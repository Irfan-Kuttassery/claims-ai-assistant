import io
import requests
import json
import base64
from flask import Flask, render_template, request, jsonify  # type: ignore
from PIL import Image  # type: ignore
import torch  # type: ignore
from transformers import pipeline, CLIPProcessor, CLIPModel  # type: ignore

app = Flask(__name__)

# ──────────────────────────────────────────────
# Model Loading (once at startup)
# ──────────────────────────────────────────────
print("🚗 Loading car damage detection models...")

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

damage_type_classifier = pipeline(
    "image-classification",
    model="beingamit99/car_damage_detection",
    device=-1,   # CPU; set to 0 for GPU
)

# 🧠 Swapped to Hive API for deepfake detection instead of local model

print("🔍 Loading Vehicle Object Detector...")
object_detector = pipeline("object-detection", device=-1)

print("✅ Models loaded successfully!")

# ──────────────────────────────────────────────
# CLIP prompts — deliberately very contrastive
# so CLIP gives a strong signal, not a 50/50 flip
# ──────────────────────────────────────────────
CLIP_TEXTS = [
    # DAMAGED (indices 0-4)
    "a car with severely crushed or crumpled metal body panels after a collision",
    "a car showing heavy accident damage with broken windshield and deformed hood",
    "a wrecked vehicle with major visible damage including dents, deep scratches, or broken parts",
    "a car involved in a road accident with structural damage to the body",
    "a vehicle with shattered windows, flat tires, or heavy crash marks",
    # UNDAMAGED (indices 5-9)
    "a brand new showroom car with flawless, pristine paintwork and no damage",
    "a perfectly intact car in excellent condition with clean body panels",
    "a stock photo of an undamaged car from a car dealership or manufacturer",
    "a car in perfect working condition with no scratches, dents, or breaks",
    "a clean, well-maintained car with no visible defects or damage anywhere",
]
N_DAMAGED = 5

# ──────────────────────────────────────────────
# Dedicated AI-Image Detection Thresholds
# ──────────────────────────────────────────────
AI_REAL_THRESHOLD = 0.50

# Thresholds for ensemble decision
CLIP_DAMAGED_MIN    = 0.60  # CLIP must say ≥60% damaged to call DAMAGED
CLIP_UNDAMAGED_MIN  = 0.60  # CLIP must say ≥60% undamaged to call UNDAMAGED
DMG_TYPE_TIEBREAK   = 0.55  # If CLIP is uncertain, use damage-type model: >55% → DAMAGED

# ──────────────────────────────────────────────
# Vehicle Detection Thresholds
# ──────────────────────────────────────────────
VEHICLE_THRESHOLD = 0.50
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "airplane", "boat", "train", "bicycle"}


def clip_classify(image: Image.Image):
    inputs = clip_processor(
        text=CLIP_TEXTS, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    damaged_score   = float(probs[:N_DAMAGED].sum())
    undamaged_score = float(probs[N_DAMAGED:].sum())
    confidence      = max(damaged_score, undamaged_score)

    if damaged_score >= CLIP_DAMAGED_MIN:
        clip_verdict = "damaged"
    elif undamaged_score >= CLIP_UNDAMAGED_MIN:
        clip_verdict = "undamaged"
    else:
        clip_verdict = "uncertain"

    return clip_verdict, damaged_score, undamaged_score, confidence

def detect_vehicle(image: Image.Image):
    """Uses Object Detection to verify if the image contains any vehicle anywhere."""
    try:
        predictions = object_detector(image)
    except Exception as e:
        print(f"Object detection failed: {e}")
        return True, 1.0  # Fallback to true if model fails
        
    highest_score = 0.0
    is_vehicle = False
    
    for p in predictions:
        if p["label"] in VEHICLE_CLASSES:
            if p["score"] > highest_score:
                highest_score = float(p["score"])
            if p["score"] >= VEHICLE_THRESHOLD:
                is_vehicle = True
                
    return is_vehicle, highest_score


def detect_ai_image(image: Image.Image):
    """Uses Hive API for Deepfake & AI Generated content detection."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        #Use your token here -- Example : 'authorization': 'Bearer wrQW1pImnnU5uAMjtt==',
        
        'authorization': 'USE YOUR TOKEN HERE',
        'Content-Type': 'application/json',
    }

    json_data = {
      "input": [
        {
          "media_base64": img_str
        }
      ]
    }

    try:
        response = requests.post(
            'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection',
            headers=headers,
            json=json_data
        )
        res_json = response.json()
        
        # Parse Hive response
        classes = res_json['output'][0]['classes']
        ai_score = 0.0
        
        for cls in classes:
            if cls['class'] == 'ai_generated' or cls['class'] == 'yes':
                ai_score = float(cls.get('score', cls.get('value', 0.0)))
                break
                
        is_ai_generated = ai_score >= AI_REAL_THRESHOLD
        
        return {
            "is_ai_generated": is_ai_generated,
            "ai_score":        float(f"{ai_score * 100.0:.1f}"),
            "real_score":      float(f"{(1.0 - ai_score) * 100.0:.1f}"),
        }
    except Exception as e:
        print(f"Hive API Error: {e}")
        return {
            "is_ai_generated": False,
            "ai_score":        0.0,
            "real_score":      100.0,
        }


# ──────────────────────────────────────────────
# Damage-type classifier
# ──────────────────────────────────────────────
DAMAGE_LABEL_MAP = {
    "Crack":         {"icon": "🔧", "severity": "moderate"},
    "Scratch":       {"icon": "🪛", "severity": "minor"},
    "Tire Flat":     {"icon": "🛞", "severity": "moderate"},
    "Dent":          {"icon": "💥", "severity": "moderate"},
    "Glass Shatter": {"icon": "💎", "severity": "severe"},
    "Lamp Broken":   {"icon": "💡", "severity": "minor"},
}

def classify_damage_type(image: Image.Image):
    results = damage_type_classifier(image, top_k=6)
    types = []
    for r in results:
        label = r["label"]
        meta  = DAMAGE_LABEL_MAP.get(label, {"icon": "⚠️", "severity": "unknown"})
        types.append({
            "label":    label,
            "score":    float(f"{float(r['score']) * 100.0:.1f}"),
            "icon":     meta["icon"],
            "severity": meta["severity"],
        })
    top_score = float(results[0]["score"]) if results else 0.0
    return types, top_score


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        image = image.resize((224, 224))

        # Step 0 — Vehicle Detection
        is_vehicle, vehicle_score = detect_vehicle(image)
        if not is_vehicle:
            return jsonify({
                "is_vehicle": False,
                "vehicle_score": float(f"{vehicle_score * 100.0:.1f}")
            }), 200

        # Step 1 — CLIP binary verdict
        clip_verdict, dam_score, undam_score, confidence = clip_classify(image)

        # Step 2 — Damage-type scores (always run; used as tiebreaker)
        damage_types, top_dmg_score = classify_damage_type(image)

        # Step 3 — AI-generated image detection (reuses CLIP, no extra model)
        ai_info = detect_ai_image(image)

        # Step 4 — Ensemble final decision
        if clip_verdict == "damaged":
            final_verdict = "damaged"
        elif clip_verdict == "undamaged":
            final_verdict = "undamaged"
        else:
            # CLIP uncertain → let damage-type model decide
            final_verdict = "damaged" if top_dmg_score >= DMG_TYPE_TIEBREAK else "uncertain"

        is_damaged = (final_verdict == "damaged")

        return jsonify({
            "is_damaged":       is_damaged,       # True / False
            "verdict":          final_verdict,    # "damaged" | "undamaged" | "uncertain"
            "clip_verdict":     clip_verdict,
            "damaged_score":    float(f"{dam_score * 100.0:.1f}"),
            "undamaged_score":  float(f"{undam_score * 100.0:.1f}"),
            "confidence":       float(f"{confidence * 100.0:.1f}"),
            "top_dmg_score":    float(f"{top_dmg_score * 100.0:.1f}"),
            "damage_types":     damage_types if is_damaged else [],
            # AI-generated image detection
            "is_ai_generated":  ai_info["is_ai_generated"],
            "ai_score":         ai_info["ai_score"],
            "real_score":       ai_info["real_score"],
            "is_vehicle":       True,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
