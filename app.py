import io
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
# CLIP prompts for AI-generated image detection
# ──────────────────────────────────────────────
AI_DETECTION_TEXTS = [
    # REAL (indices 0-4)
    "a real photograph taken by a camera showing a real car",
    "an authentic photo of an actual car taken on a street or parking lot",
    "a genuine camera photo of a real vehicle with natural lighting",
    "a real car photo with authentic road environment in the background",
    "a candid real-world photograph of a physical car",
    # AI-GENERATED (indices 5-9)
    "an AI-generated digital artwork of a car with perfect unrealistic rendering",
    "a synthetic computer-generated image of a car created by artificial intelligence",
    "a digitally generated car illustration with hyper-realistic AI rendering",
    "an artificially created image of a car produced by a generative AI model",
    "a fake AI-generated car photo with unnaturally perfect details and textures",
]
N_AI_REAL = 5  # first N_AI_REAL prompts = real

# Threshold: if real_score >= this, call it REAL
AI_REAL_THRESHOLD = 0.55

# Thresholds for ensemble decision
CLIP_DAMAGED_MIN    = 0.60  # CLIP must say ≥60% damaged to call DAMAGED
CLIP_UNDAMAGED_MIN  = 0.60  # CLIP must say ≥60% undamaged to call UNDAMAGED
DMG_TYPE_TIEBREAK   = 0.55  # If CLIP is uncertain, use damage-type model: >55% → DAMAGED


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


def detect_ai_image(image: Image.Image):
    """Uses CLIP (already loaded) to detect if the image is AI-generated or a real photo."""
    inputs = clip_processor(
        text=AI_DETECTION_TEXTS, images=image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    real_score = float(probs[:N_AI_REAL].sum())
    ai_score   = float(probs[N_AI_REAL:].sum())

    is_ai_generated = real_score < AI_REAL_THRESHOLD
    return {
        "is_ai_generated": is_ai_generated,
        "ai_score":        float(f"{ai_score * 100.0:.1f}"),
        "real_score":      float(f"{real_score * 100.0:.1f}"),
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
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
