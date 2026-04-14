import io
import requests
import json
import base64
import os
import smtplib
import threading
from email.message import EmailMessage
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
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
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


def send_alert_email(verdict, damage_score, ai_verdict, real_score):
    sender_email = os.environ.get("SENDER_EMAIL", "irfankuttassery713@gmail.com")
    # For Gmail, you will need an App Password if 2FA is enabled
    sender_password = os.environ.get("SENDER_PASSWORD", "bnibcbamehpippwp")
    receiver_email = os.environ.get("RECEIVER_EMAIL", sender_email)

    msg = EmailMessage()
    msg.set_content(
        f"Alert: A genuine damaged car was detected!\n\n"
        f"Verdict: {verdict.capitalize()}\n"
        f"Damage Confidence: {damage_score}%\n"
        f"AI Detection Verdict: {ai_verdict}\n"
        f"Real Photo Confidence: {real_score}%"
    )
    msg['Subject'] = 'Car Damage Alert - Real Incident Detected'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
    except Exception:
        pass

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


def detect_ai_image(image_bytes: bytes):
    """Uses Hive API for Deepfake & AI Generated content detection."""
    img_str = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        'authorization': 'Bearer GtCYFYPu3787AI0lmZfqWg==',
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
        print("📡 Calling Hive AI Detection API...")
        response = requests.post(
            'https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection',
            headers=headers,
            json=json_data,
            timeout=30
        )
        response.raise_for_status()
        res_json = response.json()
        
        # Hive response usually looks like:
        # {"output": [{"classes": [{"class": "ai_generated", "score": 0.99}, ...]}]}
        # or for deepfakes: {"output": [{"classes": [{"class": "yes", "score": 0.99}, ...]}]}
        
        output = res_json.get('output', [{}])[0]
        classes = output.get('classes', [])
        
        ai_score = 0.0
        for cls in classes:
            # We look for 'ai_generated' (for AI Image) or 'yes' (for Deepfake head)
            if cls['class'] in ['ai_generated', 'yes']:
                score = float(cls.get('score', cls.get('value', 0.0)))
                # Keep the highest score among relevant classes
                if score > ai_score:
                    ai_score = score
        
        print(f"📊 Hive API Result: AI Score = {ai_score:.4f}")
        
        print(f"📊 Hive API Result: AI Score = {ai_score:.4f}")
        
        # Granular Tiered Logic:
        # > 50%       → "AI Generated"
        # 20% to 50%  → "Likely AI"
        # 10% to 20%  → "Likely Real"
        # < 10%       → "Real"
        
        if ai_score > 0.50:
            verdict = "AI Generated"
        elif ai_score > 0.20:
            verdict = "Likely AI"
        elif ai_score > 0.10:
            verdict = "Likely Real"
        else:
            verdict = "Real"
            
        return {
            "verdict":         verdict,
            "ai_score":        float(f"{ai_score * 100.0:.1f}"),
            "real_score":      float(f"{(1.0 - ai_score) * 100.0:.1f}"),
        }
    except Exception as e:
        print(f"❌ Hive API Error: {e}")
        return {
            "verdict":         "Error",
            "ai_score":        0.0,
            "real_score":      0.0, 
            "error":           str(e)
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
        image_data = request.files["image"].read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Keep original bytes for Hive (don't resize)
        # Resize ONLY for CLIP and local classification models
        image_resized = image.resize((224, 224))

        # Step 0 — Vehicle Detection (on resized is fine, but can use original)
        is_vehicle, vehicle_score = detect_vehicle(image_resized)
        if not is_vehicle:
            return jsonify({
                "is_vehicle": False,
                "vehicle_score": float(f"{vehicle_score * 100.0:.1f}")
            }), 200

        # Step 1 — CLIP binary verdict
        clip_verdict, dam_score, undam_score, confidence = clip_classify(image_resized)

        # Step 2 — Damage-type scores (always run; used as tiebreaker)
        damage_types, top_dmg_score = classify_damage_type(image_resized)

        # Step 3 — AI-generated image detection (USES ORIGINAL BYTES)
        ai_info = detect_ai_image(image_data)

        # Step 4 — Ensemble final decision
        if clip_verdict == "damaged":
            final_verdict = "damaged"
        elif clip_verdict == "undamaged":
            final_verdict = "undamaged"
        else:
            # CLIP uncertain → let damage-type model decide
            final_verdict = "damaged" if top_dmg_score >= DMG_TYPE_TIEBREAK else "uncertain"

        is_damaged = (final_verdict == "damaged")

        # Step 5 — Send Email Alert if Damaged and Real
        email_dispatched = False
        if is_damaged and ai_info.get("verdict") in ["Real", "Likely Real"]:
            email_dispatched = True
            dam_score_formatted = float(f"{dam_score * 100.0:.1f}") if final_verdict == "damaged" else float(f"{top_dmg_score * 100.0:.1f}")
            threading.Thread(
                target=send_alert_email,
                args=(
                    final_verdict,
                    dam_score_formatted,
                    ai_info["verdict"],
                    ai_info["real_score"]
                ),
                daemon=True
            ).start()

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
            "ai_verdict":       ai_info["verdict"],
            "ai_score":         ai_info["ai_score"],
            "real_score":       ai_info["real_score"],
            "is_vehicle":       True,
            "email_dispatched": email_dispatched,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
