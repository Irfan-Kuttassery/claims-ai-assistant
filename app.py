import io
import re
from flask import Flask, render_template, request, jsonify
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

app = Flask(__name__)

# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
print("🚗 Loading car damage detection models...")

clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

damage_type_classifier = pipeline(
    "image-classification",
    model="beingamit99/car_damage_detection",
    device=-1,
)

print("🧠 Loading AI-image detector...")
deepfake_detector = pipeline(
    "image-classification",
    model="prithivMLmods/Deep-Fake-Detector-v2-Model",
    device=-1
)

print("✅ All models loaded successfully!")

# ──────────────────────────────────────────────
# CLIP prompts (Damage)
# ──────────────────────────────────────────────
CLIP_TEXTS = [
    "a car with severely crushed or crumpled metal body panels after a collision",
    "a car showing heavy accident damage with broken windshield and deformed hood",
    "a wrecked vehicle with major visible damage including dents, deep scratches, or broken parts",
    "a car involved in a road accident with structural damage to the body",
    "a vehicle with shattered windows, flat tires, or heavy crash marks",
    "a brand new showroom car with flawless, pristine paintwork and no damage",
    "a perfectly intact car in excellent condition with clean body panels",
    "a stock photo of an undamaged car from a car dealership or manufacturer",
    "a car in perfect working condition with no scratches, dents, or breaks",
    "a clean, well-maintained car with no visible defects or damage anywhere",
]

N_DAMAGED = 5

CLIP_DAMAGED_MIN    = 0.35  
CLIP_UNDAMAGED_MIN  = 0.75  
DMG_TYPE_TIEBREAK   = 0.50  

# ──────────────────────────────────────────────
# Friend's CLIP AI detection prompts
# ──────────────────────────────────────────────
AI_DETECTION_TEXTS = [
    "a real photograph taken by a camera showing a real car",
    "an authentic photo of an actual car taken on a street or parking lot",
    "a genuine camera photo of a real vehicle with natural lighting",
    "a real car photo with authentic road environment in the background",
    "a candid real-world photograph of a physical car",
    "an AI-generated digital artwork of a car with perfect unrealistic rendering",
    "a synthetic computer-generated image of a car created by artificial intelligence",
    "a digitally generated car illustration with hyper-realistic AI rendering",
    "an artificially created image of a car produced by a generative AI model",
    "a fake AI-generated car photo with unnaturally perfect details and textures",
]

N_AI_REAL = 5
AI_REAL_THRESHOLD = 0.55

# ──────────────────────────────────────────────
# CLIP Damage Classification
# ──────────────────────────────────────────────
def clip_classify(image):
    inputs = clip_processor(text=CLIP_TEXTS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    damaged_score   = float(probs[:N_DAMAGED].sum())
    undamaged_score = float(probs[N_DAMAGED:].sum())
    confidence      = max(damaged_score, undamaged_score)

    if damaged_score >= CLIP_DAMAGED_MIN:
        verdict = "damaged"
    elif undamaged_score >= CLIP_UNDAMAGED_MIN:
        verdict = "undamaged"
    else:
        verdict = "uncertain"

    return verdict, damaged_score, undamaged_score, confidence

# ──────────────────────────────────────────────
# Friend CLIP AI detector
# ──────────────────────────────────────────────
def friend_clip_ai_detect(image):
    inputs = clip_processor(text=AI_DETECTION_TEXTS, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    real_score = float(probs[:N_AI_REAL].sum())
    ai_score   = float(probs[N_AI_REAL:].sum())

    return {
        "real_score": round(float(real_score * 100), 1),
        "ai_score": round(float(ai_score * 100), 1),
        "is_ai": real_score < AI_REAL_THRESHOLD
    }

# ──────────────────────────────────────────────
# Damage Type Classification
# ──────────────────────────────────────────────
DAMAGE_LABEL_MAP = {
    "Crack": {"icon": "🔧", "severity": "moderate"},
    "Scratch": {"icon": "🪛", "severity": "minor"},
    "Tire Flat": {"icon": "🛞", "severity": "moderate"},
    "Dent": {"icon": "💥", "severity": "moderate"},
    "Glass Shatter": {"icon": "💎", "severity": "severe"},
    "Lamp Broken": {"icon": "💡", "severity": "minor"},
}

def classify_damage_type(image):
    results = damage_type_classifier(image, top_k=6)

    types = []
    for r in results:
        label = r["label"]
        meta = DAMAGE_LABEL_MAP.get(label, {"icon": "⚠️", "severity": "unknown"})
        types.append({
            "label": label,
            "score": round(float(r["score"]) * 100, 1),
            "icon": meta["icon"],
            "severity": meta["severity"],
        })

    top_score = float(results[0]["score"]) if results else 0.0
    return types, top_score

# ──────────────────────────────────────────────
# YOUR EXISTING AI DETECTORS (UNCHANGED)
# ──────────────────────────────────────────────
def clip_fake_detect(image):
    texts = [
        "a real camera photo of a car with natural lighting and minor imperfections",
        "a genuine photograph taken outdoors with visible road grime or dust on the car",
        "a candid insurance claim photo of a car taken with a phone or camera",
        "an AI-generated image of a car with flawless paintwork and perfect studio lighting",
        "a synthetic computer-generated car photo with unrealistically clean appearance",
        "a watermarked stock photo of a car from a photo agency like Dreamstime or Shutterstock",
        "a photorealistic AI artwork of a car that looks too cinematic and too perfect",
    ]

    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    real_score = float(probs[:3].sum()) / 3
    fake_score = float(probs[3:].sum()) / 4

    total = real_score + fake_score
    real_score /= total
    fake_score /= total

    diff = abs(real_score - fake_score)

    if real_score > fake_score and diff > 0.10:
        verdict = "real"
    elif fake_score > real_score and diff > 0.15:
        verdict = "fake"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict,
        "real_score": round(float(real_score * 100), 1),
        "fake_score": round(float(fake_score * 100), 1)
    }

def check_if_fake_v3(image):
    views = []
    for resample in [Image.Resampling.LANCZOS, Image.Resampling.BILINEAR]:
        v_img = image.resize((224, 224), resample=resample)
        results = deepfake_detector(v_img)

        f_score, r_score = 0.0, 0.0
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            if "fake" in label:
                f_score = max(f_score, score)
            elif "real" in label:
                r_score = max(r_score, score)

        views.append({"fake": f_score, "real": r_score})

    avg_fake = sum(v["fake"] for v in views) / 2
    avg_real = sum(v["real"] for v in views) / 2

    return {
        "fake_score": round(float(avg_fake * 100), 1),
        "real_score": round(float(avg_real * 100), 1)
    }

# ──────────────────────────────────────────────
# Metadata check (unchanged)
# ──────────────────────────────────────────────
def check_metadata(raw_bytes, filename):
    signals = []
    name_lower = filename.lower()

    found_keywords = any(k in name_lower for k in ["ai", "generated", "midjourney", "dalle", "stock"])
    signals.append(("fake" if found_keywords else "real", 0.9 if found_keywords else 0.3))

    return {
        "details": "Filename contains AI keywords" if found_keywords else "Filename looks real"
    }

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    raw_bytes = file.read()
    filename = file.filename

    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    # DAMAGE (UNCHANGED)
    clip_verdict, dam_score, undam_score, confidence = clip_classify(pil_img)
    damage_types, top_dmg_score = classify_damage_type(pil_img)

    if clip_verdict == "damaged":
        final_verdict = "damaged"
    elif clip_verdict == "uncertain" and top_dmg_score >= DMG_TYPE_TIEBREAK:
        final_verdict = "damaged"
    elif clip_verdict == "undamaged" and top_dmg_score >= 0.70:
        final_verdict = "damaged"
    elif clip_verdict == "undamaged":
        final_verdict = "undamaged"
    else:
        final_verdict = "uncertain"
    MIN_DAMAGE_DISPLAY_THRESHOLD = 2.0  # %

    is_damaged = (
        final_verdict == "damaged" and
        (dam_score * 100) >= MIN_DAMAGE_DISPLAY_THRESHOLD
    )

    # Override verdict if below threshold
    if final_verdict == "damaged" and (dam_score * 100) < MIN_DAMAGE_DISPLAY_THRESHOLD:
        final_verdict = "undamaged"

    # 🔥 CLIP AI DETECTION
    clip_auth = clip_fake_detect(pil_img)
    final_auth = clip_auth["verdict"]

    w_fake = clip_auth["fake_score"]
    w_real = clip_auth["real_score"]

    deep_auth = {"fake_score": 0, "real_score": 0}
    verified = False

    # Second pass if CLIP thinks it's real (60/40 weight)
    if final_auth == "real":
        verified = True
        deep_auth = check_if_fake_v3(pil_img)

        # Weighted average (51% Deepfake / 49% CLIP as suggested)
        w_fake = round(float((float(clip_auth["fake_score"]) * 0.49) + (float(deep_auth["fake_score"]) * 0.51)), 1)
        w_real = round(float((float(clip_auth["real_score"]) * 0.49) + (float(deep_auth["real_score"]) * 0.51)), 1)

        final_auth = "fake" if w_fake > w_real else "real"

    return jsonify({
        "is_damaged":       is_damaged,
        "verdict":          final_verdict,
        "clip_verdict":     clip_verdict,

        "damaged_score":    round(dam_score * 100, 1),
        "undamaged_score":  round(undam_score * 100, 1),
        "confidence":       round(confidence * 100, 1),
        "top_dmg_score":    round(top_dmg_score * 100, 1),

        "damage_types":     damage_types if is_damaged else [],

        "authenticity": {
            "verdict": final_auth,
            "weighted_fake": w_fake,
            "weighted_real": w_real,
            "clip_real": clip_auth["real_score"],
            "clip_fake": clip_auth["fake_score"],
            "deep_real": deep_auth["real_score"],
            "deep_fake": deep_auth["fake_score"],
            "verified": verified
        }
    })
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)