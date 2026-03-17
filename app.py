import io
import re
from flask import Flask, render_template, request, jsonify
from PIL import Image
from PIL.ExifTags import TAGS
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

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
# CLIP prompts
# ──────────────────────────────────────────────
CLIP_TEXTS = [
    # DAMAGED
    "a car with severely crushed or crumpled metal body panels after a collision",
    "a car showing heavy accident damage with broken windshield and deformed hood",
    "a wrecked vehicle with major visible damage including dents, deep scratches, or broken parts",
    "a car involved in a road accident with structural damage to the body",
    "a vehicle with shattered windows, flat tires, or heavy crash marks",

    # UNDAMAGED
    "a brand new showroom car with flawless, pristine paintwork and no damage",
    "a perfectly intact car in excellent condition with clean body panels",
    "a stock photo of an undamaged car from a car dealership or manufacturer",
    "a car in perfect working condition with no scratches, dents, or breaks",
    "a clean, well-maintained car with no visible defects or damage anywhere",
]

N_DAMAGED = 5

# Thresholds (Aggressive for Insurance Triage)
CLIP_DAMAGED_MIN    = 0.35  
CLIP_UNDAMAGED_MIN  = 0.75  
DMG_TYPE_TIEBREAK   = 0.50  
FAKE_THRESHOLD      = 0.60


# ──────────────────────────────────────────────
# CLIP Classification
# ──────────────────────────────────────────────
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
        verdict = "damaged"
    elif undamaged_score >= CLIP_UNDAMAGED_MIN:
        verdict = "undamaged"
    else:
        verdict = "uncertain"

    return verdict, damaged_score, undamaged_score, confidence


# ──────────────────────────────────────────────
# Damage Type Classification
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
            "label": label,
            "score": round(float(r["score"]) * 100, 1),
            "icon": meta["icon"],
            "severity": meta["severity"],
        })

    top_score = float(results[0]["score"]) if results else 0.0
    return types, top_score



# ──────────────────────────────────────────────
# Deepfake Detection (CLIP zero-shot — car-specific prompts)
# ──────────────────────────────────────────────
def clip_fake_detect(image: Image.Image):
    texts = [
        # REAL prompts — imperfect, candid, natural
        "a real camera photo of a car with natural lighting and minor imperfections",
        "a genuine photograph taken outdoors with visible road grime or dust on the car",
        "a candid insurance claim photo of a car taken with a phone or camera",
        # AI / FAKE prompts — overly perfect, stock, watermarked
        "an AI-generated image of a car with flawless paintwork and perfect studio lighting",
        "a synthetic computer-generated car photo with unrealistically clean appearance",
        "a watermarked stock photo of a car from a photo agency like Dreamstime or Shutterstock",
        "a photorealistic AI artwork of a car that looks too cinematic and too perfect",
    ]

    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0).numpy()

    real_score = float(probs[:3].sum()) / 3   # avg per real prompt
    fake_score = float(probs[3:].sum()) / 4   # avg per fake prompt

    # Re-normalise to sum to 1 for comparison
    total = real_score + fake_score
    real_score = real_score / total
    fake_score = fake_score / total

    diff = abs(real_score - fake_score)

    if real_score > fake_score and diff > 0.10:
        verdict = "real"
    elif fake_score > real_score and diff > 0.15:
        verdict = "fake"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict,
        "real_score": round(real_score * 100, 1),
        "fake_score": round(fake_score * 100, 1)
    }


# ──────────────────────────────────────────────
# Dedicated Deepfake Model Check (Consensus View)
# ──────────────────────────────────────────────
def check_if_fake_v3(image: Image.Image):
    # To prevent false positives on real images, we check the model's opinion
    # using two different resizing algorithms. If the model is consistent, it's a stronger signal.
    # We use LANCZOS (high quality) and BILINEAR (faster/softer).
    
    views = []
    for resample in [Image.Resampling.LANCZOS, Image.Resampling.BILINEAR]:
        v_img = image.resize((224, 224), resample=resample)
        results = deepfake_detector(v_img)
        
        f_score = 0.0
        r_score = 0.0
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            if "deepfake" in label or "fake" in label:
                f_score = max(f_score, score)
            elif "realism" in label or "real" in label:
                r_score = max(r_score, score)
        views.append({"fake": f_score, "real": r_score})

    # Average scores across views
    avg_fake = sum(v["fake"] for v in views) / len(views)
    avg_real = sum(v["real"] for v in views) / len(views)
    
    # Check consistency (if views differ wildly, the model is 'hallucinating' on artifacts)
    diff = abs(views[0]["fake"] - views[1]["fake"])
    is_confused = diff > 0.25 # Model is unsure due to resizing artifacts

    return {
        "fake_score": round(avg_fake * 100, 1),
        "real_score": round(avg_real * 100, 1),
        "is_confused": is_confused
    }

# ──────────────────────────────────────────────
# Metadata-based Authenticity Check
# ──────────────────────────────────────────────
AI_FILENAME_KEYWORDS = [
    # AI generator tools
    "ai", "generated", "midjourney", "dalle", "dall-e", "stable-diffusion",
    "stablediffusion", "synthetic", "fake", "artificial", "aicar", "aigc",
    "ai_generated", "ai-generated", "openai", "sora", "flux", "firefly",
    # Stock photo sites (watermarked = not claimant's own photo = fraud signal)
    "dreamstime", "shutterstock", "gettyimages", "istockphoto", "istock",
    "123rf", "adobestock", "stock", "freepik", "depositphotos", "alamy",
    "bigstock", "canstockphoto", "vecteezy"
]

def check_metadata(raw_bytes: bytes, filename: str) -> dict:
    signals = []
    verdict = "unknown"

    # ── Signal 1: Filename keywords ──
    name_lower = filename.lower()
    name_clean = re.sub(r'[^a-z0-9]', ' ', name_lower)
    found_keywords = [kw for kw in AI_FILENAME_KEYWORDS if kw in name_clean]
    filename_signal = "fake" if found_keywords else "real"
    signals.append((filename_signal, 0.9 if found_keywords else 0.3))  # high confidence if keyword found
    if found_keywords:
        signals_info = f"Filename contains AI keywords: {', '.join(found_keywords)}"
    else:
        signals_info = "Filename looks real"

    # ── Signal 2: EXIF metadata ──
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        exif_data = pil_img._getexif() if hasattr(pil_img, '_getexif') else None

        if exif_data:
            readable = {TAGS.get(k, k): v for k, v in exif_data.items()}
            has_camera = "Make" in readable or "Model" in readable
            has_settings = "ExposureTime" in readable or "FNumber" in readable or "ISOSpeedRatings" in readable

            if has_camera and has_settings:
                # Definitely taken by a real camera
                signals.append(("real", 0.85))
                signals_info += f" · EXIF: {readable.get('Make','')} {readable.get('Model','')}"
            elif has_camera:
                signals.append(("real", 0.6))
                signals_info += " · EXIF: Camera info found"
            else:
                # Has EXIF but no camera info — software-generated or stripped
                signals.append(("fake", 0.5))
                signals_info += " · EXIF: App-processed"
        else:
            # No EXIF - don't penalize, just stay neutral
            signals.append(("uncertain", 0.0))
            signals_info += " · EXIF: Missing (not a fraud signal)"
    except Exception:
        signals_info += " · EXIF: Error"

    # ── Aggregate signals ──
    fake_weight = sum(w for (v, w) in signals if v == "fake")
    real_weight = sum(w for (v, w) in signals if v == "real")

    if fake_weight > real_weight:
        verdict = "fake"
    elif real_weight > fake_weight:
        verdict = "real"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict,
        "details": signals_info,
        "fake_weight": round(fake_weight, 2),
        "real_weight": round(real_weight, 2),
    }


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
        file_obj   = request.files["image"]
        filename   = file_obj.filename or ""
        raw_bytes  = file_obj.read()

        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        # Step 1 — CLIP binary verdict
        clip_verdict, dam_score, undam_score, confidence = clip_classify(pil_img)

        # Step 2 — Damage types
        damage_types, top_dmg_score = classify_damage_type(pil_img)

        # Step 3 — Damage final decision
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

        is_damaged = (final_verdict == "damaged")

        # Step 4 — Authenticity (Multi-View Model + Metadata + SILENT CLIP GUARD)
        meta_auth   = check_metadata(raw_bytes, filename)
        deep_auth   = check_if_fake_v3(pil_img)
        clip_auth   = clip_fake_detect(pil_img)

        # 1. Filename AI Keywords (Strongest Fraud Signal)
        if "Filename contains AI keywords" in meta_auth["details"]:
            final_auth_verdict = "fake"
        
        # 2. Camera EXIF Metadata (Strongest Real Proof)
        elif "EXIF:" in meta_auth["details"] and ("info" in meta_auth["details"] or "Model" in meta_auth["details"] or "Make" in meta_auth["details"]):
            final_auth_verdict = "real"

        # 3. Model Consensus Check
        else:
            # If the model is confused (unstable across resizes), it's likely a real photo with noise
            if deep_auth["is_confused"]:
                final_auth_verdict = "real"
            # Strict majority 50% threshold
            elif deep_auth["fake_score"] >= 50.0:
                # FINAL SAFETY GUARD: If CLIP is very sure it's real (>90%), trust CLIP over model
                if clip_auth["real_score"] >= 90.0:
                    final_auth_verdict = "real"
                else:
                    final_auth_verdict = "fake"
            else:
                final_auth_verdict = "real"

        authenticity_result = {
            "verdict":    final_auth_verdict,
            "fake_score": deep_auth["fake_score"],
            "real_score": deep_auth["real_score"],
            "meta":       meta_auth["details"]
        }

        return jsonify({
            "is_damaged":       is_damaged,
            "verdict":          final_verdict,
            "clip_verdict":     clip_verdict,
            "damaged_score":    round(dam_score * 100, 1),
            "undamaged_score":  round(undam_score * 100, 1),
            "confidence":       round(confidence * 100, 1),
            "top_dmg_score":    round(top_dmg_score * 100, 1),
            "damage_types":     damage_types if is_damaged else [],
            "authenticity":     authenticity_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Run App
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)