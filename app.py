import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
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
            "score":    round(float(r["score"]) * 100, 1),
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

        # Step 3 — Ensemble final decision
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
            "damaged_score":    round(dam_score   * 100, 1),
            "undamaged_score":  round(undam_score * 100, 1),
            "confidence":       round(confidence  * 100, 1),
            "top_dmg_score":    round(top_dmg_score * 100, 1),
            "damage_types":     damage_types if is_damaged else [],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
