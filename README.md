# 🚗 FNOL FastTrack — AI Claims Assistant

> **First Notice of Loss (FNOL) automation using computer vision & GenAI**  
> Phase 1: Instant car damage detection from photos — no training required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green)](https://flask.palletsprojects.com)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📖 What is this?

When a car accident happens, insurance companies receive photos, emails, and forms. Traditionally, a human reads everything and manually enters data — even for simple cases like a cracked windshield.

**FNOL FastTrack** automates this using AI:

| Phase | Feature | Status |
|---|---|---|
| **Phase 1** | Car damage detection from photos (Damaged / Undamaged / Uncertain) | ✅ Done |
| **Phase 2** | AI-generated / deepfake photo detection (fraud prevention) | 🔜 Upcoming |
| **Phase 3** | Full FNOL pipeline — parse emails, forms, auto-triage claims | 🔜 Upcoming |

---

## 🧠 How It Works (Architecture)

We use **two pre-trained models from HuggingFace** — no custom training needed:

```
User uploads car photo
         │
         ▼
   Flask Web Server (app.py)
         │
         ▼
   ┌─────────────────────────────────────┐
   │  Model 1: CLIP (Zero-Shot)          │
   │  openai/clip-vit-base-patch32       │
   │                                     │
   │  10 text prompts (5 damaged,        │
   │  5 undamaged) → similarity scores   │
   │                                     │
   │  damaged ≥ 60%   → DAMAGED          │
   │  undamaged ≥ 60% → UNDAMAGED        │
   │  both < 60%      → UNCERTAIN        │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │  Model 2: Fine-tuned ViT            │
   │  beingamit99/car_damage_detection   │
   │                                     │
   │  • Tiebreaker when CLIP uncertain   │
   │  • Damage type: Crack / Dent /      │
   │    Scratch / Flat Tire / Glass /    │
   │    Lamp Broken                      │
   └─────────────────────────────────────┘
         │
         ▼
   JSON response → Beautiful web UI
```

**This is NOT:**
- ❌ AI Agents (no autonomous decision-making)
- ❌ Custom-trained model (we use existing pre-trained weights)

**This IS:**
- ✅ Calling pre-trained models locally (no cloud API cost)
- ✅ Zero-shot classification via CLIP
- ✅ Ensemble / tiebreaker logic for reliability

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~2GB free disk (for model weights, downloaded once)
- Internet connection (first run only, to download models)

### 1. Clone the repo
```bash
git clone https://github.com/Irfan-Kuttassery/claims-ai-assistant.git
cd claims-ai-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ If you get a Pillow error, run: `pip install --upgrade Pillow`

### 3. Run the server
```bash
python3 app.py
```

**First launch** takes ~5-10 minutes — it downloads ~950MB of model weights from HuggingFace.  
**Subsequent launches** take ~10 seconds (models are cached at `~/.cache/huggingface/`).

### 4. Open the app
```
http://127.0.0.1:5000
```

Upload any car photo and the AI will classify it instantly.

---

## 📁 Project Structure

```
claims-ai-assistant/
├── app.py              ← Flask backend + AI inference logic
├── requirements.txt    ← Python dependencies
└── templates/
    └── index.html      ← Frontend (dark-mode UI, drag & drop)
```

---

## 🖥️ UI Preview

Dark-mode web interface with:
- Drag & drop photo upload
- Verdict card (Damaged 🚨 / Undamaged ✅ / Uncertain ⚠️)
- CLIP probability score bars
- Damage type breakdown (Crack, Dent, Scratch, etc.)

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server |
| `transformers` | HuggingFace model loading |
| `torch` + `torchvision` | PyTorch backend for inference |
| `Pillow` | Image processing |

---

## 🛑 Stop / Restart the Server

```bash
# Stop
Ctrl + C

# Restart
python3 app.py
```

---

## 🔮 Roadmap

- [ ] **Phase 2**: Detect AI-generated / deepfake car photos (fraud detection)
- [ ] **Phase 3**: Parse claim emails & forms, generate 1-page claim summary, auto-triage simple vs complex claims
- [ ] GPU support (set `device=0` in `app.py` for CUDA)
- [ ] Fine-tune on custom dataset for higher accuracy
- [ ] REST API mode (headless, no UI)

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

---

## 📄 License

MIT — see [LICENSE](LICENSE)
