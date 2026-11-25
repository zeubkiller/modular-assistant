# 🧠 Portable AI Core — Offline & Modular Personal Assistant

A **100% offline (Offline-First)**, lightweight, and modular intelligent assistant framework designed to run on low-resource portable devices (Raspberry Pi 5, Cyberdecks, Linux mini-PCs).

## 🎯 Project Vision
The goal is to build an autonomous field companion capable of reasoning, querying a local knowledge base (RAG), analyzing images via local vision models, and generating clean, structured outputs without any internet connection.

Powered by a **Profile Cartridge architecture (`profiles/`)**, the same core engine can instantly switch domains:
* 👨‍🍳 **Personal Chef (`chef` — Primary Prototype):** Fridge photo ingredient recognition (`Moondream`), voice/text pantry additions, and recipe generation tailored to family preferences, allergies, and kitchen equipment stored locally.
* 🔌 **Electronics Engineer (`electronics`):** PDF datasheet analysis, exact component calculations via Python, real-time circuit schematic generation (`Schemdraw`), and live voltage probing via ESP32.
* 🌱 **Gardener (`gardener`):** Planting schedules, visual plant disease diagnosis, and soil moisture sensor monitoring.

---

## 🏗️ Technical Architecture

```text
portable-ai-core/
├── core/                   # Core engine (Chat loop, Tool Calling, RAG, self-correction)
│   ├── config_loader.py    # Profile YAML configuration loader & validator
│   ├── engine.py           # Ollama orchestrator + automatic error self-correction loop
│   └── rag.py              # Local vector indexing and retrieval (ChromaDB)
├── profiles/               # Interchangeable domain "Cartridges"
│   ├── chef/               # Personal Chef Profile (Phase 1 reference implementation)
│   │   ├── config.yaml     # System prompt, target model, and enabled tools
│   │   ├── knowledge/      # Family preferences, allergies, pantry basics, favorite recipes
│   │   └── tools/          # Python tools (fridge photo analyzer, Markdown recipe saver)
│   └── electronics/        # Electronics Profile (Phase 2 extension)
├── tests/                  # Automated unit & integration test suite (pytest + mocks)
├── output/                 # Real-time generated files (Markdown recipes, PNG schematics)
├── main.py                 # CLI entry point (Rich terminal interface)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Software Stack (100% Open-Source & Offline)

* **LLM Engine:** [Ollama](https://ollama.com) (Local execution on CPU/GPU)
* **Edge Models (Calibrated for 8GB RAM):**
  * `qwen2.5-coder:3b` — Default reasoning model.
  * `qwen2.5:1.5b` — Recommended faster CPU option for Raspberry Pi.
  * `moondream` — Lightweight vision-language model for fridge/component photo analysis.
  * `nomic-embed-text` — Local embeddings for document and preference retrieval.
* **Knowledge Base (RAG):** `ChromaDB` + `PyMuPDF` (Clean extraction of PDFs, Markdown, and text).
* **Testing & Quality:** `pytest` + `pytest-mock` (Fast, deterministic offline test suite).
* **Interface:** `Rich` (Interactive CLI).

### Inference latency and memory

Chef requests use a 2,048-token context window, four CPU threads and keep the
reasoning model resident (`model_keep_alive: -1`). The fixed dietary profile
from `data/profil_alimentaire.yaml` is prepended to the system prompt. Only
short recipe cards from `profiles/chef/knowledge/recipes/` are retrieved as
RAG examples (at most two, 250 estimated tokens); the complete request is
limited to 550 estimated tokens. The CLI warms the model at startup.

---

## 🚀 Installation & Quick Start

### 1. System Prerequisites (Ubuntu / Debian / Raspberry Pi OS)
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git portaudio19-dev ffmpeg libgl1 zstd
```

### 2. Install Ollama & Pull Models
```bash
curl -fsSL https://ollama.cominstall.sh | sh
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
ollama pull moondream
```

### 3. Project Setup
```bash
git clone https://github.com
cd portable-ai-core
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Run Tests & Launch the Assistant
```bash
# Run the automated test suite
pytest

# Run live end-to-end tests with the faster CPU model
OLLAMA_MODEL=qwen2.5:1.5b pytest -m e2e

# Show prompts, model responses, and generated Markdown
pytest -m e2e --e2e-debug

# Start the Personal Chef assistant with the fast CPU model
OLLAMA_MODEL=qwen2.5:1.5b python main.py --profile chef
```

### Chef's two-stage workflow
Enter available ingredients and a direction (for example, `chicken, zucchini,
rice; quick dinner`). The assistant displays three familiar and two exploratory
ideas, each on one compact line. Enter a number from `1` to `3` to generate and
save only that recipe under `output/recipes/`. To include a fridge photo, enter
its image path in the same request as any typed ingredients.

---

## 🗺️ Roadmap

- [ ] **Phase 1 (Core & Chef Prototype):** Modular cartridge engine, automated `pytest` suite, local RAG for family preferences/pantry, `Moondream` fridge photo analysis, and Markdown recipe generation.
- [ ] **Phase 2 (Voice & Hands-Free UX):** Offline voice dictation (`faster-whisper`) to combine fridge photos with spoken ingredients.
- [ ] **Phase 3 (Electronics Cartridge):** Add `profiles/electronics` with PDF datasheet RAG, Python math execution, and real-time `Schemdraw` circuit rendering.
- [ ] **Phase 4 (Hardware Bridge):** USB serial integration with an **ESP32** for live sensor and voltage probing.
- [ ] **Phase 5 (Standalone Portable Build):** Deploy onto a Raspberry Pi 5 + camera module + microphone + battery enclosure.

---

## 🤝 Contributing & License
Open-source project licensed under the **MIT License**. Contributions, new domain profiles (`profiles/`), and hardware integrations are welcome!
