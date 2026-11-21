# 📋 Portable AI Core — Functional & Performance Specifications

This document serves as the **Single Source of Truth** for system behavior, feature acceptance criteria, hardware constraints, and performance budgets. All End-to-End (`e2e`) tests must validate against these specifications.

---

## 1. Target Hardware & Physical Constraints

The software is developed on an x86_64 Intel VM (4 vCPUs, 8GB RAM) and targets standalone deployment on a **Raspberry Pi 5 (8GB RAM, ARM64 CPU-only, passive/active cooling, small 5"–7" screen)**.

* **Network:** 100% Offline during execution (zero external API calls).
* **Memory Ceiling:** Total system RAM footprint (Python process + ChromaDB + active Ollama model) must never exceed **5.5 GB** to prevent OS swapping on SD/NVMe storage.
* **Model Swapping Rule:** Vision models (`moondream`) must unload immediately after inference (`keep_alive=0`) before loading the text reasoning model (`qwen2.5-coder:3b`).
* **Display Constraint:** Small screen form factor. Intermediate choices must be ultra-compact (max 1 line per item). Final outputs must use clean, predictable Markdown templates.

---

## 2. Performance Budgets & Telemetry Metrics

Every pipeline execution logs telemetry to `output/logs/perf_metrics.jsonl`. **Any E2E test exceeding the hard timeout limits below must fail.**

| Pipeline Stage | Metric Tracked | Target (Optimal) | Hard Limit (E2E Fail Threshold) |
| :--- | :--- | :--- | :--- |
| **RAG Retrieval** | `rag_retrieval_ms` | `< 250 ms` | `1,000 ms` (1.0 s) |
| **Vision Analysis (`moondream`)** | `vision_inference_ms` | `< 4,000 ms` | `12,000 ms` (12.0 s) |
| **Step 1: 5 Proposals (`qwen2.5`)** | `step1_total_ms` | `< 4,000 ms` | `15,000 ms` (15.0 s) |
| **Step 2: Full Recipe (`qwen2.5`)** | `step2_total_ms` | `< 10,000 ms` | `30,000 ms` (30.0 s) |
| **Generation Speed** | `tokens_per_second` | `> 10.0 tok/s` | `> 4.0 tok/s` |
| **Peak Process RAM** | `ram_usage_mb` | `< 4,500 MB` | `< 5,500 MB` |

---

## 3. Core Architecture Features (`core/`)

### Feature C1: Cartridge Profile Loader (`core/config_loader.py`)
* **Objective:** Dynamically load domain-specific behavior without modifying core code.
* **Acceptance Criteria:**
  * Reads `profiles/<name>/config.yaml`.
  * Validates required fields (`profile_name`, `model`, `system_prompt`, `tools`).
  * Raises a clear `ConfigurationError` if a profile or required key is missing.

### Feature C2: Local Knowledge Base / RAG (`core/rag.py`)
* **Objective:** Ground LLM responses in user-provided documents without exceeding context window limits.
* **Acceptance Criteria:**
  * Indexes `.md`, `.txt`, and `.pdf` files from `profiles/<name>/knowledge/` into a local `ChromaDB` instance using `nomic-embed-text`.
  * Retrieves top-$k$ ($k \le 3$) relevant chunks per query to keep prompt size under `1,024` tokens.

### Feature C3: Self-Correcting Tool Orchestrator (`core/engine.py`)
* **Objective:** Execute deterministic Python tools and recover gracefully from small-model syntax/argument errors.
* **Acceptance Criteria:**
  * Supports structured outputs via Pydantic schemas.
  * Captures tool execution exceptions (`stderr`), feeds the error back to the LLM, and retries up to **2 times** before failing gracefully.

---

## 4. Reference Cartridge: Personal Chef (`profiles/chef/`)

### Feature F1: Pantry Staples & Family Context
* **Objective:** Eliminate repetitive input of basic ingredients while strictly enforcing dietary rules.
* **Acceptance Criteria:**
  * Automatically assumes availability of ingredients listed in `profiles/chef/knowledge/pantry_staples.md` (oil, salt, pepper, basic spices, flour, etc.).
  * Strictly respects allergies, dislikes, household size, and available appliances defined in `profiles/chef/knowledge/family_preferences.md`.

### Feature F2: Multimodal Ingredient Ingestion
* **Objective:** Combine visual fridge inventory with manual text/voice additions.
* **Acceptance Criteria:**
  * `analyze_fridge_photo(image_path)` uses `moondream` (`keep_alive=0`) to return a concise list of visible food items.
  * Merges detected items with user-typed ingredients into a unified session inventory.
  * Returns a clean error message if `image_path` does not exist.

### Feature F3: Two-Stage Minimalist Recipe Workflow
* **Objective:** Minimize screen clutter, preserve context window memory, and balance safety with culinary discovery.
* **Acceptance Criteria:**
  * **Stage 1 (Brainstorming):** Given ingredients and user direction, outputs a Pydantic-validated list of **exactly 5 proposals**:
    * **3 `Classique`:** Safe, familiar dishes matching household preferences.
    * **2 `Exploration`:** Creative or international dishes (e.g., Chinese, Indian, Mediterranean) still respecting allergies.
    * Each proposal pitch must be **$\le 12$ words** to fit on a compact portable screen.
  * **Stage 2 (Generation & Templating):** Upon user selection (`1` to `5`), clears Stage 1 clutter and generates **only** the chosen recipe using a strict Pydantic schema (`title`, `category`, `prep_time`, `cook_time`, `servings`, `ingredients`, `steps`, `chef_tip`).

### Feature F4: Standardized Recipe Export
* **Objective:** Produce consistent, readable recipe cards for offline viewing or printing.
* **Acceptance Criteria:**
  * `save_recipe_markdown` injects the structured recipe object into `profiles/chef/templates/recipe_template.md`.
  * Saves the resulting file to `output/recipes/<sanitized_title>.md` and returns the file path.

---

## 5. End-to-End (`e2e`) Test Matrix

All `e2e` tests run with `temperature=0.0` and `seed=42` against the live local Ollama instance (`pytest -m e2e`).

1. **`test_e2e_chef_two_stage_pipeline`:**
   * *Input:* Ingredients `"chicken breast, zucchini, rice"`, preference `"quick dinner"`.
   * *Assertions:* Stage 1 returns 5 items (3 `Classique`, 2 `Exploration`), all pitches $\le 12$ words, completes in `< 15s`. Selecting item `4` generates a valid Markdown file in `output/recipes/` matching all template headers in `< 30s`.
2. **`test_e2e_strict_allergy_enforcement`:**
   * *Input:* Temporary profile with `"STRICT ALLERGY: Peanuts and tree nuts"`, request `"Asian stir-fry exploration"`.
   * *Assertions:* Neither Stage 1 proposals nor Stage 2 recipe ingredients/steps contain `"peanut"`, `"cashew"`, `"almond"`, or `"nut"`.
3. **`test_e2e_vision_to_recipe_flow`:**
   * *Input:* Sample test image (`tests/assets/sample_fridge.jpg`) + text `"also have heavy cream"`.
   * *Assertions:* Vision tool completes in `< 12s`, `moondream` unloads from memory, and final recipe incorporates both visual and text ingredients.
