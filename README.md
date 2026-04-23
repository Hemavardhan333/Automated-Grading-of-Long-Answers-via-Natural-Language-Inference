<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Chart.js-4.x-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧠 ALAG Engine — Automated Long Answer Grading System

> **An end-to-end NLP-powered grading pipeline that leverages Transformer-based semantic similarity to evaluate long-form student submissions against professor-defined rubrics, delivered through a real-time enterprise analytics dashboard.**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How the Model Works](#how-the-model-works)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Analytics Dashboard](#analytics-dashboard)
- [Sample Data](#sample-data)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional long-answer grading is labour-intensive, subjective, and does not scale. **ALAG Engine** addresses this by implementing a zero-shot semantic entailment pipeline:

1. A professor uploads a **reference answer** (the gold-standard rubric source).
2. The system **auto-generates a structured rubric** by decomposing the reference into discrete conceptual checkpoints using NLTK sentence tokenization and keyword extraction.
3. Student submissions (individual or **batched via ZIP**) are encoded into dense vector representations using a **Sentence-BERT Bi-Encoder** (`all-MiniLM-L6-v2`).
4. Cosine similarity scores between rubric embeddings and student answer windows are computed, then passed through a **configurable sigmoid activation function** to produce calibrated grades.
5. Results are rendered on a **real-time analytics dashboard** with class-wide distribution curves and concept attrition heatmaps.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (Browser)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  File Upload │  │  Strictness  │  │   Chart.js Viz  │ │
│  │  Drop Zones  │  │  Calibrator  │  │  (Distribution  │ │
│  │  (PDF/ZIP)   │  │  (Sigmoid k) │  │   + Heatmap)    │ │
│  └──────┬───────┘  └──────┬───────┘  └────────▲────────┘ │
│         │                 │                    │          │
│         └────────┬────────┘                    │          │
│                  ▼                             │          │
│         ┌────────────────┐                     │          │
│         │  POST /grade   │─────────────────────┘          │
│         │  (FormData)    │     JSON Response               │
│         └────────┬───────┘                                │
└──────────────────┼────────────────────────────────────────┘
                   │  HTTP
┌──────────────────▼────────────────────────────────────────┐
│                  Flask Backend (app.py)                    │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  PDF / DOCX  │  │   Rubric     │  │  SentenceBERT   │ │
│  │  Text Extract │  │  Generator   │  │  Bi-Encoder     │ │
│  │  (PyMuPDF)   │  │  (NLTK)      │  │  (MiniLM-L6)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────────┘ │
│         │                 │                  │            │
│         ▼                 ▼                  ▼            │
│  ┌────────────────────────────────────────────────────┐   │
│  │        Cosine Similarity + Sigmoid Grading         │   │
│  │     score = Σ (weight × σ(15·(sim - threshold)))   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                           │
│  Device: CUDA (auto) │ CPU (fallback)                     │
└───────────────────────────────────────────────────────────┘
```

---

## How the Model Works

The ALAG grading pipeline operates in **five sequential stages**, transforming raw document uploads into calibrated numerical grades. This section describes each stage in technical depth.

### Stage 1 — Document Ingestion & Text Extraction

When files are uploaded via the dashboard, the backend reads the raw bytes and dispatches to a format-specific extraction engine:

| Format | Engine | Method |
|---|---|---|
| `.pdf` | **PyMuPDF** (`fitz`) | Extracts text blocks, sorted by vertical coordinate (`y`-offset) to preserve reading order across multi-column layouts. |
| `.docx` / `.doc` | **python-docx** | Iterates over `paragraphs` and joins their text content. |
| `.txt` / `.csv` | **bytes decode** | Attempts UTF-8 decoding, falls back to Latin-1 for non-standard encodings. |
| `.zip` (batch) | **zipfile** | Iterates over all entries (filtering out `__MACOSX` and hidden files), then dispatches each inner file to the appropriate extractor above. |

This produces a clean, flat string of text for the reference answer and for each student submission.

---

### Stage 2 — Automatic Rubric Generation

The professor's reference answer is decomposed into discrete **rubric items** (conceptual checkpoints) using the `RubricGenerator` class:

1. **Sentence Tokenization** — NLTK's Punkt tokenizer (`sent_tokenize`) splits the reference into individual sentences.
2. **Short Sentence Filtering** — Sentences with fewer than 2 words are discarded (e.g., headings or labels).
3. **Keyword Extraction** — A predefined set of English stopwords (`the`, `a`, `is`, `and`, `to`, `in`, `of`, etc.) is removed from each sentence. The first **4 non-stopwords** are extracted and title-cased to form a human-readable **concept label** (e.g., `"Deep Learning Subset Machine..."`).
4. **Rubric Construction** — Each sentence becomes a rubric item containing:
   - `concept` — the keyword-derived label (displayed in the heatmap)
   - `hypothesis` — the original full sentence (used for embedding)

**Example:**

> Reference: *"Deep learning is a subset of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features from the raw input."*

| Rubric Item | Concept Label | Hypothesis (Full Sentence) |
|---|---|---|
| 1 | `Deep Learning Subset Machine...` | Deep learning is a subset of machine learning based on artificial neural networks. |
| 2 | `It Uses Multiple Layers...` | It uses multiple layers to progressively extract higher-level features from the raw input. |

---

### Stage 3 — Embedding & Contextual Sliding Windows

This stage converts both the rubric facts and the student's answer into dense vector representations for comparison.

#### 3a. Rubric Fact Factualization

Before encoding, each rubric hypothesis is **factualized** — a regex-based preprocessing step strips pedagogical meta-verbs that do not carry semantic meaning:

```
Pattern: ^(?:correctly\s+)?(?:cites|relates|explains|mentions|states|describes)(?:\s+that)?\s+
```

For example: *"Correctly explains that deep learning uses neural networks"* → *"deep learning uses neural networks"*

This prevents the similarity model from matching on structural phrasing rather than actual content.

#### 3b. Contextual Sliding Window Construction

Student answers are tokenized into sentences, then grouped into **overlapping 2-sentence windows**:

```
Sentences: [S₁, S₂, S₃, S₄]
Windows:   [S₁+S₂, S₂+S₃, S₃+S₄]
```

This sliding window approach captures **cross-sentence reasoning** — a concept may span two adjacent sentences, and evaluating single sentences in isolation would miss this contextual dependency.

If the student answer contains only 1 sentence, that sentence is used directly as the sole window.

#### 3c. Dense Vector Encoding

Both the factualized rubric facts and student windows are encoded into **384-dimensional dense vectors** using the `all-MiniLM-L6-v2` Sentence-BERT Bi-Encoder:

```python
rubric_embeddings = embedder.encode(facts, convert_to_tensor=True)       # Shape: [R, 384]
student_embeddings = embedder.encode(student_windows, convert_to_tensor=True)  # Shape: [W, 384]
```

Where `R` = number of rubric items and `W` = number of student windows.

> **Note:** The research notebook (`ALAG_Best.ipynb`) uses the heavier **DeBERTa-v3 Zero-Shot NLI** classifier (`MoritzLaurer/deberta-v3-base-zeroshot-v1`) for higher QWK accuracy. The dashboard deployment uses **Sentence-BERT** (`all-MiniLM-L6-v2`) for faster inference at the cost of some precision.

---

### Stage 4 — Cosine Similarity Matrix & Sigmoid Grade Calibration

#### 4a. Best-Match Extraction

A full **cosine similarity matrix** is computed between all student windows and all rubric facts:

```python
cosine_scores = util.cos_sim(student_embeddings, rubric_embeddings)  # Shape: [W, R]
```

For each rubric fact `rⱼ`, the **maximum similarity** across all student windows is extracted:

```
sim(rⱼ) = max(cosine_scores[:, j])
```

This means a rubric concept is evaluated by its **best-matching** student window — the student only needs to address each concept once, and it can appear anywhere in their answer.

#### 4b. Sigmoid Strictness Transformation

Raw cosine similarity scores (`sim ∈ [0.0, 1.0]`) are transformed into grade multipliers using a **logistic (sigmoid) function** controlled by two parameters:

```
σ(sim) = 1 / (1 + e^(−k · (sim − threshold)))
```

Where:
- **`k` (steepness)** = 15 (hardcoded in the dashboard; adjustable in the notebook)
- **`threshold`** = dynamically mapped from the user's strictness slider:

```
threshold = 0.30 + ((strictness − 1) / 24.0) × 0.55
```

| Strictness Slider | Threshold | Effect |
|---|---|---|
| 1.0 (Lenient) | 0.30 | Low cosine scores still receive full credit |
| 8.5 (Default) | ~0.47 | Balanced grading |
| 25.0 (Strict) | 0.85 | Only near-perfect semantic matches receive credit |

The sigmoid function acts as a **soft binary gate** — similarities above the threshold rapidly approach `1.0` (full credit), while those below it rapidly approach `0.0` (no credit). The steep `k=15` ensures a crisp transition without hard cutoffs.

#### 4c. Weighted Score Accumulation

Each rubric item is assigned an equal weight:

```
item_weight = max_marks / |rubric|
```

The grade contribution from each rubric item is:

```
grade(rⱼ) = item_weight × σ(sim(rⱼ))
```

The total raw grade is the sum:

```
total_grade = Σⱼ grade(rⱼ)
```

---

### Stage 5 — Final Grade Normalization & Aggregation

#### 5a. Individual Student Grade

The raw total is normalized and rounded to the nearest **0.5 increment**:

```
final_grade = min(⌊(total_grade / max_marks) × max_marks × 2⌋ / 2, max_marks)
```

This produces clean half-mark grades (e.g., 6.0, 6.5, 7.0).

#### 5b. Batch Aggregation

When processing a ZIP batch of `N` students:

- **Class Average** = mean of all `N` final grades
- **Highest Score** = maximum across all `N` final grades
- **Concept Attrition** = for each rubric item, the average cosine similarity across all students, reported as a percentage. This reveals **which concepts the class understood** and **which concepts were systematically missed**.

---

### Worked Example

**Reference Answer:**
> *"Deep learning is a subset of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features from the raw input."*

**Student Answer:**
> *"Deep learning uses neural networks with many layers. These layers help in extracting features from the data step by step."*

**Settings:** `max_marks = 10.0`, `strictness = 8.5` → `threshold ≈ 0.47`

| Step | Rubric 1 (`Deep Learning Subset Machine...`) | Rubric 2 (`It Uses Multiple Layers...`) |
|---|---|---|
| Best cosine sim | 0.82 | 0.78 |
| σ(15 × (sim − 0.47)) | ≈ 1.00 | ≈ 1.00 |
| item_weight | 5.0 | 5.0 |
| grade contribution | 5.0 | 5.0 |

**Total raw grade:** 10.0 → **Final: 10.0 / 10.0**

If the student had *missed* Rubric 2 entirely (sim ≈ 0.15):

| σ(15 × (0.15 − 0.47)) | ≈ 0.008 |
|---|---|
| grade contribution | 0.04 |

**Total:** 5.04 → **Rounded: 5.0 / 10.0** — correctly penalising the missing concept.

---

## Key Features

| Feature | Description |
|---|---|
| **Auto-Rubric Generation** | Decomposes any reference answer into granular conceptual checkpoints via sentence tokenization and stopword-filtered keyword extraction. |
| **Semantic Similarity Grading** | Uses `all-MiniLM-L6-v2` Sentence-BERT to compute dense cosine similarity between rubric concepts and student answer windows. |
| **Sigmoid Strictness Control** | A tunable `k`-value slider (1–25) maps to a logistic threshold (0.30–0.85), allowing professors to adjust grading curve sensitivity in real time. |
| **Batch Processing** | Upload a `.zip` archive containing multiple student PDFs/DOCX files for bulk evaluation in a single request. |
| **Configurable Max Score** | Professors can define any maximum score for an assignment; all grades scale proportionally. |
| **Grade Distribution Chart** | Live histogram plotting the class-wide score distribution across dynamically computed bins (Chart.js). |
| **Concept Attrition Heatmap** | Per-rubric-item confidence bar chart identifying which concepts students mastered vs. missed. |
| **CSV/Excel Export** | One-click export of `Student ID`, `Score`, and `Percentage` to a downloadable `.csv` file. |
| **Multi-Format Ingestion** | Supports `.pdf`, `.docx`, `.doc`, `.txt`, and `.csv` input files with automatic encoding detection. |
| **GPU Acceleration** | Automatically detects and utilises CUDA-capable GPUs for embedding computation; falls back to CPU seamlessly. |

---

## Technology Stack

### Backend
| Component | Technology |
|---|---|
| Web Framework | Flask 2.x with Flask-CORS |
| NLP Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Deep Learning Runtime | PyTorch 2.0+ (CUDA / CPU) |
| Text Extraction | PyMuPDF (`fitz`) for PDFs, `python-docx` for DOCX |
| Tokenization | NLTK `sent_tokenize` (Punkt) |
| Math Utilities | NumPy (Sigmoid activation) |

### Frontend
| Component | Technology |
|---|---|
| Markup | Semantic HTML5 |
| Styling | Vanilla CSS with CSS Custom Properties (design tokens) |
| Charts | Chart.js 4.x (Line + Horizontal Bar) |
| Typography | Google Fonts (Inter) |
| UX Patterns | Async Fetch API, FormData, progress animations |

---

## Project Structure

```
ALAG-Engine/
│
├── app.py                          # Flask backend — grading API server
├── index.html                      # Dashboard frontend entry point
├── script.js                       # Client-side logic, chart rendering, API calls
├── style.css                       # Premium SaaS-grade UI styling
│
├── ALAG_Modified.ipynb             # Jupyter notebook — model experimentation
│
├── Professor Q&A Reference - *.pdf # Sample reference rubric document
├── STU_2026_111_Submission_*.pdf   # Sample individual student submission
├── ALAG_Student_Submissions_Batch.zip  # Sample batch archive (10 students)
│
└── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites

- Python **3.10+**
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### 1. Clone the Repository

```bash
git clone https://github.com/Hemavardhan333/ALAG-Engine.git
cd ALAG-Engine
```

### 2. Install Dependencies

```bash
pip install flask flask-cors torch sentence-transformers PyMuPDF python-docx nltk numpy
```

### 3. Launch the Backend Server

```bash
python app.py
```

Expected console output:
```
=======================================
  INITIALIZING ALAG BACKEND ENGINE
=======================================
Loading all-MiniLM-L6-v2 Bi-Encoder Model (Will use CPU/GPU automatically)...
-> SentenceTransformer Loaded Active Engine Ready.
 * Running on http://127.0.0.1:5000
```

### 4. Open the Dashboard

Navigate to **`http://127.0.0.1:5000`** in your browser. The Flask server serves the frontend assets (`index.html`, `script.js`, `style.css`) directly.

---

## Usage

### Single Submission Grading

1. Upload a **Reference Q&A PDF** (professor's gold-standard answer).
2. Upload a **Student Submission** (`.pdf`, `.docx`, or `.txt`).
3. Adjust the **Strictness Slider** to calibrate grading sensitivity.
4. Set the **Maximum Score** for the assignment.
5. Click **"Initialize Grading Engine"** to process.

### Batch Grading (Recommended for Classrooms)

1. Prepare a **`.zip` archive** containing all student submissions.
2. Upload the ZIP as the student file — the engine will automatically extract and evaluate each submission individually.
3. View aggregated class analytics: distribution curve, concept attrition, highest score, and class average.

### Export Results

Click **"Export Grades to CSV/Excel"** to download a structured spreadsheet with columns: `Student ID`, `Score`, `Percentage`.

---

## API Reference

### `GET /ping`

Health check endpoint.

**Response:**
```json
{
  "status": "Online",
  "device": "GPU"
}
```

### `POST /grade`

Core grading endpoint. Accepts multipart form data.

**Request Parameters:**

| Field | Type | Description |
|---|---|---|
| `reference` | File | Professor's reference answer document |
| `student` | File | Student submission (single file or `.zip` batch) |
| `strictness` | Float | Sigmoid threshold calibration (1.0–25.0) |
| `max_score` | Float | Maximum achievable score for the assignment |

**Response:**
```json
{
  "final_grade": 75.0,
  "max_marks": 100.0,
  "total_graded": 10,
  "class_avg": 75.0,
  "student_grades": [
    { "student_id": "STU_2026_101", "score": 58.0, "percentage": 58.0 },
    { "student_id": "STU_2026_102", "score": 85.0, "percentage": 85.0 }
  ],
  "rubric_names": ["Core Alignment...", "Vocabulary Check...", "Logical Chain..."],
  "attrition_rates": [88, 45, 12],
  "logs": [
    { "concept": "Core Alignment...", "confidence": 88 }
  ]
}
```

---

## Analytics Dashboard

The browser-based dashboard provides three tiers of real-time analytics:

### Metric Cards
- **Total Papers Graded** — Count of successfully processed submissions
- **Class Average** — Mean score across all students (scaled to max marks)
- **Highest Score** — Top-performing student's score in the batch

### Class Grade Distribution
A line chart with filled gradient plotting the number of students per score bin. Bins are dynamically computed relative to the configured maximum score.

### Concept Attrition Heatmap
A horizontal bar chart showing the average AI confidence (cosine similarity × 100) for each rubric concept across all students. Color-coded:
- 🟢 **Green (>75%)** — Concept well-mastered by the class
- 🟡 **Amber (45–75%)** — Partial understanding
- 🔴 **Red (<45%)** — Significant knowledge gap detected

---

## Sample Data

The repository includes ready-to-use test data for immediate evaluation:

| File | Description |
|---|---|
| `Professor Q&A Reference - Deep Learning & NLP.pdf` | Reference rubric covering Deep Learning and NLP concepts |
| `STU_2026_111_Submission_DL_NLP.pdf` | Single student submission for individual grading test |
| `ALAG_Student_Submissions_Batch.zip` | Batch of 10 student submissions for classroom-scale evaluation |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improved-embedding`)
3. Commit your changes (`git commit -m 'feat: add cross-encoder reranking'`)
4. Push to the branch (`git push origin feature/improved-embedding`)
5. Open a Pull Request

---

## License

This project is developed as part of an academic capstone. All rights reserved by the author.

---

<p align="center">
  <strong>Built with 🧠 Transformer Intelligence</strong><br/>
  <sub>Powered by Sentence-BERT · PyTorch · Flask · Chart.js</sub>
</p>
