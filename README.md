# Automated-Grading-of-Long-Answers-via-Natural-Language-Inference
An NLP pipeline that grades long answers using transformer-based Natural Language Inference (NLI). It breaks reference text into key rubric points and uses semantic entailment to check if a student’s answer captures the meaning, then applies a logistic model for fair, explainable scoring.

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
