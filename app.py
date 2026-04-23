import os
import torch
import fitz
import re
import numpy as np
import io
import zipfile
import docx

import nltk
# Crucial: Secure local NLTK tokenizer resources dynamically on boot since Colab isn't managing it
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
CORS(app) 

print("=======================================")
print("  INITIALIZING ALAG BACKEND ENGINE")
print("=======================================")
print("Loading all-MiniLM-L6-v2 Bi-Encoder Model (Will use CPU/GPU automatically)...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("-> SentenceTransformer Loaded Active Engine Ready.")

class RubricGenerator:
    def generate_rubric(self, reference_answer):
        sentences = sent_tokenize(reference_answer)
        rubric = []
        stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "that", "it", "with", "for", "on", "by", "as", "these", "those"}
        for sent in sentences:
            if len(sent.split()) < 2: continue 
            words = sent.split()
            key_words = [w for w in words if w.lower() not in stopwords]
            concept = " ".join(key_words[:min(4, max(1, len(key_words)))]).title() + "..."
            rubric.append({"concept": concept, "hypothesis": sent})
        return rubric

rubric_engine = RubricGenerator()

def extract_text_from_bytes(file_bytes, filename):
    ext = filename.split('.')[-1].lower() if '.' in filename else ''
    full_text = ""
    
    if ext == 'pdf':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: b[1]) 
                for block in blocks:
                    text = block[4].strip()
                    if text: full_text += text + " "
            doc.close()
        except Exception as e:
            print(f"PDF extraction error: {e}")
    elif ext in ['docx', 'doc']:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = " ".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"DOCX extraction error: {e}")
    else:
        # Fallback for txt or csv
        try:
            full_text = file_bytes.decode('utf-8')
        except:
            try:
                full_text = file_bytes.decode('latin-1')
            except:
                pass
                
    return full_text.strip()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "Online", "device": "GPU" if device == 0 else "CPU"})

@app.route('/grade', methods=['POST'])
def grade_submission():
    ref_file = request.files.get('reference')
    student_file = request.files.get('student')
    strictness = float(request.form.get('strictness', 8.5))
    max_marks_input = float(request.form.get('max_score', 8.0))
    
    if not ref_file or not student_file:
        return jsonify({'error': 'Missing PDF files in request payloads!'}), 400
        
    print(f"Incoming Request: Grading newly uploaded files: {ref_file.filename} vs {student_file.filename}")
    
    ref_bytes = ref_file.read()
    student_bytes = student_file.read()
    
    if not ref_bytes or not student_bytes:
        return jsonify({'error': 'One or both uploaded files are completely empty (0 bytes)!'}), 400
        
    ref_text = extract_text_from_bytes(ref_bytes, ref_file.filename)
    if not ref_text:
        return jsonify({'error': 'Could not extract readable text from the reference document.'}), 400
        
    rubric = rubric_engine.generate_rubric(ref_text)
    
    # Process Student file (Detect ZIP archives dynamically)
    student_texts = []
    
    if student_file.filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(io.BytesIO(student_bytes)) as z:
                for name in z.namelist():
                    # Ignore metadata folders inside ZIPs
                    if not name.startswith('__MACOSX') and not name.startswith('.') and '/' not in name:
                        with z.open(name) as f:
                            text = extract_text_from_bytes(f.read(), name)
                            if text.strip():
                                student_texts.append((name.rsplit('.', 1)[0], text))
        except Exception as e:
            return jsonify({'error': f'Failed to extract ZIP payload: {e}'}), 400
    else:
        text = extract_text_from_bytes(student_bytes, student_file.filename)
        if text.strip():
            student_texts.append((student_file.filename.rsplit('.', 1)[0], text))
            
    if not student_texts:
        return jsonify({'error': 'No readable text could be retrieved from the student submissions.'}), 400
    
    # Evaluate across all student submissions (Batched Classroom Output)
    all_final_grades = []
    student_grades_export = []
    total_graded = 0
    aggregate_attrition = {item['concept']: 0.0 for item in rubric}
    
    for s_name, s_text in student_texts:
        student_sentences = sent_tokenize(s_text)
        student_windows = []
        if len(student_sentences) == 1:
            student_windows = [student_sentences[0]]
        elif len(student_sentences) > 1:
            for i in range(len(student_sentences) - 1):
                student_windows.append(student_sentences[i] + " " + student_sentences[i+1])
                
        if not student_windows:
            continue
            
        facts = [re.sub(r"^(?:correctly\s+)?(?:cites|relates|explains|mentions|states|describes)(?:\s+that)?\s+", "", item['hypothesis'], flags=re.IGNORECASE).strip() for item in rubric]
        
        rubric_embeddings = embedder.encode(facts, convert_to_tensor=True)
        student_embeddings = embedder.encode(student_windows, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(student_embeddings, rubric_embeddings)
        
        best_matches_tensor, _ = torch.max(cosine_scores, dim=0)
        best_matches = {fact: max(0.0, best_matches_tensor[i].item()) for i, fact in enumerate(facts)}
                    
        total_grade = 0
        max_marks = max_marks_input
        item_weight = max_marks / len(rubric) if rubric else 0
        
        for item, fact in zip(rubric, facts):
            prob = best_matches[fact]
            # Map strictness (1 to 25) to a threshold (0.3 to 0.85)
            threshold = 0.3 + ((strictness - 1) / 24.0) * 0.55
            logistic_mult = 1 / (1 + np.exp(-15 * (prob - threshold)))
            grade = item_weight * logistic_mult
            total_grade += grade
            aggregate_attrition[item['concept']] += prob
            
        final_grade = min(round((total_grade / max_marks) * max_marks * 2) / 2, max_marks)
        all_final_grades.append(final_grade)
        
        percentage = min((final_grade / max_marks) * 100, 100.0) if max_marks > 0 else 0
        student_grades_export.append({
            "student_id": s_name,
            "score": final_grade,
            "percentage": round(percentage, 2)
        })
        
        total_graded += 1
        
    if total_graded == 0:
        return jsonify({'error': 'Could not parse meaningful data metrics.'}), 400

    class_avg = sum(all_final_grades) / total_graded
    class_avg = round(class_avg, 1)

    # Average attrition across all students
    attrition_rates = []
    rubric_names = []
    logs = []
    
    for item in rubric:
        avg_prob = aggregate_attrition[item['concept']] / total_graded
        confidence_percent = round(avg_prob * 100)
        attrition_rates.append(confidence_percent)
        rubric_names.append(item['concept'])
        logs.append({"concept": item['concept'], "confidence": confidence_percent})
        
    final_grade_returned = all_final_grades[0] if total_graded == 1 else class_avg

    return jsonify({
        "final_grade": final_grade_returned, 
        "max_marks": max_marks_input,
        "logs": logs,
        "attrition_rates": attrition_rates,
        "rubric_names": rubric_names,
        "class_avg": class_avg, 
        "total_graded": total_graded,
        "student_grades": student_grades_export
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)
