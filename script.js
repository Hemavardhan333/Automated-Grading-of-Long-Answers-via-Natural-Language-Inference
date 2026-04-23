document.addEventListener('DOMContentLoaded', () => {

    // --- Strictness Slider Interactivity ---
    const slider = document.getElementById('strictnessSlider');
    const display = document.getElementById('k-valueDisplay');

    slider.addEventListener('input', (e) => {
        display.textContent = parseFloat(e.target.value).toFixed(1);
    });

    // --- File Upload Logic mapping to HTML DOM Nodes ---
    const refFileInput = document.getElementById('ref-file');
    const studentFileInput = document.getElementById('student-file');
    
    // Convert drag areas to click targets
    document.getElementById('drop-zone-1').addEventListener('click', () => refFileInput.click());
    document.getElementById('drop-zone-2').addEventListener('click', () => studentFileInput.click());

    // Update UX element tags with file name cleanly
    refFileInput.addEventListener('change', (e) => {
        if(e.target.files.length > 0) {
            document.querySelector('#drop-zone-1 .subtext').innerHTML = `<span style="color: #10b981; font-weight: 600;">✔️ ${e.target.files[0].name}</span>`;
            document.getElementById('drop-zone-1').style.borderColor = '#10b981';
        }
    });

    studentFileInput.addEventListener('change', (e) => {
        if(e.target.files.length > 0) {
            document.querySelector('#drop-zone-2 .subtext').innerHTML = `<span style="color: #10b981; font-weight: 600;">✔️ ${e.target.files[0].name}</span>`;
            document.getElementById('drop-zone-2').style.borderColor = '#10b981';
        }
    });

    // --- Progress Execution & FETCH API Post Handling ---
    const initBtn = document.getElementById('initialize-btn');
    const progressBarContainer = document.getElementById('progress-container');
    const progressBarFill = document.querySelector('.progress-bar-fill');
    const progressLabel = document.querySelector('.progress-label');
    const progressPercent = document.querySelector('.progress-percent');

    // Chart Object Scopes
    let bellCurveChartInstance = null;
    let heatmapChartInstance = null;
    let lastStudentGrades = [];

    initBtn.addEventListener('click', async () => {
        const refFile = refFileInput.files[0];
        const studentFile = studentFileInput.files[0];

        if (!refFile || !studentFile) {
            alert('Hold on! Please upload both a Reference PDF and a Student PDF to grade.');
            return;
        }

        // Trigger animations
        initBtn.style.display = 'none';
        progressBarContainer.style.display = 'block';
        progressBarFill.style.width = `20%`;
        progressLabel.textContent = `Pinging DeBERTa Server Engine on Port:5000...`;
        progressPercent.textContent = `20%`;

        const formData = new FormData();
        formData.append('reference', refFile);
        formData.append('student', studentFile);
        formData.append('strictness', slider.value);
        
        const maxScoreEl = document.getElementById('maxScoreInput');
        if (maxScoreEl) {
            formData.append('max_score', maxScoreEl.value);
        }

        try {
            // Forward HTTP Post to Local ALAG Flask Engine
            const response = await fetch('http://127.0.0.1:5000/grade', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Backend server declined payload.");
            
            const data = await response.json();
            
            if (data.student_grades) {
                lastStudentGrades = data.student_grades;
            }
            
            // Artificial delay to make UX smooth
            progressBarFill.style.width = `100%`;
            progressLabel.textContent = `Evaluating entails via GPU...`;
            progressPercent.textContent = `100%`;
            
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                initBtn.style.display = 'block';
                initBtn.textContent = 'Process New Batch';
                
                // Update Main Metric Card dynamically!
                document.querySelector('.metric-box.success .metric-value').innerHTML = 
                    `${data.final_grade} <span style="font-size: 1rem; color: #64748b;">/ ${data.max_marks}</span>`;
                
                document.querySelector('.metric-box .metric-value').textContent = data.total_graded;

                let highestScore = data.final_grade;
                if (data.student_grades && data.student_grades.length > 0) {
                    highestScore = Math.max(...data.student_grades.map(s => s.score));
                }
                document.querySelector('.metric-box.highlight .metric-value').textContent = highestScore;

                // Sync the actual PyTorch output to the HTML Analytics Charts!
                renderCharts(data.final_grade, data.max_marks, data.rubric_names, data.attrition_rates);

            }, 500);

        } catch (error) {
            console.error('Request failed:', error);
            alert('Failed to connect to backend! Make sure you are running: `python app.py`');
            progressBarContainer.style.display = 'none';
            initBtn.style.display = 'block';
        }
    });

    const exportBtn = document.getElementById('export-btn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            if (lastStudentGrades.length === 0) {
                alert("No grades available to export yet. Please grade a batch first.");
                return;
            }
            
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "Student ID,Score,Percentage\r\n";
            
            lastStudentGrades.forEach(grade => {
                let row = `"${grade.student_id}",${grade.score},${grade.percentage}%`;
                csvContent += row + "\r\n";
            });
            
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "ALAG_Grades_Export.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // --- Dynamic Chart Rendering Engine ---
    function renderCharts(finalGrade, maxMarks, rubricNames, attritionRates) {
        
        // 1. Chart 1: Bell Curve Distribution
        const ctxBellCurve = document.getElementById('bellCurveChart').getContext('2d');
        if (bellCurveChartInstance) bellCurveChartInstance.destroy(); // Clear existing graph
        
        let gradientFill = ctxBellCurve.createLinearGradient(0, 0, 0, 300);
        gradientFill.addColorStop(0, 'rgba(59, 130, 246, 0.4)'); 
        gradientFill.addColorStop(1, 'rgba(59, 130, 246, 0.05)');

        // Dynamically compute chart bins relative to the configured maximum score
        let dynamicLabels = [];
        let actualSpread = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        
        for (let i = 0; i <= 8; i++) {
            let labelVal = (i / 8) * maxMarks;
            dynamicLabels.push(Number.isInteger(labelVal) ? labelVal.toString() : labelVal.toFixed(1));
        }

        if (lastStudentGrades && lastStudentGrades.length > 0) {
            // Plot true student grade distribution across the 9 bins
            lastStudentGrades.forEach(student => {
                let binIndex = Math.round((student.score / maxMarks) * 8);
                binIndex = Math.max(0, Math.min(8, binIndex));
                actualSpread[binIndex]++;
            });
        } else {
            // Fallback: mock standard distribution for dashboard pre-render
            let normalizedFinal = (finalGrade / maxMarks) * 8;
            actualSpread = actualSpread.map((_, i) => {
                return Math.floor(Math.max(0, 30 - Math.pow(i - normalizedFinal, 2) * 5));
            });
        }

        bellCurveChartInstance = new Chart(ctxBellCurve, {
            type: 'line',
            data: {
                labels: dynamicLabels,
                datasets: [{
                    label: 'Number of Students',
                    data: actualSpread,
                    backgroundColor: gradientFill,
                    borderColor: '#3b82f6',
                    borderWidth: 3,
                    pointBackgroundColor: '#ffffff',
                    pointBorderColor: '#3b82f6',
                    fill: true,
                    tension: 0.4 
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, grid: { color: '#e2e8f0' }, title: { display: true, text: 'No. of Students' } },
                    x: { grid: { display: false }, title: { display: true, text: `Final Score (Maximum: ${maxMarks})` } }
                }
            }
        });

        // 2. Chart 2: True Concept Attrition Heatmap
        const ctxHeatmap = document.getElementById('heatmapChart').getContext('2d');
        if (heatmapChartInstance) heatmapChartInstance.destroy();

        const backgroundColors = attritionRates.map(value => {
            if (value > 75) return '#10b981'; // Mastered
            if (value > 45) return '#f59e0b'; // Medium
            return '#ef4444'; // Poorly captured
        });

        heatmapChartInstance = new Chart(ctxHeatmap, {
            type: 'bar',
            data: {
                labels: rubricNames,
                datasets: [{
                    label: 'AI Confidence Profile (%)',
                    data: attritionRates,
                    backgroundColor: backgroundColors,
                    borderRadius: 4,
                    barPercentage: 0.8
                }]
            },
            options: {
                indexAxis: 'y', 
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { max: 100, beginAtZero: true, title: { display: true, text: 'DeBERTa Sequence Entailment Probability' }, grid: { color: '#e2e8f0' } },
                    y: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 11 }, color: '#475569' } }
                }
            }
        });
    }

    // Initialize Default Graphics
    renderCharts(6.5, 8.0, ['Core Alignment', 'Vocabulary Check', 'Logical Chain'], [88, 45, 12]);
});
