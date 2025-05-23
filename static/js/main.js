document.addEventListener('DOMContentLoaded', function() {
    // Tab navigation
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs
            tabLinks.forEach(tab => tab.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to the clicked tab
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    function handleMethodChange(summaryTypeSelect, modelSelect) {
        const method = summaryTypeSelect.value;
        const abstractiveOptGroup = modelSelect.querySelector('optgroup[label="Abstractive"]');
        const extractiveOptGroup = modelSelect.querySelector('optgroup[label="Extractive"]');
    
        if (method === 'abstractive') {
            abstractiveOptGroup.style.display = 'block';
            extractiveOptGroup.style.display = 'none';
            // Reset to default abstractive model
            modelSelect.value = 't5-small';
        } else {
            abstractiveOptGroup.style.display = 'none';
            extractiveOptGroup.style.display = 'block';
            // Reset to default extractive model
            modelSelect.value = 'bert';
        }
    }

    const textMethodSelect = document.getElementById('summary_type_text');
    const textModelSelect = document.getElementById('model_text');
    const fileMethodSelect = document.getElementById('summary_type_file');
    const fileModelSelect = document.getElementById('model_file');

    handleMethodChange(textMethodSelect, textModelSelect);
    handleMethodChange(fileMethodSelect, fileModelSelect);

    textMethodSelect.addEventListener('change', () => handleMethodChange(textMethodSelect, textModelSelect));
    fileMethodSelect.addEventListener('change', () => handleMethodChange(fileMethodSelect, fileModelSelect));
    
    // Text Summarization Form
    const textForm = document.getElementById('textForm');
    
    if (textForm) {
        textForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('text').value;
            const summaryType = document.getElementById('summary_type_text').value;
            const model = document.getElementById("model_text").value;
            const maxLength = document.getElementById('max_length_text').value;
            
            if (text.trim().length < 10) {
                alert('Please enter at least 10 characters of text.');
                return;
            }
            
            showSpinner();
            
            try {
                const response = await fetch(`/summarize/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        summary_type: summaryType,
                        model: model,
                        max_length: parseInt(maxLength)
                    })
                });
                
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'An error occurred during summarization');
                }
                
                const data = await response.json();
                
                // Display results
                document.getElementById('text-method').textContent = data.summary_type;
                document.getElementById('text-originalLength').textContent = data.original_length;
                document.getElementById('text-summaryLength').textContent = data.summary_length;
                document.getElementById('text-summary').textContent = data.summary;
                
                document.getElementById('text-results').style.display = 'block';
                
            } catch (error) {
                alert(`Error: ${error.message}`);
                console.error('Error:', error);
            } finally {
                hideSpinner();
            }
        });
    }
    
    // File Summarization Form
    const fileForm = document.getElementById('fileForm');
    
    if (fileForm) {
        fileForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('file');
            const summaryType = document.getElementById('summary_type_file').value;
            const model = document.getElementById('model_file').value;
            const maxLength = document.getElementById('max_length_file').value;
            
            if (!fileInput.files || fileInput.files.length === 0) {
                alert('Please select a PDF file to upload.');
                return;
            }
            
            const file = fileInput.files[0];
            
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                alert('Please upload a PDF file.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('summary_type', summaryType);
            formData.append('model', model);
            formData.append('max_length', maxLength);

            const compareEnabled = document.getElementById('compare-toggle').checked;
            const model2 = compareEnabled ? document.getElementById('model2').value : null;
            const reference_summary = compareEnabled ? document.getElementById('reference-summary-file').value : null

            formData.append('compare_enabled', compareEnabled);
            if(compareEnabled) {
                formData.append('model2', model2);
                formData.append('reference_summary',reference_summary)
            }
            
            showSpinner();
            
            try {
                const response = await fetch(`/summarize-pdf`, {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'An error occurred during PDF summarization');
                }
                console.log(response)
                const data = await response.json();
                
                document.getElementById('file-name').textContent = data.filename;
                document.getElementById('file-method').textContent = data.summary_type;
                document.getElementById('file-originalLength').textContent = data.original_length;
                document.getElementById('file-summaryLength').textContent = data.summary_length;
                document.getElementById('file-summary').textContent = data.summary;
                
                document.getElementById('file-results').style.display = 'block';
                console.log(data)
                if (data.comparison) {
                    console.log('Comparison data received:', data.comparison);
                    
                    document.getElementById('comparison-results').style.display = 'block';
                    
                    document.getElementById('model1-name').textContent = data.comparison.model1;
                    document.getElementById('model2-name').textContent = data.comparison.model2;
                    
                    document.getElementById('summary1').textContent = data.summary;
                    document.getElementById('summary2').textContent = data.comparison.summary2;
                    
                    const comparisonScores = data.comparison.scores;
                    console.log('Scores object:', comparisonScores);
                    
                    const model1Scores = comparisonScores.model1_scores;
                    const model2Scores = comparisonScores.model2_scores;
                    
                    // Scores visualization
                    const scoresDiv = document.querySelector('#comparison-results .scores');
                    scoresDiv.innerHTML = `
                        <h3>Evaluation Scores</h3>
                        <div class="table-responsive">
                        <table>
                        <thead>
                            <tr>
                            <th>Model</th>
                            <th>ROUGE-1</th>
                            <th>ROUGE-2</th>
                            <th>ROUGE-L</th>
                            <th>BLEU-1</th>
                            <th>BLEU-2</th>
                            <th>BLEU-4</th>
                            <th>BERTScore-Precision</th>
                            <th>BERTScore-Recall</th>
                            <th>BERTScore-F1</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <td>${data.comparison.model1}</td>
                            <td>${(model1Scores.rouge1 * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.rouge2 * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.rougeL * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bleu1 * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bleu2 * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bleu4 * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bertscore_precision * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bertscore_recall * 100).toFixed(1)}%</td>
                            <td>${(model1Scores.bertscore_f1 * 100).toFixed(1)}%</td>
                            </tr>
                            <tr>
                            <td>${data.comparison.model2}</td>
                            <td>${(model2Scores.rouge1 * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.rouge2 * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.rougeL * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bleu1 * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bleu2 * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bleu4 * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bertscore_precision * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bertscore_recall * 100).toFixed(1)}%</td>
                            <td>${(model2Scores.bertscore_f1 * 100).toFixed(1)}%</td>
                            </tr>
                        </tbody>
                        </table>
                        </div>
                        <div class="recommendation">
                            <h4>Recommendation:</h4>
                            <p>Based on ROUGE-L scores, the recommended model is: <strong>${comparisonScores.recommended_method.replace('Model 1', data.comparison.model1).replace('Model 2', data.comparison.model2)}</strong></p>
                        </div>
                    `;
                } else {
                    console.log('No comparison data in response');
                }
  
                
            } catch (error) {
                alert(`Error: ${error.message}`);
                console.error('Error:', error);
            } finally {
                hideSpinner();
            }
        });
    }
    
    // Compare Summaries Form
    const compareForm = document.getElementById('compareForm');
    
    if (compareForm) {
        compareForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('compare-text').value;
            const referenceSummary = document.getElementById('reference-summary').value;
            const maxLength = document.getElementById('max_length_compare').value;
            const extractiveModel = document.getElementById('extractive_model').value;
            const abstractiveModel = document.getElementById('abstractive_model').value;
            
            if (text.trim().length < 10) {
                alert('Please enter at least 10 characters of text.');
                return;
            }
            
            if (referenceSummary.trim().length < 10) {
                alert('Please enter a reference summary with at least 10 characters.');
                return;
            }
            
            const requestData = {
                text: text,
                reference_summary: referenceSummary,
                max_length: parseInt(maxLength),
                summary_type: 'abstractive',
                extractive_model: extractiveModel,
                abstractive_model: abstractiveModel
            };
            
            showSpinner();
            
            try {
                const response = await fetch(`/compare-summaries/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'An error occurred during summary comparison');
                }
                
                const data = await response.json();
                
                // Display results
                document.getElementById('extractive-rouge1').textContent = (data.extractive_scores.rouge1 * 100).toFixed(1) + '%';
                document.getElementById('extractive-rouge2').textContent = (data.extractive_scores.rouge2 * 100).toFixed(1) + '%';
                document.getElementById('extractive-rougeL').textContent = (data.extractive_scores.rougeL * 100).toFixed(1) + '%';
                document.getElementById('extractive-bleu1').textContent = (data.extractive_scores.bleu1 * 100).toFixed(1) + '%';
                document.getElementById('extractive-bleu2').textContent = (data.extractive_scores.bleu2 * 100).toFixed(1) + '%';
                document.getElementById('extractive-bleu4').textContent = (data.extractive_scores.bleu4 * 100).toFixed(1) + '%';
                document.getElementById('extractive-bertscore_f1').textContent = (data.extractive_scores.bertscore_f1 * 100).toFixed(1) + '%';
                document.getElementById('extractive-bertscore_precision').textContent = (data.extractive_scores.bertscore_precision * 100).toFixed(1) + '%';
                document.getElementById('extractive-bertscore_recall').textContent = (data.extractive_scores.bertscore_recall * 100).toFixed(1) + '%';
                
                document.getElementById('abstractive-rouge1').textContent = (data.abstractive_scores.rouge1 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-rouge2').textContent = (data.abstractive_scores.rouge2 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-rougeL').textContent = (data.abstractive_scores.rougeL * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bleu1').textContent = (data.abstractive_scores.bleu1 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bleu2').textContent = (data.abstractive_scores.bleu2 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bleu4').textContent = (data.abstractive_scores.bleu4 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bertscore_f1').textContent = (data.abstractive_scores.bertscore_f1 * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bertscore_precision').textContent = (data.abstractive_scores.bertscore_precision * 100).toFixed(1) + '%';
                document.getElementById('abstractive-bertscore_recall').textContent = (data.abstractive_scores.bertscore_recall * 100).toFixed(1) + '%';

                document.getElementById('extractive-model-used').textContent = requestData.extractive_model;
                document.getElementById('abstractive-model-used').textContent = requestData.abstractive_model;
                
                document.getElementById('recommended-method').textContent = data.recommended_method;
                
                document.getElementById('compare-results').style.display = 'block';
                
            } catch (error) {
                alert(`Error: ${error.message}`);
                console.error('Error:', error);
            } finally {
                hideSpinner();
            }
        });
    }
    
    // Spinner functions
    function showSpinner() {
        document.getElementById('spinner').style.display = 'flex';
    }
    
    function hideSpinner() {
        document.getElementById('spinner').style.display = 'none';
    }

    const compareToggle = document.getElementById('compare-toggle');
    const secondModelGroup = document.getElementById('second-model-group');

    compareToggle.addEventListener('change', (e) => {
    secondModelGroup.style.display = e.target.checked ? 'block' : 'none';
    });

});