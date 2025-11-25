// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const manualTab = document.getElementById('manualTab');
    const uploadTab = document.getElementById('uploadTab');
    const manualForm = document.getElementById('manualForm');
    const uploadForm = document.getElementById('uploadForm');

    function switchTab(activeTab, activeForm) {
        // Update tabs
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.classList.remove('active', 'border-blue-500', 'text-blue-600');
            tab.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
        });
        activeTab.classList.add('active', 'border-blue-500', 'text-blue-600');
        activeTab.classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');

        // Update forms
        document.querySelectorAll('.tab-content').forEach(form => {
            form.classList.add('hidden');
        });
        activeForm.classList.remove('hidden');
    }

    manualTab.addEventListener('click', () => switchTab(manualTab, manualForm));
    uploadTab.addEventListener('click', () => switchTab(uploadTab, uploadForm));

    // Form submission for manual input
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading state
            const submitButton = predictionForm.querySelector('button[type="submit"]');
            const originalButtonText = submitButton.innerHTML;
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';

            try {
                // Get form data
                const formData = new FormData(predictionForm);
                const data = {};
                formData.forEach((value, key) => {
                    data[key] = parseFloat(value) || value;
                });

                // Send prediction request
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                const result = await response.json();
                displayResults(result);
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while processing your request.');
            } finally {
                // Reset button state
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
            }
        });
    }

    // File upload handling
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');

    if (fileInput && fileList) {
        fileInput.addEventListener('change', async function(e) {
            const files = e.target.files;
            if (files.length > 0) {
                fileList.innerHTML = '';
                const formData = new FormData();
                
                for (let i = 0; i < files.length; i++) {
                    formData.append('files', files[i]);
                    
                    const fileItem = document.createElement('div');
                    fileItem.className = 'flex items-center justify-between p-2 bg-gray-50 rounded mb-2';
                    fileItem.innerHTML = `
                        <div class="flex items-center">
                            <i class="fas fa-file-medical text-blue-500 mr-3"></i>
                            <span>${files[i].name}</span>
                        </div>
                        <span class="text-sm text-gray-500">${(files[i].size / 1024).toFixed(1)} KB</span>
                    `;
                    fileList.appendChild(fileItem);
                }

                try {
                    // Show loading state
                    const uploadButton = document.querySelector('label[for="fileInput"]');
                    const originalButtonText = uploadButton.innerHTML;
                    uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Uploading...';
                    uploadButton.classList.add('opacity-50', 'cursor-not-allowed');

                    // Upload files
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData,
                    });

                    const result = await response.json();
                    
                    if (response.ok) {
                        // Process extracted data
                        processExtractedData(result.extracted_data);
                    } else {
                        throw new Error(result.error || 'Failed to process files');
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while uploading files.');
                } finally {
                    // Reset button state
                    const uploadButton = document.querySelector('label[for="fileInput"]');
                    uploadButton.innerHTML = '<i class="fas fa-cloud-upload-alt mr-2"></i>Choose Files';
                    uploadButton.classList.remove('opacity-50', 'cursor-not-allowed');
                }
            }
        });
    }

    // Function to display results
    function displayResults(data) {
        const resultsSection = document.getElementById('results');
        const riskLevel = document.getElementById('riskLevel');
        const probability = document.getElementById('probability');
        const recommendation = document.getElementById('recommendation');
        const detailedResults = document.getElementById('detailedResults');

        // Update risk level with appropriate styling
        riskLevel.textContent = data.risk_level;
        riskLevel.className = 'text-4xl font-bold mb-2 ' + 
            (data.risk_level === 'High' ? 'text-red-600' : 
             data.risk_level === 'Moderate' ? 'text-yellow-600' : 'text-green-600');

        // Update probability
        probability.textContent = `${Math.round(data.probability * 100)}%`;
        probability.className = 'text-4xl font-bold mb-2 ' + 
            (data.probability > 0.7 ? 'text-red-600' : 
             data.probability > 0.4 ? 'text-yellow-600' : 'text-green-600');

        // Update recommendation
        let recommendationText = '';
        if (data.risk_level === 'High') {
            recommendationText = 'Please consult with a healthcare professional as soon as possible.';
        } else if (data.risk_level === 'Moderate') {
            recommendationText = 'Consider lifestyle changes and regular check-ups.';
        } else {
            recommendationText = 'Maintain a healthy lifestyle and regular exercise.';
        }
        recommendation.textContent = recommendationText;

        // Show results section
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // Function to process extracted data from documents
    function processExtractedData(data) {
        // This function would be called after OCR processing
        // For now, we'll just log the data
        console.log('Extracted data:', data);
        
        // Here you would populate the form with extracted data
        // and then trigger the prediction
    }
});