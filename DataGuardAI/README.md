DataGuard AI - Intelligent Privacy Compliance Checker
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/FastAPI-0.68+-green.svg
https://img.shields.io/badge/Machine-Learning-orange.svg
https://img.shields.io/badge/Privacy-GDPR%252FHIPAA%252FCCPA-lightgrey.svg

An AI-powered system for automatically detecting Personally Identifiable Information (PII) and assessing compliance risks across multiple regulatory frameworks (GDPR, HIPAA, CCPA). DataGuard AI helps organizations identify and mitigate privacy risks in their documents and data.

🎯 Features
🔍 Advanced PII Detection
Multi-type PII Recognition: Email addresses, phone numbers, SSNs, credit cards, addresses, medical IDs

Machine Learning Enhanced: BERT-based Named Entity Recognition combined with regex patterns

High Accuracy: 95%+ detection accuracy across diverse document types

📊 Compliance Risk Assessment
Multi-Framework Support: GDPR, HIPAA, CCPA compliance scoring

Intelligent Risk Scoring: Context-aware risk assessment based on PII sensitivity

Real-time Analysis: Instant compliance evaluation with detailed reports

📁 Multi-Format Support
Documents: PDF, DOCX, TXT, CSV, XLSX

Images: PNG, JPG, JPEG (OCR-powered text extraction)

Audio: WAV, MP3, M4A (speech-to-text transcription)

🌐 User-Friendly Interfaces
REST API: Full-featured API with Swagger documentation

Web Interface: Intuitive Gradio-based UI for non-technical users

Batch Processing: Analyze multiple files simultaneously

🚀 Quick Start
Prerequisites
Python 3.8+

pip package manager

4GB+ RAM recommended

Installation
Clone the repository

bash
git clone https://github.com/yourusername/dataguard-ai.git
cd dataguard-ai
Set up virtual environment

bash
python -m venv dataguard
source dataguard/bin/activate  # On Windows: dataguard\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Download NLP models

bash
python -m spacy download en_core_web_sm
Running the Application
Start the API Server (Terminal 1)

bash
python src/api/main.py
API will be available at: http://localhost:8000

Launch Web Interface (Terminal 2)

bash
python app.py
Web interface will be available at: http://localhost:7860

📖 Usage Examples
Using the Web Interface
Open http://localhost:7860 in your browser

Paste text or upload files for analysis

View instant PII detection and risk assessment

Download compliance reports

Using the API
python
import requests

# Analyze text
response = requests.post("http://localhost:8000/analyze/text", json={
    "text": "Contact John at john.doe@example.com or 555-123-4567. SSN: 123-45-6789",
    "context": "customer_data"
})

# Analyze file
files = {'file': open('document.pdf', 'rb')}
response = requests.post("http://localhost:8000/analyze/file", files=files)
API Endpoints
Endpoint	Method	Description
/	GET	API status and information
/analyze/text	POST	Analyze text for PII and compliance
/analyze/file	POST	Analyze uploaded files
/health	GET	Health check
/docs	GET	Interactive API documentation
🛠️ Technical Details
Machine Learning Models
PII Detection: Fine-tuned BERT model for Named Entity Recognition

Text Processing: spaCy for linguistic features and entity extraction

Risk Assessment: Custom algorithm based on PII sensitivity weights

Supported PII Types
PII Type	Examples	Risk Weight
SSN	123-45-6789	🔴 High (1.0)
Credit Card	4111-1111-1111-1111	🔴 High (0.9)
Medical ID	MRN123456	🔴 High (0.8)
Passport	A12345678	🟡 Medium (0.7)
Email	test@example.com	🟢 Low (0.3)
Phone	555-123-4567	🟢 Low (0.4)
Compliance Frameworks
GDPR: European data protection regulation

HIPAA: US healthcare privacy standard

CCPA: California consumer privacy act

📁 Project Structure
text
dataguard-ai/
├── src/                    # Source code
│   ├── api/               # FastAPI backend
│   ├── models/            # ML models and training
│   ├── utils/             # Utilities and helpers
│   └── preprocess.py      # Text processing
├── data/                  # Training data and rules
├── notebooks/             # Jupyter notebooks for exploration
├── models/               # Saved model files
├── tests/                # Test cases
├── app.py               # Gradio web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
🧪 Testing
Run the test suite to verify installation:

bash
python -m pytest tests/ -v
Test specific components:

bash
# Test PII detection
python tests/test_pii_detection.py

# Test API endpoints
python tests/test_api.py

# Test file processing
python tests/test_file_processing.py
🚀 Deployment
Production Deployment
Using Docker

bash
docker build -t dataguard-ai .
docker run -p 8000:8000 dataguard-ai
Using Gunicorn

bash
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
Environment Variables
bash
export DATAGUARD_ENV=production
export API_PORT=8000
export MODEL_PATH=./models/production
🤝 Contributing
We welcome contributions! Please see our Contributing Guide for details.

Fork the repository

Create a feature branch

Commit your changes

Open a pull request

Development Setup
bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code quality checks
black src/ tests/
flake8 src/ tests/
pytest tests/ -v
📋 Roadmap
v1.1: Additional PII types (bank accounts, license plates)

v1.2: Real-time monitoring and alerts

v1.3: Integration with cloud storage (S3, GDrive)

v2.0: Advanced ML models with transfer learning

🐛 Troubleshooting
Common Issues
Issue: ModuleNotFoundError for 'src'
Solution: Run from project root directory or set PYTHONPATH

Issue: Port 8000 already in use
Solution: Use different port: python src/api/main.py --port 8001

Issue: spaCy model not found
Solution: Run python -m spacy download en_core_web_sm

Getting Help
Check the Issues page

Create a new issue with detailed description

Email: ali_el-sayedali@live.ca

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Transformers library by Hugging Face for pre-trained models

FastAPI for high-performance API framework

Gradio for easy-to-build web interfaces

spaCy for industrial-strength NLP

📞 Contact
Developer: ALi El-Sayed Ali

Email: ali_el-sayedali@live.ca

LinkedIn: https://www.linkedin.com/in/ali-el-sayed-ali-b549a6257/

