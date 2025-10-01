"""
FastAPI backend for DataGuard AI
"""

import sys
import os
from pathlib import Path

#add the project root to sys.path
project_root = Path(__file__).parent.parent.parent  # Goes up to DataGuard-AI folder
sys.path.insert(0, str(project_root))

print(f"🔧 Python path fixed. Project root: {project_root}")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import logging
from typing import Dict, Any

#import your modules
try:
    from src.data_loader import FileDataLoader
    from src.preprocess import TextPreprocessor, ComplianceAnalyzer
    from src.models.risk_assessor import RiskAssessor
    from src.utils.validators import PIIValidator, SecurityValidator

    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")


    # Create simple fallbacks for testing
    class FileDataLoader:
        def extract_text_from_file(self, path):
            return "Sample text from file"


    class TextPreprocessor:
        def detect_pii_patterns(self, text):
            # Simple regex-based PII detection
            import re
            detected = {}
            # Email pattern
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            if emails:
                detected['email'] = emails
            # Phone pattern
            phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
            if phones:
                detected['phone'] = phones
            return detected


    class ComplianceAnalyzer:
        def __init__(self, rules):
            self.rules = rules

        def generate_compliance_report(self, detected_pii, risk_score, filename):
            return {
                'filename': filename,
                'risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
                'recommendations': ['Review detected PII items'] if detected_pii else ['No issues found']
            }


    class RiskAssessor:
        def __init__(self, rules):
            self.rules = rules

        def assess_document_risk(self, detected_pii, document_context="general", file_type="txt"):
            risk_score = min(len(detected_pii) * 0.2, 1.0)
            return {
                'final_risk_score': risk_score,
                'risk_level': 'HIGH' if risk_score > 0.7 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
            }


    class PIIValidator:
        pass


    class SecurityValidator:
        pass

# Remove the problematic routes import for now
# from src.api.routes import router as api_router

# Initialize FastAPI app
app = FastAPI(
    title="DataGuard AI API",
    description="Intelligent Privacy Compliance Checker API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Comment out routes for now until we fix the import
# app.include_router(api_router, prefix="/api/v1")

# Initialize components
file_loader = FileDataLoader()
text_processor = TextPreprocessor()
pii_validator = PIIValidator()
security_validator = SecurityValidator()

# Initialize compliance rules and risk assessor
compliance_rules = {
    'gdpr': ['email', 'phone', 'address', 'name', 'ssn', 'passport'],
    'hipaa': ['medical_record_number', 'health_plan_id', 'account_number'],
    'ccpa': ['ssn', 'credit_card', 'passport']
}
risk_assessor = RiskAssessor(compliance_rules)
compliance_analyzer = ComplianceAnalyzer(compliance_rules)

logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DataGuard AI API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": [
            "/",
            "/health",
            "/analyze/text",
            "/test"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DataGuard AI API"}


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify everything is working"""
    return {
        "message": "✅ Server is working!",
        "test_pii_detection": text_processor.detect_pii_patterns("test@example.com, phone: 555-123-4567")
    }


@app.post("/analyze/text")
async def analyze_text(request: Dict[str, Any]):
    """Analyze text directly for PII and compliance risks"""
    try:
        text = request.get('text', '')
        context = request.get('context', 'general')

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided")

        print(f"📝 Analyzing text: {text[:100]}...")

        # Detect PII
        detected_pii = text_processor.detect_pii_patterns(text)
        print(f"🔍 Detected PII: {detected_pii}")

        # Assess risk
        risk_assessment = risk_assessor.assess_document_risk(
            detected_pii=detected_pii,
            document_context=context,
            file_type="txt"
        )

        # Generate compliance report
        compliance_report = compliance_analyzer.generate_compliance_report(
            detected_pii=detected_pii,
            risk_score=risk_assessment['final_risk_score'],
            filename="text_input"
        )

        return JSONResponse(content={
            "status": "success",
            "analysis": {
                "text_length": len(text),
                "pii_detected": detected_pii,
                "risk_assessment": risk_assessment,
                "compliance_report": compliance_report
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file for PII"""
    try:
        print(f"📁 Processing file: {file.filename}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Extract text from file
            extracted_text = file_loader.extract_text_from_file(tmp_file_path)
            print(f"📄 Extracted text length: {len(extracted_text)}")

            # Analyze the text
            request_data = {"text": extracted_text, "context": "file_upload"}
            return await analyze_text(request_data)

        finally:
            # Clean up temporary file
            import os
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting DataGuard AI API server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload for stability
    )