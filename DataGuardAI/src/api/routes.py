"""
API routes for DataGuard AI
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import tempfile
import os

from src.data_loader import FileDataLoader
from src.preprocess import TextPreprocessor, ComplianceAnalyzer
from src.models.pii_detector import PIIDetectorTrainer
from src.models.risk_assessor import RiskAssessor
from src.utils.validators import PIIValidator, SecurityValidator

router = APIRouter()

# Initialize components (in production, use dependency injection)
file_loader = FileDataLoader()
text_processor = TextPreprocessor()
pii_validator = PIIValidator()
security_validator = SecurityValidator()

# Initialize models
pii_detector = PIIDetectorTrainer()
compliance_rules = {
    'gdpr': ['email', 'phone', 'address', 'name', 'ssn', 'passport'],
    'hipaa': ['medical_record_number', 'health_plan_id', 'account_number'],
    'ccpa': ['ssn', 'credit_card', 'passport']
}
risk_assessor = RiskAssessor(compliance_rules)
compliance_analyzer = ComplianceAnalyzer(compliance_rules)


@router.post("/analyze/file")
async def analyze_file_route(file: UploadFile = File(...)):
    """Analyze file endpoint"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Extract text
            extracted_text = file_loader.extract_text_from_file(tmp_file_path)

            # Detect PII
            detected_pii = text_processor.detect_pii_patterns(extracted_text)

            # Assess risk
            risk_assessment = risk_assessor.assess_document_risk(
                detected_pii=detected_pii,
                file_type=os.path.splitext(file.filename)[1][1:]
            )

            # Generate compliance report
            compliance_report = compliance_analyzer.generate_compliance_report(
                detected_pii=detected_pii,
                risk_score=risk_assessment['final_risk_score'],
                filename=file.filename
            )

            return {
                "status": "success",
                "filename": file.filename,
                "risk_assessment": risk_assessment,
                "compliance_report": compliance_report,
                "pii_detected": detected_pii
            }

        finally:
            os.unlink(tmp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "DataGuard AI API"}