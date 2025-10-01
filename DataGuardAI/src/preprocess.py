preprocess_py = """
import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer
import spacy
import logging

class TextPreprocessor:
    \"\"\"Preprocess text data for PII detection\"\"\"

    def __init__(self, model_name: str = "bert-base-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        self.logger = logging.getLogger(__name__)

        # PII patterns
        self.pii_patterns = {
            'email': r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b',
            'phone': r'\\b(?:\\+?1[-.]?)?\\(?([0-9]{3})\\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\\b',
            'ssn': r'\\b(?!000|666|9\\d{2})\\d{3}[-.]?(?!00)\\d{2}[-.]?(?!0000)\\d{4}\\b',
            'credit_card': r'\\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\\b',
            'ip_address': r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b',
            'address': r'\\d+\\s+[A-Za-z\\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Court|Ct|Lane|Ln)',
        }

    def clean_text(self, text: str) -> str:
        \"\"\"Basic text cleaning\"\"\"
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\\w\\s.,!?;:-]', ' ', text)

        return text.strip()

    def detect_pii_patterns(self, text: str) -> Dict[str, List[str]]:
        \"\"\"Detect PII using regex patterns\"\"\"
        detected_pii = {}

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches

        return detected_pii

    def tokenize_for_ner(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        \"\"\"Tokenize text for NER model\"\"\"
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        return tokens

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        \"\"\"Extract named entities using spaCy\"\"\"
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })

        return entities

    def preprocess_for_training(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        \"\"\"Preprocess text data for model training\"\"\"
        processed_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }

        for text, label in zip(texts, labels):
            # Clean text
            cleaned_text = self.clean_text(text)

            # Tokenize
            tokens = self.tokenize_for_ner(cleaned_text)

            processed_data['input_ids'].append(tokens['input_ids'].squeeze())
            processed_data['attention_mask'].append(tokens['attention_mask'].squeeze())
            processed_data['labels'].append(label)

        return processed_data

class ComplianceAnalyzer:
    \"\"\"Analyze compliance risk based on detected PII\"\"\"

    def __init__(self, compliance_rules: Dict[str, List[str]]):
        self.compliance_rules = compliance_rules
        self.risk_weights = {
            'email': 0.3,
            'phone': 0.4,
            'ssn': 1.0,
            'credit_card': 0.9,
            'address': 0.5,
            'name': 0.6,
            'medical_record_number': 0.8,
            'passport': 0.7
        }

    def calculate_risk_score(self, detected_pii: Dict[str, List[str]]) -> float:
        \"\"\"Calculate overall risk score\"\"\"
        total_risk = 0.0
        max_risk = 0.0

        for pii_type, matches in detected_pii.items():
            if pii_type in self.risk_weights:
                risk_value = self.risk_weights[pii_type] * len(matches)
                total_risk += risk_value
                max_risk = max(max_risk, self.risk_weights[pii_type])

        # Normalize risk score between 0 and 1
        normalized_risk = min(total_risk / 10.0, 1.0) if total_risk > 0 else 0.0

        return normalized_risk

    def classify_risk_level(self, risk_score: float) -> str:
        \"\"\"Classify risk into categories\"\"\"
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_compliance_report(self, detected_pii: Dict[str, List[str]], 
                                 risk_score: float, filename: str) -> Dict[str, Any]:
        \"\"\"Generate detailed compliance report\"\"\"
        risk_level = self.classify_risk_level(risk_score)

        report = {
            'filename': filename,
            'timestamp': pd.Timestamp.now().isoformat(),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'detected_pii': detected_pii,
            'pii_count': sum(len(matches) for matches in detected_pii.values()),
            'recommendations': self._generate_recommendations(detected_pii, risk_level)
        }

        return report

    def _generate_recommendations(self, detected_pii: Dict[str, List[str]], 
                                risk_level: str) -> List[str]:
        \"\"\"Generate remediation recommendations\"\"\"
        recommendations = []

        if 'ssn' in detected_pii:
            recommendations.append("⚠️ SSN detected - Implement data encryption and access controls")

        if 'credit_card' in detected_pii:
            recommendations.append("⚠️ Credit card numbers found - Ensure PCI DSS compliance")

        if 'email' in detected_pii:
            recommendations.append("📧 Email addresses detected - Review data sharing policies")

        if risk_level in ['HIGH', 'CRITICAL']:
            recommendations.append("🚨 High risk detected - Conduct immediate security review")
            recommendations.append("🔒 Consider data anonymization or pseudonymization")

        if not recommendations:
            recommendations.append("✅ Low risk detected - Continue monitoring")

        return recommendations
"""
