data_loader_py = """
import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

class DataLoader:
    \"\"\"Handle loading and preprocessing of various data formats\"\"\"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.supported_formats = self.config['supported_formats']
        self.logger = logging.getLogger(__name__)

    def load_training_data(self, data_path: str) -> Dict[str, Any]:
        \"\"\"Load PII training datasets\"\"\"
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
            elif data_path.endswith('.csv'):
                data = pd.read_csv(data_path).to_dict('records')
            else:
                raise ValueError(f"Unsupported training data format: {data_path}")

            self.logger.info(f"Loaded {len(data)} training samples")
            return data

        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            raise

    def load_compliance_rules(self, rules_path: str = "data/raw/compliance_rules.yaml") -> Dict[str, List[str]]:
        \"\"\"Load compliance rules and PII patterns\"\"\"
        try:
            with open(rules_path, 'r') as f:
                rules = yaml.safe_load(f)
            return rules
        except Exception as e:
            self.logger.error(f"Error loading compliance rules: {str(e)}")
            return self.config['compliance_rules']

    def save_processed_data(self, data: Any, output_path: str) -> None:
        \"\"\"Save processed data to disk\"\"\"
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if output_path.endswith('.pkl'):
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
            elif output_path.endswith('.npy'):
                np.save(output_path, data)
            elif output_path.endswith('.json'):
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

            self.logger.info(f"Saved processed data to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

class FileDataLoader:
    \"\"\"Load and extract text from various file formats\"\"\"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_from_file(self, file_path: str) -> str:
        \"\"\"Extract text from supported file formats\"\"\"
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == '.txt':
                return self._extract_from_txt(file_path)
            elif file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext == '.docx':
                return self._extract_from_docx(file_path)
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                return self._extract_from_image(file_path)
            elif file_ext in ['.wav', '.mp3', '.m4a']:
                return self._extract_from_audio(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise

    def _extract_from_txt(self, file_path: str) -> str:
        \"\"\"Extract text from .txt files\"\"\"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_from_pdf(self, file_path: str) -> str:
        \"\"\"Extract text from PDF files\"\"\"
        import PyPDF2

        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\\n"
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        \"\"\"Extract text from Word documents\"\"\"
        import docx

        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\\n"
        return text

    def _extract_from_image(self, file_path: str) -> str:
        \"\"\"Extract text from images using OCR\"\"\"
        import pytesseract
        from PIL import Image

        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text

    def _extract_from_audio(self, file_path: str) -> str:
        \"\"\"Extract text from audio files using Whisper\"\"\"
        import whisper

        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
"""
