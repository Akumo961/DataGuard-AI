import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import joblib
import logging


class ComplianceClassifier(nn.Module):
    """Document compliance risk classifier"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", num_classes: int = 4):
        super(ComplianceClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class ComplianceTrainer:
    """Training pipeline for compliance classifier"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, texts: List[str], labels: List[int]):
        """Prepare training data"""
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the compliance classifier"""

        # Prepare model
        self.model = ComplianceClassifier(self.model_name, num_classes=4)

        # Prepare data
        train_data = self.prepare_data(train_texts, train_labels)
        train_dataset = torch.utils.data.TensorDataset(
            train_data['input_ids'],
            train_data['attention_mask'],
            train_data['labels']
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Training setup
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict compliance risk for text"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()

        risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

        return {
            'predicted_class': predicted_class,
            'risk_level': risk_levels[predicted_class],
            'confidence': probabilities[0][predicted_class].item(),
            'probabilities': probabilities[0].tolist()
        }