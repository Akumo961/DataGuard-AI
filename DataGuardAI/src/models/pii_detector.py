pii_detector_py = """
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

class PIIDataset(Dataset):
    \"\"\"Custom dataset for PII detection training\"\"\"

    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Label to ID mapping
        self.label2id = {
            'O': 0,           # Outside
            'B-PERSON': 1,    # Beginning of person name
            'I-PERSON': 2,    # Inside person name
            'B-EMAIL': 3,     # Beginning of email
            'I-EMAIL': 4,     # Inside email
            'B-PHONE': 5,     # Beginning of phone
            'I-PHONE': 6,     # Inside phone
            'B-SSN': 7,       # Beginning of SSN
            'I-SSN': 8,       # Inside SSN
            'B-ADDRESS': 9,   # Beginning of address
            'I-ADDRESS': 10   # Inside address
        }

        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            is_split_into_words=True
        )

        # Align labels with tokens
        aligned_labels = self._align_labels_with_tokens(labels, encoding)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

    def _align_labels_with_tokens(self, labels: List[str], encoding) -> List[int]:
        \"\"\"Align BIO labels with tokenized text\"\"\"
        aligned_labels = []

        for i in range(len(encoding['input_ids'][0])):
            if i == 0 or i == len(encoding['input_ids'][0]) - 1:  # [CLS] and [SEP]
                aligned_labels.append(0)  # 'O'
            else:
                if i - 1 < len(labels):
                    label = labels[i - 1] if labels[i - 1] in self.label2id else 'O'
                    aligned_labels.append(self.label2id[label])
                else:
                    aligned_labels.append(0)  # 'O'

        return aligned_labels

class PIIDetector(nn.Module):
    \"\"\"BERT-based PII detection model\"\"\"

    def __init__(self, model_name: str = "bert-base-cased", num_labels: int = 11):
        super(PIIDetector, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        \"\"\"Initialize classifier weights\"\"\"
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten for loss calculation
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {'loss': loss, 'logits': logits}

class PIIDetectorTrainer:
    \"\"\"Training pipeline for PII detection model\"\"\"

    def __init__(self, model_name: str = "bert-base-cased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.logger = logging.getLogger(__name__)

    def prepare_model(self, num_labels: int = 11):
        \"\"\"Initialize the model\"\"\"
        self.model = PIIDetector(self.model_name, num_labels)

    def train(self, train_texts: List[str], train_labels: List[List[str]], 
              val_texts: List[str] = None, val_labels: List[List[str]] = None,
              output_dir: str = "models/pii_ner_model",
              epochs: int = 3, batch_size: int = 16):
        \"\"\"Train the PII detection model\"\"\"

        # Prepare datasets
        train_dataset = PIIDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = PIIDataset(val_texts, val_labels, self.tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train
        self.logger.info("Starting PII detection model training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        self.logger.info(f"Model saved to {output_dir}")

    def predict(self, text: str, model_path: str = "models/pii_ner_model") -> List[Dict[str, Any]]:
        \"\"\"Predict PII entities in text\"\"\"
        # Load model if not loaded
        if self.model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**{k: v for k, v in inputs.items() if k != 'offset_mapping'})
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class = predictions.argmax(dim=-1)

        # Convert predictions to entities
        entities = self._convert_predictions_to_entities(
            text, predicted_token_class[0], inputs['offset_mapping'][0]
        )

        return entities

    def _convert_predictions_to_entities(self, text: str, predictions: torch.Tensor, 
                                       offsets: torch.Tensor) -> List[Dict[str, Any]]:
        \"\"\"Convert model predictions to entity format\"\"\"
        id2label = {
            0: 'O', 1: 'B-PERSON', 2: 'I-PERSON', 3: 'B-EMAIL', 4: 'I-EMAIL',
            5: 'B-PHONE', 6: 'I-PHONE', 7: 'B-SSN', 8: 'I-SSN',
            9: 'B-ADDRESS', 10: 'I-ADDRESS'
        }

        entities = []
        current_entity = None

        for i, (pred, offset) in enumerate(zip(predictions, offsets)):
            if offset[0] == 0 and offset[1] == 0:  # Skip special tokens
                continue

            label = id2label[pred.item()]

            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                start_pos = offset[0].item()
                end_pos = offset[1].item()

                current_entity = {
                    'text': text[start_pos:end_pos],
                    'label': entity_type,
                    'start': start_pos,
                    'end': end_pos,
                    'confidence': torch.max(torch.nn.functional.softmax(predictions[i])).item()
                }

            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = label[2:]
                if current_entity['label'] == entity_type:
                    current_entity['end'] = offset[1].item()
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]

            elif label == 'O' and current_entity:
                # End current entity
                entities.append(current_entity)
                current_entity = None

        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities
"""