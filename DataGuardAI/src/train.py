#!/usr/bin/env python3
"""
Main training script for DataGuard AI models
"""

import argparse
import yaml
import logging
from pathlib import Path

from src.data_loader import DataLoader
from src.models.pii_detector import PIIDetectorTrainer
from src.models.compliance_classifier import ComplianceTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pii_detector(config: dict):
    """Train PII detection model"""
    logger.info("Training PII Detection Model...")

    # Initialize trainer
    trainer = PIIDetectorTrainer(
        model_name=config['models']['pii_detector']['model_name']
    )

    # Load training data
    data_loader = DataLoader()
    training_data = data_loader.load_training_data("data/raw/pii_training_data.json")

    # Prepare data (simplified - you'd need to adapt to your data format)
    texts = [item['text'] for item in training_data]
    labels = [item['labels'] for item in training_data]

    # Train model
    trainer.train(
        train_texts=texts,
        train_labels=labels,
        output_dir="models/pii_ner_model",
        epochs=config['models']['pii_detector'].get('epochs', 3),
        batch_size=config['models']['pii_detector'].get('batch_size', 16)
    )

    logger.info("PII Detection Model training completed!")


def train_compliance_classifier(config: dict):
    """Train compliance classification model"""
    logger.info("Training Compliance Classifier...")

    trainer = ComplianceTrainer(
        model_name=config['models']['compliance_classifier']['model_name']
    )

    # This would be replaced with actual compliance training data
    # For demonstration, using dummy data
    sample_texts = [
        "This document contains sensitive customer information including emails and phone numbers.",
        "Public announcement with no personal data.",
        "Medical records with patient identifiers."
    ]
    sample_labels = [2, 0, 3]  # HIGH, LOW, CRITICAL

    trainer.train(sample_texts, sample_labels, epochs=3)

    logger.info("Compliance Classifier training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train DataGuard AI models")
    parser.add_argument('--model', type=str, choices=['pii', 'compliance', 'all'],
                        default='all', help='Model to train')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create models directory
    Path("models").mkdir(exist_ok=True)

    try:
        if args.model in ['pii', 'all']:
            train_pii_detector(config)

        if args.model in ['compliance', 'all']:
            train_compliance_classifier(config)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()