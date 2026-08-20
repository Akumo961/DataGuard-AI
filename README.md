# DataGuard AI — PII Detection & Compliance Automation


> AI-powered document analysis platform for detecting Personally Identifiable Information (PII) and assessing compliance risks using fine-tuned BERT models.


**Python** · **BERT** · **NLP** · **FastAPI** · **Gradio** · **PII Detection** · **Compliance**


## Overview


DataGuard AI automates the detection of sensitive personal information in unstructured documents and provides compliance-oriented analysis.


The system combines NLP model inference with a FastAPI backend and Gradio interface for interactive document analysis.


## Problem


Organizations process large volumes of documents containing sensitive personal information.


Manual identification of PII can be:

- Time-consuming

- Difficult to scale

- Inconsistent

- Expensive


DataGuard AI explores an automated approach using transformer-based NLP.


## Architecture


```text

Document

   │

   ▼

Document Ingestion

   │

   ▼

Text Extraction

   │

   ▼

Preprocessing

   │

   ▼

Fine-tuned BERT

   │

   ▼

PII Detection

   │

   ▼

Compliance Analysis

   │

   ▼

FastAPI / Gradio

   │

   ▼

User Results

```


## Core Capabilities


- PII detection

- Document analysis

- Multi-format document support

- Transformer-based NLP

- Real-time inference

- FastAPI backend

- Gradio interface

- Compliance-oriented analysis


## Machine Learning


The project uses a fine-tuned BERT-based model for PII detection.


The ML pipeline includes:


```text

Training Data

     ↓

Preprocessing

     ↓

Tokenization

     ↓

BERT Fine-tuning

     ↓

Evaluation

     ↓

Inference

```


## Results


The project achieved **95%+ PII detection accuracy** based on the project's evaluation.


The workflow was designed to reduce manual document review effort, with an estimated **90% reduction in manual review**.


> Evaluation metrics should be interpreted according to the dataset and evaluation methodology used by the project.


## API


The inference service is exposed through FastAPI, enabling integration into other applications and document-processing workflows.


## User Interface


A Gradio interface provides an interactive way to submit documents and inspect detection results.


## Technology Stack


- Python

- PyTorch / Transformers

- BERT

- NLP

- FastAPI

- Gradio


## Engineering Focus


**Machine Learning + NLP + Model Inference + Document Processing + API Engineering**


## Limitations


PII detection systems should be evaluated against representative datasets and validated for the specific regulatory and organizational context in which they are deployed.


The system should not be treated as a substitute for legal or compliance advice.


## Disclaimer


This project is for research and technical demonstration purposes.

