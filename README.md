# Online-Sexism-Detection

## Overview

This project aims to detect and classify sexist content in online text using deep learning models enhanced with Single-Valued Neutrosophic Sets (SVNS). The system performs classification at different levels:

Subtask A: Binary classification (Sexist vs. Non-Sexist)

Subtask B: Fine-grained categorization of sexist content

Subtask C: Severity assessment of sexist language

The implementation leverages RoBERTa, BiLSTM, and ensemble techniques for robust and context-aware classification.

## Subtasks

### Subtask A: Binary Classification

Objective: Identify whether a given text is Sexist or Non-Sexist.

Model: RoBERTa-based classification with SVNS-driven post-processing.

Performance: Uses Decision Tree classifiers to refine classification decisions.

### Subtask B: Fine-Grained Categorization

Objective: Classify sexist text into predefined categories such as derogation, threats, objectification, and stereotyping.

Model: BiLSTM for feature extraction, followed by a Random Forest classifier.

Performance: Uses an attention mechanism to enhance contextual understanding.

### Subtask C: Severity Classification

Objective: Assess the severity of sexist content based on intensity and impact.

Model: Hierarchical BiLSTM with word-level and sentence-level attention.

Performance: Uses ordinal classification with dropout regularization to improve reliability.

## Installation & Setup
### Prerequisites

Python 3.x

PyTorch

Transformers (Hugging Face)

Scikit-learn

Pandas, NumPy
