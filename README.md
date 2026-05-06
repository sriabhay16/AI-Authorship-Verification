# Repository Guide

## Dataset
The experiments are conducted on the PAN25 Generative AI Detection: Voight-Kampff AI Detection Sensitivity shared task dataset which contains:
Human-written text, AI-generated text and Multiple genres and LLM sources
- Shared Task: https://pan.webis.de/clef25/pan25-web/generated-content-analysis.html#task1
- Dataset Link: https://zenodo.org/records/14962653 

---
## Workflow Overview

```text
EDA → Dataset Balancing → Stylometric Extraction → Feature Selection →
Scenario Construction → ML/DL Training → Evaluation
```
---

## Repository Structure

### 1. [complete_eda.ipynb](./complete_eda.ipynb)
Exploratory Data Analysis of the PAN25 dataset, including:
- Genre-wise distribution
- Human vs AI sample analysis
- Model distribution
- Post-balancing dataset analysis
- Visualization of dataset splits and statistics

---

### 2. [dataset_balancing.ipynb](./dataset_balancing.ipynb)
Code for balancing the PAN25 training and validation datasets:
- Equal genre balancing
- Human vs AI class balancing
- Proportional AI-model sampling
- Generation of balanced JSONL datasets

---

### 3. [authorship_extractor.py](./authorship_extractor.py)
Core stylometric feature extraction module containing:
- Stylometric metric implementations
- Feature extraction classes and helper functions
- Definitions and computation of handcrafted linguistic features

---

### 4. [stylometry_tables.ipynb](./stylometry_tables.ipynb)
Generates genre-wise stylometric feature tables using the authorship extractor:
- Feature aggregation
- Genre-specific metric analysis
- Comparative statistical tables

---

### 5. [feature_set_extract.ipynb](./feature_set_extract.ipynb)
Feature selection and scenario generation pipeline:
- Identification of top 15 most discriminative stylometric features per genre
- Construction of feature-set scenarios using set operations:
  - **S3: Union** - Union of all selected discriminative features across genres.
  - **S4: Pairwise Intersection** - Features shared between exactly two genres using Pairwise + Intersection sets.
  - **S5: Intersection** - Features common across all three genres (Intersection set only).
---

### 6. [ML.ipynb](./ML.ipynb)
Machine learning training, validation, and testing pipeline for all feature scenarios.

#### Implemented Models
- Logistic Regression
- Ridge Classifier
- SGD Classifier
- SVM (Linear + RBF)
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- Extra Trees
- XGBoost
- LightGBM
- 1D-CNN
- GRU
- LSTM
- BERT Fine-tuning Baseline (S1: Text-Only)

Includes:
- Cross-scenario evaluation
- Performance comparison
- Validation and test evaluation with visualizations
