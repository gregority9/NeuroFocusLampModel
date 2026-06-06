# EEG Focus Classifier

EEG Focus Classifier is a research-oriented Python project for building a
modular offline pipeline that distinguishes resting-state EEG from EEG recorded
during cognitive tasks.

The current methodological target is deliberately conservative:

```text
resting state vs cognitive engagement
```

The project should not be interpreted as a clinical ADHD classifier or as a
direct psychological measurement of "focus". At this stage, the model learns to
separate rest recordings from task recordings under a controlled experimental
setup.

## Project Overview

The project uses EEG recordings from 9 participants:

```text
5 participants with ADHD
4 control participants
```

Each participant has recordings from:

```text
rest
cognitive task 1
cognitive task 2
```

The EEG montage contains 8 channels:

```text
AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

The intended sampling rate is:

```text
250 Hz
```

ADHD/control status is treated as a grouping and reporting variable, not as the
main prediction target.

## Methodological Principles

The most important methodological rule is to avoid leakage between train and
test data.

EEG windows from the same participant are strongly dependent. Therefore, the
project must not use random window-level train/test splits such as:

```python
train_test_split(X_windows, y_windows)
```

The main validation strategy is:

```text
Leave-One-Subject-Out Cross-Validation
```

In each fold:

```text
train: all participants except one
test: the held-out participant
```

This makes the evaluation stricter and better aligned with the real question:
whether the model generalizes to a new person.

## Pipeline

The full planned research pipeline is:

```text
raw EEG data
-> preprocessing
-> windowing
-> feature extraction
-> modeling
-> evaluation
-> reports
```

Each stage should have clear inputs, outputs, and configuration files. The
project is intentionally designed as a modular pipeline rather than a single
one-off notebook.

## Current Modeling Scope

The current modeling code focuses on the first classical baselines:

```text
Bandpower features + Logistic Regression
Bandpower features + Linear SVM
```

The main training script uses scikit-learn pipelines:

```text
scaler -> classifier
```

This ensures that scaling is fitted only on the training data inside each
cross-validation fold.

Planned future modeling work includes:

```text
Covariance matrices -> Tangent Space -> Logistic Regression
subject-relative normalization based on rest baseline
task-transfer validation
artifact-control experiments
```

## Feature Table Contract

The modeling pipeline expects a ready feature table from the feature extraction
stage. It does not create EEG features by itself.

Default input:

```text
data/processed/features_bandpower.csv
```

Required metadata columns:

```text
subject_id
session_id
group
task
label
artifact_score
is_rejected
```

All remaining columns are interpreted as model features.

Example feature columns:

```text
alpha_O1
alpha_O2
theta_F3
theta_F4
beta_F3
beta_F4
theta_beta_F3
theta_beta_F4
```

Labels:

```text
0 = rest
1 = cognitive task
```

The full contract is described in:

```text
docs/feature_table_contract.md
```

## Repository Structure

Current important files and directories:

```text
configs/
  preprocessing.yaml
  model_bandpower_logreg.yaml
  model_bandpower_svm.yaml

data/
  raw/

docs/
  eeg_focus_classifier_project_plan.md
  podzial_zadan_eeg_focus_classifier.md
  feature_table_contract.md

src/
  preprocessing/
  models/
  training/

requirements.txt
README.md
```

Expected future directories:

```text
data/interim/
data/processed/
reports/qc/
reports/figures/
reports/experiments/
tests/
```

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Current core dependencies:

```text
pandas
mne
PyYAML
scikit-learn
```

Additional dependencies may be added later for Riemannian models, plotting, and
automated reports.

## Running Experiments

After the feature table is available at:

```text
data/processed/features_bandpower.csv
```

run the logistic regression baseline:

```powershell
python -m src.training.train configs/model_bandpower_logreg.yaml
```

run the linear SVM baseline:

```powershell
python -m src.training.train configs/model_bandpower_svm.yaml
```

Experiment outputs are saved to:

```text
reports/experiments/{experiment_name}/
```

Expected files:

```text
metrics.json
metrics_per_subject.csv
predictions.csv
```

## Evaluation Metrics

The baseline training pipeline reports:

```text
balanced accuracy
F1-score
precision
recall
ROC-AUC
confusion matrix
per-subject metrics
```

Future reports should also include:

```text
metrics per task
metrics per ADHD/control group
confusion matrix plots
ROC curves
feature importance
experiment notes
```

## Artifact Controls

Because frontal EEG channels can capture eye movements, facial muscle activity,
jaw tension, and speech-related artifacts, model results must be interpreted
carefully.

Planned control experiments include:

```text
training without AF3/AF4
training without the 30-40 Hz band
training only on O1/O2
training only on frontal channels
correlation between artifact_score and model predictions
comparison before and after artifact rejection
```

These checks are necessary to determine whether the model is learning EEG
patterns related to task engagement or mostly artifacts.

## Notes

This project is intended for academic experimentation and method development.
It is not a medical diagnostic system and should not be used for clinical
decision-making.
