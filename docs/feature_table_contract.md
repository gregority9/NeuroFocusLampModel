# Feature table contract

This file describes the input table expected by the modeling pipeline.

The modeling code does not create EEG features. It expects a ready CSV file
from the feature extraction part of the project.

Default input file:

```text
data/processed/features_bandpower.csv
```

## Required metadata columns

```text
subject_id
session_id
group
task
label
artifact_score
is_rejected
```

Column meanings:

```text
subject_id      participant identifier, for example Subject203
session_id      session identifier, for example ses-001
group           ADHD or control
task            R, TASK1, TASK2, or equivalent task name
label           0 for rest, 1 for cognitive task
artifact_score  numeric artifact score for the window
is_rejected     true/false flag saying whether the window should be ignored
```

## Feature columns

Every column that is not listed as metadata is treated as a model feature.

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

## Minimal valid example

```text
subject_id,session_id,group,task,label,artifact_score,is_rejected,alpha_O1,theta_F3,beta_F3
Subject203,ses-001,ADHD,R,0,0.12,false,0.31,0.20,0.17
Subject203,ses-001,ADHD,TASK1,1,0.18,false,0.25,0.29,0.21
Subject204,ses-001,control,R,0,0.10,false,0.34,0.22,0.15
Subject204,ses-001,control,TASK1,1,0.16,false,0.27,0.31,0.20
```

## Methodological rule

The modeling pipeline uses `subject_id` for Leave-One-Subject-Out validation.
Windows from the same participant must never be split between train and test.
