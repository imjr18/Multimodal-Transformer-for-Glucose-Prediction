# OhioT1DM Data Instructions

Part A uses the OhioT1DM 2018 cohort and only reads the glucose and heart-rate
signals required by the prompt.

## 1. Request access

Request the dataset from the official OhioT1DM page:
http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html

## 2. Place the XML files

The preprocessing code supports the original Ohio folder layout and a flat
directory layout.

Preferred layout:

```text
glucose_transformer/data/raw/
└── OhioT1DM/
    └── 2018/
        ├── train/
        │   ├── 559-ws-training.xml
        │   ├── 563-ws-training.xml
        │   ├── 570-ws-training.xml
        │   ├── 575-ws-training.xml
        │   ├── 588-ws-training.xml
        │   └── 591-ws-training.xml
        └── test/
            ├── 559-ws-testing.xml
            ├── 563-ws-testing.xml
            ├── 570-ws-testing.xml
            ├── 575-ws-testing.xml
            ├── 588-ws-testing.xml
            └── 591-ws-testing.xml
```

Also supported:

```text
glucose_transformer/data/raw/
├── 559-ws-training.xml
├── 559-ws-testing.xml
...
```

## 3. Cohort split used in Part A

- Training patients: `559`, `563`, `570`
- Validation patient: `588`
- Test patients: `575`, `591`

The preprocessor merges each patient's available training and testing XML files
into one continuous patient-level dataframe, then performs a patient-level split
for model development. This preserves the requested held-out patient evaluation.

## 4. Signals used

Only these XML sections are used in Part A:

- `glucose_level`
- `basis_heart_rate` or `heart_rate`

All other OhioT1DM fields are ignored until later parts.
