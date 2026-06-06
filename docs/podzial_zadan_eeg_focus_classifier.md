# Podział zadań w projekcie EEG Focus Classifier

## Cel podziału

Celem podziału jest takie rozdzielenie pracy między trzy osoby, aby:

- każdy miał porównywalną ilość zadań,
- każdy odpowiadał za wyraźnie oddzielony moduł projektu,
- projekt dało się łatwo rozwijać w przyszłości,
- nie powstał jednorazowy notebook, tylko modularny pipeline badawczy,
- kolejne etapy mogły być wymieniane lub rozszerzane bez przepisywania całego kodu.

Projekt powinien być budowany jako pipeline:

```text
raw data
-> preprocessing
-> windowing
-> feature extraction
-> modeling
-> evaluation
-> reports
```

Każdy etap powinien mieć:

- osobne moduły w `src/`,
- osobną konfigurację YAML,
- jasno określone wejścia i wyjścia,
- możliwość łatwego rozszerzania.

---

## Podział odpowiedzialności

| Osoba | Główny obszar | Odpowiedzialność |
|---|---|---|
| **Grzechu** | Preprocessing + QC | Przygotowanie oczyszczonych sygnałów EEG oraz raportów jakości |
| **Szymon** | Okienkowanie + cechy | Zamiana sygnałów EEG na okna i tabele cech gotowe do modeli |
| **Wiciu** | Modele + walidacja + raporty | Trenowanie modeli, walidacja LOSO, metryki i raportowanie eksperymentów |

Szacunkowy udział pracy:

```text
Grzechu: 30–35%
Szymon: 30–35%
Wiciu: 30–35%
```

---

# 1. Grzechu — preprocessing i kontrola jakości

## Główna odpowiedzialność

Grzechu odpowiada za przygotowanie danych EEG do dalszego przetwarzania.

Jego część powinna być zrobiona tak, aby w przyszłości można było łatwo dodać:

- nowe filtry,
- nowe metody referencji,
- inne kanały EEG,
- nowe metody wykrywania artefaktów,
- bardziej szczegółowe raporty jakości.

## Moduły

```text
src/eeg_focus/io/
  loaders.py
  validators.py

src/eeg_focus/preprocessing/
  filters.py
  referencing.py
  artifact_detection.py
  quality_report.py

src/eeg_focus/pipelines/
  preprocess.py
```

## Zadania

### 1.1. Wczytywanie danych

Przygotowanie loadera plików CSV zawierających dane EEG.

Oczekiwane kanały:

```text
AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

Przykładowy format wejściowy:

```text
timestamp, AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

### 1.2. Walidacja danych

Dla każdego nagrania należy sprawdzić:

- liczbę próbek,
- częstotliwość próbkowania,
- brakujące próbki,
- duplikaty timestampów,
- wartości `NaN`,
- wartości `Inf`,
- martwe kanały,
- kanały o bardzo niskiej wariancji,
- kanały o bardzo wysokiej wariancji,
- clipping / saturację,
- gwałtowne skoki amplitudy.

### 1.3. Filtrowanie sygnału

Podstawowa konfiguracja:

```yaml
preprocessing:
  filter:
    highpass_hz: 1.0
    lowpass_hz: 40.0
    notch_hz: 50.0
```

Filtry nie powinny być wpisane na sztywno w kodzie. Powinny być pobierane z pliku konfiguracyjnego.

### 1.4. Referencja

Należy przygotować możliwość testowania kilku wariantów:

```text
none
common_average
average_good_channels
```

Przykład konfiguracji:

```yaml
preprocessing:
  reference:
    method: common_average
```

### 1.5. Artefakty

Na start należy przygotować podstawowy artifact scoring.

Przykładowe kryteria:

- peak-to-peak amplitude,
- RMS,
- kurtosis,
- gwałtowne skoki między próbkami,
- nietypowo wysoka moc 30–40 Hz,
- bardzo niska wariancja kanału,
- bardzo wysoka wariancja kanału.

Progi powinny być konfigurowalne:

```yaml
preprocessing:
  artifact_rejection:
    enabled: true
    rms_z_threshold: 4.0
    kurtosis_z_threshold: 4.0
    high_beta_z_threshold: 4.0
```

## Efekty pracy Grzecha

```text
data/interim/sub-001/rest_clean.csv
data/interim/sub-001/focus_memory_clean.csv
data/interim/sub-001/focus_words_clean.csv

reports/qc/sub-001_rest_qc.json
reports/qc/sub-001_rest_qc.html
```

---

# 2. Szymon — okienkowanie, cechy i sanity check

## Główna odpowiedzialność

Szymon odpowiada za zamianę oczyszczonego sygnału EEG na dane gotowe do trenowania modeli.

Jego część powinna być zrobiona tak, aby w przyszłości można było łatwo dodać:

- inne długości okien,
- inny overlap,
- nowe pasma EEG,
- nowe cechy,
- nowe agregaty kanałów,
- cechy kowariancyjne,
- cechy czasowe,
- cechy pod tryb online.

## Moduły

```text
src/eeg_focus/epoching/
  windows.py

src/eeg_focus/features/
  bandpower.py
  ratios.py
  asymmetry.py
  covariance.py
  feature_table.py

src/eeg_focus/visualization/
  psd.py
  diagnostics.py

src/eeg_focus/pipelines/
  extract_features.py
```

## Zadania

### 2.1. Okienkowanie

Podstawowa konfiguracja:

```yaml
epoching:
  window_sec: 4.0
  overlap: 0.5
```

Przy częstotliwości próbkowania 250 Hz:

```text
4 sekundy = 1000 próbek
overlap 50% = krok 2 sekundy
```

### 2.2. Etykiety

Podstawowe etykiety:

```text
rest -> 0
focus_memory -> 1
focus_words -> 1
```

Projekt powinien być opisywany jako klasyfikacja:

```text
relaks vs zadanie poznawcze
```

lub:

```text
resting state vs cognitive engagement
```

Nie należy na tym etapie twierdzić, że model mierzy „skupienie” w pełnym psychologicznym sensie.

### 2.3. Metadata okien

Dla każdego okna należy zapisać:

```text
subject_id
session_id
group
task
label
window_start_sec
window_end_sec
artifact_score
is_rejected
```

### 2.4. Cechy pasmowe

Podstawowe pasma:

```yaml
features:
  bands:
    delta: [1, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    high_beta: [30, 40]
```

Dla każdego kanału i okna należy policzyć:

- absolute bandpower,
- relative bandpower,
- log bandpower.

### 2.5. Cechy pochodne

Do przygotowania:

- theta/beta ratio,
- alpha/beta ratio,
- frontal theta,
- occipital alpha,
- frontal-to-occipital ratio,
- asymetria F3/F4,
- asymetria O1/O2.

Przykłady:

```text
theta_beta_F3 = theta_power_F3 / beta_power_F3
theta_beta_F4 = theta_power_F4 / beta_power_F4

alpha_occipital = mean(alpha_O1, alpha_O2)

frontal_theta = mean(theta_AF3, theta_AF4, theta_F3, theta_F4)

asymmetry_F3_F4_alpha = log(alpha_F3) - log(alpha_F4)
asymmetry_O1_O2_alpha = log(alpha_O1) - log(alpha_O2)
```

### 2.6. Cechy kowariancyjne

Dla każdego okna:

```text
X_window: channels x samples
```

czyli:

```text
8 x 1000
```

Należy przygotować macierz kowariancji:

```text
C = covariance(X_window)
```

Macierz ma wymiar:

```text
8 x 8
```

Będzie ona używana później przez pipeline riemannowski.

### 2.7. Sanity check

Szymon przygotowuje podstawowe wykresy i porównania:

- PSD relaks vs zadanie poznawcze,
- alpha O1/O2 relaks vs zadanie,
- theta F3/F4 relaks vs zadanie,
- memory vs words,
- ADHD vs control eksploracyjnie.

Celem jest sprawdzenie, czy dane zawierają sensowny sygnał i czy nie są zdominowane przez artefakty.

## Konfiguracja cech

Cechy powinny być sterowane z YAML-a:

```yaml
features:
  type: bandpower
  bands:
    delta: [1, 4]
    theta: [4, 8]
    alpha: [8, 13]
    beta: [13, 30]
    high_beta: [30, 40]
  use_relative_power: true
  use_log_power: true
  use_ratios: true
  use_asymmetry: true
  use_covariance: true
```

## Efekty pracy Szymona

```text
data/processed/windows_metadata.csv
data/processed/features_bandpower.csv
data/processed/covariance_matrices.npy

reports/figures/psd_rest_vs_focus.png
reports/figures/bandpower_summary.png
```

---

# 3. Wiciu — modele, walidacja i raporty

## Główna odpowiedzialność

Wiciu odpowiada za trenowanie modeli, poprawną walidację i raportowanie wyników.

Jego część powinna być zrobiona tak, aby w przyszłości można było łatwo dodać:

- nowe modele,
- nowe metryki,
- inne strategie walidacji,
- nowe eksperymenty,
- raporty porównawcze,
- pipeline predykcji dla nowej osoby.

## Moduły

```text
src/eeg_focus/models/
  classical.py
  riemannian.py
  calibration.py
  registry.py

src/eeg_focus/evaluation/
  loso.py
  task_transfer.py
  metrics.py
  reports.py

src/eeg_focus/pipelines/
  train.py
  evaluate.py
  predict.py
```

## Zadania

### 3.1. Modele baseline

Pierwszy model:

```text
Bandpower + Logistic Regression
```

Drugi model:

```text
Bandpower + Linear SVM
```

### 3.2. Model riemannowski

Model główny do porównania:

```text
Covariance
-> Tangent Space
-> Logistic Regression
```

### 3.3. Walidacja

Obowiązkowa strategia:

```text
Leave-One-Subject-Out Cross Validation
```

Czyli:

```text
trenujemy na 8 osobach
testujemy na 1 osobie
powtarzamy dla każdej osoby
```

Nie wolno robić losowego podziału okien:

```python
train_test_split(X_windows, y_windows)
```

ponieważ prowadzi to do leakage między oknami tej samej osoby.

### 3.4. Normalizacja

Scaler musi być uczony tylko na danych treningowych.

Poprawnie:

```text
fit scaler only on train
transform train
transform test
```

Niepoprawnie:

```text
fit scaler on all data before cross-validation
```

### 3.5. Metryki

Należy raportować:

- balanced accuracy,
- F1-score,
- precision,
- recall,
- ROC-AUC,
- confusion matrix,
- wyniki per subject,
- wyniki per task,
- wyniki per grupa ADHD/control.

### 3.6. Eksperymenty dodatkowe

Wiciu odpowiada też za eksperymenty porównawcze:

```text
Eksperyment 1:
Bandpower + Logistic Regression

Eksperyment 2:
Bandpower + Linear SVM

Eksperyment 3:
Riemannian Tangent Space + Logistic Regression

Eksperyment 4:
z normalizacją względem relaksu osoby vs bez niej

Eksperyment 5:
train: rest vs memory
test: rest vs words

Eksperyment 6:
train: rest vs words
test: rest vs memory

Eksperyment 7:
kontrola artefaktów
```

### 3.7. Kontrola artefaktów

Testy kontrolne:

- wyniki bez kanałów AF3/AF4,
- wyniki bez pasma 30–40 Hz,
- wyniki tylko dla O1/O2,
- wyniki tylko dla kanałów frontalnych,
- korelacja `artifact_score` z predykcją,
- porównanie wyników przed i po odrzuceniu artefaktów.

Celem jest sprawdzenie, czy model nie klasyfikuje głównie artefaktów.

## Konfiguracja modeli

Modele powinny być wybierane z YAML-a:

```yaml
model:
  type: logistic_regression
  class_weight: balanced
  C: 1.0

validation:
  strategy: leave_one_subject_out
  group_column: subject_id

normalization:
  scaler: robust
  subject_relative_to_rest: true
```

## Efekty pracy Wicia

```text
reports/experiments/bandpower_logreg/
  metrics.json
  metrics_per_subject.csv
  confusion_matrix.png
  roc_curve.png
  feature_importance.csv
  config.yaml
  notes.md

reports/experiments/bandpower_svm/
  metrics.json
  metrics_per_subject.csv
  confusion_matrix.png
  roc_curve.png
  config.yaml
  notes.md

reports/experiments/riemannian_logreg/
  metrics.json
  metrics_per_subject.csv
  confusion_matrix.png
  config.yaml
  notes.md
```

---

# Wymagania architektoniczne dla wszystkich

## 1. Zero hardcodowania

Nie wpisujemy na sztywno w kodzie:

```python
window_sec = 4
channels = ["AF3", "AF4", "F3", "F4"]
model = LogisticRegression()
```

Zamiast tego parametry powinny pochodzić z konfiguracji:

```yaml
data:
  sampling_rate: 250
  channels:
    - AF3
    - AF4
    - F3
    - F4
    - FC5
    - FC6
    - O1
    - O2
```

## 2. Jasne wejścia i wyjścia

Każdy etap powinien produkować plik używany przez kolejny etap.

```text
Grzechu:
raw CSV
-> clean/interim CSV
-> QC reports

Szymon:
clean/interim CSV
-> windows metadata
-> feature table
-> PSD plots

Wiciu:
feature table
-> models
-> metrics
-> plots
-> experiment reports
```

## 3. Jeden pipeline, wiele konfiguracji

Nie tworzymy wielu osobnych skryptów typu:

```text
train_logreg.py
train_svm.py
train_riemannian.py
```

Lepiej mieć jeden pipeline:

```text
train.py --config configs/model_bandpower_logreg.yaml
train.py --config configs/model_bandpower_svm.yaml
train.py --config configs/model_riemannian_logreg.yaml
```

## 4. Registry dla modeli i cech

Warto przygotować prosty system rejestrów.

Przykład dla modeli:

```python
MODEL_REGISTRY = {
    "logistic_regression": build_logistic_regression,
    "linear_svm": build_linear_svm,
    "riemannian_logreg": build_riemannian_logreg,
}
```

Przykład dla cech:

```python
FEATURE_REGISTRY = {
    "bandpower": compute_bandpower,
    "ratios": compute_ratios,
    "asymmetry": compute_asymmetry,
    "covariance": compute_covariance,
}
```

Dzięki temu dodanie nowego modelu lub cechy wymaga dopisania funkcji i wpisu w registry, a nie przepisywania całego pipeline’u.

## 5. Testy

Każda osoba powinna przygotować minimalne testy swojego modułu.

```text
Grzechu:
tests/test_loaders.py
tests/test_filters.py
tests/test_artifact_detection.py

Szymon:
tests/test_epoching.py
tests/test_features.py
tests/test_covariance.py

Wiciu:
tests/test_loso_split.py
tests/test_metrics.py
tests/test_model_registry.py
```

Najważniejszy test całego projektu:

```text
Czy Leave-One-Subject-Out naprawdę nie miesza okien tej samej osoby między train i test.
```

---

# Proponowana struktura repozytorium

```text
eeg-focus-classifier/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│
├── configs/
│   ├── preprocessing.yaml
│   ├── features_bandpower.yaml
│   ├── features_covariance.yaml
│   ├── model_bandpower_logreg.yaml
│   ├── model_bandpower_svm.yaml
│   └── model_riemannian_logreg.yaml
│
├── src/
│   └── eeg_focus/
│       ├── io/
│       ├── preprocessing/
│       ├── epoching/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── visualization/
│       └── pipelines/
│
├── notebooks/
│   ├── 01_signal_quality.ipynb
│   ├── 02_psd_exploration.ipynb
│   ├── 03_feature_exploration.ipynb
│   └── 04_model_comparison.ipynb
│
├── reports/
│   ├── qc/
│   ├── figures/
│   └── experiments/
│
├── tests/
│   ├── test_loaders.py
│   ├── test_filters.py
│   ├── test_epoching.py
│   ├── test_features.py
│   ├── test_loso_split.py
│   └── test_metrics.py
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

# Kolejność pracy

## Etap 1 — MVP

```text
1. Grzechu:
   loader, walidacja, filtrowanie, QC

2. Szymon:
   okienkowanie, etykiety, bandpower, tabela cech

3. Wiciu:
   Logistic Regression, LOSO, podstawowe metryki
```

## Etap 2 — rozszerzenie

```text
1. Grzechu:
   artifact scoring i lepsze raporty jakości

2. Szymon:
   ratio features, asymetrie, covariance features, PSD sanity check

3. Wiciu:
   Linear SVM, Riemannian pipeline, raporty eksperymentów
```

## Etap 3 — obrona wyników

```text
1. Grzechu:
   analiza problematycznych kanałów i jakości danych

2. Szymon:
   porównanie cech relaks vs zadanie poznawcze

3. Wiciu:
   eksperymenty kontrolne, porównanie modeli, finalne raporty
```

---

# Najkrótsze podsumowanie

```text
Grzechu:
Modułowy preprocessing.
Ma umożliwić łatwe dodanie nowych filtrów, metod referencji, kanałów i artefaktów.

Szymon:
Modułowe okienkowanie i cechy.
Ma umożliwić łatwe dodanie nowych długości okien, pasm, cech, agregatów i macierzy kowariancji.

Wiciu:
Modułowe modele i eksperymenty.
Ma umożliwić łatwe dodanie nowych modeli, walidacji, metryk, raportów i eksperymentów kontrolnych.
```

---

# Najważniejsza zasada projektu

```text
Nie piszemy jednorazowego notebooka.
Piszemy pipeline badawczy, który teraz robi MVP,
ale później można go rozbudować bez przepisywania wszystkiego od zera.
```

Projekt ma być prosty na start, ale przygotowany pod rozwój:

```text
MVP teraz
-> więcej cech później
-> więcej modeli później
-> więcej osób później
-> personalizacja później
-> online inference później
```
