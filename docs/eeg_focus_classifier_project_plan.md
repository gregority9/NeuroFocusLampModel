# Projekt klasyfikacji skupienia na podstawie EEG

## 1. Cel projektu

Celem projektu jest zbudowanie skalowalnego pipeline'u do klasyfikacji stanu poznawczego na podstawie sygnału EEG, w szczególności rozróżniania:

- stanu relaksu,
- stanu zaangażowania poznawczego / skupienia.

W aktualnej wersji dane pochodzą od 9 osób:

- 5 osób z ADHD,
- 4 osoby bez ADHD,
- każda osoba posiada nagranie relaksu oraz dwa nagrania zadań poznawczych.

Dane EEG pochodzą z czepka BrainAccess Mini, z kanałów:

```text
AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

Parametry danych:

```text
Częstotliwość próbkowania: 250 Hz
Relaks: 180 sekund
Skupienie / zadanie 1: 60 sekund, liczenie w pamięci
Skupienie / zadanie 2: 60 sekund, generowanie jak największej liczby słów na daną literę
Liczba kanałów: 8
```

Ważne: na obecnym etapie projekt powinien być opisywany ostrożnie jako klasyfikator:

```text
relaks vs zadanie poznawcze
```

albo:

```text
resting state vs cognitive engagement
```

Nie należy jeszcze twierdzić, że model klasyfikuje „skupienie” w pełnym psychologicznym sensie, ponieważ dane nie zawierają bezpośredniej, ciągłej miary skupienia, np. samooceny, reakcji behawioralnych, wyników zadania w czasie albo markerów rozproszenia.

---

## 2. Główne założenia metodologiczne

### 2.1. Mała liczba osób

Najważniejsze ograniczenie projektu to bardzo mała liczba niezależnych uczestników:

```text
N = 9 osób
```

Choć po podziale sygnału na okna liczba próbek treningowych będzie większa, realnie okna pochodzące od tej samej osoby są silnie zależne. Nie wolno traktować ich jako całkowicie niezależnych przykładów.

Największe ryzyko:

```text
model nauczy się rozpoznawać osoby, artefakty lub specyfikę sesji,
zamiast faktycznego stanu poznawczego.
```

Dlatego walidacja musi być prowadzona po osobach, a nie po losowych oknach.

---

### 2.2. ADHD jako zmienna grupująca, nie główna etykieta

W obecnej wersji projektu ADHD nie powinno być główną etykietą klasyfikacyjną. Przy 5 osobach z ADHD i 4 bez ADHD nie ma podstaw do budowy wiarygodnego klasyfikatora ADHD vs control.

Rekomendowane użycie zmiennej ADHD:

```text
- analiza odporności modelu,
- raportowanie wyników osobno dla ADHD i grupy kontrolnej,
- sprawdzanie, czy model nie działa wyraźnie gorzej w jednej z grup,
- analiza różnic eksploracyjnych, bez mocnych wniosków klinicznych.
```

Główny cel modelowania:

```text
relaks vs zaangażowanie poznawcze
```

---

### 2.3. Unikanie leakage

Nie należy robić klasycznego:

```python
train_test_split(X_windows, y_windows)
```

jeżeli `X_windows` zawiera okna z tych samych osób.

To prowadzi do leakage, ponieważ okna tej samej osoby mogą trafić jednocześnie do zbioru treningowego i testowego. Model może wtedy nauczyć się cech osobniczych zamiast różnicy relaks/skupienie.

Poprawna strategia:

```text
Leave-One-Subject-Out Cross Validation
```

czyli:

```text
trenujemy na 8 osobach,
testujemy na 1 osobie,
powtarzamy dla każdej osoby.
```

---

## 3. Struktura danych

### 3.1. Surowe dane

Rekomendowana organizacja:

```text
data/
  raw/
    sub-001/
      rest.csv
      focus_memory.csv
      focus_words.csv
    sub-002/
      rest.csv
      focus_memory.csv
      focus_words.csv
    ...
```

Każdy plik powinien zawierać:

```text
timestamp, AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

Opcjonalnie:

```text
marker, sample_index, impedance, battery, quality_flag
```

jeżeli takie informacje są dostępne.

---

### 3.2. Metadane uczestników

Plik:

```text
data/metadata/participants.tsv
```

Przykładowa struktura:

```tsv
subject_id	group	age	sex	handedness	medication	notes
sub-001	ADHD	NA	NA	NA	NA	NA
sub-002	control	NA	NA	NA	NA	NA
```

Na przyszłość warto zbierać:

- wiek,
- płeć,
- dominującą rękę,
- status leków,
- sen,
- kofeinę,
- zmęczenie,
- porę dnia,
- jakość kontaktu elektrod,
- informację, czy osoba była wcześniej zaznajomiona z zadaniem.

---

### 3.3. Metadane nagrań

Plik:

```text
data/metadata/recordings.tsv
```

Przykład:

```tsv
subject_id	session_id	task	file	sampling_rate	duration_sec	channels
sub-001	ses-001	rest	data/raw/sub-001/rest.csv	250	180	AF3,AF4,F3,F4,FC5,FC6,O1,O2
sub-001	ses-001	focus_memory	data/raw/sub-001/focus_memory.csv	250	60	AF3,AF4,F3,F4,FC5,FC6,O1,O2
sub-001	ses-001	focus_words	data/raw/sub-001/focus_words.csv	250	60	AF3,AF4,F3,F4,FC5,FC6,O1,O2
```

---

## 4. Preprocessing EEG

Preprocessing powinien być osobnym, konfigurowalnym etapem. Nie powinien być wymieszany z kodem modelu.

Proponowane moduły:

```text
src/eeg_focus/preprocessing/
  filters.py
  referencing.py
  artifact_detection.py
  quality_report.py
```

---

### 4.1. Kontrola jakości sygnału

Dla każdego nagrania należy automatycznie sprawdzić:

```text
- liczbę próbek,
- częstotliwość próbkowania,
- brakujące próbki,
- duplikaty timestampów,
- NaN / Inf,
- clipping / saturację,
- ekstremalne wartości,
- kanały martwe,
- kanały z bardzo niską wariancją,
- kanały z bardzo wysoką wariancją,
- gwałtowne skoki amplitudy,
- globalny poziom szumu.
```

Raport jakości powinien być zapisywany np. do:

```text
reports/qc/sub-001_rest_qc.json
reports/qc/sub-001_rest_qc.html
```

Minimalne statystyki per kanał:

```text
mean
std
median
MAD
min
max
peak-to-peak
RMS
kurtosis
skewness
percentage_of_missing_samples
percentage_of_clipped_samples
```

---

### 4.2. Filtrowanie

Rekomendowane parametry bazowe:

```yaml
filter:
  highpass_hz: 1.0
  lowpass_hz: 40.0
  notch_hz: 50.0
```

Uzasadnienie:

- high-pass 1 Hz usuwa wolne dryfty,
- low-pass 40 Hz ogranicza artefakty mięśniowe i wysokoczęstotliwościowy szum,
- notch 50 Hz ogranicza zakłócenia sieciowe.

Na obecnym etapie nie rekomenduje się używania pasm powyżej 40 Hz jako głównego źródła informacji, ponieważ przy małej liczbie kanałów i zadaniach wymagających mowy/myślenia łatwo złapać napięcie mięśni twarzy, czoła lub szczęki.

---

### 4.3. Referencja

Należy przetestować co najmniej trzy warianty:

```text
1. brak zmiany referencji,
2. common average reference z dobrych kanałów,
3. średnia referencja po kanałach zaakceptowanych po QC.
```

Wybór referencji powinien być dokonany na podstawie wyników walidacji po osobach oraz stabilności sygnału, nie na podstawie pojedynczego nagrania.

---

### 4.4. Detekcja i odrzucanie artefaktów

Przy kanałach AF3, AF4, F3, F4 należy spodziewać się silnych artefaktów od:

```text
- mrugnięć,
- ruchów oczu,
- napięcia czoła,
- ruchów szczęki,
- mowy,
- mikroruchów głowy,
- złego kontaktu elektrody.
```

Na start rekomenduję podejście konserwatywne:

```text
nie usuwać automatycznie komponentów ICA,
tylko oznaczać i odrzucać złe okna.
```

Przykładowe kryteria odrzucania okien:

```text
- zbyt duży peak-to-peak amplitude,
- RMS znacznie powyżej mediany osoby,
- kurtosis powyżej progu,
- gwałtowne skoki między próbkami,
- nietypowo wysoka moc 30-40 Hz,
- bardzo niska wariancja kanału,
- bardzo wysoka korelacja kanału z innymi kanałami w sposób niefizjologiczny.
```

Progi powinny być możliwie relatywne względem osoby/sesji, np.:

```text
z-score względem mediany i MAD
```

zamiast sztywnych progów globalnych.

---

### 4.5. ICA

ICA może być dodane jako opcjonalny etap w przyszłości, ale nie powinno być obowiązkowe w pierwszej wersji pipeline'u.

Powody:

- tylko 8 kanałów,
- mały dataset,
- automatyczne ICA może usuwać użyteczny sygnał,
- ryzyko niekontrolowanego wpływu preprocessingu na wynik,
- trudność z walidacją, czy usuwane komponenty są rzeczywiście artefaktami.

Rekomendacja:

```text
v1: artifact rejection per window
v2: opcjonalne ICA z ręczną/automatyczną oceną komponentów
v3: porównanie pipeline'u z ICA i bez ICA w LOSO
```

---

## 5. Segmentacja

Dane należy podzielić na okna czasowe.

Rekomendacja bazowa:

```yaml
epoching:
  window_sec: 4.0
  overlap: 0.5
```

Przy 250 Hz:

```text
4 s = 1000 próbek
overlap 50% = krok 2 s
```

Szacunkowa liczba okien na osobę:

```text
relaks: 180 s -> około 89 okien
skupienie łącznie: 120 s -> około 59 okien
razem: około 148 okien/osobę
```

Łącznie dla 9 osób:

```text
około 1330 okien
```

Ważne: to nadal nie oznacza 1330 niezależnych przykładów. Niezależnych jednostek jest 9 osób.

---

## 6. Etykiety

Proponowane etykiety bazowe:

```text
rest -> 0
focus_memory -> 1
focus_words -> 1
```

Dodatkowe kolumny:

```text
task:
  rest
  focus_memory
  focus_words

group:
  ADHD
  control

subject_id:
  sub-001, sub-002, ...

session_id:
  ses-001
```

W pliku z oknami:

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

---

## 7. Ekstrakcja cech

W pierwszej wersji projektu rekomendowane są dwie równoległe ścieżki cech:

```text
A. cechy pasmowe i interpretowalne,
B. cechy kowariancyjne / riemannowskie.
```

---

### 7.1. Cechy pasmowe

Dla każdego kanału i okna należy policzyć moc w pasmach:

```yaml
bands:
  delta: [1, 4]
  theta: [4, 8]
  alpha: [8, 13]
  beta: [13, 30]
  high_beta: [30, 40]
```

Cechy:

```text
absolute_bandpower
relative_bandpower
log_bandpower
```

Rekomendowane cechy pochodne:

```text
theta_beta_ratio
alpha_beta_ratio
frontal_theta
occipital_alpha
left_right_asymmetry
frontal_to_occipital_ratio
```

Przykłady:

```text
theta_beta_F3 = theta_power_F3 / beta_power_F3
theta_beta_F4 = theta_power_F4 / beta_power_F4

alpha_occipital = mean(alpha_O1, alpha_O2)

frontal_theta = mean(theta_AF3, theta_AF4, theta_F3, theta_F4)

asymmetry_F3_F4_alpha = log(alpha_F3) - log(alpha_F4)
asymmetry_O1_O2_alpha = log(alpha_O1) - log(alpha_O2)
```

---

### 7.2. Cechy specyficzne dla kanałów

Kanały można pogrupować funkcjonalnie:

```text
frontal/attention:
  AF3, AF4, F3, F4

fronto-central:
  FC5, FC6

occipital:
  O1, O2
```

Możliwe agregaty:

```text
frontal_theta_mean
frontal_beta_mean
occipital_alpha_mean
occipital_alpha_suppression
frontal_beta_to_occipital_alpha
left_frontal_vs_right_frontal
left_occipital_vs_right_occipital
```

---

### 7.3. Cechy kowariancyjne

Dla każdego okna:

```text
X_window: channels x samples
```

czyli:

```text
8 x 1000
```

Liczymy macierz kowariancji:

```text
C = covariance(X_window)
```

Macierz ma wymiar:

```text
8 x 8
```

Następnie można użyć metod riemannowskich:

```text
CovarianceEstimator
TangentSpace
LogisticRegression
```

albo:

```text
Minimum Distance to Mean
```

To podejście często dobrze sprawdza się przy EEG i małych datasetach, ponieważ wykorzystuje relacje przestrzenne między kanałami.

---

## 8. Normalizacja

Normalizacja jest krytyczna.

### 8.1. Zasada najważniejsza

Scaler nie może być uczony na danych testowych.

W każdej iteracji LOSO:

```text
fit scaler tylko na train
transform train
transform test
```

Nie wolno robić:

```python
scaler.fit(X_all)
```

przed podziałem na foldy.

---

### 8.2. Normalizacja globalna

Dla cech klasycznych:

```text
StandardScaler
RobustScaler
```

Rekomendacja bazowa:

```text
RobustScaler albo StandardScaler
```

wybierany w eksperymencie.

---

### 8.3. Normalizacja względem relaksu osoby

Bardzo ważna opcja:

```text
feature_relative = (feature_window - mean_rest_subject) / std_rest_subject
```

To pozwala modelowi wykrywać:

```text
czy dana osoba jest bardziej zaangażowana niż jej własny baseline
```

zamiast:

```text
czy osoba ma ogólnie wysoką/niższą moc EEG.
```

Uwaga metodologiczna:

- w testowaniu nowej osoby można użyć jej krótkiego nagrania relaksu jako kalibracji,
- trzeba jasno opisać, że model wymaga baseline'u osoby,
- należy porównać wersję z kalibracją i bez kalibracji.

---

## 9. Modele

### 9.1. Baseline 0 — analiza bez ML

Zanim rozpocznie się uczenie modelu, należy wykonać sanity check:

```text
PSD relaks vs skupienie
alpha O1/O2 relaks vs skupienie
theta F3/F4 relaks vs skupienie
porównanie memory vs words
porównanie ADHD vs control eksploracyjnie
```

Celem jest sprawdzenie, czy sygnał zawiera spodziewalne wzorce i czy nie jest zdominowany przez artefakty.

---

### 9.2. Baseline 1 — bandpower + Logistic Regression

Pierwszy właściwy model:

```text
cechy pasmowe
+ StandardScaler/RobustScaler
+ LogisticRegression(class_weight="balanced")
```

Zalety:

```text
- prosty,
- szybki,
- interpretowalny,
- odporniejszy na overfitting niż modele złożone,
- dobry do raportu.
```

---

### 9.3. Baseline 2 — bandpower + Linear SVM

Drugi klasyczny model:

```text
cechy pasmowe
+ scaler
+ LinearSVC albo SVC(kernel="linear")
```

Zalety:

```text
- prosty,
- dobry przy małych datasetach,
- może działać lepiej niż regresja logistyczna przy określonej geometrii cech.
```

Wady:

```text
- prawdopodobieństwa wymagają kalibracji,
- mniej bezpośrednia interpretacja niż logistic regression.
```

---

### 9.4. Model główny — Riemannian Tangent Space + Logistic Regression

Rekomendowany główny kandydat:

```text
okno EEG
-> macierz kowariancji
-> przestrzeń styczna
-> Logistic Regression
```

Schemat:

```text
X_window: 8 x 1000
C_window: 8 x 8
TangentSpace(C_window): vector
Classifier: LogisticRegression
```

Zalety:

```text
- dobrze dopasowane do EEG,
- wykorzystuje relacje między kanałami,
- relatywnie dobre przy małych datasetach,
- szybkie obliczeniowo,
- skalowalne na więcej osób.
```

---

### 9.5. Random Forest / XGBoost

Można przetestować jako dodatkowe modele, ale nie jako główne.

Ryzyka:

```text
- łatwe przeuczenie,
- kuszące wysokie wyniki przy leakage,
- mniejsza stabilność przy małej liczbie osób.
```

Jeżeli używane:

```text
- tylko z LOSO,
- z ograniczoną głębokością drzew,
- z raportem wyników per subject,
- z porównaniem do prostszych baseline'ów.
```

---

### 9.6. Deep learning

Na obecnym etapie nie rekomenduję deep learningu jako głównego podejścia.

Nie zaczynać od:

```text
- LSTM,
- Transformer,
- dużego CNN,
- EEGNet jako głównego modelu.
```

Powody:

```text
- tylko 9 osób,
- duże ryzyko overfittingu,
- ryzyko uczenia się artefaktów,
- trudniejsza interpretacja,
- brak potrzeby obliczeniowej.
```

Deep learning można dodać później, gdy będą:

```text
- dziesiątki lub setki osób,
- wiele sesji na osobę,
- więcej zadań,
- lepsze etykiety,
- większa kontrola jakości.
```

---

## 10. Walidacja

### 10.1. Główna walidacja

Podstawowa strategia:

```text
Leave-One-Subject-Out Cross Validation
```

Implementacyjnie:

```python
from sklearn.model_selection import LeaveOneGroupOut

groups = subject_ids
cv = LeaveOneGroupOut()

for train_idx, test_idx in cv.split(X, y, groups=groups):
    ...
```

Każdy fold:

```text
train: 8 osób
test: 1 osoba
```

Raport końcowy:

```text
średnia po foldach
odchylenie standardowe
wyniki per osoba
```

---

### 10.2. Walidacja między zadaniami

Należy sprawdzić, czy model uczy się ogólnego zaangażowania poznawczego, czy tylko specyfiki jednego zadania.

Eksperyment A:

```text
train: rest vs focus_memory
test: rest vs focus_words
```

Eksperyment B:

```text
train: rest vs focus_words
test: rest vs focus_memory
```

Jeżeli model działa dobrze tylko wewnątrz jednego zadania, a słabo między zadaniami, to nie można twierdzić, że klasyfikuje ogólne skupienie.

---

### 10.3. Walidacja per grupa

Raportować osobno:

```text
wyniki dla ADHD
wyniki dla control
```

Metryki:

```text
balanced accuracy ADHD
balanced accuracy control
F1 ADHD
F1 control
false positive rate ADHD
false positive rate control
false negative rate ADHD
false negative rate control
```

Przy tak małej próbie traktować te wyniki jako eksploracyjne.

---

### 10.4. Metryki

Nie wystarczy accuracy.

Rekomendowane metryki:

```text
balanced accuracy
F1-score
ROC-AUC
precision
recall
confusion matrix
wyniki per subject
wyniki per task
```

Ważne: metryki można liczyć zarówno per okno, jak i per sesja.

Dla decyzji sesyjnej można uśredniać predykcje z okien:

```text
p_session = mean(p_focus_window)
```

albo użyć mediany:

```text
p_session = median(p_focus_window)
```

Mediana może być odporniejsza na pojedyncze artefakty.

---

## 11. Raportowanie wyników

Każdy eksperyment powinien generować raport:

```text
reports/experiments/{experiment_name}/
  metrics.json
  metrics_per_subject.csv
  confusion_matrix.png
  roc_curve.png
  feature_importance.csv
  config.yaml
  notes.md
```

Minimalny raport:

```text
- nazwa eksperymentu,
- data,
- hash/wersja danych,
- konfiguracja preprocessingu,
- konfiguracja cech,
- konfiguracja modelu,
- strategia walidacji,
- metryki ogólne,
- metryki per osoba,
- metryki per grupa ADHD/control,
- metryki per task,
- wykresy PSD,
- komentarz interpretacyjny.
```

---

## 12. Interpretowalność

Dla modelu bandpower + logistic regression można analizować wagi cech:

```text
które kanały i pasma zwiększają prawdopodobieństwo klasy „focus”,
które zmniejszają,
czy dominują kanały frontalne,
czy pojawia się alpha suppression w O1/O2,
czy model używa głównie high_beta, co może sugerować artefakty mięśniowe.
```

Dla modeli riemannowskich interpretacja jest trudniejsza, ale można:

```text
- porównywać macierze kowariancji relaks vs focus,
- analizować wzorce połączeń między kanałami,
- sprawdzać stabilność wyników po usunięciu kanałów,
- robić permutation importance.
```

---

## 13. Kontrola artefaktów i sanity checks

Należy sprawdzić, czy model nie klasyfikuje artefaktów.

Testy kontrolne:

```text
1. Wyniki po usunięciu kanałów frontalnych AF3/AF4.
2. Wyniki po usunięciu pasma 30-40 Hz.
3. Wyniki osobno dla kanałów potylicznych O1/O2.
4. Wyniki osobno dla kanałów frontalnych.
5. Porównanie okien odrzuconych i zaakceptowanych.
6. Korelacja predykcji modelu z artifact_score.
7. Sprawdzenie, czy focus_words nie generuje większych artefaktów mięśniowych przez cichą mowę.
```

Jeżeli model mocno opiera się na high beta / 30-40 Hz albo kanałach AF3/AF4, trzeba ostrożnie interpretować wynik.

---

## 14. Proponowana architektura repozytorium

```text
eeg-focus-classifier/
│
├── data/
│   ├── raw/
│   │   ├── sub-001/
│   │   │   ├── rest.csv
│   │   │   ├── focus_memory.csv
│   │   │   └── focus_words.csv
│   │   └── sub-002/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│       ├── participants.tsv
│       └── recordings.tsv
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
│       ├── __init__.py
│       │
│       ├── io/
│       │   ├── loaders.py
│       │   ├── validators.py
│       │   └── export.py
│       │
│       ├── preprocessing/
│       │   ├── filters.py
│       │   ├── referencing.py
│       │   ├── artifact_detection.py
│       │   └── quality_report.py
│       │
│       ├── epoching/
│       │   └── windows.py
│       │
│       ├── features/
│       │   ├── bandpower.py
│       │   ├── ratios.py
│       │   ├── asymmetry.py
│       │   ├── covariance.py
│       │   └── feature_table.py
│       │
│       ├── models/
│       │   ├── classical.py
│       │   ├── riemannian.py
│       │   ├── calibration.py
│       │   └── registry.py
│       │
│       ├── evaluation/
│       │   ├── loso.py
│       │   ├── task_transfer.py
│       │   ├── metrics.py
│       │   └── reports.py
│       │
│       ├── visualization/
│       │   ├── psd.py
│       │   ├── topography.py
│       │   └── diagnostics.py
│       │
│       └── pipelines/
│           ├── preprocess.py
│           ├── extract_features.py
│           ├── train.py
│           ├── evaluate.py
│           └── predict.py
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
│   └── test_loso_split.py
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 15. Konfiguracja eksperymentów

Parametry powinny być sterowane przez YAML, nie hardcodowane.

Przykład:

```yaml
project:
  name: eeg_focus_classifier
  random_state: 42

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

preprocessing:
  filter:
    highpass_hz: 1.0
    lowpass_hz: 40.0
    notch_hz: 50.0
  reference:
    method: common_average
  artifact_rejection:
    enabled: true
    rms_z_threshold: 4.0
    kurtosis_z_threshold: 4.0
    high_beta_z_threshold: 4.0

epoching:
  window_sec: 4.0
  overlap: 0.5

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

normalization:
  scaler: robust
  subject_relative_to_rest: true

model:
  type: logistic_regression
  class_weight: balanced
  C: 1.0

validation:
  strategy: leave_one_subject_out
  group_column: subject_id

outputs:
  save_predictions: true
  save_feature_importance: true
  save_reports: true
```

---

## 16. Główne pipeline'y

### 16.1. Pipeline preprocessingowy

```text
load raw EEG
-> validate sampling
-> reorder channels
-> detect missing samples
-> interpolate/mark missing data
-> filter 1-40 Hz
-> notch 50 Hz
-> rereference
-> generate QC report
-> save clean/interim signal
```

---

### 16.2. Pipeline okienkowania

```text
load preprocessed signal
-> split into windows
-> assign labels
-> calculate artifact score per window
-> mark rejected windows
-> save window metadata
```

---

### 16.3. Pipeline cech

```text
load accepted windows
-> compute bandpower features
-> compute ratio features
-> compute asymmetry features
-> compute covariance features
-> save feature table
```

---

### 16.4. Pipeline treningu

```text
load features
-> split by Leave-One-Subject-Out
-> fit scaler on train
-> transform train/test
-> train model
-> predict test
-> save predictions
-> repeat for all subjects
-> aggregate metrics
-> generate report
```

---

### 16.5. Pipeline predykcji dla nowej osoby

Wariant bez kalibracji:

```text
new raw EEG
-> preprocessing
-> windowing
-> feature extraction
-> global scaler/model
-> p_focus per window
-> session-level score
```

Wariant z kalibracją relaksową:

```text
new rest baseline
-> compute subject baseline statistics
new task EEG
-> compute features
-> normalize relative to subject rest
-> model prediction
```

---

## 17. Plan eksperymentów

### Eksperyment 0: jakość sygnału

Cel:

```text
ocenić, czy dane nadają się do modelowania
```

Wyniki:

```text
- QC per osoba,
- liczba odrzuconych okien,
- PSD per kanał,
- porównanie relaks vs focus,
- wykrycie kanałów problematycznych.
```

---

### Eksperyment 1: bandpower + Logistic Regression

Konfiguracja:

```text
okna 4 s
overlap 50%
filtr 1-40 Hz
notch 50 Hz
common average reference
cechy bandpower + ratios + asymmetry
RobustScaler
LogisticRegression
LOSO
```

Cel:

```text
uzyskanie pierwszego interpretowalnego baseline'u.
```

---

### Eksperyment 2: bandpower + Linear SVM

Jak Eksperyment 1, ale model:

```text
Linear SVM
```

Cel:

```text
porównanie prostych modeli liniowych.
```

---

### Eksperyment 3: Riemannian Tangent Space

Konfiguracja:

```text
okna 4 s
filtr 1-40 Hz
macierze kowariancji
TangentSpace
LogisticRegression
LOSO
```

Cel:

```text
sprawdzenie, czy cechy kowariancyjne dają lepszą generalizację.
```

---

### Eksperyment 4: bez subject-relative normalization

Porównanie:

```text
model z normalizacją względem relaksu osoby
vs
model bez tej normalizacji
```

Cel:

```text
ocenić, czy model korzysta z indywidualnego baseline'u.
```

---

### Eksperyment 5: generalizacja między zadaniami

Wariant A:

```text
train: rest vs memory
test: rest vs words
```

Wariant B:

```text
train: rest vs words
test: rest vs memory
```

Cel:

```text
sprawdzić, czy model rozpoznaje ogólne zaangażowanie poznawcze,
a nie specyfikę jednego zadania.
```

---

### Eksperyment 6: analiza grup ADHD/control

Cel:

```text
sprawdzić, czy model działa podobnie w grupie ADHD i kontrolnej.
```

Raport:

```text
- balanced accuracy per grupa,
- F1 per grupa,
- false positives/false negatives per grupa,
- predykcje per osoba.
```

---

### Eksperyment 7: kontrola artefaktów

Testy:

```text
- bez pasma 30-40 Hz,
- bez kanałów AF3/AF4,
- tylko kanały O1/O2,
- tylko kanały frontalne,
- korelacja artifact_score z p_focus,
- porównanie wyników przed i po odrzuceniu artefaktów.
```

Cel:

```text
upewnić się, że model nie bazuje głównie na artefaktach.
```

---

## 18. Wymagania obliczeniowe

Przy obecnym rozmiarze danych trening nie będzie wymagający.

Szacunkowo:

```text
9 osób x 300 s x 250 Hz x 8 kanałów
= około 5,4 mln próbek sygnału EEG
```

Po okienkowaniu:

```text
około 1330 okien
```

To jest mały dataset.

Wystarczy:

```text
CPU: zwykły laptop
RAM: 8-16 GB
GPU: niepotrzebne
```

Najcięższe etapy:

```text
- filtrowanie sygnału,
- liczenie PSD/bandpower,
- wielokrotna walidacja LOSO,
- generowanie raportów.
```

Modele klasyczne i riemannowskie powinny trenować się bardzo szybko.

GPU będzie potrzebne dopiero przy:

```text
- deep learningu,
- dużej liczbie osób,
- długich nagraniach,
- wielu sesjach,
- dużym strojeniu hiperparametrów.
```

---

## 19. Biblioteki

Rekomendowany stack:

```text
Python
NumPy
SciPy
pandas
scikit-learn
MNE-Python
pyRiemann
matplotlib
seaborn opcjonalnie do raportów
PyYAML / OmegaConf
MLflow opcjonalnie
pytest
```

Uwaga: w kodzie produkcyjnym warto ograniczyć zależność notebooków od logiki eksperymentu. Notebooki powinny służyć do eksploracji, a właściwy pipeline powinien być w `src/`.

---

## 20. Minimalny pseudokod treningu

```python
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score

X, y, groups = load_feature_table()

cv = LeaveOneGroupOut()

all_predictions = []

for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fold_result = {
        "fold_id": fold_id,
        "test_subject": groups[test_idx][0],
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    all_predictions.append(fold_result)
```

---

## 21. Minimalny pseudokod pipeline'u riemannowskiego

```python
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.metrics import balanced_accuracy_score

X_windows, y, groups = load_windows_as_array()
# X_windows shape: n_windows x n_channels x n_samples

cv = LeaveOneGroupOut()

for train_idx, test_idx in cv.split(X_windows, y, groups=groups):
    X_train, X_test = X_windows[train_idx], X_windows[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = balanced_accuracy_score(y_test, y_pred)
```

---

## 22. Ryzyka projektu

### 22.1. Ryzyko overfittingu

Powód:

```text
mało osób, dużo okien zależnych od siebie.
```

Mitigacja:

```text
LOSO,
proste modele,
raport per subject,
kontrole artefaktów,
unikanie deep learningu na start.
```

---

### 22.2. Ryzyko uczenia się artefaktów

Powód:

```text
kanały frontalne,
zadanie słowne,
napięcie twarzy,
mrugnięcia,
ruch oczu.
```

Mitigacja:

```text
artifact rejection,
analiza high beta,
testy bez kanałów frontalnych,
korelacja artifact_score z predykcją.
```

---

### 22.3. Ryzyko zbyt szerokiej interpretacji

Model może klasyfikować:

```text
relaks vs zadanie
```

a nie:

```text
skupienie jako stan psychologiczny.
```

Mitigacja:

```text
ostrożna nomenklatura,
dodanie w przyszłości metryk behawioralnych,
dodanie samooceny skupienia,
dodanie wielu poziomów trudności.
```

---

### 22.4. Ryzyko wpływu ADHD jako confoundera

Przy małej próbie model może mylić różnice osobnicze lub grupowe z etykietą zadania.

Mitigacja:

```text
LOSO,
wyniki per grupa,
analiza per subject,
nieużywanie ADHD jako głównej etykiety,
zwiększenie liczby osób w przyszłości.
```

---

## 23. Roadmapa rozwoju

### Faza 1: MVP badawcze

Cel:

```text
zbudować kompletny offline pipeline i uzyskać wiarygodny baseline.
```

Zakres:

```text
- import danych,
- QC,
- filtering,
- windowing,
- bandpower features,
- Logistic Regression,
- LOSO,
- podstawowy raport.
```

---

### Faza 2: Pipeline riemannowski

Zakres:

```text
- macierze kowariancji,
- Tangent Space,
- Logistic Regression,
- porównanie z bandpower,
- raport generalizacji między zadaniami.
```

---

### Faza 3: Kontrola artefaktów

Zakres:

```text
- artifact scoring,
- testy kanałów,
- testy pasm,
- korelacja artefaktów z predykcją,
- raport odporności.
```

---

### Faza 4: Skalowanie danych

Zakres:

```text
- więcej osób,
- więcej sesji,
- powtarzalność między dniami,
- dodatkowe poziomy trudności,
- markerowanie wyników zadań,
- subiektywna ocena skupienia.
```

---

### Faza 5: Personalizacja

Zakres:

```text
- baseline relaksowy nowej osoby,
- adaptacja modelu do osoby,
- kalibracja per subject,
- porównanie global model vs personalized model.
```

---

### Faza 6: Online inference

Zakres:

```text
- bufor danych 4 s,
- predykcja co 1-2 s,
- rolling average p_focus,
- wykrywanie artefaktów online,
- prosty dashboard.
```

---

## 24. Rekomendacja końcowa

Najlepsza architektura na obecnym etapie:

```text
MNE-Python + scikit-learn + pyRiemann
```

Dwa główne pipeline'y modelowe:

```text
1. Bandpower + Logistic Regression
2. Covariance + Tangent Space + Logistic Regression
```

Obowiązkowa walidacja:

```text
Leave-One-Subject-Out
```

Obowiązkowe raportowanie:

```text
- wyniki per osoba,
- wyniki per grupa ADHD/control,
- wyniki per task,
- balanced accuracy,
- F1,
- confusion matrix,
- kontrola artefaktów.
```

Najważniejsze zasady:

```text
- nie dzielić losowo okien na train/test,
- nie zaczynać od deep learningu,
- nie interpretować wyników klinicznie,
- nie twierdzić, że model mierzy czyste skupienie,
- zacząć od prostego, interpretowalnego i skalowalnego pipeline'u.
```

Najbardziej obronne sformułowanie celu:

```text
Budujemy klasyfikator EEG rozróżniający stan spoczynku od stanu zaangażowania poznawczego, z docelową możliwością rozszerzenia w kierunku personalizowanej klasyfikacji skupienia.
```
