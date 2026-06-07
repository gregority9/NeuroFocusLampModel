# EEG Focus Classifier

Pipeline do klasyfikacji okien EEG na stan spoczynku (`rest`, etykieta `0`) oraz zadanie poznawcze (`task`, etykieta `1`). Projekt jest częścią prac FocusLamp i służy do eksperymentów offline na danych BrainAccess Mini.

Model nie jest klasyfikatorem klinicznym ADHD. Zmienna `ADHD/control` jest używana do raportowania i analizy, a nie jako główny target predykcji.

## Dane

Zbiór zawiera nagrania od 9 osób:

- 5 osób z grupy ADHD,
- 4 osoby z grupy kontrolnej.

Każdy uczestnik ma trzy warunki eksperymentalne:

- `R` - rest, 180 sekund,
- `TASK1` - zadanie pamięciowe/liczenie w pamięci, 60 sekund,
- `TASK2` - zadanie słowne/generowanie słów na literę, 60 sekund.

Sygnał EEG pochodzi z 8 kanałów:

```text
AF3, AF4, F3, F4, FC5, FC6, O1, O2
```

Częstotliwość próbkowania: `250 Hz`.

## Struktura Projektu

```text
configs/
  base/                         # bazowe configi z dziedziczeniem
  benchmarks/                   # definicje siatek benchmarkowych
  model_*.yaml                  # krótkie override'y konkretnych eksperymentów

data/
  raw/                          # surowe dane uczestników
  processed/                    # dane po preprocessingu i feature table

docs/
  Dokumentacja Projektu - EEG Focus Classifier.txt
  Dokumentacja Projektu - EEG Focus Classifier.docx
  feature_table_contract.md

reports/
  experiments/                  # wyniki pojedynczych eksperymentów
  benchmarks/                   # manifesty i top-rankingi benchmarków

src/
  eeg_focus/                    # preprocessing i ekstrakcja cech
  models/                       # budowa sklearn Pipeline
  training/                     # trening LOSO i config loader
  evaluation/                   # metryki, raporty, benchmark runner
```

## Instalacja

Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

W dalszych komendach zakładamy, że środowisko jest aktywne.

## Pełny Workflow

### 1. Preprocessing

```bash
python -m src.eeg_focus.main
```

Co robi:

- ładuje pliki surowe z `data/raw`,
- waliduje długości i strukturę nagrań,
- filtruje EEG zgodnie z `configs/preprocessing.yaml`,
- liczy artefakty EEG/ruchowe,
- zapisuje wynik do `data/processed/preprocessing/preprocessed_all.csv`.

### 2. Ekstrakcja Cech

Projekt nie ma osobnego CLI dla feature extraction, więc etap odpalamy przez klasę pipeline'u:

```bash
python -B - <<'PY'
import yaml
from src.eeg_focus.pipelines.feature_extraction import FeatureExtractionPipeline

with open('configs/preprocessing.yaml', encoding='utf-8') as file:
    config = yaml.safe_load(file)

pipeline = FeatureExtractionPipeline(config)
result = pipeline.run('data/processed/preprocessing/preprocessed_all.csv')
print(result)
PY
```

Co robi:

- dzieli sygnał na okna 2-sekundowe,
- liczy bandpower dla pasm `delta`, `theta`, `alpha`, `beta`,
- dodaje cechy pochodne: `log_bandpower`, `ratios`, `regional`, `asymmetry`,
- zapisuje `data/processed/features_bandpower.csv`.

Szybki sanity check:

```bash
python -B - <<'PY'
import pandas as pd

df = pd.read_csv('data/processed/features_bandpower.csv', nrows=1)
print('columns:', len(df.columns))
print('log_theta_AF3' in df.columns)
print('ratio_theta_over_alpha_AF3' in df.columns)
print('theta_frontal_mean' in df.columns)
print('asym_alpha_F3_minus_F4' in df.columns)
PY
```

Po aktualnej ekstrakcji tabela ma `175` kolumn.

## Trening Pojedynczego Modelu

Baseline logistic regression:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.training.train configs/model_bandpower_logreg.yaml
```

Linear SVM:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.training.train configs/model_bandpower_svm.yaml
```

Wariant z pełnym zestawem feature groups:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.training.train configs/model_features_all_logreg.yaml
```

Wyniki pojedynczego eksperymentu zapisują się do:

```text
reports/experiments/<experiment_name>/
```

Najważniejsze pliki w katalogu eksperymentu:

```text
metrics.json                  # główne metryki
config_used.yaml              # pełny config po merge'u
metrics_per_subject.csv       # wyniki LOSO per osoba
metrics_per_task.csv          # wyniki per R/TASK1/TASK2
metrics_per_group.csv         # wyniki per ADHD/control
predictions.csv               # predykcje per okno
confusion_matrix.png          # macierz pomyłek
roc_curve.png                 # krzywa ROC
model.joblib                  # finalny model na całości danych
model_metadata.json           # metadane modelu
```

## Benchmark 300 Kombinacji

Definicja benchmarku:

```text
configs/benchmarks/features_models_full.yaml
```

Siatka obejmuje:

- 10 zestawów cech,
- 5 modeli,
- 3 warianty normalizacji,
- 2 tryby artefaktów.

Łącznie:

```text
10 x 5 x 3 x 2 = 300 eksperymentów
```

Uruchomienie benchmarku:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.evaluation.run_benchmark configs/benchmarks/features_models_full.yaml
```

Runner jest wznawialny. Jeśli eksperyment ma już `metrics.json`, zostanie pominięty.

Wymuszenie przeliczenia wszystkiego od zera:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.evaluation.run_benchmark configs/benchmarks/features_models_full.yaml --force
```

Testowe odpalenie pierwszych N eksperymentów:

```bash
env MPLCONFIGDIR=/tmp/mplconfig python -B -m src.evaluation.run_benchmark configs/benchmarks/features_models_full.yaml --limit 5
```

Manifest benchmarku:

```text
reports/benchmarks/features_models_full_manifest.json
```

Zbiorcza tabela wszystkich eksperymentów:

```text
reports/experiments/comparison.csv
```

Podgląd top 20:

```bash
python -B - <<'PY'
import pandas as pd

df = pd.read_csv('reports/experiments/comparison.csv')
print(df.head(20).to_string(index=False))
PY
```

## Top 15 Benchmarku

Top 15 opcji zapisano w:

```text
reports/benchmarks/features_models_full_top15/
```

W tym katalogu są:

```text
README.md
top15_summary.csv
01_<experiment>/
02_<experiment>/
...
15_<experiment>/
```

Najlepszy wariant:

```text
feature_set: regional
model: random_forest
normalization: subject_relative
artifacts: reject
balanced accuracy: 0.7537
F1: 0.6527
ROC AUC: 0.8117
```

## Aktualne Wnioski Z Benchmarku

Najważniejszy wynik metodologiczny: `subject_relative` wygrywa zdecydowanie. Normalizacja względem restu tej samej osoby redukuje różnice osobnicze i poprawia generalizację LOSO.

Najlepszy zestaw cech to `regional`, czyli uśrednione cechy regionalne. Wrzucenie wszystkich cech naraz (`all`) nie jest najlepsze, co sugeruje, że nadmiar cech dodaje szum.

Najlepiej wypadają modele drzewiaste, szczególnie `random_forest`. `rbf_svm` jest drugim najsilniejszym wariantem. Modele liniowe pozostają przydatne jako baseline i do interpretacji, ale nie dominują rankingu.

Odrzucanie artefaktów najczęściej pomaga w topowych wynikach, choć tryb `keep` również pojawia się wysoko. Wnioski o artefaktach należy traktować ostrożnie, bo ruch/napięcie mięśniowe może częściowo korelować z wykonywaniem zadania.

## Config Inheritance

Configi modeli używają dziedziczenia przez `extends`.

Przykład:

```yaml
extends: configs/base/model_bandpower_logreg.yaml

experiment:
  name: features_all_logreg

features:
  feature_groups:
    - raw_bandpower
    - log_bandpower
    - ratios
    - regional
    - asymmetry
```

Loader znajduje się w:

```text
src/training/config_loader.py
```

Obsługuje deep merge, więc override `features.feature_groups` nie usuwa `features.type` z bazowego configu.

## Metryki

`balanced_accuracy` jest główną metryką, bo klasy `rest` i `task` nie są idealnie zbalansowane. Liczy średnią skuteczność dla obu klas:

```text
balanced_accuracy = (recall_rest + recall_task) / 2
```

`F1` mierzy jakość predykcji klasy `task`, łącząc precision i recall.

`ROC AUC` mierzy zdolność modelu do separacji klas niezależnie od konkretnego progu decyzyjnego. `0.5` oznacza losowość, `1.0` wynik idealny.

## Najważniejsze Pliki Źródłowe

```text
src/eeg_focus/pipelines/preprocess.py              # preprocessing
src/eeg_focus/pipelines/feature_extraction.py      # ekstrakcja cech
src/eeg_focus/features/bandpower.py                # bandpower
src/eeg_focus/features/derived_bandpower.py        # cechy pochodne
src/models/model_pipeline.py                       # modele sklearn
src/training/train.py                              # trening LOSO
src/training/feature_selection.py                  # wybór grup cech
src/training/config_loader.py                      # extends/deep merge YAML
src/evaluation/run_benchmark.py                    # benchmark grid
src/evaluation/reports.py                          # zapis raportów
```

## Uwagi Metodologiczne

- Nie używać losowego `train_test_split` po oknach EEG, bo powoduje leakage między próbkami tej samej osoby.
- Raportować wyniki LOSO per subject, bo średnia może ukrywać słabe wyniki na konkretnych osobach.
- Interpretować topowe wyniki ostrożnie: próba ma tylko 9 osób.
- Najlepsze konfiguracje warto dalej testować na transferze `TASK1 -> TASK2` i `TASK2 -> TASK1` oraz na nowych osobach.
