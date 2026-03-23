# k-NN Classifier

A from-scratch implementation of the k-Nearest Neighbours classification algorithm in Python, with a terminal interface, optional GUI, and accuracy visualization.

---

## Features

- Pure Python kNN — no scikit-learn or ML libraries
- Euclidean distance across all attributes (dynamic, not hardcoded)
- Distance-based tie-breaking when majority vote results in a draw
- Labeled and unlabeled test data support
- Accuracy vs k chart (matplotlib)
- Optional Tkinter GUI with file browser, live classification, and embedded chart
- Label normalization (case, whitespace, underscore/dash)
- Three included datasets for testing

---

## Project Structure

```
├── main.py              # Entry point — terminal mode or GUI launcher
├── knn.py               # Core algorithm: distance, neighbours, classify
├── data_loader.py       # File parser and label normalizer
├── evaluator.py         # Evaluation engine and results printer
├── knn_gui.py           # Tkinter GUI (optional)
│
├── iris_training_1.txt  # Iris dataset — training split (120 samples)
├── iris_test_1.txt      # Iris dataset — test split (30 samples)
├── noisy_training.txt   # 2D noisy dataset — 3 classes
├── noisy_test.txt       # 2D noisy dataset — test
├── complex_training.txt # 4D dataset — 10 classes (alpha–kappa)
├── complex_test.txt     # 4D dataset — test
│
├── test_knn.py          # Quick manual test script
└── test_data_loader.py  # Data loader sanity check
```

---

## Algorithm

### Distance

Euclidean distance across all feature columns:

```
d(x, y) = sqrt( Σ (xi − yi)² )
```

### Classification

1. Compute distance from the test sample to every training sample.
2. Sort by distance ascending, take the `k` closest.
3. **Majority vote** — the class with the most neighbours wins.
4. **Tie-breaking** — if two or more classes tie, the class of the single closest neighbour wins. Tie resolution is always distance-based.

### Tie-breaking detail

```python
# neighbours already sorted by distance ascending
for dist, sample in neighbours:
    if sample["label"] in top_classes:
        return sample["label"]
```

This guarantees a deterministic result without any arbitrary random choice.

---

## File Format

Tab-separated columns, comma as decimal separator. Last column is the class label (optional).

```
5,1    3,5    1,4    0,2    Iris-setosa
6,4    3,2    4,5    1,5    Iris-versicolor
```

The loader handles:
- Leading/trailing whitespace
- Mixed case labels (`Iris-Setosa` → `iris-setosa`)
- Underscore vs dash (`iris_setosa` → `iris-setosa`)
- Comma decimal separators (`5,1` → `5.1`)
- Files without a label column (`has_label=False`)

---

## Usage

### Requirements

```bash
pip install matplotlib
```

Tkinter is included with standard Python. Matplotlib is only required for the accuracy chart and GUI chart view.

### Run

```bash
python main.py
```

**Configuration** is at the top of `main.py`:

```python
ASK_FILE_PATHS = False          # True = prompt for paths, False = use below
TRAINING_FILE  = "iris_training_1.txt"
TEST_FILE      = "iris_test_1.txt"
HAS_LABEL      = True           # False if test file has no class column
CHART_MAX_K    = None           # None = full training set, or e.g. 20
CHART_ENABLED  = True
GUI_ENABLED    = True           # True = launch GUI, False = terminal
```

### Terminal mode example

```
============================================================
        k-NN Classifier
============================================================

Training file    : iris_training_1.txt
Test file        : iris_test_1.txt
Training samples : 120
Test samples     : 30
Attributes       : 4

Enter k (number of neighbours, max 120): 3

#     True Label                Predicted                 Result
-----------------------------------------------------------------
1     iris-setosa               iris-setosa               correct
2     iris-setosa               iris-setosa               correct
...
-----------------------------------------------------------------
k                   : 3
Total samples       : 30
Correctly classified: 30
Errors              : 0
Accuracy            : 100.00%
```

### GUI mode

Set `GUI_ENABLED = True` in `main.py`. The GUI provides:

- File browser for training and test files
- k selector (spinner)
- Evaluate button with scrollable results panel
- Accuracy vs k chart in a separate window (computed in a background thread)
- Manual sample classifier — enter attribute values and predict on the fly

---

## Datasets

| Dataset | Attributes | Classes | Training | Test |
|---|---|---|---|---|
| Iris | 4 | 3 | 120 | 30 |
| Noisy | 2 | 3 (a, b, c) | 90 | 30 |
| Complex | 4 | 10 (alpha–kappa) | ~200 | 60 |

---

## Module Reference

### `knn.py`

| Function | Description |
|---|---|
| `euclidean_distance(x, y)` | Distance between two attribute vectors |
| `get_neighbours(training_data, test_sample, k)` | Returns k nearest as `(dist, sample)` tuples |
| `classify(training_data, test_sample, k)` | Returns predicted class label |

### `data_loader.py`

| Function | Description |
|---|---|
| `load_data(filepath, has_label)` | Parses file → list of `{attributes, label}` dicts |
| `normalize_label(label)` | Normalizes casing, whitespace, separators |
| `get_num_attributes(samples)` | Returns number of features in first sample |

### `evaluator.py`

| Function | Description |
|---|---|
| `evaluate(training_data, test_data, k)` | Runs classification, prints report, returns accuracy |

Returns `float` (accuracy %) if test data has labels, `None` otherwise.
