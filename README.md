# Speech Emotion Recognition

Research experiments using MFCC and discrete wavelet transform (DWT) features with LSTM, CNN-LSTM and ResNet-based networks, plus microphone prediction prototypes.

## Recorded validation results

These are saved experimental results, not independently reproduced benchmarks. Each row is a separate dataset experiment; no cross-dataset generalisation claim is implied.

| Dataset | CNN-LSTM validation accuracy | MFCC coefficients | DWT level | Epochs |
|---|---:|---:|---:|---:|
| TESS | 99.64% | 40 | 1 | 500 |
| CREMA-D | 48.96% | 40 | 2 | 500 |
| SAVEE | 75.00% | 40 | 1 | 500 |
| RAVDESS | 59.03% | 40 | 1 | 500 |

Source: [saved validation table](results/metrics/Validation_Highest_Accuracy_Metrics_CNN_LSTM.csv). Values describe validation, not a separate held-out test evaluation.

## Explore the project

- Training: [CNN-LSTM](cnn%20lstm.ipynb), [PyTorch training](pytorch%20train.ipynb), [ResNet-LSTM](Resnet%20LSTM.ipynb).
- Features: [MFCC experiments](Experimenting%20MFCC.ipynb).
- Demonstrations: [Tkinter app](app.ipynb), [real-time experiment](Real_Time.ipynb), [PyTorch prediction](pytorch%20predict.ipynb).
- Evidence: [metrics](results/metrics/), [figures](results/figures/), [report](reports/CSH2.pdf).
- [Repository inventory and duplicate analysis](docs/INVENTORY.md).
- [Reproduction notes and known issues](docs/REPRODUCIBILITY.md).

## Environment and data

Create a separate Python environment, install the candidate dependencies in requirements.txt, then launch Jupyter from the repository root. Dependency versions were not supplied with the upload; this is not a tested environment lock.

Obtain TESS, CREMA-D, RAVDESS and SAVEE from their original providers and follow their usage conditions. Full audio datasets are not included. Set dataset and checkpoint paths in the chosen notebook before running it.

The included LSTM checkpoint is not automatically interchangeable with the CNN-LSTM checkpoint referenced by the GUI. The demo also references an absent microphone image.

Original notebook filenames and checkpoint locations are preserved to avoid silently changing experiment behaviour. Metrics, figures and the report are grouped by purpose; see the inventory for old-to-new paths.
