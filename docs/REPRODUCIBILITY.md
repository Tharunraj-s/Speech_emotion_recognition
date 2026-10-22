# Reproduction review

This review inspected selected training/demo source and the saved CNN-LSTM validation CSV. Training and microphone inference have not been executed.

## Before running
1. Select one dataset, architecture and corresponding checkpoint.
2. Replace machine-specific Windows dataset/checkpoint paths.
3. Restore the exact label order used in training.
4. Match MFCC count, wavelet, decomposition level and preprocessing.
5. Supply the missing `mic button.png` used by app.ipynb.
6. Run Jupyter from the repository root. Metric-writing cells may still target old root filenames; update those outputs to results/metrics when rerunning.

## Observed issues
- Real_Time.ipynb configures six output classes but one live-prediction label list contains seven labels. Confirm the dataset mapping before interpreting predictions.
- The GUI loads an external CNN-LSTM checkpoint from a Windows path. The uploaded LSTM_model_pytorch.pth has not been proven compatible.
- Real_Time.ipynb uses pyaudio and wave without importing them in the inspected source.
- The inspected CNN-LSTM forward pass applies softmax. Check the training loss pairing; CrossEntropyLoss expects logits.
- Validation maxima selected across experiments are not independent test-set performance.
- Audit speaker independence and feature-fitting boundaries before making generalisation claims.

## Preserved variants
Case-only notebook names contain different Git blobs, so they were retained. Exact duplicate files are recorded in INVENTORY.md rather than deleted.

No model deserialisation, dataset download, retraining or metric recalculation was performed.
