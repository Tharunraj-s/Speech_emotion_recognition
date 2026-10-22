# Demonstration guide

## Walkthrough
1. Introduce emotion classification from speech.
2. Explain MFCC features and why wavelet decomposition was explored.
3. Show the report and compare LSTM/CNN-LSTM variants.
4. Show dataset-specific validation results and original graphs.
5. Explain why TESS and CREMA-D differ rather than treating the strongest result as universal accuracy.
6. Open app.py and predict.py to show the model-to-interface implementation.

## Local desktop demo
From the repository root, install requirements.txt in a separate environment:

```bash
python app.py
```

Select a compatible LSTM checkpoint, a JSON array of labels in its verified training order, and an audio recording. Tkinter must be installed locally.

```bash
python predict.py sample.wav --labels labels.json
```

The maintained UI uses prerecorded audio. Earlier microphone experiments remain preserved separately and require their own checkpoint/assets.

## Presenting before dependencies are restored
Use the report, original graphs, CSV evidence and a code walkthrough. Describe the interface as a prototype. Historical research outputs and the later maintained inference implementation are separate artifacts.

Historical notebook contents are unchanged. Launch Jupyter from the repository root and configure data/output paths before rerunning experiments.
