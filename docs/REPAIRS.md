# Maintained entry points and validation

Run `python app.py` for the desktop file-selection interface, or:

```bash
python predict.py sample.wav --labels labels.json
```

`labels.json` must be a JSON array in the exact order used when training the selected checkpoint. Do not infer this order from result tables. The old prediction notebook suggests a seven-class mapping, but the checkpoint training provenance has not been independently established.

The maintained LSTM implementation derives hidden/output dimensions from the checkpoint, uses CPU loading, strict state loading, a `(1,40,1)` input, raw logits and explicit label-count checks. Feature extraction follows the old MFCC-only prediction notebook (40 coefficients, three seconds after a half-second offset). It does not pretend an LSTM checkpoint belongs to the DWT CNN-LSTM model.

The desktop interface runs inference off the UI thread and updates Tkinter only on the main thread. It requires no missing microphone artwork or PyAudio. Record a WAV with your preferred recorder, then select it. Microphone streaming remains an archived experiment until its matching CNN-LSTM checkpoint and training metadata are supplied.

Historical training notebooks remain research artifacts, not maintained entry points. Changing their softmax/loss pairing requires retraining; archived validation figures must not be attributed to a changed implementation.

Validation: Python syntax and notebook structure checked. The current workspace lacks PyTorch/librosa, the source audio datasets and a verified label-order file, so checkpoint inference and retraining have not been run.
