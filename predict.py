"""MFCC inference for the uploaded LSTM checkpoint; CPU by default."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def predict(audio, checkpoint, labels):
    import numpy as np
    import librosa
    import torch
    from torch import nn
    class LSTMModel(nn.Module):
        def __init__(self, hidden, classes):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, 1, batch_first=True)
            self.fc1 = nn.Linear(hidden, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, classes)
        def forward(self, x):
            output, _ = self.lstm(x)
            output = torch.relu(output[:, -1, :])
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(output)))))
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
    if not Path(audio).is_file():
        raise FileNotFoundError(f'Missing audio file: {audio}')
    # Do not guess label order. Supply the mapping from the checkpoint training run.
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Supply unique labels in the exact training order')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or 'lstm.weight_hh_l0' not in state or 'fc3.weight' not in state:
        raise ValueError('Checkpoint is not the supported LSTM state dictionary')
    if state['fc3.weight'].shape[0] != len(labels):
        raise ValueError('Checkpoint output count does not match supplied labels')
    hidden = state['lstm.weight_hh_l0'].shape[1]
    model = LSTMModel(hidden, len(labels))
    model.load_state_dict(state, strict=True)
    model.eval()
    y, sr = librosa.load(audio, duration=3, offset=0.5)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Audio is empty after the 0.5-second offset or contains invalid samples')
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    tensor = torch.tensor(features, dtype=torch.float32).reshape(1, 40, 1)
    with torch.inference_mode():
        logits = model(tensor)
    return labels[int(logits.argmax(dim=1).item())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'LSTM_model_pytorch.pth')
    parser.add_argument('--labels', type=Path, required=True, help='JSON array in checkpoint training order')
    args = parser.parse_args()
    try:
        labels = json.loads(args.labels.read_text())
        if not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels):
            raise ValueError('Labels must be a JSON array of nonempty strings')
        print(predict(args.audio, args.checkpoint, labels))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f'Prediction failed: {exc}\n')
