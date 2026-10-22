"""Desktop file-based speech demo. No missing artwork or microphone dependency."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from predict import predict, ROOT


def main():
    root = tk.Tk()
    root.title('Speech Emotion Recognition')
    root.geometry('480x240')
    status = tk.StringVar(value='Choose a checkpoint, its label mapping and an audio recording.')
    tk.Label(root, textvariable=status, wraplength=440).pack(padx=20, pady=25)
    pool = ThreadPoolExecutor(max_workers=1)
    def start():
        checkpoint = filedialog.askopenfilename(title='Select LSTM checkpoint', initialdir=ROOT)
        if not checkpoint: return
        mapping = filedialog.askopenfilename(title='Select label-order JSON')
        if not mapping: return
        audio = filedialog.askopenfilename(title='Select audio recording')
        if not audio: return
        try:
            labels = json.loads(Path(mapping).read_text())
            if not isinstance(labels, list) or not all(isinstance(x, str) and x for x in labels):
                raise ValueError('Expected a JSON array of labels')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Invalid label mapping', str(exc)); return
        button.config(state='disabled')
        status.set('Analysing recording…')
        future = pool.submit(predict, audio, checkpoint, labels)
        def poll():
            if not future.done():
                root.after(100, poll); return
            try: status.set('Predicted emotion: ' + future.result())
            except Exception as exc: status.set('Prediction unavailable: ' + str(exc))
            button.config(state='normal')
        root.after(100, poll)
    button = tk.Button(root, text='Analyse audio', command=start)
    button.pack(pady=15)
    root.mainloop()
    pool.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
