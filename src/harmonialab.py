
import os, io, base64, typer
import numpy as np
import soundfile as sf
import librosa
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from jinja2 import Template
import matplotlib.pyplot as plt

app = typer.Typer(add_completion=False)

MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def key_estimate(chroma):
    scores = []
    for shift in range(12):
        maj = np.dot(chroma, np.roll(MAJOR, shift))
        minr = np.dot(chroma, np.roll(MINOR, shift))
        if maj >= minr:
            scores.append(("{} major".format(shift), maj))
        else:
            scores.append(("{} minor".format(shift), minr))
    best = max(scores, key=lambda x: x[1])[0]
    return best

CHORDS = {
    # triads major/minor
}
# build chord templates
def build_chords():
    chords = {}
    names = []
    for root in range(12):
        # major triad
        tpl = np.zeros(12); tpl[root]=1; tpl[(root+4)%12]=1; tpl[(root+7)%12]=1
        chords[f"{root}:maj"]=tpl
        # minor triad
        tpl2 = np.zeros(12); tpl2[root]=1; tpl2[(root+3)%12]=1; tpl2[(root+7)%12]=1
        chords[f"{root}:min"]=tpl2
    return chords
CHORDS = build_chords()

def chord_from_chroma(ch):
    best = None; best_score = -1
    for name, tpl in CHORDS.items():
        s = float(np.dot(ch, tpl) / (np.linalg.norm(ch)+1e-9))
        if s > best_score:
            best_score = s; best = name
    return best, best_score

def novelty_from_chroma(C):
    # simple frame-wise difference
    D = np.diff(C, axis=1)
    n = np.maximum(0.0, np.sum(np.abs(D), axis=0))
    n = (n - n.min()) / (n.max() - n.min() + 1e-9)
    return n

def to_png(y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y)
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")

def save_midi_chords(path, chords, tempo_bpm=120, ppq=480):
    mid = MidiFile(ticks_per_beat=ppq)
    track = MidiTrack(); mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=int(60_000_000/tempo_bpm), time=0))
    for name, dur_beats in chords:
        if name is None: 
            # rest
            track.append(Message('note_off', note=60, velocity=0, time=int(ppq*dur_beats)))
            continue
        root, qual = name.split(":")
        root = int(root)
        if qual=="maj":
            notes = [root, (root+4)%12, (root+7)%12]
        else:
            notes = [root, (root+3)%12, (root+7)%12]
        # map to MIDI around C4
        base = 60 + root
        midi_notes = [60 + (n - 0) for n in notes]  # simplistic map
        for n in midi_notes:
            track.append(Message('note_on', note=60+(n%12), velocity=80, time=0))
        track.append(Message('note_off', note=60+(midi_notes[0]%12), velocity=0, time=int(ppq*dur_beats)))
        for n in midi_notes[1:]:
            track.append(Message('note_off', note=60+(n%12), velocity=0, time=0))
    mid.save(path)

@app.command()
def analyze(audio: str, report: bool = typer.Option(False), midi: str = typer.Option(None)):
    y, sr = librosa.load(audio, sr=None, mono=True)
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(C, axis=1)
    key = key_estimate(chroma_mean)
    # per-beat chords
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    hop = 512
    chords_seq = []
    last_b = 0
    if len(beats) < 2:
        beats = np.arange(0, C.shape[1], sr//hop//2)
    for i in range(len(beats)-1):
        a = int(beats[i]); b = int(beats[i+1])
        seg = C[:, a:b]
        ch, score = chord_from_chroma(np.mean(seg, axis=1))
        chords_seq.append((ch, 1.0))  # 1 beat
    # novelty and sections (simple peaks)
    nov = novelty_from_chroma(C)
    # report
    print(f"Key: {key}")
    print(f"Acordes (amostra): {chords_seq[:8]} ... total {len(chords_seq)}")
    if midi:
        save_midi_chords(midi, chords_seq, tempo_bpm=tempo if tempo>0 else 120)
        print(f"MIDI: {midi}")
    if report:
        os.makedirs("reports", exist_ok=True)
        png = to_png(nov)
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>HarmoniaLab</title></head>
<body><h1>Relatório HarmoniaLab</h1>
<p>Arquivo: {os.path.basename(audio)}</p>
<p>Key estimada: {key}</p>
<p>Acordes estimados (parcial): {", ".join([c for c,_ in chords_seq[:16]])}</p>
<img src="data:image/png;base64,{png}">
</body></html>"""
        out = os.path.join("reports", os.path.splitext(os.path.basename(audio))[0] + "_harmonia.html")
        with open(out,"w",encoding="utf-8") as f: f.write(html)
        print(f"Relatório: {out}")

if __name__ == "__main__":
    app()
