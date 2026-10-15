# SoundSort — Upgraded Music Analysis System

SoundSort now supports two workflows:

1. **Dataset Analysis** — upload a CSV containing `tempo`, `energy`, and `danceability`; SoundSort standardizes features, selects K with silhouette score, performs K-Means clustering, assigns interpretable rule-based categories, plots results, and exports a labelled CSV.
2. **Individual Song Analysis** — upload an MP3 or WAV; SoundSort uses Librosa to extract tempo, RMS energy, chroma, zero-crossing rate, spectral centroid/bandwidth/rolloff, and 13 MFCC coefficients.

> Current audio mode performs **feature extraction**, not trained genre prediction yet. A supervised classifier is the next upgrade.

## Setup (Windows / VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Project structure

```text
SoundSort-Upgraded/
├── app.py
├── model.py
├── audio_features.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── result.html
│   ├── audio_result.html
│   ├── 404.html
│   └── 500.html
├── static/
├── uploads/
├── processed/
└── sample_data/
    └── hindi_songs_soundsort.csv
```

## Test

- Dataset mode: upload `sample_data/hindi_songs_soundsort.csv`.
- Song mode: upload a local `.mp3` or `.wav` file under 25 MB.

## Important implementation notes

- CSV and audio extensions are validated separately.
- Temporary uploads are deleted after processing.
- Upload size is capped at 25 MB.
- Generated CSV/plot files are ignored by Git.
- `gunicorn` is included for future Linux/Render deployment.
