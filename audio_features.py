import librosa
import numpy as np


def extract_audio_features(file_path):
    # Load first 30 seconds of audio
    y, sr = librosa.load(file_path, duration=30)

    if len(y) == 0:
        raise ValueError("Could not read audio data.")

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr
    )

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr
    )

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr
    )

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr
    )

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    features = {
        "tempo": round(tempo, 2),
        "energy": round(energy, 4),
        "zero_crossing_rate": round(zcr_mean, 4),
        "spectral_centroid": round(float(np.mean(spectral_centroid)), 2),
        "spectral_bandwidth": round(float(np.mean(spectral_bandwidth)), 2),
        "spectral_rolloff": round(float(np.mean(spectral_rolloff)), 2),
        "chroma": round(float(np.mean(chroma)), 4),
    }

    # Store mean of every MFCC
    for i in range(13):
        features[f"mfcc_{i + 1}"] = round(
            float(np.mean(mfcc[i])),
            4
        )

    return features