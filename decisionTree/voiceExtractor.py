import librosa
import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, sr=44100)
    
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitches = pitches[pitches > 0]

    meanfreq = np.mean(pitches) if len(pitches) > 0 else 0
    sd = np.std(pitches) if len(pitches) > 0 else 0
    median = np.median(pitches) if len(pitches) > 0 else 0
    Q25 = np.percentile(pitches, 25) if len(pitches) > 0 else 0
    Q75 = np.percentile(pitches, 75) if len(pitches) > 0 else 0
    IQR = Q75 - Q25
    skewness = skew(pitches) if len(pitches) > 0 else 0
    kurt = kurtosis(pitches) if len(pitches) > 0 else 0
    
    ps = S**2
    ps_norm = ps / ps.sum(axis=0, keepdims=True)
    spectral_entropy = -np.sum(ps_norm * np.log2(ps_norm + 1e-10), axis=0).mean()
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    
    mode_index = np.argmax(S.mean(axis=1))
    mode_freq = freqs[mode_index]
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    
    meanfun = meanfreq
    minfun = np.min(pitches) if len(pitches) > 0 else 0
    maxfun = np.max(pitches) if len(pitches) > 0 else 0
    
    dom_freqs = freqs[np.argmax(S, axis=0)]
    meandom = np.mean(dom_freqs)
    mindom = np.min(dom_freqs)
    maxdom = np.max(dom_freqs)
    
    dfrange = maxdom - mindom
    
    modindx = (maxfun - minfun) / meanfun if meanfun != 0 else 0
    
    features = np.array([
        meanfreq, sd, median, Q25, Q75, IQR, skewness, kurt,
        spectral_entropy, spectral_flatness, mode_freq, centroid,
        meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx
    ])
    
    return features.reshape(1, -1)
