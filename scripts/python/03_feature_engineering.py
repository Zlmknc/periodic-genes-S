import pandas as pd
import numpy as np
from scipy.signal import lombscargle
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
EXPERIMENTS = ["alpha", "cdc15", "cdc28", "elu"]

# Maya hücre döngüsü periyodu tahmini aralığı (dakika) - literatürden yaklaşık
CYCLE_PERIOD_RANGE = (50, 120)


def safe_autocorr(values, times, lag_steps=1):
    """NaN'lari cikararak, sadece gercekten ardisik (index bazinda) ciftler uzerinden
    lag-k otokorelasyonu hesaplar. Zaman eşit araliklarla ornek­lenmedigi icin bu
    bir yaklastirmadir; sinirlilik olarak raporda belirtilecek."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n <= lag_steps + 1:
        return 0.0
    a = v[:-lag_steps]
    b = v[lag_steps:]
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def count_peaks(values):
    v = np.asarray(values, dtype=float)
    valid_idx = np.where(~np.isnan(v))[0]
    v_valid = v[valid_idx]
    if len(v_valid) < 3:
        return 0
    peaks = 0
    for i in range(1, len(v_valid) - 1):
        if v_valid[i] > v_valid[i - 1] and v_valid[i] > v_valid[i + 1]:
            peaks += 1
    return peaks


def lomb_scargle_features(values, times):
    """Eksik/duzensiz zaman noktalarini doğru ele alan Lomb-Scargle periodogrami.
    FFT'nin aksine esit araliksiz orneklemede gecerlidir - literatur taramasinda
    onerilen ama onceki calismada hic kullanilmayan yontem."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    mask = ~np.isnan(v)
    v, t = v[mask], t[mask]
    if len(v) < 4:
        return {"ls_dominant_period": 0.0, "ls_max_power": 0.0, "ls_band_power": 0.0}

    v = v - np.mean(v)
    # Aday periyotlar: 2x en kucuk ornekleme araligindan, toplam sure*1.5'e kadar
    span = t.max() - t.min()
    periods = np.linspace(max(10, span / 40), span * 1.5, 200)
    ang_freqs = 2 * np.pi / periods

    power = lombscargle(t, v, ang_freqs, normalize=True)
    dominant_idx = np.argmax(power)
    dominant_period = periods[dominant_idx]
    max_power = power[dominant_idx]

    band_mask = (periods >= CYCLE_PERIOD_RANGE[0]) & (periods <= CYCLE_PERIOD_RANGE[1])
    band_power = power[band_mask].sum() if band_mask.sum() > 0 else 0.0

    return {
        "ls_dominant_period": dominant_period,
        "ls_max_power": max_power,
        "ls_band_power": band_power,
    }

from scipy.optimize import curve_fit

def sinusoidal_fit_features(values, times):
    """Genlik, frekans, faz parametrelerini cikarir. Eski calismadaki
    fikri koruyoruz ama artik missing-value'lari dogru disliyoruz."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    mask = ~np.isnan(v)
    v, t = v[mask], t[mask]
    if len(v) < 5:
        return {"sin_amplitude": 0.0, "sin_freq": 0.0, "sin_phase": 0.0}

    def sinusoid(t, A, f, phi, offset):
        return A * np.sin(2 * np.pi * f * t + phi) + offset

    try:
        guess_freq = 1 / 80.0   # ~80 dk periyot baslangic tahmini
        popt, _ = curve_fit(sinusoid, t, v,
                             p0=[np.std(v), guess_freq, 0, np.mean(v)],
                             maxfev=2000)
        return {"sin_amplitude": abs(popt[0]), "sin_freq": abs(popt[1]), "sin_phase": popt[2] % (2*np.pi)}
    except Exception:
        return {"sin_amplitude": 0.0, "sin_freq": 0.0, "sin_phase": 0.0}

def extract_features_for_experiment(name):
    df = pd.read_csv(PROCESSED / f"{name}_aligned.csv").set_index("ORF")
    times = np.load(PROCESSED / f"{name}_times.npy")

    rows = []
    for orf, row in df.iterrows():
        values = row.values.astype(float)
        valid = values[~np.isnan(values)]

        feat = {"ORF": orf}
        feat["mean"] = np.mean(valid) if len(valid) else 0.0
        feat["std"] = np.std(valid) if len(valid) else 0.0
        feat["range"] = (np.max(valid) - np.min(valid)) if len(valid) else 0.0
        feat["missing_ratio"] = np.isnan(values).mean()
        feat["autocorr_lag1"] = safe_autocorr(values, times, 1)
        feat["autocorr_lag2"] = safe_autocorr(values, times, 2)
        feat["peak_count"] = count_peaks(values)
        feat.update(lomb_scargle_features(values, times))
        feat.update(sinusoidal_fit_features(values, times))

        rows.append(feat)

    out = pd.DataFrame(rows)
    # sutun isimlerini deney onekiyle isaretle (birlestirmede karismasin)
    out = out.rename(columns={c: f"{name}_{c}" for c in out.columns if c != "ORF"})
    return out


if __name__ == "__main__":
    for name in EXPERIMENTS:
        print(f"İşleniyor: {name} ...")
        feats = extract_features_for_experiment(name)
        feats.to_csv(PROCESSED / f"{name}_features.csv", index=False)
        print(f"  -> {feats.shape[0]} gen x {feats.shape[1]-1} özellik kaydedildi.")
        print(feats.describe().T[["mean", "std", "min", "max"]].round(3))