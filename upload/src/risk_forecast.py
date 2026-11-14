"""Stage 2 - Disease RISK forecasting (Random Forest baseline).

IMPORTANT: this demo uses SYNTHETIC weather and a rule-based label (cool + humid
spells favour late blight). It shows the pipeline only. For real use, replace
make_data() with real weather (e.g. IMD / NASA POWER) and field-observed outbreaks.
An LSTM can replace the Random Forest once enough real time-series data exists.

Run:  python src/risk_forecast.py
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

WINDOW, HORIZON, LONG = 5, 5, 30   # last 5 days (+30-day context) -> risk in next 5 days


def make_data(n_days=1500, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n_days)
    temp = 15 + 8 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2.5, n_days)
    hum = np.clip(75 + 15 * np.sin(2 * np.pi * (t + 60) / 90) + rng.normal(0, 8, n_days), 30, 100)
    rain = np.clip(rng.gamma(1.2, 3, n_days) * (hum > 80), 0, None)
    wet = np.clip((hum - 70) / 30 * 12 + rain * 0.5 + rng.normal(0, 1.5, n_days), 0, 24)
    X_raw = np.stack([temp, hum, rain, wet], 1)
    favourable = (temp >= 10) & (temp <= 25) & (hum >= 90) & (wet >= 8)   # blight-favourable day
    return X_raw, favourable


def build_xy(X_raw, favourable):
    X, y = [], []
    for i in range(LONG, len(X_raw) - HORIZON):
        w = X_raw[i - WINDOW:i]
        feats = list(w.ravel())                               # raw last 5 days
        feats += list(w.mean(0))                              # 5-day average
        feats += list(X_raw[i - 15:i].mean(0))                # 15-day average
        feats += list(X_raw[i - LONG:i].mean(0))              # 30-day average
        feats += [favourable[i - WINDOW:i].sum(),             # favourable days in last 5
                  favourable[i - LONG:i].sum()]               # favourable days in last 30
        X.append(feats)
        y.append(int(favourable[i:i + HORIZON].sum() >= 2))   # >=2 favourable days ahead = high risk
    return np.array(X), np.array(y)


if __name__ == "__main__":
    X_raw, fav = make_data()
    X, y = build_xy(X_raw, fav)
    n = len(X)                                    # chronological 64 / 16 / 20 split (no shuffling)
    a, b = int(n * 0.64), int(n * 0.80)
    Xtr, Xva, Xte = X[:a], X[a:b], X[b:]
    ytr, yva, yte = y[:a], y[a:b], y[b:]
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                min_samples_leaf=3, random_state=0).fit(Xtr, ytr)

    # choose the decision threshold on VALIDATION data (never on test) to maximise high-risk F1
    pv = rf.predict_proba(Xva)[:, 1]
    ths = np.linspace(0.05, 0.70, 27)
    th = max(ths, key=lambda t: f1_score(yva, pv >= t, zero_division=0))

    pred = (rf.predict_proba(Xte)[:, 1] >= th).astype(int)
    print(f"high-risk share: train={ytr.mean():.2f} val={yva.mean():.2f} test={yte.mean():.2f}")
    print(f"chosen threshold (from validation) = {th:.2f}")
    print(classification_report(yte, pred, target_names=["low risk", "high risk"], zero_division=0))
