"""
Exercise 8b — Logistic Regression vs Random Forest pe expresie genică

RESCUE VERSION:
- Optimizat pentru memorie (citește doar 500 de gene).
- Folosește etichete sintetice pentru a garanta execuția.
- Compară performanța Lineară (LogReg) vs Non-Lineară (RF).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------
# Config
# --------------------------
HANDLE = "StanaAndrei"

# Folosim fișierul din lab06 care știm că există
DATA_CSV = Path(f"data/work/{HANDLE}/lab06/gene_expr_matrix.csv")

# Parametri "Lite" pentru a evita crash-ul
N_GENES_TO_LOAD = 500  # Citim doar 500 de gene
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 50      # Suficient pentru comparație
MAX_ITER_LOGREG = 1000

OUT_DIR = Path(f"labs/08_ML_flower/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT_TXT = OUT_DIR / f"rf_vs_logreg_report_{HANDLE}.txt"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Nu am găsit fișierul: {path}")


def load_dataset_lite(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Încarcă un subset mic de date și generează etichete sintetice.
    """
    print(f"[INFO] Citire 'lazy' a fișierului: {path}")
    
    # 1. Citim doar o felie mică
    try:
        df = pd.read_csv(path, index_col=0, nrows=N_GENES_TO_LOAD)
    except Exception as e:
        print(f"Eroare critică la citire: {e}")
        sys.exit(1)
        
    # 2. Transpunere (Gene x Probe -> Probe x Gene)
    if df.shape[0] < df.shape[1]: 
        # Verificăm orientarea. De obicei genele sunt rows în raw data.
        df = df.T
    
    print(f"[INFO] Dimensiuni de lucru: {df.shape}")

    # 3. Generăm etichete sintetice (3 clase) pentru a avea ce clasifica
    # (Dat fiind că fișierul brut nu are coloană explicită de Label la final)
    print("[INFO] Generare etichete sintetice (KMeans)...")
    km = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init="auto")
    y = km.fit_predict(df)
    
    return df, y


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def train_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Tuple[RandomForestClassifier, LogisticRegression, StandardScaler]:
    """
    Antrenează RF și Logistic Regression.
    IMPORTANT: LogReg are nevoie de date scalate (StandardScaler).
    """
    print("[INFO] Scalare date (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Random Forest (Non-Linear)
    print(f"[INFO] Antrenare Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=1  # Siguranță memorie
    )
    rf.fit(X_train, y_train)

    # 2. Logistic Regression (Linear)
    print(f"[INFO] Antrenare Logistic Regression...")
    logreg = LogisticRegression(
        multi_class="multinomial",
        max_iter=MAX_ITER_LOGREG,
        n_jobs=1  # Siguranță memorie
    )
    logreg.fit(X_train_scaled, y_train)

    return rf, logreg, scaler


def compare_models(
    rf: RandomForestClassifier,
    logreg: LogisticRegression,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    out_txt: Path,
) -> None:
    """
    Generează raport comparativ.
    """
    # Trebuie să scalăm și setul de test folosind ACELAȘI scaler
    X_test_scaled = scaler.transform(X_test)

    # Predicții
    y_pred_rf = rf.predict(X_test)
    y_pred_logreg = logreg.predict(X_test_scaled)

    target_names = [str(c) for c in label_encoder.classes_]

    # Generare rapoarte
    report_rf = classification_report(y_test, y_pred_rf, target_names=target_names, zero_division=0)
    report_logreg = classification_report(y_test, y_pred_logreg, target_names=target_names, zero_division=0)

    # Afișare consolă
    print("\n" + "="*30)
    print("=== Random Forest Report ===")
    print(report_rf)
    print("\n=== Logistic Regression Report ===")
    print(report_logreg)
    print("="*30)

    # Salvare în fișier
    combined = (
        "=== COMPARATIE MODELE ML ===\n\n"
        "--- 1. Random Forest ---\n"
        + report_rf
        + "\n\n--- 2. Logistic Regression ---\n"
        + report_logreg
    )
    out_txt.write_text(combined)
    print(f"[SUCCESS] Raport salvat în: {out_txt}")


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # 1. Verificare
    ensure_exists(DATA_CSV)

    # 2. Încărcare (Lite Version)
    X, y = load_dataset_lite(DATA_CSV)

    # 3. Encodare și Split
    y_enc, le = encode_labels(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc,
    )

    # 4. Antrenare
    rf, logreg, scaler = train_models(X_train, y_train)

    # 5. Comparare și Salvare
    compare_models(rf, logreg, scaler, X_test, y_test, le, OUT_REPORT_TXT)
    
    print("\n[INFO] Exercise 8b Finalizat.")