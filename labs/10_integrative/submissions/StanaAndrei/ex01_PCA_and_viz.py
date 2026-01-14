"""
Exercise 10 — PCA Single-Omics vs Joint

TODO:
- încărcați SNP și Expression
- normalizați fiecare strat (z-score)
- rulați PCA pe:
    1) strat SNP
    2) strat Expression
    3) strat Joint (concat)
- generați 3 figuri PNG
- comparați vizual distribuția probelor
"""

from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

HANDLE = "StanaAndrei"

SNP_CSV = Path(f"data/work/{HANDLE}/lab10/snp_matrix_{HANDLE}.csv")
EXP_CSV = Path(f"data/work/{HANDLE}/lab10/expression_matrix_{HANDLE}.csv")

OUT_DIR = Path(f"labs/10_integrative/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- REZOLVARE TODO ---

def run_and_plot_pca(df, title, filename):
    """
    Funcție auxiliară pentru a rula PCA și a salva graficul.
    """
    # Rulăm PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df)
    
    # Extragem varianța explicată pentru context
    var_explicata = pca.explained_variance_ratio_
    total_var = var_explicata.sum() * 100
    
    # Generăm plot-ul
    plt.figure(figsize=(8, 6))
    plt.scatter(pcs[:, 0], pcs[:, 1], alpha=0.7, edgecolors='k')
    plt.title(f"{title}\nTotal Explained Variance: {total_var:.2f}%")
    plt.xlabel(f"PC1 ({var_explicata[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({var_explicata[1]*100:.2f}%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Salvăm
    out_path = OUT_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"Generat: {out_path}")

def main():
    # 1. Încărcare date
    # Presupunem că prima coloană conține ID-urile probelor (index_col=0)
    print("Se încarcă datele...")
    try:
        snp_df = pd.read_csv(SNP_CSV, index_col=0)
        exp_df = pd.read_csv(EXP_CSV, index_col=0)
    except FileNotFoundError as e:
        print(f"Eroare: Fișierul nu a fost găsit. Verifică calea: {e}")
        return

    # 2. Aliniere probe (Intersectăm indecșii pentru a avea aceleași probe)
    common_samples = snp_df.index.intersection(exp_df.index)
    snp_df = snp_df.loc[common_samples]
    exp_df = exp_df.loc[common_samples]
    
    print(f"Probe comune identificate: {len(common_samples)}")

    # 3. Normalizare (Z-score)
    # Este crucial să normalizăm înainte de concatenare pentru ca un tip de date 
    # (ex: expresie genică cu valori mari) să nu domine PCA-ul joint.
    scaler = StandardScaler()
    
    snp_norm = pd.DataFrame(scaler.fit_transform(snp_df), 
                            index=snp_df.index, columns=snp_df.columns)
    
    exp_norm = pd.DataFrame(scaler.fit_transform(exp_df), 
                            index=exp_df.index, columns=exp_df.columns)

    # 4. Creare Strat Joint (Concatenare)
    joint_norm = pd.concat([snp_norm, exp_norm], axis=1)

    # 5. Rulare PCA și Export Figuri
    print("Se rulează PCA...")
    
    # a) Strat SNP
    run_and_plot_pca(snp_norm, "PCA - SNP Layer", "pca_snp.png")
    
    # b) Strat Expression
    run_and_plot_pca(exp_norm, "PCA - Expression Layer", "pca_expression.png")
    
    # c) Strat Joint
    run_and_plot_pca(joint_norm, "PCA - Joint Layer (Early Integration)", "pca_joint.png")

    print("Gata! Verifică folderul de output.")

if __name__ == "__main__":
    main()