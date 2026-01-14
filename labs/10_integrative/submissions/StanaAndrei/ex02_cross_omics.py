"""
Exercise 10.2 — Identify top SNP–Gene correlations

TODO:
- încărcați matricea integrată multi-omics
- împărțiți rândurile în SNPs vs gene (după indice sau după nume)
- calculați corelații între fiecare SNP și fiecare genă
- filtrați |r| > 0.5
- exportați snp_gene_pairs_<handle>.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

HANDLE = "StanaAndrei"


JOINT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/multiomics_concat_{HANDLE}.csv")

OUT_CSV = Path(f"labs/10_integrative/submissions/{HANDLE}/snp_gene_pairs_{HANDLE}.csv")

# Asigurăm crearea directorului de output dacă nu există
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# --- REZOLVARE TODO ---

def main():
    print(f"1. Se încarcă matricea integrată din {JOINT_CSV} ...")
    if not JOINT_CSV.exists():
        print(f"Eroare: Fișierul {JOINT_CSV} nu există. Rulează întâi Exercițiul 10 (partea 1).")
        return

    # Încărcăm datele (index_col=0 pentru a folosi ID-urile probelor ca index)
    joint_df = pd.read_csv(JOINT_CSV, index_col=0)
    
    print(f"   Dimensiuni matrice încărcată: {joint_df.shape}")

    # 2. Separăm coloanele în SNP-uri și Gene
    # Presupunere: Numele SNP-urilor încep cu 'rs'. Genele au alte nume.
    all_cols = joint_df.columns
    snp_cols = [col for col in all_cols if col.startswith('rs')]
    gene_cols = [col for col in all_cols if not col.startswith('rs')]
    
    if len(snp_cols) == 0:
        print("Avertisment: Nu s-au găsit coloane care încep cu 'rs'. Verifică numele coloanelor!")
        # Fallback opțional: dacă primele N coloane sunt SNP-uri, poți folosi slicing, ex: joint_df.iloc[:, :100]
        return

    print(f"   Identificate: {len(snp_cols)} SNP-uri și {len(gene_cols)} Gene.")

    # 3. Calculăm matricea de corelație
    # Calculăm corelația pe tot DataFrame-ul, apoi extragem doar dreptunghiul de interes (SNP vs Gene)
    # Metoda Pearson este default
    print("3. Se calculează corelațiile (acest pas poate dura puțin)...")
    full_corr_matrix = joint_df.corr(method='pearson')
    
    # Extragem doar intersecția rândurilor (SNP) cu coloanele (Gene)
    # Astfel ignorăm corelațiile SNP-SNP sau Gene-Gene
    snp_gene_corr = full_corr_matrix.loc[snp_cols, gene_cols]

    # 4. Transformăm matricea în listă de perechi și filtrăm
    print("4. Se filtrează perechile cu |r| > 0.5 ...")
    
    # 'unstack' transformă matricea în serie lungă (Gene, SNP) -> valoare
    pairs = snp_gene_corr.unstack().reset_index()
    pairs.columns = ['Gene', 'SNP', 'Correlation']
    
    # Filtrare valoare absolută > 0.5
    strong_pairs = pairs[pairs['Correlation'].abs() > 0.5]
    
    # Sortăm descrescător după puterea corelației (modul)
    strong_pairs = strong_pairs.sort_values(by='Correlation', key=abs, ascending=False)

    # 5. Export
    print(f"5. Se exportă {len(strong_pairs)} perechi în {OUT_CSV} ...")
    strong_pairs.to_csv(OUT_CSV, index=False)
    
    print("Gata! Iată primele 5 rezultate:")
    print(strong_pairs.head())

if __name__ == "__main__":
    main()