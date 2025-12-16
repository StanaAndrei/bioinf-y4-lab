"""
Exercise 9.2 — Disease Proximity and Drug Ranking

Scop:
- să calculați distanța medie dintre fiecare medicament și un set de gene asociate unei boli
- să ordonați medicamentele în funcție de proximitate (network-based prioritization)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, List, Tuple

import networkx as nx
import pandas as pd
import numpy as np
import pickle

# --------------------------
# Config
# --------------------------
# NOTĂ: Înlocuiește "<handle>" cu handle-ul tău real dacă rulezi local
HANDLE = "StanaAndrei"

# Input: graful bipartit (salvat anterior) SAU tabelul drug-gene
GRAPH_DRUG_GENE = Path(f"labs/09_repurposing/submissions/{HANDLE}/network_drug_gene_{HANDLE}.gpickle")
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Input: lista genelor bolii
DISEASE_GENES_TXT = Path(f"data/work/{HANDLE}/lab09/disease_genes_{HANDLE}.txt")

# Output directory & file
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_PRIORITY = OUT_DIR / f"drug_priority_{HANDLE}.csv"


# --------------------------
# Utils
# --------------------------
def ensure_exists(path: Path) -> None:
    """
    Verifică dacă fișierul există. Dacă nu, ridică FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Fișierul nu a fost găsit: {path}")
    print(f"[OK] Fișier găsit: {path}")


def load_bipartite_graph_or_build() -> nx.Graph:
    """
    Dacă GRAPH_DRUG_GENE există, îl încarcă.
    Altfel, reconstruiește graful plecând de la DRUG_GENE_CSV.
    """
    # 1. Încercăm să încărcăm pickle-ul
    if GRAPH_DRUG_GENE.exists():
        print(f"[INFO] Loading graph from {GRAPH_DRUG_GENE}...")
        try:
            # Notă: nx.read_gpickle este deprecated în NetworkX 3.0+
            # Folosim pickle standard dacă nx eșuează sau direct nx.read_gpickle pentru versiuni vechi
            with open(GRAPH_DRUG_GENE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[WARN] Could not load pickle ({e}). Rebuilding from CSV.")

    # 2. Reconstrucția din CSV (Fallback)
    ensure_exists(DRUG_GENE_CSV)
    print(f"[INFO] Building graph from {DRUG_GENE_CSV}...")
    
    df = pd.read_csv(DRUG_GENE_CSV)
    # Presupunem coloane: 'drug', 'gene' (sau similar, ajustați după CSV-ul real)
    # Dacă CSV-ul nu are header, folosiți names=['drug', 'gene']
    if 'drug' not in df.columns or 'gene' not in df.columns:
        # Fallback simplu dacă numele coloanelor diferă, luăm primele 2
        df.columns = ['drug', 'gene']

    B = nx.Graph()
    
    # Adăugăm nodurile cu atributul 'bipartite'
    drugs = df['drug'].unique()
    genes = df['gene'].unique()
    
    B.add_nodes_from(drugs, bipartite='drug')
    B.add_nodes_from(genes, bipartite='gene')
    
    # Adăugăm muchiile
    edges = list(zip(df['drug'], df['gene']))
    B.add_edges_from(edges)
    
    print(f"[INFO] Graph built: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges.")
    return B


def load_disease_genes(path: Path) -> Set[str]:
    """
    Încarcă fișierul text cu gene (una pe linie) și returnează un set.
    """
    ensure_exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        # Citim liniile, eliminăm whitespace și filtrăm liniile goale
        genes = {line.strip() for line in f if line.strip()}
    
    print(f"[INFO] Loaded {len(genes)} disease genes.")
    return genes


def get_drug_nodes(B: nx.Graph) -> List[str]:
    """
    Extrage lista nodurilor de tip 'drug'.
    """
    return [n for n, d in B.nodes(data=True) if d.get("bipartite") == "drug"]


def compute_drug_disease_distance(
    B: nx.Graph,
    drug: str,
    disease_genes: Set[str],
    mode: str = "mean",
    max_dist: int = 5,
) -> float:
    """
    Calculează distanța agregată (min sau mean) de la un medicament la setul de gene.
    """
    # Optimizare: Calculăm shortest paths de la 'drug' către toate nodurile accesibile
    # până la o distanță limită (cutoff) pentru a nu traversa tot graful inutil.
    # Dacă graful e mic, putem scoate cutoff-ul.
    
    try:
        lengths = nx.single_source_shortest_path_length(B, source=drug, cutoff=max_dist)
    except nx.NetworkXError:
        # Nodul nu există în graf (caz rar, dar posibil)
        return float('nan')

    distances = []
    
    # Intersectăm genele bolii cu genele găsite în graf
    # (Exercițiul cere să ignorăm genele care nu sunt în graf)
    valid_disease_genes = [g for g in disease_genes if g in B.nodes()]
    
    if not valid_disease_genes:
        return float('nan') # Sau o penalizare maximă

    for gene in valid_disease_genes:
        if gene in lengths:
            distances.append(lengths[gene])
        else:
            # Gena este în graf, dar nu e accesibilă (componentă diferită sau > max_dist)
            distances.append(max_dist + 1)
    
    if not distances:
        return float(max_dist + 1)

    if mode == "min":
        return float(np.min(distances))
    else:  # mode == "mean"
        return float(np.mean(distances))


def rank_drugs_by_proximity(
    B: nx.Graph,
    disease_genes: Set[str],
    mode: str = "mean",
) -> pd.DataFrame:
    """
    Calculează scorul de distanță pentru fiecare medicament și sortează.
    """
    drug_nodes = get_drug_nodes(B)
    results = []
    
    print(f"[INFO] Ranking {len(drug_nodes)} drugs using mode='{mode}'...")
    
    # Putem filtra disease_genes doar la cele prezente în graf pentru eficiență
    # dar funcția compute_drug_disease_distance gestionează deja asta.
    
    for drug in drug_nodes:
        dist = compute_drug_disease_distance(B, drug, disease_genes, mode=mode)
        results.append({
            "drug": drug,
            "distance": dist
        })
        
    df = pd.DataFrame(results)
    
    # Eliminăm valorile NaN (dacă există medicamente complet deconectate și vrem să le scoatem)
    # Sau le punem la coadă. Aici aleg să sortez, NaN ajunge la final.
    df = df.dropna()
    
    # Sortăm crescător (distanță mică = proximitate mare = mai bun)
    df = df.sort_values(by="distance", ascending=True)
    
    return df


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # TODO 1: Verificăm input-urile (deja incluse în funcțiile de load, dar verificăm path-urile principale)
    # Aici presupunem că DRUG_GENE_CSV este sursa dacă pickle nu există.
    if not GRAPH_DRUG_GENE.exists():
        try:
            ensure_exists(DRUG_GENE_CSV)
        except FileNotFoundError:
            print("[STOP] Nu există nici graful salvat, nici CSV-ul sursă.")
            exit(1)
            
    try:
        ensure_exists(DISEASE_GENES_TXT)
    except FileNotFoundError:
        print(f"[STOP] Lipsește fișierul cu genele bolii: {DISEASE_GENES_TXT}")
        exit(1)

    # TODO 2: Încărcați / construiți graful bipartit
    G = load_bipartite_graph_or_build()

    # TODO 3: Încărcați setul de disease genes
    d_genes = load_disease_genes(DISEASE_GENES_TXT)

    # TODO 4: Calculați ranking-ul medicamentelor după proximitate
    # Folosim 'mean' distance (distanța medie către genele bolii)
    df_ranking = rank_drugs_by_proximity(G, d_genes, mode="mean")

    # Afișăm top 5
    print("\n[RESULTS] Top 5 drugs closer to disease module:")
    print(df_ranking.head(5))

    # TODO 5: Salvați rezultatele
    df_ranking.to_csv(OUT_DRUG_PRIORITY, index=False)
    print(f"\n[INFO] Results saved to: {OUT_DRUG_PRIORITY}")