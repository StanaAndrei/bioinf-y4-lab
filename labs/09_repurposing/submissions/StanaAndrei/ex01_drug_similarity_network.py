"""
Exercise 9.1 — Drug–Gene Bipartite Network & Drug Similarity Network

Scop:
- să construiți o rețea bipartită drug–gene plecând de la un CSV
- să proiectați layer-ul de medicamente folosind similaritatea dintre seturile de gene
- să exportați un fișier cu muchiile de similaritate între medicamente
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, List

import itertools
import networkx as nx
import pandas as pd
import pickle # Necesar pentru salvarea grafului in mod compatibil

# --------------------------
# Config — adaptați pentru handle-ul vostru
# --------------------------
HANDLE = "StanaAndrei"  # TODO: Schimbă aici cu handle-ul tău real (ex: 'popescu')

# Input: fișier cu coloane cel puțin: drug, gene
# Asigură-te că acest path este corect relativ la locul de unde rulezi scriptul
DRUG_GENE_CSV = Path(f"data/work/{HANDLE}/lab09/drug_gene_{HANDLE}.csv")

# Output directory & files
OUT_DIR = Path(f"labs/09_repurposing/submissions/{HANDLE}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DRUG_SUMMARY = OUT_DIR / f"drug_summary_{HANDLE}.csv"
OUT_DRUG_SIMILARITY = OUT_DIR / f"drug_similarity_{HANDLE}.csv"
OUT_GRAPH_DRUG_GENE = OUT_DIR / f"network_drug_gene_{HANDLE}.gpickle"


def ensure_exists(path: Path) -> None:
    """
    Verifică dacă fișierul există. Dacă nu, ridică FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(f"[EROARE] Fișierul de input nu a fost găsit la calea: {path}")


def load_drug_gene_table(path: Path) -> pd.DataFrame:
    """
    Citește CSV-ul cu pandas și validează coloanele.
    """
    df = pd.read_csv(path)
    required_cols = {'drug', 'gene'}
    
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[EROARE] Tabelul trebuie să conțină coloanele {required_cols}. "
            f"Coloane găsite: {list(df.columns)}"
        )
    
    # Curățăm eventualele spații sau valori nule, opțional dar recomandat
    df = df.dropna(subset=['drug', 'gene'])
    return df


def build_drug2genes(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Construiește un dict: drug -> set de gene țintă.
    """
    # Grupăm după 'drug', luăm coloana 'gene', aplicăm set și convertim la dict
    drug2genes = df.groupby("drug")["gene"].apply(set).to_dict()
    return drug2genes


def build_bipartite_graph(drug2genes: Dict[str, Set[str]]) -> nx.Graph:
    """
    Construiește graful bipartit (drug-gene).
    """
    G = nx.Graph()
    
    # 1. Adăugăm nodurile de tip 'drug'
    drug_nodes = list(drug2genes.keys())
    G.add_nodes_from(drug_nodes, bipartite="drug")
    
    # 2. Identificăm și adăugăm nodurile de tip 'gene'
    all_genes = set().union(*drug2genes.values())
    G.add_nodes_from(list(all_genes), bipartite="gene")
    
    # 3. Adăugăm muchiile
    edges = []
    for drug, genes in drug2genes.items():
        for gene in genes:
            edges.append((drug, gene))
            
    G.add_edges_from(edges)
    return G


def summarize_drugs(drug2genes: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Construiește un DataFrame cu statistici per medicament.
    """
    data = []
    for drug, genes in drug2genes.items():
        data.append({
            "drug": drug,
            "num_targets": len(genes)
        })
    
    df_summary = pd.DataFrame(data)
    # Putem sorta descrescător după numărul de gene
    df_summary = df_summary.sort_values(by="num_targets", ascending=False)
    return df_summary


def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    """
    Calculați similaritatea Jaccard între două seturi de gene:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def compute_drug_similarity_edges(
    drug2genes: Dict[str, Set[str]],
    min_sim: float = 0.0,
) -> List[Tuple[str, str, float]]:
    """
    Calculează similaritatea Jaccard pentru toate perechile de medicamente.
    """
    edges = []
    drugs = list(drug2genes.keys())
    
    # itertools.combinations generează perechi unice (d1, d2) fără repetiție
    for d1, d2 in itertools.combinations(drugs, 2):
        s1 = drug2genes[d1]
        s2 = drug2genes[d2]
        
        sim = jaccard_similarity(s1, s2)
        
        if sim >= min_sim:
            edges.append((d1, d2, sim))
            
    return edges


def edges_to_dataframe(edges: List[Tuple[str, str, float]]) -> pd.DataFrame:
    """
    Transformă lista de muchii într-un DataFrame.
    """
    df = pd.DataFrame(edges, columns=["drug1", "drug2", "weight"])
    return df


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print(f"[INFO] Rulez pentru handle: {HANDLE}")
    
    # TODO 1: verificați că fișierul de input există
    try:
        ensure_exists(DRUG_GENE_CSV)
        print(f"[OK] Fișier găsit: {DRUG_GENE_CSV}")
    except FileNotFoundError as e:
        print(e)
        print("-> Asigură-te că ai modificat variabila HANDLE și calea fișierului este corectă.")
        exit(1)

    # TODO 2: încărcați tabelul drug-gene
    df_dg = load_drug_gene_table(DRUG_GENE_CSV)
    print(f"[OK] Tabel încărcat. Dimensiuni: {df_dg.shape}")

    # TODO 3: construiți mapping-ul drug -> set de gene
    d2g = build_drug2genes(df_dg)
    print(f"[OK] Mapping construit pentru {len(d2g)} medicamente.")

    # TODO 4: construiți graful bipartit și salvați-l (opțional)
    B = build_bipartite_graph(d2g)
    print(f"[OK] Graf bipartit construit: {B.number_of_nodes()} noduri, {B.number_of_edges()} muchii.")
    
    # Salvare graf (folosind pickle standard pentru compatibilitate maximă)
    try:
        with open(OUT_GRAPH_DRUG_GENE, 'wb') as f:
            pickle.dump(B, f)
        print(f"[SAVE] Graful a fost salvat în: {OUT_GRAPH_DRUG_GENE}")
    except Exception as e:
        print(f"[WARN] Nu s-a putut salva graful: {e}")

    # TODO 5: generați și salvați sumarul pe medicamente
    df_summary = summarize_drugs(d2g)
    df_summary.to_csv(OUT_DRUG_SUMMARY, index=False)
    print(f"[SAVE] Sumarul medicamentelor salvat în: {OUT_DRUG_SUMMARY}")

    # TODO 6: calculați similaritatea între medicamente
    # Setăm un prag minim mic pentru a nu salva chiar toate perechile cu 0
    similarity_edges = compute_drug_similarity_edges(d2g, min_sim=0.01)
    
    df_sim = edges_to_dataframe(similarity_edges)
    df_sim.to_csv(OUT_DRUG_SIMILARITY, index=False)
    
    print(f"[SAVE] Similaritățile (edge list) salvate în: {OUT_DRUG_SIMILARITY}")
    print(f"[INFO] Proces complet. Au fost găsite {len(df_sim)} muchii de similaritate.")