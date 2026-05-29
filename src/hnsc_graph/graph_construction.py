import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Mapeo MIMAT → nombre miRNA via miRBase (mature.fa)
# ============================================================================

def build_mimat_name_maps(mature_fa_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Parsea el archivo mature.fa de miRBase y construye dos diccionarios:
      - mimat_to_name : MIMAT0000076 → hsa-miR-21-5p
      - name_to_mimat : hsa-miR-21-5p → MIMAT0000076

    Solo se incluyen entradas humanas (hsa-).

    Parameters
    ----------
    mature_fa_path : Path
        Ruta al archivo mature.fa descargado de miRBase.

    Returns
    -------
    mimat_to_name, name_to_mimat : Tuple[Dict, Dict]
    """
    mimat_to_name: Dict[str, str] = {}
    name_to_mimat: Dict[str, str] = {}

    if not mature_fa_path.exists():
        raise FileNotFoundError(f"mature.fa no encontrado: {mature_fa_path}")

    with open(mature_fa_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name  = parts[0][1:]   # quitar '>'
            mimat = parts[1]
            if name.startswith("hsa-") and mimat.startswith("MIMAT"):
                mimat_to_name[mimat] = name
                name_to_mimat[name]  = mimat

    logger.info("miRBase: %d entradas hsa cargadas", len(mimat_to_name))
    return mimat_to_name, name_to_mimat


# ============================================================================
# 2. Carga de TargetScan y construcción de interacciones isomiR → isoforma
# ============================================================================

def build_mirna_isoform_edges(
    targetscan_path: Path,
    name_to_mimat: Dict[str, str],
    isoform_to_gene: Dict[str, str],
    valid_mimats: List[str],
    valid_isoforms: List[str],
    context_score_threshold: float = 0.0,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Construye el DataFrame de aristas isomiR → isoforma usando TargetScan.

    Flujo:
    1. Lee TargetScan en chunks filtrando por miRNAs hsa- y genes en nuestro diccionario.
    2. Mapea nombre miRNA → MIMAT via name_to_mimat.
    3. Mapea gen diana → isoformas via isoform_to_gene (invertido).
    4. Filtra a isomiRs y isoformas presentes en nuestras matrices.
    5. Filtra por context++ score < threshold (más negativo = interacción más fuerte).

    Parameters
    ----------
    targetscan_path : Path
        Ruta al archivo TargetScan (Predicted_Targets_Context_Scores...).
    name_to_mimat : Dict[str, str]
        Mapeo nombre miRNA → MIMAT (de build_mimat_name_maps).
    isoform_to_gene : Dict[str, str]
        Mapeo isoforma_id → gene_symbol.
    valid_mimats : List[str]
        MIMAT presentes en nuestra matriz de isomiRs.
    valid_isoforms : List[str]
        IDs de isoformas presentes en nuestra matriz mRNA.
    context_score_threshold : float
        Solo interacciones con context++ score < threshold. Default 0.0.
    chunksize : int
        Tamaño de chunk para lectura de TargetScan.

    Returns
    -------
    pd.DataFrame con columnas [mirna_id, isoform_id, context_score]
    """
    if not targetscan_path.exists():
        raise FileNotFoundError(f"TargetScan no encontrado: {targetscan_path}")

    # Diccionario inverso: gen → lista de isoformas válidas
    gene_to_isoforms: Dict[str, List[str]] = {}
    for iso, gene in isoform_to_gene.items():
        if iso in valid_isoforms:
            gene_to_isoforms.setdefault(gene, []).append(iso)

    valid_mimat_set = set(valid_mimats)
    valid_gene_set  = set(gene_to_isoforms.keys())

    chunks = []
    for chunk in pd.read_csv(
        targetscan_path, sep="\t",
        usecols=["Gene Symbol", "miRNA", "context++ score"],
        chunksize=chunksize, low_memory=False,
    ):
        c = chunk[
            chunk["miRNA"].str.startswith("hsa-", na=False) &
            chunk["Gene Symbol"].isin(valid_gene_set)
        ].copy()
        if not c.empty:
            chunks.append(c)

    if not chunks:
        logger.warning("TargetScan: ninguna interacción encontrada para nuestros genes.")
        return pd.DataFrame(columns=["mirna_id", "isoform_id", "context_score"])

    df_ts = pd.concat(chunks, ignore_index=True)
    logger.info("TargetScan: %d interacciones hsa para nuestros genes", len(df_ts))

    # Mapear nombre → MIMAT
    df_ts["mirna_id"] = df_ts["miRNA"].map(name_to_mimat)
    df_ts = df_ts[
        df_ts["mirna_id"].notna() &
        df_ts["mirna_id"].isin(valid_mimat_set)
    ].copy()
    logger.info("Tras mapear MIMAT y filtrar a nuestros isomiRs: %d", len(df_ts))

    # Mapear gen → isoformas
    df_ts["isoform_id"] = df_ts["Gene Symbol"].map(gene_to_isoforms)
    df_ts = df_ts.dropna(subset=["isoform_id"])
    df_ts = df_ts.explode("isoform_id")
    df_ts = df_ts[df_ts["isoform_id"].isin(valid_isoforms)].copy()

    # Filtrar por context++ score
    df_ts["context_score"] = pd.to_numeric(df_ts["context++ score"], errors="coerce")
    df_ts = df_ts[df_ts["context_score"] < context_score_threshold].copy()

    # Resultado final — un par (mirna, isoforma) único
    result = (
        df_ts[["mirna_id", "isoform_id", "context_score"]]
        .drop_duplicates(subset=["mirna_id", "isoform_id"])
        .reset_index(drop=True)
    )
    logger.info(
        "Aristas isomiR→isoforma finales: %d | isomiRs: %d | isoformas: %d",
        len(result), result["mirna_id"].nunique(), result["isoform_id"].nunique()
    )
    return result


# ============================================================================
# 3. Construcción de aristas isomiR ↔ isomiR por correlación de expresión
# ============================================================================

def build_mirna_mirna_edges(
    mirna_expr: pd.DataFrame,
    r_threshold: float = 0.55,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Construye aristas entre pares de isomiRs con correlación de expresión
    |r| >= r_threshold calculada sobre todos los pacientes de entrenamiento.

    La correlación entre isomiRs captura co-regulación funcional:
    dos isomiRs que co-expresan probablemente regulan genes diana similares
    o participan en los mismos procesos oncológicos.

    IMPORTANTE: se calcula sobre la cohorte tumor completa (474 pacientes),
    no sobre subconjuntos. Las aristas son fijas para todos los pacientes.

    Parameters
    ----------
    mirna_expr : pd.DataFrame
        Matriz [muestras x isomiRs] en escala log2(RPM+1).
        Filas = pacientes, columnas = MIMATs.
    r_threshold : float
        Umbral de |r| para crear arista. Default 0.55.
    method : str
        "spearman" o "pearson". Default "spearman".

    Returns
    -------
    pd.DataFrame con columnas [mirna_src, mirna_dst, correlation]
    Aristas no dirigidas (src < dst para evitar duplicados en este DataFrame;
    el HeteroGraphBuilder las añade en ambas direcciones).
    """
    logger.info(
        "Calculando correlaciones %s entre %d isomiRs sobre %d pacientes...",
        method, mirna_expr.shape[1], mirna_expr.shape[0]
    )

    if method == "spearman":
        corr_matrix = mirna_expr.rank().corr(method="pearson")
    else:
        corr_matrix = mirna_expr.corr(method="pearson")

    isomirs   = corr_matrix.columns.tolist()
    corr_vals = corr_matrix.values
    n         = len(isomirs)

    src_list, dst_list, corr_list = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_vals[i, j]
            if not np.isnan(r) and abs(r) >= r_threshold:
                src_list.append(isomirs[i])
                dst_list.append(isomirs[j])
                corr_list.append(float(r))

    result = pd.DataFrame({
        "mirna_src"   : src_list,
        "mirna_dst"   : dst_list,
        "correlation" : corr_list,
    })

    logger.info(
        "Aristas isomiR↔isomiR: %d pares con |r|>=%.2f",
        len(result), r_threshold
    )
    return result


# ============================================================================
# 4. Constructor del HeteroData heterogéneo
# ============================================================================

@dataclass
class HeteroGraphConfig:
    mirna_node_type:         str                    = "mirna"
    isoform_node_type:       str                    = "isoform"
    mirna_mirna_edge_type:   Tuple[str, str, str]   = ("mirna", "coexpresses", "mirna")
    mirna_isoform_edge_type: Tuple[str, str, str]   = ("mirna", "represses", "isoform")


class HeteroGraphBuilder:
    """
    Construye un único objeto HeteroData de PyTorch Geometric con TODOS los
    pacientes (fallecidos + censurados) para entrenamiento con Cox Partial
    Likelihood Loss.

    Diseño del grafo:
    ─────────────────
    Nodos 'mirna'    : isomiRs MIMAT. x = [n_isomirs  x n_pacientes] log2(RPM+1)
    Nodos 'isoform'  : isoformas mRNA. x = [n_isoformas x n_pacientes] log2(UQ+1)

    Aristas 'coexpresses' (mirna↔mirna):
        Correlación Spearman |r| >= umbral entre isomiRs.
        No dirigidas → se añaden en ambas direcciones para paso de mensajes
        bidireccional en GATv2.
        edge_attr = correlación Spearman [n_aristas x 1]

    Aristas 'represses' (mirna→isoform):
        Interacciones biológicas reales de TargetScan (context++ score < 0).
        Dirigidas: isomiR → isoforma que reprime.
        edge_attr = context++ score [n_aristas x 1]

    Targets de supervivencia (Overall Survival únicamente):
        data.os_time  : [n_pacientes] — tiempo de supervivencia en días
        data.os_event : [n_pacientes] — 1=fallecido, 0=censurado
        data.barcodes : lista de barcodes TCGA en orden de pacientes

   

    Interpretabilidad:
        Las capas GATv2 se configuran con return_attention_weights=True en NB04.
        Los node_ids guardados en data['mirna'].node_ids y data['isoform'].node_ids
        permiten mapear índices de atención a nombres biológicos en NB04.
    """

    def __init__(self, config: HeteroGraphConfig) -> None:
        self.config = config

    @staticmethod
    def _build_index_map(ids: List[str]) -> Dict[str, int]:
        return {str(id_): i for i, id_ in enumerate(ids)}

    def build(
        self,
        mirna_expr: pd.DataFrame,
        isoform_expr: pd.DataFrame,
        mirna_mirna_edges: pd.DataFrame,
        mirna_isoform_edges: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> HeteroData:
        """
        Construye el HeteroData completo con todos los pacientes.

        Parameters
        ----------
        mirna_expr : pd.DataFrame
            [muestras x isomiRs] log2(RPM+1). Índice = barcodes 12 chars.
        isoform_expr : pd.DataFrame
            [muestras x isoformas] log2(UQ+1). Índice = barcodes 12 chars.
        mirna_mirna_edges : pd.DataFrame
            Columnas [mirna_src, mirna_dst, correlation].
        mirna_isoform_edges : pd.DataFrame
            Columnas [mirna_id, isoform_id, context_score].
        targets : pd.DataFrame
            Columnas mínimas: ['OS', 'OS.time']. Índice = barcodes 12 chars.
            Debe estar alineado con mirna_expr e isoform_expr.

        Returns
        -------
        HeteroData con:
            data['mirna'].x                              [n_isomirs x n_pacientes]
            data['mirna'].node_ids                       lista de MIMATs
            data['isoform'].x                            [n_isoformas x n_pacientes]
            data['isoform'].node_ids                     lista de isoforma IDs
            data['mirna','coexpresses','mirna'].edge_index  [2 x n_aristas_mm]
            data['mirna','coexpresses','mirna'].edge_attr   [n_aristas_mm x 1]
            data['mirna','represses','isoform'].edge_index  [2 x n_aristas_mi]
            data['mirna','represses','isoform'].edge_attr   [n_aristas_mi x 1]
            data.os_time    [n_pacientes] float32
            data.os_event   [n_pacientes] float32  (1=fallecido, 0=censurado)
            data.barcodes   lista de barcodes en orden de pacientes
        """
        # ── Alinear barcodes ──────────────────────────────────────────────
        common_barcodes = sorted(
            set(mirna_expr.index) & set(isoform_expr.index) & set(targets.index)
        )
        if len(common_barcodes) == 0:
            raise ValueError(
                "Intersección de barcodes vacía. Verifica que mirna_expr, "
                "isoform_expr y targets tienen el mismo formato de índice."
            )

        mirna_expr_   = mirna_expr.loc[common_barcodes]
        isoform_expr_ = isoform_expr.loc[common_barcodes]
        targets_      = targets.loc[common_barcodes]
        n_patients    = len(common_barcodes)

        logger.info("Pacientes en el grafo: %d", n_patients)
        logger.info(
            "  Fallecidos (OS=1): %d | Censurados (OS=0): %d",
            int((targets_["OS"] == 1).sum()),
            int((targets_["OS"] == 0).sum()),
        )

        # ── Índices de nodos ──────────────────────────────────────────────
        mirna_ids   = list(mirna_expr_.columns)
        isoform_ids = list(isoform_expr_.columns)
        mirna_idx   = self._build_index_map(mirna_ids)
        isoform_idx = self._build_index_map(isoform_ids)

        data = HeteroData()

        # ── Features de nodos [n_moleculas x n_pacientes] ─────────────────
        # Transpuesta: filas = moléculas, columnas = pacientes
        data[self.config.mirna_node_type].x = torch.tensor(
            mirna_expr_.T.to_numpy(), dtype=torch.float32
        )   # [n_isomirs x n_pacientes]

        data[self.config.isoform_node_type].x = torch.tensor(
            isoform_expr_.T.to_numpy(), dtype=torch.float32
        )   # [n_isoformas x n_pacientes]

        # IDs biológicos para interpretabilidad en NB04
        data[self.config.mirna_node_type].node_ids   = mirna_ids
        data[self.config.isoform_node_type].node_ids = isoform_ids
        data.barcodes = common_barcodes

        # ── Aristas isomiR ↔ isomiR (bidireccionales) ─────────────────────
        mm_src, mm_dst, mm_corr = [], [], []
        n_skip_mm = 0
        for _, row in mirna_mirna_edges.iterrows():
            si = mirna_idx.get(str(row["mirna_src"]))
            di = mirna_idx.get(str(row["mirna_dst"]))
            if si is None or di is None:
                n_skip_mm += 1
                continue
            # Bidireccional para paso de mensajes simétrico
            mm_src  += [si, di]
            mm_dst  += [di, si]
            mm_corr += [float(row["correlation"]), float(row["correlation"])]

        if not mm_src:
            logger.warning("No se crearon aristas isomiR↔isomiR.")
        else:
            data[self.config.mirna_mirna_edge_type].edge_index = torch.tensor(
                [mm_src, mm_dst], dtype=torch.long)
            data[self.config.mirna_mirna_edge_type].edge_attr = torch.tensor(
                mm_corr, dtype=torch.float32).unsqueeze(1)
            logger.info("Aristas isomiR↔isomiR: %d dirigidas (%d pares)",
                        len(mm_src), len(mm_src) // 2)

        # ── Aristas isomiR → isoforma (dirigidas) ─────────────────────────
        mi_src, mi_dst, mi_score = [], [], []
        n_skip_mi = 0
        for _, row in mirna_isoform_edges.iterrows():
            si = mirna_idx.get(str(row["mirna_id"]))
            di = isoform_idx.get(str(row["isoform_id"]))
            if si is None or di is None:
                n_skip_mi += 1
                continue
            mi_src.append(si)
            mi_dst.append(di)
            mi_score.append(float(row["context_score"]))

        if not mi_src:
            logger.warning("No se crearon aristas isomiR→isoforma.")
        else:
            data[self.config.mirna_isoform_edge_type].edge_index = torch.tensor(
                [mi_src, mi_dst], dtype=torch.long)
            data[self.config.mirna_isoform_edge_type].edge_attr = torch.tensor(
                mi_score, dtype=torch.float32).unsqueeze(1)
            logger.info("Aristas isomiR→isoforma: %d", len(mi_src))

        if n_skip_mm or n_skip_mi:
            logger.info(
                "Aristas descartadas por IDs no presentes — "
                "mirna-mirna: %d | mirna-isoforma: %d",
                n_skip_mm, n_skip_mi
            )

        # ── Targets de supervivencia — solo Overall Survival ──────────────
        # DFI descartado definitivamente del proyecto.
        os_time_vals = targets_["OS.time"].values.astype(float)
        if np.any(np.isnan(os_time_vals)):
            n_nan = int(np.isnan(os_time_vals).sum())
            logger.warning("%d pacientes con OS.time NaN — serán ignorados por Cox Loss", n_nan)

        data.os_time  = torch.tensor(os_time_vals,               dtype=torch.float32)
        data.os_event = torch.tensor(targets_["OS"].values,      dtype=torch.float32)

        logger.info(
            "Grafo construido: %d isomiRs | %d isoformas | %d pacientes | "
            "aristas mm=%d | aristas mi=%d",
            len(mirna_ids), len(isoform_ids), n_patients,
            len(mm_src) // 2 if mm_src else 0,
            len(mi_src),
        )
        return data


__all__ = [
    "build_mimat_name_maps",
    "build_mirna_isoform_edges",
    "build_mirna_mirna_edges",
    "HeteroGraphConfig",
    "HeteroGraphBuilder",
]
