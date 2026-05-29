"""
baseline_models.py — Modelos de línea base para estudio de ablación (TFM HNSCC)

Baselines para comparar con SupraGraphNet:

1. Baseline_MLP      — Sin grafo. Solo características + pooling + clasificador.
2. Baseline_GCN      — Grafo isotrópico (GCNConv). Agregación sin pesos aprendidos.
3. Baseline_VanillaGAT — GAT básico (GATConv, 1 head) sin residuales ni LayerNorm.

Justificación de la selección en docstring de cada clase.

Interfaz idéntica a SupraGraphNet:
  - Entrada : HeteroData (mismo objeto del pipeline)
  - _patient_pooling : función estática heredada del modelo principal
  - Salida  : logits [n_patients] (sin sigmoid)


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


# ──────────────────────────────────────────────────────────────────────────────
# Configuración compartida para los tres baselines
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaselineConfig:
    """
    Configuración común para los modelos de ablación.
    Replica los parámetros esenciales de SupraGraphNetConfig para
    mantener comparabilidad directa de capacidad de representación.
    """
    num_mirna:      int
    num_mrna:       int
    num_patients:   int
    hidden_dim:     int   = 64
    os_head_dim:    int   = 32
    dropout:        float = 0.4
    expr_embed_dim: int   = 32
    use_hpv:        bool  = True
    hpv_embed_dim:  int   = 8


# ──────────────────────────────────────────────────────────────────────────────
# 1. Baseline_MLP  Sin grafo
# ──────────────────────────────────────────────────────────────────────────────

class Baseline_MLP(nn.Module):
    """
    Baseline sin estructura de grafo (ablación de las aristas).

    Justificación teórica
    ---------------------
    El Estado del Arte justifica el uso de GNNs en oncología precisamente porque
    los datos moleculares son intrínsecamente relacionales (§ GNNs en oncología).
    Este baseline cuantifica cuánto aporta la estructura del grafo: ignora
    completamente las aristas (co-expresión miRNA-miRNA y represión miRNA-isoforma)
    y procesa cada tipo de nodo con capas lineales independientes.

    El diseño replica la rama de embedding del modelo principal (misma proyección
    expr_embed → hidden) para que la diferencia de rendimiento se deba únicamente
    a la presencia o ausencia del message-passing, no a una diferencia en
    capacidad paramétrica.

    Corresponde al experimento de ablación más puro: «¿qué añade el grafo?»
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Embeddings de expresión (igual que SupraGraphNet)
        self.mirna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )
        self.mrna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )

        # Proyección a espacio oculto — sin message-passing
        self.mirna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )
        self.mrna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )

        # HPV (late fusion, igual que SupraGraphNet)
        if config.use_hpv:
            self.hpv_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, config.hpv_embed_dim),
            )
            hpv_dim = config.hpv_embed_dim
        else:
            hpv_dim = 0

        # La dimensión del vector de paciente es hidden_dim (mirna) + hidden_dim (isoforma)
        patient_emb_dim = config.hidden_dim + config.hidden_dim + hpv_dim

        self.os_head = nn.Sequential(
            nn.Linear(patient_emb_dim, config.os_head_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.os_head_dim, 1),
        )

    @staticmethod
    def _patient_pooling(expr_raw: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        """Idéntico a SupraGraphNet: pooling ponderado por expresión."""
        weights = F.softmax(expr_raw.T.clamp(min=0) * 5, dim=1)
        return torch.matmul(weights, node_emb)

    def forward(self, data: HeteroData, hpv: Optional[torch.Tensor] = None):
        expr_mirna    = data["mirna"].x       # [n_mirna,    n_patients]
        expr_isoforma = data["isoform"].x     # [n_isoform,  n_patients]

        # Embedding de expresión
        e_mirna    = self.mirna_expr_embed(expr_mirna)
        e_isoforma = self.mrna_expr_embed(expr_isoforma)

        # Proyección MLP pura (sin message-passing)
        h_mirna   = self.mirna_proj(e_mirna)      # [n_mirna,   hidden_dim]
        h_isoforma = self.mrna_proj(e_isoforma)    # [n_isoform, hidden_dim]

        # Pooling a nivel de paciente (igual que SupraGraphNet)
        g_mirna   = self._patient_pooling(expr_mirna,    h_mirna)
        g_isoforma = self._patient_pooling(expr_isoforma, h_isoforma)
        g_patient  = torch.cat([g_mirna, g_isoforma], dim=-1)

        # Late fusion HPV
        if self.config.use_hpv:
            if hpv is None and hasattr(data, "hpv"):
                hpv = data.hpv.float()
            if hpv is not None:
                hpv_emb   = self.hpv_proj(hpv.unsqueeze(-1))
                g_patient = torch.cat([g_patient, hpv_emb], dim=-1)

        return self.os_head(g_patient).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Baseline_GCN  Grafo isotrópico
# ──────────────────────────────────────────────────────────────────────────────

class Baseline_GCN(nn.Module):
    """
    Baseline con agregación isotrópica mediante GCNConv (Kipf & Welling, 2017).

    Justificación teórica
    ---------------------
    GCN es la arquitectura de referencia dominante en la literatura multi-ómica
    revisada (§ Arquitecturas principales, § Frameworks representativos):
    MOGONET, MoGCN, SUPREME y MVGNN usan GCNConv como bloque central.
    Según el Estado del Arte, su limitación fundamental es la agregación
    isotrópica: «cada vecino contribuye con el mismo peso, escalado por el grado
    del nodo» (Tabla comparativa de arquitecturas), sin distinguir la importancia
    relativa de distintas conexiones moleculares.

    Este baseline cuantifica el coste de renunciar a la atención diferencial
    del modelo principal (GATv2 multi-head) en favor de la simplicidad de GCN.

    Para operar sobre el grafo heterogéneo, se aplica GCNConv en dos pasos:
      1. Capa homo miRNA→miRNA sobre las aristas de co-expresión.
      2. Capa homo isoforma actualizada con los miRNA ya propagados
         (proyectando miRNA al espacio de isoforma antes de la convolución
         sobre las aristas de represión, tratadas como no dirigidas).

    
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Embeddings de expresión (igual que SupraGraphNet)
        self.mirna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )
        self.mrna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )
        self.mirna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )
        self.mrna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )

        # GCNConv miRNA-miRNA (homoGCN, grafo homogéneo — sin problemas de versión)
        self.gcn_mirna_mirna = GCNConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            add_self_loops=True,
            normalize=True,
        )

        # SAGEConv bipartita miRNA→isoforma.
        # SAGEConv acepta x=(x_src, x_dst) de forma nativa en todas las versiones
        # de PyG, a diferencia de GCNConv cuyo argumento 'size' fue eliminado.
        # El mecanismo es mean-aggregation: isotrópico como GCN pero sin el
        # requisito de simetría, lo que lo hace apto para grafos dirigidos bipartitos.
        self.sage_mirna_isoform = SAGEConv(
            in_channels=(config.hidden_dim, config.hidden_dim),
            out_channels=config.hidden_dim,
            aggr='mean',
        )

        self.dropout = nn.Dropout(config.dropout)

        # HPV
        if config.use_hpv:
            self.hpv_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, config.hpv_embed_dim),
            )
            hpv_dim = config.hpv_embed_dim
        else:
            hpv_dim = 0

        patient_emb_dim = config.hidden_dim + config.hidden_dim + hpv_dim

        self.os_head = nn.Sequential(
            nn.Linear(patient_emb_dim, config.os_head_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.os_head_dim, 1),
        )

    @staticmethod
    def _patient_pooling(expr_raw: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(expr_raw.T.clamp(min=0) * 5, dim=1)
        return torch.matmul(weights, node_emb)

    def forward(self, data: HeteroData, hpv: Optional[torch.Tensor] = None):
        ei_mm = data[("mirna", "coexpresses", "mirna")].edge_index
        ei_mi = data[("mirna", "represses", "isoform")].edge_index

        expr_mirna    = data["mirna"].x
        expr_isoforma = data["isoform"].x

        # Embedding inicial
        e_mirna    = self.mirna_expr_embed(expr_mirna)
        e_isoforma = self.mrna_expr_embed(expr_isoforma)
        h_mirna_0  = self.mirna_proj(e_mirna)
        h_iso_0    = self.mrna_proj(e_isoforma)

        # Capa 1: GCN isotrópica miRNA-miRNA
        h_mirna_1 = F.elu(self.gcn_mirna_mirna(h_mirna_0, ei_mm))
        h_mirna_1 = self.dropout(h_mirna_1)

        # Capa 2: SAGEConv bipartita miRNA→isoforma
        h_iso_1 = F.elu(
            self.sage_mirna_isoform((h_mirna_1, h_iso_0), ei_mi)
        )
        h_iso_1 = self.dropout(h_iso_1)

        # Pooling por paciente
        g_mirna    = self._patient_pooling(expr_mirna,    h_mirna_1)
        g_isoforma = self._patient_pooling(expr_isoforma, h_iso_1)
        g_patient  = torch.cat([g_mirna, g_isoforma], dim=-1)

        # HPV
        if self.config.use_hpv:
            if hpv is None and hasattr(data, "hpv"):
                hpv = data.hpv.float()
            if hpv is not None:
                hpv_emb   = self.hpv_proj(hpv.unsqueeze(-1))
                g_patient = torch.cat([g_patient, hpv_emb], dim=-1)

        return self.os_head(g_patient).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Baseline_VanillaGAT  GAT básico sin residuales ni normalización
# ──────────────────────────────────────────────────────────────────────────────

class Baseline_VanillaGAT(nn.Module):
    """
    Baseline con atención de grafos clásica (GATConv, Veličković et al., 2018).

    Justificación teórica
    ---------------------
    GAT es la arquitectura de referencia con pesos de atención en la literatura
    multi-ómica revisada: MOGAT (Tanvir et al., 2024), Li & Nabavi (2024) y
    LASSO-MOGAT (Alharbi et al., 2025) la usan como bloque central, con AUC
    reportados superiores a GCN en hasta un 46% relativo (Tabla de frameworks).

    Este baseline aísla la contribución de las mejoras arquitectónicas del modelo
    principal (GATv2 vs GATv1, multi-head, conexiones residuales, LayerNorm) frente
    a la atención básica. Concretamente:
      · GATConv clásico (v1) en lugar de GATv2Conv
      · 1 cabeza de atención (heads=1) en lugar de 4
      · Sin conexión residual (no res_proj)
      · Sin LayerNorm

    La comparación MLP < GCN < VanillaGAT < SupraGraphNet debería validar la
    escalera de complejidad arquitectónica; cualquier resultado que se desvíe
    de este orden es en sí mismo un hallazgo relevante para el TFM.

    
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Embeddings de expresión (igual que SupraGraphNet)
        self.mirna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )
        self.mrna_expr_embed = nn.Sequential(
            nn.Linear(config.num_patients, config.expr_embed_dim),
            nn.LayerNorm(config.expr_embed_dim),
            nn.ELU(),
        )
        self.mirna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )
        self.mrna_proj = nn.Sequential(
            nn.Linear(config.expr_embed_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
        )

        # GATConv v1 miRNA-miRNA (1 head, sin residual, sin LayerNorm)
        self.gat_mirna_mirna = GATConv(
            in_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            heads=1,
            concat=False,           # heads=1 → concat no cambia nada, False por claridad
            dropout=config.dropout,
            add_self_loops=False,
        )

        # GATConv v1 bipartita miRNA→isoforma (1 head)
        self.gat_mirna_isoform = GATConv(
            in_channels=(config.hidden_dim, config.hidden_dim),
            out_channels=config.hidden_dim,
            heads=1,
            concat=False,
            dropout=config.dropout,
            add_self_loops=False,
        )

        self.dropout = nn.Dropout(config.dropout)

        # HPV
        if config.use_hpv:
            self.hpv_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, config.hpv_embed_dim),
            )
            hpv_dim = config.hpv_embed_dim
        else:
            hpv_dim = 0

        patient_emb_dim = config.hidden_dim + config.hidden_dim + hpv_dim

        self.os_head = nn.Sequential(
            nn.Linear(patient_emb_dim, config.os_head_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.os_head_dim, 1),
        )

    @staticmethod
    def _patient_pooling(expr_raw: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(expr_raw.T.clamp(min=0) * 5, dim=1)
        return torch.matmul(weights, node_emb)

    def forward(self, data: HeteroData, hpv: Optional[torch.Tensor] = None):
        ei_mm = data[("mirna", "coexpresses", "mirna")].edge_index
        ei_mi = data[("mirna", "represses", "isoform")].edge_index

        expr_mirna    = data["mirna"].x
        expr_isoforma = data["isoform"].x

        # Embedding inicial
        e_mirna    = self.mirna_expr_embed(expr_mirna)
        e_isoforma = self.mrna_expr_embed(expr_isoforma)
        h_mirna_0  = self.mirna_proj(e_mirna)
        h_iso_0    = self.mrna_proj(e_isoforma)

        # Capa 1: GAT v1 miRNA-miRNA (sin residual, sin LayerNorm)
        h_mirna_1 = F.elu(self.gat_mirna_mirna(h_mirna_0, ei_mm))
        h_mirna_1 = self.dropout(h_mirna_1)

        # Capa 2: GAT v1 bipartita miRNA→isoforma (sin residual, sin LayerNorm)
        h_iso_1 = F.elu(
            self.gat_mirna_isoform((h_mirna_1, h_iso_0), ei_mi)
        )
        h_iso_1 = self.dropout(h_iso_1)

        # Pooling por paciente
        g_mirna    = self._patient_pooling(expr_mirna,    h_mirna_1)
        g_isoforma = self._patient_pooling(expr_isoforma, h_iso_1)
        g_patient  = torch.cat([g_mirna, g_isoforma], dim=-1)

        # HPV
        if self.config.use_hpv:
            if hpv is None and hasattr(data, "hpv"):
                hpv = data.hpv.float()
            if hpv is not None:
                hpv_emb   = self.hpv_proj(hpv.unsqueeze(-1))
                g_patient = torch.cat([g_patient, hpv_emb], dim=-1)

        return self.os_head(g_patient).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "BaselineConfig",
    "Baseline_MLP",
    "Baseline_GCN",
    "Baseline_VanillaGAT",
]
