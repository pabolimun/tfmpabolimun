"""
gnn_models.py — SupraGraphNet para predicción de supervivencia (HNSC) v8

"""

import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Barra de progreso
# ──────────────────────────────────────────────────────────────────────────────

def _progress(epoch, total, loss, auc, width=30):
    filled = int(width * epoch / total)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] ep {epoch:>4}/{total}  loss {loss:.4f}  AUC {auc:.3f}",
          end="", flush=True)
    if epoch == total:
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Capas GAT 
# ──────────────────────────────────────────────────────────────────────────────

class BipartiteGATLayer(nn.Module):
    def __init__(self, in_src, in_dst, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=(in_src, in_dst),
            out_channels=out_channels,
            heads=heads, concat=True,
            dropout=dropout, add_self_loops=False,
        )
        self.res_proj = nn.Linear(in_dst, out_channels * heads, bias=False)
        self.norm = nn.LayerNorm(out_channels * heads)

    def forward(self, x_src, x_dst, edge_index):
        out = self.gat((x_src, x_dst), edge_index)
        return F.elu(self.norm(out + self.res_proj(x_dst)))

    def forward_with_attention(self, x_src, x_dst, edge_index):
        out_raw, (ei_out, alpha) = self.gat(
            (x_src, x_dst), edge_index, return_attention_weights=True
        )
        out = F.elu(self.norm(out_raw + self.res_proj(x_dst)))
        return out, ei_out, alpha


class HomoGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=True,
            dropout=dropout, add_self_loops=False,
        )
        self.res_proj = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.norm = nn.LayerNorm(out_channels * heads)

    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)
        return F.elu(self.norm(out + self.res_proj(x)))

    def forward_with_attention(self, x, edge_index):
        out_raw, (ei_out, alpha) = self.gat(
            x, edge_index, return_attention_weights=True
        )
        out = F.elu(self.norm(out_raw + self.res_proj(x)))
        return out, ei_out, alpha


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SupraGraphNetConfig:
    num_mirna:      int
    num_mrna:       int
    num_patients:   int
    hidden_dim:     int   = 64
    os_head_dim:    int   = 32
    heads:          int   = 4
    dropout:        float = 0.4
    expr_embed_dim: int   = 32
    use_hpv:        bool  = True
    hpv_embed_dim:  int   = 8


# ──────────────────────────────────────────────────────────────────────────────
# Modelo 
# ──────────────────────────────────────────────────────────────────────────────

class SupraGraphNet(nn.Module):
    """
    SupraGraphNet — clasificación binaria con Focal Loss + Label Smoothing.

    """

    def __init__(self, config: SupraGraphNetConfig):
        super().__init__()
        self.config = config

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

        self.mirna_mirna_layer = HomoGATLayer(
            config.hidden_dim, config.hidden_dim,
            config.heads, config.dropout,
        )
        self.layer1 = BipartiteGATLayer(
            config.hidden_dim * config.heads, config.hidden_dim,
            config.hidden_dim, config.heads, config.dropout,
        )
        self.layer2 = BipartiteGATLayer(
            config.hidden_dim * config.heads,
            config.hidden_dim * config.heads,
            config.hidden_dim, 1, config.dropout,
        )

        if config.use_hpv:
            self.hpv_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, config.hpv_embed_dim),
            )
            hpv_dim = config.hpv_embed_dim
        else:
            hpv_dim = 0

        patient_emb_dim = config.hidden_dim * config.heads + config.hidden_dim + hpv_dim

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

    def _encode(self, data: HeteroData):
        ei_mm = data[("mirna", "coexpresses", "mirna")].edge_index
        ei_mi = data[("mirna", "represses", "isoform")].edge_index
        expr_mirna    = data["mirna"].x
        expr_isoforma = data["isoform"].x
        e_mirna    = self.mirna_expr_embed(expr_mirna)
        e_isoforma = self.mrna_expr_embed(expr_isoforma)
        h_mirna_0  = self.mirna_proj(e_mirna)
        h_mrna_0   = self.mrna_proj(e_isoforma)
        return h_mirna_0, h_mrna_0, expr_mirna, expr_isoforma, ei_mm, ei_mi

    def forward(self, data: HeteroData, hpv: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        h_mirna_0, h_mrna_0, expr_mirna, expr_isoforma, ei_mm, ei_mi = self._encode(data)

        if return_attention:
            h_mirna_1, ei_mm_out, alpha_mm = \
                self.mirna_mirna_layer.forward_with_attention(h_mirna_0, ei_mm)
            h_mrna_1, ei_mi_l1_out, alpha_mi_l1 = \
                self.layer1.forward_with_attention(h_mirna_1, h_mrna_0, ei_mi)
            h_mrna_2, ei_mi_l2_out, alpha_mi_l2 = \
                self.layer2.forward_with_attention(h_mirna_1, h_mrna_1, ei_mi)
            attn_dict = {
                "mirna_mirna":       {"edge_index": ei_mm_out,    "alpha": alpha_mm},
                "mirna_isoforma_l1": {"edge_index": ei_mi_l1_out, "alpha": alpha_mi_l1},
                "mirna_isoforma_l2": {"edge_index": ei_mi_l2_out, "alpha": alpha_mi_l2},
            }
        else:
            h_mirna_1 = self.mirna_mirna_layer(h_mirna_0, ei_mm)
            h_mrna_1  = self.layer1(h_mirna_1, h_mrna_0, ei_mi)
            h_mrna_2  = self.layer2(h_mirna_1, h_mrna_1, ei_mi)

        g_mirna   = self._patient_pooling(expr_mirna,    h_mirna_1)
        g_mrna    = self._patient_pooling(expr_isoforma, h_mrna_2)
        g_patient = torch.cat([g_mirna, g_mrna], dim=-1)

        if self.config.use_hpv:
            if hpv is None and hasattr(data, "hpv"):
                hpv = data.hpv.float()
            if hpv is not None:
                hpv_emb   = self.hpv_proj(hpv.unsqueeze(-1))
                g_patient = torch.cat([g_patient, hpv_emb], dim=-1)

        logits = self.os_head(g_patient).squeeze(-1)

        if return_attention:
            return logits, attn_dict
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Loss: Focal Loss + Label Smoothing
# ──────────────────────────────────────────────────────────────────────────────

def focal_loss_with_smoothing(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Focal Loss con Label Smoothing para clasificación binaria.

    Focal Loss: FL(p) = -(1-p_t)^gamma * log(p_t)
    - gamma=0 → BCE estándar
    - gamma=2 → penaliza más los ejemplos fáciles, fuerza al modelo
      a aprender los casos difíciles (los acumulados cerca de 0.5)

    Label Smoothing: convierte targets duros 0/1 en suaves epsilon/(1-epsilon)
    - smoothing=0.1 → 0→0.1, 1→0.9
    - Reduce overconfidence, mejora calibración de probabilidades
    - Especialmente útil cuando la señal biológica es débil

    Parameters
    ----------
    logits : Tensor [n]
        Salida cruda del modelo (sin sigmoid).
    labels : Tensor [n]
        Labels binarios 0/1.
    gamma : float
        Parámetro de focusing. Default 2.0.
    smoothing : float
        Factor de label smoothing. Default 0.1.

    Returns
    -------
    loss : Tensor escalar
    """
    labels = labels.float()

    # Label smoothing: 0 → smoothing, 1 → 1-smoothing
    labels_smooth = labels * (1.0 - smoothing) + smoothing * 0.5

    # Probabilidades
    probs    = torch.sigmoid(logits)
    probs_t  = torch.where(labels >= 0.5, probs, 1.0 - probs)

    # BCE con labels suavizados
    bce = F.binary_cross_entropy_with_logits(logits, labels_smooth, reduction='none')

    # Factor focal: (1 - p_t)^gamma
    focal_weight = (1.0 - probs_t).pow(gamma)

    loss = (focal_weight * bce).mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Métricas 
# ──────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """AUC-ROC, F1, precisión y recall a partir de logits crudos."""
    probs = 1 / (1 + np.exp(-logits_np))
    preds = (probs >= threshold).astype(int)
    auc  = roc_auc_score(labels_np, probs)   if len(np.unique(labels_np)) > 1 else 0.5
    f1   = f1_score(labels_np, preds,        zero_division=0)
    prec = precision_score(labels_np, preds, zero_division=0)
    rec  = recall_score(labels_np, preds,    zero_division=0)
    return {
        "auc": float(auc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "probs": probs, "preds": preds,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train_supra_graph_classifier(
    data, labels, train_idx, val_idx,
    config: SupraGraphNetConfig,
    hpv=None,
    num_epochs: int = 600,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    early_stop_patience: int = 200,
    noise_std: float = 0.02,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    cindex_every: int = 5,
    trial=None,
):
    """
    Entrena SupraGraphNet con Focal Loss + Label Smoothing.

    focal_gamma : float
        Parámetro focusing de Focal Loss (0=BCE pura, 2=default óptimo).
    label_smoothing : float
        Factor de suavizado de etiquetas (0=sin smoothing, 0.1=default).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)
    labels_dev = labels.float().to(device)
    if hpv is not None:
        hpv = hpv.float().to(device)

    train_idx = torch.as_tensor(list(train_idx), dtype=torch.long, device=device)
    val_idx   = torch.as_tensor(list(val_idx),   dtype=torch.long, device=device)

    model     = SupraGraphNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01,
    )

    best_auc = 0.0; best_state = None
    epochs_no_improve = 0; current_auc = 0.5
    mirna_x_orig   = data["mirna"].x.clone()
    isoform_x_orig = data["isoform"].x.clone()
    train_history = []; val_history = []

    for epoch in range(1, num_epochs + 1):
        model.train(); optimizer.zero_grad()

        if noise_std > 0:
            data["mirna"].x   = (mirna_x_orig   + torch.randn_like(mirna_x_orig)   * noise_std).clamp(min=0)
            data["isoform"].x = (isoform_x_orig + torch.randn_like(isoform_x_orig) * noise_std).clamp(min=0)

        logits = model(data, hpv=hpv)

        # Focal Loss + Label Smoothing
        loss = focal_loss_with_smoothing(
            logits[train_idx],
            labels_dev[train_idx],
            gamma=focal_gamma,
            smoothing=label_smoothing,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        model.eval()
        data["mirna"].x   = mirna_x_orig
        data["isoform"].x = isoform_x_orig
        train_history.append(float(loss.item()))

        if epoch % cindex_every == 0 or epoch == 1:
            with torch.no_grad():
                logits_all = model(data, hpv=hpv)
            m = compute_classification_metrics(
                logits_all[val_idx].cpu().numpy(),
                labels_dev[val_idx].cpu().numpy(),
            )
            current_auc = m["auc"]
            val_history.append((epoch, current_auc))

            if trial is not None:
                trial.report(1 - current_auc, epoch)
                if trial.should_prune():
                    raise __import__("optuna").exceptions.TrialPruned()

            if current_auc > best_auc:
                best_auc = current_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += cindex_every
                if epochs_no_improve >= early_stop_patience:
                    _progress(epoch, num_epochs, loss.item(), current_auc)
                    print(f"\n  ⏹️  Early stopping en época {epoch}  |  mejor AUC: {best_auc:.4f}")
                    break

        if epoch % 10 == 0 or epoch == 1:
            _progress(epoch, num_epochs, loss.item(), current_auc)

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    data["mirna"].x   = mirna_x_orig
    data["isoform"].x = isoform_x_orig

    with torch.no_grad():
        logits_final = model(data, hpv=hpv)

    final_m = compute_classification_metrics(
        logits_final[val_idx].cpu().numpy(),
        labels_dev[val_idx].cpu().numpy(),
    )
    return model, {
        "val_auc":       final_m["auc"],
        "val_f1":        final_m["f1"],
        "val_precision": final_m["precision"],
        "val_recall":    final_m["recall"],
        "val_probs":     final_m["probs"],
        "val_preds":     final_m["preds"],
        "val_labels":    labels_dev[val_idx].cpu().numpy(),
        "train_history": train_history,
        "val_history":   val_history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optuna
# ──────────────────────────────────────────────────────────────────────────────

def make_optuna_objective_classifier(
    data, labels, q14_indices, base_config_kwargs,
    hpv=None, num_epochs_trial=200,
    early_stop_patience_trial=80, n_folds=4, seed=42,
):
    """
    Objetivo Optuna para clasificación con Focal Loss + Label Smoothing.
    Añade focal_gamma y label_smoothing al espacio de búsqueda.
    Minimiza 1-AUC (media K-Fold).
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    labels_q14 = labels[q14_indices].numpy().astype(int)

    def objective(trial):
        hidden_dim     = trial.suggest_categorical("hidden_dim",     [64, 96, 128])
        heads          = trial.suggest_categorical("heads",          [2, 4])
        dropout        = trial.suggest_float("dropout",              0.2, 0.5)
        expr_embed_dim = trial.suggest_categorical("expr_embed_dim", [16, 32, 64])
        hpv_embed_dim  = trial.suggest_categorical("hpv_embed_dim",  [4, 8, 16])
        lr             = trial.suggest_float("lr",                   1e-4, 5e-3, log=True)
        weight_decay   = trial.suggest_float("weight_decay",         1e-3, 5e-2, log=True)
        noise_std      = trial.suggest_float("noise_std",            0.0, 0.05)
        focal_gamma    = trial.suggest_float("focal_gamma",          0.5, 4.0)
        label_smoothing= trial.suggest_float("label_smoothing",      0.05, 0.2)

        config = SupraGraphNetConfig(
            **base_config_kwargs,
            hidden_dim=hidden_dim, heads=heads, dropout=dropout,
            expr_embed_dim=expr_embed_dim, hpv_embed_dim=hpv_embed_dim,
        )

        fold_aucs = []
        for fold, (train_rel, val_rel) in enumerate(skf.split(q14_indices, labels_q14)):
            torch.manual_seed(seed + fold)
            _, metrics = train_supra_graph_classifier(
                data=data, labels=labels,
                train_idx=q14_indices[train_rel],
                val_idx=q14_indices[val_rel],
                config=config, hpv=hpv,
                num_epochs=num_epochs_trial, lr=lr,
                weight_decay=weight_decay,
                early_stop_patience=early_stop_patience_trial,
                noise_std=noise_std,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
                trial=trial if fold == 0 else None,
            )
            fold_aucs.append(metrics["val_auc"])
            trial.report(1 - float(np.mean(fold_aucs)), fold)
            if trial.should_prune():
                raise __import__("optuna").exceptions.TrialPruned()

        return float(1 - np.mean(fold_aucs))

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Interpretabilidad 
# ──────────────────────────────────────────────────────────────────────────────

def get_attention_weights(model, data, hpv=None):
    model.eval()
    device = next(model.parameters()).device
    data   = data.to(device)
    if hpv is not None:
        hpv = hpv.float().to(device)
    mirna_ids   = np.array(data["mirna"].node_ids)
    isoform_ids = np.array(data["isoform"].node_ids)
    with torch.no_grad():
        _, attn_dict = model(data, hpv=hpv, return_attention=True)
    results = {}
    ei_mm  = attn_dict["mirna_mirna"]["edge_index"].cpu().numpy()
    alp_mm = attn_dict["mirna_mirna"]["alpha"].cpu().numpy()
    if alp_mm.ndim > 1: alp_mm = alp_mm.mean(axis=1)
    results["mirna_mirna"] = pd.DataFrame({
        "mirna_src_id": mirna_ids[ei_mm[0]],
        "mirna_dst_id": mirna_ids[ei_mm[1]],
        "attention":    alp_mm.astype(float),
    }).sort_values("attention", ascending=False).reset_index(drop=True)
    ei_l1  = attn_dict["mirna_isoforma_l1"]["edge_index"].cpu().numpy()
    alp_l1 = attn_dict["mirna_isoforma_l1"]["alpha"].cpu().numpy()
    if alp_l1.ndim > 1: alp_l1 = alp_l1.mean(axis=1)
    df_l1 = pd.DataFrame({
        "mirna_id":    mirna_ids[ei_l1[0]],
        "isoforma_id": isoform_ids[ei_l1[1]],
        "attention":   alp_l1.astype(float),
    }).sort_values("attention", ascending=False).reset_index(drop=True)
    results["mirna_isoforma_l1"] = df_l1
    ei_l2  = attn_dict["mirna_isoforma_l2"]["edge_index"].cpu().numpy()
    alp_l2 = attn_dict["mirna_isoforma_l2"]["alpha"].cpu().numpy()
    if alp_l2.ndim > 1: alp_l2 = alp_l2.mean(axis=1)
    df_l2 = pd.DataFrame({
        "mirna_id":    mirna_ids[ei_l2[0]],
        "isoforma_id": isoform_ids[ei_l2[1]],
        "attention":   alp_l2.astype(float),
    }).sort_values("attention", ascending=False).reset_index(drop=True)
    results["mirna_isoforma_l2"] = df_l2
    df_merge = df_l1.merge(df_l2, on=["mirna_id","isoforma_id"],
                            suffixes=("_l1","_l2"), how="outer").fillna(0)
    df_merge["attention"] = (df_merge["attention_l1"] + df_merge["attention_l2"]) / 2
    results["mirna_isoforma"] = (
        df_merge[["mirna_id","isoforma_id","attention"]]
        .sort_values("attention", ascending=False).reset_index(drop=True)
    )
    return results


def get_attention_edge_importance(model, data, edge_type="mirna_isoforma",
                                  top_k=None, hpv=None):
    df = get_attention_weights(model, data, hpv=hpv)[edge_type]
    return df.head(top_k) if top_k else df


def run_gnnexplainer(
    model: SupraGraphNet, data: HeteroData,
    patient_indices: List[int],
    hpv: Optional[torch.Tensor] = None,
) -> Dict:
    """GNNExplainer por gradiente sobre logit de clasificación."""
    model.eval()
    device = next(model.parameters()).device
    data   = data.to(device)
    hpv_dev = hpv.float().to(device) if hpv is not None else None

    mirna_ids   = list(data["mirna"].node_ids)
    isoform_ids = list(data["isoform"].node_ids)
    n_mirna     = len(mirna_ids)
    n_isoform   = len(isoform_ids)

    mirna_imp_acc   = np.zeros(n_mirna)
    isoform_imp_acc = np.zeros(n_isoform)
    ei_mm = data[("mirna","coexpresses","mirna")].edge_index.cpu().numpy()
    ei_mi = data[("mirna","represses","isoform")].edge_index.cpu().numpy()
    edge_mm_imp_acc = np.zeros(ei_mm.shape[1])
    edge_mi_imp_acc = np.zeros(ei_mi.shape[1])
    n_explained = 0

    for pat_idx in patient_indices:
        try:
            mirna_x = data["mirna"].x.clone().detach().requires_grad_(True)
            iso_x   = data["isoform"].x.clone().detach().requires_grad_(True)
            data_grad = data.clone()
            data_grad["mirna"].x   = mirna_x
            data_grad["isoform"].x = iso_x
            logits = model(data_grad, hpv=hpv_dev)
            logits[pat_idx].backward()

            if mirna_x.grad is not None:
                g = mirna_x.grad.abs().mean(dim=1).cpu().numpy()
                mirna_imp_acc += g
                for e in range(ei_mm.shape[1]):
                    edge_mm_imp_acc[e] += g[ei_mm[0, e]]
            if iso_x.grad is not None:
                gi = iso_x.grad.abs().mean(dim=1).cpu().numpy()
                isoform_imp_acc += gi
                if mirna_x.grad is not None:
                    gm = mirna_x.grad.abs().mean(dim=1).cpu().numpy()
                    for e in range(ei_mi.shape[1]):
                        edge_mi_imp_acc[e] += (gm[ei_mi[0,e]] + gi[ei_mi[1,e]]) / 2
            n_explained += 1
        except Exception as ex:
            logger.warning(f"Error paciente {pat_idx}: {ex}")

    if n_explained == 0:
        raise RuntimeError("No se pudo explicar ningún paciente.")

    def _df_nodes(ids, imp):
        return pd.DataFrame({"id": ids, "mean_importance": imp / n_explained})\
                 .sort_values("mean_importance", ascending=False).reset_index(drop=True)

    return {
        "mirna_importance":     _df_nodes(mirna_ids,   mirna_imp_acc).rename(columns={"id":"mirna_id"}),
        "isoform_importance":   _df_nodes(isoform_ids, isoform_imp_acc).rename(columns={"id":"isoform_id"}),
        "edge_mm_importance":   pd.DataFrame({
            "mirna_src":       [mirna_ids[ei_mm[0,i]] for i in range(ei_mm.shape[1])],
            "mirna_dst":       [mirna_ids[ei_mm[1,i]] for i in range(ei_mm.shape[1])],
            "mean_importance": edge_mm_imp_acc / n_explained,
        }).sort_values("mean_importance", ascending=False).reset_index(drop=True),
        "edge_mi_importance":   pd.DataFrame({
            "mirna_id":        [mirna_ids[ei_mi[0,i]]   for i in range(ei_mi.shape[1])],
            "isoform_id":      [isoform_ids[ei_mi[1,i]] for i in range(ei_mi.shape[1])],
            "mean_importance": edge_mi_imp_acc / n_explained,
        }).sort_values("mean_importance", ascending=False).reset_index(drop=True),
        "n_patients_explained": n_explained,
    }


__all__ = [
    "SupraGraphNetConfig", "SupraGraphNet",
    "focal_loss_with_smoothing", "compute_classification_metrics",
    "train_supra_graph_classifier",
    "make_optuna_objective_classifier",
    "get_attention_weights", "get_attention_edge_importance",
    "run_gnnexplainer",
]
