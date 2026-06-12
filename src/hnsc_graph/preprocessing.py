import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logging sencillo para scripts y notebooks."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


# ============================================================================
# 1. Curación clínica de la cohorte
# ============================================================================


@dataclass
class ClinicalFilterConfig:
    clinical_path: Path
    histology_column: str = "histological_type"
    allowed_histology_keywords: Tuple[str, ...] = (
        "adenocarcinoma ductal",
        "ductal adenocarcinoma",
    )
    sample_type_column: Optional[str] = None

    # ── Columnas de supervivencia ─────────────────────────────────────────────
    # OS.time : tiempo en días hasta muerte (fallecidos) o censura (vivos)
    # OS      : 1 = fallecido (evento completo), 0 = vivo/censurado
    os_time_column: str = "OS.time"
    os_status_column: str = "OS"

    # ── Columnas de recaída ───────────────────────────────────────────────────
    # DFI.time : tiempo en días hasta recaída o última observación libre
    # DFI      : 1 = recaída, 0 = sin recaída, NaN = sin datos (≠ sin recaída)
    dfi_time_column: str = "DFI.time"
    dfi_status_column: str = "DFI"

    # ── Columna de progresión ─────────────────────────────────────────────────
    pfi_time_column: str = "PFI.time"
    pfi_status_column: str = "PFI"

    # ── Criterio de filtrado por tiempo nulo ──────────────────────────────────
    # Se elimina un paciente si OS.time=0 Y PFI.time=0 Y DFI.time=NaN.
    # Un paciente con OS.time=0 pero PFI.time>0 o DFI.time>0 SE MANTIENE.
    remove_zero_followup: bool = True


class ClinicalFilter:


    def __init__(self, config: ClinicalFilterConfig) -> None:
        self.config = config

    def load_clinical(self) -> pd.DataFrame:
        clinical_path = self.config.clinical_path
        if not clinical_path.exists():
            raise FileNotFoundError(f"Clinical file not found: {clinical_path}")
        try:
            sep = "\t" if clinical_path.suffix.lower() in {".tsv", ".txt"} else ","
            df = pd.read_csv(clinical_path, sep=sep)
        except Exception as exc:
            raise RuntimeError(f"Error reading clinical file: {exc}") from exc
        logger.info("Archivo clínico cargado: %d filas x %d columnas", *df.shape)
        return df

    def _is_ductal_histology(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        return any(k in value.lower() for k in self.config.allowed_histology_keywords)

    @staticmethod
    def _is_primary_tumor_barcode(barcode: str) -> bool:
        if not isinstance(barcode, str):
            return False
        return "-01" in barcode[12:16]

    def _remove_zero_followup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina pacientes con seguimiento completamente nulo.
        Criterio: OS.time=0 Y PFI.time=0 Y DFI.time=NaN.
        """
        os_t  = self.config.os_time_column
        pfi_t = self.config.pfi_time_column
        dfi_t = self.config.dfi_time_column

        if os_t not in df.columns:
            logger.warning("Columna %s no encontrada, no se filtra seguimiento nulo.", os_t)
            return df

        if pfi_t in df.columns and dfi_t in df.columns:
            mask_zero = (
                (df[os_t].fillna(0) == 0) &
                (df[pfi_t].fillna(0) == 0) &
                (df[dfi_t].isna())
            )
        elif pfi_t in df.columns:
            mask_zero = (df[os_t].fillna(0) == 0) & (df[pfi_t].fillna(0) == 0)
        else:
            mask_zero = df[os_t].fillna(0) == 0

        for idx in df.loc[mask_zero].index:
            logger.info("Eliminado paciente con seguimiento nulo: %s", idx)
        if not mask_zero.any():
            logger.info("No hay pacientes con seguimiento nulo.")

        return df.loc[~mask_zero].copy()

    def filter_pdac_primary(
        self,
        clinical_df: pd.DataFrame,
        barcode_column: str = "bcr_patient_barcode",
    ) -> pd.DataFrame:
        """
        Filtra para retener adenocarcinomas ductales primarios con seguimiento válido.

        Pasos:
        1. Filtro por histología ductal.
        2. Filtro por tumor primario (código 01).
        3. Eliminación de pacientes con seguimiento nulo.
        """
        for col in (self.config.histology_column, barcode_column):
            if col not in clinical_df.columns:
                raise KeyError(f"Required clinical column not found: {col}")

        hist_mask = clinical_df[self.config.histology_column].map(self._is_ductal_histology)
        logger.info("Histología ductal: %d / %d", hist_mask.sum(), len(hist_mask))

        if self.config.sample_type_column and self.config.sample_type_column in clinical_df.columns:
            prim_mask = clinical_df[self.config.sample_type_column].astype(str).str.contains(
                "01", regex=False)
        else:
            prim_mask = clinical_df[barcode_column].map(self._is_primary_tumor_barcode)
        logger.info("Tumor primario (código 01): %d", prim_mask.sum())

        filtered = clinical_df.loc[hist_mask & prim_mask].copy()
        logger.info("Tras histología + tumor primario: %d pacientes", len(filtered))

        if self.config.remove_zero_followup:
            filtered = self._remove_zero_followup(filtered)
            logger.info("Tras eliminar seguimiento nulo: %d pacientes finales", len(filtered))

        if filtered.empty:
            logger.warning("El filtrado produjo 0 pacientes. Revisa los criterios.")
        return filtered

    def get_valid_barcodes(self, barcode_column: str = "bcr_patient_barcode") -> Set[str]:
        df = self.load_clinical()
        filtered = self.filter_pdac_primary(df, barcode_column=barcode_column)
        return set(filtered[barcode_column].astype(str).str.upper())


# ============================================================================
# 2. Targets de supervivencia para el regresor GNN
# ============================================================================


def build_survival_targets(
    clinical_df: pd.DataFrame,
    os_time_col:    str = "OS.time",
    os_status_col:  str = "OS",
    dfi_time_col:   str = "DFI.time",
    dfi_status_col: str = "DFI",
) -> pd.DataFrame:
    """
    Construye el DataFrame de targets continuos para el regresor GNN.
    """
    for col in [os_time_col, os_status_col]:
        if col not in clinical_df.columns:
            raise KeyError(f"Columna requerida no encontrada: {col}")

    t = pd.DataFrame(index=clinical_df.index)
    t["os_time"]  = clinical_df[os_time_col].astype(float)
    t["os_event"] = clinical_df[os_status_col].astype(float)
    t["os_group"] = t["os_event"].map({1.0: "deceased", 0.0: "censored"})

    if dfi_time_col in clinical_df.columns:
        t["dfi_time"] = clinical_df[dfi_time_col].astype(float)
    else:
        t["dfi_time"] = np.nan
        logger.warning("Columna %s no encontrada. dfi_time=NaN.", dfi_time_col)

    if dfi_status_col in clinical_df.columns:
        t["dfi_event"] = clinical_df[dfi_status_col].astype(float)
        t["dfi_group"] = t["dfi_event"].map({1.0: "recurrence", 0.0: "no_recurrence"})
    else:
        t["dfi_event"] = np.nan
        t["dfi_group"] = np.nan
        logger.warning("Columna %s no encontrada. dfi_event=NaN.", dfi_status_col)

    logger.info(
        "Targets: %d pacientes | fallecidos=%d | censurados=%d | DFI disponible=%d",
        len(t),
        int((t["os_event"] == 1).sum()),
        int((t["os_event"] == 0).sum()),
        int(t["dfi_event"].notna().sum()),
    )
    return t


def get_dfi_mask(targets_df: pd.DataFrame) -> pd.Series:
    """
    Máscara booleana de pacientes con DFI disponible.
    Usar en el loss de la cabeza DFI del GNN para que solo esos pacientes
    contribuyan al gradiente.
    """
    return targets_df["dfi_event"].notna()


# ============================================================================
# 3. Procesamiento de isomiRs miRNA (conteos crudos)
# ============================================================================


@dataclass
class MiRNAIsoformProcessorConfig:
    isoforms_dir: Path
    file_suffix: str = ".isoforms.quantification.txt"
    region_column: str = "miRNA_region"
    count_column: str = "read_count"
    barcode_column: Optional[str] = None


class MiRNAIsoformProcessor:
    """
    Procesa archivos `isoforms.quantification.txt` de TCGA a nivel de isomiR.

    - Filtra por regiones "mature".
    - Extrae identificador MIMAT (incluye brazo 5p/3p implícitamente).
    - Agrupa por MIMAT sumando read_count.

    Resultado: matriz [n_muestras x n_isomiRs] con conteos crudos.
    Normalizar con: filter_low_expression_by_cpm + log2_cpm.
    """

    def __init__(self, config: MiRNAIsoformProcessorConfig) -> None:
        self.config = config

    def _iter_isoform_files(self) -> Iterable[Path]:
        if not self.config.isoforms_dir.exists():
            raise FileNotFoundError(
                f"miRNA isoforms directory not found: {self.config.isoforms_dir}")
        yield from sorted(
            p for p in self.config.isoforms_dir.iterdir()
            if p.name.endswith(self.config.file_suffix)
        )

    @staticmethod
    def _extract_mimat(region_value: str) -> Optional[str]:
        if not isinstance(region_value, str):
            return None
        import re
        m = re.search(r"(MIMAT\d+)", region_value)
        return m.group(1) if m else None

    @staticmethod
    def _infer_barcode_from_filename(path: Path) -> str:
        return path.name.split(".")[0].upper()

    def _load_one_file(self, path: Path) -> Tuple[str, pd.Series]:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            raise RuntimeError(f"Error reading miRNA isoform file {path}: {exc}") from exc

        for col in (self.config.region_column, self.config.count_column):
            if col not in df.columns:
                raise KeyError(f"Required column {col} not found in {path}")

        mature = df[self.config.region_column].astype(str).str.contains(
            "mature", case=False, na=False)
        df_m = df.loc[mature].copy()
        df_m["MIMAT"] = df_m[self.config.region_column].map(self._extract_mimat)
        df_m = df_m.dropna(subset=["MIMAT"])

        grouped = df_m.groupby("MIMAT")[self.config.count_column].sum().astype(int)

        if self.config.barcode_column and self.config.barcode_column in df.columns:
            barcode = str(df[self.config.barcode_column].dropna().unique()[0]).upper()
        else:
            barcode = self._infer_barcode_from_filename(path)

        return barcode, grouped

    def build_count_matrix(self, valid_barcodes: Optional[Set[str]] = None) -> pd.DataFrame:
        """Construye la matriz [n_muestras x n_MIMAT] de conteos crudos."""
        series_list: List[pd.Series] = []
        sample_ids: List[str] = []

        for path in self._iter_isoform_files():
            barcode, counts = self._load_one_file(path)
            if valid_barcodes is not None and barcode not in valid_barcodes:
                continue
            series_list.append(counts)
            sample_ids.append(barcode)

        if not series_list:
            raise RuntimeError("Empty miRNA isoform matrix after processing.")

        matrix = pd.DataFrame(series_list, index=sample_ids).fillna(0).astype(int)
        logger.info("Matriz isomiRs: %s (muestras x MIMAT)", matrix.shape)
        return matrix


# ============================================================================
# 4. Normalización isomiRs miRNA: filtrado CPM + log2(RPM+1)
# ============================================================================


def _compute_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    """CPM = 10^6 * counts_ij / library_size_j. Corrige por profundidad."""
    if (counts < 0).any().any():
        raise ValueError("Counts matrix contains negative values.")
    library_sizes = counts.sum(axis=0).replace(0, np.nan)
    return counts.divide(library_sizes, axis=1) * 1e6


def log2_cpm(counts: pd.DataFrame, pseudo_count: float = 1.0) -> pd.DataFrame:

    cpm = _compute_cpm(counts)
    transformed = np.log2(cpm + float(pseudo_count))
    return pd.DataFrame(transformed, index=counts.index, columns=counts.columns, dtype=float)


def filter_low_expression_by_cpm(
    counts: pd.DataFrame,
    min_cpm: float = 1.0,
    min_samples_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Filtra isomiRs miRNA con baja expresión.
    Criterio: CPM >= min_cpm en al menos min_samples_fraction de muestras.
    Aplicar sobre matriz de conteos crudos [features x muestras].
    """
    if not (0.0 < min_samples_fraction <= 1.0):
        raise ValueError("min_samples_fraction must be in (0, 1].")
    cpm = _compute_cpm(counts)
    mask = (cpm >= min_cpm).sum(axis=1) >= (min_samples_fraction * cpm.shape[1])
    filtered = counts.loc[mask].copy()
    logger.info("Filtrado CPM isomiRs: %d -> %d (umbral=%.2f, fracción=%.2f)",
                counts.shape[0], filtered.shape[0], min_cpm, min_samples_fraction)
    return filtered


# ============================================================================
# 5. Filtrado y normalización de isoformas mRNA (RSEM UQ-normalized)
# ============================================================================


def filter_mrna_isoforms(
    mrna_matrix: pd.DataFrame,
    min_uq_fraction: float = 0.2,
    max_zero_fraction: float = 0.75,
    min_median: float = 1.0,
) -> pd.DataFrame:
    
    n_orig = mrna_matrix.shape[1]

    # F1: presencia mínima
    mask_f1 = (mrna_matrix >= 1.0).mean(axis=0) >= min_uq_fraction
    m1 = mrna_matrix.loc[:, mask_f1]
    logger.info("F1 UQ>=1 en >=%.0f%% muestras: %d -> %d isoformas",
                min_uq_fraction * 100, n_orig, m1.shape[1])

    # F2: máximo de ceros
    mask_f2 = (m1 == 0).mean(axis=0) <= max_zero_fraction
    m2 = m1.loc[:, mask_f2]
    logger.info("F2 ceros <=%.0f%%: %d -> %d isoformas",
                max_zero_fraction * 100, m1.shape[1], m2.shape[1])

    # F3: mediana mínima
    mask_f3 = m2.median(axis=0) > min_median
    m3 = m2.loc[:, mask_f3].copy()
    logger.info("F3 mediana >%.1f: %d -> %d isoformas (%.0f%% reducción total)",
                min_median, m2.shape[1], m3.shape[1],
                (1 - m3.shape[1] / n_orig) * 100)

    return m3


def log2_uq(mrna_matrix: pd.DataFrame, pseudo_count: float = 1.0) -> pd.DataFrame:
    """
    log2(UQ_normalized + pseudo_count) para isoformas mRNA.
    Aplicar DESPUÉS de filter_mrna_isoforms, sobre valores lineales UQ.
    NO usar log2_cpm aquí — los datos ya están UQ-normalizados.
    """
    t = np.log2(mrna_matrix + float(pseudo_count))
    return pd.DataFrame(t, index=mrna_matrix.index, columns=mrna_matrix.columns, dtype=float)


# ============================================================================
# 6. Selección de isoformas mRNA por señal de supervivencia (Spearman)
# ============================================================================


def select_isoforms_by_survival_correlation(
    mrna_expr_log2: pd.DataFrame,
    os_time_deceased: pd.Series,
    top_n_variance: int = 10000,
    p_thresh: float = 0.05,
    r_thresh: float = 0.2,
    r_thresh_soft: float = 0.15,
    min_selected: int = 3000,
    top_var_complement: int = 5000,
    max_selected: int = 10000,
) -> Tuple[List[str], pd.DataFrame]:
   
    # Preselección por varianza
    top_var_idx = mrna_expr_log2.var(axis=0).nlargest(top_n_variance).index
    mrna_top    = mrna_expr_log2[top_var_idx].copy()

    # Alinear fallecidos con datos de expresión
    dec_in_expr     = [i for i in os_time_deceased.index if i in mrna_top.index]
    os_time_aligned = os_time_deceased.loc[dec_in_expr]
    mrna_dec        = mrna_top.loc[dec_in_expr]

    logger.info("Spearman: %d fallecidos x %d isoformas (preselección varianza)",
                len(dec_in_expr), mrna_top.shape[1])

    # Correlación Spearman
    corr_vals, pvals = [], []
    for col in mrna_top.columns:
        r, p = stats.spearmanr(mrna_dec[col].values, os_time_aligned.values)
        corr_vals.append(r)
        pvals.append(p)

    _, fdr, _, _ = multipletests(pvals, method="fdr_bh")

    corr_results = pd.DataFrame({
        "isoform"    : mrna_top.columns,
        "spearman_r" : corr_vals,
        "pval"       : pvals,
        "FDR"        : fdr,
        "abs_r"      : np.abs(corr_vals),
        "neg_log10p" : -np.log10(np.array(pvals) + 1e-300),
    }).set_index("isoform")

    # Clasificación para visualización
    corr_results["category"] = "NS"
    corr_results.loc[
        (corr_results["pval"] < p_thresh) & (corr_results["spearman_r"] >= r_thresh),
        "category"] = "POS"
    corr_results.loc[
        (corr_results["pval"] < p_thresh) & (corr_results["spearman_r"] <= -r_thresh),
        "category"] = "NEG"
    corr_results.loc[
        (corr_results["pval"] < p_thresh) & (corr_results["abs_r"] < r_thresh),
        "category"] = "SIG_WEAK"

    logger.info("Resultados Spearman: FDR<0.05=%d | p<%.2f=%d | p<%.2f+|r|>=%.2f=%d",
                int((corr_results["FDR"] < 0.05).sum()),
                p_thresh, int((corr_results["pval"] < p_thresh).sum()),
                p_thresh, r_thresh,
                int(((corr_results["pval"] < p_thresh) &
                     (corr_results["abs_r"] >= r_thresh)).sum()))

    # Selección escalonada
    selected = set()

    sel_main = corr_results[
        (corr_results["pval"] < p_thresh) & (corr_results["abs_r"] >= r_thresh)
    ].index.tolist()
    selected.update(sel_main)
    logger.info("Paso 1 — p<%.2f y |r|>=%.2f: %d isoformas", p_thresh, r_thresh, len(sel_main))

    if len(selected) < min_selected:
        sel_soft = corr_results[
            (corr_results["pval"] < p_thresh) & (corr_results["abs_r"] >= r_thresh_soft)
        ].index.tolist()
        selected.update(sel_soft)
        logger.info("Paso 2 — |r|>=%.2f: acumulado %d", r_thresh_soft, len(selected))

    if len(selected) < min_selected:
        top_var_comp = mrna_expr_log2.var(axis=0).nlargest(top_var_complement).index.tolist()
        selected.update(top_var_comp)
        logger.info("Paso 3 — top varianza %d: acumulado %d", top_var_complement, len(selected))

    corr_ext = corr_results.reindex(list(selected))
    corr_ext["abs_r"] = corr_ext["abs_r"].fillna(0)
    selected_ordered = (
        corr_ext.sort_values("abs_r", ascending=False)
        .head(max_selected)
        .index.tolist()
    )
    logger.info("Isoformas seleccionadas final: %d (cap=%d)", len(selected_ordered), max_selected)
    return selected_ordered, corr_results


__all__ = [
    # Setup
    "setup_logging",
    # Curación clínica
    "ClinicalFilterConfig",
    "ClinicalFilter",
    # Targets de supervivencia
    "build_survival_targets",
    "get_dfi_mask",
    # isomiRs miRNA
    "MiRNAIsoformProcessorConfig",
    "MiRNAIsoformProcessor",
    "log2_cpm",
    "filter_low_expression_by_cpm",
    # Isoformas mRNA
    "filter_mrna_isoforms",
    "log2_uq",
    "select_isoforms_by_survival_correlation",
]
