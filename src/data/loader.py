"""
Unified loaders for ACI-IoT'23 and CIC-IoT'23 raw data.

Goal: both datasets exposed through the same interface so downstream
code (preprocessing, classifiers, agent tools) never branches on
dataset identity.

Both loaders return a DataFrame whose feature columns are dataset-specific
(the two datasets share no column names) but whose label columns are always
the same three:

    label         canonical fine-grained class  (12 for ACI, 34 for CIC)
    label_group   coarse group                  (Benign / DoS / DDoS / ...)
    label_binary  "Benign" or "Attack"

A ``LoadReport`` is attached to ``df.attrs["load_report"]`` describing what was
read, what was sampled, and what was dropped.

Why CIC is streamed rather than concatenated
--------------------------------------------
CIC-IoT'23 is 45,019,243 rows across 63 files (~8.9 GB of CSV). Materializing
it before sampling would need roughly 14 GB as float64, so the loader samples
*during* a single streaming pass and only concatenates the survivors.

Sampling is Bernoulli with a per-class acceptance probability
``min(1, cap / global_count[class])`` taken from the measured counts frozen in
``config/datasets.yaml``. This draws a uniform sample across all 63 files in
one pass. The alternative — taking the first N rows per class — would pull the
majority classes entirely from the earliest files and bake in the capture
ordering, since the files are chronological.

The random stream is consumed sequentially, one draw per row read, so the
selected sample depends on the seed and the file order but NOT on
``chunksize``: reading 2 x 500k draws exactly the same stream as 1 x 1M.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from . import feature_schema as fs

ACI = fs.ACI
CIC = fs.CIC

DEFAULT_CONFIG_PATH = "config/datasets.yaml"
DEFAULT_CHUNKSIZE = 500_000

BENIGN = "Benign"
ATTACK = "Attack"


# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

@dataclass
class LoadReport:
    """What the loader actually did. Printed rather than inferred."""

    dataset: str
    files_read: int = 0
    rows_read: int = 0
    rows_returned: int = 0
    rows_dropped_null_label: int = 0
    rows_skipped_malformed: int = 0
    per_class_cap: Optional[int] = None
    sampled: bool = False
    acceptance_rates: Dict[str, float] = field(default_factory=dict)
    class_counts_returned: Dict[str, int] = field(default_factory=dict)
    count_drift: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"LoadReport({self.dataset})",
            f"  files read            : {self.files_read}",
            f"  rows read             : {self.rows_read:,}",
            f"  rows returned         : {self.rows_returned:,}",
            f"  classes returned      : {len(self.class_counts_returned)}",
        ]
        if self.rows_skipped_malformed:
            lines.append(f"  malformed rows skipped: {self.rows_skipped_malformed:,}")
        if self.rows_dropped_null_label:
            lines.append(f"  null labels dropped   : {self.rows_dropped_null_label:,}")
        if self.sampled:
            lines.append(f"  per-class cap         : {self.per_class_cap:,}")
            capped = sum(1 for r in self.acceptance_rates.values() if r < 1.0)
            lines.append(f"  classes downsampled   : {capped}")
        if self.count_drift:
            lines.append("  COUNT DRIFT vs frozen config:")
            for cls, (expected, observed) in sorted(self.count_drift.items()):
                lines.append(f"    {cls}: expected {expected:,}, observed {observed:,}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Read config/datasets.yaml."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dataset_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    if name not in config:
        raise KeyError(
            f"Dataset {name!r} not present in config. Available: "
            f"{[k for k in config if k not in ('split', 'cleaning')]}"
        )
    return config[name]


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def normalize_labels(
    raw: pd.Series,
    label_map: Dict[str, str],
    *,
    dataset: str,
) -> pd.Series:
    """
    Map raw label strings to canonical names via an explicit table.

    Normalization is ``raw.strip().upper()`` to form the lookup key, so
    "Benign", "BENIGN" and " benign " all resolve to the same entry, and the
    mixed-case spellings used by the original CIC release
    (``DDoS-ICMP_Flood``) resolve identically to these pre-uppercased files.

    An unmapped key raises. Silently passing an unknown label through is how a
    35th class enters a 34-class experiment unnoticed.
    """
    keys = raw.astype("string").str.strip().str.upper()
    mapped = keys.map(label_map)

    unknown = keys[mapped.isna() & keys.notna()]
    if len(unknown):
        counts = unknown.value_counts()
        detail = ", ".join(f"{k!r} ({v:,} rows)" for k, v in counts.head(10).items())
        raise ValueError(
            f"[{dataset}] {len(counts)} label value(s) are missing from "
            f"`label_map` in config/datasets.yaml: {detail}. Add them to the "
            f"map (with a `label_groups` entry and a `class_order` position) "
            f"rather than letting them through."
        )
    return mapped


def _attach_label_columns(
    df: pd.DataFrame,
    label_col: str,
    ds_cfg: Dict[str, Any],
    dataset: str,
) -> pd.DataFrame:
    """Replace the raw label column with label / label_group / label_binary."""
    label_map = ds_cfg["label_map"]
    label_groups = ds_cfg["label_groups"]

    canonical = normalize_labels(df[label_col], label_map, dataset=dataset)

    missing_groups = sorted(set(canonical.dropna()) - set(label_groups))
    if missing_groups:
        raise ValueError(
            f"[{dataset}] `label_groups` is missing entries for: {missing_groups}."
        )

    df = df.drop(columns=[label_col])
    df["label"] = canonical
    df["label_group"] = canonical.map(label_groups)
    df["label_binary"] = np.where(canonical == BENIGN, BENIGN, ATTACK)
    return df


def class_order(dataset: str, config: Dict[str, Any]) -> List[str]:
    """
    Frozen class ordering for label -> int encoding.

    Pinned in config so the encoding is identical across runs and machines;
    sklearn's default alphabetical order would otherwise shift if the set of
    classes present in a split ever changes.
    """
    ds_cfg = _dataset_config(dataset, config)
    order = list(ds_cfg["class_order"])
    declared = set(ds_cfg["label_map"].values())
    if set(order) != declared:
        raise ValueError(
            f"[{dataset}] `class_order` and `label_map` disagree. "
            f"Only in class_order: {sorted(set(order) - declared)}; "
            f"only in label_map: {sorted(declared - set(order))}."
        )
    return order


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------

def _read_header(path: str) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _validate_header(path: str, expected: List[str], dataset: str) -> None:
    found = _read_header(path)
    if found != expected:
        missing = [c for c in expected if c not in found]
        extra = [c for c in found if c not in expected]
        raise ValueError(
            f"[{dataset}] Header mismatch in {os.path.basename(path)}: "
            f"expected {len(expected)} columns, found {len(found)}. "
            f"Missing: {missing[:5]}. Unexpected: {extra[:5]}. "
            f"Concatenating mismatched headers would silently produce NaN "
            f"columns, so the load is refused."
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _acceptance_rates(
    class_counts: Dict[str, int], cap: Optional[int]
) -> Dict[str, float]:
    """min(1, cap / count) per class; all 1.0 when no cap is set."""
    if cap is None:
        return {cls: 1.0 for cls in class_counts}
    return {cls: min(1.0, cap / count) for cls, count in class_counts.items()}


def _bernoulli_keep(
    labels: pd.Series, rates: Dict[str, float], draws: np.ndarray
) -> np.ndarray:
    """
    Boolean mask: keep row i when draws[i] < rate[label_i].

    A class with no entry in ``rates`` falls back to 1.0 (keep everything).
    Defaulting to 0.0 would silently discard every row of a class that is
    missing from the frozen counts; the caller reports it as a warning instead.
    """
    thresholds = labels.map(rates).to_numpy(dtype="float64", na_value=1.0)
    return draws < thresholds


# ---------------------------------------------------------------------------
# ACI
# ---------------------------------------------------------------------------

def load_aci_iot(
    config: Dict[str, Any],
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    downcast_float32: bool = False,
) -> pd.DataFrame:
    """
    Load ACI-IoT'23.

    The full dataset is 1,231,411 rows over 85 columns (~409 MB as float32) and
    fits in memory, so ``per_class_cap`` is null in the shipped config and every
    row is returned. Float64 is kept by default: the Idle* columns hold values
    around 1.7e15, which float32 cannot represent without visible error.
    """
    ds_cfg = _dataset_config(ACI, config)
    report = LoadReport(dataset=ACI, per_class_cap=ds_cfg.get("per_class_cap"))

    raw_dir = ds_cfg["raw_path"]
    paths = [os.path.join(raw_dir, f) for f in ds_cfg["files"]]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{ACI}] Expected raw file not found: {p}")

    expected_cols = fs.all_columns(ACI)
    for p in paths:
        _validate_header(p, expected_cols, ACI)

    df = _stream_files(
        paths=paths,
        dataset=ACI,
        ds_cfg=ds_cfg,
        config=config,
        chunksize=chunksize,
        downcast_float32=downcast_float32,
        report=report,
    )
    return df


# ---------------------------------------------------------------------------
# CIC
# ---------------------------------------------------------------------------

def load_cic_iot(
    config: Dict[str, Any],
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    downcast_float32: bool = True,
) -> pd.DataFrame:
    """
    Load CIC-IoT'23 by streaming all 63 Merged*.csv files in one pass.

    With the shipped ``per_class_cap: 50000`` this returns roughly 1.29M rows
    (24 classes capped, 10 rare classes taken whole), keeping every one of the
    34 classes. Set ``per_class_cap: null`` in the config to stream all 45M
    rows — expect ~7 GB resident as float32.
    """
    ds_cfg = _dataset_config(CIC, config)
    report = LoadReport(dataset=CIC, per_class_cap=ds_cfg.get("per_class_cap"))

    raw_dir = ds_cfg["raw_path"]
    pattern = os.path.join(raw_dir, ds_cfg.get("file_glob", "*.csv"))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"[{CIC}] No files matched {pattern!r}.")

    expected_files = ds_cfg.get("n_files_expected")
    if expected_files is not None and len(paths) != expected_files:
        report.warnings.append(
            f"expected {expected_files} files, found {len(paths)}"
        )

    # All 63 headers were verified byte-identical; re-check so a re-download
    # that changes one file fails loudly instead of producing NaN columns.
    expected_cols = fs.all_columns(CIC)
    for p in paths:
        _validate_header(p, expected_cols, CIC)

    df = _stream_files(
        paths=paths,
        dataset=CIC,
        ds_cfg=ds_cfg,
        config=config,
        chunksize=chunksize,
        downcast_float32=downcast_float32,
        report=report,
    )
    return df


# ---------------------------------------------------------------------------
# Shared streaming core
# ---------------------------------------------------------------------------

def _stream_files(
    *,
    paths: List[str],
    dataset: str,
    ds_cfg: Dict[str, Any],
    config: Dict[str, Any],
    chunksize: int,
    downcast_float32: bool,
    report: LoadReport,
) -> pd.DataFrame:
    label_col = ds_cfg["label_column"]
    cap = ds_cfg.get("per_class_cap")
    frozen_counts: Dict[str, int] = ds_cfg.get("class_counts", {})
    label_map = ds_cfg["label_map"]

    cleaning = config.get("cleaning", {})
    on_bad_lines = cleaning.get("on_bad_lines", "warn")
    seed = config.get("split", {}).get("seed", 42)

    rates = _acceptance_rates(frozen_counts, cap)
    report.per_class_cap = cap
    report.sampled = cap is not None
    report.acceptance_rates = rates

    if cap is not None and not frozen_counts:
        raise ValueError(
            f"[{dataset}] per_class_cap is set but `class_counts` is empty. "
            f"Sampling rates are derived from the measured counts; regenerate "
            f"them before capping."
        )

    # One generator for the whole load. Draws are consumed one per row read, in
    # file then row order, so the sample is independent of chunksize.
    rng = np.random.default_rng(seed)

    dtypes: Dict[str, str] = {}
    if downcast_float32:
        dtypes = dict(fs.numeric_dtypes(dataset))
        for col in fs.categorical_columns(dataset):
            dtypes.pop(col, None)

    kept: List[pd.DataFrame] = []
    observed_raw: Dict[str, int] = {}

    for path in paths:
        report.files_read += 1
        reader = pd.read_csv(
            path,
            chunksize=chunksize,
            dtype=dtypes or None,
            on_bad_lines=on_bad_lines,
            low_memory=False,
        )
        with warnings.catch_warnings():
            # on_bad_lines only governs rows with too MANY fields; a short row
            # is padded with NaN and read normally. CIC's 9 truncated final
            # rows therefore arrive with a null label and are removed by the
            # null-label filter below, not by the parser. Suppress the
            # per-row ParserWarning; both counts land in the report instead.
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            for chunk in reader:
                n = len(chunk)
                report.rows_read += n
                draws = rng.random(n)  # always n draws, kept or not

                null_labels = int(chunk[label_col].isna().sum())
                if null_labels:
                    report.rows_dropped_null_label += null_labels
                    keep_notna = chunk[label_col].notna().to_numpy()
                    chunk = chunk[keep_notna]
                    draws = draws[keep_notna]
                    if chunk.empty:
                        continue

                canonical = normalize_labels(
                    chunk[label_col], label_map, dataset=dataset
                )
                for cls, cnt in canonical.value_counts().items():
                    observed_raw[cls] = observed_raw.get(cls, 0) + int(cnt)

                if cap is not None:
                    mask = _bernoulli_keep(canonical, rates, draws)
                    if not mask.any():
                        continue
                    chunk = chunk[mask]

                kept.append(chunk)

    if not kept:
        raise ValueError(f"[{dataset}] Loaded zero rows from {len(paths)} file(s).")

    df = pd.concat(kept, ignore_index=True)
    del kept

    # rows_read already counts rows that were later dropped for a null label,
    # so the shortfall is against rows_read alone: what the parser refused.
    expected_rows = ds_cfg.get("n_rows_expected")
    if expected_rows is not None:
        shortfall = expected_rows - report.rows_read
        if shortfall > 0:
            report.rows_skipped_malformed = shortfall
        elif shortfall < 0:
            report.warnings.append(
                f"read {report.rows_read:,} rows, more than the frozen "
                f"n_rows_expected of {expected_rows:,}"
            )

    # Drift check: the frozen counts drive the sampling rates, so stale counts
    # silently change the sample. Surface it rather than let it pass.
    #
    # Only meaningful after a complete pass. Comparing per-class totals from a
    # partial read (a file subset, a dev run) against whole-dataset counts would
    # report every class as drifted, which is noise, not signal.
    full_pass = expected_rows is not None and report.rows_read >= expected_rows * 0.999
    if full_pass:
        for cls, expected in frozen_counts.items():
            observed = observed_raw.get(cls, 0)
            if expected and abs(observed - expected) / expected > 0.01:
                report.count_drift[cls] = (expected, observed)
    elif frozen_counts:
        report.warnings.append(
            "partial read: per-class drift against the frozen counts was not "
            "checked, and sampling rates still assume whole-dataset totals"
        )
    unexpected = sorted(set(observed_raw) - set(frozen_counts))
    if unexpected:
        report.warnings.append(
            f"classes present in data but absent from class_counts: {unexpected}"
        )

    df = _attach_label_columns(df, label_col, ds_cfg, dataset)

    if cleaning.get("replace_inf_with_nan", True):
        inf_cols = [c for c in fs.columns_with(dataset, "has_inf") if c in df.columns]
        if inf_cols:
            df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)

    if cleaning.get("drop_zero_variance", True):
        drop = [
            c for c, spec in fs.schema(dataset).items()
            if spec["zero_variance"] and c in df.columns
        ]
        if drop:
            df = df.drop(columns=drop)

    report.rows_returned = len(df)
    report.class_counts_returned = {
        str(k): int(v) for k, v in df["label"].value_counts().items()
    }

    expected_classes = set(ds_cfg["label_map"].values())
    missing = sorted(expected_classes - set(report.class_counts_returned))
    if missing:
        report.warnings.append(f"classes absent from the loaded sample: {missing}")

    df.attrs["load_report"] = report
    df.attrs["dataset"] = dataset
    df.attrs["class_order"] = class_order(dataset, config)
    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    ACI: load_aci_iot,
    CIC: load_cic_iot,
}


def load_dataset(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
    use_cache: bool = True,
    rebuild_cache: bool = False,
    chunksize: int = DEFAULT_CHUNKSIZE,
    return_report: bool = False,
):
    """
    Load either dataset through one interface.

    Downstream code calls ``load_dataset("aci_iot_2023", cfg)`` or
    ``load_dataset("cic_iot_2023", cfg)`` and never branches on which.

    With ``use_cache`` the processed frame is written to (and read back from)
    ``processed_path``, so the multi-minute CIC pass runs once. Note the cache
    is keyed only by path: after changing ``per_class_cap``, the seed, or the
    label map, pass ``rebuild_cache=True``.
    """
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset {name!r}. Expected one of {sorted(_LOADERS)}.")

    config = load_config(config_path) if config is None else config
    ds_cfg = _dataset_config(name, config)
    cache_path = ds_cfg.get("processed_path")

    if use_cache and not rebuild_cache and cache_path and os.path.exists(cache_path):
        df = _read_cache(cache_path)
        df.attrs.setdefault("dataset", name)
        df.attrs.setdefault("class_order", class_order(name, config))
        # Parquet does not round-trip DataFrame.attrs, so a cache hit cannot
        # return the original report. Say so rather than hand back empty counts
        # that look like a load which read nothing.
        report = LoadReport(dataset=name, rows_returned=len(df))
        report.class_counts_returned = {
            str(k): int(v) for k, v in df["label"].value_counts().items()
        }
        report.warnings.append(
            f"served from cache {cache_path}; file/row counts are not available. "
            f"Pass rebuild_cache=True after changing per_class_cap, the seed, "
            f"or the label map."
        )
        df.attrs["load_report"] = report
        return (df, report) if return_report else df

    df = _LOADERS[name](config, chunksize=chunksize)

    if use_cache and cache_path:
        _write_cache(df, cache_path)

    return (df, df.attrs["load_report"]) if return_report else df


# ---------------------------------------------------------------------------
# Parquet cache
# ---------------------------------------------------------------------------

_PYARROW_HINT = (
    "Parquet caching needs pyarrow (`pip install pyarrow`, already listed in "
    "requirements.txt). Pass use_cache=False to skip caching entirely."
)


def _read_cache(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"{_PYARROW_HINT} Original error: {exc}") from exc


def _write_cache(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # pandas serializes DataFrame.attrs into the parquet metadata as JSON, and
    # LoadReport is not JSON-serializable. Write from a view with attrs cleared
    # rather than let the whole cache write fail (or degrade to a warning that
    # silently discards the other attrs).
    to_write = df.copy(deep=False)
    to_write.attrs = {}
    try:
        to_write.to_parquet(path, index=False)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"{_PYARROW_HINT} Original error: {exc}") from exc


__all__ = [
    "ACI",
    "CIC",
    "LoadReport",
    "class_order",
    "load_aci_iot",
    "load_cic_iot",
    "load_config",
    "load_dataset",
    "normalize_labels",
]
