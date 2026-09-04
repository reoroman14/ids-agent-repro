"""
Cleaning, encoding, and normalization shared across both datasets.

Order of operations matters, and the pipeline is deliberately shaped around
one constraint: anything fitted from the data must be fitted on TRAIN only.

    clean(df, config)                     split-independent transforms only
      |
    train_val_test_split(df, config)      applies the configured split policy
      |
    encode_categoricals(train) -> categories \\  fitted on train,
    normalize_numeric(train)   -> transform   /  reused for val and test

``clean`` therefore does not impute: median imputation is fitted on train and
lives in ``normalize_numeric`` alongside the scaler, since the two must be fit
together and applied in a fixed order (impute, then scale).

``prepare_splits`` wires the whole chain and is the single call intended for
``src/classifiers/train_eval.py``.

All policy comes from config — ``split`` and ``cleaning`` in
config/datasets.yaml, ``preprocessing`` in config/models.yaml. Nothing here
hardcodes a ratio, a seed, or a class name.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from . import feature_schema as fs
from .loader import class_order

LABEL_COLUMNS = ("label", "label_group", "label_binary")

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


# ---------------------------------------------------------------------------
# Reports and fitted state
# ---------------------------------------------------------------------------

@dataclass
class SplitReport:
    """Which class went where, and why. Printed rather than inferred."""

    dataset: Optional[str] = None
    seed: int = 0
    ratios: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stratified: bool = True
    min_rows_for_split: int = 0
    rare_class_policy: str = "train_only"

    n_total: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    normal_classes: List[str] = field(default_factory=list)
    rare_classes: Dict[str, int] = field(default_factory=dict)
    holdout_classes: Dict[str, int] = field(default_factory=dict)
    dropped_classes: Dict[str, int] = field(default_factory=dict)
    classes_absent_from_test: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        tr, va, te = self.ratios
        lines = [
            f"SplitReport({self.dataset})",
            f"  seed / ratios         : {self.seed} / {tr:.2f}-{va:.2f}-{te:.2f}"
            f"{' stratified' if self.stratified else ''}",
            f"  rows total            : {self.n_total:,}",
            f"  train / val / test    : {self.n_train:,} / {self.n_val:,} / {self.n_test:,}",
            f"  classes split normally: {len(self.normal_classes)}",
        ]
        if self.rare_classes:
            lines.append(
                f"  rare (<{self.min_rows_for_split} rows, policy "
                f"'{self.rare_class_policy}'):"
            )
            for cls, n in sorted(self.rare_classes.items()):
                lines.append(f"    {cls}: {n:,} rows")
        if self.holdout_classes:
            lines.append("  held out (test only, unseen in training):")
            for cls, n in sorted(self.holdout_classes.items()):
                lines.append(f"    {cls}: {n:,} rows")
        if self.dropped_classes:
            lines.append("  dropped entirely:")
            for cls, n in sorted(self.dropped_classes.items()):
                lines.append(f"    {cls}: {n:,} rows")
        if self.classes_absent_from_test:
            lines.append("  absent from test (excluded from test-set metrics):")
            for cls, why in sorted(self.classes_absent_from_test.items()):
                lines.append(f"    {cls}: {why}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


@dataclass
class NumericTransform:
    """
    Median imputer + scaler, fitted together on train.

    Bundled because both are fitted on the same rows and must be applied in a
    fixed order: impute, then scale. Fitting the scaler before imputation would
    let NaN handling shift the scale statistics.
    """

    columns: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)
    scaler: Any = None
    scaler_kind: str = "standard"
    degenerate_medians: List[str] = field(default_factory=list)


@dataclass
class Splits:
    """Everything ``train_eval.py`` needs, and nothing it has to recompute."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray

    feature_names: List[str]
    classes_: List[str]
    target: str
    dataset: Optional[str] = None
    categories: Dict[str, List[Any]] = field(default_factory=dict)
    numeric_transform: Optional[NumericTransform] = None
    report: Optional[SplitReport] = None

    def summary(self) -> str:
        lines = [
            f"Splits({self.dataset}, target={self.target!r})",
            f"  features              : {len(self.feature_names)}",
            f"  X_train / X_val / X_test: {self.X_train.shape} / "
            f"{self.X_val.shape} / {self.X_test.shape}",
            f"  classes in encoding   : {len(self.classes_)}",
            f"  classes seen in train : {len(np.unique(self.y_train))}",
            f"  classes seen in test  : {len(np.unique(self.y_test))}",
        ]
        if self.numeric_transform is not None:
            lines.append(f"  scaler                : {self.numeric_transform.scaler_kind}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------

def _dataset_of(df: pd.DataFrame, dataset: Optional[str]) -> Optional[str]:
    return dataset or df.attrs.get("dataset")

def _require_dataset(df: pd.DataFrame, dataset: Optional[str]) -> str:
    ds = _dataset_of(df, dataset)
    if ds is None:
        raise ValueError(
            "Dataset name unknown. The loader sets df.attrs['dataset']; if the "
            "frame was rebuilt or read from a plain file, pass dataset= "
            "explicitly."
        )
    return ds


def _split_ratios(split_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    tr = float(split_cfg.get("train_ratio", 0.7))
    va = float(split_cfg.get("val_ratio", 0.15))
    te = float(split_cfg.get("test_ratio", 0.15))
    total = tr + va + te
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"split ratios must sum to 1.0, got train={tr} + val={va} + "
            f"test={te} = {total}. Fix `split` in config/datasets.yaml."
        )
    for name, value in (("train", tr), ("val", va), ("test", te)):
        if value <= 0:
            raise ValueError(f"{name}_ratio must be > 0, got {value}.")
    return tr, va, te


# ---------------------------------------------------------------------------
# 1. clean
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    *,
    dataset: Optional[str] = None,
    drop_identifiers: bool = True,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Split-independent cleaning. Safe to run on the full frame before splitting.

    Does NOT impute — median imputation is fitted on train and lives in
    ``normalize_numeric``.

    ``drop_duplicates`` defaults to False on purpose. Flood traffic legitimately
    produces identical feature vectors at this resolution, so de-duplicating
    would silently rewrite the class balance that ``per_class_cap`` was chosen
    to control. Turn it on only with that trade-off in mind.

    Zero-variance detection looks at the whole frame. That is a structural
    decision (a column carries no information at all), not a statistical one
    fitted from labels, so it does not leak test information in the way a
    scaler or imputer would.
    """
    ds = _dataset_of(df, dataset)
    cleaning = (config or {}).get("cleaning", {})
    out = df.copy()

    label_cols = [c for c in LABEL_COLUMNS if c in out.columns]
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns if c not in label_cols
    ]

    if cleaning.get("replace_inf_with_nan", True) and numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if drop_identifiers and ds is not None:
        ids = [c for c in fs.identifier_columns(ds) if c in out.columns]
        if ids:
            out = out.drop(columns=ids)

    if cleaning.get("drop_zero_variance", True):
        candidates = [c for c in out.columns if c not in label_cols]
        constant = [c for c in candidates if out[c].nunique(dropna=True) <= 1]
        if constant:
            out = out.drop(columns=constant)

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        if len(out) != before:
            warnings.warn(
                f"clean() dropped {before - len(out):,} duplicate rows; class "
                f"balance has changed.",
                stacklevel=2,
            )

    out.attrs.update(df.attrs)
    return out


# ---------------------------------------------------------------------------
# 2. train_val_test_split
# ---------------------------------------------------------------------------

def _classify_classes(
    counts: pd.Series,
    *,
    holdout: Sequence[str],
    min_rows: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition class names into (normal, rare, holdout), or raise on conflict."""
    holdout_set = set(holdout)
    below_min = {str(c) for c, n in counts.items() if n < min_rows}

    conflict = sorted(holdout_set & below_min)
    if conflict:
        detail = ", ".join(f"{c!r} ({int(counts[c]):,} rows)" for c in conflict)
        raise ValueError(
            f"Contradictory split policy for {detail}. `holdout_classes` sends a "
            f"class to test only, while falling under `min_rows_for_split` "
            f"({min_rows}) with rare_class_policy 'train_only' sends it to train "
            f"only. A class cannot be both. Either remove it from "
            f"`holdout_classes` or lower `min_rows_for_split` below its row "
            f"count — guessing here would silently produce a meaningless "
            f"zero-day result."
        )

    present_holdout = [str(c) for c in counts.index if str(c) in holdout_set]
    rare = sorted(below_min)
    normal = [
        str(c) for c in counts.index
        if str(c) not in holdout_set and str(c) not in below_min
    ]
    return normal, rare, present_holdout


def _check_stratifiable(
    counts: pd.Series, classes: Sequence[str], tr: float, va: float, te: float
) -> None:
    """A stratified two-stage split needs enough members per class in each stage."""
    need_temp = 2.0 / (va + te)          # >= 2 rows must survive into val+test
    need_train = 1.0 / tr                # >= 1 row must land in train
    required = int(np.ceil(max(need_temp, need_train)))
    too_small = {c: int(counts[c]) for c in classes if counts[c] < required}
    if too_small:
        detail = ", ".join(f"{c!r} ({n} rows)" for c, n in sorted(too_small.items()))
        raise ValueError(
            f"Stratified splitting needs at least {required} rows per class at "
            f"these ratios, but {detail} fall short. Raise "
            f"`min_rows_for_split` above their counts so they route through "
            f"`rare_class_policy` instead, or set `stratify: false`."
        )


def train_val_test_split(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    dataset: Optional[str] = None,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test according to ``config['split']``.

    Every class falls into exactly one of three buckets:

    ``holdout``  listed in ``holdout_classes`` -> 100% to test, absent from
                 train and val. This is the zero-day setup: the model meets the
                 class for the first time at evaluation.
    ``rare``     fewer than ``min_rows_for_split`` rows -> routed by
                 ``rare_class_policy`` (``train_only`` / ``drop`` / ``error``).
    ``normal``   everything else -> stratified split at the configured ratios.

    A class in both the holdout list and under the row threshold raises: the two
    policies point in opposite directions and picking one silently would
    invalidate the result.

    The seed from ``config['split']['seed']`` drives both stratified splits and
    the final shuffle, so the same frame and config always give the same split.
    A ``SplitReport`` is attached to each returned frame's ``.attrs``.
    """
    ds = _dataset_of(df, dataset)
    split_cfg = config.get("split", {})

    seed = int(split_cfg.get("seed", 42))
    tr, va, te = _split_ratios(split_cfg)
    stratify_on = bool(split_cfg.get("stratify", True))
    min_rows = int(split_cfg.get("min_rows_for_split", 0))
    policy = str(split_cfg.get("rare_class_policy", "train_only"))
    holdout = list(split_cfg.get("holdout_classes") or [])

    if policy not in ("train_only", "drop", "error"):
        raise ValueError(
            f"Unknown rare_class_policy {policy!r}. Expected 'train_only', "
            f"'drop' or 'error'."
        )
    if label_column not in df.columns:
        raise KeyError(
            f"Label column {label_column!r} not in the frame. Available label "
            f"columns: {[c for c in LABEL_COLUMNS if c in df.columns]}."
        )

    # A misspelled holdout class would silently hold out nothing.
    if holdout and ds is not None and ds in config:
        known = set(config[ds]["label_map"].values())
        unknown = sorted(set(holdout) - known)
        if unknown:
            raise ValueError(
                f"`holdout_classes` names {unknown}, which are not canonical "
                f"classes of {ds}. Expected names from `label_map` values in "
                f"config/datasets.yaml."
            )

    report = SplitReport(
        dataset=ds,
        seed=seed,
        ratios=(tr, va, te),
        stratified=stratify_on,
        min_rows_for_split=min_rows,
        rare_class_policy=policy,
        n_total=len(df),
    )

    counts = df[label_column].value_counts()
    normal, rare, present_holdout = _classify_classes(
        counts, holdout=holdout, min_rows=min_rows
    )

    missing_holdout = sorted(set(holdout) - set(present_holdout))
    if missing_holdout:
        report.warnings.append(
            f"holdout classes absent from this frame: {missing_holdout}"
        )

    report.normal_classes = normal
    report.rare_classes = {c: int(counts[c]) for c in rare}
    report.holdout_classes = {c: int(counts[c]) for c in present_holdout}

    if rare and policy == "error":
        raise ValueError(
            f"rare_class_policy is 'error' and these classes fall under "
            f"min_rows_for_split={min_rows}: {report.rare_classes}."
        )

    labels = df[label_column].astype(str)
    holdout_rows = df[labels.isin(present_holdout)]
    rare_rows = df[labels.isin(rare)]
    normal_rows = df[labels.isin(normal)]

    if normal_rows.empty:
        raise ValueError(
            "No classes remain for the stratified split; every class was routed "
            "to holdout or rare. Check `holdout_classes` and "
            "`min_rows_for_split`."
        )

    if stratify_on:
        _check_stratifiable(counts, normal, tr, va, te)

    y_normal = normal_rows[label_column] if stratify_on else None
    train_part, temp_part = train_test_split(
        normal_rows,
        train_size=tr,
        random_state=seed,
        shuffle=True,
        stratify=y_normal,
    )

    # val's share of what is left, derived from the configured ratios so that
    # changing them (80/10/10, say) needs no change here.
    val_share = va / (va + te)
    y_temp = temp_part[label_column] if stratify_on else None
    val_part, test_part = train_test_split(
        temp_part,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=y_temp,
    )

    train_parts = [train_part]
    if len(rare_rows):
        if policy == "train_only":
            train_parts.append(rare_rows)
            for cls in rare:
                report.classes_absent_from_test[cls] = (
                    f"only {int(counts[cls]):,} rows (< min_rows_for_split="
                    f"{min_rows}); routed to train only"
                )
        elif policy == "drop":
            report.dropped_classes = {c: int(counts[c]) for c in rare}
            for cls in rare:
                report.classes_absent_from_test[cls] = "dropped by rare_class_policy"

    train = pd.concat(train_parts) if len(train_parts) > 1 else train_part
    val = val_part
    test = pd.concat([test_part, holdout_rows]) if len(holdout_rows) else test_part

    for cls in present_holdout:
        report.classes_absent_from_test.pop(cls, None)

    # Shuffle so appended rare / holdout rows are not a contiguous tail block.
    train = train.sample(frac=1, random_state=seed)
    val = val.sample(frac=1, random_state=seed)
    test = test.sample(frac=1, random_state=seed)

    _assert_partition(df, train, val, test, report, policy)

    report.n_train, report.n_val, report.n_test = len(train), len(val), len(test)
    for part in (train, val, test):
        part.attrs.update(df.attrs)
        part.attrs["split_report"] = report

    return train, val, test


def _assert_partition(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    report: SplitReport,
    policy: str,
) -> None:
    """The splits must be disjoint, and must account for every row."""
    i_tr, i_va, i_te = set(train.index), set(val.index), set(test.index)
    for a, b, names in (
        (i_tr, i_va, "train/val"),
        (i_tr, i_te, "train/test"),
        (i_va, i_te, "val/test"),
    ):
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"{names} splits overlap on {len(overlap)} rows — the split is "
                f"invalid and any metric computed from it would be inflated."
            )

    accounted = len(train) + len(val) + len(test)
    expected = len(df) - sum(report.dropped_classes.values())
    if accounted != expected:
        raise AssertionError(
            f"split lost rows: {accounted:,} across train/val/test but "
            f"{expected:,} expected ({len(df):,} input minus "
            f"{sum(report.dropped_classes.values()):,} dropped by policy "
            f"{policy!r})."
        )


# ---------------------------------------------------------------------------
# 3. encode_categoricals
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    *,
    dataset: Optional[str] = None,
    categories: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    """
    One-hot the schema's categorical feature columns.

    Fit by calling with ``categories=None`` on train; pass the returned mapping
    when transforming val and test so the column set and its order stay
    identical across splits. A value unseen at fit time yields an all-zero row
    across that column's dummies and a warning, rather than a new column that
    would silently misalign the matrices.

    On ACI this handles ``Connection Type`` (wired / wireless). CIC has no
    categorical features, so this is a no-op there.
    """
    ds = _dataset_of(df, dataset)
    out = df.copy()

    cat_cols = (
        [c for c in fs.categorical_columns(ds) if c in out.columns] if ds else []
    )
    fitting = categories is None
    categories = {} if fitting else dict(categories)

    for col in cat_cols:
        if fitting:
            categories[col] = sorted(out[col].dropna().astype(str).unique().tolist())
        known = categories[col]
        values = out[col].astype(str)

        if not fitting:
            unseen = sorted(set(values.dropna().unique()) - set(known))
            if unseen:
                warnings.warn(
                    f"{col!r} has values unseen at fit time: {unseen}. They "
                    f"encode as all-zero dummies.",
                    stacklevel=2,
                )

        for value in known:
            out[f"{col}={value}"] = (values == value).astype("uint8")
        out = out.drop(columns=[col])

    # Columns present in the fitted mapping but missing from this frame.
    for col, known in categories.items():
        for value in known:
            name = f"{col}={value}"
            if name not in out.columns:
                out[name] = np.uint8(0)

    out.attrs.update(df.attrs)
    return out, categories


def one_hot_columns(categories: Dict[str, List[Any]]) -> List[str]:
    """Names of the dummy columns produced by ``encode_categoricals``."""
    return [f"{col}={value}" for col, values in categories.items() for value in values]


# ---------------------------------------------------------------------------
# 4. normalize_numeric
# ---------------------------------------------------------------------------

def normalize_numeric(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    models_config: Optional[Dict[str, Any]] = None,
    *,
    fitted: Optional[NumericTransform] = None,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, NumericTransform]:
    """
    Median-impute then scale the numeric feature columns.

    Fit by calling with ``fitted=None`` on train; pass the returned
    ``NumericTransform`` when transforming val and test. Both the medians and
    the scaler statistics come from train alone — ``config/datasets.yaml`` sets
    ``impute_fit_on: train``, and computing either on the full frame would leak
    test distribution into training.

    The scaler comes from ``preprocessing.scaler`` in config/models.yaml
    (``standard`` / ``minmax`` / ``robust`` / ``none``).
    """
    cleaning = (config or {}).get("cleaning", {})
    pre_cfg = (models_config or {}).get("preprocessing", {})

    if cleaning.get("impute_fit_on", "train") != "train":
        warnings.warn(
            "impute_fit_on is not 'train'; this function only ever fits on the "
            "frame it is given, so pass the train split.",
            stacklevel=2,
        )
    strategy = cleaning.get("impute_strategy", "median")
    if strategy != "median":
        raise NotImplementedError(
            f"impute_strategy={strategy!r} is not implemented; only 'median' is."
        )

    out = df.copy()
    fitting = fitted is None

    if fitting:
        if columns is None:
            label_cols = set(LABEL_COLUMNS)
            columns = [
                c for c in out.select_dtypes(include=[np.number]).columns
                if c not in label_cols
            ]
        kind = str(pre_cfg.get("scaler", "standard")).lower()
        if kind not in _SCALERS:
            raise ValueError(
                f"Unknown scaler {kind!r}. Expected one of {sorted(_SCALERS)}. "
                f"Set `preprocessing.scaler` in config/models.yaml."
            )
        fitted = NumericTransform(columns=list(columns), scaler_kind=kind)
    else:
        missing = [c for c in fitted.columns if c not in out.columns]
        if missing:
            raise KeyError(
                f"Columns fitted on train are missing from this frame: "
                f"{missing[:10]}. Transform the splits with the same fitted "
                f"object produced from train."
            )

    cols = fitted.columns
    if not cols:
        return out, fitted

    if fitting:
        medians = out[cols].median(numeric_only=True)
        degenerate = [c for c in cols if pd.isna(medians.get(c, np.nan))]
        if degenerate:
            # An all-NaN column in train has no median; 0.0 keeps the matrix
            # finite and the column is named so it can be dropped deliberately.
            warnings.warn(
                f"no median available (all-NaN in the fitting split) for "
                f"{degenerate}; imputing 0.0.",
                stacklevel=2,
            )
        fitted.medians = {
            c: (0.0 if pd.isna(medians.get(c, np.nan)) else float(medians[c]))
            for c in cols
        }
        fitted.degenerate_medians = degenerate

    out[cols] = out[cols].fillna(value=fitted.medians)

    scaler_cls = _SCALERS[fitted.scaler_kind]
    if scaler_cls is not None:
        if fitting:
            fitted.scaler = scaler_cls()
            fitted.scaler.fit(out[cols])
        out[cols] = fitted.scaler.transform(out[cols])

    out.attrs.update(df.attrs)
    return out, fitted


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(
    y: pd.Series, dataset: str, config: Dict[str, Any], *, target: str = "label"
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode labels to ints using the frozen ``class_order`` from config.

    Not sklearn's ``LabelEncoder``: that fits alphabetically on whatever classes
    happen to be present, so an integer's meaning would shift between a run that
    holds a class out and one that does not. The pinned order keeps label -> int
    identical across runs, machines and splits.

    ``label_group`` and ``label_binary`` have no pinned order, so those fall back
    to sorted unique values, which is stable for a fixed group vocabulary.
    """
    if target == "label":
        classes = class_order(dataset, config)
    else:
        groups = config[dataset]["label_groups"]
        classes = (
            sorted(set(groups.values())) if target == "label_group" else ["Attack", "Benign"]
        )

    index = {name: i for i, name in enumerate(classes)}
    unknown = sorted(set(y.astype(str).unique()) - set(index))
    if unknown:
        raise ValueError(
            f"Values {unknown} in {target!r} are absent from the frozen class "
            f"list for {dataset}. Update config/datasets.yaml rather than "
            f"encoding them ad hoc."
        )
    return y.astype(str).map(index).to_numpy(dtype=np.int64), classes


def decode_labels(y: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Inverse of :func:`encode_labels`."""
    lookup = np.asarray(classes, dtype=object)
    return lookup[np.asarray(y, dtype=int)]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def prepare_splits(
    df: pd.DataFrame,
    config: Dict[str, Any],
    models_config: Optional[Dict[str, Any]] = None,
    *,
    dataset: Optional[str] = None,
    target: str = "label",
    include_high_leakage: bool = True,
    drop_duplicates: bool = False,
) -> Splits:
    """
    Run the whole chain: clean -> split -> fit on train -> transform all three.

    This is the single entry point for ``src/classifiers/train_eval.py``.

    ``include_high_leakage`` defaults to True, matching the Phase 1 decision to
    keep ``Src Port`` / ``Dst Port`` and the ``Idle*`` group. Pass False for the
    conservative feature set; the ``critical`` leakage columns (IPs, Flow ID,
    Timestamp) are excluded either way.

    The split is always computed on the fine-grained ``label``, whatever
    ``target`` is set to. Splitting on ``label_binary`` instead would put every
    row in a two-class stratification, so the rare-class and holdout policies
    would never fire and the 34-class and 2-class runs would sit on different
    row partitions — not comparable. ``target`` selects y only.
    """
    if target not in LABEL_COLUMNS:
        raise ValueError(f"target must be one of {LABEL_COLUMNS}, got {target!r}.")

    ds = _require_dataset(df, dataset)
    cleaned = clean(df, config, dataset=ds, drop_duplicates=drop_duplicates)
    train, val, test = train_val_test_split(
        cleaned, config, dataset=ds, label_column="label"
    )
    report = train.attrs.get("split_report")

    # Fit on train, then apply the same fitted objects to val and test.
    train, categories = encode_categoricals(train, config, dataset=ds)
    val, _ = encode_categoricals(val, config, dataset=ds, categories=categories)
    test, _ = encode_categoricals(test, config, dataset=ds, categories=categories)

    allowed = set(
        fs.feature_columns(ds, include_high_leakage=include_high_leakage)
    ) | set(one_hot_columns(categories))
    feature_names = [c for c in train.columns if c in allowed]
    if not feature_names:
        raise ValueError(f"No feature columns survived for {ds}.")

    dummies = set(one_hot_columns(categories))
    scale_one_hot = bool(
        (models_config or {}).get("preprocessing", {}).get("scale_one_hot", False)
    )
    numeric_cols = [
        c for c in feature_names if scale_one_hot or c not in dummies
    ]

    train, fitted = normalize_numeric(
        train, config, models_config, columns=numeric_cols
    )
    val, _ = normalize_numeric(val, config, models_config, fitted=fitted)
    test, _ = normalize_numeric(test, config, models_config, fitted=fitted)

    y_train, classes = encode_labels(train[target], ds, config, target=target)
    y_val, _ = encode_labels(val[target], ds, config, target=target)
    y_test, _ = encode_labels(test[target], ds, config, target=target)

    return Splits(
        X_train=train[feature_names],
        y_train=y_train,
        X_val=val[feature_names],
        y_val=y_val,
        X_test=test[feature_names],
        y_test=y_test,
        feature_names=feature_names,
        classes_=classes,
        target=target,
        dataset=ds,
        categories=categories,
        numeric_transform=fitted,
        report=report,
    )


__all__ = [
    "LABEL_COLUMNS",
    "NumericTransform",
    "SplitReport",
    "Splits",
    "clean",
    "decode_labels",
    "encode_categoricals",
    "encode_labels",
    "normalize_numeric",
    "one_hot_columns",
    "prepare_splits",
    "train_val_test_split",
]
