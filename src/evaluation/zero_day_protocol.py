"""
Choosing zero-day holdouts by behavioural similarity rather than by the
dataset's attack-family labels.

Why this module exists
----------------------
The first zero-day protocol withheld whole taxonomic families and assumed that
made the task hard. Measurement showed the assumption is wrong. Withholding all
twelve of CIC's DDoS classes scores an AUC of 0.998, because the DoS and Mirai
families are also floods and stay in training. Withholding ACI's five DoS
classes scores 0.594, because that removes flooding behaviour entirely. The
taxonomy label is not what determines difficulty; whether a behaviourally
similar attack survives in training is.

A protocol built on family labels therefore measures something close to
arbitrary, and flatters the model whenever the taxonomy happens to split
behaviourally redundant classes across families.

What this module does instead
-----------------------------
1. Represent each class by its centroid in the standardised feature space.
2. Take the distance between centroids as behavioural similarity.
3. Cluster classes on that distance to obtain *behavioural* families.
4. For any candidate holdout, report the distance from the withheld classes to
   the nearest attack class still in training.

That last number is the design variable. It predicts difficulty, so a protocol
can deliberately span it instead of stumbling across it, and results can be
reported as a curve against it rather than as a single figure whose difficulty
is unstated.

The centroid distance is deliberately a simple, a-priori choice — standardised
features, Euclidean distance, Ward linkage — rather than something tuned until
it correlated well. :func:`validate_distance_predicts_difficulty` exists to
check it against measured AUCs honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BENIGN = "Benign"


@dataclass
class BehaviouralSpace:
    """Class centroids and the distances between them."""

    class_names: List[str]
    centroids: pd.DataFrame          # index: class name, columns: features
    distances: pd.DataFrame          # square, symmetric, index/columns: class name
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def attack_classes(self) -> List[str]:
        return [c for c in self.centroids.index if c != BENIGN]

    def nearest_neighbours(self, cls: str, k: int = 3) -> List[Tuple[str, float]]:
        """The k behaviourally closest classes to ``cls``."""
        row = self.distances.loc[cls].drop(index=cls)
        return [(str(name), float(value)) for name, value in row.nsmallest(k).items()]

    def distance_to_benign(self, cls: str) -> float:
        return float(self.distances.loc[cls, BENIGN])


def build_behavioural_space(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: Sequence[str],
    *,
    min_rows: int = 30,
) -> BehaviouralSpace:
    """
    Build the behavioural space from a training split.

    ``X`` must already be scaled — pass ``Splits.X_train``, whose scaler was fit
    on train only. Unscaled features would let a single wide-ranged column
    dominate every distance.

    Classes with fewer than ``min_rows`` examples are excluded: a centroid from
    a handful of rows is noise, and ACI's 5-row class would otherwise anchor a
    cluster on nothing.
    """
    class_names = [str(c) for c in class_names]
    frame = pd.DataFrame(np.asarray(X), columns=list(X.columns))
    labels = np.asarray(y)

    rows, counts = {}, {}
    for index, name in enumerate(class_names):
        mask = labels == index
        n = int(mask.sum())
        if n < min_rows:
            continue
        rows[name] = frame.loc[mask].mean(axis=0)
        counts[name] = n

    if len(rows) < 2:
        raise ValueError(
            f"need at least two classes with >= {min_rows} rows to build a "
            f"behavioural space; found {len(rows)}."
        )

    centroids = pd.DataFrame(rows).T
    values = centroids.to_numpy(dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    matrix = np.sqrt((diff ** 2).sum(axis=-1))
    distances = pd.DataFrame(matrix, index=centroids.index, columns=centroids.index)

    return BehaviouralSpace(
        class_names=list(centroids.index),
        centroids=centroids,
        distances=distances,
        counts=counts,
    )


def behavioural_clusters(
    space: BehaviouralSpace, n_clusters: int = 4, *, method: str = "ward"
) -> Dict[str, List[str]]:
    """
    Cluster attack classes by behaviour into ``n_clusters`` groups.

    Benign is excluded — it is the thing being distinguished from, not one of
    the behaviours being grouped.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    attacks = space.attack_classes
    if len(attacks) <= n_clusters:
        return {f"B{i + 1}": [c] for i, c in enumerate(attacks)}

    # .copy(): pandas may hand back a read-only view, which fill_diagonal rejects.
    sub = space.distances.loc[attacks, attacks].to_numpy(dtype=float).copy()
    np.fill_diagonal(sub, 0.0)
    sub = (sub + sub.T) / 2.0  # guard against float asymmetry
    tree = linkage(squareform(sub, checks=False), method=method)
    assignments = fcluster(tree, t=n_clusters, criterion="maxclust")

    clusters: Dict[str, List[str]] = {}
    for name, group in zip(attacks, assignments):
        clusters.setdefault(f"B{int(group)}", []).append(name)
    return dict(sorted(clusters.items()))


def nearest_remaining_distance(
    space: BehaviouralSpace, holdout: Sequence[str]
) -> Optional[float]:
    """
    Distance from the withheld classes to the closest attack class still in
    training.

    This is the difficulty variable. A small value means the model has a near
    neighbour to generalise from and the task is easy; a large value means the
    withheld behaviour is genuinely unrepresented.

    Returns None when every attack class is withheld, or when none of the
    withheld classes is in the space (all too small to have a centroid).
    """
    held = [c for c in holdout if c in space.distances.index]
    remaining = [c for c in space.attack_classes if c not in set(holdout)]
    if not held or not remaining:
        return None
    return float(space.distances.loc[held, remaining].to_numpy().min())


def describe_holdout(
    space: BehaviouralSpace, holdout: Sequence[str]
) -> Dict[str, Any]:
    """Summarise a candidate holdout: its size, isolation, and closest survivor."""
    held = [c for c in holdout if c in space.distances.index]
    remaining = [c for c in space.attack_classes if c not in set(holdout)]
    out: Dict[str, Any] = {
        "holdout": list(holdout),
        "n_classes": len(holdout),
        "n_rows": int(sum(space.counts.get(c, 0) for c in holdout)),
        "nearest_remaining_distance": nearest_remaining_distance(space, holdout),
        "nearest_remaining_class": None,
        "n_remaining_attack_classes": len(remaining),
    }
    if held and remaining:
        block = space.distances.loc[held, remaining]
        flat = block.to_numpy()
        position = np.unravel_index(np.argmin(flat), flat.shape)
        out["nearest_remaining_class"] = str(block.columns[position[1]])
        out["closest_withheld_class"] = str(block.index[position[0]])
    return out


def validate_distance_predicts_difficulty(
    records: Sequence[Mapping[str, Any]],
    *,
    distance_key: str = "nearest_remaining_distance",
    score_key: str = "auc",
) -> Dict[str, Any]:
    """
    Check honestly whether the distance actually predicts measured difficulty.

    ``records`` should pair a holdout's distance with its measured AUC. Returns
    Pearson and Spearman correlations and the fitted slope. A weak result here
    means the similarity measure is wrong and the protocol should not be built
    on it — this function exists so that outcome is visible rather than
    quietly assumed away.
    """
    pairs = [
        (float(r[distance_key]), float(r[score_key]))
        for r in records
        if r.get(distance_key) is not None and r.get(score_key) is not None
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": float("nan"), "spearman": float("nan")}

    distance = np.array([p[0] for p in pairs])
    score = np.array([p[1] for p in pairs])
    order_d = pd.Series(distance).rank().to_numpy()
    order_s = pd.Series(score).rank().to_numpy()
    slope, intercept = np.polyfit(distance, score, 1)

    return {
        "n": len(pairs),
        "pearson": float(np.corrcoef(distance, score)[0, 1]),
        "spearman": float(np.corrcoef(order_d, order_s)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
        "distance_range": (float(distance.min()), float(distance.max())),
        "score_range": (float(score.min()), float(score.max())),
    }


def design_protocol(
    space: BehaviouralSpace,
    *,
    n_clusters: int = 4,
    n_singletons: int = 3,
) -> List[Dict[str, Any]]:
    """
    Propose a set of holdouts that deliberately spans the difficulty range.

    Includes each behavioural cluster (hard end: a whole behaviour removed) and
    the least isolated individual classes (easy end: a near-duplicate survives),
    so results can be reported as a curve against isolation rather than as one
    number whose difficulty is unstated.
    """
    proposals: List[Dict[str, Any]] = []

    for name, members in behavioural_clusters(space, n_clusters).items():
        entry = describe_holdout(space, members)
        entry.update({"label": f"behavioural cluster {name} ({len(members)})",
                      "regime": "behavioural-cluster"})
        proposals.append(entry)

    singles = [describe_holdout(space, [c]) for c in space.attack_classes]
    singles = [s for s in singles if s["nearest_remaining_distance"] is not None]
    singles.sort(key=lambda s: s["nearest_remaining_distance"])
    for entry in singles[:n_singletons]:
        entry.update({"label": f"{entry['holdout'][0]} (single)",
                      "regime": "single-class"})
        proposals.append(entry)
    for entry in singles[-n_singletons:]:
        if entry not in proposals:
            entry = dict(entry)
            entry.update({"label": f"{entry['holdout'][0]} (single, isolated)",
                          "regime": "single-class"})
            proposals.append(entry)

    return proposals


__all__ = [
    "BehaviouralSpace",
    "behavioural_clusters",
    "build_behavioural_space",
    "describe_holdout",
    "design_protocol",
    "nearest_remaining_distance",
    "validate_distance_predicts_difficulty",
]
