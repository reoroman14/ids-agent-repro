"""
Explicit feature schema.

Purpose: from Phase 1 onward, every feature is tagged with where it
originates and whether it will later be attacker-controllable
(e.g. telemetry fields that Phase 2's injection surface will target).
This file is intentionally created now so Phase 2/3 do not require
restructuring the data layer.

Every column of both datasets is listed. The two datasets share no column
names — ACI ships 85 CICFlowMeter netflow columns, CIC ships 40 packet-level
aggregates — so each has its own table and models are trained per dataset.

Field meanings
--------------
role
    ``identifier``  Not a feature. Excluded from X unconditionally.
    ``feature``     Model input.
    ``label``       Target column.
role is about what the column *is*; ``leakage_risk`` is about what it would
do to the results if used.

leakage_risk
    ``critical``  Would trivially reveal the label in a single-testbed
                  capture. Never include.
    ``high``      Plausibly encodes the capture setup rather than attack
                  behaviour. Excluded by default; opt in deliberately.
    ``none``      Ordinary behavioural feature.

attacker_controllable
    False everywhere in Phase 1, by design. Phase 2 flips these for the
    telemetry fields an attacker can influence; the structure does not change.

Flags below (``zero_variance``, ``has_inf``, ``has_nan``, ``suspect``) were
measured from the local raw files, not assumed. ACI figures come from a full
pass over all 1,231,411 rows; CIC figures from a six-file spread sample.
"""

from typing import Any, Dict, List

ACI = "aci_iot_2023"
CIC = "cic_iot_2023"


def _spec(
    dtype: str,
    role: str,
    source: str,
    *,
    leakage_risk: str = "none",
    attacker_controllable: bool = False,
    manipulation_cost: str = "n/a",
    zero_variance: bool = False,
    has_inf: bool = False,
    has_nan: bool = False,
    suspect: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "dtype": dtype,
        "role": role,
        "source": source,
        "leakage_risk": leakage_risk,
        "attacker_controllable": attacker_controllable,
        "manipulation_cost": manipulation_cost,
        "zero_variance": zero_variance,
        "has_inf": has_inf,
        "has_nan": has_nan,
        "suspect": suspect,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# ACI-IoT-2023 — 85 columns (83 numeric + Label + Connection Type)
# ---------------------------------------------------------------------------
# Sources:
#   flow_key      the 5-tuple / flow identity written by CICFlowMeter
#   flow_record   a statistic CICFlowMeter computed over the flow
#   capture_meta  a property of the testbed capture, not of the traffic

_ACI_SCHEMA: Dict[str, Dict[str, Any]] = {
    # -- flow identity, in raw file order -------------------------------------
    # Insertion order here IS the expected file header: all_columns() is used to
    # validate every CSV before it is read. Keep these in file order, not
    # thematic order.
    "Flow ID": _spec(
        "str", "identifier", "flow_key", leakage_risk="critical",
        notes="Concatenation of the 5-tuple; uniquely keys the flow.",
    ),
    "Src IP": _spec(
        "str", "identifier", "flow_key", leakage_risk="critical",
        notes="In a single-testbed capture the attacker host address nearly "
              "perfectly predicts the label. Including it inflates F1 toward "
              "1.0 and silently invalidates the reproduction.",
    ),
    "Src Port": _spec(
        "int64", "feature", "flow_key", leakage_risk="high",
        notes="Kept as a feature deliberately: port behaviour is what defines "
              "the Port Scan and OS Scan classes. Ephemeral source ports may "
              "still encode host identity — report results with and without "
              "if contested.",
    ),
    "Dst IP": _spec(
        "str", "identifier", "flow_key", leakage_risk="critical",
        notes="Victim host address; same leakage argument as Src IP.",
    ),
    "Dst Port": _spec(
        "int64", "feature", "flow_key", leakage_risk="high",
        notes="Kept deliberately; see Src Port. Destination service port is "
              "genuinely discriminative for scan and flood classes.",
    ),
    "Protocol": _spec("int64", "feature", "flow_key"),
    "Timestamp": _spec(
        "str", "identifier", "capture_meta", leakage_risk="critical",
        notes="Attack phases were captured in contiguous time windows, so "
              "wall-clock time separates the classes on its own.",
    ),

    # -- basic flow shape -----------------------------------------------------
    "Flow Duration": _spec("int64", "feature", "flow_record"),
    "Total Fwd Packet": _spec("int64", "feature", "flow_record"),
    "Total Bwd packets": _spec("int64", "feature", "flow_record"),
    "Total Length of Fwd Packet": _spec("float64", "feature", "flow_record"),
    "Total Length of Bwd Packet": _spec("float64", "feature", "flow_record"),

    # -- per-direction packet length statistics ------------------------------
    "Fwd Packet Length Max": _spec("float64", "feature", "flow_record"),
    "Fwd Packet Length Min": _spec("float64", "feature", "flow_record"),
    "Fwd Packet Length Mean": _spec("float64", "feature", "flow_record"),
    "Fwd Packet Length Std": _spec("float64", "feature", "flow_record"),
    "Bwd Packet Length Max": _spec("float64", "feature", "flow_record"),
    "Bwd Packet Length Min": _spec("float64", "feature", "flow_record"),
    "Bwd Packet Length Mean": _spec("float64", "feature", "flow_record"),
    "Bwd Packet Length Std": _spec("float64", "feature", "flow_record"),

    # -- rates: division by a zero-length flow yields inf --------------------
    "Flow Bytes/s": _spec(
        "float64", "feature", "flow_record", has_inf=True, has_nan=True,
        notes="915 inf and 1,009 NaN over the full file.",
    ),
    "Flow Packets/s": _spec(
        "float64", "feature", "flow_record", has_inf=True,
        notes="1,924 inf over the full file.",
    ),

    # -- inter-arrival times -------------------------------------------------
    "Flow IAT Mean": _spec("float64", "feature", "flow_record"),
    "Flow IAT Std": _spec("float64", "feature", "flow_record"),
    "Flow IAT Max": _spec("float64", "feature", "flow_record"),
    "Flow IAT Min": _spec("float64", "feature", "flow_record"),
    "Fwd IAT Total": _spec("float64", "feature", "flow_record"),
    "Fwd IAT Mean": _spec("float64", "feature", "flow_record"),
    "Fwd IAT Std": _spec("float64", "feature", "flow_record"),
    "Fwd IAT Max": _spec("float64", "feature", "flow_record"),
    "Fwd IAT Min": _spec("float64", "feature", "flow_record"),
    "Bwd IAT Total": _spec("float64", "feature", "flow_record"),
    "Bwd IAT Mean": _spec("float64", "feature", "flow_record"),
    "Bwd IAT Std": _spec("float64", "feature", "flow_record"),
    "Bwd IAT Max": _spec("float64", "feature", "flow_record"),
    "Bwd IAT Min": _spec("float64", "feature", "flow_record"),

    # -- per-direction flags / headers ---------------------------------------
    "Fwd PSH Flags": _spec("int64", "feature", "flow_record"),
    "Bwd PSH Flags": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Fwd URG Flags": _spec(
        "int64", "feature", "flow_record",
        notes="Constant in a 300k-row head sample but NOT over the full file — "
              "do not drop on sample evidence.",
    ),
    "Bwd URG Flags": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Fwd Header Length": _spec("int64", "feature", "flow_record"),
    "Bwd Header Length": _spec("int64", "feature", "flow_record"),
    "Fwd Packets/s": _spec("float64", "feature", "flow_record"),
    "Bwd Packets/s": _spec("float64", "feature", "flow_record"),

    # -- aggregate packet length ---------------------------------------------
    "Packet Length Min": _spec("float64", "feature", "flow_record"),
    "Packet Length Max": _spec("float64", "feature", "flow_record"),
    "Packet Length Mean": _spec("float64", "feature", "flow_record"),
    "Packet Length Std": _spec("float64", "feature", "flow_record"),
    "Packet Length Variance": _spec("float64", "feature", "flow_record"),

    # -- TCP flag counts ------------------------------------------------------
    "FIN Flag Count": _spec("int64", "feature", "flow_record"),
    "SYN Flag Count": _spec("int64", "feature", "flow_record"),
    "RST Flag Count": _spec("int64", "feature", "flow_record"),
    "PSH Flag Count": _spec("int64", "feature", "flow_record"),
    "ACK Flag Count": _spec("int64", "feature", "flow_record"),
    "URG Flag Count": _spec(
        "int64", "feature", "flow_record",
        notes="Constant in a 300k-row head sample but NOT over the full file.",
    ),
    "CWR Flag Count": _spec(
        "int64", "feature", "flow_record",
        notes="Constant in a 300k-row head sample but NOT over the full file.",
    ),
    "ECE Flag Count": _spec(
        "int64", "feature", "flow_record",
        notes="Constant in a 300k-row head sample but NOT over the full file.",
    ),

    # -- ratios and segment sizes --------------------------------------------
    "Down/Up Ratio": _spec("float64", "feature", "flow_record"),
    "Average Packet Size": _spec("float64", "feature", "flow_record"),
    "Fwd Segment Size Avg": _spec("float64", "feature", "flow_record"),
    "Bwd Segment Size Avg": _spec("float64", "feature", "flow_record"),

    # -- bulk transfer statistics --------------------------------------------
    "Fwd Bytes/Bulk Avg": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Fwd Packet/Bulk Avg": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Fwd Bulk Rate Avg": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Bwd Bytes/Bulk Avg": _spec("int64", "feature", "flow_record"),
    "Bwd Packet/Bulk Avg": _spec("int64", "feature", "flow_record"),
    "Bwd Bulk Rate Avg": _spec("int64", "feature", "flow_record"),

    # -- subflow statistics ---------------------------------------------------
    "Subflow Fwd Packets": _spec("int64", "feature", "flow_record"),
    "Subflow Fwd Bytes": _spec("int64", "feature", "flow_record"),
    "Subflow Bwd Packets": _spec(
        "int64", "feature", "flow_record", zero_variance=True,
        notes="Single distinct value across all 1,231,411 rows.",
    ),
    "Subflow Bwd Bytes": _spec("int64", "feature", "flow_record"),

    # -- window / segment ------------------------------------------------------
    "FWD Init Win Bytes": _spec("int64", "feature", "flow_record"),
    "Bwd Init Win Bytes": _spec("int64", "feature", "flow_record"),
    "Fwd Act Data Pkts": _spec("int64", "feature", "flow_record"),
    "Fwd Seg Size Min": _spec("int64", "feature", "flow_record"),

    # -- active periods --------------------------------------------------------
    "Active Mean": _spec("float64", "feature", "flow_record"),
    "Active Std": _spec("float64", "feature", "flow_record"),
    "Active Max": _spec("float64", "feature", "flow_record"),
    "Active Min": _spec("float64", "feature", "flow_record"),

    # -- idle periods: known ACI defect ---------------------------------------
    # These hold raw epoch-microsecond timestamps (~1.7e15), not durations.
    # Because they are absolute capture times they also separate the classes by
    # when each attack phase was recorded, which is leakage, not behaviour.
    "Idle Mean": _spec(
        "float64", "feature", "flow_record", leakage_risk="high", suspect=True,
        notes="Holds an absolute epoch-microsecond timestamp (median 8.49e14, "
              "max 1.70e15), not an idle duration. Capture-time correlated.",
    ),
    "Idle Std": _spec(
        "float64", "feature", "flow_record", suspect=True,
        notes="Companion of the Idle Mean/Max/Min defect.",
    ),
    "Idle Max": _spec(
        "float64", "feature", "flow_record", leakage_risk="high", suspect=True,
        notes="Absolute epoch-microsecond timestamp, not a duration.",
    ),
    "Idle Min": _spec(
        "float64", "feature", "flow_record", leakage_risk="high", suspect=True,
        notes="Absolute epoch-microsecond timestamp, not a duration.",
    ),

    # -- label and testbed metadata -------------------------------------------
    "Label": _spec("str", "label", "annotation"),
    "Connection Type": _spec(
        "category", "feature", "capture_meta",
        notes="Testbed medium: wired (488,653) / wireless (742,758). "
              "One-hot encoded to two columns.",
    ),
}


# ---------------------------------------------------------------------------
# CIC-IoT-2023 — 40 columns (39 numeric + Label)
# ---------------------------------------------------------------------------
# No IP, port, or timestamp columns exist in this dataset, so it carries none
# of ACI's identity-leakage problem. Sources:
#   packet_header   read directly off packet headers
#   packet_agg      aggregated over the packet window

_CIC_SCHEMA: Dict[str, Dict[str, Any]] = {
    "Header_Length": _spec("float64", "feature", "packet_header"),
    "Protocol Type": _spec("int64", "feature", "packet_header"),
    "Time_To_Live": _spec("float64", "feature", "packet_header"),
    "Rate": _spec(
        "float64", "feature", "packet_agg", has_inf=True,
        notes="Packet rate; inf where the observation window has zero span.",
    ),

    # -- TCP flag indicators (fractions in [0, 1]) ----------------------------
    "fin_flag_number": _spec("float64", "feature", "packet_header"),
    "syn_flag_number": _spec("float64", "feature", "packet_header"),
    "rst_flag_number": _spec("float64", "feature", "packet_header"),
    "psh_flag_number": _spec("float64", "feature", "packet_header"),
    "ack_flag_number": _spec("float64", "feature", "packet_header"),
    "ece_flag_number": _spec("float64", "feature", "packet_header"),
    "cwr_flag_number": _spec("float64", "feature", "packet_header"),

    # -- TCP flag counts -------------------------------------------------------
    "ack_count": _spec("int64", "feature", "packet_agg"),
    "syn_count": _spec("int64", "feature", "packet_agg"),
    "fin_count": _spec("int64", "feature", "packet_agg"),
    "rst_count": _spec("int64", "feature", "packet_agg"),

    # -- application / transport protocol indicators --------------------------
    "HTTP": _spec("float64", "feature", "packet_header"),
    "HTTPS": _spec("float64", "feature", "packet_header"),
    "DNS": _spec("float64", "feature", "packet_header"),
    "Telnet": _spec("float64", "feature", "packet_header"),
    "SMTP": _spec("float64", "feature", "packet_header"),
    "SSH": _spec("float64", "feature", "packet_header"),
    "IRC": _spec("float64", "feature", "packet_header"),
    "TCP": _spec("float64", "feature", "packet_header"),
    "UDP": _spec("float64", "feature", "packet_header"),
    "DHCP": _spec("float64", "feature", "packet_header"),
    "ARP": _spec("float64", "feature", "packet_header"),
    "ICMP": _spec("float64", "feature", "packet_header"),
    "IGMP": _spec("float64", "feature", "packet_header"),
    "IPv": _spec("float64", "feature", "packet_header"),
    "LLC": _spec("float64", "feature", "packet_header"),

    # -- packet size statistics ------------------------------------------------
    "Tot sum": _spec("int64", "feature", "packet_agg"),
    "Min": _spec("int64", "feature", "packet_agg"),
    "Max": _spec("int64", "feature", "packet_agg"),
    "AVG": _spec("float64", "feature", "packet_agg"),
    "Std": _spec(
        "float64", "feature", "packet_agg", has_nan=True,
        notes="NaN where the window holds a single packet (no dispersion).",
    ),
    "Tot size": _spec("float64", "feature", "packet_agg"),
    "IAT": _spec("float64", "feature", "packet_agg"),
    "Number": _spec("int64", "feature", "packet_agg"),
    "Variance": _spec(
        "float64", "feature", "packet_agg", has_nan=True,
        notes="NaN alongside Std, for the same reason.",
    ),

    "Label": _spec("str", "label", "annotation"),
}


# ---------------------------------------------------------------------------
# Phase 2: the attacker-controllable surface
# ---------------------------------------------------------------------------
# Phase 1 left `attacker_controllable` False everywhere, as the file said it
# would. Phase 2 turns it on, and the honest answer is that an attacker who
# generates the traffic controls essentially every feature — these are
# statistics of packets the attacker sent.
#
# So the interesting question is not *whether* a feature can be changed but
# what changing it costs the attacker. A SYN flood that stops setting SYN flags
# has evaded detection by ceasing to be a SYN flood, which is not an attack.
# `manipulation_cost` records that:
#
#   low     free to change without affecting the attack: padding, TTL, rarely
#           used flag bits, small timing jitter.
#   medium  changeable but degrades throughput or stealth: packet counts,
#           rates, inter-arrival times.
#   high    definitional — changing it means abandoning the attack: the
#           transport protocol, and the flags that constitute the flood.
#
# An evasion result is only meaningful if it respects these. Perturbing a
# `high` feature produces a "successful evasion" that no real attacker would
# accept, which is how adversarial-ML papers end up overstating attack success.

_CIC_MANIPULATION_COST = {
    "Header_Length": "low", "Time_To_Live": "low",
    "ece_flag_number": "low", "cwr_flag_number": "low",
    "Tot sum": "low", "Min": "low", "Max": "low", "AVG": "low",
    "Std": "low", "Tot size": "low", "Variance": "low",
    "IAT": "medium", "Rate": "medium", "Number": "medium",
    "psh_flag_number": "medium", "ack_flag_number": "medium",
    "ack_count": "medium", "fin_flag_number": "medium", "fin_count": "medium",
    "rst_flag_number": "medium", "rst_count": "medium",
    "Protocol Type": "high", "syn_flag_number": "high", "syn_count": "high",
    "TCP": "high", "UDP": "high", "ICMP": "high", "IGMP": "high", "ARP": "high",
    "HTTP": "high", "HTTPS": "high", "DNS": "high", "Telnet": "high",
    "SMTP": "high", "SSH": "high", "IRC": "high", "DHCP": "high",
    "IPv": "high", "LLC": "high",
}

_ACI_MANIPULATION_COST_PREFIXES = {
    "low": ("Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min",
            "FWD Init Win Bytes", "Bwd Init Win Bytes", "CWR Flag Count",
            "ECE Flag Count", "URG Flag Count", "Fwd URG Flags", "Bwd URG Flags"),
    "high": ("Protocol", "SYN Flag Count", "Src Port", "Dst Port"),
}


def _apply_attack_surface() -> None:
    """Turn on Phase 2's attacker-controllable flags and cost labels."""
    for column, spec in _CIC_SCHEMA.items():
        if spec["role"] != "feature":
            continue
        spec["attacker_controllable"] = True
        spec["manipulation_cost"] = _CIC_MANIPULATION_COST.get(column, "medium")

    for column, spec in _ACI_SCHEMA.items():
        if spec["role"] != "feature":
            continue
        # Connection Type is a property of the testbed link, not of the traffic.
        if column == "Connection Type":
            spec["attacker_controllable"] = False
            continue
        spec["attacker_controllable"] = True
        cost = "medium"
        for level, prefixes in _ACI_MANIPULATION_COST_PREFIXES.items():
            if column in prefixes:
                cost = level
                break
        else:
            if column.startswith(("Packet Length", "Fwd Packet Length",
                                  "Bwd Packet Length", "Average Packet Size",
                                  "Subflow", "Total Length")):
                cost = "low"
        spec["manipulation_cost"] = cost


_apply_attack_surface()


FEATURE_SCHEMA: Dict[str, Dict[str, Dict[str, Any]]] = {
    ACI: _ACI_SCHEMA,
    CIC: _CIC_SCHEMA,
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def schema(dataset: str) -> Dict[str, Dict[str, Any]]:
    """Column table for one dataset."""
    if dataset not in FEATURE_SCHEMA:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Expected one of {sorted(FEATURE_SCHEMA)}."
        )
    return FEATURE_SCHEMA[dataset]


def all_columns(dataset: str) -> List[str]:
    """Every column, in raw file order."""
    return list(schema(dataset))


def label_column(dataset: str) -> str:
    cols = [c for c, s in schema(dataset).items() if s["role"] == "label"]
    if len(cols) != 1:
        raise ValueError(f"{dataset} must declare exactly one label column, found {cols}.")
    return cols[0]


def identifier_columns(dataset: str) -> List[str]:
    return [c for c, s in schema(dataset).items() if s["role"] == "identifier"]


def categorical_columns(dataset: str) -> List[str]:
    return [
        c for c, s in schema(dataset).items()
        if s["role"] == "feature" and s["dtype"] == "category"
    ]


def numeric_dtypes(dataset: str) -> Dict[str, str]:
    """float32 read dtypes for the numeric feature columns, for memory-bounded loads."""
    return {
        c: "float32"
        for c, s in schema(dataset).items()
        if s["role"] == "feature" and s["dtype"] in ("int64", "float64")
    }


def feature_columns(
    dataset: str,
    *,
    include_high_leakage: bool = True,
    include_zero_variance: bool = False,
    include_suspect: bool = True,
) -> List[str]:
    """
    Model input columns.

    Defaults reflect the Phase 1 decisions: ``critical`` leakage columns are
    always excluded, ``high`` risk columns (Src Port / Dst Port, the Idle
    group) are kept but tagged, and measured zero-variance columns are dropped.
    """
    out = []
    for col, spec in schema(dataset).items():
        if spec["role"] != "feature":
            continue
        if spec["leakage_risk"] == "critical":
            continue
        if not include_high_leakage and spec["leakage_risk"] == "high":
            continue
        if not include_zero_variance and spec["zero_variance"]:
            continue
        if not include_suspect and spec["suspect"]:
            continue
        out.append(col)
    return out


def columns_with(dataset: str, flag: str) -> List[str]:
    """Columns whose boolean flag is set, e.g. ``columns_with(ACI, 'has_inf')``."""
    return [c for c, s in schema(dataset).items() if s.get(flag) is True]


# ---------------------------------------------------------------------------
# Human-readable flow description
# ---------------------------------------------------------------------------

#: IANA protocol numbers that appear in these captures.
_PROTOCOL_NAMES = {
    0: "HOPOPT", 1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP", 47: "GRE",
    50: "ESP", 58: "ICMPv6", 89: "OSPF", 132: "SCTP",
}

#: CIC flag-fraction columns, and the flag each one reports.
_CIC_FLAGS = {
    "fin_flag_number": "FIN", "syn_flag_number": "SYN", "rst_flag_number": "RST",
    "psh_flag_number": "PSH", "ack_flag_number": "ACK", "ece_flag_number": "ECE",
    "cwr_flag_number": "CWR",
}

#: CIC protocol-indicator columns.
_CIC_PROTOCOLS = ["HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP",
                  "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC"]


def _num(record: Dict[str, Any], key: str) -> Optional[float]:
    value = record.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value  # drop NaN


def describe_flow(dataset: str, record: Dict[str, Any]) -> str:
    """
    Render a flow in words rather than numbers.

    The classifiers consume a standardised feature vector, in which
    ``Protocol Type`` is something like -0.31 and a packet rate is 1.87. That
    is the right representation for them and a useless one for a language
    model, which cannot tell that -0.31 means TCP.

    This turns the RAW (pre-scaling) values into statements a reader can reason
    about — protocol names, which TCP flags are set, packet sizes in bytes,
    rates per second. It is the one input the agent can have that the
    classifiers genuinely do not: not more data, but the same data in a form a
    language model can actually use.

    Falls back to a plain listing for a dataset without a bespoke renderer.
    """
    if dataset == CIC:
        return _describe_cic_flow(record)
    if dataset == ACI:
        return _describe_aci_flow(record)
    return "\n".join(f"{k} = {v}" for k, v in record.items())


def _describe_cic_flow(record: Dict[str, Any]) -> str:
    lines: List[str] = []

    proto = _num(record, "Protocol Type")
    if proto is not None:
        name = _PROTOCOL_NAMES.get(int(round(proto)), f"protocol {int(round(proto))}")
        lines.append(f"Transport protocol: {name} (number {int(round(proto))}).")

    ttl = _num(record, "Time_To_Live")
    header = _num(record, "Header_Length")
    bits = []
    if ttl is not None:
        bits.append(f"time-to-live {ttl:.0f}")
    if header is not None:
        bits.append(f"header length {header:.0f} bytes")
    if bits:
        lines.append("Header: " + ", ".join(bits) + ".")

    rate = _num(record, "Rate")
    number = _num(record, "Number")
    iat = _num(record, "IAT")
    bits = []
    if rate is not None:
        bits.append(f"{rate:,.0f} packets/second")
    if number is not None:
        bits.append(f"{number:.0f} packets observed")
    if iat is not None:
        bits.append(f"mean inter-arrival {iat:.3g} s")
    if bits:
        lines.append("Rate: " + ", ".join(bits) + ".")

    lo, hi = _num(record, "Min"), _num(record, "Max")
    avg, std = _num(record, "AVG"), _num(record, "Std")
    bits = []
    if lo is not None and hi is not None:
        bits.append(f"{lo:.0f}-{hi:.0f} bytes")
    if avg is not None:
        bits.append(f"mean {avg:.1f}")
    if std is not None:
        bits.append(f"std {std:.1f}")
    if bits:
        lines.append("Packet size: " + ", ".join(bits) + ".")

    set_flags = [
        f"{label} ({_num(record, col):.2f} of packets)"
        for col, label in _CIC_FLAGS.items()
        if (_num(record, col) or 0.0) > 0.01
    ]
    lines.append(
        "TCP flags present: " + (", ".join(set_flags) if set_flags else "none")
        + "."
    )

    counts = {k: _num(record, f"{k}_count") for k in ("ack", "syn", "fin", "rst")}
    named = [f"{k} {v:.0f}" for k, v in counts.items() if v]
    if named:
        lines.append("Flag counts: " + ", ".join(named) + ".")

    seen = [
        f"{p} ({_num(record, p):.2f})"
        for p in _CIC_PROTOCOLS
        if (_num(record, p) or 0.0) > 0.01
    ]
    lines.append(
        "Protocols indicated: " + (", ".join(seen) if seen else "none identified")
        + "."
    )
    return "\n".join(lines)


def _describe_aci_flow(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    proto = _num(record, "Protocol")
    if proto is not None:
        name = _PROTOCOL_NAMES.get(int(round(proto)), f"protocol {int(round(proto))}")
        lines.append(f"Transport protocol: {name}.")
    src, dst = _num(record, "Src Port"), _num(record, "Dst Port")
    if src is not None and dst is not None:
        lines.append(f"Ports: source {src:.0f}, destination {dst:.0f}.")

    duration = _num(record, "Flow Duration")
    fwd, bwd = _num(record, "Total Fwd Packet"), _num(record, "Total Bwd packets")
    if duration is not None:
        lines.append(
            f"Duration: {duration / 1e6:.3f} s"
            + (f", {fwd:.0f} packets forward and {bwd:.0f} back." if fwd is not None else ".")
        )

    flags = {"FIN": "FIN Flag Count", "SYN": "SYN Flag Count", "RST": "RST Flag Count",
             "PSH": "PSH Flag Count", "ACK": "ACK Flag Count", "URG": "URG Flag Count"}
    named = [f"{k} {_num(record, c):.0f}" for k, c in flags.items() if _num(record, c)]
    lines.append("TCP flag counts: " + (", ".join(named) if named else "none") + ".")

    mean, mx = _num(record, "Packet Length Mean"), _num(record, "Packet Length Max")
    if mean is not None:
        lines.append(f"Packet length: mean {mean:.1f} bytes, max {mx:.0f}.")
    conn = record.get("Connection Type")
    if conn:
        lines.append(f"Link: {conn}.")
    return "\n".join(lines)


def attacker_controllable_columns(
    dataset: str, *, max_cost: str = "medium"
) -> List[str]:
    """
    The Phase 2 attack surface: features an attacker can change, restricted to
    those they can change without abandoning the attack.

    ``max_cost`` of ``"low"`` gives only cost-free manipulations (padding, TTL,
    unused flag bits); ``"medium"`` also allows changes that degrade throughput
    or stealth; ``"high"`` allows everything, including the protocol and the
    flags that define the attack — useful only as an upper bound, since a
    "successful evasion" there may just be a different, harmless flow.
    """
    order = {"low": 0, "medium": 1, "high": 2}
    if max_cost not in order:
        raise ValueError(f"max_cost must be one of {sorted(order)}, got {max_cost!r}")
    ceiling = order[max_cost]
    return [
        c
        for c, s in schema(dataset).items()
        if s["attacker_controllable"]
        and order.get(s.get("manipulation_cost", "high"), 2) <= ceiling
    ]


def manipulation_cost(dataset: str, column: str) -> str:
    return schema(dataset).get(column, {}).get("manipulation_cost", "n/a")
