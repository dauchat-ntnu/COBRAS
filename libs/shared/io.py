"""
Input/output loaders for PowerSOC.

Supported formats:
  - CSV Folder (MATPOWER-style CSVs)
  - MATPOWER .m files (legacy, via pandapower converter)
"""

from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import replace
import re
import math
import warnings
import pandas as pd
from pandas.errors import ParserWarning

from .data import (
    BusData,
    BranchData,
    GeneratorData,
    LoadData,
    PowerFlowCase,
)


def _merge_dot_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize columns by splitting on '.' and summing duplicates.

        Example:
            - '30' + '30.1' + '30.2' -> single column '30' (row-wise sum)
            - '31.1' -> '31'
        """
        normalized = df.copy()
        normalized.columns = [str(col).split(".", 1)[0] for col in normalized.columns]

        # If multiple columns map to the same base name, merge by row-wise sum.
        if normalized.columns.duplicated().any():
                normalized = normalized.T.groupby(level=0, sort=False).sum().T

        return normalized


def _sanitize_load_profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid demands instead of silently turning them into zero load."""
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    if not numeric_df.apply(lambda column: column.map(math.isfinite)).all().all():
        raise ValueError("Load profiles must contain finite numeric values; fill missing demands explicitly")
    return numeric_df


def _select_snapshot_row(
    df: pd.DataFrame,
    *,
    load_at: Optional[str],
    load_index: Optional[int],
    file_path: Path,
) -> pd.Series:
    """Select one snapshot row using the same policy as load selection."""
    if load_index is not None and load_index < 0:
        raise _format_error_with_file("load_index must be non-negative", file_path)
    try:
        if load_at is not None:
            row = df.loc[load_at]
        elif load_index is not None:
            row = df.iloc[load_index]
        else:
            row = df.iloc[0]
    except Exception as exc:
        raise _format_error_with_file(
            f"Could not select snapshot row using load_at={load_at!r}, load_index={load_index!r}",
            file_path,
        ) from exc

    if isinstance(row, pd.DataFrame):
        raise _format_error_with_file(
            "Snapshot selection is ambiguous because the index is not unique",
            file_path,
        )
    return row


def _should_use_normalized_load_mode(
    *,
    candidate: bool,
    p_row: pd.Series,
    q_row: pd.Series,
    max_p_by_bus: dict[int, float],
    max_q_by_bus: dict[int, float],
    bus_mapping: Optional[dict[str, int]] = None,
    eps: float = 1e-12,
) -> tuple[bool, Optional[str]]:
    """Decide if normalized load mode is valid for the selected snapshot."""
    if not candidate:
        return False, None

    has_nonzero_factor_p = False
    has_nonzero_factor_q = False
    has_nonzero_pmax = False
    has_nonzero_qmax = False

    for bus_col in p_row.index:
        bus_col_str = str(bus_col).strip()
        if bus_mapping and bus_col_str in bus_mapping:
            bus_id = bus_mapping[bus_col_str]
        else:
            try:
                bus_id = int(bus_col_str)
            except Exception:
                continue

        p_factor = float(p_row[bus_col])
        q_factor = float(q_row[bus_col])

        if abs(p_factor) > eps:
            has_nonzero_factor_p = True
            if abs(float(max_p_by_bus.get(bus_id, 0.0))) > eps:
                has_nonzero_pmax = True

        if abs(q_factor) > eps:
            has_nonzero_factor_q = True
            if abs(float(max_q_by_bus.get(bus_id, 0.0))) > eps:
                has_nonzero_qmax = True

    if has_nonzero_factor_p and not has_nonzero_pmax:
        return False, "non-zero p_load factors but all corresponding mpc_bus Pd are zero"
    if has_nonzero_factor_q and not has_nonzero_qmax:
        return False, "non-zero q_load factors but all corresponding mpc_bus Qd are zero"

    return True, None


def _load_generator_profiles(
    folder_path: Path,
    profiles_filename: str,
) -> Optional[pd.DataFrame]:
    """Load optional normalized generator profiles from network folder.
    
    Handles both comma-delimited and semicolon-delimited CSV files.
    """
    profiles_path = folder_path / profiles_filename

    if not profiles_path.exists():
        print(f"Note: profiles file not found at {profiles_path}; proceeding without generator profiles.")
        return None

    try:
        # Try auto-detection first, but if it fails, explicitly try common delimiters
        profiles_df = pd.read_csv(profiles_path, sep=None, engine="python", index_col=0)
        
        # If auto-detection resulted in a single column (likely delimiter not detected),
        # try semicolon explicitly
        if profiles_df.shape[1] == 0 and profiles_df.shape[0] > 0:
            profiles_df = pd.read_csv(profiles_path, sep=";", index_col=0)
    except Exception as exc:
        raise _format_error_with_file("Invalid profiles format", profiles_path) from exc

    if profiles_df.empty or profiles_df.shape[1] == 0:
        print(
            f"Warning: profiles.csv at {profiles_path} has no usable profile columns. "
            f"Generators will use static p_max/q_max limits regardless of profile_name."
        )
        return None

    numeric_profiles = profiles_df.apply(pd.to_numeric, errors="coerce")
    if numeric_profiles.isna().any().any():
        raise _format_error_with_file(
            "profiles.csv contains non-numeric values",
            profiles_path,
        )

    # Factors above one support capacity-scaling sensitivity studies.
    if not numeric_profiles.apply(lambda column: column.map(math.isfinite)).all().all() or (numeric_profiles < 0).any().any():
        raise _format_error_with_file("Generator profiles must contain finite non-negative factors", profiles_path)

    return numeric_profiles


def _format_error_with_file(message: str, file_path: Path) -> ValueError:
    """Build a ValueError that points to the source file causing a format error."""
    return ValueError(f"{message} (file: {file_path})")


def _compact_row_snapshot(row: pd.Series, max_items: int = 8) -> str:
    """Return a compact one-line preview of a row for error messages."""
    items = []
    for idx, (key, value) in enumerate(row.items()):
        if idx >= max_items:
            items.append("...")
            break
        items.append(f"{key}={value!r}")
    return ", ".join(items)


def _build_row_format_error(
    *,
    file_path: Path,
    table_name: str,
    row_idx: object,
    row: pd.Series,
    expected_columns: list[str],
    exc: Exception,
) -> ValueError:
    """Build a detailed row-level format error with context and expectations."""
    row_columns = [str(c) for c in row.index]
    missing = [c for c in expected_columns if c not in row_columns]
    expected_str = ", ".join(expected_columns)
    message = (
        f"Invalid row in {table_name} at row index {row_idx} (file: {file_path}). "
        f"Reason: {exc}. "
        f"Expected columns include: [{expected_str}]. "
        f"Missing expected columns in parsed row: {missing}. "
        f"Row snapshot: {_compact_row_snapshot(row)}"
    )
    return ValueError(message)


def _parse_first_float_token(text: str) -> Optional[float]:
    """Return the first floating-point token found in text, if any."""
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def _extract_base_mva_from_dataframe(df: pd.DataFrame, file_path: Path) -> float:
    """Extract base MVA from a table that may also include a Base kV column."""
    normalized = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}

    # Support common explicit names.
    for key in ("base_mva", "basemva", "mpc_base_mva"):
        if key in normalized:
            return float(df[normalized[key]].iloc[0])

    # Accept broader names that include both base and mva.
    for norm_name, original_col in normalized.items():
        if "base" in norm_name and "mva" in norm_name:
            return float(df[original_col].iloc[0])

    # Fallback: first numeric value in first row across columns.
    if not df.empty:
        first_row = df.iloc[0]
        numeric_row = pd.to_numeric(first_row, errors="coerce")
        for value in numeric_row:
            if pd.notna(value):
                return float(value)

    # Last resort: parse first numeric token from first cell text.
    if not df.empty and df.shape[1] >= 1:
        token = _parse_first_float_token(str(df.iloc[0, 0]))
        if token is not None:
            return float(token)

    raise _format_error_with_file("Invalid base_mva value", file_path)


def _normalize_column_name(name: str) -> str:
    """Normalize column names for tolerant format matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _canonicalize_mpc_bus_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map accepted mpc_bus header variants to canonical MATPOWER-style names."""
    normalized_to_canonical = {
        "bus": "bus",
        "busi": "bus",        # bus_i
        "type": "type",
        "bustype": "type",    # bus_type
        "pd": "Pd",
        "qd": "Qd",
        "gs": "Gs",
        "bs": "Bs",
        "area": "area",
        "busarea": "area",    # bus_area
        "vm": "Vm",
        "va": "Va",
        "basekv": "baseKV",   # baseKV / base_kV
        "zone": "zone",
        "vmax": "Vmax",
        "vmin": "Vmin",
    }

    rename_map = {}
    for original_col in df.columns:
        normalized = _normalize_column_name(original_col)
        canonical = normalized_to_canonical.get(normalized)
        if canonical is None:
            continue
        # Avoid overwriting if canonical column is already present.
        if canonical in df.columns and original_col != canonical:
            continue
        if original_col != canonical:
            rename_map[original_col] = canonical

    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _canonicalize_mpc_branch_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map accepted mpc_branch header variants to canonical MATPOWER-style names."""
    normalized_to_canonical = {
        "fbus": "fbus",
        "fbusbus": "fbus",
        "fbusid": "fbus",
        "fbusno": "fbus",
        "f_bus": "fbus",
        "tbus": "tbus",
        "tbusbus": "tbus",
        "tbusid": "tbus",
        "tbusno": "tbus",
        "t_bus": "tbus",
        "r": "r",
        "brr": "r",          # br_r
        "x": "x",
        "brx": "x",          # br_x
        "b": "b",
        "brb": "b",          # br_b
        "ratea": "rateA",
        "ratea": "rateA",
        "rate_a": "rateA",
        "rateb": "rateB",
        "rate_b": "rateB",
        "ratec": "rateC",
        "rate_c": "rateC",
        "ratio": "ratio",
        "tap": "ratio",
        "angle": "angle",
        "shift": "angle",
        "status": "status",
        "brstatus": "status",  # br_status
        "angmin": "angmin",
        "angmax": "angmax",
    }

    rename_map = {}
    for original_col in df.columns:
        normalized = _normalize_column_name(original_col)
        canonical = normalized_to_canonical.get(normalized)
        if canonical is None:
            continue
        if canonical in df.columns and original_col != canonical:
            continue
        if original_col != canonical:
            rename_map[original_col] = canonical

    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def _read_mpc_bus_csv(path: Path) -> pd.DataFrame:
    """Read bus CSV without allowing pandas to promote bus_i to an implicit index."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParserWarning)
        return pd.read_csv(path, sep=None, engine="python", index_col=False)


def _load_bus_mapping(mapping_path: Path) -> dict[str, int]:
    """Load load-column to bus-id mapping from CSV."""
    try:
        mapping_df = pd.read_csv(mapping_path, sep=None, engine="python")
    except Exception as exc:
        raise _format_error_with_file(
            f"Invalid bus mapping format. Parser error: {exc}",
            mapping_path,
        ) from exc

    if mapping_df.empty:
        raise _format_error_with_file("Bus mapping file is empty", mapping_path)

    normalized_cols = {_normalize_column_name(col): col for col in mapping_df.columns}

    # Preferred/expected format from user spec:
    #   bus_i, time_series_ID, existing_load
    target_col = normalized_cols.get("busi") or normalized_cols.get("bus")
    source_col = normalized_cols.get("timeseriesid")
    existing_col = normalized_cols.get("existingload")

    # Fallback for legacy mapping files.
    if source_col is None:
        for key in (
            "loadcolumn",
            "profilecolumn",
            "column",
            "name",
            "busname",
            "loadprofile",
            "source",
        ):
            if key in normalized_cols:
                source_col = normalized_cols[key]
                break
    if target_col is None:
        for key in ("busid", "target", "mappedbus"):
            if key in normalized_cols:
                target_col = normalized_cols[key]
                break

    # Last fallback: first two columns if explicit headers are not provided.
    if source_col is None or target_col is None:
        if len(mapping_df.columns) < 2:
            raise _format_error_with_file(
                "Bus mapping file must have at least two columns: load column name and bus id",
                mapping_path,
            )
        source_col = mapping_df.columns[0]
        target_col = mapping_df.columns[1]

    mapping: dict[str, int] = {}
    for row_idx, row in mapping_df.iterrows():
        try:
            if existing_col is not None:
                existing_raw = str(row.get(existing_col, "")).strip().lower()
                if existing_raw in {"false", "0", "no", "n", "f"}:
                    continue

            source_name = str(row[source_col]).strip()
            if not source_name or source_name.lower() == "nan":
                continue
            bus_id = int(row[target_col])
            mapping[source_name] = bus_id
        except Exception as exc:
            raise ValueError(
                f"Invalid row in bus mapping at row index {row_idx} (file: {mapping_path}). "
                f"Reason: {exc}. Row snapshot: {_compact_row_snapshot(row)}"
            ) from exc

    if not mapping:
        raise _format_error_with_file("Bus mapping file produced no valid mappings", mapping_path)

    return mapping


def _pick_default_bus_mapping_file(folder_path: Path) -> Optional[str]:
    """Return a default bus mapping filename if one exists in the folder."""
    if not folder_path.exists() or not folder_path.is_dir():
        return None

    preferred = ["bus_mapping.csv", "bus_maping.csv"]
    for name in preferred:
        if (folder_path / name).exists():
            return name

    candidates = []
    for file_path in folder_path.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".csv":
            continue
        stem_lower = file_path.stem.lower()
        if stem_lower.startswith("bus_mapping") or stem_lower.startswith("bus_maping"):
            candidates.append(file_path.name)

    if not candidates:
        return None
    return sorted(candidates)[0]


def _loads_from_snapshot(p_row, q_row, *, buses, base_mva,
                         load_interpretation_mode, bus_mapping,
                         bus_mapping_path, p_load_path, q_load_path, mpc_bus_path):
    """Convert and validate a snapshot identically for single and range runs."""
    eps = 1e-12
    normalized_mode_candidate = any(abs(b.p_d) > eps or abs(b.q_d) > eps for b in buses)
    max_p_by_bus = {b.bus_id: b.p_d for b in buses}
    max_q_by_bus = {b.bus_id: b.q_d for b in buses}
    load_mode = str(load_interpretation_mode or "auto").strip().lower()
    if load_mode not in {"auto", "normalized", "absolute"}:
        raise ValueError(
            "Invalid load_interpretation_mode. "
            f"Expected one of ['auto', 'normalized', 'absolute'], got {load_interpretation_mode!r}."
        )

    normalized_fallback_reason: Optional[str] = None
    if load_mode == "normalized":
        force_normalized_profiles = True
    elif load_mode == "absolute":
        force_normalized_profiles = False
    else:
        force_normalized_profiles, normalized_fallback_reason = _should_use_normalized_load_mode(
            candidate=normalized_mode_candidate,
            p_row=p_row,
            q_row=q_row,
            max_p_by_bus=max_p_by_bus,
            max_q_by_bus=max_q_by_bus,
            bus_mapping=bus_mapping,
            eps=eps,
        )
        if normalized_mode_candidate and not force_normalized_profiles and normalized_fallback_reason:
            print(
                "Warning: mpc_bus Pd/Qd suggested normalized load mode, "
                f"but loader switched to absolute p_load/q_load values because {normalized_fallback_reason}."
            )

    loads = []
    missing_max_bus_ids = set()
    zero_max_with_nonzero_factor = set()
    for bus_col in p_row.index:
        bus_col_str = str(bus_col).strip()
        if bus_mapping and bus_col_str in bus_mapping:
            bus_id = bus_mapping[bus_col_str]
        else:
            try:
                bus_id = int(bus_col_str)
            except ValueError as exc:
                if bus_mapping:
                    raise ValueError(
                        f"Load column '{bus_col_str}' was not found in bus mapping file "
                        f"{bus_mapping_path} (load files: {p_load_path}, {q_load_path})"
                    ) from exc
                raise ValueError(
                    f"Invalid bus column '{bus_col}' in load files; expected numeric bus IDs. "
                    f"If load columns are named differently, provide bus_mapping.csv "
                    f"via bus_mapping_filename (files: {p_load_path}, {q_load_path})"
                ) from exc

        if force_normalized_profiles:
            if bus_id not in max_p_by_bus or bus_id not in max_q_by_bus:
                missing_max_bus_ids.add(bus_id)
                continue

            p_factor = float(p_row[bus_col])
            q_factor = float(q_row[bus_col])
            if not math.isfinite(p_factor) or not math.isfinite(q_factor):
                raise ValueError(
                    f"Invalid load factor for bus column '{bus_col_str}' in load files "
                    f"(files: {p_load_path}, {q_load_path}). "
                    f"Selected row contains non-finite values: P={p_row[bus_col]!r}, Q={q_row[bus_col]!r}."
                )
            p_max = float(max_p_by_bus[bus_id])
            q_max = float(max_q_by_bus[bus_id])

            if (abs(p_factor) > eps and abs(p_max) <= eps) or (abs(q_factor) > eps and abs(q_max) <= eps):
                zero_max_with_nonzero_factor.add(bus_id)

            # Forced normalized mode: p_load/q_load are interpreted as load factors.
            p_pu = (p_factor * p_max) / base_mva
            q_pu = (q_factor * q_max) / base_mva
            load = LoadData(bus_id=bus_id, p_d=p_pu, q_d=q_pu)
        else:
            if load_mode == "absolute":
                # Explicit absolute-demand mode: convert MW/MVAr values to p.u. before
                # they enter the solver-facing LoadData container.
                p_value = float(p_row[bus_col]) / base_mva
                q_value = float(q_row[bus_col]) / base_mva
            else:
                p_value = float(p_row[bus_col])
                q_value = float(q_row[bus_col])

            load = LoadData(
                bus_id=bus_id,
                p_d=p_value,
                q_d=q_value,
            )
            if not math.isfinite(load.p_d) or not math.isfinite(load.q_d):
                raise ValueError(
                    f"Invalid load value for bus column '{bus_col_str}' in load files "
                    f"(files: {p_load_path}, {q_load_path}). "
                    f"Selected row contains non-finite values: P={p_row[bus_col]!r}, Q={q_row[bus_col]!r}."
                )
        loads.append(load)

    if missing_max_bus_ids:
        raise ValueError(
            "Normalized load mode was triggered by non-zero Pd/Qd, but some buses in p_load/q_load "
            f"do not exist in {mpc_bus_path}: {sorted(missing_max_bus_ids)} "
            f"(load files: {p_load_path}, {q_load_path})"
        )

    return loads


def load_case_from_folder(
    folder_path: Path,
    load_at: Optional[str] = None,
    load_index: Optional[int] = None,
    override_voltage_bounds: bool = False,
    default_vmin: float = 0.9,
    default_vmax: float = 1.1,
    enable_popup_warning: bool = True,
    mpc_base_mva_filename: str = "mpc_base_mva",
    mpc_bus_filename: str = "mpc_bus.csv",
    mpc_branch_filename: str = "mpc_branch.csv",
    p_load_filename: str = "p_load.csv",
    q_load_filename: str = "q_load.csv",
    profiles_filename: str = "profiles.csv",
    bus_mapping_filename: Optional[str] = None,
    generators_filename: str = "generators.xlsx",
    load_interpretation_mode: str = "auto",
    cache: Optional[dict[str, Any]] = None,
) -> PowerFlowCase:
    """
    Load case from folder with MATPOWER-style CSV files.
    
    Expected files in folder:
      - mpc_base_mva (single value or CSV with header 'base_mva')
        - mpc_bus.csv (columns accepted in either naming style):
            bus/type/Pd/Qd/Gs/Bs/area/Vm/Va/baseKV/zone/Vmax/Vmin
            or bus_i/bus_type/Pd/Qd/Gs/Bs/bus_area/Vm/Va/base_kV/zone/Vmax/Vmin
        - mpc_branch.csv (columns accepted in either naming style):
            fbus/tbus/r/x/b/rateA/rateB/rateC/ratio/angle/status/angmin/angmax
            or f_bus/t_bus/br_r/br_x/br_b/rate_A/rate_B/rate_C/tap/shift/br_status
                - p_load.csv (columns: date/time + load values at each bus)
            - q_load.csv (columns: date/time + load values at each bus)
            - profiles.csv (optional; normalized [0,1] generator profile timeseries)
    - bus_mapping.csv (optional; map non-numeric load column names to bus ids)
      - generators.xlsx (columns: bus, Pmin, Pmax, Qmin, Qmax, Vmax, Cp, Cq, C0)
    
    Args:
        folder_path: Path to folder
        load_at: Date string to select from p_load.csv (if time series)
        load_index: Row index to select from load files (if time series)
        override_voltage_bounds: If True, ignore Vmin/Vmax from mpc_bus.csv
        default_vmin, default_vmax: Default voltage bounds to use
        enable_popup_warning: If True, show popup warnings when load mode is auto-switched
        mpc_base_mva_filename: Base-MVA file name (plain text or CSV)
        mpc_bus_filename: Bus data file name
        mpc_branch_filename: Branch data file name
        p_load_filename: Active load profile file name
        q_load_filename: Reactive load profile file name
        profiles_filename: Optional generator profile file name
        bus_mapping_filename: Optional CSV mapping non-numeric load columns to bus ids
        generators_filename: Generator data file name (.xlsx or .csv)
        load_interpretation_mode: One of "auto", "normalized", "absolute".
            - auto: infer from mpc_bus Pd/Qd with safety fallback
            - normalized: force p_load/q_load as factors against bus Pd/Qd maxima
            - absolute: force p_load/q_load as absolute demands and convert to p.u. using base_mva
    
    Returns:
        PowerFlowCase object
    """
    
    folder_path = Path(folder_path)

    # Optional per-run cache to avoid reparsing static inputs each timestep.
    if cache is None:
        cache = {}

    def _cache_get_or_load(cache_key: tuple, loader):
        if cache_key in cache:
            return cache[cache_key]
        value = loader()
        cache[cache_key] = value
        return value
    
    # Load base MVA
    base_mva_path = folder_path / mpc_base_mva_filename
    if base_mva_path.exists() and base_mva_path.suffix.lower() != ".csv":
        try:
            with open(base_mva_path) as f:
                raw_text = f.read().strip()
            token = _parse_first_float_token(raw_text)
            if token is None:
                raise _format_error_with_file("Invalid mpc_base_mva format", base_mva_path)
            base_mva = float(token)
        except Exception as exc:
            raise _format_error_with_file("Invalid mpc_base_mva format", base_mva_path) from exc
    else:
        # Try CSV from the explicit file name first (if applicable), then no-extension + .csv fallback.
        if base_mva_path.suffix.lower() == ".csv":
            base_mva_csv = base_mva_path
        else:
            base_mva_csv = folder_path / f"{mpc_base_mva_filename}.csv"

        if base_mva_csv.exists():
            try:
                # Auto-detect delimiter to support files like:
                # "Base MVA;Base kV" with a data row "10;22".
                df = _cache_get_or_load(
                    ("base_mva_csv", str(base_mva_csv)),
                    lambda: pd.read_csv(base_mva_csv, sep=None, engine="python"),
                )
            except Exception as exc:
                raise _format_error_with_file("Invalid mpc_base_mva CSV format", base_mva_csv) from exc
            try:
                base_mva = _extract_base_mva_from_dataframe(df, base_mva_csv)
            except Exception as exc:
                raise _format_error_with_file("Invalid base_mva value", base_mva_csv) from exc
        else:
            print("Warning: mpc_base_mva file not found; defaulting to 100.0 MVA")
            base_mva = 100.0
    
    if not math.isfinite(base_mva) or base_mva <= 0:
        raise _format_error_with_file("base_mva must be finite and positive", base_mva_path)

    # Load buses
    mpc_bus_path = folder_path / mpc_bus_filename
    try:
        mpc_bus = _cache_get_or_load(
            ("mpc_bus", str(mpc_bus_path)),
            lambda: _canonicalize_mpc_bus_columns(
                _read_mpc_bus_csv(mpc_bus_path)
            ),
        )
    except Exception as exc:
        raise _format_error_with_file("Invalid mpc_bus format", mpc_bus_path) from exc
    
    buses = []
    expected_bus_columns = [
        "bus", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin",
    ]
    for row_idx, row in mpc_bus.iterrows():
        try:
            bus_id = int(row["bus"])
        
            vmax = float(default_vmax if override_voltage_bounds else row.get("Vmax", default_vmax))**2
            vmin = float(default_vmin if override_voltage_bounds else row.get("Vmin", default_vmin))**2
        
            bus = BusData(
                bus_id=bus_id,
                bus_type=int(row.get("type", 1)),
                p_d=float(row.get("Pd", 0.0)),
                q_d=float(row.get("Qd", 0.0)),
                g_s=float(row.get("Gs", 0.0)),
                b_s=float(row.get("Bs", 0.0)),
                area=int(row.get("area", 1)),
                v_m=float(row.get("Vm", 1.0)),
                v_a=float(row.get("Va", 0.0)),
                base_kv=float(row.get("baseKV", 1.0)),
                zone=int(row.get("zone", 1)),
                v_max=vmax,
                v_min=vmin,
            )
            buses.append(bus)
        except Exception as exc:
            raise _build_row_format_error(
                file_path=mpc_bus_path,
                table_name="mpc_bus",
                row_idx=row_idx,
                row=row,
                expected_columns=expected_bus_columns,
                exc=exc,
            ) from exc

    # Load branches
    mpc_branch_path = folder_path / mpc_branch_filename
    try:
        mpc_branch = _cache_get_or_load(
            ("mpc_branch", str(mpc_branch_path)),
            lambda: _canonicalize_mpc_branch_columns(
                pd.read_csv(mpc_branch_path, sep=None, engine="python")
            ),
        )
    except Exception as exc:
        raise _format_error_with_file("Invalid mpc_branch format", mpc_branch_path) from exc
    
    branches = []
    expected_branch_columns = [
        "fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle", "status", "angmin", "angmax",
    ]
    for idx, row in mpc_branch.iterrows():
        try:
            # Normalize endpoint ordering: ensure from_bus < to_bus for consistency.
            fbus_raw = int(row["fbus"])
            tbus_raw = int(row["tbus"])
            branch = BranchData(
                branch_id=idx,
                from_bus=fbus_raw,
                to_bus=tbus_raw,
                r=float(row["r"]),
                x=float(row["x"]),
                b=float(row.get("b", 0.0)),
                rateA=float(row.get("rateA", 9999.0)),
                rateB=float(row.get("rateB", 9999.0)),
                rateC=float(row.get("rateC", 9999.0)),
                ratio=float(row.get("ratio", 0.0)),
                angle=float(row.get("angle", 0.0)),
                status=int(row.get("status", 1)),
                angmin=float(row.get("angmin", -360.0)),
                angmax=float(row.get("angmax", 360.0)),
                lmax=(float(row.get("rateA", 9999.0)) / base_mva) ** 2,
            )
            branches.append(branch)
        except Exception as exc:
            raise _build_row_format_error(
                file_path=mpc_branch_path,
                table_name="mpc_branch",
                row_idx=idx,
                row=row,
                expected_columns=expected_branch_columns,
                exc=exc,
            ) from exc
    
    # Load loads
    p_load_path = folder_path / p_load_filename
    q_load_path = folder_path / q_load_filename
    if not bus_mapping_filename:
        bus_mapping_filename = _pick_default_bus_mapping_file(folder_path)
    bus_mapping_path = folder_path / bus_mapping_filename if bus_mapping_filename else None
    bus_mapping: dict[str, int] = {}
    if bus_mapping_path is not None:
        if not bus_mapping_path.exists():
            raise _format_error_with_file("Bus mapping file not found", bus_mapping_path)
        bus_mapping = _cache_get_or_load(
            ("bus_mapping", str(bus_mapping_path)),
            lambda: _load_bus_mapping(bus_mapping_path),
        )

    try:
        p_load_df = _cache_get_or_load(
            ("p_load", str(p_load_path)),
            lambda: pd.read_csv(p_load_path, sep=None, engine="python", index_col=0),
        )
    except Exception as exc:
        raise _format_error_with_file("Invalid p_load format", p_load_path) from exc
    try:
        q_load_df = _cache_get_or_load(
            ("q_load", str(q_load_path)),
            lambda: pd.read_csv(q_load_path, sep=None, engine="python", index_col=0),
        )
    except Exception as exc:
        raise _format_error_with_file("Invalid q_load format", q_load_path) from exc

    # Handle duplicated bus columns and sanitize once per cached dataset.
    p_load_df = _cache_get_or_load(
        ("p_load_processed", str(p_load_path)),
        lambda: _sanitize_load_profile_dataframe(_merge_dot_suffix_columns(p_load_df)),
    )
    q_load_df = _cache_get_or_load(
        ("q_load_processed", str(q_load_path)),
        lambda: _sanitize_load_profile_dataframe(_merge_dot_suffix_columns(q_load_df)),
    )

    # Sanity check: active/reactive load files must reference the same bus columns.
    if not p_load_df.index.equals(q_load_df.index):
        raise ValueError("Active and reactive load profiles must have identical ordered timestep labels")
    p_cols = set(p_load_df.columns)
    q_cols = set(q_load_df.columns)
    if p_cols != q_cols:
        missing_in_q = sorted(p_cols - q_cols)
        missing_in_p = sorted(q_cols - p_cols)
        raise ValueError(
            f"{p_load_path} and {q_load_path} have mismatched bus columns after normalization. "
            f"Missing in q_load: {missing_in_q}. Missing in p_load: {missing_in_p}."
        )
    
    p_row = _select_snapshot_row(
        p_load_df,
        load_at=load_at,
        load_index=load_index,
        file_path=p_load_path,
    )
    q_row = _select_snapshot_row(
        q_load_df,
        load_at=load_at,
        load_index=load_index,
        file_path=q_load_path,
    )

    loads = _loads_from_snapshot(
        p_row, q_row, buses=buses, base_mva=base_mva,
        load_interpretation_mode=load_interpretation_mode,
        bus_mapping=bus_mapping, bus_mapping_path=bus_mapping_path,
        p_load_path=p_load_path, q_load_path=q_load_path, mpc_bus_path=mpc_bus_path,
    )

    # Load generators
    gen_file = folder_path / generators_filename
    if gen_file.exists():
        try:
            if gen_file.suffix.lower() == ".csv":
                gen_df = _cache_get_or_load(
                    ("generators_csv", str(gen_file)),
                    lambda: pd.read_csv(gen_file, sep=None, engine="python"),
                )
            else:
                gen_df = _cache_get_or_load(
                    ("generators_excel", str(gen_file)),
                    lambda: pd.read_excel(gen_file),
                )
        except Exception as exc:
            raise _format_error_with_file(
                f"Invalid generators file format. Parser error: {exc}",
                gen_file,
            ) from exc
        
        generators = []
        expected_gen_columns = [
            "bus", "pmin", "pmax", "qmin", "qmax", "status", "cp", "cq", "inflow_profile",
        ]
        for idx, row in gen_df.iterrows():
            try:
                status_raw = row.get("status", row.get("Status", row.get("active", row.get("active", 1))))
                status = int(status_raw) if pd.notna(status_raw) else 0
                inflow_profile_raw = row.get("inflow_profile", None)
                inflow_profile = None
                if pd.notna(inflow_profile_raw):
                    inflow_profile = str(inflow_profile_raw).strip() or None

                gen = GeneratorData(
                    gen_id=idx,
                    bus_id=int(row["bus"]),
                    p_min=float(row.get("pmin", 0.0)),
                    p_max=float(row.get("pmax", 1.0)),
                    q_min=float(row.get("qmin", -1.0)),
                    q_max=float(row.get("qmax", 1.0)),
                    status=status,
                    c_p=float(row.get("cp", 1.0)),
                    c_q=float(row.get("cq", 0.0)),
                    profile_name=inflow_profile,
                )
                generators.append(gen)
            except Exception as exc:
                raise _build_row_format_error(
                    file_path=gen_file,
                    table_name="generators",
                    row_idx=idx,
                    row=row,
                    expected_columns=expected_gen_columns,
                    exc=exc,
                ) from exc
    else:
        # Default: generator at root bus
        root_bus = buses[0].bus_id if buses else 1
        generators = [
            GeneratorData(
                gen_id=0,
                bus_id=root_bus,
                p_max=1.0,
                q_max=1.0,
                p_min=0.0,
                q_min=0.0,
                status=1,
                c_p=1.0,
                c_q=0.0
            )
        ]

    profiles_df = _cache_get_or_load(
        ("generator_profiles", str(folder_path / profiles_filename)),
        lambda: _load_generator_profiles(folder_path, profiles_filename),
    )
    if profiles_df is not None and len(profiles_df) != len(p_load_df):
        raise ValueError("Generator and load profiles must have identical snapshot counts")
    profile_row: Optional[pd.Series] = None
    if profiles_df is not None:
        profile_row = _select_snapshot_row(
            profiles_df,
            load_at=load_at,
            load_index=load_index,
            file_path=folder_path / profiles_filename,
        )

    for gen in generators:
        gen.p_max_available = gen.p_max
        gen.q_max_available = gen.q_max
        gen.q_min_available = gen.q_min
        
        # If generator has no profile_name, it uses static limits
        if not gen.profile_name:
            continue
        
        # If no profile row was loaded, skip (will use static limits)
        if profile_row is None:
            continue
        
        # Apply profile factor if the profile column exists
        if gen.profile_name not in profile_row.index:
            print(
                f"Warning: Generator {gen.gen_id} at bus {gen.bus_id} has profile_name='{gen.profile_name}' "
                f"but this profile column was not found in profiles.csv. "
                f"Using static p_max={gen.p_max}. Available profiles: {sorted(profile_row.index)}"
            )
            continue
        
        profile_factor = float(profile_row[gen.profile_name])
        gen.p_max_available = gen.p_max * profile_factor
        gen.q_max_available = gen.q_max * profile_factor
        gen.q_min_available = gen.q_min * profile_factor

        
    
    # Determine root bus (slack bus with type 3, or highest Pmax generator)
    root_bus = None
    for bus in buses:
        if bus.bus_type == 3:
            root_bus = bus.bus_id
            break
    
    if root_bus is None and generators:
        root_bus = generators[0].bus_id
    
    if root_bus is None:
        root_bus = buses[0].bus_id if buses else 1
    
    return PowerFlowCase(
        buses=buses,
        branches=branches,
        generators=generators,
        loads=loads,
        root_bus=root_bus,
        base_mva=base_mva,
    )


def build_case_loader_from_folder(
    folder_path: Path,
    *,
    override_voltage_bounds: bool = False,
    default_vmin: float = 0.9,
    default_vmax: float = 1.1,
    enable_popup_warning: bool = True,
    mpc_base_mva_filename: str = "mpc_base_mva",
    mpc_bus_filename: str = "mpc_bus.csv",
    mpc_branch_filename: str = "mpc_branch.csv",
    p_load_filename: str = "p_load.csv",
    q_load_filename: str = "q_load.csv",
    profiles_filename: str = "profiles.csv",
    bus_mapping_filename: Optional[str] = None,
    generators_filename: str = "generators.xlsx",
    load_interpretation_mode: str = "auto",
) -> Callable[[int], PowerFlowCase]:
    """Build a fast per-timestep loader with static topology and cached timeseries."""
    folder_path = Path(folder_path)
    cache: dict[str, Any] = {}

    # Materialize static topology once and warm cache.
    base_case = load_case_from_folder(
        folder_path=folder_path,
        load_index=0,
        override_voltage_bounds=override_voltage_bounds,
        default_vmin=default_vmin,
        default_vmax=default_vmax,
        enable_popup_warning=enable_popup_warning,
        mpc_base_mva_filename=mpc_base_mva_filename,
        mpc_bus_filename=mpc_bus_filename,
        mpc_branch_filename=mpc_branch_filename,
        p_load_filename=p_load_filename,
        q_load_filename=q_load_filename,
        profiles_filename=profiles_filename,
        bus_mapping_filename=bus_mapping_filename,
        generators_filename=generators_filename,
        load_interpretation_mode=load_interpretation_mode,
        cache=cache,
    )

    static_buses = base_case.buses
    static_branches = base_case.branches
    static_root = base_case.root_bus
    static_base_mva = base_case.base_mva

    # Build generator templates once; only profile-dependent availability changes per timestep.
    generator_templates = [replace(g) for g in base_case.generators]

    p_load_path = folder_path / p_load_filename
    q_load_path = folder_path / q_load_filename

    if not bus_mapping_filename:
        bus_mapping_filename = _pick_default_bus_mapping_file(folder_path)
    bus_mapping_path = folder_path / bus_mapping_filename if bus_mapping_filename else None

    bus_mapping: dict[str, int] = {}
    if bus_mapping_path is not None:
        if not bus_mapping_path.exists():
            raise _format_error_with_file("Bus mapping file not found", bus_mapping_path)
        bus_mapping = _load_bus_mapping(bus_mapping_path)

    p_load_df = cache.get(("p_load_processed", str(p_load_path)))
    q_load_df = cache.get(("q_load_processed", str(q_load_path)))
    if p_load_df is None:
        p_load_df = _sanitize_load_profile_dataframe(
            _merge_dot_suffix_columns(pd.read_csv(p_load_path, sep=None, engine="python", index_col=0))
        )
        cache[("p_load_processed", str(p_load_path))] = p_load_df
    if q_load_df is None:
        q_load_df = _sanitize_load_profile_dataframe(
            _merge_dot_suffix_columns(pd.read_csv(q_load_path, sep=None, engine="python", index_col=0))
        )
        cache[("q_load_processed", str(q_load_path))] = q_load_df
    if len(p_load_df) != len(q_load_df):
        raise ValueError(
            f"{p_load_path} and {q_load_path} have different snapshot counts: "
            f"{len(p_load_df)} vs {len(q_load_df)}."
        )

    profiles_df = cache[("generator_profiles", str(folder_path / profiles_filename))]
    if profiles_df is not None and len(profiles_df) != len(p_load_df):
        raise ValueError(
            f"{folder_path / profiles_filename} and {p_load_path} have different snapshot counts: "
            f"{len(profiles_df)} vs {len(p_load_df)}."
        )

    def _build_at(load_index: int) -> PowerFlowCase:
        p_row = _select_snapshot_row(
            p_load_df,
            load_at=None,
            load_index=load_index,
            file_path=p_load_path,
        )
        q_row = _select_snapshot_row(
            q_load_df,
            load_at=None,
            load_index=load_index,
            file_path=q_load_path,
        )

        loads = _loads_from_snapshot(
            p_row, q_row, buses=static_buses, base_mva=static_base_mva,
            load_interpretation_mode=load_interpretation_mode,
            bus_mapping=bus_mapping, bus_mapping_path=bus_mapping_path,
            p_load_path=p_load_path, q_load_path=q_load_path, mpc_bus_path=folder_path / mpc_bus_filename,
        )

        profile_row: Optional[pd.Series] = None
        if profiles_df is not None:
            profile_row = _select_snapshot_row(
                profiles_df,
                load_at=None,
                load_index=load_index,
                file_path=folder_path / profiles_filename,
            )

        generators = [replace(gen) for gen in generator_templates]
        for gen in generators:
            gen.p_max_available = gen.p_max
            gen.q_max_available = gen.q_max
            gen.q_min_available = gen.q_min
            if not gen.profile_name or profile_row is None:
                continue
            if gen.profile_name not in profile_row.index:
                continue
            profile_factor = float(profile_row[gen.profile_name])
            gen.p_max_available = gen.p_max * profile_factor
            gen.q_max_available = gen.q_max * profile_factor
            gen.q_min_available = gen.q_min * profile_factor

        return PowerFlowCase(
            buses=[replace(bus) for bus in static_buses],
            branches=[replace(branch) for branch in static_branches],
            generators=generators,
            loads=loads,
            root_bus=static_root,
            base_mva=static_base_mva,
        )

    _build_at.snapshot_count = len(p_load_df)  # type: ignore[attr-defined]
    return _build_at

