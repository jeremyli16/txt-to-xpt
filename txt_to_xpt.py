"""
txt_to_xpt.py
-------------
Converts a delimited text file (millions of rows, hundreds of variables)
to a SAS Transport (XPT) file that follows SAS Version 5 Transport conventions.

Usage
-----
# Auto mode — picks first .txt/.csv from Inputs/, writes to Outputs/
python txt_to_xpt.py

# Explicit input, auto output to Outputs/
python txt_to_xpt.py Inputs/myfile.txt

# Fully explicit
python txt_to_xpt.py Inputs/myfile.txt Outputs/myfile.xpt

# With options
python txt_to_xpt.py Inputs/myfile.txt Outputs/myfile.xpt --delimiter "," --dataset-name MYDATA

Options
-------
--delimiter DELIM     Field delimiter (default: auto-detect from tab, comma, pipe, semicolon)
--encoding ENC        Input file encoding (default: utf-8)
--chunksize N         Rows to read per chunk (default: 100000)
--dataset-name NAME   SAS dataset name, max 8 chars (default: derived from filename)
--label TEXT          Dataset label, max 40 chars (default: empty)
--var-map FILE        Optional JSON file mapping column -> {name, label, format, type}
--date-cols COL,...   Comma-separated column names to parse as SAS dates
--missing-values V,.. Comma-separated string values treated as missing (default: .,NA,NaN,"")

SAS XPT Conventions Enforced
-----------------------------
- Dataset name    : uppercase, max 8 chars, starts with letter/underscore
- Variable names  : uppercase, max 8 chars, starts with letter/underscore, alphanumeric+underscore
- Variable labels : max 40 chars (truncated with warning)
- Format names    : max 8 chars
- Numeric columns : stored as 8-byte IEEE double (SAS default)
- Character cols  : stored as fixed-width, width = max observed length (max 200 bytes)
- SAS dates       : stored as days since 1960-01-01
- SAS datetimes   : stored as seconds since 1960-01-01 00:00:00
- Missing numeric : stored as NaN (SAS displays as ".")
- Missing char    : stored as spaces
"""

import argparse
import json
import logging
import math
import os
import re
import struct
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAS conventions
# ---------------------------------------------------------------------------
SAS_EPOCH = date(1960, 1, 1)
SAS_EPOCH_DT = datetime(1960, 1, 1)
MAX_NAME_LEN = 8
MAX_LABEL_LEN = 40
MAX_DATASET_NAME_LEN = 8
MAX_CHAR_WIDTH = 200          # practical cap; SAS itself allows up to 32767
XPT_VERSION = b"5"


def _sas_name(raw: str, seen: set, kind: str = "variable") -> str:
    """
    Convert an arbitrary string to a valid SAS name:
      - uppercase
      - max 8 characters
      - only letters, digits, underscores
      - must start with a letter or underscore
    Deduplicates by appending a numeric suffix.
    """
    name = raw.upper()
    name = re.sub(r"[^A-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    if not name:
        name = "VAR"
    name = name[:MAX_NAME_LEN]

    base = name
    suffix = 1
    while name in seen:
        tail = str(suffix)
        name = base[: MAX_NAME_LEN - len(tail)] + tail
        suffix += 1

    seen.add(name)
    if name != raw.upper()[:MAX_NAME_LEN]:
        log.warning("  %s name '%s' → '%s'", kind, raw, name)
    return name


def _sas_label(text: str) -> str:
    if len(text) > MAX_LABEL_LEN:
        log.warning("Label truncated: '%s'", text[:60])
    return text[:MAX_LABEL_LEN]


def _days_since_sas_epoch(val) -> Optional[float]:
    """Convert a date-like value to SAS date (days since 1960-01-01)."""
    if pd.isna(val):
        return float("nan")
    if isinstance(val, (pd.Timestamp, datetime)):
        d = val.date() if isinstance(val, datetime) else val
        return float((d - SAS_EPOCH).days)
    return float("nan")


# ---------------------------------------------------------------------------
# XPT writer (SAS Version 5 Transport format, pure Python)
# ---------------------------------------------------------------------------
# Reference: SAS Technical Support TS-140 / SAS Institute documentation.

def _pad(s: bytes, length: int, pad: bytes = b" ") -> bytes:
    return s[:length].ljust(length, pad)


def _xpt_header_record(name: bytes) -> bytes:
    """80-byte HEADER RECORD with given name."""
    return _pad(b"HEADER RECORD*******" + name + b"HEADER RECORD!!!!!!!000000000000000000000000000000", 80)


def _ieee_to_ibm(value: float) -> bytes:
    """Convert IEEE 754 double to IBM mainframe 8-byte real (SAS XPT numeric)."""
    if math.isnan(value):
        return b"\x2e" + b"\x00" * 7   # SAS missing (.)

    if value == 0.0:
        return b"\x00" * 8

    sign = 0 if value > 0 else 1
    value = abs(value)

    # IEEE exponent base-2, IBM exponent base-16
    ieee_bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    ieee_exp = ((ieee_bits >> 52) & 0x7FF) - 1023   # unbiased
    ieee_mant = (ieee_bits & 0x000FFFFFFFFFFFFF) | 0x0010000000000000

    # Convert to base-16 exponent
    # IBM: value = mantissa * 16^(exp-64)  mantissa in [1/16, 1)
    ibm_exp = math.ceil((ieee_exp + 1) / 4)
    mant_shift = ibm_exp * 4 - (ieee_exp + 1)
    mantissa = ieee_mant << mant_shift     # 56-bit mantissa

    ibm_exp_biased = ibm_exp + 64
    if ibm_exp_biased < 0:
        return b"\x00" * 8
    if ibm_exp_biased > 127:
        # Overflow → largest representable
        ibm_exp_biased = 127
        mantissa = (1 << 56) - 1

    byte0 = (sign << 7) | (ibm_exp_biased & 0x7F)
    mant_bytes = (mantissa >> 8) & 0xFFFFFFFFFFFFFF   # top 7 bytes of 56-bit

    result = struct.pack(">Q", (byte0 << 56) | mant_bytes)
    return result


class XPTWriter:
    """
    Writes a single-dataset SAS Version 5 XPT file.

    Parameters
    ----------
    path        : output file path
    dataset_name: SAS dataset name (≤8 uppercase chars)
    label       : dataset label (≤40 chars)
    variables   : list of dicts with keys:
                    name   (str, ≤8), label (str, ≤40),
                    type   ('N' or 'C'),
                    length (int; 8 for numeric, 1-200 for char),
                    format (str, ≤8, optional)
    """

    def __init__(self, path: str, dataset_name: str, label: str, variables: list):
        self.path = path
        self.dataset_name = _pad(dataset_name.encode("ascii", "replace"), 8)
        self.label = _pad(label.encode("ascii", "replace"), 40)
        self.variables = variables
        self._fh = None
        self._obs_count = 0
        self._row_len = sum(v["length"] for v in variables)

    # -- lifecycle --
    def __enter__(self):
        self._fh = open(self.path, "wb")
        self._write_file_header()
        self._write_member_header()
        self._write_namestr_header()
        self._write_namestrs()
        self._write_obs_header()
        return self

    def __exit__(self, *_):
        self._finalize()
        self._fh.close()

    # -- header sections --
    def _write_file_header(self):
        now = datetime.utcnow()
        date_str = _pad(now.strftime("%d%b%y").upper().encode(), 8)
        time_str = _pad(now.strftime("%H:%M:%S").encode(), 8)

        rec1 = _pad(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!000000000000000000000000000000", 80)
        rec2 = _pad(b"SAS     SAS     SASLIB  " + XPT_VERSION + b"   " + b" " * 52, 80)
        rec3 = _pad(date_str + b" " * 8 + time_str + b" " * 16 + date_str + b" " * 32, 80)

        self._fh.write(rec1 + rec2 + rec3)

    def _write_member_header(self):
        now = datetime.utcnow()
        date_str = _pad(now.strftime("%d%b%y").upper().encode(), 8)
        time_str = _pad(now.strftime("%H:%M:%S").encode(), 8)
        modified = _pad((now.strftime("%d%b%y").upper() + ":" + now.strftime("%H:%M:%S")).encode(), 16)

        rec1 = _pad(b"HEADER RECORD*******MEMBER  HEADER RECORD!!!!!!!000000000000000001600000000140", 80)
        rec2 = _pad(b"HEADER RECORD*******DSCRPTR HEADER RECORD!!!!!!!000000000000000000000000000000", 80)
        rec3 = _pad(b"SAS     " + self.dataset_name + b"SASDATA " + XPT_VERSION + b"   " + b" " * 52, 80)
        # modified(16) + _(16) + label(40) + type(8) = 80  — layout pandas expects
        rec4 = _pad(modified + b" " * 16 + self.label + b" " * 8, 80)

        self._fh.write(rec1 + rec2 + rec3 + rec4)

    def _write_namestr_header(self):
        nvar = len(self.variables)
        rec = _pad(
            b"HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!000000%04d0000000000000000000000" % nvar,
            80,
        )
        self._fh.write(rec)

    def _write_namestrs(self):
        """Each NAMESTR is 140 bytes; padded to a multiple of 80."""
        namestr_bytes = b""
        pos = 0
        for i, var in enumerate(self.variables, start=1):
            ntype = 1 if var["type"] == "N" else 2
            length = var["length"]
            name = _pad(var["sas_name"].encode("ascii", "replace"), 8)
            label = _pad(var["label"].encode("ascii", "replace"), 40)
            fmt = _pad(var.get("format", "").encode("ascii", "replace"), 8)
            infmt = _pad(var.get("informat", "").encode("ascii", "replace"), 8)

            # XPT NAMESTR: 140 bytes big-endian
            # ntype(2) nhfun(2) nlng(2) nvar0(2) nname(8) nlabel(40)
            # nform(8) nfl(2) nfd(2) nfj(2) nfill(2) niform(8)
            # nifl(2) nifd(2) npos(4) pad(52) = 140
            ns = (
                struct.pack(">h", ntype)    # 2  ntype
                + struct.pack(">h", 0)      # 2  nhfun
                + struct.pack(">h", length) # 2  nlng
                + struct.pack(">h", i)      # 2  nvar0
                + name                      # 8  nname
                + label                     # 40 nlabel
                + fmt                       # 8  nform
                + struct.pack(">h", 0)      # 2  nfl
                + struct.pack(">h", 0)      # 2  nfd
                + struct.pack(">h", 0)      # 2  nfj
                + b"\x00\x00"              # 2  nfill
                + infmt                     # 8  niform
                + struct.pack(">h", 0)      # 2  nifl
                + struct.pack(">h", 0)      # 2  nifd
                + struct.pack(">l", pos)    # 4  npos
                + b"\x00" * 52             # 52 padding
            )
            assert len(ns) == 140, f"Namestr length {len(ns)} != 140"
            namestr_bytes += ns
            pos += length

        # Pad to 80-byte boundary
        remainder = len(namestr_bytes) % 80
        if remainder:
            namestr_bytes += b" " * (80 - remainder)

        self._fh.write(namestr_bytes)

    def _write_obs_header(self):
        rec = _pad(b"HEADER RECORD*******OBS     HEADER RECORD!!!!!!!000000000000000000000000000000", 80)
        self._fh.write(rec)

    # -- data --
    def write_chunk(self, df: pd.DataFrame):
        """Encode and write a DataFrame chunk. Columns must match self.variables in order."""
        buf = bytearray()
        for _, row in df.iterrows():
            for var in self.variables:
                val = row[var["_col"]]   # original column reference
                if var["type"] == "N":
                    try:
                        f = float(val)
                        if math.isnan(f):
                            buf += b"\x2e" + b"\x00" * 7
                        else:
                            buf += _ieee_to_ibm(f)
                    except (TypeError, ValueError):
                        buf += b"\x2e" + b"\x00" * 7
                else:
                    length = var["length"]
                    if pd.isna(val) or val is None:
                        buf += b" " * length
                    else:
                        encoded = str(val).encode("ascii", "replace")[:length]
                        buf += encoded.ljust(length, b" ")

            self._obs_count += 1

        self._fh.write(bytes(buf))

    def _finalize(self):
        """Pad observation section to 80-byte boundary."""
        obs_bytes = self._obs_count * self._row_len
        remainder = obs_bytes % 80
        if remainder:
            self._fh.write(b" " * (80 - remainder))
        log.info("Wrote %d observations to %s", self._obs_count, self.path)


# ---------------------------------------------------------------------------
# Auto-detect delimiter
# ---------------------------------------------------------------------------
def _detect_delimiter(path: str, encoding: str) -> str:
    candidates = ["\t", ",", "|", ";"]
    with open(path, encoding=encoding, errors="replace") as fh:
        sample = fh.read(8192)
    counts = {d: sample.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    log.info("Auto-detected delimiter: %r (counts: %s)", best, counts)
    return best


# ---------------------------------------------------------------------------
# Schema inference pass (first chunksize rows)
# ---------------------------------------------------------------------------
def _infer_schema(
    path: str,
    delimiter: str,
    encoding: str,
    chunksize: int,
    missing_values: list,
    date_cols: list,
    var_map: dict,
) -> dict:
    """
    Returns a dict:
        col_original -> {sas_name, label, type, length, format, informat}
    """
    log.info("Schema inference pass (first %d rows)…", chunksize)
    sample = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        nrows=chunksize,
        na_values=missing_values,
        keep_default_na=True,
        low_memory=False,
    )

    seen_names: set = set()
    schema = {}

    for col in sample.columns:
        override = var_map.get(col, {})
        raw_name = override.get("name", col)
        sas_name = _sas_name(raw_name, seen_names)
        label = _sas_label(override.get("label", col))
        fmt = override.get("format", "")
        infmt = override.get("informat", "")

        forced_type = override.get("type", "").upper()

        if col in date_cols:
            vtype = "N"
            length = 8
            if not fmt:
                fmt = "DATE9"
        elif forced_type in ("N", "C"):
            vtype = forced_type
            if vtype == "N":
                length = 8
            else:
                maxlen = sample[col].dropna().astype(str).str.len().max()
                length = min(int(maxlen) if not math.isnan(maxlen) else 1, MAX_CHAR_WIDTH)
        else:
            s = sample[col]
            if pd.api.types.is_numeric_dtype(s):
                vtype = "N"
                length = 8
            else:
                vtype = "C"
                maxlen = s.dropna().astype(str).str.len().max()
                length = min(int(maxlen) if not math.isnan(maxlen) else 1, MAX_CHAR_WIDTH)
                length = max(length, 1)

        schema[col] = {
            "sas_name": sas_name,
            "label": label,
            "type": vtype,
            "length": length,
            "format": fmt,
            "informat": infmt,
            "_col": col,
        }

    log.info("Schema: %d variables (%d numeric, %d character)",
             len(schema),
             sum(1 for v in schema.values() if v["type"] == "N"),
             sum(1 for v in schema.values() if v["type"] == "C"))
    return schema


# ---------------------------------------------------------------------------
# Full-file character-width pass (optional, for exact char widths)
# ---------------------------------------------------------------------------
def _scan_char_widths(
    path: str,
    delimiter: str,
    encoding: str,
    chunksize: int,
    missing_values: list,
    schema: dict,
) -> None:
    """Update character variable lengths by scanning the entire file."""
    char_cols = [col for col, v in schema.items() if v["type"] == "C"]
    if not char_cols:
        return

    log.info("Scanning full file for max character widths…")
    reader = pd.read_csv(
        path,
        sep=delimiter,
        encoding=encoding,
        usecols=char_cols,
        chunksize=chunksize,
        na_values=missing_values,
        keep_default_na=True,
        low_memory=False,
    )
    for chunk in reader:
        for col in char_cols:
            if col in chunk.columns:
                mx = chunk[col].dropna().astype(str).str.len().max()
                if not math.isnan(mx):
                    schema[col]["length"] = max(schema[col]["length"], min(int(mx), MAX_CHAR_WIDTH))


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(args):
    # Resolve delimiter
    delimiter = args.delimiter or _detect_delimiter(args.input, args.encoding)
    missing_values = [v for v in (args.missing_values or ".,NA,NaN,").split(",")]
    date_cols = [c.strip() for c in (args.date_cols or "").split(",") if c.strip()]

    # Load optional variable map
    var_map = {}
    if args.var_map:
        with open(args.var_map) as f:
            var_map = json.load(f)

    # Dataset name
    dataset_name = args.dataset_name or Path(args.input).stem
    dataset_name = re.sub(r"[^A-Z0-9_]", "_", dataset_name.upper())
    if dataset_name and dataset_name[0].isdigit():
        dataset_name = "_" + dataset_name
    dataset_name = dataset_name[:MAX_DATASET_NAME_LEN] or "DATASET"

    label = (args.label or "")[:MAX_LABEL_LEN]

    # Schema inference
    schema = _infer_schema(
        args.input, delimiter, args.encoding,
        args.chunksize, missing_values, date_cols, var_map,
    )

    # Full-file width scan for character variables
    _scan_char_widths(
        args.input, delimiter, args.encoding,
        args.chunksize, missing_values, schema,
    )

    variables = list(schema.values())   # ordered as in file

    # Estimate output size
    row_len = sum(v["length"] for v in variables)
    log.info("Row byte length: %d", row_len)

    # Write XPT
    log.info("Writing XPT: %s (dataset=%s, label=%r)", args.output, dataset_name, label)
    with XPTWriter(args.output, dataset_name, label, variables) as writer:
        parse_dates_arg = date_cols if date_cols else False
        reader = pd.read_csv(
            args.input,
            sep=delimiter,
            encoding=args.encoding,
            chunksize=args.chunksize,
            na_values=missing_values,
            keep_default_na=True,
            low_memory=False,
            parse_dates=parse_dates_arg if parse_dates_arg else False,
        )

        chunk_num = 0
        for chunk in reader:
            chunk_num += 1
            if chunk_num % 10 == 0:
                log.info("  Processing chunk %d (~%d rows so far)…",
                         chunk_num, chunk_num * args.chunksize)

            # Convert date columns to SAS dates
            for col in date_cols:
                if col in chunk.columns:
                    chunk[col] = chunk[col].apply(_days_since_sas_epoch)

            # Ensure numeric columns are float, char are str
            for var in variables:
                col = var["_col"]
                if col not in chunk.columns:
                    continue
                if var["type"] == "N":
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                else:
                    chunk[col] = chunk[col].where(chunk[col].notna(), other=None)

            writer.write_chunk(chunk)

    size_mb = os.path.getsize(args.output) / 1_048_576
    log.info("Done. Output: %s (%.1f MB)", args.output, size_mb)

    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"  Input         : {args.input}")
    print(f"  Output        : {args.output}  ({size_mb:.1f} MB)")
    print(f"  Dataset name  : {dataset_name}")
    print(f"  Variables     : {len(variables)}")
    print(f"  Observations  : {writer._obs_count:,}")
    print(f"  Row byte len  : {row_len}")
    print("\n  First 10 variables:")
    for v in variables[:10]:
        print(f"    {v['sas_name']:<8}  {v['type']}  len={v['length']:<4}  label={v['label']!r}")
    if len(variables) > 10:
        print(f"    … and {len(variables)-10} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert a large delimited text file to a SAS XPT file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  nargs="?", default=None, help="Path to input .txt / .csv file (default: first file in Inputs/)")
    parser.add_argument("output", nargs="?", default=None, help="Path for output .xpt file (default: Outputs/<input_stem>.xpt)")
    parser.add_argument("--delimiter",     default=None,    help="Field delimiter (default: auto)")
    parser.add_argument("--encoding",      default="utf-8", help="Input encoding (default: utf-8)")
    parser.add_argument("--chunksize",     default=100_000, type=int, help="Rows per chunk (default: 100000)")
    parser.add_argument("--dataset-name",  default=None,    help="SAS dataset name ≤8 chars")
    parser.add_argument("--label",         default="",      help="Dataset label ≤40 chars")
    parser.add_argument("--var-map",       default=None,    help="JSON file with variable metadata")
    parser.add_argument("--date-cols",     default="",      help="Comma-separated date column names")
    parser.add_argument("--missing-values",default=".,NA,NaN,", help="Comma-separated missing value strings")

    args = parser.parse_args()

    if args.input is None:
        inputs_dir = Path("Inputs")
        candidates = sorted(inputs_dir.glob("*.[ct][sx][tv]")) if inputs_dir.exists() else []
        if not candidates:
            sys.exit("ERROR: No input file given and no .txt/.csv found in Inputs/")
        args.input = str(candidates[0])
        log.info("Using input: %s", args.input)

    if args.output is None:
        os.makedirs("Outputs", exist_ok=True)
        args.output = os.path.join("Outputs", Path(args.input).stem + ".xpt")

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")

    convert(args)


if __name__ == "__main__":
    main()
