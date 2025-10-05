"""
Build promoter- and enhancer-aware target regions for testis genes (symbols or ENSG IDs).

Outputs:
- testis_tss±2kb.bed
- testis_tss±10kb.bed
- testis_genespan±25kb.bed
- testis_target_regions_merged.bed  (TSS±2kb ∪ TSS±10kb ∪ Testis eQTL sites)
- testis_target_regions_qc.tsv
"""

import os
import re
import sys
from typing import Optional, Dict

import pandas as pd
from tqdm import tqdm


gene_list_path = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/initial/testis_genes.txt"  # symbols in your file
mane_path      = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/initial/MANE.GRCh38.v1.4.summary.txt"
eqtl_parquet   = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/initial/Testis.v10.eQTLs.signif_pairs.parquet"

out_dir        = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate"
os.makedirs(out_dir, exist_ok=True)

out_bed_tss2   = os.path.join(out_dir, "testis_tss±2kb.bed")
out_bed_tss10  = os.path.join(out_dir, "testis_tss±10kb.bed")
out_bed_span25 = os.path.join(out_dir, "testis_genespan±25kb.bed")
out_bed_target = os.path.join(out_dir, "testis_target_regions_merged.bed")
out_qc_tsv     = os.path.join(out_dir, "testis_target_regions_qc.tsv")

# window sizes
PAD_TSS_CORE   = 2_000
PAD_TSS_PROX   = 10_000
PAD_SPAN       = 25_000

CANON_CHR_RE = re.compile(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$")


def strip_ens_version(x: str) -> str:
    return re.sub(r"\.\d+$", "", x) if isinstance(x, str) else x

def looks_like_ensg(s: str) -> bool:
    return isinstance(s, str) and s.startswith("ENSG")

def nc_to_ucsc_chr(x: str) -> Optional[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str) and x.startswith("chr"):
        return x
    m = re.match(r"^NC_0+(\d+)\.\d+$", str(x))
    if not m:
        # Some MANE exports already have numeric chr (e.g., "17" or "X")
        s = str(x)
        if s in {"X", "Y"}:
            return f"chr{s}"
        if s.isdigit():
            return f"chr{s}"
        return None
    n = int(m.group(1))
    if n == 23: return "chrX"
    if n == 24: return "chrY"
    return f"chr{n}"

def ensure_bed4(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    bed = df.rename(columns={"chrom": "chr"}).loc[:, ["chr", "start", "end", name_col]].copy()
    bed.columns = ["chr", "start", "end", "name"]
    bed["start"] = pd.to_numeric(bed["start"], errors="coerce").fillna(0).astype(int)
    bed["end"]   = pd.to_numeric(bed["end"], errors="coerce").fillna(0).astype(int)
    bed = bed[bed["end"] > bed["start"]]
    bed.loc[bed["start"] < 0, "start"] = 0
    bed = bed.sort_values(["chr", "start", "end", "name"]).reset_index(drop=True)
    return bed

def merge_bed_like(bed: pd.DataFrame) -> pd.DataFrame:
    merged = []
    for (chrom, name), grp in bed.groupby(["chr", "name"], sort=False):
        s0, e0 = None, None
        for _, r in grp.sort_values(["start", "end"]).iterrows():
            s, e = int(r["start"]), int(r["end"])
            if s0 is None:
                s0, e0 = s, e
            elif s <= e0:
                e0 = max(e0, e)
            else:
                merged.append([chrom, s0, e0, name])
                s0, e0 = s, e
        if s0 is not None:
            merged.append([chrom, s0, e0, name])
    return pd.DataFrame(merged, columns=["chr", "start", "end", "name"])

def qc_report(label: str, bed: pd.DataFrame):
    print(f"\n[QC] {label}")
    print(f"  intervals: {len(bed):,}")
    if len(bed):
        lengths = bed["end"] - bed["start"]
        print(f"  length (bp): min={lengths.min():,} median={int(lengths.median()):,} max={lengths.max():,}")
        print(f"  unique genes: {bed['name'].nunique():,}")
        bad = ((bed["end"] <= bed["start"]) | (bed["start"] < 0)).sum()
        if bad:
            print(f"  [WARN] {bad} malformed intervals (end<=start or negative start)")

# gene list
symbols_or_ids = pd.read_csv(gene_list_path, header=None, names=["input"], dtype=str, comment="#")["input"].str.strip()
symbols_or_ids = symbols_or_ids[symbols_or_ids != ""].dropna().drop_duplicates()
print(f"[INFO] Loaded {len(symbols_or_ids)} inputs from: {gene_list_path}")

num_ensg = sum(looks_like_ensg(x) for x in symbols_or_ids)
mode = "ENSG" if num_ensg == len(symbols_or_ids) else ("MIXED" if 0 < num_ensg < len(symbols_or_ids) else "SYMBOL")
print(f"[INFO] Detected input mode: {mode}")

# mane
mane = pd.read_csv(mane_path, sep="\t", dtype=str)
print(f"[INFO] MANE columns: {list(mane.columns)[:10]} ... total={len(mane.columns)}")

# Column detection
col_map: Dict[str, Optional[str]] = {
    "ensg": None,
    "symbol": None,
    "chrom": None,
    "start": None,
    "end": None,
    "strand": None,
}

# Ensembl Gene ID
for c in ["Ensembl_Gene", "Ensembl gene", "Ensembl gene ID", "EnsemblGene"]:
    if c in mane.columns:
        col_map["ensg"] = c
        break

# Symbol
for c in ["Approved_Symbol", "HGNC_symbol", "symbol", "Gene", "Approved symbol", "HGNC Symbol"]:
    if c in mane.columns:
        col_map["symbol"] = c
        break

# Chromosome (GRCh38)
for c in ["GRCh38_chr", "chr", "GRCh38 chromosome", "Chromosome"]:
    if c in mane.columns:
        col_map["chrom"] = c
        break

# Coordinates
for c in ["chr_start", "GRCh38_start", "start"]:
    if c in mane.columns:
        col_map["start"] = c
        break
for c in ["chr_end", "GRCh38_end", "end"]:
    if c in mane.columns:
        col_map["end"] = c
        break

# Strand
for c in ["chr_strand", "strand", "GRCh38_strand"]:
    if c in mane.columns:
        col_map["strand"] = c
        break

missing_keys = [k for k, v in col_map.items() if v is None and k in ("ensg","chrom","start","end","strand")]
if missing_keys:
    print(f"[ERROR] MANE missing required columns: {missing_keys}")
    sys.exit(1)

mane_use = mane[[col_map["ensg"], col_map.get("symbol"), col_map["chrom"], col_map["start"], col_map["end"], col_map["strand"]]].copy()
mane_use.columns = ["Ensembl_Gene", "Symbol", "GRCh38_chr", "chr_start", "chr_end", "chr_strand"]

mane_use["Ensembl_Gene"] = mane_use["Ensembl_Gene"].map(strip_ens_version)
mane_use["chrom"]        = mane_use["GRCh38_chr"].map(nc_to_ucsc_chr)
mane_use["chr_start"]    = pd.to_numeric(mane_use["chr_start"], errors="coerce")
mane_use["chr_end"]      = pd.to_numeric(mane_use["chr_end"], errors="coerce")

mane_use = mane_use.dropna(subset=["chrom", "chr_start", "chr_end", "chr_strand"])
mane_use = mane_use[mane_use["chrom"].str.match(CANON_CHR_RE)].copy()

# TSS
mane_use["TSS"] = mane_use.apply(lambda r: int(r["chr_start"]) if r["chr_strand"] == "+" else int(r["chr_end"]), axis=1)

# Deduplicate (keep longest span if multiple)
mane_use["span"] = mane_use["chr_end"] - mane_use["chr_start"]
mane_use = mane_use.sort_values(["Ensembl_Gene", "span"], ascending=[True, False]).drop_duplicates("Ensembl_Gene", keep="first")

# input mapping
if mode == "ENSG":
    in_df = pd.DataFrame({"Ensembl_Gene": symbols_or_ids.map(strip_ens_version)})
    merged = in_df.merge(mane_use, on="Ensembl_Gene", how="left")
elif mode == "SYMBOL":
    if "Symbol" not in mane_use.columns or mane_use["Symbol"].isna().all():
        print("[ERROR] Your gene list is SYMBOLs but MANE lacks a symbol column I recognize.")
        sys.exit(1)
    in_df = pd.DataFrame({"Symbol": symbols_or_ids})
    merged = in_df.merge(mane_use, on="Symbol", how="left")
else:  # MIXED
    in_df = pd.DataFrame({"raw": symbols_or_ids})
    in_df["Ensembl_Gene"] = in_df["raw"].where(in_df["raw"].map(looks_like_ensg), None).map(strip_ens_version)
    in_df["Symbol"] = in_df["raw"].where(~in_df["raw"].map(looks_like_ensg), None)
    merged = in_df.merge(mane_use, on=["Ensembl_Gene","Symbol"], how="left")  # will only match one of the keys

missing = merged["chrom"].isna().sum()
if missing:
    print(f"[WARN] {missing} input gene(s) not found in MANE mapping. Examples:")
    print(merged.loc[merged["chrom"].isna(), ["Ensembl_Gene","Symbol"]].head())
merged = merged.dropna(subset=["chrom"]).copy()

# Build a 'name' column (prefer Ensembl; append symbol if present)
merged["name"] = merged["Ensembl_Gene"].fillna(merged["Symbol"])
if "Symbol" in merged.columns:
    merged["name"] = merged.apply(
        lambda r: f"{r['Ensembl_Gene']}|{r['Symbol']}" if pd.notna(r.get("Ensembl_Gene")) and pd.notna(r.get("Symbol")) else r["name"],
        axis=1
    )

print(f"[INFO] Successfully mapped {merged.shape[0]} genes.")

# A) TSS +-2 kb
tss2 = merged.apply(
    lambda r: pd.Series({"chrom": r["chrom"], "start": max(0, int(r["TSS"]) - PAD_TSS_CORE), "end": int(r["TSS"]) + PAD_TSS_CORE, "name": r["name"]}),
    axis=1
)
tss2 = ensure_bed4(tss2)

# B) TSS +-10 kb
tss10 = merged.apply(
    lambda r: pd.Series({"chrom": r["chrom"], "start": max(0, int(r["TSS"]) - PAD_TSS_PROX), "end": int(r["TSS"]) + PAD_TSS_PROX, "name": r["name"]}),
    axis=1
)
tss10 = ensure_bed4(tss10)

# C) Gene span +-25 kb
span25 = merged.apply(
    lambda r: pd.Series({"chrom": r["chrom"], "start": max(0, int(r["chr_start"]) - PAD_SPAN), "end": int(r["chr_end"]) + PAD_SPAN, "name": r["name"]}),
    axis=1
)
span25 = ensure_bed4(span25)

# INTEGRATE eQTL sites

def load_testis_eqtls(parquet_path: str) -> pd.DataFrame:
    eq = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in eq.columns}
    # chrom
    chr_col = cols.get("chrom") or cols.get("chr") or cols.get("variant_chrom") or None
    pos_col = cols.get("position") or cols.get("pos") or cols.get("variant_pos") or None
    gene_id_col = cols.get("gene_id") or cols.get("gene") or None
    gene_sym_col = cols.get("gene_symbol") or cols.get("symbol") or None
    if not chr_col or not pos_col:
        raise ValueError("Cannot find chrom/pos in eQTL parquet.")
    out = eq[[chr_col, pos_col]].copy()
    out.columns = ["chr", "pos"]
    if gene_id_col:  out["gene_id"] = eq[gene_id_col].astype(str).map(strip_ens_version)
    if gene_sym_col: out["gene_symbol"] = eq[gene_sym_col].astype(str)
    out["chr"] = out["chr"].astype(str).apply(lambda x: x if x.startswith("chr") else f"chr{x}")
    out = out[out["chr"].str.match(CANON_CHR_RE)]
    out = out[pd.to_numeric(out["pos"], errors="coerce").notna()]
    out["pos"] = out["pos"].astype(int)
    return out.reset_index(drop=True)

try:
    eqtl_df = load_testis_eqtls(eqtl_parquet)
    print(f"[INFO] Loaded Testis eQTL rows: {len(eqtl_df)}")
except Exception as e:
    print(f"[WARN] Could not parse eQTL parquet: {e}")
    eqtl_df = pd.DataFrame(columns=["chr","pos","gene_id","gene_symbol"])

# Build mapping from Ensembl  and display name used in BEDs
ensg_to_name = {}
for _, r in merged.iterrows():
    ensg = r.get("Ensembl_Gene")
    if pd.isna(ensg): continue
    ensg_to_name[ensg] = r["name"]

# Match eQTLs to our genes
eq_match = pd.DataFrame()
if not eqtl_df.empty:
    if "gene_id" in eqtl_df.columns:
        eq_match = eqtl_df[eqtl_df["gene_id"].isin(set(ensg_to_name.keys()))].copy()
        eq_match["name"] = eq_match["gene_id"].map(ensg_to_name)
    elif "gene_symbol" in eqtl_df.columns and "Symbol" in merged.columns:
        sym_to_name = dict(zip(merged["Symbol"], merged["name"]))
        eq_match = eqtl_df[eqtl_df["gene_symbol"].isin(set(sym_to_name.keys()))].copy()
        eq_match["name"] = eq_match["gene_symbol"].map(sym_to_name)
    if not eq_match.empty:
        eq_match["start"] = (eq_match["pos"] - 1).clip(lower=0)
        eq_match["end"]   = eq_match["pos"]
        eq_match = eq_match[["chr","start","end","name"]]
        eq_match = ensure_bed4(eq_match)
print(f"[INFO] eQTL sites matched to {eq_match['name'].nunique() if not eq_match.empty else 0} gene(s).")


parts = [tss2, tss10]
if not eq_match.empty:
    parts.append(eq_match)

target = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["chr","start","end","name"])
target = target.sort_values(["chr","name","start","end"]).reset_index(drop=True)
if not target.empty:
    target = merge_bed_like(target)


qc_report("TSS±2kb", tss2)
qc_report("TSS±10kb", tss10)
qc_report("GeneSpan±25kb", span25)
qc_report("TARGET (promoters ∪ eQTLs)", target)

# per-gene target length QC
if not target.empty:
    per_gene = target.assign(length=lambda d: d["end"] - d["start"]).groupby("name", as_index=False)["length"].sum()
    per_gene = per_gene.rename(columns={"length": "target_bp"}).sort_values("target_bp", ascending=False)
    per_gene.to_csv(out_qc_tsv, sep="\t", index=False)
    print(f"[INFO] Wrote per-gene target QC → {out_qc_tsv}")

tss2.to_csv(out_bed_tss2, sep="\t", header=False, index=False)
tss10.to_csv(out_bed_tss10, sep="\t", header=False, index=False)
span25.to_csv(out_bed_span25, sep="\t", header=False, index=False)
target.to_csv(out_bed_target, sep="\t", header=False, index=False)

print("\n[OK] BEDs written:")
print(f"  - {out_bed_tss2}")
print(f"  - {out_bed_tss10}")
print(f"  - {out_bed_span25}")
print(f"  - {out_bed_target}")