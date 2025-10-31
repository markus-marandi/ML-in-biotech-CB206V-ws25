import os
import glob
import gzip
import shutil
import pandas as pd
from dotenv import load_dotenv
import re
import numpy as np
from typing import Any

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from alphagenome.models import dna_client as _dc


VAR_TSV  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal/muscle_skeletal_variants_all.tsv.gz"
OUT_TSV  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal/muscle_skeletal_alphagenome_scores_all_aggs_variantids.tsv.gz"
OUT_DIR  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal/muscle_skeletal_alphagenome_batches_variantids"

CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

SEQ_LENS_GM = [131_072] # 16_384,  524_288,
SEQ_LEN_CM  = 131_072
BATCH       = 512
REVERSE = True

RNA = dna_client.OutputType.RNA_SEQ
Agg = variant_scorers.AggregationType
ORG = _dc.Organism.HOMO_SAPIENS

SCORERS_GENE = [
    variant_scorers.GeneMaskLFCScorer(requested_output=RNA),
]

SCORERS_CENTER = [
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.L2_DIFF_LOG1P),
]

RENAME_FRIENDLY = {
    "GeneMaskLFCScorer": "gene_exonmask_delta_log2",
    "CenterMaskScorer(width=2001).DIFF_MEAN": "center2.001kb_diff_mean",
    "CenterMaskScorer(width=10001).DIFF_MEAN": "center10.001kb_diff_mean",
    "CenterMaskScorer(width=100001).DIFF_MEAN": "center100.001kb_diff_mean",
    "CenterMaskScorer(width=2001).L2_DIFF_LOG1P": "center2.001kb_l2_log1p",
    "CenterMaskScorer(width=10001).L2_DIFF_LOG1P": "center10.001kb_l2_log1p",
    "CenterMaskScorer(width=100001).L2_DIFF_LOG1P": "center100.001kb_l2_log1p",
}

def _interval_to_str(x):
    if isinstance(x, str):
        return x
    # alphagenome Interval-like object
    if hasattr(x, "chromosome") and hasattr(x, "start") and hasattr(x, "end"):
        return f"{x.chromosome}:{int(x.start)}-{int(x.end)}:."
    return str(x)

def normalize_tidy(tidy: pd.DataFrame) -> pd.DataFrame:
    tidy = tidy.copy()
    if "variant_id" in tidy.columns:
        tidy["variant_id"] = tidy["variant_id"].map(lambda v: v.name if hasattr(v, "name") else str(v))
    elif "variant" in tidy.columns:
        tidy["variant_id"] = tidy["variant"].map(lambda v: v.name if hasattr(v, "name") else str(v))
    if "scored_interval" in tidy.columns:
        tidy["scored_interval_str"] = tidy["scored_interval"].map(_interval_to_str)
    else:
        tidy["scored_interval_str"] = pd.NA
    return tidy

def map_friendly(name: str) -> str:
    s = str(name)
    for k, v in RENAME_FRIENDLY.items():
        if k in s:
            return v
    return s

def strip_ensg(x):
    if pd.isna(x): return pd.NA
    m = re.match(r"(ENSG\d+)", str(x))
    return m.group(1) if m else pd.NA

load_dotenv()
API_KEY = os.getenv("API_KEY_PERSONAL")
assert API_KEY, "Set API_KEY_SCILIFELAB in your .env"
model = dna_client.create(api_key=API_KEY)

def inject_into_anndata(scores, meta_df):
    """push variant metadata into AnnData.obs for tidy_scores.

    scores: list[list[AnnData]] as returned by score_variants.
    meta_df: dataframe slice aligned with the outer list.
    """
    for per_var, (_, row) in zip(scores, meta_df.iterrows()):
        if not isinstance(per_var, (list, tuple)):
            per_var = [per_var]
        for ad in per_var:
            ad.obs["variant_id"] = str(row.variant_id)
            ad.obs["CHROM"] = str(row.CHROM)
            ad.obs["POS"] = int(row.POS)
            ad.obs["REF"] = str(row.REF)
            ad.obs["ALT"] = str(row.ALT)
            ad.obs["gene_tag"] = str(row.gene_tag) if pd.notna(row.gene_tag) else ""

# ------------------------------------------------------------------
# helper to flatten non-gene-centric scorer outputs (center-mask, etc.)
# ------------------------------------------------------------------


def scores_to_df(scores, meta_df):
    """flatten center-mask outputs into tidy rows.

    args:
        scores: nested list returned by score_variants.
        meta_df: dataframe slice with the same ordering as the outer list.
    returns:
        pd.DataFrame with one row per (gene_idx, track_idx).
    """

    rows: list[dict[str, object]] = []

    for per_variant, (_, mrow) in zip(scores, meta_df.iterrows()):
        if not isinstance(per_variant, (list, tuple)):
            per_variant = [per_variant]

        for ad_obj in per_variant:
            X = np.asarray(ad_obj.X)
            if X.ndim == 1:
                X = X[None, :]

            n_genes, n_tracks = X.shape

            interval = ad_obj.uns.get("scored_interval", None)
            iv_str = _interval_to_str(interval) if interval is not None else pd.NA
            scorer_name = str(ad_obj.uns.get("variant_scorer", ""))
            out_type = str(ad_obj.uns.get("output_type", ""))

            track_meta = getattr(ad_obj, "var", pd.DataFrame(index=range(n_tracks)))

            for g_idx in range(n_genes):
                gene_row = ad_obj.obs.iloc[g_idx] if g_idx < ad_obj.obs.shape[0] else {}
                for t_idx in range(n_tracks):
                    val = float(X[g_idx, t_idx])
                    trow = track_meta.iloc[t_idx] if t_idx < len(track_meta) else {}

                    rows.append({
                        "variant_id": str(mrow.variant_id),
                        "scored_interval_str": iv_str,
                        "output_type": out_type,
                        "variant_scorer": scorer_name,
                        "track_name": trow.get("name", pd.NA),
                        "track_strand": trow.get("strand", pd.NA),
                        "Assay title": trow.get("assay_title", pd.NA),
                        "ontology_curie": trow.get("ontology_curie", pd.NA),
                        "biosample_name": trow.get("biosample_name", pd.NA),
                        "biosample_type": trow.get("biosample_type", pd.NA),
                        "gtex_tissue": trow.get("gtex_tissue", pd.NA),
                        "raw_score": val,
                        "gene_id": gene_row.get("gene_id", pd.NA),
                        "gene_name": gene_row.get("gene_name", pd.NA),
                        "gene_type": gene_row.get("gene_type", pd.NA),
                        "gene_strand": gene_row.get("gene_strand", pd.NA),
                        "CHROM": mrow.CHROM,
                        "POS": int(mrow.POS),
                        "REF": mrow.REF,
                        "ALT": mrow.ALT,
                        "gene_tag": mrow.gene_tag,
                    })

    return pd.DataFrame(rows)

def _ensure_variant_ids(df: pd.DataFrame, interval_to_varid: dict[str, str]) -> None:
    """fill missing variant_id values from scored_interval_str using a lookup map.

    modifies df in-place.
    """
    if "variant_id" not in df.columns:
        df["variant_id"] = pd.NA
    mask = df["variant_id"].isna() | (df["variant_id"] == "")
    if mask.any():
        df.loc[mask, "variant_id"] = df.loc[mask, "scored_interval_str"].map(interval_to_varid)

df = pd.read_csv(VAR_TSV, sep="\t", compression="infer", low_memory=False)
need_base = {"CHROM", "POS", "REF", "ALT", "gene_tag"}
missing = need_base - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {VAR_TSV}: {missing}")

if "variant_id" not in df.columns:
    df["variant_id"] = (
        df["CHROM"].astype(str)
        + ":" + pd.to_numeric(df["POS"], errors="coerce").astype("Int64").astype(str)
        + ":" + df["REF"].astype(str) + ">" + df["ALT"].astype(str)
    )

df = df.loc[:, ["variant_id", "gene_tag", "CHROM", "POS", "REF", "ALT"]].copy()
df["POS"] = pd.to_numeric(df["POS"], errors="coerce")
df = df.dropna(subset=["POS"]).copy()
df["POS"] = df["POS"].astype(int)

def make_intervals(seq_len: int, chroms, poses):
    half = seq_len // 2
    poses = np.asarray(poses, dtype=int)
    starts = np.maximum(poses - half, 1)
    ends   = poses + half
    return [
        genome.Interval(chromosome=str(c), start=int(s), end=int(e))
        for c, s, e in zip(chroms, starts, ends)
    ]

def make_variants(chroms, poses, refs, alts, names):
    return [genome.Variant(chromosome=str(c),
                           position=int(p),
                           reference_bases=str(rf),
                           alternate_bases=str(al),
                           name=str(nm))
            for c, p, rf, al, nm in zip(chroms, poses, refs, alts, names)]

def chunk_path(s, e):
    return os.path.join(CHUNK_DIR, f"chunk_{s:07d}_{e:07d}.tsv.gz")

done = set()
for p in glob.glob(os.path.join(CHUNK_DIR, "chunk_*.tsv.gz")):
    b = os.path.basename(p)
    s, e = b.replace("chunk_", "").replace(".tsv.gz", "").split("_")
    done.add((int(s), int(e)))

n = len(df)
for start in range(0, n, BATCH):
    end = min(start + BATCH, n)
    if (start, end) in done:
        print(f"Skipping existing chunk {start}-{end}")
        continue

    meta = df.iloc[start:end].copy()
    intervals_cm = make_intervals(SEQ_LEN_CM, meta["CHROM"].to_numpy(), meta["POS"].to_numpy())
    variants_chunk = make_variants(meta["CHROM"], meta["POS"], meta["REF"], meta["ALT"], meta["variant_id"])
    cm_interval_to_varid = { _interval_to_str(iv): vid for iv, vid in zip(intervals_cm, meta["variant_id"]) }

    gm_frames = []
    for L in SEQ_LENS_GM:
        iv = make_intervals(L, meta["CHROM"].to_numpy(), meta["POS"].to_numpy())
        gm_interval_to_varid = { _interval_to_str(ivv): vid for ivv, vid in zip(iv, meta["variant_id"]) }
        scores = model.score_variants(
            intervals=iv,
            variants=variants_chunk,
            variant_scorers=[variant_scorers.GeneMaskLFCScorer(requested_output=RNA)],
            organism=ORG,
            progress_bar=False,
        )
        inject_into_anndata(scores, meta)
        tidy = variant_scorers.tidy_scores(scores, match_gene_strand=False)
        tidy = normalize_tidy(tidy)
        _ensure_variant_ids(tidy, gm_interval_to_varid)
        tidy["seq_len"] = L
        gm_frames.append(tidy)

    gm_all = pd.concat(gm_frames, ignore_index=True)

    # gene-masked on 131k interval length
    scores_gene_cm = model.score_variants(
        intervals=intervals_cm,
        variants=variants_chunk,
        variant_scorers=SCORERS_GENE,
        organism=ORG,
        progress_bar=False,
    )
    inject_into_anndata(scores_gene_cm, meta)
    gm_cm = variant_scorers.tidy_scores(scores_gene_cm, match_gene_strand=False)
    gm_cm = normalize_tidy(gm_cm)
    _ensure_variant_ids(gm_cm, cm_interval_to_varid)
    gm_cm["seq_len"] = SEQ_LEN_CM

    gm_all = pd.concat([gm_all, gm_cm], ignore_index=True)

    # center-mask scorers
    scores_cm_center = model.score_variants(
        intervals=intervals_cm,
        variants=variants_chunk,
        variant_scorers=SCORERS_CENTER,
        organism=ORG,
        progress_bar=False,
    )
    cm = scores_to_df(scores_cm_center, meta)
    _ensure_variant_ids(cm, cm_interval_to_varid)
    cm["seq_len"] = SEQ_LEN_CM


    # meta columns already present from scores_to_df

    gm_all["gene_id"] = gm_all.get("gene_id", pd.Series(pd.NA, index=gm_all.index)).map(strip_ensg)

    vm_gid = (gm_all.dropna(subset=["variant_id", "gene_id"])
              .groupby("variant_id")["gene_id"]
              .agg(lambda s: ";".join(sorted(set(s)))))

    vm_gname = (gm_all.dropna(subset=["variant_id", "gene_name"])
                .groupby("variant_id")["gene_name"]
                .agg(lambda s: ";".join(sorted(set(map(str, s))))))

    # Use the HASHABLE key produced by normalize_tidy
    si_gid = (gm_all.dropna(subset=["scored_interval_str", "gene_id"])
              .groupby("scored_interval_str")["gene_id"]
              .agg(lambda s: ";".join(sorted(set(s)))))

    si_gname = (gm_all.dropna(subset=["scored_interval_str", "gene_name"])
                .groupby("scored_interval_str")["gene_name"]
                .agg(lambda s: ";".join(sorted(set(map(str, s))))))

    for c in ["gene_id", "gene_name", "gene_tag"]:
        if c not in cm.columns:
            cm[c] = pd.NA

    miss = cm["gene_id"].isna()
    cm.loc[miss, "gene_id"] = cm.loc[miss, "variant_id"].map(vm_gid)
    miss = cm["gene_id"].isna()
    cm.loc[miss, "gene_id"] = cm.loc[miss, "scored_interval_str"].map(si_gid)
    cm["gene_id"] = cm["gene_id"].map(strip_ensg)

    miss = cm["gene_name"].isna()
    cm.loc[miss, "gene_name"] = cm.loc[miss, "variant_id"].map(vm_gname)
    miss = cm["gene_name"].isna()
    cm.loc[miss, "gene_name"] = cm.loc[miss, "scored_interval_str"].map(si_gname)

    miss = cm["gene_tag"].isna()
    cm.loc[miss, "gene_tag"] = cm.loc[miss, "variant_id"].map(vm_gid)
    miss = cm["gene_tag"].isna()
    cm.loc[miss, "gene_tag"] = cm.loc[miss, "scored_interval_str"].map(si_gid)

    for t in (gm_all, cm):
        t["scorer_friendly"] = t["variant_scorer"].map(map_friendly)

    out_chunk = chunk_path(start, end)
    tmp_chunk = out_chunk + ".tmp"
    pd.concat([gm_all, cm], ignore_index=True).to_csv(tmp_chunk, sep="\t", index=False, compression="gzip")
    os.replace(tmp_chunk, out_chunk)
    print(f"Scored {end}/{n} → wrote {os.path.basename(out_chunk)}")

chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.tsv.gz")))
assert chunk_files, "No chunk files found — nothing to stitch."

tmp_final = OUT_TSV + ".tmp"
with gzip.open(tmp_final, "wt") as w:
    wrote_header = False
    for p in chunk_files:
        with gzip.open(p, "rt") as r:
            if not wrote_header:
                shutil.copyfileobj(r, w)
                wrote_header = True
            else:
                r.readline()  # skip header
                shutil.copyfileobj(r, w)

os.replace(tmp_final, OUT_TSV)
print(f"Done. Wrote {OUT_TSV} from {len(chunk_files)} chunks")