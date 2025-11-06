import os
import glob
import gzip
import shutil
import math
import time
import pandas as pd
from dotenv import load_dotenv
import re
import numpy as np
from typing import Any

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from alphagenome.models import dna_client as _dc


VAR_TSV  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal/muscle_skeletal_variants_all.tsv.gz"
TSS_BED = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal/muscle_skeletal_tss±2kb.bed"
OUT_TSV  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/results/new_muscle_skeletal/muscle_skeletal_alphagenome_scores_all_aggs_variantids_long.tsv.gz"
OUT_DIR  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/results/new_muscle_skeletal/muscle_skeletal_alphagenome_batches_variantids_long"


CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

SEQ_LENS_GM = [1048576] # 16_384,  524_288,
SEQ_LEN_CM  = 1048576
BATCH       = 512
REVERSE = True

RNA = dna_client.OutputType.RNA_SEQ
Agg = variant_scorers.AggregationType
ORG = _dc.Organism.HOMO_SAPIENS

SCORERS_GENE = [
    variant_scorers.GeneMaskLFCScorer(requested_output=RNA),
]

SCORERS_CENTER = [
    #variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.DIFF_MEAN),
    #variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.L2_DIFF_LOG1P),
]

RENAME_FRIENDLY = {
    "GeneMaskLFCScorer": "gene_exonmask_delta_log2",
    "CenterMaskScorer(width=10001).DIFF_MEAN": "center10.001kb_diff_mean",
    "CenterMaskScorer(width=100001).DIFF_MEAN": "center100.001kb_diff_mean",
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

# helper to normalize gene tags to ensg core
ENS_RE = re.compile(r'(ENSG\d+)')

def ensg_core(x: str) -> str | None:
    # extract ensg id without version or symbol
    if x is None:
        return None
    m = ENS_RE.search(str(x))
    return m.group(1) if m else None

def _normalize_label(s):
    # normalize labels for robust matching
    s = '' if s is None else str(s)
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

load_dotenv()
API_KEY = os.getenv("API_KEY_PERSONAL")
assert API_KEY, "Set API_KEY_SCILIFELAB in your .env"
# measure client init to debug slow starts
t0_client = time.time()
model = dna_client.create(api_key=API_KEY)
print(f"api client ready in {time.time() - t0_client:.1f}s")

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
df["gene_tag_core"] = df["gene_tag"].map(ensg_core)

# add anchors so every target gene receives a 1 mb window
try:
    bed = pd.read_csv(
        TSS_BED,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "name"],
        dtype={"chr": str, "start": int, "end": int, "name": str},
    )
except FileNotFoundError as e:
    raise FileNotFoundError(f"missing tss bed: {TSS_BED}") from e

# ensure ucsc-style chromosome names
bed["chr"] = bed["chr"].astype(str).map(lambda c: c if c.startswith("chr") else f"chr{c}")
bed["tss_pos"] = ((bed["start"].astype(int) + bed["end"].astype(int)) // 2).astype(int)
bed["gene_tag"] = bed["name"].astype(str)
bed["ensg_core"] = bed["name"].map(ensg_core)

have_genes = set(df["gene_tag_core"].dropna().astype(str).unique())
all_genes = set(bed["ensg_core"].dropna().astype(str).unique())
missing_core = sorted(all_genes - have_genes)

anchors = pd.DataFrame(columns=["variant_id", "gene_tag", "CHROM", "POS", "REF", "ALT", "is_anchor"])
if missing_core:
    bed_missing = bed[bed["ensg_core"].isin(missing_core)].copy()
    # default synthetic alleles
    bed_missing["REF"] = "A"
    bed_missing["ALT"] = "C"

    # clash guard: if a real variant exists at same chrom:pos for the same gene, flip to A>G
    clash_keys = set(zip(
        df["CHROM"].astype(str),
        df["POS"].astype(int),
        df["gene_tag_core"].astype(str),
    ))

    def _maybe_flip(row):
        # guard collisions per ensg core
        tup = (str(row["chr"]), int(row["tss_pos"]), str(row["ensg_core"]))
        return "G" if tup in clash_keys else "C"

    bed_missing["ALT"] = bed_missing.apply(_maybe_flip, axis=1)

    anchors = pd.DataFrame(
        {
            "variant_id": (
                bed_missing["chr"].astype(str)
                + ":"
                + bed_missing["tss_pos"].astype(int).astype(str)
                + ":A>"
                + bed_missing["ALT"].astype(str)
            ),
            # write plain ensg to keep schema aligned with df
            "gene_tag": bed_missing["ensg_core"].astype(str),
            "CHROM": bed_missing["chr"].astype(str),
            "POS": bed_missing["tss_pos"].astype(int),
            "REF": "A",
            "ALT": bed_missing["ALT"].astype(str),
            "is_anchor": True,
        }
    )

# attach is_anchor flag and append anchors
if "is_anchor" not in df.columns:
    df["is_anchor"] = False
else:
    df["is_anchor"] = df["is_anchor"].fillna(False).astype(bool)

if len(anchors):
    df = (
        pd.concat([df, anchors.loc[:, df.columns]], ignore_index=True)
        .drop_duplicates(subset=["variant_id", "gene_tag"], keep="first")
    )

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

def parse_existing_chunks(dirpath):
    done = set()
    for p in glob.glob(os.path.join(dirpath, "chunk_*.tsv.gz")):
        b = os.path.basename(p)
        s, e = b.replace("chunk_", "").replace(".tsv.gz", "").split("_")
        done.add((int(s), int(e)))
    return done

done = parse_existing_chunks(CHUNK_DIR)
num_batches = math.ceil(n / BATCH)

# Reverse: process batches from the end of the TSV toward the start
batch_ids = list(range(num_batches - 1, -1, -1))

for b in batch_ids:
    start = b * BATCH
    end   = min(start + BATCH, n)
    out_chunk = chunk_path(start, end)
    lock_path = out_chunk + ".lock"
    tmp_chunk = out_chunk + ".tmp"

    # Skip if already produced here or already present
    if (start, end) in done or os.path.exists(out_chunk):
        print(f"Skipping existing chunk {start}-{end}")
        continue

    # Acquire a local lock to avoid duplicate work if re-run
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"Locked by another process, skip {start}-{end}")
        continue

    try:
        print(f"start batch {start}-{end} size {end - start}")
        meta = df.iloc[start:end].copy()
        intervals_cm = make_intervals(SEQ_LEN_CM, meta["CHROM"].to_numpy(), meta["POS"].to_numpy())
        variants_chunk = make_variants(meta["CHROM"], meta["POS"], meta["REF"], meta["ALT"], meta["variant_id"])
        cm_interval_to_varid = { _interval_to_str(iv): vid for iv, vid in zip(intervals_cm, meta["variant_id"]) }

        gm_frames = []
        for L in SEQ_LENS_GM:
            iv = make_intervals(L, meta["CHROM"].to_numpy(), meta["POS"].to_numpy())
            gm_interval_to_varid = { _interval_to_str(ivv): vid for ivv, vid in zip(iv, meta["variant_id"]) }
            t0 = time.time()
            print(f"gm score L={L} n={len(meta)} start")
            scores = model.score_variants(
                intervals=iv,
                variants=variants_chunk,
                variant_scorers=[variant_scorers.GeneMaskLFCScorer(requested_output=RNA)],
                organism=ORG,
                progress_bar=False,
            )
            print(f"gm score L={L} done in {time.time() - t0:.1f}s")
            inject_into_anndata(scores, meta)
            tidy = variant_scorers.tidy_scores(scores, match_gene_strand=False)
            tidy = normalize_tidy(tidy)
            _ensure_variant_ids(tidy, gm_interval_to_varid)
            anchor_map = dict(zip(meta["variant_id"], meta["is_anchor"]))
            tidy["is_anchor"] = tidy["variant_id"].map(anchor_map).fillna(False)
            tidy["seq_len"] = L
            gm_frames.append(tidy)

        gm_all = pd.concat(gm_frames, ignore_index=True)

        # center-mask scorers at 1 Mb
        t0 = time.time()
        print(f"cm score 1mb n={len(meta)} start")
        scores_cm_center = model.score_variants(
            intervals=intervals_cm,
            variants=variants_chunk,
            variant_scorers=SCORERS_CENTER,
            organism=ORG,
            progress_bar=False,
        )
        print(f"cm score 1mb done in {time.time() - t0:.1f}s")

        cm = scores_to_df(scores_cm_center, meta)
        _ensure_variant_ids(cm, cm_interval_to_varid)
        anchor_map = dict(zip(meta["variant_id"], meta["is_anchor"]))
        cm["is_anchor"] = cm["variant_id"].map(anchor_map).fillna(False)
        cm["seq_len"] = SEQ_LEN_CM

        # metadata harmonization identical to your forward loop
        gm_all["gene_id"] = gm_all.get("gene_id", pd.Series(pd.NA, index=gm_all.index)).map(strip_ensg)
        vm_gid = (gm_all.dropna(subset=["variant_id", "gene_id"]) \
                  .groupby("variant_id")["gene_id"] \
                  .agg(lambda s: ";".join(sorted(set(s)))))
        vm_gname = (gm_all.dropna(subset=["variant_id", "gene_name"]) \
                    .groupby("variant_id")["gene_name"] \
                    .agg(lambda s: ";".join(sorted(set(map(str, s))))))
        si_gid = (gm_all.dropna(subset=["scored_interval_str", "gene_id"]) \
                  .groupby("scored_interval_str")["gene_id"] \
                  .agg(lambda s: ";".join(sorted(set(s)))))
        si_gname = (gm_all.dropna(subset=["scored_interval_str", "gene_name"]) \
                    .groupby("scored_interval_str")["gene_name"] \
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

        out_df = pd.concat([gm_all, cm], ignore_index=True)

        # optional tissue filter (unchanged)
        mask_muscle = pd.Series(False, index=out_df.index)
        if "gtex_tissue" in out_df.columns:
            norm = out_df["gtex_tissue"].astype(str).map(_normalize_label)
            mask_muscle = norm.str.contains("muscle", na=False) & norm.str.contains("skeletal", na=False)
        mask_uberon = pd.Series(False, index=out_df.index)
        if "ontology_curie" in out_df.columns:
            mask_uberon = out_df["ontology_curie"].astype(str).str.contains("UBERON:0001134", case=False, na=False)
        if mask_muscle.any() or mask_uberon.any():
            out_df = out_df[mask_muscle | mask_uberon].copy()

        # atomic write
        t0w = time.time()
        out_df.to_csv(tmp_chunk, sep="\t", index=False, compression="gzip")
        os.replace(tmp_chunk, out_chunk)
        print(f"wrote {os.path.basename(out_chunk)} in {time.time() - t0w:.1f}s, progress {end}/{n}")
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass

if os.getenv("STITCH", "0") == "1":
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