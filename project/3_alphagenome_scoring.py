import os
import pandas as pd
from dotenv import load_dotenv

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from alphagenome.models import dna_client as _dc


VAR_TSV = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/testis_variants_all.tsv.gz"
OUT_TSV = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/testis_alphagenome_scores_all_aggs.tsv.gz"

SEQ_LEN = 131_072
BATCH   = 512
RNA     = dna_client.OutputType.RNA_SEQ
Agg     = variant_scorers.AggregationType
ONTOLOGY_TERMS = None

load_dotenv()
API_KEY = os.getenv("API_KEY_PERSONAL")
assert API_KEY, "Set API_KEY_PERSONAL in your .env"
model = dna_client.create(api_key=API_KEY, model_version=None, timeout=None, address=None)

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

half = SEQ_LEN // 2
def make_interval(chrom, pos):
    start = max(1, int(pos) - half)
    end   = int(pos) + half
    return genome.Interval(chromosome=str(chrom), start=start, end=end)

intervals = [make_interval(r.CHROM, r.POS) for r in df.itertuples(index=False)]
variants = [
    genome.Variant(
        chromosome=str(r.CHROM),
        position=int(r.POS),
        reference_bases=str(r.REF),
        alternate_bases=str(r.ALT),
        name=str(r.variant_id),     # use variant_id as the Variant name
    )
    for r in df.itertuples(index=False)
]

scorers = [
    variant_scorers.GeneMaskLFCScorer(requested_output=RNA),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.DIFF_MEAN),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=2_001,   aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=10_001,  aggregation_type=Agg.L2_DIFF_LOG1P),
    variant_scorers.CenterMaskScorer(requested_output=RNA, width=100_001, aggregation_type=Agg.L2_DIFF_LOG1P),
]

ORGANISM = _dc.Organism.HOMO_SAPIENS
all_chunks = []

for i in range(0, len(variants), BATCH):
    v_chunk  = variants[i:i+BATCH]
    iv_chunk = intervals[i:i+BATCH]

    scores = model.score_variants(
        intervals=iv_chunk,
        variants=v_chunk,
        variant_scorers=scorers,
        organism=ORGANISM,
        progress_bar=False,
    )

    # Tidy to a long DataFrame (one row per variant × scorer × track)
    tidy = variant_scorers.tidy_scores(scores)

    # Normalize variant_id so it's a plain string key
    if "variant_id" in tidy.columns:
        tidy["variant_id"] = tidy["variant_id"].map(lambda v: v.name if hasattr(v, "name") else str(v))
    elif "variant" in tidy.columns:
        tidy["variant_id"] = tidy["variant"].map(lambda v: v.name if hasattr(v, "name") else str(v))

    # Attach original metadata (variant_id, CHROM, POS, REF, ALT, gene_tag)
    meta = df.iloc[i:i+len(v_chunk)][["variant_id", "CHROM", "POS", "REF", "ALT", "gene_tag"]].copy()
    meta["variant_id"] = meta["variant_id"].astype(str)
    tidy = tidy.merge(meta, on="variant_id", how="left")

    all_chunks.append(tidy)
    print(f"Scored {i+len(v_chunk)}/{len(variants)}")

out = pd.concat(all_chunks, ignore_index=True)

rename_map = {
    "GeneMaskLFCScorer": "gene_exonmask_delta_log2",
    "CenterMaskScorer(width=2001).DIFF_MEAN": "center2.001kb_diff_mean",
    "CenterMaskScorer(width=10001).DIFF_MEAN": "center10.001kb_diff_mean",
    "CenterMaskScorer(width=100001).DIFF_MEAN": "center100.001kb_diff_mean",
    "CenterMaskScorer(width=2001).L2_DIFF_LOG1P": "center2.001kb_l2_log1p",
    "CenterMaskScorer(width=10001).L2_DIFF_LOG1P": "center10.001kb_l2_log1p",
    "CenterMaskScorer(width=100001).L2_DIFF_LOG1P": "center100.001kb_l2_log1p",
}
if "scorer" in out.columns:
    out["scorer_friendly"] = out["scorer"].map(lambda s: rename_map.get(s, s))

out.to_csv(OUT_TSV, sep="\t", index=False, compression="gzip")
print(f"Wrote {OUT_TSV} with {out.shape[0]:,} rows")