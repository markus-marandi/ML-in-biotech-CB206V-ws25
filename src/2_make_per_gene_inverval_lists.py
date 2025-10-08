import os, glob, pandas as pd

IN_DIR  = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/primary_files_hail_muscle_skeletal"
OUT_TSV = "/Users/markus/university/ML-in-biotech-CB206V-ws25/data/intermediate/muscle_skeletal_variants_all.tsv.gz"

REQUIRED = {"CHROM","POS","REF","ALT"}

def load_one(p):
    gene = os.path.basename(p).replace(".gnomad_v4.1.tsv.bgz","")
    # bgz is gzip-compatible
    try:
        df = pd.read_csv(p, sep="\t", compression="gzip", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(p, sep="\t", compression="infer", engine="python", low_memory=False)

    ren = {}
    up = {c.upper(): c for c in df.columns}
    for want in ["CHROM","POS","REF","ALT","AF","AC","AN"]:
        if want in up:
            ren[up[want]] = want
        elif want.lower() in df.columns:
            ren[want.lower()] = want
    df = df.rename(columns=ren)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(p)} missing columns: {missing}. Columns seen: {list(df.columns)}")

    # types
    df["CHROM"] = df["CHROM"].astype(str)
    df["POS"]   = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
    for col in ["REF","ALT"]:
        df[col] = df[col].astype(str)

    df["gene_tag"] = gene
    df["variant_id"] = df["CHROM"] + ":" + df["POS"].astype(str) + ":" + df["REF"] + ">" + df["ALT"]
    return df

files = sorted(glob.glob(os.path.join(IN_DIR, "*.gnomad_v4.1.tsv.bgz")))
assert files, "No per-gene TSVs found"

all_df = pd.concat([load_one(p) for p in files], ignore_index=True)
all_df = all_df.dropna(subset=["POS"]).drop_duplicates(subset=["variant_id","gene_tag"])
all_df.to_csv(OUT_TSV, sep="\t", index=False, compression="gzip")
print(all_df.shape, "→", OUT_TSV)
print(all_df.head(3).to_string(index=False))