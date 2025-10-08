import os, json
import pandas as pd
import hail as hl
from collections import defaultdict

TARGET_BED = "/mnt/sdb/markus_files/gene_exp/11_prep_variants/testis_target_regions_merged.bed"
OUTDIR     = "/mnt/sdb/markus_files/gene_exp/11_prep_variants/primary_files_hail_testis"
os.makedirs(OUTDIR, exist_ok=True)

GNOMAD_HT = "gs://gcp-public-data--gnomad/release/4.1/ht/genomes/gnomad.genomes.v4.1.sites.ht/"

with open("/mnt/sdb/markus_files/gene_exp/11_prep_variants/spark_conf.json") as f:
    spark_conf = json.load(f)
hail_home = "/mnt/sdb/markus_files/hail"
for k, v in spark_conf.items():
    if isinstance(v, str):
        spark_conf[k] = v.replace("{hail_home}", hail_home)

bed = pd.read_csv(TARGET_BED, sep="\t", header=None, names=["chr","start","end","name"], dtype={"chr":str})
assert len(bed) > 0
assert (bed["end"] > bed["start"]).all()
assert bed["chr"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$").all()

gene_intervals = defaultdict(list)
for r in bed.itertuples(index=False):
    gene_intervals[r.name].append((r.chr, int(r.start), int(r.end)))
genes = sorted(gene_intervals.keys())
print(f"Loaded {len(bed)} intervals across {len(genes)} genes")

hl.init(app_name="gnomad_v41_by_target_regions",
        spark_conf=spark_conf,
        tmp_dir="/mnt/sdb/tmp",
        default_reference="GRCh38")

ht = hl.read_table(GNOMAD_HT)

def to_hl_interval(chrom, start0, end1):
    # BED 0-based, half-open -> Hail 1-based inclusive
    return hl.parse_locus_interval(f"{chrom}:{start0+1}-{end1}", reference_genome="GRCh38")

for gene in genes:
    ivals = [to_hl_interval(c, s, e) for (c, s, e) in gene_intervals[gene]]

    sub = hl.filter_intervals(ht, ivals)
    sub = sub.key_by()  # unkey to avoid key-based surprises

    sub = sub.annotate(
        CHROM = sub.locus.contig,
        POS   = sub.locus.position,
        REF   = sub.alleles[0],
        ALT   = sub.alleles[1],
        AF = hl.or_missing(hl.len(sub.freq) > 0, sub.freq[0].AF),
        AC = hl.or_missing(hl.len(sub.freq) > 0, sub.freq[0].AC),
        AN = hl.or_missing(hl.len(sub.freq) > 0, sub.freq[0].AN),
    )
    sub = sub.filter(hl.len(sub.alleles) == 2)  # biallelic only

    n = sub.count()
    if n == 0:
        print(f" {gene}: no variants; skipping")
        continue

    out_path = os.path.join(OUTDIR, f"{gene}.gnomad_v4.1.tsv.bgz")
    sub.select("CHROM","POS","REF","ALT","AF","AC","AN").export(out_path)
    print(f"{gene}: {n} variants → {out_path}")

hl.stop()
print("Done.")