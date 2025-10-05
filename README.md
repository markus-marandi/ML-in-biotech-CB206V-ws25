# Genomic Variant Analysis for Machine Learning in Biotechnology

This repository provides a modular pipeline for genomic variant analysis and regulatory effect prediction. It integrates DeepMind’s AlphaGenome API to estimate RNA-seq and promoter-level impacts of DNA sequence variants, supporting analyses of cis-regulatory regions and gene expression changes.

The workflow includes variant extraction (via Hail), per-gene aggregation, and AlphaGenome-based batch scoring. Input variant files must contain the following columns: CHROM (chromosome), POS (1-based position), REF (reference allele), and ALT (alternate allele).

Dependencies: alphagenome>=0.2.0, pandas, pybedtools, pysam, requests, python-dotenv, hail>=0.2.133.

References:
1. DeepMind (2025). AlphaGenome Python SDK 0.2.0 – Model for regulatory genome predictions.
2. Avsec Ž. et al. (2025). AlphaGenome: Advancing regulatory variant effect prediction with a unified DNA sequence model (preprint).
3. Library documentation for Pandas 2.3.3, PyBedTools 0.12.0, pysam 0.23.3, and Requests 2.32.5.
4. AlphaGenome Tutorials (2025). Quick Start and variant scoring examples.
5. BioMCP Tutorial (2025). Predicting Variant Effects with AlphaGenome – illustrated regulatory variant analysis.