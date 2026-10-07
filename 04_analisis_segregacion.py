"""
Compatibility wrapper for 04_analisis_segregacion.py.
Forwards to brecha_educativa.analysis.
"""

import pandas as pd
from brecha_educativa.config import CONSOLIDATED_FILE, METRICS_SUMMARY_FILE
from brecha_educativa.processor import process_raw_data
from brecha_educativa.analysis import compute_duncan_segregation_index, compute_commune_metrics_summary

def main():
    if not CONSOLIDATED_FILE.is_file():
        process_raw_data()
    df = pd.read_csv(CONSOLIDATED_FILE)
    duncan = compute_duncan_segregation_index(df)
    summary = compute_commune_metrics_summary(df)
    summary.to_csv(METRICS_SUMMARY_FILE, index=False)
    print(f"Índice de Segregación Duncan (D): {duncan:.3f}")

if __name__ == "__main__":
    main()