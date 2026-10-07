"""
Compatibility wrapper for 02_analisis_exploratorio.py.
Forwards to brecha_educativa.
"""

import pandas as pd
from brecha_educativa.config import CONSOLIDATED_FILE, FIGURES_DIR
from brecha_educativa.processor import process_raw_data
from brecha_educativa.visualizer import plot_fee_distribution
from brecha_educativa.report import generate_educational_report

def main():
    if not CONSOLIDATED_FILE.is_file():
        process_raw_data()
    df = pd.read_csv(CONSOLIDATED_FILE)
    plot_fee_distribution(df, output_path=FIGURES_DIR / "distribucion_financiamiento.png")
    generate_educational_report(df)

if __name__ == "__main__":
    main()