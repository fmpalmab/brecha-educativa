"""
Procesamiento, limpieza y consolidación de datos educativos (MINEDUC y SIMCE).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
import pandas as pd

from brecha_educativa.config import (
    RAW_DIR,
    PROCESSED_DIR,
    CONSOLIDATED_FILE,
    BBOX,
    MIN_COLEGIOS_POR_COMUNA,
)


def normalizar_texto(texto: Any) -> str:
    """Estandariza cadenas de texto: mayúsculas, sin tildes ni caracteres especiales."""
    if pd.isna(texto):
        return "DESCONOCIDO"
    texto_str = str(texto).upper().strip()
    texto_norm = unicodedata.normalize("NFD", texto_str)
    limpio = "".join(c for c in texto_norm if unicodedata.category(c) != "Mn")
    return limpio.strip()


def clean_coord(val: Any) -> Optional[float]:
    """Convierte coordenadas con comas decimales o formato string a float seguro."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).replace(",", ".").strip()
    try:
        return float(val_str)
    except ValueError:
        return None


def clasificar_pago_consolidado(row: pd.Series) -> str:
    """Clasifica en 'Gratuito' o 'Pagado' según tipo de pago y dependencia institucional."""
    pago = str(row.get("PAGO_MENSUAL", "")).upper()
    dep = str(row.get("categoria_dependencia", "")).upper()

    if pago == "GRATUITO":
        return "Gratuito"
    if pago == "SIN INFORMACION" and any(x in dep for x in ["MUNICIPAL", "SLEP", "ADMIN"]):
        return "Gratuito"
    return "Pagado"


def generate_synthetic_educational_dataset(
    n_schools: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético representativo de colegios en Santiago (RM)
    para pruebas unitarias y entornos sin acceso a los CSVs oficiales completos.
    """
    rng = np.random.default_rng(seed)
    comunas = [
        "SANTIAGO", "PROVIDENCIA", "LAS CONDES", "VITACURA", "LA FLORIDA",
        "MAIPU", "PUENTE ALTO", "PUDAHUEL", "QUILICURA", "NUNOA", "PENALOLEN",
    ]

    rows = []
    for i in range(1, n_schools + 1):
        comuna = rng.choice(comunas)
        is_paid = bool(rng.random() < (0.75 if comuna in ("LAS CONDES", "VITACURA", "PROVIDENCIA") else 0.20))
        pago = "MAS DE $100.000" if is_paid else "GRATUITO"
        cod_depe = 4 if is_paid else rng.choice([1, 2, 3, 6])
        dep_cat = "PARTICULAR_PAGADO" if is_paid else "PARTICULAR_SUBV"

        # Coordenadas dentro de Santiago BBox
        lon = rng.uniform(BBOX[0] + 0.05, BBOX[2] - 0.05)
        lat = rng.uniform(BBOX[1] + 0.05, BBOX[3] - 0.05)

        docentes = rng.integers(15, 90)
        cursos = rng.integers(8, 36)
        mat_total = int(cursos * rng.uniform(22.0, 38.0))

        # SIMCE base: más alto en colegios pagados
        simce_base = 295.0 if is_paid else 245.0
        simce_4b = float(np.clip(simce_base + rng.normal(0, 15.0), 180.0, 340.0))
        simce_2m = float(np.clip(simce_base - 5.0 + rng.normal(0, 16.0), 175.0, 335.0))

        rows.append({
            "RBD": 1000 + i,
            "NOM_RBD": f"COLEGIO_{i}",
            "COD_COM_RBD": 13100 + (hash(comuna) % 30),
            "NOM_COM_RBD": comuna,
            "COD_DEPE": cod_depe,
            "categoria_dependencia": dep_cat,
            "LATITUD": lat,
            "LONGITUD": lon,
            "PAGO_MENSUAL": pago,
            "TIPO_PAGO": "Pagado" if is_paid else "Gratuito",
            "orden_precio": 5 if is_paid else 0,
            "MAT_TOTAL": mat_total,
            "CUR_SIM_TOT": cursos,
            "DC_TOT": docentes,
            "ratio_alumno_docente": round(mat_total / docentes, 2),
            "ratio_alumno_curso": round(mat_total / cursos, 2),
            "SIMCE_4B_AVG": round(simce_4b, 1),
            "SIMCE_2M_AVG": round(simce_2m, 1),
        })

    df = pd.DataFrame(rows)
    return df


def process_raw_data(
    raw_dir: Union[str, Path] = RAW_DIR,
    output_path: Union[str, Path] = CONSOLIDATED_FILE,
    bbox: List[float] = BBOX,
    min_schools_per_commune: int = MIN_COLEGIOS_POR_COMUNA,
) -> pd.DataFrame:
    """Procesa los archivos oficiales de MINEDUC y SIMCE en un CSV consolidado."""
    p_raw = Path(raw_dir)
    ee_file = p_raw / "EE_2024.csv"
    mat_file = p_raw / "Matricula_2024.csv"
    doc_file = p_raw / "Docente_2024.csv"

    if not (ee_file.is_file() and mat_file.is_file() and doc_file.is_file()):
        print("[Aviso] No se encontraron todos los archivos RAW; generando dataset estructurado sintético...")
        df_synthetic = generate_synthetic_educational_dataset(n_schools=200)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_synthetic.to_csv(out_p, index=False)
        return df_synthetic

    print("Cargando archivos oficiales RAW...")
    df_ee = pd.read_csv(ee_file, sep=";", encoding="utf-8", low_memory=False)
    df_mat = pd.read_csv(mat_file, sep=";", encoding="utf-8", low_memory=False)
    df_doc = pd.read_csv(doc_file, sep=";", encoding="utf-8", low_memory=False)

    for df in [df_ee, df_mat, df_doc]:
        df["RBD"] = pd.to_numeric(df["RBD"], errors="coerce")
        df.dropna(subset=["RBD"], inplace=True)
        df["RBD"] = df["RBD"].astype(int)

    # Filtrar solo RM (Región 13)
    df_ee_rm = df_ee[df_ee["COD_REG_RBD"] == 13].copy()
    df_ee_rm["NOM_COM_RBD"] = df_ee_rm["NOM_COM_RBD"].apply(normalizar_texto)
    # Selección de columnas base para evitar duplicación con df_mat
    cols_ee = [
        "RBD", "NOM_RBD", "COD_COM_RBD", "NOM_COM_RBD", "COD_DEPE",
        "LATITUD", "LONGITUD", "RURAL_RBD", "PAGO_MENSUAL",
    ]
    df_ee_rm = df_ee_rm[[c for c in cols_ee if c in df_ee_rm.columns]].copy()
    df_ee_rm["LATITUD"] = df_ee_rm["LATITUD"].apply(clean_coord)
    df_ee_rm["LONGITUD"] = df_ee_rm["LONGITUD"].apply(clean_coord)

    # Agregación Matrícula y Cursos
    df_mat_g = df_mat.groupby("RBD")["MAT_TOTAL"].sum().reset_index()
    if "CUR_SIM_TOT" in df_mat.columns:
        df_cursos = df_mat.groupby("RBD")["CUR_SIM_TOT"].sum().reset_index()
        df_mat_g = df_mat_g.merge(df_cursos, on="RBD", how="left")

    df_doc_g = df_doc.groupby("RBD")["DC_TOT"].sum().reset_index()

    df_master = df_ee_rm.merge(df_mat_g, on="RBD", how="left")
    df_master = df_master.merge(df_doc_g, on="RBD", how="left")
    df_master = df_master.dropna(subset=["LATITUD", "LONGITUD", "DC_TOT", "MAT_TOTAL"])
    df_master = df_master[(df_master["DC_TOT"] > 0) & (df_master["MAT_TOTAL"] > 0)]

    df_master["ratio_alumno_docente"] = df_master["MAT_TOTAL"] / df_master["DC_TOT"]
    if "CUR_SIM_TOT" in df_master.columns:
        df_master["ratio_alumno_curso"] = df_master.apply(
            lambda x: x["MAT_TOTAL"] / x["CUR_SIM_TOT"] if x["CUR_SIM_TOT"] > 0 else np.nan, axis=1
        )

    dep_map = {
        1: "MUNICIPAL_CORP", 2: "MUNICIPAL_DAEM", 3: "PARTICULAR_SUBV",
        4: "PARTICULAR_PAGADO", 5: "ADMIN_DELEGADA", 6: "SLEP",
    }
    df_master["categoria_dependencia"] = df_master["COD_DEPE"].map(dep_map).fillna("OTRO")
    df_master["TIPO_PAGO"] = df_master.apply(clasificar_pago_consolidado, axis=1)

    precio_map = {
        "GRATUITO": 0, "$1.000 A $10.000": 1, "$10.001 A $25.000": 2,
        "$25.001 A $50.000": 3, "$50.001 A $100.000": 4, "MAS DE $100.000": 5, "SIN INFORMACION": -1,
    }
    df_master["PAGO_MENSUAL_NORM"] = df_master["PAGO_MENSUAL"].apply(lambda x: str(x).upper().strip())
    df_master["orden_precio"] = df_master["PAGO_MENSUAL_NORM"].map(precio_map).fillna(-1)

    # Integración SIMCE si existe
    s4_file = p_raw / "simce4b2024_rbd_preliminar.csv"
    s2_file = p_raw / "simce2m2024_rbd_preliminar.csv"
    if s4_file.is_file() and s2_file.is_file():
        try:
            df_s4 = pd.read_csv(s4_file, sep=";", encoding="latin-1")
            df_s2 = pd.read_csv(s2_file, sep=";", encoding="latin-1")
            cols_s4 = {"rbd": "RBD", "prom_lect4b_rbd": "SIMCE_4B_LECT", "prom_mate4b_rbd": "SIMCE_4B_MATE"}
            cols_s2 = {"rbd": "RBD", "prom_lect2m_rbd": "SIMCE_2M_LECT", "prom_mate2m_rbd": "SIMCE_2M_MATE"}
            df_s4_sel = df_s4[[c for c in cols_s4 if c in df_s4.columns]].rename(columns=cols_s4)
            df_s2_sel = df_s2[[c for c in cols_s2 if c in df_s2.columns]].rename(columns=cols_s2)
            df_s4_sel["RBD"] = pd.to_numeric(df_s4_sel["RBD"], errors="coerce")
            df_s2_sel["RBD"] = pd.to_numeric(df_s2_sel["RBD"], errors="coerce")
            df_s4_sel["SIMCE_4B_AVG"] = df_s4_sel[["SIMCE_4B_LECT", "SIMCE_4B_MATE"]].mean(axis=1)
            df_s2_sel["SIMCE_2M_AVG"] = df_s2_sel[["SIMCE_2M_LECT", "SIMCE_2M_MATE"]].mean(axis=1)
            df_master = df_master.merge(df_s4_sel[["RBD", "SIMCE_4B_AVG"]], on="RBD", how="left")
            df_master = df_master.merge(df_s2_sel[["RBD", "SIMCE_2M_AVG"]], on="RBD", how="left")
        except Exception as e:
            print(f"[Aviso SIMCE]: {e}")

    # Filtro BBox y Comunas válidas
    mask_bbox = (
        (df_master["LONGITUD"] >= bbox[0]) & (df_master["LONGITUD"] <= bbox[2]) &
        (df_master["LATITUD"] >= bbox[1]) & (df_master["LATITUD"] <= bbox[3])
    )
    df_final = df_master[mask_bbox].copy()

    conteo_comunal = df_final["NOM_COM_RBD"].value_counts()
    comunas_validas = conteo_comunal[conteo_comunal > min_schools_per_commune].index
    df_final = df_final[df_final["NOM_COM_RBD"].isin(comunas_validas)].copy()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_p, index=False)
    print(f"Base consolidada guardada exitosamente en: {out_p} ({len(df_final)} colegios)")

    return df_final
