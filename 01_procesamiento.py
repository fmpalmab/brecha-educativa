import pandas as pd
import numpy as np
import unicodedata
import os

# --- CONFIGURACIÓN ---
# Bounding Box Urbano [min_lon, min_lat, max_lon, max_lat]
BBOX = [-70.872116, -33.642527, -70.469742, -33.327552]
# Mínimo de colegios para considerar la comuna válida estadísticamente
MIN_COLEGIOS_POR_COMUNA = 3

def normalizar_texto(texto):
    """Estandariza strings: Mayúsculas, sin tildes, sin espacios extra."""
    if pd.isna(texto):
        return "DESCONOCIDO"
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def clean_coord(val):
    """Limpia coordenadas que pueden venir con coma decimal."""
    if isinstance(val, str):
        val = val.replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

def clasificar_pago_consolidado(row):
    """Regla de Negocio: Gratuito vs Pagado (para visualización Rojo/Verde)"""
    pago = str(row['PAGO_MENSUAL']).upper()
    dep = str(row['categoria_dependencia']).upper()
    
    if pago == 'GRATUITO': 
        return 'Gratuito'
    # Asumir gratuidad si es público y no hay info
    if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN']): 
        return 'Gratuito'
    
    return 'Pagado'

def main():
    print(">>> INICIANDO PROCESAMIENTO CONSOLIDADO (Bloque 1 - Corregido) <<<")
    os.makedirs('data/processed', exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CARGA Y LIMPIEZA INICIAL
    # -------------------------------------------------------------------------
    print("1. Cargando archivos RAW...")
    try:
        df_ee = pd.read_csv('data/raw/EE_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_mat = pd.read_csv('data/raw/Matricula_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_doc = pd.read_csv('data/raw/Docente_2024.csv', sep=';', encoding='utf-8', low_memory=False)
    except FileNotFoundError as e:
        print(f"❌ Error: Falta archivo raw -> {e}")
        return

    # Normalizar RBD
    for df in [df_ee, df_mat, df_doc]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')
        df.dropna(subset=['RBD'], inplace=True)
        df['RBD'] = df['RBD'].astype(int)

    # Filtro: Eliminar solo párvulos
    print("   -> Filtrando jardines infantiles puros...")
    cols_ens = [f'ENS_{i:02d}' for i in range(1, 12)]
    ens_data = df_ee[cols_ens].replace(0, np.nan)
    
    def es_solo_parvulo(row):
        niveles = set(row.dropna().unique())
        return len(niveles) == 1 and 10 in niveles # 10 es Parvularia

    mask_parvulos = ens_data.apply(es_solo_parvulo, axis=1)
    df_ee = df_ee[~mask_parvulos].copy()

    # Filtro: Solo RM (Región 13)
    df_ee_rm = df_ee[df_ee['COD_REG_RBD'] == 13].copy()
    
    # Normalizar Comunas
    df_ee_rm['NOM_COM_RBD'] = df_ee_rm['NOM_COM_RBD'].apply(normalizar_texto)

    # Limpieza de Coordenadas
    df_ee_rm['LATITUD'] = df_ee_rm['LATITUD'].apply(clean_coord)
    df_ee_rm['LONGITUD'] = df_ee_rm['LONGITUD'].apply(clean_coord)

    # Selección de columnas base
    cols_ee = ['RBD', 'NOM_RBD', 'COD_COM_RBD', 'NOM_COM_RBD', 'COD_DEPE', 
               'LATITUD', 'LONGITUD', 'RURAL_RBD', 'PAGO_MENSUAL']
    df_ee_rm = df_ee_rm[[c for c in cols_ee if c in df_ee_rm.columns]]

    # -------------------------------------------------------------------------
    # 2. AGREGACIÓN DE MATRÍCULA Y DOCENTES
    # -------------------------------------------------------------------------
    print("2. Calculando métricas de capacidad...")
    
    # Matrícula y Cursos
    df_mat_g = df_mat.groupby('RBD')['MAT_TOTAL'].sum().reset_index()
    if 'CUR_SIM_TOT' in df_mat.columns:
        df_cursos = df_mat.groupby('RBD')['CUR_SIM_TOT'].sum().reset_index()
        df_mat_g = df_mat_g.merge(df_cursos, on='RBD', how='left')
    
    # Docentes
    df_doc_g = df_doc.groupby('RBD')['DC_TOT'].sum().reset_index()
    
    # Merge
    df_master = df_ee_rm.merge(df_mat_g, on='RBD', how='left')
    df_master = df_master.merge(df_doc_g, on='RBD', how='left')
    
    # Limpieza nulos críticos
    df_master = df_master.dropna(subset=['LATITUD', 'LONGITUD', 'DC_TOT', 'MAT_TOTAL'])
    df_master = df_master[(df_master['DC_TOT'] > 0) & (df_master['MAT_TOTAL'] > 0)]

    # Cálculo de Ratios
    df_master['ratio_alumno_docente'] = df_master['MAT_TOTAL'] / df_master['DC_TOT']
    if 'CUR_SIM_TOT' in df_master.columns:
        df_master['ratio_alumno_curso'] = df_master.apply(
            lambda x: x['MAT_TOTAL'] / x['CUR_SIM_TOT'] if x['CUR_SIM_TOT'] > 0 else np.nan, axis=1
        )

    # -------------------------------------------------------------------------
    # 3. CLASIFICACIONES DE NEGOCIO
    # -------------------------------------------------------------------------
    print("3. Aplicando reglas de negocio...")
    dep_map = {1:'MUNICIPAL_CORP', 2:'MUNICIPAL_DAEM', 3:'PARTICULAR_SUBV', 
               4:'PARTICULAR_PAGADO', 5:'ADMIN_DELEGADA', 6:'SLEP'}
    df_master['categoria_dependencia'] = df_master['COD_DEPE'].map(dep_map).fillna('OTRO')
    df_master['TIPO_PAGO'] = df_master.apply(clasificar_pago_consolidado, axis=1)

    precio_map = {'GRATUITO': 0, '$1.000 A $10.000': 1, '$10.001 A $25.000': 2,
                  '$25.001 A $50.000': 3, '$50.001 A $100.000': 4, 
                  'MAS DE $100.000': 5, 'SIN INFORMACION': -1}
    df_master['PAGO_MENSUAL_NORM'] = df_master['PAGO_MENSUAL'].apply(lambda x: str(x).upper().strip())
    df_master['orden_precio'] = df_master['PAGO_MENSUAL_NORM'].map(precio_map).fillna(-1)

    # -------------------------------------------------------------------------
    # 4. INTEGRACIÓN SIMCE
    # -------------------------------------------------------------------------
    print("4. Integrando SIMCE...")
    try:
        df_s4 = pd.read_csv('data/raw/simce4b2024_rbd_preliminar.csv', sep=';', encoding='latin-1')
        df_s2 = pd.read_csv('data/raw/simce2m2024_rbd_preliminar.csv', sep=';', encoding='latin-1')
        
        cols_s4 = {'rbd':'RBD', 'prom_lect4b_rbd':'SIMCE_4B_LECT', 'prom_mate4b_rbd':'SIMCE_4B_MATE'}
        cols_s2 = {'rbd':'RBD', 'prom_lect2m_rbd':'SIMCE_2M_LECT', 'prom_mate2m_rbd':'SIMCE_2M_MATE'}
        
        df_s4_sel = df_s4[cols_s4.keys()].rename(columns=cols_s4)
        df_s2_sel = df_s2[cols_s2.keys()].rename(columns=cols_s2)
        
        for df_s in [df_s4_sel, df_s2_sel]:
            df_s['RBD'] = pd.to_numeric(df_s['RBD'], errors='coerce')
            for c in df_s.columns:
                if c != 'RBD': df_s[c] = pd.to_numeric(df_s[c], errors='coerce')

        df_s4_sel['SIMCE_4B_AVG'] = df_s4_sel[['SIMCE_4B_LECT', 'SIMCE_4B_MATE']].mean(axis=1)
        df_s2_sel['SIMCE_2M_AVG'] = df_s2_sel[['SIMCE_2M_LECT', 'SIMCE_2M_MATE']].mean(axis=1)

        df_master = df_master.merge(df_s4_sel, on='RBD', how='left')
        df_master = df_master.merge(df_s2_sel, on='RBD', how='left')

    except Exception as e:
        print(f"⚠️ Advertencia SIMCE: {e}")

    # -------------------------------------------------------------------------
    # 5. FILTRO GEOGRÁFICO Y REPRESENTATIVIDAD (CORRECCIÓN CRÍTICA)
    # -------------------------------------------------------------------------
    print("5. Aplicando Filtro Geográfico y Limpieza de 'Retazos'...")
    n_before = len(df_master)
    
    # A. Recorte Geográfico (Bounding Box)
    mask_bbox = (
        (df_master['LONGITUD'] >= BBOX[0]) & (df_master['LONGITUD'] <= BBOX[2]) & 
        (df_master['LATITUD'] >= BBOX[1]) & (df_master['LATITUD'] <= BBOX[3])
    )
    df_final = df_master[mask_bbox].copy()
    
    # B. Filtro de Representatividad (Eliminar comunas "mutiladas")
    # Contamos cuántos colegios quedaron por comuna tras el recorte
    conteo_comunal = df_final['NOM_COM_RBD'].value_counts()
    
    # Identificamos comunas con más de 3 colegios
    comunas_validas = conteo_comunal[conteo_comunal > MIN_COLEGIOS_POR_COMUNA].index
    comunas_eliminadas = conteo_comunal[conteo_comunal <= MIN_COLEGIOS_POR_COMUNA].index.tolist()
    
    # Filtramos
    df_final = df_final[df_final['NOM_COM_RBD'].isin(comunas_validas)].copy()
    
    print(f"   -> Establecimientos en RM: {n_before}")
    print(f"   -> Establecimientos tras BBox: {mask_bbox.sum()}")
    print(f"   -> Establecimientos finales (Comunas válidas): {len(df_final)}")
    print(f"   -> 🗑️ Comunas eliminadas por tener <= {MIN_COLEGIOS_POR_COMUNA} colegios (Retazos):")
    print(f"      {', '.join(comunas_eliminadas)}")

    # -------------------------------------------------------------------------
    # 6. GUARDAR
    # -------------------------------------------------------------------------
    output_path = 'data/processed/base_consolidada_rm_2024_final.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ PROCESO COMPLETADO. Archivo maestro actualizado en: {output_path}")

if __name__ == "__main__":
    main()