import pandas as pd
import numpy as np
import unicodedata
import os

def normalizar_texto(texto):
    """
    Estandariza strings: Mayúsculas, sin tildes, sin espacios extra.
    """
    if pd.isna(texto):
        return "DESCONOCIDO"
    
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def main():
    print("--- Iniciando Pre-procesamiento (V2: Sin Parvularios) ---")
    
    os.makedirs('data/processed', exist_ok=True)

    # 1. Cargar los datos raw
    print("Cargando archivos CSV...")
    try:
        # Ajusta low_memory=False para evitar warnings de tipos mixtos
        df_ee = pd.read_csv('data/raw/EE_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_mat = pd.read_csv('data/raw/Matricula_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_doc = pd.read_csv('data/raw/Docente_2024.csv', sep=';', encoding='utf-8', low_memory=False)
    except FileNotFoundError as e:
        print(f"ERROR: Falta un archivo. {e}")
        return

    # 2. Normalización de Llaves (RBD)
    print("Normalizando llaves RBD...")
    for df in [df_ee, df_mat, df_doc]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')
        df.dropna(subset=['RBD'], inplace=True)
        df['RBD'] = df['RBD'].astype(int)

    # ==============================================================================
    # NUEVO FILTRO: ELIMINAR ESTABLECIMIENTOS SOLO DE PÁRVULOS
    # ==============================================================================
    print("Filtrando establecimientos que SOLO imparten Educación Parvularia...")
    
    # Las columnas ENS_01 a ENS_11 indican los niveles que ofrece el colegio.
    # Código 10 = Educación Parvularia.
    # Estrategia: Si un colegio tiene SOLO el código 10 (y ceros/nulos), se va.
    
    # 1. Seleccionamos columnas de enseñanza
    cols_ens = [f'ENS_{i:02d}' for i in range(1, 12)]
    
    # 2. Reemplazamos 0 por NaN para no contarlos
    ens_data = df_ee[cols_ens].replace(0, np.nan)
    
    # 3. Función para detectar si es "Solo Párvulo"
    def es_solo_parvulo(row):
        niveles = set(row.dropna().unique())
        # Si el único nivel presente es 10 (o 10.0), es parvulario puro
        if len(niveles) == 1 and 10 in niveles:
            return True
        # Si tiene 10 y otros (ej. 110 Básica), es colegio mixto -> SE QUEDA.
        # Si no tiene niveles (caso raro), lo marcamos para revisión (o borrar).
        return False

    # 4. Aplicar filtro
    mask_parvulos = ens_data.apply(es_solo_parvulo, axis=1)
    n_antes = len(df_ee)
    df_ee = df_ee[~mask_parvulos].copy()
    n_despues = len(df_ee)
    
    print(f"   -> Eliminados {n_antes - n_despues} jardines infantiles/salas cuna.")
    print(f"   -> Quedan {n_despues} establecimientos escolares.")
    # ==============================================================================

    # 3. Filtrar solo Región Metropolitana (13)
    df_ee_rm = df_ee[df_ee['COD_REG_RBD'] == 13].copy()

    # 4. Estandarización de Texto (Comunas)
    print("Estandarizando nombres de comunas...")
    df_ee_rm['NOM_COM_RBD'] = df_ee_rm['NOM_COM_RBD'].apply(normalizar_texto)
    
    # 5. Selección de Columnas
    cols_ee = [
        'RBD', 'NOM_RBD', 'COD_COM_RBD', 'NOM_COM_RBD', 
        'COD_DEPE', 'COD_DEPE2', 'LATITUD', 'LONGITUD', 'RURAL_RBD',
        'PAGO_MENSUAL'
    ]
    # Asegurarnos de que existen
    cols_final = [c for c in cols_ee if c in df_ee_rm.columns]
    df_ee_rm = df_ee_rm[cols_final]

    # 6. Merge con Matrícula y Docentes
    print("Cruzando bases de datos...")
    # Agrupar matrícula (Total por colegio)
    df_mat_g = df_mat.groupby('RBD')['MAT_TOTAL'].sum().reset_index()
    # Recuperar cursos para ratio de hacinamiento
    if 'CUR_SIM_TOT' in df_mat.columns:
        df_cursos = df_mat.groupby('RBD')['CUR_SIM_TOT'].sum().reset_index()
        df_mat_g = df_mat_g.merge(df_cursos, on='RBD', how='left')
    
    # Agrupar docentes
    df_doc_g = df_doc.groupby('RBD')['DC_TOT'].sum().reset_index()
    
    # Merge maestro
    df_master = df_ee_rm.merge(df_mat_g, on='RBD', how='left')
    df_master = df_master.merge(df_doc_g, on='RBD', how='left')
    
    # 7. Limpieza Final y Cálculos
    df_master = df_master.dropna(subset=['LATITUD', 'LONGITUD', 'DC_TOT', 'MAT_TOTAL'])
    df_master = df_master[(df_master['DC_TOT'] > 0) & (df_master['MAT_TOTAL'] > 0)]

    # Indicadores
    df_master['ratio_alumno_docente'] = df_master['MAT_TOTAL'] / df_master['DC_TOT']
    
    if 'CUR_SIM_TOT' in df_master.columns:
        df_master['ratio_alumno_curso'] = df_master.apply(
            lambda x: x['MAT_TOTAL'] / x['CUR_SIM_TOT'] if x['CUR_SIM_TOT'] > 0 else np.nan, axis=1
        )

    # Clasificación Dependencia
    def clasificar_dependencia(cod):
        if cod == 1: return 'MUNICIPAL_CORP'
        elif cod == 2: return 'MUNICIPAL_DAEM'
        elif cod == 3: return 'PARTICULAR_SUBV'
        elif cod == 4: return 'PARTICULAR_PAGADO'
        elif cod == 5: return 'ADMIN_DELEGADA'
        elif cod == 6: return 'SLEP'
        else: return 'OTRO'

    df_master['categoria_dependencia'] = df_master['COD_DEPE'].apply(clasificar_dependencia)

    # Limpieza Precio
    precio_map = {
        'GRATUITO': 0,
        '$1.000 A $10.000': 1,
        '$10.001 A $25.000': 2,
        '$25.001 A $50.000': 3,
        '$50.001 A $100.000': 4,
        'MAS DE $100.000': 5,
        'SIN INFORMACION': -1
    }
    df_master['PAGO_MENSUAL_NORM'] = df_master['PAGO_MENSUAL'].apply(lambda x: str(x).upper().strip())
    df_master['orden_precio'] = df_master['PAGO_MENSUAL_NORM'].map(precio_map).fillna(-1)

    # 8. Guardar
    output_path = 'data/processed/base_consolidada_rm_2024.csv'
    df_master.to_csv(output_path, index=False)
    
    print(f"¡Listo! Base filtrada guardada en {output_path}")
    print(f"Total establecimientos finales: {len(df_master)}")

if __name__ == "__main__":
    main()