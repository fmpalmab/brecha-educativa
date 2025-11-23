import pandas as pd
import numpy as np
import unicodedata
import os

def normalizar_texto(texto):
    """
    Estandariza strings: Mayúsculas, sin tildes, sin espacios extra.
    Maneja el caso de la Ñ separando tildes de caracteres base.
    """
    if pd.isna(texto):
        return "DESCONOCIDO"
    
    # 1. Convertir a string y mayúsculas
    texto = str(texto).upper()
    
    # 2. Normalizar caracteres unicode (NFD separa letras de tildes)
    texto = unicodedata.normalize('NFD', texto)
    
    # 3. Eliminar caracteres de combinación (las tildes)
    # Ojo: Esto transformará 'Ñ' en 'N' + '~', y borrará la '~'. 
    # Si prefieres mantener la Ñ, avísame, pero para agrupar datos 'N' es más seguro.
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    
    # 4. Eliminar espacios al inicio y final
    return texto.strip()

def main():
    print("--- Iniciando Pre-procesamiento (Corrección de Texto) ---")
    
    # Asegurar directorios
    os.makedirs('data/processed', exist_ok=True)

    # 1. Cargar los datos raw
    print("Cargando archivos CSV...")
    try:
        df_ee = pd.read_csv('data/raw/EE_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_mat = pd.read_csv('data/raw/Matricula_2024.csv', sep=';', encoding='utf-8', low_memory=False)
        df_doc = pd.read_csv('data/raw/Docente_2024.csv', sep=';', encoding='utf-8', low_memory=False)
    except FileNotFoundError as e:
        print(f"ERROR: Falta un archivo. Detalle: {e}")
        return

    # 2. Normalización de Llaves (RBD) - CRÍTICO
    print("Normalizando llaves RBD...")
    for df in [df_ee, df_mat, df_doc]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')
        df.dropna(subset=['RBD'], inplace=True)
        df['RBD'] = df['RBD'].astype(int)

    # 3. Filtrar solo Región Metropolitana (13) en el Directorio
    # Hacemos esto ANTES de normalizar texto para ahorrar tiempo
    df_ee_rm = df_ee[df_ee['COD_REG_RBD'] == 13].copy()

    # 4. APLICAR ESTANDARIZACIÓN DE TEXTO (EL FIX)
    print("Estandarizando nombres de comunas...")
    # Aplicamos la función a la columna de Comuna
    df_ee_rm['NOM_COM_RBD'] = df_ee_rm['NOM_COM_RBD'].apply(normalizar_texto)
    
    # (Opcional) Estandarizar también el nombre del colegio para búsquedas futuras
    df_ee_rm['NOM_RBD'] = df_ee_rm['NOM_RBD'].apply(normalizar_texto)

    # 5. Selección de Columnas
    cols_ee = [
        'RBD', 'NOM_RBD', 'COD_COM_RBD', 'NOM_COM_RBD', 
        'COD_DEPE', 'COD_DEPE2', 'LATITUD', 'LONGITUD', 'RURAL_RBD',
        'PAGO_MENSUAL'
    ]
    
    # Limpiar espacios en nombres de columnas por si acaso
    df_ee_rm.columns = df_ee_rm.columns.str.strip()
    # Intersección de columnas que realmente existen
    cols_final = [c for c in cols_ee if c in df_ee_rm.columns]
    df_ee_rm = df_ee_rm[cols_final]

    # 6. Merge con Matrícula y Docentes
    print("Cruzando bases de datos...")
    # Agrupar matrícula por si viene desglosada (aunque el archivo Resumen ya debería venir listo)
    # Aseguramos que sea único por RBD
    df_mat_g = df_mat.groupby('RBD')['MAT_TOTAL'].sum().reset_index()
    
    # Agrupar docentes
    df_doc_g = df_doc.groupby('RBD')['DC_TOT'].sum().reset_index()
    
    # Merge
    df_master = df_ee_rm.merge(df_mat_g, on='RBD', how='left')
    df_master = df_master.merge(df_doc_g, on='RBD', how='left')
    
    # Recuperar CUR_SIM_TOT si existe en Matricula original (útil para hacinamiento)
    if 'CUR_SIM_TOT' in df_mat.columns:
        df_cursos = df_mat.groupby('RBD')['CUR_SIM_TOT'].sum().reset_index()
        df_master = df_master.merge(df_cursos, on='RBD', how='left')

    # 7. Limpieza Final y Cálculos
    df_master = df_master.dropna(subset=['LATITUD', 'LONGITUD', 'DC_TOT', 'MAT_TOTAL'])
    df_master = df_master[(df_master['DC_TOT'] > 0) & (df_master['MAT_TOTAL'] > 0)]

    # Indicadores
    df_master['ratio_alumno_docente'] = df_master['MAT_TOTAL'] / df_master['DC_TOT']
    
    if 'CUR_SIM_TOT' in df_master.columns:
        # Evitar división por cero en cursos
        df_master['ratio_alumno_curso'] = df_master.apply(
            lambda x: x['MAT_TOTAL'] / x['CUR_SIM_TOT'] if x['CUR_SIM_TOT'] > 0 else np.nan, axis=1
        )

    # Clasificación Dependencia (Legible)
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
    # Aplicar normalización al precio también por si acaso viene con espacios
    df_master['PAGO_MENSUAL_NORM'] = df_master['PAGO_MENSUAL'].apply(lambda x: str(x).upper().strip())
    df_master['orden_precio'] = df_master['PAGO_MENSUAL_NORM'].map(precio_map).fillna(-1)

    # 8. Guardar
    output_path = 'data/processed/base_consolidada_rm_2024.csv'
    df_master.to_csv(output_path, index=False)
    
    print(f"¡Listo! Base consolidada guardada en {output_path}")
    print(f"Total establecimientos: {len(df_master)}")
    print("Ejemplo de comunas normalizadas:", df_master['NOM_COM_RBD'].unique()[:5])

if __name__ == "__main__":
    main()