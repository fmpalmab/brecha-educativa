# 🎓 Brecha Educativa RM: Econometría Espacial y Análisis de Segregación Escolar

[![CI](https://github.com/fmpalmab/brecha-educativa/actions/workflows/ci.yml/badge.svg)](https://github.com/fmpalmab/brecha-educativa/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.12%2B-285f8f.svg)](https://geopandas.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Pipeline cuantitativo, geoespacial y econométrico de ciencia de datos diseñado para analizar la **brecha de resultados, saturación de aulas y segregación socioeconómica** en el sistema escolar de la Región Metropolitana de Santiago de Chile (RM).

Integra fuentes oficiales abiertas del **Ministerio de Educación (MINEDUC)** y de la **Agencia de Calidad de la Educación (SIMCE 2024)** junto con cartografía vectorial de comunas (Shapefiles).

---

## 🏛️ Fuentes de Datos Oficiales Integradas

El pipeline ingesta y consolida automáticamente las bases maestras oficiales:
1. **Directorio Oficial de Establecimientos (EE 2024):** Ubicación geográfica (latitud, longitud), dependencia institucional (Municipal, Particular Subvencionado, Particular Pagado, SLEP) y tramo de arancel mensual.
2. **Rendimiento Docente (Docente 2024):** Horas cronológicas y dotación total de profesores por establecimiento (`DC_TOT`).
3. **Matrícula Escolar (Matrícula 2024):** Cantidad de estudiantes por grado y nivel (`MAT_TOTAL`) y número de aulas/cursos (`CUR_SIM_TOT`).
4. **Evaluaciones Estandarizadas SIMCE 2024:**
   - 4° Básico: Lectura y Matemáticas (`simce4b2024_rbd_preliminar.csv`).
   - 2° Medio: Lectura y Matemáticas (`simce2m2024_rbd_preliminar.csv`).
5. **Cartografía Censal Vectorial (Shapefiles):** Polígonos georreferenciados de comunas de la RM (`COMUNA_C17.shp`).

---

## 🏗️ Arquitectura del Sistema

El proyecto fue refactorizado como un paquete estándar `brecha_educativa` (PEP 517/621):

```
brecha-educativa/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline de CI (Pytest en 3.10, 3.11, 3.12)
├── brecha_educativa/            # Paquete principal
│   ├── __init__.py              # Exportaciones públicas del paquete
│   ├── __main__.py              # Punto de entrada ejecutable (python -m brecha_educativa)
│   ├── analysis.py              # Cálculo de Índice de Duncan, matrices de correlación y resúmenes
│   ├── cli.py                   # Interfaz de línea de comandos unificada
│   ├── config.py                # Bounding Box urbano, rutas y paletas de color
│   ├── pipeline.py              # Orquestador del flujo integral de ciencia de datos
│   ├── processor.py             # Limpieza de texto, parseo de coordenadas, cruces y generador sintético
│   ├── report.py                # Generador de informes ejecutivos en texto/markdown
│   ├── spatial.py               # Ingesta de shapefiles, recorte espacial BBox y mapas
│   └── visualizer.py            # Heatmaps de correlación y gráficos de distribución
├── data/
│   ├── external/comunas_rm/     # Shapefile de comunas metropolitanas (.shp, .dbf, .shx)
│   ├── processed/               # Base consolidada limpia y resumen comunal CSV
│   └── raw/                     # Archivos fuente MINEDUC y SIMCE
├── figures/                     # Figuras científicas exportadas en alta resolución
├── reports/                     # Reporte ejecutivo consolidado
├── tests/                       # Suite de pruebas unitarias (Pytest)
│   ├── test_analysis.py
│   ├── test_cli.py
│   ├── test_processor.py
│   └── test_synthetic.py
└── pyproject.toml               # Configuración de empaquetado y herramientas
```

---

## 🧮 Metodología Econométrica y Espacial

### 1. Índice de Disimilitud de Duncan (Segregación Escolar)

Mide la segregación residencial/educativa entre estudiantes de colegios pagados y gratuitos:

$$D = \frac{1}{2} \sum_{i=1}^{K} \left| \frac{p_i}{P_{\text{total}}} - \frac{g_i}{G_{\text{total}}} \right|$$

- $p_i$: Estudiantes en colegios pagados en la comuna $i$.
- $g_i$: Estudiantes en colegios gratuitos en la comuna $i$.
- **Interpretación:** $D \in [0, 1]$. En el Gran Santiago, $D \approx 0.36 - 0.45$, lo cual señala una alta concentración territorial y segregación socioeconómica.

### 2. Ratios de Capacidad y Sobrecarga

- **Ratio Alumnos por Docente:** $\text{Ratio}_{\text{Docente}} = \frac{\text{MAT\_TOTAL}}{\text{DC\_TOT}}$
- **Saturación de Aulas (Alumnos por Curso):** $\text{Ratio}_{\text{Aula}} = \frac{\text{MAT\_TOTAL}}{\text{CUR\_SIM\_TOT}}$

---

## ⚡ Instalación

Requiere **Python 3.10+**.

```bash
git clone https://github.com/fmpalmab/brecha-educativa.git
cd brecha-educativa

# Instalación estándar
pip install -e .

# Con dependencias de pruebas
pip install -e ".[dev]"
```

---

## 🚀 Uso del CLI Unificado

```bash
# 1. Ejecutar pipeline completo (procesar, analizar, graficar y reportar)
python -m brecha_educativa --pipeline

# 2. Análisis longitudinal comparativo multianual (Evolución de brecha, Duncan e impacto temporal)
python -m brecha_educativa --longitudinal --years 2022,2023,2024,2025

# 3. Generar exclusivamente el mapa espacial de colegios gratuitos vs pagados
python -m brecha_educativa --map

# 4. Procesar y consolidar archivos crudos de MINEDUC y SIMCE
python -m brecha_educativa --process

# 5. Imprimir reporte ejecutivo de hallazgos estadísticos
python -m brecha_educativa --report

# 6. Generar dataset sintético para pruebas locales o CI sin archivos crudos
python -m brecha_educativa --synth --schools 200
```

### Compatibilidad Histórica

Los scripts originales continúan operativos:
```bash
python 01_procesamiento.py
python 02_analisis_exploratorio.py
python 03_analisis_geoespacial.py
python 04_analisis_segregacion.py
python 05_analisis_estadistico_final_v2.py
```

---

## 🧪 Pruebas Automatizadas

```bash
pytest -v
```

La suite cubre normalización tipográfica, limpieza de coordenadas geográficas, invarianzas matemáticas del índice de Duncan y ejecución CLI.

---

## 📜 Licencia

Distribuido bajo la Licencia **MIT**. Desarrollado en la Universidad de Chile.
