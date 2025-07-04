# ⚾ Trabajo Fin de Máster – Sistema de Proyección de Rendimiento en MLB con Machine Learning

![Vista previa del sistema](images/preview_radar.png)


![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

> **Trabajo Fin de Máster** - Máster en Data Science, Big Data & Business Analytics  
> **Universidad Complutense de Madrid (UCM)** - 2024  
> **Autor**: Sergio Grigorow

## 🎯 **Objetivo del Proyecto**

Desarrollar un sistema híbrido que combine técnicas de clustering y modelado predictivo para proyectar el rendimiento ofensivo de jugadores de MLB, superando la precisión de métodos tradicionales.  

- **Clustering de arquetipos de jugadores** (ajustado por era temporal)
- **Modelado de series temporales** para capturar patrones de envejecimiento
- **Features de contexto** que mejoren las predicciones individuales

## 📊 **Resultados Principales**

### Performance del Modelo
- **25.6% mejora** vs baseline (OPS año anterior)
- **MAE: 0.0589** (vs 0.0759 del baseline)
- **R²: 0.5276** - explica >50% de la varianza
- **Competitivo vs sistemas profesionales**: 39.0% casos ganados vs ZiPS (36.2%)

### Benchmarking contra Sistemas Profesionales
| Sistema | MAE | R² | % Casos Ganados |
|---------|-----|----|-----------------| 
| **Nuestro Modelo** | 0.0814 | 0.15 | **39.4%** |
| ZiPS | 0.0804 | 0.17 | 36.6% |
| Steamer | 0.0829 | 0.16 | 24.0% |

## 🏗️ **Arquitectura del Sistema**

### 1. **Clustering Robusto con Ajuste por Era** 
- 7 arquetipos de jugadores identificados
- Variables ajustadas por contexto temporal (1940-2023)
- Eliminación de sesgos entre diferentes eras del baseball

### 2. **Feature Engineering Avanzado**
- **48.9%** contribución de features temporales
- **18.8%** contribución del clustering
- **17.8%** contribución de variables era-ajustadas
- Más de 90 variables predictivas generadas

### 3. **Modelado Híbrido**
- Random Forest optimizado con 200 árboles
- Integración de patrones individuales y grupales
- Validación temporal estricta (2015-2023)

## 📁 **Estructura del Proyecto**

```
baseball-career-projection/
├── data/
│   ├── raw/                   # Datos originales (Batting.csv, Fielding.csv, People.csv)
│   └── processed/             # Datos procesados y features
├── models/                    # Modelos entrenados y metadata
├── TFMasterV2.ipynb           # Notebook principal completo
├── app.py                     # Aplicación Streamlit
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## 🚀 **Instalación y Uso**

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tgrigorow1974/TrabajoFinDeMaster.git
cd Trabajo-Version-2
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Notebook
```bash
jupyter notebook notebooks/TFMasterV2.ipynb
```

### 4. Lanzar la Aplicación Web
```bash
streamlit run app.py
```

## 🔍 **Datos Utilizados**

- **Periodo**: 1940-2023 (83 años de historia de MLB)
- **Registros de bateo**: 113,799
- **Registros de fildeo**: 151,507  
- **Jugadores únicos**: 21,010
- **Fuente**: Bases de datos oficiales de MLB vía Lahman Database

## 🧮 **Metodología**

### Preprocessing y Feature Engineering
1. **Limpieza y consolidación** de datos multi-equipo
2. **Normalización temporal** para ajustar por era
3. **Cálculo de métricas avanzadas** (ISO, wOBA, BABIP, etc.)
4. **Aging curves** mediante método delta

### Clustering
1. **Selección de variables** representativas del estilo de juego
2. **Ajuste por era** para eliminar sesgos temporales
3. **Optimización** mediante Silhouette Score (7 clusters óptimos)
4. **Asignación** de todos los registros históricos

### Modelado Predictivo
1. **División temporal** estricta (train hasta 2014, test 2015-2023)
2. **Feature selection** automática por importancia
3. **Optimización de hiperparámetros** con grid search
4. **Validación** con casos históricos reales

## 📈 **Aplicaciones Prácticas**

- **Evaluación de contratos**: Análisis de riesgo para agentes libres
- **Desarrollo de talento**: Identificación de trayectorias esperadas  
- **Análisis comparativo**: Rankings y evaluaciones entre jugadores
- **Herramienta de scouting**: Proyecciones para toma de decisiones

## 🔬 **Validación del Modelo**

### Caso de Estudio: Jeff Bagwell
- **Trayectoria completa**: 1991-2005 (15 temporadas)
- **Validación histórica**: Proyecciones vs performance real
- **Captura de tendencias**: Identificación correcta de patrones de declive

### Validación Externa
- **Comparación directa** con ZiPS y Steamer (2024)
- **Caso de prueba**: Bagwell
- **Competitividad demostrada** en escenarios reales

## 👨‍💼 **Autor y Supervisión**

- **Autor**: Sergio Grigorow
- **Tutores**: Carlos Ortega, Santiago Mota  
- **Universidad**: Complutense de Madrid
- **Programa**: Máster Data Science, Big Data & Business Analytics
- **Año**: 2024-2025

## 📄 **Licencia**

Este proyecto está desarrollado con fines académicos como Trabajo Fin de Máster.

## 🤝 **Contribuciones**

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 🚀 **Aplicación Interactiva**

- Aplicación interactiva: [Abrir en Streamlit](https://tu-url.streamlit.app)
---

⚾ **¡Disfruta explorando el futuro del baseball analytics!**
