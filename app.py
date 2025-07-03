import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import BytesIO
import base64

# Diccionario de logos de equipos
team_logos = {
    "ANA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "ARI": "https://a.espncdn.com/i/teamlogos/mlb/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/mlb/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/mlb/500/bal.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/mlb/500/bos.png",
    "CHN": "https://a.espncdn.com/i/teamlogos/mlb/500/chc.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/mlb/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/mlb/500/cle.png",
    "COL": "https://a.espncdn.com/i/teamlogos/mlb/500/col.png",
    "CHA": "https://a.espncdn.com/i/teamlogos/mlb/500/chw.png",
    "DET": "https://a.espncdn.com/i/teamlogos/mlb/500/det.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/mlb/500/hou.png",
    "KCR": "https://a.espncdn.com/i/teamlogos/mlb/500/kc.png",
    "LAA": "https://a.espncdn.com/i/teamlogos/mlb/500/laa.png",
    "LAN": "https://a.espncdn.com/i/teamlogos/mlb/500/lad.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/mlb/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/mlb/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/mlb/500/min.png",
    "NYN": "https://a.espncdn.com/i/teamlogos/mlb/500/nym.png",
    "NYA": "https://a.espncdn.com/i/teamlogos/mlb/500/nyy.png",
    "OAK": "https://a.espncdn.com/i/teamlogos/mlb/500/oak.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/mlb/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/mlb/500/pit.png",
    "SDN": "https://a.espncdn.com/i/teamlogos/mlb/500/sd.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/mlb/500/sea.png",
    "SFG": "https://a.espncdn.com/i/teamlogos/mlb/500/sf.png",
    "STL": "https://a.espncdn.com/i/teamlogos/mlb/500/stl.png",
    "TBR": "https://a.espncdn.com/i/teamlogos/mlb/500/tb.png",
    "TEX": "https://a.espncdn.com/i/teamlogos/mlb/500/tex.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/mlb/500/tor.png",
    "WSN": "https://a.espncdn.com/i/teamlogos/mlb/500/wsh.png"
}

st.set_page_config(page_title="Predicción MLB OPS", page_icon="⚾", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================

@st.cache_data
def load_data():
    """Carga datos completos y lista de jugadores elegibles."""
    try:
        df_all = pd.read_csv("data/processed/baseball_with_cluster_features.csv")
        eligible_players = pd.read_csv("data/processed/eligible_players_app.csv")
        return df_all, eligible_players
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None

@st.cache_resource
def load_model():
    """Carga modelo y metadata."""
    try:
        model = joblib.load('models/best_model_final.pkl')
        feature_cols = joblib.load('models/feature_columns_final.pkl')
        model_metadata = joblib.load('models/model_metadata_final.pkl')
        return model, feature_cols, model_metadata
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None, None

# =============================================================================
# FUNCIONES DE PREDICCIÓN (SOLO FEATURES ERA-AJUSTADAS)
# =============================================================================

def predict_individual_player(player_id, df_all, model, feature_cols, model_metadata, years=3):
    """Predice OPS para un jugador individual usando solo features era-ajustadas."""
    
    player_data = df_all[df_all['playerID'] == player_id].sort_values('yearID')
    
    if len(player_data) == 0:
        return None, "No se encontraron datos para este jugador"
    
    latest_data = player_data.iloc[-1].copy()
    player_name = latest_data['name']
    player_team = latest_data['teamID']
    
    # Obtener estadísticas de carrera
    career_ops = player_data['OPS'].mean()
    ops_2023 = latest_data['OPS']
    
    # Verificar que tenemos todas las features necesarias
    missing_features = [f for f in feature_cols if f not in latest_data.index or pd.isna(latest_data[f])]
    if missing_features:
        return None, f"Faltan features: {missing_features[:5]}..."
    
    model_error = model_metadata['validation_mae']
    
    predictions = []
    current_data = latest_data.copy()
    
    for year_offset in range(1, years + 1):
        pred_year = 2023 + year_offset
        
        # Actualizar características dependientes de la edad
        current_data['age'] = latest_data['age'] + year_offset
        current_data['experience'] = latest_data['experience'] + year_offset
        current_data['age_squared'] = current_data['age'] ** 2
        current_data['BMI_age_interaction'] = current_data['BMI'] * current_data['age']
        current_data['is_peak_years'] = 1 if 26 <= current_data['age'] <= 30 else 0
        current_data['is_rookie'] = 0
        
        # Usar solo las features que el modelo espera
        X_pred = current_data[feature_cols].values.reshape(1, -1)
        pred = model.predict(X_pred)[0]
        
        predictions.append({
            'year': pred_year,
            'age': int(current_data['age']),
            'pesimista': max(pred - model_error, 0.400),
            'realista': pred,
            'optimista': min(pred + model_error, 1.400)
        })
    
    return {
        'player_name': player_name,
        'player_id': player_id,
        'age_2024': int(latest_data['age'] + 1),
        'team': player_team,
        'career_ops': career_ops,
        'ops_2023': ops_2023,
        'predictions': predictions
    }, None

def predict_multiple_players(player_ids, df_all, model, feature_cols, model_metadata, eligible_players):
    """Predice OPS para múltiples jugadores usando solo features era-ajustadas."""
    
    results = []
    
    for player_id in player_ids:
        player_info = eligible_players[eligible_players['playerID'] == player_id].iloc[0]
        player_data = df_all[df_all['playerID'] == player_id].sort_values('yearID')
        latest_data = player_data.iloc[-1].copy()
        
        # Verificar que tenemos todas las features necesarias
        missing_features = [f for f in feature_cols if f not in latest_data.index or pd.isna(latest_data[f])]
        if missing_features:
            continue
        
        # Preparar datos para 2024
        pred_data = latest_data.copy()
        pred_data['age'] = latest_data['age'] + 1
        pred_data['experience'] = latest_data['experience'] + 1
        pred_data['age_squared'] = pred_data['age'] ** 2
        pred_data['BMI_age_interaction'] = pred_data['BMI'] * pred_data['age']
        pred_data['is_peak_years'] = 1 if 26 <= pred_data['age'] <= 30 else 0
        pred_data['is_rookie'] = 0
        
        # Usar solo las features que el modelo espera
        X_pred = pred_data[feature_cols].values.reshape(1, -1)
        pred = model.predict(X_pred)[0]
        
        model_error = model_metadata['validation_mae']
        
        results.append({
            'player_name': player_info['name'],
            'age_2024': int(player_info['age_2023'] + 1),
            'team': player_info['team_2023'],
            'pesimista': max(pred - model_error, 0.400),
            'realista': pred,
            'optimista': min(pred + model_error, 1.400)
        })
    
    return results

# =============================================================================
# CARGAR DATOS Y MODELO
# =============================================================================

df_all, eligible_players = load_data()
model, feature_cols, model_metadata = load_model()

if df_all is None or model is None:
    st.stop()

# Verificar features era-ajustadas
era_features = [f for f in feature_cols if 'era_adjusted' in f] if feature_cols else []
era_percentage = (len(era_features) / len(feature_cols) * 100) if feature_cols else 0

# Lista de jugadores disponibles
lista_jugadores = eligible_players['name'].tolist()

# =============================================================================
# INFORMACIÓN EN SIDEBAR
# =============================================================================

st.sidebar.markdown("### 📊 Información del Sistema")
st.sidebar.markdown(f"**Jugadores disponibles:** {len(lista_jugadores):,}")
st.sidebar.markdown(f"**Registros totales:** {len(df_all):,}")
st.sidebar.markdown(f"**Años cubiertos:** {df_all['yearID'].min()}-{df_all['yearID'].max()}")
st.sidebar.markdown(f"**MAE del modelo:** {model_metadata['validation_mae']:.4f}")
st.sidebar.markdown("### 🎯 Características del Modelo")
st.sidebar.markdown(f"**Features era-ajustadas:** {len(era_features)}/{len(feature_cols)} ({era_percentage:.1f}%)")
st.sidebar.markdown("✅ **Consistencia metodológica**: Solo features era-ajustadas")
st.sidebar.markdown("✅ **Sin redundancia**: No variables originales")

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================

st.markdown(
    """
    <style>
        @media only screen and (max-width: 768px) {
            .header-container {
                flex-direction: column !important;
                text-align: center !important;
            }
            .header-container img {
                margin: 10px auto !important;
            }
        }
    </style>

    <div class='header-container' style='
        background-color:#002654;
        padding:18px;
        border-radius:12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    '>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Major_League_Baseball_logo.svg/1024px-Major_League_Baseball_logo.svg.png'
             width='140' style='margin: 10px; border-radius:12px; background:white; padding:6px 14px;'/>
        <div style='flex-grow:1; text-align:center; min-width: 250px;'>
            <h2 style='color:white; margin-bottom: 9px; font-size: 1.8em;'>⚾ Predicción de Rendimiento OPS en MLB ⚾</h2>
            <p style='color:white; font-size:18px; margin:0;'>Herramienta interactiva de Machine Learning para proyectar desempeño ofensivo</p>
        </div>
        <img src='https://sabr.org/sites/default/files/SABR_logo-square-700px.png'
             width='100' style='margin: 10px; border-radius:14px; background:white; padding:6px 14px;'/>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# SELECCIÓN DE MODO Y JUGADORES
# =============================================================================

modo = st.radio(
    "Selecciona el tipo de predicción:",
    ("Predicción individual", "Predicción múltiple"),
    horizontal=True
)

if modo == "Predicción individual":
    jugador_seleccionado = st.selectbox("Selecciona el jugador:", lista_jugadores)
else:
    jugadores_seleccionados = st.multiselect(
        "Selecciona uno o más jugadores:", 
        lista_jugadores, 
        default=lista_jugadores[:2] if len(lista_jugadores) >= 2 else lista_jugadores[:1]
    )

# =============================================================================
# BOTÓN Y RESULTADOS
# =============================================================================

if st.button("Predecir"):
    
    if modo == "Predicción individual":
        # Obtener ID del jugador
        player_id = eligible_players[eligible_players['name'] == jugador_seleccionado]['playerID'].iloc[0]
        
        resultado, error = predict_individual_player(
            player_id, df_all, model, feature_cols, model_metadata
        )
        
        if error:
            st.error(error)
        else:
            team_id = resultado['team']
            logo_url = team_logos.get(team_id, None)

            # Mostrar información del jugador
            st.markdown(f"""
            <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 16px;'>
                <div style='font-size:1.38em; line-height:1.22; text-align: center;'>
                    <b>Jugador:</b> {resultado['player_name']}<br>
                    <b>Edad en 2024:</b> {resultado['age_2024']}<br>
                    <b>OPS Carrera:</b> {resultado['career_ops']:.3f}<br>
                    <b>OPS 2023:</b> {resultado['ops_2023']:.3f}
                </div>
                <div style='margin-left:36px; text-align:center;'>
                    {"<img src='" + logo_url + "' width='155' style='border-radius:12px; background:white; padding:6px 18px; vertical-align:middle;'/>" if logo_url else f"<span style='color:#ccc;'>Sin logo<br>({team_id})</span>"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar predicciones por año
            for pred in resultado['predictions']:
                year = pred['year']
                age = pred['age'] 
                pesimista = pred['pesimista']
                realista = pred['realista']
                optimista = pred['optimista']

                # Determinar color y emoji según rendimiento
                if realista >= 0.900:
                    color = "#28a745"
                    emoji = "🔥"
                    text_color = "#1e7e34"
                elif realista >= 0.800:
                    color = "#17a2b8"
                    emoji = "⭐"
                    text_color = "#117a8b"
                elif realista >= 0.700:
                    color = "#ffc107"
                    emoji = "👍"
                    text_color = "#d39e00"
                else:
                    color = "#dc3545"
                    emoji = "⚠️"
                    text_color = "#bd2130"

                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}15, {color}05); 
                            border-left: 4px solid {color}; 
                            padding: 15px 20px; 
                            margin: 8px auto; 
                            border-radius: 8px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            max-width: 600px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='font-size: 20px; font-weight: bold; color: {text_color};'>
                            {emoji} {year} (Edad {age})
                        </div>
                        <div style='font-size: 28px; font-weight: bold; color: {text_color};'>
                            {realista:.3f}
                        </div>
                    </div>
                    <div style='margin-top: 8px; display: flex; justify-content: space-between; font-size: 14px;'>
                        <span style='color: #721c24;'>📉 Pesimista: <b>{pesimista:.3f}</b></span>
                        <span style='color: #495057;'>📊 Realista: <b>{realista:.3f}</b></span>
                        <span style='color: #155724;'>📈 Optimista: <b>{optimista:.3f}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Gráfico de proyección mejorado con histórico
            if resultado['predictions']:
                # Obtener datos históricos del jugador
                player_historical = df_all[df_all['playerID'] == player_id].sort_values('yearID')
                
                # Datos históricos (últimos 3 años)
                last_3_years = player_historical.tail(3)
                years_hist = last_3_years['yearID'].values
                ops_hist = last_3_years['OPS'].values
                
                # Datos de predicción
                years_pred = [pred['year'] for pred in resultado['predictions']]
                ops_pred = [pred['realista'] for pred in resultado['predictions']]
                ops_lower = [pred['pesimista'] for pred in resultado['predictions']]
                ops_upper = [pred['optimista'] for pred in resultado['predictions']]
                
                # Conectar 2023 con predicciones
                years_pred_full = np.array([years_hist[-1]] + years_pred)
                ops_pred_full = np.array([ops_hist[-1]] + ops_pred)
                ops_lower_full = np.array([ops_hist[-1]] + ops_lower)
                ops_upper_full = np.array([ops_hist[-1]] + ops_upper)
                
                # Crear gráfico
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(14, 8))
                fig.patch.set_facecolor('white')
                
                # Línea histórica (azul)
                ax.plot(years_hist, ops_hist, marker='o', markersize=8, 
                        linewidth=3, color='#1e40af', label='Histórico')
                
                # Línea de predicción (naranja)
                ax.plot(years_pred_full, ops_pred_full, marker='s', markersize=8,
                        linewidth=3, color='#f59e0b', label='Predicción')
                
                # Bandas de confianza que se abren desde 2023
                ax.fill_between(years_pred_full, ops_lower_full, ops_upper_full, 
                               alpha=0.3, color='#f59e0b', label='Banda Confianza')
                
                # Configuración del gráfico
                ax.set_title(f'Proyección de Carrera - {resultado["player_name"]}', 
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Año', fontsize=12, fontweight='bold')
                ax.set_ylabel('OPS', fontsize=12, fontweight='bold')
                ax.legend(fontsize=11, frameon=True, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#f8fafc')
                
                # Ajustar límites del eje Y
                all_values = np.concatenate([ops_hist, ops_lower, ops_upper])
                y_min, y_max = all_values.min(), all_values.max()
                y_margin = (y_max - y_min) * 0.2
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    else:  # Predicción múltiple
        if not jugadores_seleccionados:
            st.error("Selecciona al menos un jugador.")
        else:
            # Obtener IDs de jugadores
            player_ids = []
            for nombre in jugadores_seleccionados:
                player_id = eligible_players[eligible_players['name'] == nombre]['playerID'].iloc[0]
                player_ids.append(player_id)
            
            resultados = predict_multiple_players(
                player_ids, df_all, model, feature_cols, model_metadata, eligible_players
            )

            if not resultados:
                st.error("No se encontraron jugadores válidos.")
            else:
                # Crear tabla HTML
                st.markdown("### 📋 Resultados por Jugador - Predicción 2024")

                tabla_html = (
                    "<table style='width:100%; border-collapse:collapse; font-size:18px; text-align:center;'>"
                    "<thead>"
                    "<tr style='background-color:#003366; color:white;'>"
                    "<th style='padding:10px;'>Jugador</th>"
                    "<th style='padding:10px;'>Edad</th>"
                    "<th style='padding:10px;'>Equipo</th>"
                    "<th style='padding:10px;'>OPS Pesimista</th>"
                    "<th style='padding:10px;'>OPS Realista</th>"
                    "<th style='padding:10px;'>OPS Optimista</th>"
                    "</tr>"
                    "</thead><tbody>"
                )

                for result in resultados:
                    tabla_html += (
                        f"<tr>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{result['player_name']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{result['age_2024']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{result['team']}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{result['pesimista']:.3f}</td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'><b>{result['realista']:.3f}</b></td>"
                        f"<td style='padding:8px; border-bottom:1px solid #ddd;'>{result['optimista']:.3f}</td>"
                        f"</tr>"
                    )

                tabla_html += "</tbody></table>"
                st.markdown(tabla_html, unsafe_allow_html=True)

                # Gráfico estilo original
                player_names = [result['player_name'][:15] for result in resultados]
                pesimistas = [result['pesimista'] for result in resultados]
                realistas = [result['realista'] for result in resultados]
                optimistas = [result['optimista'] for result in resultados]

                n_players = len(player_names)

                # Posiciones x para cada grupo
                x_pesimista = np.arange(n_players)
                x_realista = x_pesimista + n_players + 0.5
                x_optimista = x_pesimista + 2 * n_players + 1

                fig, ax = plt.subplots(figsize=(16, 6))

                # Barras por tipo
                ax.bar(x_pesimista, pesimistas, color='lightcoral', alpha=0.8, label='Pesimista')
                ax.bar(x_realista, realistas, color='steelblue', alpha=0.8, label='Realista')
                ax.bar(x_optimista, optimistas, color='lightgreen', alpha=0.8, label='Optimista')

                # Valores encima de cada barra
                for i in range(n_players):
                    ax.text(x_pesimista[i], pesimistas[i] + 0.005, f'{pesimistas[i]:.3f}', 
                           ha='center', fontsize=9)
                    ax.text(x_realista[i], realistas[i] + 0.005, f'{realistas[i]:.3f}', 
                           ha='center', fontsize=9, fontweight='bold')
                    ax.text(x_optimista[i], optimistas[i] + 0.005, f'{optimistas[i]:.3f}', 
                           ha='center', fontsize=9)

                # Configurar ejes
                all_x = np.concatenate([x_pesimista, x_realista, x_optimista])
                all_labels = player_names + player_names + player_names
                ax.set_xticks(all_x)
                ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=10)
                ax.set_xlabel('Jugadores por Tipo de Predicción', fontsize=12)
                ax.set_ylabel('OPS 2024', fontsize=12)

                ax.set_ylim([0.35, min(1.45, max(optimistas + realistas + pesimistas) * 1.15)])

                # Líneas divisorias entre grupos
                if n_players > 1:
                    ax.axvline(x=n_players - 0.5, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=2 * n_players + 0.5, color='gray', linestyle='--', alpha=0.5)

                # Etiquetas de grupo
                y_top = ax.get_ylim()[1]
                ax.text(np.mean(x_pesimista), y_top * 0.96, 'PESIMISTA', 
                       ha='center', fontsize=12, fontweight='bold', color='red')
                ax.text(np.mean(x_realista), y_top * 0.96, 'REALISTA', 
                       ha='center', fontsize=12, fontweight='bold', color='blue')
                ax.text(np.mean(x_optimista), y_top * 0.96, 'OPTIMISTA', 
                       ha='center', fontsize=12, fontweight='bold', color='green')

                # Título
                model_name = model_metadata['model_name'] if model_metadata else 'Random Forest'
                ax.set_title(
                    f"Sistema Solo Era-Ajustado OPS 2024: Pesimista | Realista | Optimista\n"
                    f"({model_name}, {len(feature_cols)} features, {len(era_features)} era-ajustadas)",
                    fontsize=14, fontweight='bold'
                )

                ax.grid(axis='y', linestyle='--', alpha=0.3)
                ax.legend()
                plt.tight_layout()

                st.pyplot(fig)

# =============================================================================
# PIE DE PÁGINA
# =============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:16px;color:gray;'>Sistema de Machine Learning para proyección de carreras en Baseball</p>",
    unsafe_allow_html=True
)