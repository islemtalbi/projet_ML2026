"""
Application Web — Eco-Smart Classifier (Version Premium)
Dark Mode | Graphiques riches | UX optimisée
"""

import os
import re
import warnings

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Eco-Smart Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# DARK THEME CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base dark ── */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color .2s;
}
.card:hover { border-color: #388e3c; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d2818 100%);
    border: 1px solid #238636;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(35,134,54,.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3fb950, #56d364, #7ee787);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #8b949e;
    font-size: 1.1rem;
    margin-top: .5rem;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #238636, #3fb950);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #3fb950;
    display: block;
}
.metric-label {
    font-size: .8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* ── Prediction result ── */
.pred-box {
    background: linear-gradient(135deg, #0d2818, #1a3320);
    border: 2px solid #238636;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.pred-emoji { font-size: 4rem; margin-bottom: .5rem; }
.pred-cat {
    font-size: 2.2rem;
    font-weight: 700;
    color: #3fb950;
    margin: .2rem 0;
}
.pred-price {
    font-size: 1.3rem;
    color: #f78166;
    font-weight: 600;
}
.pred-conf { color: #8b949e; font-size: .9rem; margin-top: .3rem; }

/* ── Agreement badge ── */
.badge-ok {
    background: #1a3320;
    border: 1px solid #238636;
    color: #3fb950;
    border-radius: 8px;
    padding: .6rem 1rem;
    text-align: center;
    font-weight: 600;
}
.badge-warn {
    background: #2d1f00;
    border: 1px solid #9e6a03;
    color: #e3b341;
    border-radius: 8px;
    padding: .6rem 1rem;
    text-align: center;
}

/* ── NLP result cards ── */
.nlp-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
}
.nlp-card h4 { color: #8b949e; font-size: .85rem; text-transform: uppercase; margin: 0 0 .5rem; }
.nlp-cat { font-size: 1.6rem; font-weight: 700; color: #e6edf3; margin: .3rem 0; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #8b949e !important;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3fb950 !important;
    border-bottom: 2px solid #3fb950 !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: #238636 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all .2s;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(35,134,54,.4);
}

/* ── Sidebar items ── */
.sidebar-stat {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: .7rem 1rem;
    margin-bottom: .5rem;
    font-size: .9rem;
}
.sidebar-stat span { color: #3fb950; font-weight: 600; }

/* ── Section titles ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 1.5rem 0 1rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid #30363d;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw", "dataset_ProjetML_2026.csv")
DATA_CLEAN  = os.path.join(BASE_DIR, "data", "processed", "dataset_clean.csv")

CAT_COLORS = {
    "Métal":     "#78909c",
    "Metal":     "#78909c",
    "Papier":    "#8d6e63",
    "Plastique": "#42a5f5",
    "Verre":     "#66bb6a",
}
CAT_EMOJI = {
    "Métal": "🔩", "Metal": "🔩",
    "Papier": "📄", "Plastique": "🧴", "Verre": "🍶",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", family="Inter, sans-serif"),
    margin=dict(t=30, b=20, l=10, r=10),
)

STOP_DOMAINE = {
    "lot", "dechet", "collecte", "volume", "poids",
    "kg", "litre", "usine", "site", "materiau", "aspect",
    "organique", "aluminium",
}

# ══════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        m = {
            "classifier":   joblib.load(f"{MODELS_DIR}/classifier_best.pkl"),
            "regressor":    joblib.load(f"{MODELS_DIR}/regressor_best.pkl"),
            "scaler":       joblib.load(f"{MODELS_DIR}/scaler.pkl"),
            "le_source":    joblib.load(f"{MODELS_DIR}/le_source.pkl"),
            "le_categorie": joblib.load(f"{MODELS_DIR}/le_categorie.pkl"),
            "tfidf":        joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl"),
            "nlp_clf":      joblib.load(f"{MODELS_DIR}/nlp_classifier.pkl"),
        }
        return m, True
    except Exception:
        return {}, False


@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv(DATA_RAW), pd.read_csv(DATA_CLEAN), True
    except Exception:
        return None, None, False


def preprocess_text(texte: str) -> str:
    stop_fr = set(stopwords.words("french")).union(STOP_DOMAINE)
    stemmer = SnowballStemmer("french")
    texte   = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", texte.lower())
    tokens  = word_tokenize(re.sub(r"\s+", " ", texte).strip(), language="french")
    return " ".join(stemmer.stem(t) for t in tokens if t not in stop_fr)


def predict_numeric(models, poids, volume, conductivite, opacite, rigidite, source):
    try:
        src_enc = models["le_source"].transform([source])[0]
    except Exception:
        src_enc = 0
    raw     = np.array([[poids, volume, conductivite, opacite, rigidite]])
    scaled  = models["scaler"].transform(raw)
    feats   = np.append(scaled[0], src_enc).reshape(1, -1)
    cat     = models["classifier"].predict(feats)[0]
    prix    = float(models["regressor"].predict(feats)[0])
    conf    = None
    probas  = None
    if hasattr(models["classifier"], "predict_proba"):
        p     = models["classifier"].predict_proba(feats)[0]
        conf  = float(np.max(p))
        probas = dict(zip(models["classifier"].classes_, p))
    return cat, prix, conf, probas


# ══════════════════════════════════════════════════════════════════
# LOADING
# ══════════════════════════════════════════════════════════════════
with st.spinner("Chargement du système…"):
    models, models_ok = load_models()
    df_raw, df_clean, data_ok = load_data()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-size:3rem">♻️</div>
        <div style="font-size:1.2rem; font-weight:700; color:#3fb950;">Eco-Smart</div>
        <div style="font-size:.8rem; color:#8b949e;">Classifier v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Statut système**")
    st.markdown(
        f'<div class="sidebar-stat">🤖 Modèles ML <span>{"✓ OK" if models_ok else "✗ ERR"}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sidebar-stat">📦 Dataset <span>{"✓ OK" if data_ok else "✗ ERR"}</span></div>',
        unsafe_allow_html=True,
    )

    if data_ok:
        st.markdown("**Statistiques**")
        st.markdown(f'<div class="sidebar-stat">Observations <span>{len(df_raw):,}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat">Labellisées <span>{df_raw["Categorie"].notna().sum():,}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat">Catégories <span>4</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#8b949e; font-size:.8rem; text-align:center;">
        Projet ML 2026<br>Classification de déchets<br>
        <span style="color:#238636;">● Pipeline complet DVC + MLflow</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <p class="hero-title">♻️ Eco-Smart Classifier</p>
    <p class="hero-sub">Pipeline ML complet · Classification de déchets · Estimation de valeur · NLP</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  Dashboard Data",
    "🎛️  Prédiction Manuelle",
    "🤖  Assistant NLP",
])

# ─────────────────────────────────────────────────────────────────
# ONGLET 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────
with tab1:
    if not data_ok:
        st.error("Dataset introuvable.")
    else:
        # KPIs
        st.markdown('<div class="metric-grid">'
            f'<div class="metric-card"><span class="metric-value">{len(df_raw):,}</span><span class="metric-label">Observations</span></div>'
            f'<div class="metric-card"><span class="metric-value">{df_raw["Categorie"].notna().sum():,}</span><span class="metric-label">Labellisées</span></div>'
            f'<div class="metric-card"><span class="metric-value">{df_raw["Poids"].isnull().mean()*100:.1f}%</span><span class="metric-label">NaN Poids</span></div>'
            f'<div class="metric-card"><span class="metric-value">4</span><span class="metric-label">Catégories</span></div>'
            '</div>', unsafe_allow_html=True)

        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">Distribution des catégories</p>', unsafe_allow_html=True)
            counts = df_raw["Categorie"].value_counts().reset_index()
            counts.columns = ["Categorie", "Count"]
            fig = px.pie(counts, values="Count", names="Categorie",
                         color="Categorie", color_discrete_map=CAT_COLORS,
                         hole=0.5)
            fig.update_traces(textposition="outside", textinfo="label+percent",
                              textfont_color="#e6edf3",
                              marker=dict(line=dict(color="#0d1117", width=2)))
            fig.update_layout(**PLOT_LAYOUT, height=340,
                              legend=dict(font=dict(color="#8b949e")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<p class="section-title">Répartition par source</p>', unsafe_allow_html=True)
            src = df_raw["Source"].value_counts().reset_index()
            src.columns = ["Source", "Count"]
            fig2 = px.bar(src, x="Count", y="Source", orientation="h",
                          color="Count", color_continuous_scale=["#0d2818", "#3fb950"])
            fig2.update_layout(**PLOT_LAYOUT, height=340,
                               coloraxis_showscale=False,
                               yaxis=dict(gridcolor="#21262d"),
                               xaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig2, use_container_width=True)

        # Row 2 — feature distribution
        st.markdown('<p class="section-title">Distribution des features par catégorie</p>', unsafe_allow_html=True)
        feat = st.selectbox("Feature", ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"],
                            key="feat_sel")
        df_plot = df_raw[df_raw["Categorie"].notna()]
        fig3 = px.violin(df_plot, x="Categorie", y=feat, color="Categorie",
                         color_discrete_map=CAT_COLORS, box=True, points="outliers")
        fig3.update_layout(**PLOT_LAYOUT, height=380, showlegend=False,
                           xaxis=dict(gridcolor="#21262d"),
                           yaxis=dict(gridcolor="#21262d"))
        st.plotly_chart(fig3, use_container_width=True)

        # Row 3 — heatmap corrélation
        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<p class="section-title">Corrélations</p>', unsafe_allow_html=True)
            num_cols = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Prix_Revente"]
            corr = df_raw[num_cols].corr().round(2)
            fig4 = px.imshow(corr, color_continuous_scale="RdYlGn",
                             zmin=-1, zmax=1, text_auto=True)
            fig4.update_layout(**PLOT_LAYOUT, height=350)
            st.plotly_chart(fig4, use_container_width=True)

        with c4:
            st.markdown('<p class="section-title">Clusters PCA 2D</p>', unsafe_allow_html=True)
            df_ml = df_clean[df_clean["Categorie"].notna()].copy()
            fp = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]
            if all(f in df_ml.columns for f in fp):
                X = df_ml[fp].dropna()
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X)
                v1 = round(pca.explained_variance_ratio_[0] * 100, 1)
                v2 = round(pca.explained_variance_ratio_[1] * 100, 1)
                df_p = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                                     "Categorie": df_ml["Categorie"].values[:len(coords)]})
                fig5 = px.scatter(df_p, x="PC1", y="PC2", color="Categorie",
                                  color_discrete_map=CAT_COLORS, opacity=0.65,
                                  labels={"PC1": f"PC1 ({v1}%)", "PC2": f"PC2 ({v2}%)"})
                fig5.update_traces(marker=dict(size=4))
                fig5.update_layout(**PLOT_LAYOUT, height=350,
                                   xaxis=dict(gridcolor="#21262d"),
                                   yaxis=dict(gridcolor="#21262d"),
                                   legend=dict(font=dict(color="#8b949e")))
                st.plotly_chart(fig5, use_container_width=True)

        # Aperçu
        st.markdown('<p class="section-title">Aperçu du dataset</p>', unsafe_allow_html=True)
        n = st.slider("Nombre de lignes", 5, 50, 10)
        st.dataframe(df_raw.head(n), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────
# ONGLET 2 — PREDICTION MANUELLE
# ─────────────────────────────────────────────────────────────────
with tab2:
    if not models_ok:
        st.error("Modèles introuvables. Lancez `dvc repro`.")
    else:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<p class="section-title">⚙️ Caractéristiques du déchet</p>', unsafe_allow_html=True)

            poids   = st.slider("⚖️ Poids (kg)",       1.0, 500.0, 65.0, 0.5)
            volume  = st.slider("📦 Volume (L)",        1.0, 600.0, 120.0, 1.0)
            cond    = st.slider("⚡ Conductivité",      0.0, 1.0,   0.5,  0.01)
            opac    = st.slider("🔍 Opacité",           0.0, 1.0,   0.5,  0.01)
            rigid   = st.slider("💪 Rigidité",          1.0, 10.0,  5.0,  0.5)
            sources = list(models["le_source"].classes_)
            source  = st.selectbox("🏭 Source", sources)

            # Radar chart
            st.markdown('<p class="section-title">Profil du déchet</p>', unsafe_allow_html=True)
            radar_vals = [poids / 500, volume / 600, cond, opac, rigid / 10]
            radar_cats = ["Poids", "Volume", "Conductivité", "Opacité", "Rigidité"]
            fig_r = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself",
                fillcolor="rgba(35,134,54,.25)",
                line=dict(color="#3fb950", width=2),
            ))
            fig_r.update_layout(
                **PLOT_LAYOUT, height=280,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor="#30363d", tickfont=dict(color="#8b949e")),
                    angularaxis=dict(gridcolor="#30363d",
                                    tickfont=dict(color="#8b949e")),
                ),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with right:
            cat, prix, conf, probas = predict_numeric(
                models, poids, volume, cond, opac, rigid, source
            )
            emoji = CAT_EMOJI.get(cat, "♻️")
            conf_pct = f"{conf * 100:.1f}%" if conf else "—"

            st.markdown('<p class="section-title">🎯 Résultat</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-emoji">{emoji}</div>
                <div class="pred-cat">{cat}</div>
                <div class="pred-price">Prix estimé : {prix:.2f} €</div>
                <div class="pred-conf">Confidence : {conf_pct}</div>
            </div>
            """, unsafe_allow_html=True)

            # Jauge
            if conf:
                st.markdown("")
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=conf * 100,
                    delta={"reference": 70, "valueformat": ".1f",
                           "increasing": {"color": "#3fb950"},
                           "decreasing": {"color": "#f78166"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                                 "tickfont": {"color": "#8b949e"}},
                        "bar": {"color": "#238636"},
                        "bgcolor": "#161b22",
                        "bordercolor": "#30363d",
                        "steps": [
                            {"range": [0, 50],  "color": "#1f0d0d"},
                            {"range": [50, 75], "color": "#1f1a0d"},
                            {"range": [75, 100], "color": "#0d1f10"},
                        ],
                        "threshold": {"line": {"color": "#3fb950", "width": 3},
                                      "thickness": .85, "value": 90},
                    },
                    number={"suffix": "%", "font": {"color": "#3fb950", "size": 28}},
                ))
                fig_g.update_layout(**PLOT_LAYOUT, height=220)
                st.plotly_chart(fig_g, use_container_width=True)

            # Barres probabilités
            if probas:
                st.markdown('<p class="section-title">Probabilités par catégorie</p>', unsafe_allow_html=True)
                df_p = pd.DataFrame(list(probas.items()), columns=["Catégorie", "Prob"])
                df_p["Prob%"] = df_p["Prob"] * 100
                df_p = df_p.sort_values("Prob%", ascending=True)
                fig_b = px.bar(df_p, x="Prob%", y="Catégorie", orientation="h",
                               color="Catégorie", color_discrete_map=CAT_COLORS,
                               text=df_p["Prob%"].map(lambda x: f"{x:.1f}%"))
                fig_b.update_traces(textposition="outside",
                                    textfont=dict(color="#e6edf3"))
                fig_b.update_layout(**PLOT_LAYOUT, height=220, showlegend=False,
                                    xaxis=dict(gridcolor="#21262d", range=[0, 110]),
                                    yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_b, use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# ONGLET 3 — ASSISTANT NLP
# ─────────────────────────────────────────────────────────────────
with tab3:
    if not models_ok:
        st.error("Modèles introuvables.")
    else:
        EXEMPLES = {
            "-- Choisir un exemple --": ("", 65.0, 120.0, 0.5, 0.5, 5.0),
            "🔩 Métal":     ("Déchet métallique conducteur rigide haute conductivité forte masse opaque collecté à l'usine", 65.0, 120.0, 0.90, 0.5, 8.0),
            "📄 Papier":    ("Feuilles et cartons légers collectés provenant de l'usine peu conducteur souple", 15.0, 33.0, 0.0, 0.8, 2.0),
            "🧴 Plastique": ("Déchet plastique semi-rigide collecté en zone urbaine aspect translucide", 30.0, 58.0, 0.0, 0.5, 4.0),
            "🍶 Verre":     ("Déchets en verre transparents fragiles collectés lors d'une collecte citoyenne", 200.0, 360.0, 0.0, 0.1, 9.0),
        }

        col_in, col_out = st.columns([1, 1], gap="large")

        with col_in:
            st.markdown('<p class="section-title">✍️ Description du déchet</p>', unsafe_allow_html=True)
            ex_key = st.selectbox("💡 Exemples prédéfinis", list(EXEMPLES.keys()))
            ex_txt, ex_p, ex_v, ex_c, ex_o, ex_r = EXEMPLES[ex_key]

            rapport = st.text_area("Description", value=ex_txt, height=130,
                                   placeholder="Ex : Déchet métallique conducteur très rigide collecté en zone industrielle…")

            with st.expander("⚙️ Paramètres numériques", expanded=(ex_key != "-- Choisir un exemple --")):
                p2 = st.number_input("Poids (kg)",      value=ex_p,  min_value=1.0, max_value=500.0)
                v2 = st.number_input("Volume (L)",      value=ex_v,  min_value=1.0, max_value=600.0)
                c2 = st.number_input("Conductivité",    value=ex_c,  min_value=0.0, max_value=1.0,  step=0.01)
                o2 = st.number_input("Opacité",         value=ex_o,  min_value=0.0, max_value=1.0,  step=0.01)
                r2 = st.number_input("Rigidité",        value=ex_r,  min_value=1.0, max_value=10.0, step=0.5)
                src_list = list(models["le_source"].classes_)
                s2 = st.selectbox("Source", src_list, key="s2")

            go_btn = st.button("🔍 Analyser le déchet", type="primary", use_container_width=True)

        with col_out:
            st.markdown('<p class="section-title">📋 Résultats de l\'analyse</p>', unsafe_allow_html=True)

            if go_btn and rapport.strip():
                with st.spinner("Pipeline NLP en cours…"):
                    # NLP
                    clean_txt  = preprocess_text(rapport)
                    vec        = models["tfidf"].transform([clean_txt])
                    nlp_pred   = models["nlp_clf"].predict(vec)[0]
                    # Numérique
                    num_pred, prix2, conf2, probas2 = predict_numeric(
                        models, p2, v2, c2, o2, r2, s2
                    )

                # Résultats côte à côte
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"""
                    <div class="nlp-card">
                        <h4>Modèle NLP</h4>
                        <div style="font-size:2.5rem">{CAT_EMOJI.get(nlp_pred,"♻️")}</div>
                        <div class="nlp-cat">{nlp_pred}</div>
                    </div>""", unsafe_allow_html=True)
                with cb:
                    st.markdown(f"""
                    <div class="nlp-card">
                        <h4>Modèle Numérique</h4>
                        <div style="font-size:2.5rem">{CAT_EMOJI.get(num_pred,"♻️")}</div>
                        <div class="nlp-cat">{num_pred}</div>
                        <div style="color:#8b949e;font-size:.85rem">Prix : {prix2:.2f} €</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")
                if nlp_pred == num_pred:
                    st.markdown(f'<div class="badge-ok">✅ Accord — Les deux modèles confirment : <b>{nlp_pred}</b></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-warn">⚠️ Désaccord — NLP : <b>{nlp_pred}</b> | Numérique : <b>{num_pred}</b><br><small>Le modèle numérique est généralement plus fiable.</small></div>',
                                unsafe_allow_html=True)

                # Tokens NLP
                with st.expander("🔬 Analyse NLP détaillée"):
                    st.markdown("**Texte après prétraitement :**")
                    tokens = clean_txt.split() if clean_txt else []
                    if tokens:
                        st.markdown(" ".join(f'`{t}`' for t in tokens))
                    else:
                        st.warning("Texte vide après suppression des stopwords.")
                    st.markdown(f"**Nombre de tokens :** {len(tokens)}")

                # Probas numérique
                if probas2:
                    st.markdown('<p class="section-title">Confiance numérique</p>', unsafe_allow_html=True)
                    df_pp = pd.DataFrame(list(probas2.items()), columns=["Cat", "P"])
                    df_pp["P%"] = df_pp["P"] * 100
                    df_pp = df_pp.sort_values("P%", ascending=True)
                    fig_pb = px.bar(df_pp, x="P%", y="Cat", orientation="h",
                                   color="Cat", color_discrete_map=CAT_COLORS,
                                   text=df_pp["P%"].map(lambda x: f"{x:.1f}%"))
                    fig_pb.update_traces(textposition="outside",
                                        textfont=dict(color="#e6edf3"))
                    fig_pb.update_layout(**PLOT_LAYOUT, height=200, showlegend=False,
                                        xaxis=dict(gridcolor="#21262d", range=[0, 115]),
                                        yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_pb, use_container_width=True)

            elif go_btn:
                st.warning("Veuillez entrer une description.")
            else:
                st.markdown("""
                <div class="card" style="margin-top:1rem;">
                    <div style="font-size:1.5rem; margin-bottom:.8rem;">💡 Guide d'utilisation</div>
                    <table style="width:100%; border-collapse:collapse; color:#e6edf3;">
                        <tr style="border-bottom:1px solid #30363d;">
                            <th style="padding:.5rem; text-align:left; color:#8b949e;">Catégorie</th>
                            <th style="padding:.5rem; text-align:left; color:#8b949e;">Mots clés efficaces</th>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:.5rem;">🔩 Métal</td>
                            <td style="padding:.5rem; color:#8b949e;">métallique · conducteur · conductivité · rigide · ferraille · masse</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:.5rem;">📄 Papier</td>
                            <td style="padding:.5rem; color:#8b949e;">carton · feuilles · léger · souple · peu conducteur</td>
                        </tr>
                        <tr style="border-bottom:1px solid #21262d;">
                            <td style="padding:.5rem;">🧴 Plastique</td>
                            <td style="padding:.5rem; color:#8b949e;">plastique · semi-rigide · translucide · collecté</td>
                        </tr>
                        <tr>
                            <td style="padding:.5rem;">🍶 Verre</td>
                            <td style="padding:.5rem; color:#8b949e;">verre · transparent · fragile · citoyenne</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
