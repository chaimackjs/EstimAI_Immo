from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "models" / "HistGradientBoostingRegressor.joblib"
MODEL_FEATURES = [
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "nombre_de_lots",
    "annee_mutation",
    "mois_mutation",
    "code_departement",
    "type_local",
]


@st.cache_resource
def charger_modele():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def construire_donnees(
    surface: float,
    pieces: int,
    terrain: float,
    lots: int,
    annee: int,
    mois: int,
    departement: str,
    type_local: str,
) -> pd.DataFrame:
    donnees = pd.DataFrame(
        [
            {
                "surface_reelle_bati": surface,
                "nombre_pieces_principales": pieces,
                "surface_terrain": terrain,
                "nombre_de_lots": lots,
                "annee_mutation": annee,
                "mois_mutation": mois,
                "code_departement": departement,
                "type_local": type_local,
            }
        ],
        columns=MODEL_FEATURES,
    )
    for colonne in ("code_departement", "type_local"):
        donnees[colonne] = donnees[colonne].astype("category")
    return donnees


st.set_page_config(
    page_title="EstimAI Immo",
    page_icon="🏠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 900px; padding-top: 3rem; }
    .hero { padding: 1.5rem 0 1rem; }
    .hero h1 { color: #17324d; font-size: 2.5rem; margin-bottom: .35rem; }
    .hero p { color: #5d6b78; font-size: 1.05rem; }
    [data-testid="stMetricValue"] { color: #0f766e; }
    </style>
    <div class="hero">
      <h1>EstimAI Immo</h1>
      <p>Une estimation du prix au m² à partir des caractéristiques du bien.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

modele = charger_modele()
if modele is None:
    st.warning(
        "Le modèle n'est pas encore disponible. Lancez `python ml.py` après avoir "
        "préparé `data/clean/dfv_dpe.csv`, puis rechargez cette page."
    )

with st.form("formulaire_estimation"):
    st.subheader("Caractéristiques du bien")
    colonne_gauche, colonne_droite = st.columns(2)
    with colonne_gauche:
        surface = st.number_input("Surface habitable (m²)", min_value=1.0, value=70.0, step=1.0)
        pieces = st.number_input("Nombre de pièces", min_value=1, value=3, step=1)
        terrain = st.number_input("Surface du terrain (m²)", min_value=0.0, value=0.0, step=10.0)
        lots = st.number_input("Nombre de lots", min_value=0, value=1, step=1)
    with colonne_droite:
        annee = st.number_input("Année de mutation", min_value=2021, max_value=2030, value=2025, step=1)
        mois = st.selectbox("Mois de mutation", range(1, 13), index=8, format_func=lambda valeur: f"{valeur:02d}")
        departement = st.text_input("Code département", value="75", max_chars=3)
        type_local = st.selectbox("Type de bien", ["Appartement", "Maison"])

    soumis = st.form_submit_button("Estimer le prix au m²", type="primary", use_container_width=True)

if soumis:
    if not departement.strip().isdigit():
        st.error("Le code département doit contenir uniquement des chiffres.")
    elif modele is None:
        st.info("Aucune estimation possible tant que le modèle n'a pas été généré.")
    else:
        donnees = construire_donnees(
            surface,
            pieces,
            terrain,
            lots,
            annee,
            mois,
            departement.strip(),
            type_local,
        )
        prediction = float(modele.predict(donnees)[0])
        st.metric("Estimation", f"{prediction:,.0f} € / m²".replace(",", " "))
        st.caption("Cette estimation est indicative et dépend de la qualité des données d'entraînement.")