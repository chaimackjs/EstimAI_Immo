"""Interface Streamlit de démonstration du meilleur modèle immobilier."""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modeles import predire_bundle


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "meilleur_modele.joblib"
NATURES_TERRAIN = {
    "Non renseignée": "",
    "Sol": "S",
    "Terrain à bâtir": "AB",
    "Terrain d'agrément": "AG",
    "Terre": "T",
    "Jardin": "J",
    "Pré": "P",
    "Verger": "VE",
    "Vigne": "VI",
}
LATITUDE_MIN, LATITUDE_MAX = 45.45, 46.31
LONGITUDE_MIN, LONGITUDE_MAX = 4.27, 5.14


@st.cache_resource
def charger_modele():
    """Charger le bundle du meilleur modèle livré avec l'application."""
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


def valeur_optionnelle(valeur, minimum=0):
    """Convertir une valeur positive ou retourner une valeur manquante."""
    return float(valeur) if valeur is not None and valeur > minimum else np.nan


def construire_donnees(date_mutation, surface, surface_carrez, terrain, pieces,
                        lots, type_local, code_insee, code_postal, latitude,
                        longitude, id_parcelle="", nature_terrain="",
                        dependance=False, adresse_numero=0,
                        adresse_code_voie="", adresse_suffixe="",
                        lot_numero=""):
    """Construire une observation avec les variables prioritaires du modèle."""
    code_insee = code_insee.strip().upper()
    code_postal = code_postal.strip()
    departement = code_insee[:3] if code_insee.startswith(("97", "98")) else code_insee[:2]
    return pd.DataFrame([{
        "date_mutation": pd.Timestamp(date_mutation),
        "surface_reelle_bati": float(surface),
        "lot1_surface_carrez": valeur_optionnelle(surface_carrez),
        "surface_terrain": valeur_optionnelle(terrain),
        "nombre_pieces_principales": int(pieces),
        "nombre_de_lots": int(lots),
        "type_local": type_local,
        "code_insee": code_insee,
        "code_postal": code_postal,
        "code_departement": departement,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "id_parcelle": id_parcelle.strip().upper() or pd.NA,
        "code_nature_culture": nature_terrain or pd.NA,
        "presence_dependance": int(dependance),
        "terrain_ambigu": 0,
        "adresse_numero": valeur_optionnelle(adresse_numero),
        "adresse_code_voie": adresse_code_voie.strip().upper() or pd.NA,
        "adresse_suffixe": adresse_suffixe.strip().upper() or pd.NA,
        "lot1_numero": lot_numero.strip() or pd.NA,
    }])


st.set_page_config(page_title="EstimAI Immo", page_icon="🏠", layout="centered")
st.title("EstimAI Immo")
st.caption("Estimation indicative d'un appartement ou d'une maison dans le Rhône.")

bundle = charger_modele()
if bundle is None:
    st.error(
        "Modèle introuvable. Le fichier `models/meilleur_modele.joblib` doit être "
        "présent dans le dépôt GitHub déployé."
    )
    st.stop()

metriques = bundle.get("metriques_test", {})
seuil_total = bundle.get("seuil_r2_prix_total", 0.70)
modele_valide = metriques.get("r2_prix_total", float("-inf")) > seuil_total
if modele_valide:
    st.success(f"Modèle validé — R² prix total : {metriques['r2_prix_total']:.3f}.")
else:
    st.warning(
        "Modèle expérimental : son R² prix total "
        f"({metriques.get('r2_prix_total', 0):.3f}) ne dépasse pas {seuil_total:.2f}."
    )

st.info(
    "Pour une estimation plus précise, renseignez surtout les surfaces, la localisation "
    "exacte et l'identifiant de parcelle. Les ratios et mailles géographiques sont "
    "calculés automatiquement."
)

with st.form("formulaire_estimation"):
    st.subheader("1. Localisation du bien")
    localisation_gauche, localisation_droite = st.columns(2)
    with localisation_gauche:
        code_insee = st.text_input(
            "Code INSEE de la commune *", value="69123", max_chars=5,
            help="Variable géographique importante. Exemple : 69123 pour Lyon.",
        )
        latitude = st.number_input(
            "Latitude *", value=45.7673, format="%.6f",
            help="Coordonnée comprise dans le périmètre d'apprentissage du Rhône.",
        )
        id_parcelle = st.text_input(
            "Identifiant de parcelle",
            help="Fortement recommandé : il permet de calculer la section cadastrale.",
        )
    with localisation_droite:
        code_postal = st.text_input(
            "Code postal *", value="69001", max_chars=5,
            help="Une des variables les plus importantes du modèle.",
        )
        longitude = st.number_input(
            "Longitude *", value=4.8343, format="%.6f",
            help="Coordonnée comprise dans le périmètre d'apprentissage du Rhône.",
        )

    st.subheader("2. Surfaces")
    surface_col, carrez_col, terrain_col = st.columns(3)
    with surface_col:
        surface = st.number_input(
            "Surface habitable (m²) *", min_value=9.0, max_value=1000.0,
            value=70.0, step=1.0,
        )
    with carrez_col:
        surface_carrez = st.number_input(
            "Surface Carrez (m²)", min_value=0.0, max_value=1000.0,
            value=70.0, step=1.0,
            help="Saisissez 0 si elle est inconnue. Le ratio Carrez/habitable est calculé automatiquement.",
        )
    with terrain_col:
        terrain = st.number_input(
            "Surface du terrain (m²)", min_value=0.0, max_value=1_000_000.0,
            value=0.0, step=10.0,
            help="Saisissez 0 si elle est inconnue ou non applicable.",
        )

    st.subheader("3. Caractéristiques du logement")
    bien_gauche, bien_droite = st.columns(2)
    with bien_gauche:
        type_local = st.selectbox("Type de bien *", ["Appartement", "Maison"])
        pieces = st.number_input("Nombre de pièces principales *", min_value=1,
                                 max_value=50, value=3, step=1)
        lots = st.number_input("Nombre de lots", min_value=0, max_value=20,
                               value=1, step=1)
    with bien_droite:
        nature_libelle = st.selectbox(
            "Nature du terrain", list(NATURES_TERRAIN),
            help="Pour une maison, « Sol » est le cas le plus fréquent dans les données DVF.",
        )
        dependance = st.checkbox("Présence d'une dépendance")
        date_mutation = st.date_input("Date de l'estimation", value=date.today())

    with st.expander("Compléments facultatifs à faible influence"):
        complement_gauche, complement_droite = st.columns(2)
        with complement_gauche:
            adresse_numero = st.number_input("Numéro de voie", min_value=0,
                                             value=0, step=1)
            adresse_code_voie = st.text_input("Code voie DVF")
        with complement_droite:
            adresse_suffixe = st.text_input("Suffixe de voie")
            lot_numero = st.text_input("Numéro du premier lot")

    soumis = st.form_submit_button("Estimer le bien", type="primary",
                                    use_container_width=True)

if soumis:
    erreurs = []
    code_insee = code_insee.strip()
    code_postal = code_postal.strip()
    if not (len(code_insee) == 5 and code_insee.isdigit() and code_insee.startswith("69")):
        erreurs.append("Le code INSEE doit contenir cinq chiffres et appartenir au Rhône (69).")
    if not (len(code_postal) == 5 and code_postal.isdigit() and code_postal.startswith("69")):
        erreurs.append("Le code postal doit contenir cinq chiffres et appartenir au Rhône (69).")
    if not (LATITUDE_MIN <= latitude <= LATITUDE_MAX and
            LONGITUDE_MIN <= longitude <= LONGITUDE_MAX):
        erreurs.append("Les coordonnées sont hors du périmètre couvert par les données d'apprentissage.")
    if surface_carrez > 0 and surface_carrez > surface * 1.5:
        erreurs.append("La surface Carrez paraît incohérente avec la surface habitable.")

    if erreurs:
        for erreur in erreurs:
            st.error(erreur)
    else:
        try:
            donnees = construire_donnees(
                date_mutation=date_mutation,
                surface=surface,
                surface_carrez=surface_carrez,
                terrain=terrain,
                pieces=pieces,
                lots=lots,
                type_local=type_local,
                code_insee=code_insee,
                code_postal=code_postal,
                latitude=latitude,
                longitude=longitude,
                id_parcelle=id_parcelle,
                nature_terrain=NATURES_TERRAIN[nature_libelle],
                dependance=dependance,
                adresse_numero=adresse_numero,
                adresse_code_voie=adresse_code_voie,
                adresse_suffixe=adresse_suffixe,
                lot_numero=lot_numero,
            )
            prix_m2 = float(predire_bundle(bundle, donnees)[0])
            prix_total = prix_m2 * float(surface)
            resultat_m2, resultat_total = st.columns(2)
            resultat_m2.metric("Prix estimé au m²", f"{prix_m2:,.0f} €".replace(",", " "))
            resultat_total.metric("Prix total estimé", f"{prix_total:,.0f} €".replace(",", " "))

            informations_manquantes = []
            if surface_carrez <= 0:
                informations_manquantes.append("surface Carrez")
            if not id_parcelle.strip():
                informations_manquantes.append("parcelle cadastrale")
            if type_local == "Maison" and terrain <= 0:
                informations_manquantes.append("surface du terrain")
            if informations_manquantes:
                st.warning(
                    "Estimation calculée, mais elle peut être moins précise sans : "
                    + ", ".join(informations_manquantes) + "."
                )
            if pd.Timestamp(date_mutation) > pd.Timestamp(bundle["fin_apprentissage"]):
                st.info(
                    "Cette date est postérieure à la fin des données d'apprentissage "
                    f"({bundle['fin_apprentissage']})."
                )
        except Exception as erreur:
            st.error(f"Estimation impossible : {erreur}")

st.caption(
    f"Modèle : {bundle.get('nom', 'inconnu')} · "
    f"MAE de test : {metriques.get('mae_eur_m2', float('nan')):,.0f} €/m²"
    .replace(",", " ")
)
