import os

import pandas as pd


COLONNES_DVF_UTILES = [
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Code type local",
    "Type local",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Nombre de lots",
    "Surface terrain",
]

COLONNES_DVF_TEXTE = {
    "Code postal": "string",
    "Commune": "string",
    "Code departement": "string",
    "Code commune": "string",
    "Nature mutation": "string",
    "Type local": "string",
}


def lire_fichier_dvf(chemin):
    return pd.read_csv(
        chemin,
        sep="|",
        usecols=COLONNES_DVF_UTILES,
        dtype=COLONNES_DVF_TEXTE,
        low_memory=False,
    )


def nettoyer_dvf(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], dayfirst=True)
    df["valeur_fonciere"] = pd.to_numeric(
        df["valeur_fonciere"].str.replace(",", "."),
        errors="coerce",
    )

    df = df[df["nature_mutation"] == "Vente"]
    df = df[df["type_local"].isin(["Maison", "Appartement"])]

    colonnes_numeriques = [
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "nombre_de_lots",
        "surface_terrain",
    ]
    for colonne in colonnes_numeriques:
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    df = df[df["surface_reelle_bati"] > 0]
    df = df[df["valeur_fonciere"].notna()]
    df["prix_m2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df["annee_mutation"] = df["date_mutation"].dt.year
    df["mois_mutation"] = df["date_mutation"].dt.month
    df["code_postal"] = df["code_postal"].str.strip().str.zfill(5)

    code_departement = df["code_departement"].str.strip()
    code_commune = df["code_commune"].str.strip()
    df["code_insee"] = code_departement + code_commune.str.zfill(3)
    outre_mer = code_departement.str.len().eq(3)
    df.loc[outre_mer, "code_insee"] = (
        code_departement[outre_mer] + code_commune[outre_mer].str.zfill(2)
    )

    return df.drop(columns="nature_mutation").reset_index(drop=True)
