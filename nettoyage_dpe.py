import pandas as pd


COLONNES_DPE_UTILES = [
    "date_etablissement_dpe",
    "etiquette_dpe",
    "etiquette_ges",
    "annee_construction",
    "periode_construction",
    "code_insee_ban",
    "type_batiment",
    "surface_habitable_logement",
    "type_energie_principale_chauffage",
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
]


def nettoyer_dpe(df):
    df = df[COLONNES_DPE_UTILES].copy()
    df = df.rename(columns={"code_insee_ban": "code_insee"})
    df["code_insee"] = df["code_insee"].astype("string").str.strip().str.zfill(5)
    df["date_etablissement_dpe"] = pd.to_datetime(
        df["date_etablissement_dpe"],
        errors="coerce",
    )
    df["annee_dpe"] = df["date_etablissement_dpe"].dt.year

    mapping_dpe = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
    for colonne in ("etiquette_dpe", "etiquette_ges"):
        df[colonne] = (
            df[colonne].astype("string").str.strip().str.upper().map(mapping_dpe)
        )

    mapping_isolation = {
        "très bonne": 4,
        "bonne": 3,
        "moyenne": 2,
        "insuffisante": 1,
    }
    for colonne in (
        "qualite_isolation_enveloppe",
        "qualite_isolation_murs",
        "qualite_isolation_menuiseries",
    ):
        df[colonne] = (
            df[colonne].astype("string").str.strip().str.lower().map(mapping_isolation)
        )

    df = df[df["type_batiment"].isin(["maison", "appartement"])].copy()
    df["code_type_local"] = df["type_batiment"].map(
        {"maison": 1, "appartement": 2}
    )
    return df.drop(columns="type_batiment")


