import os

import matplotlib.pyplot as plt
import pandas as pd

from acquisition_donnees import recuperer_dpe
from nettoyage_dpe import nettoyer_dpe
from rapprochement_dvf_dpe import rapprocher_dvf_dpe
from nettoyage_dvf import lire_fichier_dvf, nettoyer_dvf


UTILISER_DONNEES_NETTOYEES = True


def afficher_correlation(df):
    correlation = df.corr(numeric_only=True)
    plt.imshow(correlation)
    plt.grid(False)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.colorbar()


def enregistrer_clean(df, nom_fichier):
    dossier = os.path.join("data", "clean")
    os.makedirs(dossier, exist_ok=True)
    df.to_csv(os.path.join(dossier, nom_fichier), index=False)


def recup_donnees_depuis_sources():
    df_dvf = pd.concat(
        [
            lire_fichier_dvf(
                os.path.join("data", "dvf", f"ValeursFoncieres-{annee}.txt")
            )
            for annee in range(2021, 2026)
        ],
        ignore_index=True,
    )
    df_dvf = nettoyer_dvf(df_dvf)
    df_dpe = nettoyer_dpe(recuperer_dpe())
    df_model = rapprocher_dvf_dpe(df_dvf, df_dpe)

    enregistrer_clean(df_dpe, "dpe.csv")
    enregistrer_clean(df_dvf, "dvf.csv")
    enregistrer_clean(df_model, "dfv_dpe.csv")
    return df_dpe, df_dvf, df_model


if __name__ == "__main__":
    fichiers_clean = [
        os.path.join("data", "clean", nom)
        for nom in ("dpe.csv", "dvf.csv", "dfv_dpe.csv")
    ]
    use_clean = UTILISER_DONNEES_NETTOYEES and all(
        os.path.isfile(chemin) for chemin in fichiers_clean
    )

    if use_clean:
        df_dpe = pd.read_csv(fichiers_clean[0])
        df_dvf = pd.read_csv(fichiers_clean[1])
        df_model = pd.read_csv(fichiers_clean[2])
    else:
        print("Récupération et nettoyage des données...")
        df_dpe, df_dvf, df_model = recup_donnees_depuis_sources()

    afficher_correlation(df_dpe)
    afficher_correlation(df_dvf)
    afficher_correlation(df_model)