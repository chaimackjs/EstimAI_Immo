"""Vérification rapide du modèle produit par la chaîne CI/CD."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modeles import CATEGORIELLES, predire_bundle
from utilitaires import ROOT


def verifier_modele(chemin_modele, chemin_donnees, nombre_lignes=20):
    """Charger le modèle et vérifier quelques prédictions réelles."""
    bundle = joblib.load(chemin_modele)
    if bundle.get('autorise_deploiement') is not True:
        raise ValueError('Le modèle n’est pas autorisé au déploiement.')

    version = bundle.get('version_modele')
    if not version or not version.get('identifiant'):
        raise ValueError('La version traçable du modèle est absente.')

    donnees = pd.read_csv(
        chemin_donnees,
        dtype={colonne: 'string' for colonne in CATEGORIELLES + ['id_mutation']},
        nrows=nombre_lignes,
        low_memory=False,
    )
    if donnees.empty:
        raise ValueError('Aucune donnée disponible pour tester le modèle.')

    predictions = predire_bundle(bundle, donnees)
    if not np.isfinite(predictions).all() or not (predictions > 0).all():
        raise ValueError('Le modèle a produit une prédiction invalide.')

    print(
        f"Modèle {version['identifiant']} vérifié sur {len(predictions)} lignes : "
        f"prédictions de {predictions.min():.0f} à {predictions.max():.0f} EUR/m²."
    )


def main():
    """Lire les arguments et lancer la vérification du modèle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modele', type=Path, default=ROOT / 'models' / 'meilleur_modele.joblib')
    parser.add_argument('--donnees', type=Path, default=ROOT / 'data' / 'clean' / 'dvf_dpe.csv')
    parser.add_argument('--nombre-lignes', type=int, default=20)
    args = parser.parse_args()

    verifier_modele(args.modele, args.donnees, args.nombre_lignes)


if __name__ == '__main__':
    main()
