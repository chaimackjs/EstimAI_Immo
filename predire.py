"""Estimer a partir d'un JSON de caracteristiques, sans prix connu ni autre cible."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from modeles import predire_bundle
from rapprochement_dvf_dpe import appliquer_profils_dpe
from utilitaires import ROOT, dates, normaliser_code, numerique


def predire_biens(biens, chemin_modele=None):
    chemin_modele = Path(chemin_modele or ROOT / 'models' / 'meilleur_modele.joblib')
    # Ne charger que des fichiers joblib produits par vous : pickle execute du code.
    bundle = joblib.load(chemin_modele)
    df = pd.DataFrame([biens] if isinstance(biens, dict) else biens)
    requis = {'date_mutation', 'code_insee', 'type_local', 'surface_reelle_bati'}
    if manquants := requis - set(df):
        raise ValueError(f'Caracteristiques manquantes : {sorted(manquants)}')
    df['date_mutation'] = dates(df['date_mutation'])
    df['surface_reelle_bati'] = numerique(df['surface_reelle_bati'])
    df['code_insee'] = normaliser_code(df['code_insee'])
    df['type_local'] = df['type_local'].astype('string').str.strip().str.capitalize()
    if (df['date_mutation'].isna().any() or not df['surface_reelle_bati'].gt(0).all()
            or not df['type_local'].isin(['Maison', 'Appartement']).all()):
        raise ValueError('Date, surface ou type de logement invalide.')
    if 'code_departement' not in df:
        outre_mer = df['code_insee'].str.startswith(('97', '98'), na=False)
        df['code_departement'] = df['code_insee'].str[:2].where(~outre_mer, df['code_insee'].str[:3])
    df['code_type_local'] = df['type_local'].map({'Maison': 1, 'Appartement': 2})
    if bundle['profils_dpe']:
        fichier = chemin_modele.parent / bundle['profils_dpe']
        if not fichier.exists():
            raise FileNotFoundError(f'Profil DPE manquant : {fichier}. Conserver ce fichier avec le modele.')
        profils = joblib.load(fichier)
        df = appliquer_profils_dpe(df, profils['profils'], profils['fenetre_jours'], profils['minimum_dpe'])
    prediction = predire_bundle(bundle, df)
    prix_m2 = prediction if bundle['cible'] == 'prix_m2' else prediction / df['surface_reelle_bati'].to_numpy()
    resultat = df[['date_mutation', 'code_insee', 'type_local', 'surface_reelle_bati']].copy()
    resultat['prix_m2_estime'] = prix_m2
    resultat['prix_total_estime'] = prix_m2 * df['surface_reelle_bati'].to_numpy()
    resultat['apres_fin_apprentissage'] = df['date_mutation'] > pd.Timestamp(bundle['fin_apprentissage'])
    if 'dpe_profil_disponible' in df:
        resultat['profil_dpe_disponible'] = df['dpe_profil_disponible'].astype(bool)
    return resultat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fichier_json', type=Path)
    parser.add_argument('--modele', type=Path, default=ROOT / 'models' / 'meilleur_modele.joblib')
    args = parser.parse_args()
    biens = json.loads(args.fichier_json.read_text(encoding='utf-8'))
    print(predire_biens(biens, args.modele).to_string(index=False))
    print('\nEstimation ponctuelle, sans intervalle de confiance calibre. '
          'Ne pas confondre le profil DPE communal avec le DPE du bien.')


if __name__ == '__main__':
    main()
