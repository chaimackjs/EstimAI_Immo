"""Nettoyage, profils DPE, rapport qualite et cache invalide si les sources changent."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from nettoyage_dvf import BornesDVF, lire_fichier_dvf, nettoyer_dvf
from nettoyage_dpe import nettoyer_dpe
from rapprochement_dvf_dpe import appliquer_profils_dpe, construire_profils_dpe
from utilitaires import ROOT


def preparer(dossier=None, force=False, minimum_dpe=20, fenetre_jours=730, sans_dpe=False,
             dossier_rapports=None):
    dossier = Path(dossier or ROOT / 'data')
    clean, rapports = dossier / 'clean', Path(dossier_rapports or ROOT / 'reports')
    clean.mkdir(parents=True, exist_ok=True)
    rapports.mkdir(parents=True, exist_ok=True)
    fichiers_dvf = sorted([*(dossier / 'dvf').glob('*.csv'), *(dossier / 'dvf').glob('*.csv.gz')])
    fichiers_dpe = [] if sans_dpe else sorted((dossier / 'dpe').glob('*.csv'))
    if not fichiers_dvf:
        raise FileNotFoundError('Aucun CSV geo-DVF dans data/dvf. '
            'Lancer acquisition_donnees.py --departements 69 --annees 2021 2022 2023 2024 2025 '
            '(adapter le departement). Les TXT bruts ne sont pas reutilises.')
    sources = [{'fichier': str(p.resolve()), 'taille': p.stat().st_size,
                'mtime_ns': p.stat().st_mtime_ns} for p in fichiers_dvf + fichiers_dpe]
    code = b''.join((ROOT / f).read_bytes() for f in ['utilitaires.py', 'nettoyage_dvf.py',
                     'nettoyage_dpe.py', 'rapprochement_dvf_dpe.py', 'preparer_donnees.py'])
    signature = {'sources': sources, 'code_sha256': hashlib.sha256(code).hexdigest(),
                 'minimum_dpe': minimum_dpe, 'fenetre_jours': fenetre_jours, 'sans_dpe': sans_dpe}
    manifeste = clean / 'manifest.json'
    sortie = clean / 'dvf_dpe.csv'  # Corrige l'ancien nom dfv_dpe.csv.
    cache_ok = (not force and sortie.exists() and manifeste.exists()
                and json.loads(manifeste.read_text()) == signature
                and (not fichiers_dpe or (clean / 'profils_dpe.joblib').exists()))
    if cache_ok:
        print(f'Cache a jour : {sortie}')
        return sortie
    ventes, qualite = [], {'dvf': {}, 'parametres_dpe': signature | {'sources': sources}}
    for chemin in fichiers_dvf:
        print(f'Nettoyage de {chemin.name}')
        df, rapport = nettoyer_dvf(lire_fichier_dvf(chemin), retourner_rapport=True)
        ventes.append(df)
        qualite['dvf'][chemin.name] = rapport
    df_dvf = pd.concat(ventes, ignore_index=True)
    if df_dvf.empty:
        raise ValueError('Aucune mutation admissible. Consulter les bornes et la structure des sources.')
    if df_dvf['id_mutation'].duplicated().any():
        raise ValueError('Des mutations figurent dans plusieurs fichiers. '
                         'Ne pas melanger plusieurs exports du meme millesime/perimetre.')
    df_dvf.to_csv(clean / 'dvf.csv', index=False)
    if fichiers_dpe:
        brut = pd.concat([pd.read_csv(p, dtype='string') for p in fichiers_dpe], ignore_index=True)
        dpe, qualite['dpe'] = nettoyer_dpe(brut, retourner_rapport=True)
        dpe.to_csv(clean / 'dpe.csv', index=False)
        profils = construire_profils_dpe(dpe)
        joblib.dump({'profils': profils, 'minimum_dpe': minimum_dpe,
                     'fenetre_jours': fenetre_jours}, clean / 'profils_dpe.joblib')
        resultat = appliquer_profils_dpe(df_dvf, profils, fenetre_jours, minimum_dpe)
    else:
        print('Pas de DPE : preparation DVF seule. Aucun logement ne sera supprime pour cela.')
        resultat = appliquer_profils_dpe(df_dvf, None, fenetre_jours, minimum_dpe)
        for ancien in [clean / 'dpe.csv', clean / 'profils_dpe.joblib']:
            ancien.unlink(missing_ok=True)
    resultat = resultat.sort_values(['date_mutation', 'id_mutation']).reset_index(drop=True)
    couverture = resultat.groupby(['annee_mutation', 'code_departement', 'type_local'], dropna=False).agg(
        ventes=('id_mutation', 'size'),
        part_avec_profil_dpe=('dpe_profil_disponible', 'mean'),
        mediane_nombre_dpe=('dpe_nb_diagnostics', 'median')).reset_index()
    couverture.to_csv(rapports / 'couverture_dpe.csv', index=False)
    qualite['mutations_finales'] = len(resultat)
    qualite['part_avec_profil_dpe'] = float(resultat['dpe_profil_disponible'].mean())
    (rapports / 'qualite_donnees.json').write_text(json.dumps(qualite, indent=2, ensure_ascii=False), encoding='utf-8')
    resultat.to_csv(sortie, index=False)
    manifeste.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    print(f'{len(resultat):,} mutations ; profils DPE exploitables : {qualite["part_avec_profil_dpe"]:.1%}')
    return sortie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--sans-dpe', action='store_true')
    parser.add_argument('--minimum-dpe', type=int, default=20)
    parser.add_argument('--fenetre-jours', type=int, default=730)
    args = parser.parse_args()
    preparer(force=args.force, sans_dpe=args.sans_dpe,
             minimum_dpe=args.minimum_dpe, fenetre_jours=args.fenetre_jours)


if __name__ == '__main__':
    main()
