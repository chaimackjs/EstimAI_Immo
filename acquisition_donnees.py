"""Collecte ciblee et exhaustive par departement/annee, avec cache brut."""
import argparse
from datetime import date
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nettoyage_dpe import COLONNES_DPE_UTILES
from utilitaires import ROOT

DVF_BASE_URL = 'https://files.data.gouv.fr/geo-dvf/latest/csv'
DPE_DATASET_URL = 'https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant'
DPE_API_URL = DPE_DATASET_URL + '/lines'
ANNEE_DEBUT = 2021
TOUS_DEPARTEMENTS = [
    *(f'{code:02d}' for code in range(1, 20)),
    '2A', '2B',
    *(f'{code:02d}' for code in range(21, 96)),
    '971', '972', '973', '974', '976',
]


def annees_disponibles(debut=ANNEE_DEBUT, fin=None):
    """Retourner les années complètes comprises dans la période demandée."""
    fin = date.today().year - 1 if fin is None else fin
    if debut < ANNEE_DEBUT or fin < debut or fin >= date.today().year:
        raise ValueError(f'Periode invalide : choisir des annees entre {ANNEE_DEBUT} et {date.today().year - 1}.')
    return range(debut, fin + 1)


def lire_departements(valeurs):
    """Accepter ``69 01`` ou ``69,01`` et refuser tout perimetre implicite."""
    departements = [dep.strip().upper().zfill(2) for valeur in valeurs for dep in valeur.split(',') if dep.strip()]
    inconnus = sorted(set(departements) - set(TOUS_DEPARTEMENTS))
    if not departements or inconnus:
        raise ValueError(f'Departement(s) invalide(s) : {", ".join(inconnus) or "aucun"}.')
    return list(dict.fromkeys(departements))


def session_http():
    """Créer une session HTTP avec reprise automatique sur erreur."""
    session = requests.Session()
    session.headers['User-Agent'] = 'EstimAI-Immo/2.0 (educational-data-analysis)'
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


def telecharger_dvf(departements, annees, dossier=None, force=False):
    """Télécharger les fichiers DVF demandés dans le cache local."""
    dossier = Path(dossier or ROOT / 'data' / 'dvf')
    dossier.mkdir(parents=True, exist_ok=True)
    with session_http() as session:
        for annee in annees:
            for dep in departements:
                dep = str(dep).upper().zfill(2)
                chemin = dossier / f'{annee}-{dep}.csv.gz'
                if chemin.exists() and not force:
                    print(f'Cache brut : {chemin.name}')
                    continue
                url = f'{DVF_BASE_URL}/{annee}/departements/{dep}.csv.gz'
                temporaire = chemin.with_suffix(chemin.suffix + '.part')
                print(f'Telechargement DVF : {annee}, departement {dep}')
                with session.get(url, stream=True, timeout=(20, 180)) as reponse:
                    reponse.raise_for_status()
                    with temporaire.open('wb') as flux:
                        for morceau in reponse.iter_content(chunk_size=1024 * 1024):
                            flux.write(morceau)
                temporaire.replace(chemin)


def recuperer_dpe(departement, annee, nb_ligne=None, taille_page=1000, session=None):
    """Sans limite par defaut. Une limite explicite refuse un resultat tronque."""
    if nb_ligne is not None and nb_ligne < 1:
        raise ValueError('nb_ligne doit etre positif ou None.')
    propre_session = session is None
    session = session or session_http()
    try:
        meta = session.get(DPE_DATASET_URL, timeout=60)
        meta.raise_for_status()
        presentes = {c['key'] for c in meta.json()['schema']}
        requis = {'numero_dpe', 'code_departement_ban', 'date_etablissement_dpe'}
        if not requis.issubset(presentes):
            raise ValueError('Le schema ADEME a change : verifier les champs de filtre et identifiant.')
        champs = [c for c in COLONNES_DPE_UTILES if c in presentes]
        dep = str(departement).upper().zfill(2)
        params = {'size': min(taille_page, nb_ligne) if nb_ligne else taille_page,
                  'select': ','.join(champs),
                  'qs': f'code_departement_ban:"{dep}" AND date_etablissement_dpe:[{annee}-01-01 TO {annee}-12-31]'}
        lignes, suivante, vues = [], DPE_API_URL, set()
        total_attendu = None
        while suivante:
            if suivante in vues:
                raise RuntimeError('Curseur ADEME repete : collecte interrompue sans publier un faux cache complet.')
            vues.add(suivante)
            reponse = session.get(suivante, params=params, timeout=90)
            reponse.raise_for_status()
            data = reponse.json()
            total = data.get('total')
            if total_attendu is None and total is not None:
                total_attendu = total
            if nb_ligne and total is not None and total > nb_ligne:
                raise ValueError(f'{total} DPE disponibles, limite {nb_ligne}. '
                                 'Utiliser nb_ligne=None pour ne pas biaiser le profil local.')
            resultats = data.get('results', [])
            lignes.extend(resultats)
            suivante = urljoin(DPE_API_URL, data['next']) if data.get('next') else None
            params = None  # Le curseur contient deja les filtres de la requete initiale.
            if suivante and (not resultats or (nb_ligne and len(lignes) >= nb_ligne)):
                raise RuntimeError('Collecte DPE incomplete : aucun cache complet ne sera cree.')
        if total_attendu is not None and len(lignes) != total_attendu:
            raise RuntimeError(f'Collecte incomplete ou source modifiee : {len(lignes)} lignes '
                               f'recues pour {total_attendu} annoncees. Relancer la collecte.')
        if nb_ligne is not None and len(lignes) > nb_ligne:
            raise RuntimeError('Limite explicite depassee ; aucun cache complet ne sera publie.')
        if not lignes:
            return pd.DataFrame(columns=champs)
        return pd.DataFrame(lignes).reindex(columns=champs)
    finally:
        if propre_session:
            session.close()


def main():
    """Exécuter la collecte depuis la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--avec-dpe', action='store_true', help='Collecter aussi les DPE du meme perimetre.')
    parser.add_argument('--force', action='store_true', help='Actualiser les fichiers deja presents.')
    parser.add_argument('--departements', nargs='+', required=True,
                        help='Departements a collecter, par exemple : 69 ou 69,01.')
    parser.add_argument('--annee-debut', type=int, default=ANNEE_DEBUT)
    parser.add_argument('--annee-fin', type=int, default=date.today().year - 1)
    parser.add_argument('--taille-page', type=int, default=10_000,
                        help='Lignes DPE par requete API ; 10 000 limite la duree sans tronquer.')
    args = parser.parse_args()
    departements = lire_departements(args.departements)
    annees = annees_disponibles(args.annee_debut, args.annee_fin)
    telecharger_dvf(departements, annees, force=args.force)
    if args.avec_dpe:
        dossier = ROOT / 'data' / 'dpe'
        dossier.mkdir(parents=True, exist_ok=True)
        with session_http() as session:
            for dep in departements:
                for annee in annees:
                    chemin = dossier / f'{annee}-{str(dep).upper().zfill(2)}.csv'
                    if chemin.exists() and not args.force:
                        continue
                    print(f'Collecte DPE complete : departement {dep}, annee {annee}')
                    df = recuperer_dpe(dep, annee, taille_page=args.taille_page, session=session)
                    temporaire = chemin.with_suffix('.csv.part')
                    df.to_csv(temporaire, index=False)
                    temporaire.replace(chemin)
                    print(f'{len(df):,} diagnostics enregistres')


if __name__ == '__main__':
    main()
