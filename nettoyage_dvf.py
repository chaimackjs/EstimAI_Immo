"""Une observation = une mutation residentielle simple, jamais une ligne brute."""
from dataclasses import asdict, dataclass
import pandas as pd
from utilitaires import dates, normaliser_code, normaliser_colonnes, numerique


@dataclass(frozen=True)
class BornesDVF:
    # Perimetre initial de travail, pas des verites universelles.
    surface_min: float = 9
    surface_max: float = 1000
    prix_min: float = 1000
    prix_m2_min: float = 100
    prix_m2_max: float = 50000


REQUIS = ['id_mutation', 'date_mutation', 'nature_mutation', 'numero_disposition',
          'valeur_fonciere', 'code_commune', 'code_departement', 'id_parcelle',
          'type_local', 'surface_reelle_bati']
NUMERIQUES = ['valeur_fonciere', 'surface_reelle_bati', 'surface_terrain',
              'nombre_pieces_principales', 'nombre_de_lots', 'latitude', 'longitude']


def lire_fichier_dvf(chemin):
    if str(chemin).lower().endswith('.txt'):
        raise ValueError(
            'Le TXT DGFiP ne fournit pas id_mutation. Il ne permet pas ici de '
            'securiser le regroupement. Utiliser les CSV geolocalises Etalab : '
            'python acquisition_donnees.py --avec-dpe. Les anciens TXT peuvent rester sur disque.')
    return pd.read_csv(chemin, dtype='string', low_memory=False)


def nettoyer_dvf(df, bornes=None, retourner_rapport=False):
    bornes = bornes or BornesDVF()
    df = normaliser_colonnes(df).rename(columns={'nombre_lots': 'nombre_de_lots'})
    absentes = sorted(set(REQUIS) - set(df.columns))
    if absentes:
        raise ValueError(f'Colonnes DVF absentes : {absentes}. Utiliser le format geo-DVF Etalab.')
    rapport = {'lignes_brutes': len(df), 'bornes': asdict(bornes)}
    df['id_mutation'] = df['id_mutation'].astype('string').str.strip().replace('', pd.NA)
    rapport['lignes_sans_id_mutation'] = int(df['id_mutation'].isna().sum())
    df = df.dropna(subset=['id_mutation']).copy()
    df['date_mutation'] = dates(df['date_mutation'])
    df['code_insee'] = normaliser_code(df['code_commune'])
    df['code_departement'] = normaliser_code(df['code_departement'], 2)
    df['code_postal'] = normaliser_code(df.get('code_postal', pd.Series(pd.NA, index=df.index)))
    df['type_local'] = df['type_local'].astype('string').str.strip().str.capitalize()
    df['id_parcelle'] = df['id_parcelle'].astype('string').str.strip().replace('', pd.NA)
    for colonne in NUMERIQUES:
        df[colonne] = numerique(df.get(colonne, pd.Series(float('nan'), index=df.index)))
    df['_logement'] = df['type_local'].isin(['Maison', 'Appartement']).astype(int)
    df['_autre_bati'] = (~df['type_local'].isin(['Maison', 'Appartement', 'D\u00e9pendance'])
                        & df['type_local'].notna()).astype(int)
    df['_dependance'] = df['type_local'].eq('D\u00e9pendance').fillna(False).astype(int)
    df['_vente'] = df['nature_mutation'].astype('string').str.strip().eq('Vente').fillna(False)
    df['_identifiants_ok'] = (df['id_parcelle'].notna() & df['numero_disposition'].notna()
                              & df['date_mutation'].notna())
    groupes = df.groupby('id_mutation', sort=False)
    stats = groupes.agg(
        logements=('_logement', 'sum'), autres_batis=('_autre_bati', 'sum'),
        dependances=('_dependance', 'sum'), ventes=('_vente', 'all'),
        identifiants_ok=('_identifiants_ok', 'all'),
        nb_parcelles=('id_parcelle', 'nunique'), nb_dispositions=('numero_disposition', 'nunique'),
        nb_prix=('valeur_fonciere', 'nunique'), nb_dates=('date_mutation', 'nunique'),
        prix=('valeur_fonciere', 'max'), nb_surfaces_terrain=('surface_terrain', 'nunique'),
        terrain=('surface_terrain', 'max'))
    # Ne pas dedoublonner les lignes logement : deux appartements identiques
    # peuvent etre deux biens distincts. Sans identifiant de local, on rejette.
    admissibles = (stats['logements'].eq(1) & stats['autres_batis'].eq(0)
                   & stats['ventes'] & stats['identifiants_ok']
                   & stats['nb_parcelles'].eq(1) & stats['nb_dispositions'].eq(1)
                   & stats['nb_prix'].eq(1) & stats['nb_dates'].eq(1))
    rapport['mutations_initiales'] = len(stats)
    rapport['mutations_rejetees_structure'] = int((~admissibles).sum())
    df = df[df['id_mutation'].isin(stats.index[admissibles]) & df['_logement'].eq(1)].copy()
    df['valeur_fonciere'] = df['id_mutation'].map(stats['prix'])
    df['presence_dependance'] = df['id_mutation'].map(stats['dependances']).gt(0).astype(int)
    df['terrain_ambigu'] = df['id_mutation'].map(stats['nb_surfaces_terrain']).gt(1).astype(int)
    # Ne pas sommer aveuglement les surfaces repetees entre locaux/cultures.
    df['surface_terrain'] = df['id_mutation'].map(stats['terrain']).where(df['terrain_ambigu'].eq(0))
    df['prix_m2'] = df['valeur_fonciere'] / df['surface_reelle_bati']
    plausibles = (df['surface_reelle_bati'].between(bornes.surface_min, bornes.surface_max)
                  & df['valeur_fonciere'].ge(bornes.prix_min)
                  & df['prix_m2'].between(bornes.prix_m2_min, bornes.prix_m2_max)
                  & df['code_insee'].str.fullmatch(r'(?:[0-9]{5}|2[AB][0-9]{3})').fillna(False))
    rapport['mutations_rejetees_plausibilite'] = int((~plausibles).sum())
    df = df[plausibles].copy()
    for colonne, borne_min, borne_max in [('nombre_pieces_principales', 1, 50),
                                        ('surface_terrain', 0, 1_000_000),
                                        ('latitude', -90, 90), ('longitude', -180, 180)]:
        df[colonne] = df[colonne].where(df[colonne].between(borne_min, borne_max))
    df['code_type_local'] = df['type_local'].map({'Maison': 1, 'Appartement': 2})
    df['annee_mutation'] = df['date_mutation'].dt.year
    df['mois_mutation'] = df['date_mutation'].dt.month
    df['trimestre_mutation'] = df['date_mutation'].dt.quarter
    df['surface_par_piece'] = df['surface_reelle_bati'] / df['nombre_pieces_principales']
    df = df.drop(columns=[c for c in df if c.startswith('_')]).reset_index(drop=True)
    rapport['mutations_conservees'] = len(df)
    df.attrs['rapport'] = rapport
    return (df, rapport) if retourner_rapport else df
