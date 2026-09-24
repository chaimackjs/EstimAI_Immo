"""Preparation des diagnostics pour des profils statistiques, pas un appariement."""
import pandas as pd
from utilitaires import dates, normaliser_code, normaliser_colonnes

ISOLATIONS = ['qualite_isolation_enveloppe', 'qualite_isolation_murs',
              'qualite_isolation_menuiseries']
# Le jeu ADEME v2 emploie les noms courts ``qualite_isol_*``. Les exports plus
# anciens emploient les noms longs : les deux sont acceptes et normalises ici.
ALIASES_ISOLATION = {
    'qualite_isolation_enveloppe': ('qualite_isolation_enveloppe', 'qualite_isol_enveloppe'),
    'qualite_isolation_murs': ('qualite_isolation_murs', 'qualite_isol_mur'),
    'qualite_isolation_menuiseries': ('qualite_isolation_menuiseries', 'qualite_isol_menuiserie'),
}
COLONNES_DPE_UTILES = ['numero_dpe', 'date_etablissement_dpe', 'date_reception_dpe',
    'date_derniere_modification_dpe', 'etiquette_dpe', 'etiquette_ges',
    'code_insee_ban', 'code_departement_ban', 'type_batiment',
    *[nom for noms in ALIASES_ISOLATION.values() for nom in noms]]


def nettoyer_dpe(df, retourner_rapport=False):
    """Nettoyer et normaliser les diagnostics DPE bruts."""
    df = normaliser_colonnes(df).rename(columns={'code_insee_ban': 'code_insee'})
    requis = {'numero_dpe', 'date_etablissement_dpe', 'code_insee', 'type_batiment', 'etiquette_dpe'}
    if manquantes := sorted(requis - set(df)):
        raise ValueError(f'Colonnes DPE absentes : {manquantes}. Regenerer la collecte ADEME.')
    rapport = {'lignes_brutes': len(df)}
    for cible, alias in ALIASES_ISOLATION.items():
        disponibles = [c for c in alias if c in df]
        if disponibles:
            # Si les deux versions sont presentes, la valeur moderne est prioritaire
            # et l'ancienne sert uniquement de repli.
            df[cible] = df[disponibles].bfill(axis=1).iloc[:, 0]
    df['numero_dpe'] = df['numero_dpe'].astype('string').str.strip().replace('', pd.NA)
    df['code_insee'] = normaliser_code(df['code_insee'])
    df['type_batiment'] = df['type_batiment'].astype('string').str.strip().str.lower()
    df['code_type_local'] = df['type_batiment'].map({'maison': 1, 'appartement': 2})
    for c in ['date_etablissement_dpe', 'date_reception_dpe', 'date_derniere_modification_dpe']:
        df[c] = dates(df.get(c, pd.Series(pd.NaT, index=df.index)))
    dates_source = ['date_etablissement_dpe', 'date_reception_dpe', 'date_derniere_modification_dpe']
    # La version actuelle d'un diagnostic ne doit pas etre retro-injectee
    # avant sa reception/modification connue. Cela reste une approximation
    # conservatrice, pas une archive historique point-in-time de l'API.
    df['date_disponibilite'] = df[dates_source].max(axis=1)
    df['disponibilite_estimee'] = df[dates_source[1:]].isna().all(axis=1).astype(int)
    for c in ['etiquette_dpe', 'etiquette_ges']:
        serie = df.get(c, pd.Series(pd.NA, index=df.index)).astype('string').str.strip().str.upper()
        df[c] = serie.where(serie.isin(list('ABCDEFG')))
    for c in ISOLATIONS:
        serie = df.get(c, pd.Series(pd.NA, index=df.index)).astype('string').str.strip().str.lower()
        df[c + '_score'] = serie.map({'tr\u00e8s bonne': 4, 'tres bonne': 4, 'bonne': 3,
                                      'moyenne': 2, 'insuffisante': 1}).astype(float)
    df = df.dropna(subset=['numero_dpe', 'date_etablissement_dpe', 'code_insee', 'code_type_local'])
    df = df[df['code_insee'].str.fullmatch(r'(?:[0-9]{5}|2[AB][0-9]{3})').fillna(False)]
    rapport['lignes_invalides'] = rapport['lignes_brutes'] - len(df)
    avant = len(df)
    df = df.sort_values('date_disponibilite').drop_duplicates('numero_dpe', keep='last')
    rapport['doublons_numero_dpe'] = avant - len(df)
    rapport['disponibilite_estimee'] = int(df['disponibilite_estimee'].sum())
    rapport['isolation_renseignee'] = {
        c: int(df[c + '_score'].notna().sum()) for c in ISOLATIONS
    }
    rapport['diagnostics_conserves'] = len(df)
    colonnes = ['numero_dpe', 'code_insee', 'code_type_local', *dates_source,
                'date_disponibilite', 'disponibilite_estimee', 'etiquette_dpe', 'etiquette_ges',
                *[c + '_score' for c in ISOLATIONS]]
    df = df[colonnes].reset_index(drop=True)
    df.attrs['rapport'] = rapport
    return (df, rapport) if retourner_rapport else df
