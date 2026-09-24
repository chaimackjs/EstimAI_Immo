"""Construire des profils DPE historiques par commune et type de bien.

Le rapprochement est statistique: aucun diagnostic individuel n'est attribué
à une vente. Les ventes sont enrichies avec les diagnostics disponibles dans
une fenêtre passée, sans utiliser les données postérieures à la vente.
"""
import numpy as np
import pandas as pd
from nettoyage_dpe import ISOLATIONS
from utilitaires import dates, normaliser_code

CLES = ['code_insee', 'code_type_local']
VARIABLES = [*[f'part_{lettre}' for lettre in 'ABCDEFG'], 'part_FG', 'ges_part_FG',
             *[c + '_score' for c in ISOLATIONS]]
DPE_FEATURES = ['dpe_nb_diagnostics', 'dpe_nb_etiquettes', 'dpe_profil_disponible',
                'dpe_anciennete_jours', *['dpe_' + c for c in VARIABLES]]


def construire_profils_dpe(df_dpe):
    """Construire les cumuls de diagnostics par commune, type et date."""
    df = df_dpe.copy()
    df['code_insee'] = normaliser_code(df['code_insee'])
    df['code_type_local'] = pd.to_numeric(df['code_type_local'], errors='coerce')
    df['date_disponibilite'] = dates(df['date_disponibilite'])
    df = df.dropna(subset=CLES + ['date_disponibilite'])
    df['code_type_local'] = df['code_type_local'].astype('int64')
    for lettre in 'ABCDEFG':
        df['part_' + lettre] = df['etiquette_dpe'].eq(lettre).astype('Float64')
    df['part_FG'] = df['etiquette_dpe'].isin(['F', 'G']).astype(float).where(df['etiquette_dpe'].notna())
    df['ges_part_FG'] = df['etiquette_ges'].isin(['F', 'G']).astype(float).where(df['etiquette_ges'].notna())
    aggregations = {'nb_diagnostics': ('numero_dpe', 'size'),
                    'nb_etiquettes': ('etiquette_dpe', 'count')}
    for c in VARIABLES:
        aggregations[c + '_somme'] = (c, 'sum')
        aggregations[c + '_nombre'] = (c, 'count')
    profil = (df.groupby(CLES + ['date_disponibilite'], as_index=False, observed=True)
              .agg(**aggregations).sort_values(['date_disponibilite', *CLES]))
    cumulables = list(aggregations)
    profil[cumulables] = profil.groupby(CLES, observed=True)[cumulables].cumsum()
    return profil.reset_index(drop=True)


def appliquer_profils_dpe(ventes, profils, fenetre_jours=730, minimum_dpe=20,
                          anciennete_max_jours=365):
    """Ajouter aux ventes les indicateurs DPE antérieurs à chaque mutation.

    ``fenetre_jours`` limite l'historique utilisé et ``minimum_dpe`` évite de
    calculer des proportions sur un nombre trop faible de diagnostics.
    """
    if fenetre_jours <= 0 or minimum_dpe <= 0 or anciennete_max_jours <= 0:
        raise ValueError('La fenetre, le minimum et l anciennete maximale doivent etre positifs.')
    resultat = ventes.drop(columns=[c for c in DPE_FEATURES if c in ventes]).copy().reset_index(drop=True)
    for c in DPE_FEATURES:
        resultat[c] = np.nan
    resultat[['dpe_nb_diagnostics', 'dpe_nb_etiquettes', 'dpe_profil_disponible']] = 0.0
    if profils is None or profils.empty or ventes.empty:
        return resultat
    base = resultat[CLES + ['date_mutation']].copy()
    base['_position'] = np.arange(len(base))
    base['code_insee'] = normaliser_code(base['code_insee'])
    base['date_mutation'] = dates(base['date_mutation'])
    base['code_type_local'] = pd.to_numeric(base['code_type_local'], errors='coerce')
    base = base.dropna(subset=CLES + ['date_mutation'])
    base['code_type_local'] = base['code_type_local'].astype('int64')
    profil = profils.copy()
    profil['code_insee'] = normaliser_code(profil['code_insee'])
    profil['date_disponibilite'] = dates(profil['date_disponibilite'])
    profil['code_type_local'] = profil['code_type_local'].astype('int64')
    profil = profil.sort_values(['date_disponibilite', *CLES])
    def passe(decalage):
        """Lire le dernier cumul antérieur à la date recherchée."""
        gauche = base.copy()
        gauche['_date_recherche'] = gauche['date_mutation'] - pd.to_timedelta(decalage, unit='D')
        return pd.merge_asof(gauche.sort_values('_date_recherche'), profil,
            left_on='_date_recherche', right_on='date_disponibilite', by=CLES,
            direction='backward', allow_exact_matches=False).set_index('_position')
    haut, bas = passe(0), passe(fenetre_jours)
    # Difference de cumuls : [vente - fenetre, vente), sans le jour de vente.
    cumulables = [c for c in profil if c not in CLES + ['date_disponibilite']]
    compte = haut[cumulables].astype(float).fillna(0) - bas[cumulables].astype(float).fillna(0)
    resultat.loc[compte.index, 'dpe_nb_diagnostics'] = compte['nb_diagnostics']
    resultat.loc[compte.index, 'dpe_nb_etiquettes'] = compte['nb_etiquettes']
    age = (haut['date_mutation'] - haut['date_disponibilite']).dt.days
    fiable = (compte['nb_etiquettes'].ge(minimum_dpe)
              & age.le(anciennete_max_jours).fillna(False))
    resultat.loc[compte.index, 'dpe_profil_disponible'] = fiable.astype(int)
    for c in VARIABLES:
        nombre = compte[c + '_nombre']
        valeur = (compte[c + '_somme'] / nombre.replace(0, np.nan)).where(fiable)
        resultat.loc[compte.index, 'dpe_' + c] = valeur
    resultat.loc[age.index, 'dpe_anciennete_jours'] = age.where(compte['nb_diagnostics'].gt(0))
    return resultat


def rapprocher_dvf_dpe(df_dvf, df_dpe, fenetre_jours=730, minimum_dpe=20,
                       anciennete_max_jours=365):
    """Construire les profils DPE puis enrichir les ventes DVF avec ceux-ci."""
    return appliquer_profils_dpe(df_dvf, construire_profils_dpe(df_dpe), fenetre_jours,
                                 minimum_dpe, anciennete_max_jours)
