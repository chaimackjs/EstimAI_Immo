"""Schema des variables et objets serialisables pour entrainement/prediction."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from utilitaires import dates, normaliser_code, numerique

CATEGORIELLES_BASE = ['code_insee', 'code_postal', 'code_departement', 'type_local']
CATEGORIELLES_ENRICHIES = ['adresse_locale', 'parcelle_locale', 'lot_cadastral', 'voie_locale',
                           'section_cadastrale', 'geo_adresse', 'geo_cellule_100m',
                           'geo_cellule_500m', 'geo_cellule_1km', 'periode_mutation',
                           'code_nature_culture']
CATEGORIELLES = [*CATEGORIELLES_BASE, *CATEGORIELLES_ENRICHIES]
MARCHE_CLES = ['code_postal', 'type_local']
MARCHE_FEATURES = ['marche_nb_365j', 'marche_prix_m2_moyen_365j',
    'marche_nb_730j', 'marche_prix_m2_moyen_730j',
    'marche_tendance_prix_m2', 'marche_profil_disponible']
BASE_FEATURES_DVF = ['surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
    'nombre_de_lots', 'presence_dependance', 'terrain_ambigu', 'surface_par_piece',
    'annee_mutation', 'mois_mutation', 'trimestre_mutation', 'latitude', 'longitude',
    *CATEGORIELLES_BASE]
BASE_FEATURES = [*BASE_FEATURES_DVF, *MARCHE_FEATURES]
FEATURES_ENRICHIES = [*BASE_FEATURES_DVF, 'adresse_numero', 'surface_carrez_totale',
    'ratio_surface_carrez_bati', *CATEGORIELLES_ENRICHIES]
FEATURES_ENRICHIES_MARCHE = [*FEATURES_ENRICHIES, *MARCHE_FEATURES]


def construire_profils_marche(ventes):
    """Cumuls journaliers de prix observes par code postal et type de bien.

    Ces profils ne portent que sur les transactions passees : les prix du jour
    de la vente et les transactions futures ne peuvent jamais servir de
    variable explicative.
    """
    requis = set(MARCHE_CLES + ['date_mutation', 'prix_m2'])
    manquantes = sorted(requis - set(ventes))
    if manquantes:
        raise ValueError(f'Colonnes absentes pour les comparables : {manquantes}')

    df = ventes[MARCHE_CLES + ['date_mutation', 'prix_m2']].copy()
    df['date_mutation'] = dates(df['date_mutation'])
    df['code_postal'] = normaliser_code(df['code_postal'])
    df['type_local'] = df['type_local'].astype('string').str.strip().str.capitalize()
    df['prix_m2'] = numerique(df['prix_m2'])
    df = df.dropna(subset=MARCHE_CLES + ['date_mutation', 'prix_m2'])

    quotidien = (df.groupby(MARCHE_CLES + ['date_mutation'], observed=True, as_index=False)
                   .agg(marche_nb_jour=('prix_m2', 'size'),
                        marche_somme_prix_m2_jour=('prix_m2', 'sum'))
                   .sort_values(['date_mutation', *MARCHE_CLES]))

    colonnes = ['marche_nb_jour', 'marche_somme_prix_m2_jour']
    quotidien[['marche_nb_cumule', 'marche_somme_prix_m2_cumule']] = (
        quotidien.groupby(MARCHE_CLES, observed=True)[colonnes].cumsum())
    return quotidien.reset_index(drop=True)


def appliquer_profils_marche(ventes, profils, minimum_ventes=10):
    """Ajouter des comparables 365/730 jours strictement anterieurs a la vente."""
    if minimum_ventes < 1:
        raise ValueError('Le minimum de ventes doit etre positif.')

    resultat = ventes.drop(columns=[c for c in MARCHE_FEATURES if c in ventes]).copy().reset_index(drop=True)
    for colonne in MARCHE_FEATURES:
        resultat[colonne] = np.nan
    resultat['marche_profil_disponible'] = 0.0
    if profils is None or profils.empty or ventes.empty:
        return resultat

    base = resultat[MARCHE_CLES + ['date_mutation']].copy()
    base['_position'] = np.arange(len(base))
    base['date_mutation'] = dates(base['date_mutation'])
    base['code_postal'] = normaliser_code(base['code_postal'])
    base['type_local'] = base['type_local'].astype('string').str.strip().str.capitalize()
    base = base.dropna(subset=MARCHE_CLES + ['date_mutation'])

    profil = profils.copy()
    profil['date_mutation'] = dates(profil['date_mutation'])
    profil['code_postal'] = normaliser_code(profil['code_postal'])
    profil['type_local'] = profil['type_local'].astype('string').str.strip().str.capitalize()
    profil = profil.sort_values(['date_mutation', *MARCHE_CLES])

    def cumul_avant(decalage_jours):
        """Lire les cumuls connus avant une date décalée."""
        gauche = base.copy()
        gauche['_date_recherche'] = gauche['date_mutation'] - pd.to_timedelta(decalage_jours, unit='D')
        return pd.merge_asof(
            gauche.sort_values('_date_recherche'), profil,
            left_on='_date_recherche', right_on='date_mutation', by=MARCHE_CLES,
            direction='backward', allow_exact_matches=False).set_index('_position')

    maintenant, il_y_a_365, il_y_a_730 = cumul_avant(0), cumul_avant(365), cumul_avant(730)

    def fenetre(debut):
        """Calculer les volumes et moyennes sur une fenêtre passée."""
        nombre = (maintenant['marche_nb_cumule'].fillna(0) - debut['marche_nb_cumule'].fillna(0))
        somme = (maintenant['marche_somme_prix_m2_cumule'].fillna(0)
                 - debut['marche_somme_prix_m2_cumule'].fillna(0))
        return nombre, somme / nombre.replace(0, np.nan)

    nb_365, moyenne_365 = fenetre(il_y_a_365)
    nb_730, moyenne_730 = fenetre(il_y_a_730)
    disponible = nb_365.ge(minimum_ventes)
    index = nb_365.index

    resultat.loc[index, 'marche_nb_365j'] = nb_365
    resultat.loc[index, 'marche_nb_730j'] = nb_730
    resultat.loc[index, 'marche_prix_m2_moyen_365j'] = moyenne_365.where(disponible)
    resultat.loc[index, 'marche_prix_m2_moyen_730j'] = moyenne_730.where(nb_730.ge(minimum_ventes))
    resultat.loc[index, 'marche_tendance_prix_m2'] = (moyenne_365 - moyenne_730).where(disponible)
    resultat.loc[index, 'marche_profil_disponible'] = disponible.astype(int)
    return resultat


def construire_X(df, colonnes):
    """Construire les variables explicatives attendues par un modèle."""
    donnees = df.copy()
    date = dates(donnees.get('date_mutation', pd.Series(pd.NaT, index=df.index)))
    donnees['annee_mutation'] = date.dt.year
    donnees['mois_mutation'] = date.dt.month
    donnees['trimestre_mutation'] = date.dt.quarter
    donnees['periode_mutation'] = date.dt.to_period('M').astype('string')

    for c in ['surface_reelle_bati', 'nombre_pieces_principales']:
        donnees[c] = numerique(donnees.get(c, pd.Series(np.nan, index=df.index)))

    latitude = numerique(donnees.get('latitude', pd.Series(np.nan, index=df.index)))
    longitude = numerique(donnees.get('longitude', pd.Series(np.nan, index=df.index)))
    donnees['latitude'], donnees['longitude'] = latitude, longitude
    for nom, facteur in [('geo_cellule_100m', 1000), ('geo_cellule_500m', 200),
                         ('geo_cellule_1km', 100)]:
        lat_cellule = (latitude * facteur).round().astype('Int64').astype('string')
        lon_cellule = (longitude * facteur).round().astype('Int64').astype('string')
        donnees[nom] = lat_cellule + '_' + lon_cellule
    donnees['geo_adresse'] = (latitude.round(5).astype('string') + '_' +
                              longitude.round(5).astype('string'))

    code_insee = normaliser_code(donnees.get('code_insee', pd.Series(pd.NA, index=df.index)))
    voie = donnees.get('adresse_code_voie', pd.Series(pd.NA, index=df.index)).astype('string').str.strip().str.upper()
    donnees['voie_locale'] = code_insee + '_' + voie
    numero = donnees.get('adresse_numero', pd.Series(pd.NA, index=df.index)).astype('string').str.strip()
    suffixe = donnees.get('adresse_suffixe', pd.Series(pd.NA, index=df.index)).astype('string').str.strip().str.upper()
    donnees['adresse_locale'] = code_insee + '_' + voie + '_' + numero + '_' + suffixe.fillna('')

    parcelle = donnees.get('id_parcelle', pd.Series(pd.NA, index=df.index)).astype('string').str.strip().str.upper()
    donnees['parcelle_locale'] = parcelle
    lots = pd.concat([
        donnees.get(f'lot{i}_numero', pd.Series(pd.NA, index=df.index)).astype('string').str.strip()
        for i in range(1, 6)
    ], axis=1).fillna('')
    donnees['lot_cadastral'] = parcelle + '_' + lots.agg('_'.join, axis=1)
    donnees['section_cadastrale'] = parcelle.str.slice(stop=-4).where(parcelle.str.len().ge(8))

    surfaces_carrez = pd.concat([
        numerique(donnees.get(f'lot{i}_surface_carrez', pd.Series(np.nan, index=df.index)))
        for i in range(1, 6)
    ], axis=1)
    donnees['surface_carrez_totale'] = surfaces_carrez.sum(axis=1, min_count=1)
    donnees['ratio_surface_carrez_bati'] = (
        donnees['surface_carrez_totale'] / donnees['surface_reelle_bati'].where(
            donnees['surface_reelle_bati'].gt(0)))
    donnees['adresse_numero'] = numerique(
        donnees.get('adresse_numero', pd.Series(np.nan, index=df.index)))
    pieces = donnees['nombre_pieces_principales'].where(donnees['nombre_pieces_principales'].gt(0))
    donnees['surface_par_piece'] = donnees['surface_reelle_bati'] / pieces

    X = donnees.reindex(columns=colonnes).copy()
    for c in colonnes:
        if c in CATEGORIELLES:
            if c in {'code_insee', 'code_postal', 'code_departement'}:
                X[c] = normaliser_code(X[c], 2 if c == 'code_departement' else 5)
            else:
                X[c] = X[c].astype('string').str.strip().str.upper()
            X[c] = X[c].fillna('__MANQUANT__').astype(str)
        else:
            X[c] = numerique(X[c])
    return X


class MedianeLocale(RegressorMixin, BaseEstimator):
    """Reference : commune/type, puis departement/type, puis type, puis globale."""

    def __init__(self, minimum=10):
        """Initialiser le nombre minimal de références locales."""
        self.minimum = minimum

    def fit(self, X, y):
        """Apprendre les médianes aux différents niveaux géographiques."""
        df = X.reset_index(drop=True).copy()
        df['_cible'] = np.asarray(y)
        self.globale_ = float(np.median(y))
        self.tables_ = []

        for cles in [['code_insee', 'type_local'], ['code_departement', 'type_local'], ['type_local']]:
            stats = df.groupby(cles, dropna=False)['_cible'].agg(['median', 'count'])
            self.tables_.append((cles, stats.loc[stats['count'].ge(self.minimum), 'median']))

        return self

    def predict(self, X):
        """Prédire avec la médiane locale la plus précise disponible."""
        prediction = np.full(len(X), np.nan)

        for cles, table in self.tables_:
            index = pd.MultiIndex.from_frame(X[cles]) if len(cles) > 1 else pd.Index(X[cles[0]])
            valeurs = table.reindex(index).to_numpy(dtype=float)
            prediction = np.where(np.isnan(prediction), valeurs, prediction)

        return np.where(np.isnan(prediction), self.globale_, prediction)


class CatBoostParType(RegressorMixin, BaseEstimator):
    """Deux modeles CatBoost, un par type de logement, avec repli global."""

    def __init__(self, parametres):
        """Initialiser les paramètres des modèles CatBoost."""
        self.parametres = parametres

    def fit(self, X, y):
        """Entraîner un modèle global et un modèle par type de bien."""
        self.modele_global_ = CatBoostRegressor(**self.parametres)
        self.modele_global_.fit(X, y, cat_features=[c for c in CATEGORIELLES if c in X], verbose=False)

        self.modeles_type_ = {}
        self.effectifs_type_ = {}

        for type_local in X['type_local'].dropna().unique():
            masque = X['type_local'].eq(type_local)
            if masque.sum() < 100:
                continue

            modele = CatBoostRegressor(**self.parametres)
            modele.fit(X.loc[masque], np.asarray(y)[masque.to_numpy()],
                       cat_features=[c for c in CATEGORIELLES if c in X], verbose=False)
            self.modeles_type_[type_local] = modele
            self.effectifs_type_[type_local] = int(masque.sum())

        return self

    def predict(self, X):
        """Prédire par type de bien avec repli sur le modèle global."""
        prediction = np.asarray(self.modele_global_.predict(X), dtype=float)

        for type_local, modele in self.modeles_type_.items():
            masque = X['type_local'].eq(type_local)
            if masque.any():
                prediction[masque.to_numpy()] = modele.predict(X.loc[masque])

        return prediction

    def get_feature_importance(self, type='PredictionValuesChange'):
        """Agréger les importances selon les effectifs de chaque type."""
        total = sum(self.effectifs_type_.values())

        if not total:
            return self.modele_global_.get_feature_importance(type=type)

        importance = np.zeros(len(self.modele_global_.feature_names_), dtype=float)
        for type_local, modele in self.modeles_type_.items():
            importance += self.effectifs_type_[type_local] * modele.get_feature_importance(type=type)
        return importance / total


def predire_bundle(bundle, df):
    """Produire des prédictions avec un modèle et son schéma sauvegardé."""
    X = construire_X(df, bundle['colonnes'])
    prediction = np.asarray(bundle['modele'].predict(X), dtype=float)

    if bundle['log_cible']:
        prediction = np.exp(prediction)

    if not np.isfinite(prediction).all():
        raise ValueError('Le modele a produit des predictions non finies.')

    return prediction
