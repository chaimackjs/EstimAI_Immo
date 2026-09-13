"""Schema des variables et objets serialisables pour entrainement/prediction."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from utilitaires import dates, normaliser_code, numerique

CATEGORIELLES = ['code_insee', 'code_postal', 'code_departement', 'type_local']
BASE_FEATURES = ['surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain',
    'nombre_de_lots', 'presence_dependance', 'terrain_ambigu', 'surface_par_piece',
    'annee_mutation', 'mois_mutation', 'trimestre_mutation', 'latitude', 'longitude',
    *CATEGORIELLES]


def construire_X(df, colonnes):
    donnees = df.copy()
    date = dates(donnees.get('date_mutation', pd.Series(pd.NaT, index=df.index)))
    donnees['annee_mutation'] = date.dt.year
    donnees['mois_mutation'] = date.dt.month
    donnees['trimestre_mutation'] = date.dt.quarter
    for c in ['surface_reelle_bati', 'nombre_pieces_principales']:
        donnees[c] = numerique(donnees.get(c, pd.Series(np.nan, index=df.index)))
    pieces = donnees['nombre_pieces_principales'].where(donnees['nombre_pieces_principales'].gt(0))
    donnees['surface_par_piece'] = donnees['surface_reelle_bati'] / pieces
    X = donnees.reindex(columns=colonnes).copy()
    for c in colonnes:
        if c in CATEGORIELLES:
            if c.startswith('code_'):
                X[c] = normaliser_code(X[c], 2 if c == 'code_departement' else 5)
            else:
                X[c] = X[c].astype('string').str.strip().str.capitalize()
            X[c] = X[c].fillna('__MANQUANT__').astype(str)
        else:
            X[c] = numerique(X[c])
    return X


class MedianeLocale(RegressorMixin, BaseEstimator):
    """Reference : commune/type, puis departement/type, puis type, puis globale."""
    def __init__(self, minimum=10):
        self.minimum = minimum

    def fit(self, X, y):
        df = X.reset_index(drop=True).copy()
        df['_cible'] = np.asarray(y)
        self.globale_ = float(np.median(y))
        self.tables_ = []
        for cles in [['code_insee', 'type_local'], ['code_departement', 'type_local'], ['type_local']]:
            stats = df.groupby(cles, dropna=False)['_cible'].agg(['median', 'count'])
            self.tables_.append((cles, stats.loc[stats['count'].ge(self.minimum), 'median']))
        return self

    def predict(self, X):
        prediction = np.full(len(X), np.nan)
        for cles, table in self.tables_:
            index = pd.MultiIndex.from_frame(X[cles]) if len(cles) > 1 else pd.Index(X[cles[0]])
            valeurs = table.reindex(index).to_numpy(dtype=float)
            prediction = np.where(np.isnan(prediction), valeurs, prediction)
        return np.where(np.isnan(prediction), self.globale_, prediction)


def predire_bundle(bundle, df):
    X = construire_X(df, bundle['colonnes'])
    prediction = np.asarray(bundle['modele'].predict(X), dtype=float)
    if bundle['log_cible']:
        prediction = np.exp(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError('Le modele a produit des predictions non finies.')
    return prediction
