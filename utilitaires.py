"""Fonctions communes de nettoyage et de normalisation des données.

Ce module ne réalise aucun apprentissage. Il prépare les noms de colonnes,
les codes, les nombres et les dates pour les étapes DVF et DPE.
"""
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def normaliser_nom(nom):
    """Convertir un nom de colonne en identifiant ASCII en minuscules."""
    texte = unicodedata.normalize('NFKD', str(nom)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '_', texte.lower()).strip('_')


def normaliser_colonnes(df):
    """Retourner une copie du DataFrame avec ses colonnes normalisées."""
    return df.rename(columns=normaliser_nom).copy()


def normaliser_code(serie, largeur=5):
    """Nettoyer une série de codes et compléter sa largeur avec des zéros."""
    return (serie.astype('string').str.strip().str.upper()
            .str.replace(r'\.0$', '', regex=True)
            .replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA, '<NA>': pd.NA})
            .str.zfill(largeur))


def numerique(serie):
    """Convertir une série en nombres, en gérant espaces et virgules décimales."""
    texte = (serie.astype('string').str.replace('\u00a0', '', regex=False)
             .str.replace(' ', '', regex=False).str.replace(',', '.', regex=False))
    return pd.to_numeric(texte, errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)


def dates(serie):
    """Convertir les dates ISO ou françaises en dates pandas normalisées."""
    # ISO (sources Etalab/ADEME) et jj/mm/aaaa (exports manuels).
    return pd.to_datetime(serie, format='mixed', dayfirst=True, errors='coerce').dt.normalize()
