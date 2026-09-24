"""Comparaison sur validation temporelle, puis une seule evaluation du test final."""
import argparse
import json
import shutil
import platform
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from modeles import (BASE_FEATURES, BASE_FEATURES_DVF, CATEGORIELLES, FEATURES_ENRICHIES,
                     FEATURES_ENRICHIES_MARCHE, MARCHE_FEATURES, CatBoostParType,
                     MedianeLocale, construire_X, predire_bundle)
from rapprochement_dvf_dpe import DPE_FEATURES
from utilitaires import ROOT, dates, numerique

RANDOM_STATE = 42
DATA_PATH = ROOT / 'data' / 'clean' / 'dvf_dpe.csv'


def decouper_chronologiquement(df, date_test=None, date_validation=None):
    """Séparer les données en ensembles temporels sans chevauchement."""
    df = df.sort_values(['date_mutation', 'id_mutation']).copy()
    if df['id_mutation'].duplicated().any():
        raise ValueError('id_mutation doit etre unique : regenerer le nettoyage transactionnel.')
    if df['date_mutation'].isna().any():
        raise ValueError('Une date de mutation manque.')
    if len(df) < 10:
        raise ValueError('Il faut au moins 10 mutations pour trois ensembles distincts.')
    debut_test = pd.Timestamp(date_test) if date_test else df['date_mutation'].iloc[int(len(df) * .8)]
    developpement = df[df['date_mutation'] < debut_test]
    test = df[df['date_mutation'] >= debut_test]
    if developpement.empty:
        raise ValueError('Pas de donnees anterieures au test.')
    debut_valid = (pd.Timestamp(date_validation) if date_validation else
                   developpement['date_mutation'].iloc[int(len(developpement) * .8)])
    train = developpement[developpement['date_mutation'] < debut_valid]
    valid = developpement[developpement['date_mutation'] >= debut_valid]
    if min(len(train), len(valid), len(test)) < 2:
        raise ValueError('Train/validation/test doivent contenir au moins deux mutations chacun. '
                         'Adapter les dates ou collecter davantage de donnees.')
    return train, valid, test


def metriques(df, prediction, cible='prix_m2'):
    """Calculer les métriques d'erreur des prédictions immobilières."""
    surface = numerique(df['surface_reelle_bati']).to_numpy()
    if cible == 'prix_m2':
        reel_m2 = numerique(df['prix_m2']).to_numpy()
        pred_m2 = prediction
    else:
        reel_m2 = numerique(df['valeur_fonciere']).to_numpy() / surface
        pred_m2 = prediction / surface
    reel_total, pred_total = reel_m2 * surface, pred_m2 * surface
    erreurs = np.abs(pred_m2 - reel_m2) / reel_m2
    return {'n': len(df),
        'mae_eur_m2': float(mean_absolute_error(reel_m2, pred_m2)),
        'rmse_eur_m2': float(np.sqrt(mean_squared_error(reel_m2, pred_m2))),
        'r2_prix_m2': float(r2_score(reel_m2, pred_m2)),
        'mae_prix_total': float(mean_absolute_error(reel_total, pred_total)),
        'rmse_prix_total': float(np.sqrt(mean_squared_error(reel_total, pred_total))),
        'r2_prix_total': float(r2_score(reel_total, pred_total)),
        'mape_pourcent': float(100 * erreurs.mean()),
        'erreur_relative_mediane_pourcent': float(100 * np.median(erreurs)),
        'part_erreur_10pct': float((erreurs <= .10).mean()),
        'part_erreur_20pct': float((erreurs <= .20).mean())}


def importance_ponderee(modele, X_validation, y_validation, log_cible,
                         poids_modele=.60, repetitions=3, seuil_pct=1.0):
    """Classer les variables sans lire le jeu de test final.

    Le score combine l'importance structurelle CatBoost et la hausse de MAE
    observee lorsque chaque variable est melangee dans le jeu de validation.
    Une variable doit donc etre utilisee par le modele et degrader la
    prediction lorsqu'elle est retiree de son contexte.
    """
    if not 0 <= poids_modele <= 1:
        raise ValueError('Le poids d importance du modele doit etre entre 0 et 1.')
    if repetitions < 1 or seuil_pct < 0:
        raise ValueError('Les repetitions et le seuil d importance doivent etre positifs.')

    def score_mae(estimator, X, y):
        """Retourner la MAE négative attendue par scikit-learn."""
        prediction = np.asarray(estimator.predict(X), dtype=float)
        if log_cible:
            prediction = np.exp(prediction)
            y = np.exp(np.asarray(y, dtype=float))
        return -mean_absolute_error(y, prediction)

    importance_modele = np.asarray(modele.get_feature_importance(type='PredictionValuesChange'), dtype=float)
    permutation = permutation_importance(
        modele, X_validation, y_validation, scoring=score_mae, n_repeats=repetitions,
        random_state=RANDOM_STATE, n_jobs=1)
    importance_permutation = np.clip(np.asarray(permutation.importances_mean, dtype=float), 0, None)

    def normaliser(valeurs):
        """Normaliser des importances afin que leur somme vaille cent."""
        total = valeurs.sum()
        return np.zeros_like(valeurs) if total <= 0 else 100 * valeurs / total

    score_modele = normaliser(importance_modele)
    score_permutation = normaliser(importance_permutation)
    score_pondere = poids_modele * score_modele + (1 - poids_modele) * score_permutation
    resultat = pd.DataFrame({
        'variable': X_validation.columns,
        'importance_catboost_pct': score_modele,
        'importance_permutation_pct': score_permutation,
        'score_pondere_pct': score_pondere,
    })
    resultat['retenue'] = resultat['score_pondere_pct'].ge(seuil_pct)
    return resultat.sort_values('score_pondere_pct', ascending=False, ignore_index=True)


def modele_ridge(colonnes=BASE_FEATURES_DVF):
    """Construire le pipeline de régression Ridge de référence."""
    numeriques = [c for c in colonnes if c not in CATEGORIELLES]
    categories = [c for c in CATEGORIELLES if c in colonnes]
    preprocess = ColumnTransformer([
        ('numeriques', Pipeline([
            ('imputation', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('standardisation', StandardScaler())]), numeriques),
        ('categories', OneHotEncoder(handle_unknown='ignore'), categories)])
    return Pipeline([('pretraitement', preprocess), ('regression', Ridge(alpha=10, solver='lsqr'))])


def echantillonner_train(df, maximum):
    """Limiter la taille de l'apprentissage de façon reproductible."""
    if maximum and len(df) > maximum:
        return df.sample(n=maximum, random_state=RANDOM_STATE).sort_values('date_mutation')
    return df


def entrainer_et_evaluer(df, dossier_models=None, dossier_reports=None, iterations=1000,
                         log_cible=True, max_train=250_000, cible='prix_m2',
                         date_test=None, date_validation=None, chemin_profils=None,
                         r2_minimum=None, r2_total_minimum=None, poids_importance_modele=.60,
                         repetitions_importance=3, seuil_importance_pct=1.0,
                         chemin_profils_marche=None):
    """Comparer les modèles, retenir le meilleur et évaluer le test final."""
    models = Path(dossier_models or ROOT / 'models')
    reports = Path(dossier_reports or ROOT / 'reports')
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    requis = {'id_mutation', 'date_mutation', 'surface_reelle_bati', 'prix_m2', 'valeur_fonciere',
              'code_insee', 'code_departement', 'type_local'}
    if manque := sorted(requis - set(df)):
        raise ValueError(f'Colonnes absentes : {manque}. Relancer preparer_donnees.py --force.')
    df['date_mutation'] = dates(df['date_mutation'])
    for c in ['surface_reelle_bati', 'prix_m2', 'valeur_fonciere']:
        df[c] = numerique(df[c])
    if not (df['surface_reelle_bati'].gt(0) & df[cible].gt(0)).all():
        raise ValueError('Surface/cible invalide. Corriger le nettoyage, sans filtrer differemment par modele.')
    if not np.allclose(df['prix_m2'], df['valeur_fonciere'] / df['surface_reelle_bati']):
        raise ValueError('prix_m2 et valeur_fonciere/surface sont incoherents.')
    train_complet, valid, test = decouper_chronologiquement(df, date_test, date_validation)
    train = echantillonner_train(train_complet, max_train)
    print(f'Train : {len(train):,} ; validation : {len(valid):,} ; test reserve : {len(test):,}')
    avec_dpe = ('dpe_profil_disponible' in train and
                pd.to_numeric(train['dpe_profil_disponible'], errors='coerce').gt(0).any())
    if avec_dpe and (chemin_profils is None or not Path(chemin_profils).exists()):
        raise ValueError('Les profils DPE sont necessaires pour rendre le modele reutilisable en prediction.')
    if avec_dpe:
        destination = models / 'profils_dpe.joblib'
        if Path(chemin_profils).resolve() != destination.resolve():
            shutil.copy2(chemin_profils, destination)
    avec_marche = set(MARCHE_FEATURES).issubset(train.columns)
    if avec_marche:
        if chemin_profils_marche is None or not Path(chemin_profils_marche).exists():
            raise ValueError('Les profils de marche sont necessaires pour predire avec les comparables historiques.')
        destination_marche = models / 'profils_marche.joblib'
        if Path(chemin_profils_marche).resolve() != destination_marche.resolve():
            shutil.copy2(chemin_profils_marche, destination_marche)
    parametres = dict(iterations=iterations, depth=7, learning_rate=.05, loss_function='RMSE',
                      random_seed=RANDOM_STATE, thread_count=4, allow_writing_files=False)
    candidats = [
        ('Mediane_locale', MedianeLocale(), BASE_FEATURES_DVF, log_cible),
        ('Ridge_DVF', modele_ridge(BASE_FEATURES_DVF), BASE_FEATURES_DVF, log_cible),
        ('CatBoost_DVF', CatBoostRegressor(**parametres), BASE_FEATURES_DVF, log_cible)]
    if train['type_local'].value_counts().ge(100).sum() >= 2:
        candidats.append(('CatBoost_DVF_Segmente', CatBoostParType(parametres),
                          BASE_FEATURES_DVF, log_cible))
    candidats.append(('CatBoost_DVF_Enrichi', CatBoostRegressor(**parametres),
                      FEATURES_ENRICHIES, log_cible))
    if log_cible:
        parametres_direct = {**parametres, 'depth': 8, 'iterations': max(iterations, 1400)}
        candidats.append(('CatBoost_DVF_Enrichi_Direct', CatBoostRegressor(**parametres_direct),
                          FEATURES_ENRICHIES, False))
    if avec_marche:
        candidats.append(('CatBoost_DVF_Marche', CatBoostRegressor(**parametres),
                          BASE_FEATURES, log_cible))
        candidats.append(('CatBoost_DVF_Enrichi_Marche_Direct',
                          CatBoostRegressor(**parametres_direct),
                          FEATURES_ENRICHIES_MARCHE, False))
    if avec_dpe:
        candidats.append(('CatBoost_DVF_DPE', CatBoostRegressor(**parametres),
                          BASE_FEATURES_DVF + DPE_FEATURES, log_cible))
    else:
        print('Pas de profil DPE suffisamment renseigne dans le train : comparaison DVF seule.')
    versions = {nom: version(nom) for nom in ['numpy', 'pandas', 'scikit-learn', 'catboost', 'joblib']}
    versions['python'] = platform.python_version()
    scores, bundles = [], {}
    for nom, modele, colonnes, candidat_log in candidats:
        print(f'\nEntrainement : {nom}')
        X_train, X_valid = construire_X(train, colonnes), construire_X(valid, colonnes)
        y_train = np.log(train[cible].to_numpy()) if candidat_log else train[cible].to_numpy()
        y_valid = np.log(valid[cible].to_numpy()) if candidat_log else valid[cible].to_numpy()
        if isinstance(modele, CatBoostRegressor):
            modele.fit(X_train, y_train, cat_features=[c for c in CATEGORIELLES if c in X_train],
                       eval_set=(X_valid, y_valid), early_stopping_rounds=80,
                       use_best_model=True, verbose=False)
        else:
            modele.fit(X_train, y_train)
        bundle = {'format_version': 2, 'versions': versions, 'nom': nom, 'modele': modele, 'colonnes': colonnes,
                  'categoriels': CATEGORIELLES, 'cible': cible, 'log_cible': candidat_log,
                  'profils_dpe': 'profils_dpe.joblib' if nom.endswith('_DPE') else None,
                  'profils_marche': ('profils_marche.joblib'
                                     if any(c in colonnes for c in MARCHE_FEATURES) else None),
                  'fin_apprentissage': str(train['date_mutation'].max().date())}
        train_scores = metriques(train, predire_bundle(bundle, train), cible)
        valid_scores = metriques(valid, predire_bundle(bundle, valid), cible)
        scores.append({'modele': nom, **valid_scores, 'mae_train_eur_m2': train_scores['mae_eur_m2']})
        bundles[nom] = bundle
        joblib.dump(bundle, models / f'{nom}.joblib')
        print(f"MAE train = {train_scores['mae_eur_m2']:.1f} EUR/m2 ; "
              f"MAE validation = {valid_scores['mae_eur_m2']:.1f} EUR/m2 ; "
              f"R2 validation = {valid_scores['r2_prix_m2']:.3f}")
    tableau = pd.DataFrame(scores).sort_values('mae_eur_m2')
    tableau.to_csv(reports / 'comparaison_validation.csv', index=False)
    # Le choix est termine AVANT tout appel predict() sur le test.
    meilleur_nom = tableau.iloc[0]['modele']
    meilleur_catboost = tableau[tableau['modele'].str.startswith('CatBoost')].iloc[0]['modele']
    bundle_importance = bundles[meilleur_catboost]
    X_importance = construire_X(valid, bundle_importance['colonnes'])
    y_importance = (np.log(valid[cible].to_numpy()) if bundle_importance['log_cible']
                    else valid[cible].to_numpy())
    importance = importance_ponderee(
        bundle_importance['modele'], X_importance, y_importance,
        bundle_importance['log_cible'], poids_importance_modele,
        repetitions_importance, seuil_importance_pct)
    importance.insert(0, 'modele_analyse', meilleur_catboost)
    importance['modele_selectionne'] = meilleur_nom
    importance.to_csv(reports / 'importance_variables.csv', index=False)
    meilleur = bundles[meilleur_nom].copy()
    selectionne = meilleur['modele']
    meilleur['modele'] = clone(selectionne)
    if isinstance(selectionne, CatBoostRegressor):
        meilleur['modele'].set_params(iterations=selectionne.tree_count_)
    developpement = echantillonner_train(pd.concat([train_complet, valid]), max_train)
    X_dev = construire_X(developpement, meilleur['colonnes'])
    y_dev = (np.log(developpement[cible].to_numpy()) if meilleur['log_cible']
             else developpement[cible].to_numpy())
    if isinstance(meilleur['modele'], CatBoostRegressor):
        meilleur['modele'].fit(X_dev, y_dev,
                               cat_features=[c for c in CATEGORIELLES if c in X_dev], verbose=False)
    else:
        meilleur['modele'].fit(X_dev, y_dev)
    meilleur['fin_apprentissage'] = str(developpement['date_mutation'].max().date())
    meilleur['debut_test'] = str(test['date_mutation'].min().date())
    prediction = predire_bundle(meilleur, test)
    score_test = metriques(test, prediction, cible)
    autorise_m2 = r2_minimum is None or score_test['r2_prix_m2'] >= r2_minimum
    autorise_total = (r2_total_minimum is None or
                      score_test['r2_prix_total'] > r2_total_minimum)
    autorise_deploiement = autorise_m2 and autorise_total
    meilleur['autorise_deploiement'] = bool(autorise_deploiement)
    meilleur['seuil_r2_prix_m2'] = r2_minimum
    meilleur['seuil_r2_prix_total'] = r2_total_minimum
    meilleur['metriques_test'] = score_test
    joblib.dump(meilleur, models / 'meilleur_modele.joblib')
    diagnostic = test[['id_mutation', 'date_mutation', 'code_insee', 'type_local',
                       'surface_reelle_bati', 'prix_m2', 'valeur_fonciere']].copy()
    diagnostic['prix_m2_predit'] = prediction if cible == 'prix_m2' else prediction / diagnostic['surface_reelle_bati']
    diagnostic['prix_total_predit'] = diagnostic['prix_m2_predit'] * diagnostic['surface_reelle_bati']
    diagnostic['erreur_absolue_eur_m2'] = (diagnostic['prix_m2_predit'] - diagnostic['prix_m2']).abs()
    diagnostic['erreur_relative'] = diagnostic['erreur_absolue_eur_m2'] / diagnostic['prix_m2']
    diagnostic.to_csv(reports / 'predictions_test.csv', index=False)
    for cles, fichier in [(['type_local'], 'erreurs_par_type.csv'),
                          (['code_insee', 'type_local'], 'erreurs_par_commune.csv')]:
        (diagnostic.groupby(cles).agg(n=('id_mutation', 'size'),
            mae_eur_m2=('erreur_absolue_eur_m2', 'mean'),
            erreur_relative_mediane=('erreur_relative', 'median'))
            .to_csv(reports / fichier))
    rapport = {'modele_selectionne_sur_validation': meilleur_nom, 'versions': versions, 'cible': cible,
        'log_cible_demandee': log_cible, 'log_cible_selectionnee': meilleur['log_cible'],
        'validation': scores, 'test_final': score_test,
        'decision_deploiement': {'autorise': bool(autorise_deploiement),
                                 'seuil_r2_prix_m2': r2_minimum,
                                 'seuil_r2_prix_total': r2_total_minimum},
        'selection_variables': {
            'modele_analyse': meilleur_catboost,
            'modele_selectionne': meilleur_nom,
            'poids_catboost': poids_importance_modele,
            'poids_permutation': 1 - poids_importance_modele,
            'seuil_score_pondere_pct': seuil_importance_pct,
            'variables_retenues': importance.loc[importance['retenue'], 'variable'].tolist(),
        },
        'decoupage': {nom: {'n': len(part), 'debut': str(part['date_mutation'].min().date()),
                            'fin': str(part['date_mutation'].max().date())}
                     for nom, part in [('train', train), ('validation', valid), ('test', test)]},
        'n_apprentissage_final': len(developpement),
        'objectif_selection': 'MAE en EUR/m2 sur validation, jamais sur test',
        'limite': 'Evaluation sur mutations simples du perimetre collecte, pas sur toutes les ventes.'}
    (reports / 'metriques.json').write_text(json.dumps(rapport, indent=2, allow_nan=False), encoding='utf-8')
    print(f'\nModele retenu : {meilleur_nom}')
    print(f"TEST FINAL : MAE = {score_test['mae_eur_m2']:.1f} EUR/m2 ; "
          f"MAE prix total = {score_test['mae_prix_total']:.0f} EUR ; "
          f"R2 prix/m2 = {score_test['r2_prix_m2']:.3f} ; "
          f"R2 prix total = {score_test['r2_prix_total']:.3f}")
    if not autorise_deploiement:
        seuils_refuses = []
        if not autorise_m2:
            seuils_refuses.append(
                f'R2 prix/m2 {score_test["r2_prix_m2"]:.3f} < {r2_minimum:.3f}')
        if not autorise_total:
            seuils_refuses.append(
                f'R2 prix total {score_test["r2_prix_total"]:.3f} <= {r2_total_minimum:.3f}')
        print(f'DEPLOIEMENT BLOQUE : {" ; ".join(seuils_refuses)}.')
    return rapport


def main():
    """Lancer l'entraînement depuis la ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donnees', type=Path, default=DATA_PATH)
    parser.add_argument('--cible', choices=['prix_m2', 'valeur_fonciere'], default='prix_m2')
    parser.add_argument('--sans-log', action='store_true')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max-train', type=int, default=250000, help='0 = toutes les mutations du train.')
    parser.add_argument('--date-test', help='Debut du test final, au format AAAA-MM-JJ.')
    parser.add_argument('--date-validation', help='Debut de la validation, avant la date de test.')
    parser.add_argument('--r2-minimum', type=float,
                        help='Seuil optionnel de R2 prix/m2 du test final.')
    parser.add_argument('--r2-total-minimum', type=float, default=.70,
                        help='Le R2 prix total doit etre strictement superieur a ce seuil.')
    parser.add_argument('--exiger-qualite', action='store_true',
                        help='Retourner un code erreur si le modele ne franchit pas les seuils.')
    parser.add_argument('--poids-importance-modele', type=float, default=.60,
                        help='Poids CatBoost dans le score des variables (0 a 1).')
    parser.add_argument('--repetitions-importance', type=int, default=3,
                        help='Nombre de permutations par variable sur la validation.')
    parser.add_argument('--seuil-importance', type=float, default=1.0,
                        help='Score pondere minimal, en pourcentage, pour retenir une variable.')
    args = parser.parse_args()
    df = pd.read_csv(args.donnees, dtype={c: 'string' for c in CATEGORIELLES + ['id_mutation']}, low_memory=False)
    rapport = entrainer_et_evaluer(df, cible=args.cible, iterations=args.iterations,
        log_cible=not args.sans_log, max_train=args.max_train,
        date_test=args.date_test, date_validation=args.date_validation,
        chemin_profils=args.donnees.parent / 'profils_dpe.joblib',
        r2_minimum=args.r2_minimum,
        r2_total_minimum=args.r2_total_minimum,
        poids_importance_modele=args.poids_importance_modele,
        repetitions_importance=args.repetitions_importance,
        seuil_importance_pct=args.seuil_importance,
        chemin_profils_marche=args.donnees.parent / 'profils_marche.joblib')
    if args.exiger_qualite and not rapport['decision_deploiement']['autorise']:
        raise SystemExit('Le modele ne respecte pas les seuils de qualite demandes.')


if __name__ == '__main__':
    main()
