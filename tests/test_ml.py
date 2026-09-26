"""Donnees artificielles : ces tests ne mesurent PAS la precision immobiliere reelle."""
import joblib
import numpy as np
import pandas as pd
import pytest
from ml import (decouper_chronologiquement, entrainer_et_evaluer,
                informations_version_modele, modele_ridge)
from modeles import BASE_FEATURES, construire_X, predire_bundle
from nettoyage_dvf import nettoyer_dvf
from nettoyage_dpe import nettoyer_dpe
from predire import predire_biens
from rapprochement_dvf_dpe import DPE_FEATURES, construire_profils_dpe, appliquer_profils_dpe
from test_donnees import vente, diagnostic
from verifier_modele import verifier_modele


def donnees_synthetiques(n=160):
    rng = np.random.default_rng(42)
    lignes = []
    for i, jour in enumerate(pd.date_range('2023-01-01', periods=n, freq='4D')):
        surface = int(rng.integers(25, 160))
        maison = bool(i % 2)
        prix_m2 = (3000 + 500 * (i % 3) - 2 * surface + .3 * i) * np.exp(rng.normal(0, .07))
        lignes.append(vente(id_mutation=f'2023-{i}', id_parcelle=f'parcelle-{i}',
            date_mutation=str(jour.date()), type_local='Maison' if maison else 'Appartement',
            surface_reelle_bati=surface, nombre_pieces_principales=max(1, round(surface / 20)),
            code_commune=['69123', '69266', '01001'][i % 3],
            code_departement='01' if i % 3 == 2 else '69',
            valeur_fonciere=prix_m2 * surface, latitude=45.0 + .1 * (i % 3),
            longitude=4.0 + .1 * (i % 3), surface_terrain=300 if maison else None))
    return nettoyer_dvf(pd.DataFrame(lignes))


def test_decoupage_chronologique_sans_chevauchement():
    df = donnees_synthetiques()
    train, valid, test = decouper_chronologiquement(df)
    assert train.date_mutation.max() < valid.date_mutation.min()
    assert valid.date_mutation.max() < test.date_mutation.min()
    assert set(train.id_mutation).isdisjoint(test.id_mutation)
    assert sum(map(len, [train, valid, test])) == len(df)


def test_version_modele_contient_sha_et_empreinte(tmp_path, monkeypatch):
    manifeste = tmp_path / 'manifest.json'
    manifeste.write_text('{"source": "test"}', encoding='utf-8')
    monkeypatch.setenv('GITHUB_SHA', '1234567890abcdef')
    monkeypatch.setenv('GITHUB_RUN_ID', '42')
    version = informations_version_modele(manifeste)
    assert version['git_sha'] == '1234567890abcdef'
    assert version['github_run_id'] == '42'
    assert version['manifest_sha256']


def test_meme_jour_non_partage_entre_ensembles():
    df = donnees_synthetiques()
    df['date_mutation'] = df['date_mutation'].dt.to_period('M').dt.to_timestamp()
    train, valid, test = decouper_chronologiquement(df)
    assert train.date_mutation.max() < valid.date_mutation.min() < test.date_mutation.min()


def test_mutation_dupliquee_refusee():
    df = donnees_synthetiques()
    with pytest.raises(ValueError, match='unique'):
        decouper_chronologiquement(pd.concat([df, df.iloc[:1]]))


def test_cible_jamais_parmi_variables():
    assert {'prix_m2', 'valeur_fonciere', 'id_mutation', 'id_parcelle'}.isdisjoint(BASE_FEATURES + DPE_FEATURES)


def test_meme_encodage_et_imputation_apres_serialisation(tmp_path):
    df = donnees_synthetiques()
    train, valid, _ = decouper_chronologiquement(df)
    modele = modele_ridge()
    X = construire_X(train, BASE_FEATURES)
    modele.fit(X, np.log(train.prix_m2))
    imputation = modele.named_steps['pretraitement'].named_transformers_['numeriques'].named_steps['imputation']
    assert imputation.statistics_[0] == np.median(train.surface_reelle_bati)
    bundle = {'modele': modele, 'colonnes': BASE_FEATURES, 'log_cible': True}
    avant = predire_bundle(bundle, valid)
    chemin = tmp_path / 'modele.joblib'
    joblib.dump(bundle, chemin)
    np.testing.assert_allclose(avant, predire_bundle(joblib.load(chemin), valid))


def test_complet_selection_refit_rechargement_et_prediction(tmp_path):
    df = donnees_synthetiques()
    lignes = []
    for code in ['69123', '69266', '01001']:
        for type_bien in ['maison', 'appartement']:
            for j in range(4):
                lignes.append(diagnostic(numero_dpe=f'{code}-{type_bien}-{j}', code_insee_ban=code,
                    type_batiment=type_bien, date_etablissement_dpe='2022-12-01',
                    date_reception_dpe='2022-12-02', date_derniere_modification_dpe='2022-12-02',
                    etiquette_dpe='C' if code == '69123' else 'F'))
    profils = construire_profils_dpe(nettoyer_dpe(pd.DataFrame(lignes)))
    df = appliquer_profils_dpe(df, profils, minimum_dpe=2)
    chemin_profils = tmp_path / 'source-profils.joblib'
    joblib.dump({'profils': profils, 'minimum_dpe': 2, 'fenetre_jours': 730,
                 'anciennete_max_jours': 730}, chemin_profils)
    models, reports = tmp_path / 'models', tmp_path / 'reports'
    rapport = entrainer_et_evaluer(df, models, reports, iterations=15, max_train=0,
                                  chemin_profils=chemin_profils)
    assert len(rapport['validation']) >= 4
    assert len({ligne['n'] for ligne in rapport['validation']}) == 1
    assert rapport['test_final']['n'] == len(decouper_chronologiquement(df)[2])
    choix = min(rapport['validation'], key=lambda x: x['mae_eur_m2'])['modele']
    assert choix == rapport['modele_selectionne_sur_validation']
    entree = dict(date_mutation='2024-06-01', code_insee='69123', type_local='Appartement',
                  surface_reelle_bati=70, nombre_pieces_principales=3)
    resultat = predire_biens(entree, models / 'meilleur_modele.joblib')
    assert resultat.prix_total_estime.iloc[0] > 0
    np.testing.assert_allclose(resultat.prix_total_estime, resultat.prix_m2_estime * 70)
    # Verifier aussi l'artefact DPE, meme si ce n'est pas le candidat retenu.
    dpe_resultat = predire_biens(entree, models / 'CatBoost_DVF_DPE.joblib')
    assert dpe_resultat.profil_dpe_disponible.iloc[0]
    assert (reports / 'metriques.json').exists()
    assert (reports / 'version_modele.json').exists()
    assert joblib.load(models / 'meilleur_modele.joblib')['version_modele']['identifiant']
    chemin_donnees = tmp_path / 'donnees.csv'
    df.to_csv(chemin_donnees, index=False)
    verifier_modele(models / 'meilleur_modele.joblib', chemin_donnees, nombre_lignes=5)
    importance = pd.read_csv(reports / 'importance_variables.csv')
    assert importance.score_pondere_pct.sum() == pytest.approx(100)
    assert {'variable', 'importance_catboost_pct', 'importance_permutation_pct', 'retenue'} <= set(importance)
