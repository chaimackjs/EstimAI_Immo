import numpy as np
import pandas as pd
import pytest
from nettoyage_dvf import nettoyer_dvf, lire_fichier_dvf
from nettoyage_dpe import nettoyer_dpe
from modeles import appliquer_profils_marche, construire_profils_marche
from rapprochement_dvf_dpe import rapprocher_dvf_dpe, construire_profils_dpe, appliquer_profils_dpe
from utilitaires import normaliser_code


def vente(**kwargs):
    return dict(id_mutation='2024-1', date_mutation='2024-06-01', nature_mutation='Vente',
        numero_disposition='1', valeur_fonciere='300000,00', code_commune='69123',
        code_departement='69', code_postal='69003', id_parcelle='691230000A0001',
        type_local='Appartement', surface_reelle_bati='60', nombre_pieces_principales='3',
        nombre_lots='1', surface_terrain=None) | kwargs


def diagnostic(**kwargs):
    return dict(numero_dpe='DPE-1', date_etablissement_dpe='2024-01-01',
        date_reception_dpe='2024-01-03', date_derniere_modification_dpe='2024-01-03',
        code_insee_ban='69123', type_batiment='appartement', etiquette_dpe='C',
        etiquette_ges='D', qualite_isolation_enveloppe='bonne',
        qualite_isolation_murs='moyenne', qualite_isolation_menuiseries='bonne') | kwargs


def test_vente_simple_prix_comma_terrain_manquant():
    df = nettoyer_dvf(pd.DataFrame([vente()]))
    assert len(df) == 1
    assert df.iloc[0].prix_m2 == 5000
    assert pd.isna(df.iloc[0].surface_terrain)
    assert df.iloc[0].surface_par_piece == 20


def test_valeur_fonciere_deja_numerique():
    assert nettoyer_dvf(pd.DataFrame([vente(valeur_fonciere=300000)])).iloc[0].prix_m2 == 5000


def test_multi_logements_exclus_avant_calcul_prix():
    df = nettoyer_dvf(pd.DataFrame([vente(surface_reelle_bati='40'), vente(surface_reelle_bati='60')]))
    assert df.empty


def test_deux_lignes_identiques_ne_sont_pas_un_logement_unique():
    assert nettoyer_dvf(pd.DataFrame([vente(), vente()])).empty


def test_dependance_ne_double_pas_le_prix():
    df = nettoyer_dvf(pd.DataFrame([vente(), vente(type_local='D\u00e9pendance', surface_reelle_bati='0')]))
    assert len(df) == 1
    assert df.iloc[0].valeur_fonciere == 300000
    assert df.iloc[0].presence_dependance == 1


@pytest.mark.parametrize('extra', [
    {'type_local': 'Local industriel. commercial ou assimil\u00e9'},
    {'type_local': 'D\u00e9pendance', 'id_parcelle': 'autre-parcelle'},
    {'type_local': 'D\u00e9pendance', 'numero_disposition': '2'},
    {'type_local': 'D\u00e9pendance', 'valeur_fonciere': '200000'},
    {'type_local': 'D\u00e9pendance', 'date_mutation': '2024-06-02'},
])
def test_mutations_ambigues_rejetees(extra):
    assert nettoyer_dvf(pd.DataFrame([vente(), vente(**extra)])).empty


def test_terrain_ambigu_non_somme():
    df = nettoyer_dvf(pd.DataFrame([vente(surface_terrain='100'),
        vente(type_local='D\u00e9pendance', surface_terrain='200')]))
    assert len(df) == 1
    assert pd.isna(df.iloc[0].surface_terrain)
    assert df.iloc[0].terrain_ambigu == 1


@pytest.mark.parametrize('extra', [
    {'date_mutation': 'inconnue'}, {'valeur_fonciere': '0'},
    {'surface_reelle_bati': '0'}, {'id_mutation': None}, {'id_parcelle': None},
])
def test_lignes_invalides(extra):
    assert nettoyer_dvf(pd.DataFrame([vente(**extra)])).empty


def test_codes_conservent_zeros_corse_outre_mer():
    codes = normaliser_code(pd.Series(['1001', '01001', '2A004', '97101', 1001.0, None]))
    assert codes.iloc[:5].tolist() == ['01001', '01001', '2A004', '97101', '01001']
    assert pd.isna(codes.iloc[-1])


def test_comparables_marche_strictement_anterieurs_a_la_vente():
    ventes = nettoyer_dvf(pd.DataFrame([
        vente(id_mutation='1', date_mutation='2024-01-01', valeur_fonciere='100000', surface_reelle_bati='50'),
        vente(id_mutation='2', date_mutation='2024-02-01', valeur_fonciere='150000', surface_reelle_bati='50'),
        vente(id_mutation='3', date_mutation='2024-03-01', valeur_fonciere='200000', surface_reelle_bati='50'),
    ]))
    resultat = appliquer_profils_marche(ventes, construire_profils_marche(ventes), minimum_ventes=1)
    assert resultat.iloc[0].marche_profil_disponible == 0
    assert resultat.iloc[2].marche_profil_disponible == 1
    assert resultat.iloc[2].marche_prix_m2_moyen_365j == 2500


def test_refus_faux_regroupement_txt():
    with pytest.raises(ValueError, match='id_mutation'):
        lire_fichier_dvf('ValeursFoncieres-2024.txt')


def test_nettoyage_dpe_types_et_version_la_plus_recente():
    brut = pd.DataFrame([diagnostic(type_batiment=' APPARTEMENT ', etiquette_dpe=' c '),
        diagnostic(etiquette_dpe='F', date_derniere_modification_dpe='2024-03-01')])
    df, rapport = nettoyer_dpe(brut, retourner_rapport=True)
    assert len(df) == 1 and rapport['doublons_numero_dpe'] == 1
    assert df.iloc[0].etiquette_dpe == 'F'
    assert df.iloc[0].date_disponibilite == pd.Timestamp('2024-03-01')


def test_dpe_etablissement_seul_est_signale():
    df = nettoyer_dpe(pd.DataFrame([diagnostic(date_reception_dpe=None, date_derniere_modification_dpe=None)]))
    assert df.iloc[0].disponibilite_estimee == 1


def test_aliases_isolation_api_ademe_actuelle():
    df = nettoyer_dpe(pd.DataFrame([diagnostic(
        qualite_isolation_enveloppe=None, qualite_isolation_murs=None,
        qualite_isolation_menuiseries=None, qualite_isol_enveloppe='bonne',
        qualite_isol_mur='moyenne', qualite_isol_menuiserie='tres bonne')]))
    assert df.iloc[0].qualite_isolation_enveloppe_score == 3
    assert df.iloc[0].qualite_isolation_murs_score == 2
    assert df.iloc[0].qualite_isolation_menuiseries_score == 4


def test_pas_de_dpe_futur_ni_du_jour_de_vente():
    dvf = nettoyer_dvf(pd.DataFrame([vente()]))
    dpe = nettoyer_dpe(pd.DataFrame([
        diagnostic(numero_dpe='1', etiquette_dpe='F'),
        diagnostic(numero_dpe='2', etiquette_dpe='A', date_derniere_modification_dpe='2024-06-01'),
        diagnostic(numero_dpe='3', etiquette_dpe='A', date_derniere_modification_dpe='2024-06-02')]))
    resultat = rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=1)
    assert resultat.iloc[0].dpe_nb_diagnostics == 1
    assert resultat.iloc[0].dpe_part_FG == 1
    assert resultat.iloc[0].dpe_part_A == 0


def test_modification_future_pas_retroinjectee():
    dvf = nettoyer_dvf(pd.DataFrame([vente()]))
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic(date_derniere_modification_dpe='2026-01-01')]))
    assert rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=1).iloc[0].dpe_nb_diagnostics == 0


def test_fenetre_temporelle_et_borne_inferieure():
    dvf = nettoyer_dvf(pd.DataFrame([vente(date_mutation='2024-06-01')]))
    lignes = []
    for i, jour in enumerate(['2024-05-01', '2024-05-02', '2024-05-15']):
        lignes.append(diagnostic(numero_dpe=str(i), date_etablissement_dpe=jour,
                      date_reception_dpe=jour, date_derniere_modification_dpe=jour))
    dpe = nettoyer_dpe(pd.DataFrame(lignes))
    # [2 mai, 1 juin) pour 30 jours.
    resultat = rapprocher_dvf_dpe(dvf, dpe, fenetre_jours=30, minimum_dpe=1)
    assert resultat.iloc[0].dpe_nb_diagnostics == 2


def test_pas_de_melange_communes_ou_types():
    dvf = nettoyer_dvf(pd.DataFrame([vente()]))
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic(numero_dpe='1', code_insee_ban='01001'),
                                    diagnostic(numero_dpe='2', type_batiment='maison')]))
    assert rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=1).iloc[0].dpe_nb_diagnostics == 0


def test_minimum_dpe_ne_supprime_pas_vente():
    dvf = nettoyer_dvf(pd.DataFrame([vente()]))
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic()]))
    resultat = rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=20)
    assert len(resultat) == 1
    assert resultat.iloc[0].dpe_nb_diagnostics == 1
    assert resultat.iloc[0].dpe_profil_disponible == 0
    assert pd.isna(resultat.iloc[0].dpe_part_C)


def test_profil_dpe_trop_ancien_refuse():
    dvf = nettoyer_dvf(pd.DataFrame([vente(date_mutation='2024-06-01')]))
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic(date_derniere_modification_dpe='2024-01-03')]))
    resultat = rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=1, anciennete_max_jours=30).iloc[0]
    assert resultat.dpe_nb_diagnostics == 1
    assert resultat.dpe_profil_disponible == 0
    assert pd.isna(resultat.dpe_part_C)


def test_etiquette_inconnue_ne_compte_pas_comme_non_passoire():
    dvf = nettoyer_dvf(pd.DataFrame([vente()]))
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic(numero_dpe='1', etiquette_dpe='F'),
                                    diagnostic(numero_dpe='2', etiquette_dpe='inconnue')]))
    r = rapprocher_dvf_dpe(dvf, dpe, minimum_dpe=1).iloc[0]
    assert r.dpe_nb_diagnostics == 2 and r.dpe_nb_etiquettes == 1
    assert r.dpe_part_FG == 1
    assert sum(r['dpe_part_' + c] for c in 'ABCDEFG') == 1


def test_dpe_vide_preserve_ventes_et_ordre():
    dvf = nettoyer_dvf(pd.DataFrame([vente(id_mutation='b'), vente(id_mutation='a')]))
    resultat = appliquer_profils_dpe(dvf, None)
    assert resultat.id_mutation.tolist() == ['b', 'a']
    assert resultat.dpe_nb_diagnostics.sum() == 0


def test_profils_vides_via_dataframe():
    dpe = nettoyer_dpe(pd.DataFrame([diagnostic()])).iloc[:0]
    profils = construire_profils_dpe(dpe)
    assert profils.empty
    assert len(appliquer_profils_dpe(nettoyer_dvf(pd.DataFrame([vente()])), profils)) == 1
