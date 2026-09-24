import pytest
from acquisition_donnees import recuperer_dpe, lire_departements, annees_disponibles


class Reponse:
    def __init__(self, contenu):
        self.contenu = contenu
    def raise_for_status(self):
        pass
    def json(self):
        return self.contenu


class Session:
    def __init__(self, pages):
        champs = ['numero_dpe', 'code_departement_ban', 'date_etablissement_dpe',
                  'code_insee_ban', 'type_batiment', 'etiquette_dpe']
        self.reponses = iter([{'schema': [{'key': c} for c in champs]}, *pages])
        self.appels = []
    def get(self, url, **kwargs):
        self.appels.append((url, kwargs))
        return Reponse(next(self.reponses))


def test_pagination_collecte_tout_et_conserve_curseur():
    session = Session([{'total': 2, 'results': [{'numero_dpe': '1'}], 'next': '?after=1'},
                       {'total': 2, 'results': [{'numero_dpe': '2'}]}])
    df = recuperer_dpe('69', 2024, session=session)
    assert df.numero_dpe.tolist() == ['1', '2']
    assert 'code_departement_ban:"69"' in session.appels[1][1]['params']['qs']
    assert '2024-01-01' in session.appels[1][1]['params']['qs']
    assert session.appels[2][1]['params'] is None


def test_troncature_refusee():
    session = Session([{'total': 100001, 'results': [{'numero_dpe': '1'}]}])
    with pytest.raises(ValueError, match='biaiser'):
        recuperer_dpe('69', 2024, nb_ligne=100000, session=session)


def test_collecte_vide_a_un_schema():
    df = recuperer_dpe('69', 2021, session=Session([{'total': 0, 'results': []}]))
    assert df.empty and 'numero_dpe' in df


def test_page_vide_avec_suite_est_erreur():
    with pytest.raises(RuntimeError, match='incomplete'):
        recuperer_dpe('69', 2024, session=Session([{'results': [], 'next': '?after=1'}]))


def test_total_incomplet_sans_curseur_est_refuse():
    with pytest.raises(RuntimeError, match='Collecte incomplete'):
        recuperer_dpe('69', 2024, session=Session([{'total': 5, 'results': [{'numero_dpe': '1'}]}]))


def test_perimetre_explicit_et_annees_bornes():
    assert lire_departements(['69,01', '69']) == ['69', '01']
    assert list(annees_disponibles(2023, 2024)) == [2023, 2024]
    with pytest.raises(ValueError):
        lire_departements(['99'])
