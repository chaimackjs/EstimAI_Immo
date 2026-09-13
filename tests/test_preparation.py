import pandas as pd
from preparer_donnees import preparer
from test_donnees import vente


def test_cache_reutilise_puis_invalide_par_source(tmp_path):
    data = tmp_path / 'data'
    (data / 'dvf').mkdir(parents=True)
    source = data / 'dvf' / '2024-69.csv'
    pd.DataFrame([vente()]).to_csv(source, index=False)
    sortie = preparer(data, sans_dpe=True, dossier_rapports=tmp_path / 'reports')
    mtime = sortie.stat().st_mtime_ns
    assert len(pd.read_csv(sortie)) == 1
    preparer(data, sans_dpe=True, dossier_rapports=tmp_path / 'reports')
    assert sortie.stat().st_mtime_ns == mtime
    pd.DataFrame([vente(), vente(id_mutation='2024-2')]).to_csv(source, index=False)
    preparer(data, sans_dpe=True, dossier_rapports=tmp_path / 'reports')
    assert len(pd.read_csv(sortie)) == 2
