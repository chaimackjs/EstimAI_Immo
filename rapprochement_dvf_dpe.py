import pandas as pd


def rapprocher_dvf_dpe(df_dvf, df_dpe):
    cles = ["code_insee", "code_type_local"]
    indicateurs_dpe = [
        "etiquette_dpe",
        "etiquette_ges",
        "qualite_isolation_enveloppe",
        "qualite_isolation_murs",
        "qualite_isolation_menuiseries",
    ]

    ventes = df_dvf.copy()
    ventes["date_mutation"] = pd.to_datetime(ventes["date_mutation"], errors="coerce")
    ventes["code_insee"] = ventes["code_insee"].astype("string").str.zfill(5)
    ventes["code_type_local"] = pd.to_numeric(
        ventes["code_type_local"], errors="coerce"
    )
    ventes = ventes.dropna(subset=cles + ["date_mutation"])
    ventes["code_type_local"] = ventes["code_type_local"].astype("int64")

    dpe = df_dpe.copy()
    dpe["date_etablissement_dpe"] = pd.to_datetime(
        dpe["date_etablissement_dpe"], errors="coerce"
    )
    dpe["code_insee"] = dpe["code_insee"].astype("string").str.zfill(5)
    dpe["code_type_local"] = pd.to_numeric(
        dpe["code_type_local"], errors="coerce"
    )
    dpe = dpe.dropna(subset=cles + ["date_etablissement_dpe"])
    dpe["code_type_local"] = dpe["code_type_local"].astype("int64")

    aggregations = {"nb_dpe": ("etiquette_dpe", "size")}
    for indicateur in indicateurs_dpe:
        aggregations[f"{indicateur}_somme"] = (indicateur, "sum")
        aggregations[f"{indicateur}_nombre"] = (indicateur, "count")

    dpe_journalier = (
        dpe.groupby(cles + ["date_etablissement_dpe"], as_index=False)
        .agg(**aggregations)
        .sort_values(["date_etablissement_dpe", *cles])
    )

    groupes = dpe_journalier.groupby(cles, observed=True)
    dpe_journalier["nb_dpe_cumule"] = groupes["nb_dpe"].cumsum()
    colonnes_profil = ["nb_dpe_cumule"]
    for indicateur in indicateurs_dpe:
        somme = f"{indicateur}_somme"
        nombre = f"{indicateur}_nombre"
        moyenne = f"{indicateur}_moyenne_locale"
        dpe_journalier[somme] = groupes[somme].cumsum()
        dpe_journalier[nombre] = groupes[nombre].cumsum()
        dpe_journalier[moyenne] = dpe_journalier[somme] / dpe_journalier[nombre]
        colonnes_profil.append(moyenne)

    profil_dpe = dpe_journalier[
        cles + ["date_etablissement_dpe", *colonnes_profil]
    ].sort_values(["date_etablissement_dpe", *cles])
    ventes = ventes.sort_values(["date_mutation", *cles])

    return pd.merge_asof(
        ventes,
        profil_dpe,
        left_on="date_mutation",
        right_on="date_etablissement_dpe",
        by=cles,
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns="date_etablissement_dpe").reset_index(drop=True)
