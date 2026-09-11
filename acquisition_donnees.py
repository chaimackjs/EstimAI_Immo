import os
from io import BytesIO
import zipfile

import pandas as pd
import requests


# URL de l'API contenant les ressources DVF.
DVF_API_URL = "https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres/"
# Dossier dans lequel les archives DVF sont extraites.
DVF_DATA_DIR = "data/dvf"
DPE_API_URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
DPE_MAX_ROWS = 100_000

HTTP_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


def telecharger_dvf():
    os.makedirs(DVF_DATA_DIR, exist_ok=True)

    reponse = requests.get(DVF_API_URL, headers=HTTP_HEADERS, timeout=60)
    reponse.raise_for_status()

    print("reponse.status_code: ", reponse.status_code)
    print("reponse.url: ", reponse.url)

    dataset = reponse.json()
    fichiers = []
    for resource in dataset["resources"]:
        url = resource.get("url", "")
        resource_format = resource.get("format", "")
        if resource_format == "txt.zip" and url.endswith(".txt.zip"):
            fichiers.append(resource)

    print("nombre de données trouvées: ", len(fichiers))

    for fichier in fichiers:
        url = fichier["url"]
        nom = fichier.get("title")
        print("\nTéléchargement : ", nom)
        reponse_telechargement = requests.get(
            url,
            headers=HTTP_HEADERS,
            timeout=120,
        )
        reponse_telechargement.raise_for_status()

        with zipfile.ZipFile(BytesIO(reponse_telechargement.content)) as archive:
            archive.extractall(DVF_DATA_DIR)


def recuperer_dpe(nb_ligne=DPE_MAX_ROWS, taille_page=10_000):
    """Récupérer les DPE par pages via le curseur fourni par l'API ADEME."""
    lignes = []
    url_suivante = DPE_API_URL

    while url_suivante and len(lignes) < nb_ligne:
        taille = min(taille_page, nb_ligne - len(lignes))
        if url_suivante == DPE_API_URL:
            reponse = requests.get(url_suivante, params={"size": taille}, timeout=60)
        else:
            reponse = requests.get(url_suivante, timeout=60)
        reponse.raise_for_status()

        data = reponse.json()
        resultats = data.get("results", [])
        lignes.extend(resultats)
        url_suivante = data.get("next")

        if not resultats:
            break

    return pd.DataFrame(lignes[:nb_ligne])


if __name__ == "__main__":
    telecharger_dvf()