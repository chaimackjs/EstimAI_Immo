import os
import requests
import zipfile
from io import BytesIO


# dataset URL est la variable contenant le lien web des donéées dvf
DATASE_URL="https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres/"
# variable dossier dans laquelle sont stockés les données récupérées
DOSSIER="data/dvf"

headers= {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0"
}



def telecharger_dfv():
    os.makedirs(DOSSIER,exist_ok=True)

    # on récupère le contenu de la page dvf 
    reponse = requests.get(DATASE_URL,headers=headers)

    print("reponse.status_code: ", reponse.status_code)
    print("reponse.url: ", reponse.url)

    dataset = reponse.json()

    fichiers =[]
    # dataset["resources"] est liste de dictionnaire que l'on parcourt
    for resource in dataset["resources"]:
        url = resource.get("url", "")
        format =  resource.get("format", "")

        # filtrer les ressources pour récuperer uniquement les ressources qui correspndent aux 5 dernières années
        if( format== "txt.zip" and url.endswith(".txt.zip") ):
            fichiers.append(resource)

    print("nombre de données trouvées: ", len(fichiers))

    for fichier in fichiers:
        url = fichier["url"]
        nom = fichier.get("title")

        print("\nTéléchargement : ",  nom)
        reponse_telechargement =  requests.get(url)

        # reconstruire et déziper les archives récupérer par la requête
        with zipfile.ZipFile(BytesIO(reponse_telechargement.content)) as archive:
            archive.extractall(DOSSIER)

if __name__== '__main__':
    telecharger_dfv()
