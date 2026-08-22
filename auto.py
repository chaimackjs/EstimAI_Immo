import os
import requests
import zipfile
from io import BytesIO


# dataset URL est la variable contenant le lien web des donéées dvf
DATASE_URL="https://www.data.gouv.fr/api/1/datasets/demandes-de-valeurs-foncieres/"
# variable dossier dans laquelle sont stockés les données récupérées
DOSSIER="data/dvf"

headers= {
    "Accep": "application/json",
    "User-Agent": "Mozilla/5.0"
}



def telecharger_dfv():
    os.makedirs(DOSSIER,exist_ok=True)

    # on récupère le contenu de la page dvf 
    reponse = requests.get(DATASE_URL,headers=headers)

    print(reponse.status_code)
    print(reponse.url)
    print(reponse.headers.get("Content-Type"))

    dataset = reponse.json()


    print(dataset["resources"])



if __name__== '__main__':
    telecharger_dfv()
