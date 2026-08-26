import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

def lire_fichier(chemin): 
    return pd.read_csv(chemin,sep="|")

def afficher_infos_generales(df):

    print("Nombre de lignes : ",df.shape[0])
    print("Nombre de colonnes : ",df.shape[1])

    print("Valeur nulles : \n", df.isnull().sum())

    print("\nInformations générales : ")
    df.info();

    print("Les 5 premières lignes : ")
    print(df.head())

def nettoyage_dvf(df):
    # Standariser les noms de colonnes (maj->min) et remplacer les espaces par des "_"
    df.columns = (df.columns.str.lower().str.replace(" ","_"))
    # Transformer les data de la colonne "date_mutation" de type"str" en vrai dates qu'on pourra utiliser
    df["date_mutation"] = pd.to_datetime(df["date_mutation"],dayfirst=True)
    # Les valeurs de la colonne contiennent des virgules ce qui empêche l'utilisation numérique des valeurs
    # on transforme donc les "," en "." puis on transfomr en valeur numérique avec "pd.to_numeric"
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"].str.replace(",","."))
    # On supprime dans la colonne "nature_mutation' les lignes qui ne correspondent pas à des mutations de nature vente
    df = df[ df["nature_mutation"]=="Vente"  ]
    # On supprime dans la colonne "type_local' les lignes qui ne correspondent pas à des locaux de type maison ou appartement
    df = df[ df["type_local"].isin(["Maison","Appartement"])  ]
    # On supprime les colonnes complétement vides
    df = df.dropna(axis=1, how="all")
    # On supprime les colonnes non pertinante après première analyse
    df = df.drop(columns=["no_disposition","b/t/q","no_plan","type_local","nature_culture","nature_culture_speciale"])

    return df
    

def recuperer_dpe(nb_ligne=10000):
    url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"


    reponse = requests.get(url, params={"size":nb_ligne} )

    data = reponse.json()

    return pd.DataFrame(data["results"])


def nettoyage_dpe(df):

    # La liste des colonns à conserver: 
    colonnes_dpe_utiles=[
        "date_etablissement_dpe",
        "etiquette_dpe",
        "etiquette_ges",
        "annee_construction",
        "adresse_brut",
        "nom_commune_brut",
        "code_postal_brut",
        "type_batiment",
        "surface_habitable_logement",
        "type_energie_principale_chauffage",
        "qualite_isolation_enveloppe",
        "qualite_isolation_murs",
        "qualite_isolation_menuiseries"
    ]

    # Changement des dataframes avec uniquelemnt les ncolonnes souhaitées
    df = df[colonnes_dpe_utiles]

    # Transformer les data de la colonne "date_etablissement_dpe" de type"str" en vrai dates qu'on pourra utiliser
    df["date_etablissement_dpe"] = pd.to_datetime(df["date_etablissement_dpe"],yearfirst=True)

    # print("df['etiquette_dpe'].unique() : ", list(df["etiquette_dpe"].unique()) )
    # print("df['etiquette_ges'].unique() : ", list(df["etiquette_ges"].unique()) )

    # Dictionnaire de mapping pour les étiquettes dpe
    mapping_dpe={'A':7, 'B':6,'C':5, 'D':4,'E':3, 'F':2,'G':1}

    # Application du mapping 
    df["etiquette_dpe"]= df["etiquette_dpe"].map(mapping_dpe)
    df["etiquette_ges"]= df["etiquette_ges"].map(mapping_dpe)

    # print("df['etiquette_dpe'].unique() : ", df["etiquette_dpe"].unique() )
    # print("df['qualite_isolation_enveloppe'].unique() : ", list(df["qualite_isolation_enveloppe"].unique()) )
    
    mapping_isolation={'très bonne':4, 'bonne':3,'moyenne':2, 'insuffisante':1}
    df["qualite_isolation_enveloppe"]= df["qualite_isolation_enveloppe"].map(mapping_isolation)
    df["qualite_isolation_murs"]= df["qualite_isolation_murs"].map(mapping_isolation)
    df["qualite_isolation_menuiseries"]= df["qualite_isolation_menuiseries"].map(mapping_isolation)
    # print(df["qualite_isolation_enveloppe"].unique() )

    # print(df["type_batiment"].unique())
    df = df[ df["type_batiment"].isin(["maison","appartement"])  ]
    mapping_type_batiment={'maison':1, 'appartement':2}
    df["type_batiment"]= df["type_batiment"].map(mapping_type_batiment)

    return df

def afficher_correlation(df):
    correlation = df.corr(numeric_only=True)

    plt.imshow(correlation)

    plt.grid(False) 

    # Optional: Remove minor ticks that can look like grid lines
    plt.xticks(range(len(correlation.columns)),correlation.columns, rotation=90 )
    plt.yticks(range(len(correlation.columns)),correlation.columns)

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', ha='center', va='center', color='white')


    plt.colorbar()
    plt.show()

# Fonction pour enregistrer les dataframes pré-traité dans un fichier
def enregistrer_clean(df,nom_fichier):
    dossier=os.path.join("data","clean")
    os.makedirs(dossier,exist_ok=True)

    chemin = os.path.join(dossier,nom_fichier)

    df.to_csv(chemin,index=False)

# Récupération des données à partir des sources:
def recup_donnes_from_source():
    df2021= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2021.txt"))
    df2022= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2022.txt"))
    df2023= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2023.txt"))
    df2024= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2024.txt"))
    df2025= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2025.txt"))

    print("\nDonnées 2021 : \n")
    afficher_infos_generales(df2021)


    print("Valeurs nature_mutation :", df2021["Nature mutation"].unique())
    print("Valeurs type_local :", df2021["Type local"].unique())


    print("\nDonnées 2022 : \n")
    afficher_infos_generales(df2022)

    print("\nDonnées 2023 : \n")
    afficher_infos_generales(df2023)

    print("\nDonnées 2024 : \n")
    afficher_infos_generales(df2024)

    print("\nDonnées 2025 : \n")
    afficher_infos_generales(df2025)

    # print(df2021.columns[df2021.isnull().all()])

    df2021 = nettoyage_dvf(df2021)
    df2022 = nettoyage_dvf(df2022)
    df2023 = nettoyage_dvf(df2023)
    df2024 = nettoyage_dvf(df2024)
    df2025 = nettoyage_dvf(df2025)

    print("\nDonnées 2021 : \n")
    afficher_infos_generales(df2021)

    # Concaténation des données des 5 années:
    df_dvf = pd.concat([df2021,df2022,df2023,df2024,df2025])

    #Récuéparation et nettoyage des données dpe:
    df_dpe = recuperer_dpe(10000)
    df_dpe = nettoyage_dpe(df_dpe)

    afficher_infos_generales(df_dpe)

    # Enregistrer les dataframes traités dans 2 fichiers csv distincts 
    enregistrer_clean(df_dpe,"dpe.csv")
    enregistrer_clean(df_dvf,"dvf.csv")

    return df_dpe,df_dvf

if __name__== '__main__':

    use_clean=False

    if(use_clean==True):
        df_dpe= pd.read_csv(os.path.join("data","clean","dpe.csv"))
        df_dvf= pd.read_csv(os.path.join("data","clean","dvf.csv"))
    else:
        df_dpe,df_dvf=recup_donnes_from_source()

    afficher_correlation(df_dpe)
    afficher_correlation(df_dvf)
