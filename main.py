import pandas as pd 

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

if __name__== '__main__':

    df2021= lire_fichier("data\\dvf\\ValeursFoncieres-2021.txt")
    df2022= lire_fichier("data\\dvf\\ValeursFoncieres-2022.txt")
    df2023= lire_fichier("data\\dvf\\ValeursFoncieres-2023.txt")
    df2024= lire_fichier("data\\dvf\\ValeursFoncieres-2024.txt")
    df2025= lire_fichier("data\\dvf\\ValeursFoncieres-2025.txt")

    print("\nDonnées 2021 : \n")
    afficher_infos_generales(df2021)

    print("\nDonnées 2022 : \n")
    afficher_infos_generales(df2022)

    print("\nDonnées 2023 : \n")
    afficher_infos_generales(df2023)

    print("\nDonnées 2024 : \n")
    afficher_infos_generales(df2024)

    print("\nDonnées 2025 : \n")
    afficher_infos_generales(df2025)
