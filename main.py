import pandas as pd 

def lire_fichier(chemin): 
    return pd.read_csv(chemin,sep="|")



if __name__== '__main__':

    df2021= lire_fichier("data\\dvf\\ValeursFoncieres-2021.txt")
    df2022= lire_fichier("data\\dvf\\ValeursFoncieres-2022.txt")
    df2023= lire_fichier("data\\dvf\\ValeursFoncieres-2023.txt")
    df2024= lire_fichier("data\\dvf\\ValeursFoncieres-2024.txt")
    df2026= lire_fichier("data\\dvf\\ValeursFoncieres-2025.txt")

