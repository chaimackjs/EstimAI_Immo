# Audit du projet EstimAI_Immo

**Archive examinée :** `EstimAI_Immo-main(3).zip`  
**Date :** 12 septembre 2026  
**Périmètre :** code Python, données et entraînement ; aucune donnée réelle ni
métrique enregistrée dans l'archive. Les numéros de lignes ci-dessous désignent
les fichiers de l'archive d'origine, pas ceux de la version corrigée.

## Conclusion

Le code contient des problèmes vérifiables qui empêchent d'interpréter correctement
les performances. Le premier est simple : **aucun des modèles ne consomme les
profils DPE calculés par le rapprochement**. Le second est l'absence de localisation
fine dans les variables d'apprentissage. La cible DVF est par ailleurs construite
par ligne brute sans traiter les ventes comprenant plusieurs biens.

Ces constats expliquent des faiblesses de conception. Ils ne permettent pas de
quantifier leur contribution aux erreurs ni de promettre un R² après correction.

## 1. Les variables DPE n'atteignent jamais les modèles

Dans `rapprochement_dvf_dpe.py`, lignes 34-55, les colonnes produites sont notamment :

```text
nb_dpe_cumule
etiquette_dpe_moyenne_locale
etiquette_ges_moyenne_locale
qualite_isolation_enveloppe_moyenne_locale
qualite_isolation_murs_moyenne_locale
qualite_isolation_menuiseries_moyenne_locale
```

Dans `ml.py`, lignes 31-49, les régressions demandent les noms individuels
`etiquette_dpe`, `annee_dpe`, `annee_construction`, etc. Ces champs ne sont pas
présents dans le résultat de la jointure. Comme les noms absents sont ignorés,
l'entraînement continue silencieusement, mais sans eux.

L'intersection vérifiée par exécution sur un exemple donne exactement six variables
pour les régressions : surface bâtie, nombre de pièces, surface de terrain, nombre
de lots, année et mois de mutation. La liste `HGB_FEATURES` des lignes 17-26 ne
contient pas non plus les profils DPE. Changer le volume de DPE ne peut donc pas
améliorer directement les prédictions de ces modèles dans cette version.

**Correction :** schéma explicite `DPE_FEATURES` partagé entre préparation,
apprentissage et prédiction ; comparaison CatBoost DVF / DVF + DPE sur les mêmes
ventes de validation. Aucun DPE individuel n'est inventé.

## 2. La localisation est insuffisante

Les régressions n'utilisent ni commune, ni code postal, ni département, ni type
de bien. HGB utilise le département et le type, mais ni commune ni coordonnées.
Pour deux observations ayant les mêmes variables conservées, un modèle
produira la même prédiction, même si les marchés locaux diffèrent fortement.

**Correction :** conservation des codes INSEE/postaux comme catégories, type
maison/appartement, département et latitude/longitude disponibles. Les CSV Etalab
fournissent ces coordonnées à la parcelle, pas à l'appartement [1].
CatBoost gère les variables catégorielles ; Ridge utilise un encodage approprié [5].
Ajouter directement des milliers de communes comme catégories natives dans HGB
ne convient pas : chaque variable catégorielle y est limitée à `max_bins`, au plus
255 catégories [6].

## 3. Une ligne DVF n'est pas forcément une vente d'un logement

Dans `nettoyage_dvf.py`, lignes 51-65, les lignes non résidentielles sont supprimées
avant toute analyse de l'ensemble de la mutation. Puis `valeur_fonciere` est divisée
par la surface de chaque ligne résidentielle. Il manque les identifiants permettant
d'identifier les ventes composées. Une mutation peut porter sur plusieurs locaux
et plusieurs parcelles ; le prix est indiqué au niveau d'une disposition dans
la source DVF [2][3].

### Reproduction artificielle du problème

Une vente groupée de deux appartements de 40 et 60 m² pour 300 000 euros peut
conduire le code d'origine à produire :

| Ligne | Prix attribué par le code | Prix au m² calculé |
|---|---:|---:|
| Appartement 40 m² | 300 000 euros | 7 500 euros/m² |
| Appartement 60 m² | 300 000 euros | 5 000 euros/m² |

Le prix de l'ensemble rapporté à sa surface est 3 000 euros/m², mais cela
**ne prouve pas** que chaque appartement vaut individuellement 3 000 euros/m².
La cible individuelle n'est tout simplement pas identifiable dans cet exemple.
Ce calcul a été reproduit avec le code fourni ; ce ne sont pas des ventes réelles.

**Correction :** utiliser les CSV Etalab et leur `id_mutation`, examiner toutes
les lignes avant de filtrer, puis conserver un périmètre conservateur : un logement
déclaré, une parcelle, une disposition, pas de local d'activité, prix cohérent.
Une mutation ambiguë n'est pas convertie artificiellement en plusieurs exemples.
Les dépendances sont signalées. Des surfaces de terrain contradictoires deviennent
manquantes plutôt que d'être additionnées au hasard.

**Coût de cette correction :** des ventes parfaitement réelles peuvent être
écartées, notamment les maisons représentées sur plusieurs lignes. Il s'agit
d'un choix de périmètre prudent, pas d'une reconstitution exhaustive des locaux.
L'identifiant de mutation Etalab n'est pas un identifiant permanent de logement.
Un simple `drop_duplicates()` ne résout pas cette ambiguïté.

## 4. Le rapprochement d'origine est un enrichissement statistique

Les clés de jointure sont `code_insee` et `code_type_local`. Le résultat est une
moyenne cumulée des diagnostics antérieurs dans ce groupe. Ce n'est pas une
correspondance entre le logement vendu et son DPE. L'orientation temporelle
`direction='backward'` et l'exclusion des correspondances du jour sont en revanche
une bonne précaution déjà présente.

**Correction :** conserver cette nature statistique et la nommer clairement,
ajouter effectifs et couverture, utiliser les proportions d'étiquettes sur une
fenêtre passée, tenir compte des dates de réception/modification disponibles.
Les seuils 730 jours / 20 DPE sont modifiables et ne sont pas des optima démontrés.

Un vrai appariement demanderait une stratégie supplémentaire : adresse normalisée,
cohérence du type et des dates, comparaison prudente des surfaces, détection des
candidats multiples et validation manuelle d'un échantillon. Même adresse et
surface proche ne prouvent pas l'identité de deux appartements. **Cette version
ne prétend pas implémenter cet appariement individuel.**

## 5. La collecte DPE n'est pas alignée avec DVF

`acquisition_donnees.py`, lignes 14 et 56-77, prend les 100 000 premières lignes
renvoyées par l'API, sans répartition géographique ni temporelle imposée.
`preparer_donnees.py`, lignes 30-42, combine en revanche cinq années de DVF.
Une troncature des résultats d'une API n'est pas un échantillon aléatoire ou
représentatif garanti. Les diagnostics de cette base commencent en juillet
2021 ; les ventes antérieures ne peuvent donc pas avoir un profil issu de cette
base disponible avant la vente [4].

**Correction :** collecte complète par département/année, cache brut, contrôle
de complétude et rapport de couverture historique. Sans les données de l'utilisateur,
le taux de couverture de l'ancienne version reste inconnu.

Les diagnostics décrivent uniquement la population observée dans les DPE collectés,
pas un recensement aléatoire exhaustif des logements de la commune. Les modifications
récentes et les diagnostics retirés limitent toute reconstruction historique.

## 6. Les comparaisons entre modèles ne sont pas équitables

Dans `ml.py`, ligne 53, `dropna()` retire toute ligne incomplète pour les régressions.
HGB conserve les variables explicatives manquantes, lignes 80-97. Les jeux de test
peuvent donc différer. Le même `random_state` ne rend pas identiques des jeux
de test construits à partir de populations différentes.

Le découpage aléatoire mélange par ailleurs les périodes anciennes et récentes.
Il n'est pas intrinsèquement interdit : il répond à une autre question que
l'estimation de ventes ultérieures. Les doublons/mutations multilignes peuvent
aussi se retrouver des deux côtés lorsqu'ils ne sont pas traités.

**Correction :** ensemble commun, découpage chronologique, schéma commun, imputation
apprise seulement sur l'apprentissage. Le choix et l'arrêt anticipé utilisent
la validation, pas le test. Ces séparations suivent les principes de prévention
des fuites de données documentés par scikit-learn [7].

Le passage à un test futur peut faire **baisser** le score, tout en le rendant
plus pertinent pour l'usage visé. L'ancien R² aléatoire et le nouveau R² temporel,
sur un périmètre nettoyé différent, ne sont pas directement comparables.

## 7. La sauvegarde n'est pas suffisante pour prédire

`ml.py`, lignes 135-142, sauvegarde uniquement l'estimateur. Le `StandardScaler`
retourné par la préparation ainsi que la liste/ordre des variables ne sont pas
sauvegardés avec les régressions. Une prédiction réutilisant les valeurs brutes
ne correspond alors pas à l'entraînement.

**Correction :** sauvegarder la pipeline de Ridge, le schéma d'entrée, la
transformation de cible, les versions logicielles et les profils DPE nécessaires.
Le script `predire.py` réutilise la même construction de variables.
La pipeline scikit-learn permet de garder les transformations et le modèle
ensemble, en apprenant les transformations sur l'apprentissage uniquement [7].

## Protocole de mesure recommandé

Commencer par la médiane locale, puis Ridge, CatBoost DVF et CatBoost DVF + DPE.
Comparer leurs MAE/RMSE et erreurs relatives sur les mêmes ventes de validation.
Vérifier si les DPE couvrent les périodes utiles, et si leur ajout réduit vraiment
l'erreur. Un enrichissement peut être inutile ou nuisible : sa présence ne garantit
pas une amélioration. Conserver un test final non utilisé pour les choix.

L'évaluation porte sur de futures mutations du périmètre collecté. Elle ne mesure
ni la généralisation spatiale à des territoires absents de l'apprentissage ni un
backtest parfait fondé sur les publications disponibles à chaque date.

## Vérifications effectuées

40 tests automatisés réussis, notamment nettoyage transactionnel, valeurs manquantes,
conservation des codes, absence d'utilisation de DPE futurs, fenêtre temporelle,
contrôle des effectifs, pagination simulée, invalidation du cache, découpage
chronologique et pipeline complète avec les quatre candidats sur données artificielles.
Les prédictions restent identiques après rechargement d'une pipeline sauvegardée.

**Non mesuré :** gain de précision réel, taux d'exclusion sur les fichiers de
l'utilisateur, représentativité/couverture DPE réelle, qualité d'un appariement
individuel (non implémenté), et collecte HTTP complète en conditions réelles.
Les appels réseau du processus Python étaient indisponibles. Aucun chiffre de
gain ni modèle artificiellement entraîné n'est présenté comme un résultat réel.

## Références officielles consultées le 12 septembre 2026

1. Etalab / data.gouv.fr, schéma CSV DVF et géolocalisation :
   https://github.com/datagouv/dvf/blob/master/README-CSV.md
2. Cerema, structure mutation/local/parcelle et limites des identifiants :
   https://doc-datafoncier.cerema.fr/doc/dv3f/local/idloc?v=11
3. Cerema, valeur foncière et dispositions :
   https://doc-datafoncier.cerema.fr/doc/dv3f/mutation/valeurfonc
4. ADEME, DPE logements existants depuis juillet 2021 et schéma de l'API :
   https://data.ademe.fr/datasets/dpe03existant
   https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant
5. CatBoost, variables catégorielles :
   https://catboost.ai/docs/en/features/categorical-features
6. scikit-learn, HistGradientBoostingRegressor et cardinalité catégorielle :
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
7. scikit-learn, prétraitement, pipelines et fuites de données :
   https://scikit-learn.org/stable/common_pitfalls.html
