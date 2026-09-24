# EstimAI Immo - version corrigée

Cette version corrige la construction de la cible, l'utilisation des variables DPE,
la localisation, la comparaison des modèles et la sauvegarde pour la prédiction.
Les corrections ne constituent pas une promesse de meilleur R² : il faut le mesurer
sur les données réelles. Lire `AUDIT.md` pour le diagnostic du code initial.

## 1. Installation

Depuis le dossier du projet, sous Windows ou Linux :

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` contient les versions exactes utilisées pour les tests
locaux. Tous les chemins internes sont relatifs au dossier du projet, pas au
répertoire courant du terminal.

## 2. Collecter un périmètre cohérent

Exemple avec le département 69 : adapter les départements et les années au projet.
Il n'est pas nécessaire de commencer par toute la France.

```bash
python acquisition_donnees.py --departements 69 --annee-debut 2021 --annee-fin 2025 --avec-dpe
```

Le programme télécharge les **CSV DVF géolocalisés Etalab**, avec identifiant de
mutation, identifiant de parcelle et coordonnées. Les fichiers DVF bruts `.txt` de
l'ancienne version ne sont pas réutilisés : fabriquer un identifiant avec seulement
la date et le prix pourrait fusionner des ventes distinctes. Ils peuvent rester
sur disque. Les anciens `data/clean/dfv_dpe.csv` ne sont pas réutilisés non plus.

Les DPE sont collectés **sans limite silencieuse à 100 000 lignes**, par département
et année d'établissement. Chaque page est suivie jusqu'à la fin du curseur. Un total
incohérent ou une interruption fait échouer la collecte au lieu de publier un cache
présenté comme complet. Une extraction via une API vivante peut changer pendant
la pagination : dans ce cas, relancer la collecte.

Sans `--avec-dpe`, seule DVF est collectée. Cela permet de mesurer d'abord un
modèle de référence DVF. `--force` actualise les sources déjà présentes.
Utiliser une même livraison DVF pour tous les fichiers ; ne pas mélanger des
extractions qui se chevauchent. Le chemin distant `latest` peut évoluer.

## 3. Nettoyer et construire les profils

```bash
python preparer_donnees.py --force
```

Les exécutions suivantes peuvent omettre `--force`. Le cache est invalidé lorsque
les fichiers sources, le code de nettoyage ou les paramètres changent.

**Unité statistique : une mutation simple avec exactement une ligne de logement,
une disposition, une parcelle, un prix cohérent et sans local d'activité.** Les
dépendances sur la même parcelle sont admises et signalées. Le prix reste celui
de l'ensemble vendu, rapporté à la surface bâtie du logement : il n'isole pas
la valeur d'un garage ou d'un jardin.

C'est volontairement restrictif. Une maison répétée sur plusieurs lignes pour
des raisons cadastrales peut être exclue. Deux lignes d'appartements identiques
ne sont pas fusionnées arbitrairement. Sans identifiant de local, on ne peut pas
savoir avec certitude s'il s'agit du même logement. Le modèle n'est donc pas
évalué sur toutes les transactions immobilières.

Les bornes initiales sont dans `BornesDVF` de `nettoyage_dvf.py` : surface de
9 à 1 000 m², prix total d'au moins 1 000 euros, prix au m² de 100 à 50 000 euros.
Ce sont des hypothèses de périmètre, pas des règles universelles. Elles peuvent
écarter des ventes réelles. Les exclusions sont comptées dans le rapport.
Les fixer avant la comparaison, sans regarder le test pour les ajuster.

Les champs manquants restent manquants ; une surface de terrain inconnue n'est
pas remplacée par zéro. Des surfaces de terrain contradictoires ne sont pas
additionnées aveuglément.

### Ce que signifie le rapprochement DPE

Il s'agit d'un **profil statistique de la commune et du type de bien**, pas du
DPE individuel du logement. Le programme ne fait pas de faux appariement
« même commune = même bien ».

Par défaut, les profils couvrent les 730 jours précédant la vente, sans inclure
le jour de la vente. Ils donnent les proportions A à G, la part F/G, des indicateurs
d'isolation, l'effectif et l'ancienneté. Une proportion n'est utilisée qu'avec
au moins 20 observations valides. Les ventes sans profil sont conservées.

```bash
python preparer_donnees.py --minimum-dpe 20 --fenetre-jours 730 --force
```

La date de disponibilité utilisée est le maximum des dates d'établissement,
de réception et de dernière modification disponibles. Cela évite de placer la
version courante d'un DPE avant une modification connue. Si seule la date
d'établissement est fournie, la disponibilité est **estimée** et signalée.
Un DPE modifié récemment peut donc ne pas enrichir une vente ancienne.

**Limite historique :** l'API courante n'est pas une archive de toutes les versions
anciennes ni de leurs dates exactes de publication. Les DPE retirés peuvent être
absents. De même, une DVF publiée aujourd'hui peut contenir des corrections
rétrospectives. Le protocole n'est pas un backtest opérationnel parfaitement
« tel que connu à la date ». Celui-ci demanderait des instantanés archivés
et les délais de publication, pour DVF comme pour DPE.

## 4. Entraîner et comparer

```bash
python ml.py --r2-minimum 0.75
```

Le jeu est découpé chronologiquement : environ 64 % apprentissage, 16 % validation,
20 % test final. Une même journée n'est pas partagée entre les ensembles.
Les proportions exactes dépendent des dates. Un identifiant de mutation dupliqué
est refusé. Le sous-échantillonnage éventuel concerne seulement l'apprentissage.

Les candidats utilisent **les mêmes ventes de validation** :

- Médiane locale avec repli commune/type, département/type, type, puis globale.
- Ridge avec imputation, standardisation numérique et encodage des catégories.
- CatBoost DVF de référence.
- CatBoost segmenté, avec un modèle appartement et un modèle maison, si les deux
  segments contiennent assez de ventes.
- CatBoost DVF enrichi de comparables historiques par code postal et type de
  bien ; il n'est retenu que s'il améliore la validation.
- CatBoost DVF + profils DPE si la couverture DPE est suffisante.

La localisation comprend le code INSEE, le code postal, le département et les
coordonnées cadastrales disponibles. Ces dernières localisent la parcelle, pas
nécessairement la porte ou l'appartement.

La cible par défaut est `log(prix_m2)`. Les prédictions sont remises en euros
avant le calcul des métriques. Le logarithme n'est pas supposé meilleur par
principe : l'alternative `--sans-log` se compare sur validation. Le logarithme
modifie la fonction de perte ; l'exponentielle ne fournit pas automatiquement
une moyenne conditionnelle non biaisée en euros.

La sélection minimise la **MAE en euros/m² sur validation**, y compris avec la
cible alternative `valeur_fonciere`. L'arrêt anticipé de CatBoost utilise la
validation, jamais le test. Le meilleur candidat est réentraîné sur apprentissage
+ validation, avec le nombre d'arbres retenu, puis le test est évalué.
Le modèle final sauvegardé n'a pas appris sur le test. Si le R² du test est sous
le seuil indiqué, il est marqué comme bloqué et `predire.py` refuse de l'utiliser.

```bash
python ml.py --date-test 2025-01-01 --date-validation 2024-01-01
python ml.py --iterations 1500 --max-train 0
```

Ces commandes supposent des données suffisantes dans les périodes concernées.
Fixer les paramètres sur validation, pas en relançant des essais jusqu'à obtenir
un meilleur score sur le même test. Une validation spatiale séparée serait
nécessaire pour mesurer la généralisation à des communes inconnues.

### Modules de préparation

`utilitaires.py` contient les conversions communes utilisées par tout le
pipeline. Il normalise les noms de colonnes, les codes géographiques, les
nombres et les dates. Les valeurs impossibles à convertir deviennent manquantes
afin de pouvoir être contrôlées par les étapes suivantes.

`rapprochement_dvf_dpe.py` construit des profils DPE par commune et type de
bien, puis les ajoute aux ventes DVF. Ce n'est pas un appariement avec le DPE
du logement vendu: les indicateurs décrivent les diagnostics observés dans la
même commune et le même type de bien pendant une fenêtre historique. Par défaut,
la fenêtre est de 730 jours et un indicateur n'est calculé qu'à partir de 20
diagnostics valides.

`modeles.py` construit aussi des comparables de marché par code postal et type
de bien : nombre de ventes et prix moyen au m² sur les 365 et 730 jours
précédents. Une vente ne peut consulter ni son propre prix, ni un prix futur.
Le profil est reconstruit lors d'une prédiction future.

### Fichiers utiles

| Fichier | Utilisation |
|---|---|
| `reports/qualite_donnees.json` | Volumes bruts, exclusions, DPE dédoublonnés, dates estimées. |
| `reports/couverture_dpe.csv` | Couverture par année, département et type de bien. |
| `reports/comparaison_validation.csv` | Comparaison des candidats et apport du DPE. |
| `reports/metriques.json` | Protocole, versions logicielles et métriques du test final. |
| `reports/predictions_test.csv` | Prédictions et erreurs individuelles du test. |
| `reports/erreurs_par_type.csv` | Erreurs appartements/maisons et effectifs. |
| `reports/erreurs_par_commune.csv` | Erreurs par commune/type, avec effectifs. |
| `models/meilleur_modele.joblib` | Modèle retenu, schéma, prétraitement et métadonnées. |
| `models/profils_dpe.joblib` | Profils historiques nécessaires aux candidats enrichis DPE. |
| `models/profils_marche.joblib` | Comparables historiques nécessaires au candidat marché. |

Les métriques incluent MAE, RMSE et R² sur le prix au m² et le prix total,
erreur relative médiane, MAPE et proportions à moins de 10 % et 20 % d'erreur.
Une moyenne par commune calculée sur deux ventes n'a pas la même fiabilité que
sur plusieurs milliers : toujours lire la colonne d'effectif.

`reports/importance_variables.csv` classe les paramètres du meilleur CatBoost
sur validation. Son score pondéré combine 60 % d'importance CatBoost et 40 %
d'importance par permutation de la MAE. Une variable est retenue à partir de
1 %, seuil modifiable avec `--seuil-importance`. Ce classement n'utilise jamais
le jeu de test final.

## 5. Prédire

`exemple_bien.json` est un logement **fictif**, sans prix connu. Adapter ses
caractéristiques. Les informations absentes restent manquantes : il vaut mieux
renseigner les caractéristiques disponibles que laisser le modèle les imputer.

```bash
python predire.py exemple_bien.json
```

Le script reconstruit les variables dans le bon ordre, recharge le prétraitement
et, si nécessaire, le profil DPE antérieur à la date de prédiction. Il fournit
un prix au m² et un prix total estimés. Il ne produit pas d'intervalle de confiance
calibré. Une estimation dans une zone, une période ou un segment peu représenté
peut être mauvaise.

Conserver les modules Python et `profils_dpe.joblib` avec les modèles concernés.
Les candidats nommés sont appris sur l'apprentissage seul ; `meilleur_modele`
est réentraîné sur apprentissage + validation. Ne charger que des fichiers
`joblib` de confiance : ils peuvent exécuter du code lors du chargement.

## 6. Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Les tests couvrent le nettoyage, les mutations ambiguës, les codes géographiques,
les bornes temporelles DPE, la pagination simulée, le cache, les découpages,
l'imputation et un entraînement complet avec rechargement/prédiction.

**Vérifié ici : 40 tests réussis.** L'entraînement de test utilise uniquement
des données artificielles. Aucun score obtenu sur ces données ne constitue une
mesure de précision immobilière. Aucun modèle artificiellement entraîné n'est
livré comme modèle exploitable.

Les appels HTTP ont été testés avec des réponses simulées. Le réseau n'était
pas accessible depuis le processus Python de l'environnement : le téléchargement
réel complet n'a pas été exécuté. Les services et documentations ont été
consultés via le navigateur le 12 septembre 2026. Les données réelles ne
figuraient pas dans l'archive fournie.

## Sources

Voir les liens et les limites d'interprétation dans `AUDIT.md`.
