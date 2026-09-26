# Protection des données — EstimAI Immo

Ce document décrit le traitement de données du projet EstimAI Immo. Il sert de
notice détaillée, de fiche de registre et d'analyse préliminaire de la nécessité
d'une AIPD. Le responsable du projet doit valider les éléments juridiques et
remplacer le contact provisoire avant tout usage commercial.

## Notice d'information

### Responsable et contact

Le responsable du traitement est la personne ou l'organisme qui exploite
l'instance EstimAI Immo. En l'absence d'une adresse dédiée, le contact
provisoire est le responsable du dépôt
[chaimackjs/EstimAI_Immo](https://github.com/chaimackjs/EstimAI_Immo), par le
canal privé indiqué sur son profil GitHub.

En production, la variable d'environnement `CONTACT_RGPD` doit contenir le nom
du responsable et une adresse de contact dédiée. Une demande relative aux
données personnelles ne doit pas être publiée dans une issue GitHub publique.

### Finalité et base légale

La finalité est de développer, évaluer et présenter un modèle d'estimation du
prix d'un bien immobilier. L'estimation est indicative et ne produit aucune
décision juridique ou financière automatique.

La base légale envisagée est l'intérêt légitime prévu à l'article 6.1.f du
RGPD : développer et démontrer un outil d'estimation immobilière. Le responsable
doit réaliser et conserver un test de mise en balance avant la mise en
production. Si le projet est exploité dans le cadre d'une mission publique, la
base légale doit être réexaminée avec le délégué à la protection des données.

### Données, sources et personnes concernées

Les données d'apprentissage proviennent de :

- [DVF, DGFiP et data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres),
  sous Licence Ouverte 2.0 ;
- [DPE des logements existants, ADEME](https://data.ademe.fr/datasets/dpe03existant).

Le traitement porte sur les caractéristiques de biens, les transactions
immobilières, les références cadastrales, les coordonnées géographiques et des
profils DPE agrégés. Même sans nom d'acquéreur ou de vendeur, la combinaison
d'une adresse, d'une parcelle, d'une date et d'un prix peut permettre une
identification indirecte. Les personnes concernées sont donc notamment les
propriétaires, vendeurs ou acquéreurs associés aux mutations réutilisées.

Dans le formulaire Streamlit, l'utilisateur fournit les caractéristiques et la
localisation d'un bien. Le code applicatif traite ces valeurs en mémoire pour
produire une estimation et ne les écrit volontairement ni dans une base ni dans
un fichier. Des journaux techniques peuvent toutefois être traités par
l'hébergeur.

### Destinataires, hébergement et transferts

Les données d'apprentissage sont accessibles uniquement aux personnes qui
administrent le projet. GitHub héberge le code, exécute les workflows et
conserve temporairement les artefacts. Streamlit Cloud héberge l'application et
traite les requêtes des utilisateurs. Le responsable doit vérifier les
conditions contractuelles, les lieux d'hébergement et les mécanismes applicables
aux éventuels transferts hors de l'Espace économique européen.

Les fichiers bruts et nettoyés ne doivent pas être publiés dans le dépôt GitHub.
Le modèle ne doit pas fournir un accès permettant de retrouver une transaction
DVF individuelle.

### Durées de conservation

<!-- markdownlint-disable MD013 -->

| Catégorie | Durée ou règle proposée |
| --- | --- |
| Saisie du formulaire | Traitement transitoire ; aucune conservation volontaire par l'application |
| Données brutes et propres sur un poste autorisé | Jusqu'au prochain entraînement validé, puis suppression ou archivage restreint ; réexamen au moins annuel |
| Artefacts du workflow principal | 14 jours |
| Artefacts du réentraînement hebdomadaire | 30 jours |
| Modèle publié | Durée de la version active, puis archivage restreint si nécessaire à l'audit |
| Rapports agrégés | Durée du projet, avec réexamen annuel |
| Journaux GitHub et Streamlit | Selon la configuration du service, à limiter au minimum nécessaire |

<!-- markdownlint-enable MD013 -->

Le responsable doit rendre ces règles effectives sur les postes locaux et dans
les services cloud. La documentation seule ne remplace pas une purge ou une
restriction d'accès.

### Droits des personnes

Selon la situation, les personnes peuvent demander l'accès, la rectification,
l'effacement ou la limitation des données qui les concernent et s'opposer au
traitement. Elles peuvent également introduire une réclamation auprès de la
[CNIL](https://www.cnil.fr/). Une demande doit préciser les éléments permettant
d'identifier la mutation concernée sans transmettre publiquement des documents
ou informations supplémentaires.

## Fiche du registre de traitement

<!-- markdownlint-disable MD013 -->

| Rubrique | Description |
| --- | --- |
| Nom du traitement | Entraînement et utilisation du modèle EstimAI Immo |
| Responsable | Exploitant de l'instance EstimAI Immo, à nommer formellement |
| Contact | Variable `CONTACT_RGPD` ; contact provisoire via le profil privé du responsable GitHub |
| Finalité | Développement, évaluation et démonstration d'une estimation immobilière indicative |
| Base légale envisagée | Intérêt légitime, article 6.1.f du RGPD, sous réserve du test de mise en balance |
| Personnes concernées | Utilisateurs du formulaire et personnes indirectement liées aux mutations DVF |
| Données | Bien, surface, prix historique, date, adresse, parcelle, localisation et statistiques DPE |
| Sources | DGFiP/data.gouv.fr pour DVF ; ADEME pour DPE ; saisie directe pour la prédiction |
| Destinataires | Équipe autorisée du projet ; GitHub et Streamlit comme prestataires techniques |
| Décision automatisée | Aucune décision produisant un effet juridique ; estimation indicative uniquement |
| Conservation | Règles du tableau précédent, à appliquer techniquement et à réexaminer annuellement |
| Sécurité | Dépôt sans données brutes, secrets GitHub, artefacts limités dans le temps, tests et audit de dépendances |
| Droits | Accès, rectification, effacement, limitation, opposition et réclamation auprès de la CNIL |
| Transferts | À vérifier selon les contrats, régions et garanties de GitHub et Streamlit |

<!-- markdownlint-enable MD013 -->

## Analyse de nécessité d'une AIPD

Cette analyse préliminaire utilise les critères publiés par la CNIL. Elle ne
remplace pas une AIPD complète.

<!-- markdownlint-disable MD013 -->

| Critère de risque | Appréciation pour EstimAI Immo |
| --- | --- |
| Évaluation ou scoring | Oui : estimation de la valeur d'un bien, sans décision automatique |
| Données à caractère hautement personnel | À examiner : localisation précise et valeur financière d'un bien |
| Traitement à grande échelle | Oui : plusieurs dizaines de milliers de mutations |
| Croisement d'ensembles de données | Oui : combinaison DVF, DPE et données géographiques |
| Usage innovant | Oui : apprentissage automatique et déploiement public |
| Surveillance systématique | Non identifiée |
| Personnes vulnérables | Non ciblées |
| Effet juridique ou exclusion d'un droit | Non, sous réserve que l'estimation reste indicative |

<!-- markdownlint-enable MD013 -->

Plusieurs critères sont réunis. Une **AIPD complète est recommandée avant une
mise en production publique**, et peut être obligatoire si le responsable
confirme que le traitement présente un risque élevé. Elle devra décrire :

1. les flux de données et les responsabilités ;
2. la nécessité de l'adresse, de la parcelle et des coordonnées exactes ;
3. les risques de réidentification, de divulgation et de mésusage du prix ;
4. les mesures de réduction de précision, d'agrégation et de contrôle d'accès ;
5. les contrats avec GitHub et Streamlit, les transferts et la gestion des
   incidents ;
6. le risque résiduel et la décision formelle du responsable.

## Mesures prioritaires avant production

- Configurer une adresse dédiée dans `CONTACT_RGPD`.
- Vérifier et documenter le test de mise en balance de l'intérêt légitime.
- Réduire ou supprimer les adresses, parcelles et coordonnées exactes qui ne
  sont pas indispensables au modèle publié.
- Mettre en œuvre la purge et les restrictions d'accès annoncées ci-dessus.
- Vérifier les contrats et transferts associés à GitHub et Streamlit Cloud.
- Finaliser l'AIPD et faire valider la documentation par le responsable ou le
  délégué à la protection des données.
