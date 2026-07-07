# COMPAS Recidivism Prediction – Explainability & Fairness Analysis

## Présentation du projet

Ce projet a été réalisé dans le cadre d'un projet de formation en Data Science et Intelligence Artificielle. Il porte sur la prédiction de la récidive judiciaire à partir du célèbre jeu de données **COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)**.

Au-delà de la simple performance prédictive, ce projet s'intéresse à deux problématiques majeures de l'intelligence artificielle moderne :

- **l'explicabilité des modèles**, afin de comprendre pourquoi une prédiction est produite ;
- **l'équité algorithmique**, afin d'identifier et de limiter les biais susceptibles d'affecter certains groupes d'individus.

L'objectif est donc de développer un modèle performant tout en évaluant sa transparence et son impact sur l'équité des décisions.

---

## Contexte

Les outils d'aide à la décision fondés sur le Machine Learning sont aujourd'hui utilisés dans de nombreux domaines sensibles tels que la santé, la finance, le recrutement ou encore la justice.

Dans le domaine judiciaire américain, l'outil **COMPAS** est utilisé pour estimer le risque de récidive d'un individu. Ces prédictions peuvent influencer certaines décisions relatives aux peines, à la libération conditionnelle ou au suivi judiciaire.

En 2016, une enquête publiée par **ProPublica** a mis en évidence que le système COMPAS présentait des différences importantes de performances selon certains groupes de population, notamment selon l'origine ethnique. Cette publication a suscité un important débat scientifique sur les biais algorithmiques, la transparence des modèles et l'équité des systèmes d'intelligence artificielle.

Depuis, le jeu de données COMPAS est devenu une référence académique pour étudier les problématiques d'explicabilité, d'équité et d'IA responsable.

---

## Objectifs du projet

L'objectif principal est de construire un modèle de Machine Learning permettant de prédire la récidive tout en étudiant son comportement et son équité.

Plus précisément, ce projet vise à :

- explorer et nettoyer les données COMPAS ;
- analyser les caractéristiques des individus et identifier différents profils à l'aide d'une étape de clustering ;
- entraîner et comparer plusieurs modèles de classification ;
- interpréter les décisions du meilleur modèle grâce à la méthode SHAP ;
- mesurer les biais éventuels du modèle selon différents groupes de population ;
- proposer une stratégie de réduction des biais ;
- comparer les performances et les indicateurs d'équité avant et après la mitigation ;
- mettre en perspective ces résultats avec les travaux récents sur l'explicabilité et l'IA responsable.

---

# Jeu de données

## Origine des données

Le projet s'appuie sur le jeu de données **COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)**, largement utilisé dans la littérature scientifique pour l'étude de l'explicabilité, des biais algorithmiques et de l'équité des systèmes de Machine Learning.

Les données proviennent d'évaluations réalisées auprès de personnes poursuivies par la justice pénale dans le comté de Broward (Floride, États-Unis). Elles regroupent des informations démographiques, judiciaires et criminelles utilisées pour estimer le risque de récidive.

Ce jeu de données est devenu une référence académique à la suite de l'enquête menée par **ProPublica** en 2016, qui a soulevé des interrogations sur l'équité des prédictions produites par le système COMPAS.

---

## Variable cible

L'objectif du projet est de prédire la variable :

| Variable | Description |
|----------|-------------|
| **is_recid** | Indique si un individu a récidivé (1) ou non (0). |

Il s'agit donc d'un **problème de classification binaire**.

---

## Variables utilisées

Le modèle exploite plusieurs catégories d'informations :

- **Caractéristiques démographiques**
  - âge
  - sexe
  - origine ethnique
  - catégorie d'âge

- **Historique judiciaire**
  - nombre d'antécédents judiciaires (*priors_count*)
  - nombre de délits commis étant mineur
  - nombre de délits mineurs
  - nombre d'autres infractions

- **Informations relatives à l'infraction**
  - gravité de l'infraction (*charge_degree*)
  - groupe d'infraction (*charge_group*)

- **Scores COMPAS**
  - *decile_score*
  - *score_text*
  - *v_decile_score*
  - *v_score_text*

Une variable de **cluster**, obtenue lors de l'analyse non supervisée, a également été intégrée afin de représenter différents profils d'individus.

---

## Variables sensibles

Certaines variables sont considérées comme **sensibles**, car elles peuvent conduire à des différences de traitement entre groupes de population.

Dans ce projet, les variables suivantes ont fait l'objet d'une attention particulière :

| Variable | Raison |
|----------|--------|
| **race** | Étude des biais liés à l'origine ethnique |
| **sex** | Analyse des différences entre hommes et femmes |
| **age_cat** | Analyse selon les tranches d'âge |

Ces variables ont été conservées lors de l'analyse des biais mais retirées lors de la phase de mitigation afin d'étudier leur influence sur les performances et l'équité du modèle.

---

## Caractéristiques générales du jeu de données

Après les différentes étapes de nettoyage et de préparation des données, le jeu utilisé pour la modélisation présente les caractéristiques suivantes :

- plusieurs milliers d'observations ;
- un nombre limité de variables explicatives pertinentes ;
- une cible relativement équilibrée entre les deux classes ;
- un mélange de variables numériques et catégorielles nécessitant un prétraitement adapté.

Cette diversité de variables justifie l'utilisation d'un pipeline de prétraitement combinant normalisation des variables numériques et encodage des variables catégorielles avant l'entraînement des modèles de Machine Learning.

---

## Limites du jeu de données

Bien que très utilisé dans la recherche, le jeu de données COMPAS présente plusieurs limites.

- Les données concernent une population localisée (Broward County, Floride) et ne sont donc pas nécessairement représentatives d'autres contextes judiciaires.
- Certaines variables résultent elles-mêmes de décisions humaines pouvant déjà contenir des biais historiques.
- Les scores COMPAS utilisés comme variables explicatives sont issus d'un algorithme propriétaire dont le fonctionnement détaillé n'est pas public.
- Enfin, la notion de récidive dépend de nombreux facteurs sociaux, économiques et judiciaires qui ne sont pas entièrement représentés dans le jeu de données.

Ces limites doivent être prises en compte lors de l'interprétation des résultats et justifient la réalisation d'une analyse approfondie de l'explicabilité et de l'équité du modèle.
