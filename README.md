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
