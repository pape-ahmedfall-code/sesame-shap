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

---

# Méthodologie

Le projet a été développé selon une démarche classique de Data Science, allant de la compréhension des données jusqu'à l'évaluation de l'équité des modèles de Machine Learning. Chaque étape a été réalisée de manière progressive afin de garantir la qualité des analyses et la fiabilité des résultats.

L'ensemble du workflow est résumé dans la figure ci-dessous :

```
Chargement des données
        │
        ▼
Exploration et nettoyage
        │
        ▼
Analyse exploratoire
        │
        ▼
Clustering (K-Means)
        │
        ▼
Prétraitement des données
        │
        ▼
Entraînement des modèles
        │
        ▼
Comparaison des performances
        │
        ▼
Explicabilité (SHAP)
        │
        ▼
Analyse des biais
        │
        ▼
Atténuation des biais
        │
        ▼
Comparaison avant / après mitigation
```

---

## 1. Exploration et préparation des données

La première étape a consisté à explorer le jeu de données afin de mieux comprendre sa structure et sa qualité.

Les principales opérations réalisées sont les suivantes :

- inspection des variables disponibles ;
- analyse des types de données ;
- recherche de valeurs manquantes et de doublons ;
- étude des distributions des principales variables ;
- analyse des corrélations entre variables.

Cette phase a permis d'identifier les traitements nécessaires avant la modélisation.

---

## 2. Analyse exploratoire des données

Une analyse exploratoire (EDA) a ensuite été réalisée afin de mieux caractériser la population étudiée.

Plusieurs visualisations ont permis d'étudier :

- la répartition des individus selon l'âge, le sexe et l'origine ethnique ;
- la distribution des antécédents judiciaires ;
- la distribution des scores COMPAS ;
- les relations entre certaines variables explicatives et la variable cible.

Cette étape a facilité la compréhension des données et orienté les choix réalisés lors de la modélisation.

---

## 3. Clustering

Avant d'entraîner les modèles supervisés, une analyse non supervisée a été menée à l'aide de l'algorithme **K-Means**.

Le nombre optimal de groupes a été déterminé grâce à la méthode du coude (*Elbow Method*).

Les clusters obtenus ont ensuite été visualisés grâce à :

- UMAP ;
- t-SNE.

Cette étape a permis d'identifier différents profils d'individus partageant des caractéristiques similaires. Le cluster obtenu a ensuite été intégré comme variable explicative dans les modèles de classification.

---

## 4. Prétraitement des données

Les variables numériques et catégorielles ont nécessité des traitements différents avant l'entraînement des modèles.

Les principales étapes du prétraitement sont :

- normalisation des variables numériques à l'aide de **StandardScaler** ;
- encodage des variables catégorielles avec **OneHotEncoder** ;
- combinaison des traitements grâce à **ColumnTransformer** ;
- séparation des données en ensembles d'entraînement et de test selon une stratégie stratifiée.

Cette approche garantit un pipeline reproductible et limite les risques de fuite d'information entre les jeux d'entraînement et de test.

---

## 5. Modélisation

Trois modèles de Machine Learning ont été entraînés puis comparés :

| Modèle | Objectif |
|---------|----------|
| Régression Logistique | Modèle de référence simple et interprétable |
| Random Forest | Ensemble d'arbres de décision robuste et performant |
| Gradient Boosting | Méthode de boosting permettant d'améliorer progressivement les prédictions |

Les modèles ont été évalués à l'aide de plusieurs métriques :

- Accuracy ;
- Precision ;
- Recall ;
- F1-score ;
- ROC-AUC ;
- matrice de confusion.

Le **Random Forest** a obtenu les meilleures performances globales et a donc été retenu pour les analyses d'explicabilité et d'équité.

---

## 6. Explicabilité du modèle

Afin de comprendre les décisions du modèle retenu, une analyse d'explicabilité a été réalisée avec la bibliothèque **SHAP (SHapley Additive exPlanations)**.

Plusieurs visualisations ont été utilisées :

- SHAP Summary Plot ;
- SHAP Beeswarm Plot ;
- SHAP Bar Plot ;

Ces graphiques permettent d'identifier les variables les plus influentes et d'expliquer les prédictions réalisées par le modèle.

---

## 7. Analyse des biais

L'équité du modèle a ensuite été évaluée selon plusieurs variables sensibles :

- race ;
- sex ;
- age_cat.

Plusieurs indicateurs ont été étudiés :

- True Positive Rate (TPR) ;
- False Positive Rate (FPR).

Cette analyse permet de vérifier si certains groupes sont favorisés ou défavorisés par les décisions du modèle.

---

## 8. Atténuation des biais

Une stratégie simple de mitigation a ensuite été mise en œuvre.

Elle consiste à supprimer certaines variables sensibles lors de l'entraînement du modèle afin de réduire leur influence sur les prédictions.

Le modèle a ensuite été réentraîné puis réévalué selon les mêmes indicateurs de performance et d'équité.

Cette comparaison permet de mesurer le compromis entre performance prédictive et réduction des biais.

---

# Résultats obtenus

## Comparaison des modèles

Trois modèles de Machine Learning ont été entraînés et comparés afin d'identifier celui offrant le meilleur compromis entre performance prédictive, robustesse et capacité d'interprétation.

Les performances obtenues sont résumées dans le tableau suivant.

| Modèle | Accuracy | F1-score | ROC-AUC | Observation |
|---------|:--------:|:--------:|:-------:|-------------|
| Régression Logistique | ~0.70 | ~0.66 | ~0.77 | Modèle simple, facilement interprétable mais limité pour capturer les relations complexes entre variables. |
| Gradient Boosting | ~0.72 | ~0.70 | ~0.80 | Meilleure capacité de généralisation que la régression logistique mais performances inférieures au Random Forest. |
| Random Forest | **~0.91** | **~0.91** | **~0.97** | Meilleur modèle de l'étude, offrant les meilleures performances globales. |

Les résultats montrent clairement que le **Random Forest** surpasse les deux autres modèles sur l'ensemble des métriques étudiées.

Cette amélioration s'explique par sa capacité à modéliser des relations non linéaires, à gérer les interactions complexes entre variables et à réduire le risque de surapprentissage grâce à l'agrégation d'un grand nombre d'arbres de décision.

Pour ces raisons, ce modèle a été retenu pour les analyses d'explicabilité et d'équité.

---

## Interprétation des prédictions avec SHAP

Les performances d'un modèle ne suffisent pas à garantir sa fiabilité dans un contexte sensible comme la justice pénale. Il est également nécessaire de comprendre les raisons qui conduisent le modèle à produire une prédiction.

Pour répondre à cette problématique, le projet utilise **SHAP (SHapley Additive exPlanations)**.

L'analyse SHAP met en évidence les variables qui influencent le plus les décisions du modèle.

Les principales variables explicatives sont notamment :

- **priors_count** (nombre d'antécédents judiciaires) ;
- **decile_score** ;
- **v_decile_score** ;
- **age** ;
- **days_b_screening_arrest** ;
- certaines variables décrivant les infractions.

Les graphiques SHAP permettent d'observer :

- l'importance globale de chaque variable ;
- l'effet positif ou négatif de chaque variable sur la prédiction ;
- l'explication individuelle d'une décision donnée.

Cette analyse montre que le modèle ne prend pas ses décisions de manière arbitraire mais s'appuie principalement sur les informations liées aux antécédents judiciaires et aux évaluations du risque déjà présentes dans les données.

---

## Analyse des biais

Une fois le modèle interprété, une analyse d'équité a été menée afin de vérifier si ses performances variaient selon différents groupes de population.

L'étude porte principalement sur trois variables sensibles :

- la race ;
- le sexe ;
- la tranche d'âge.

Deux indicateurs ont été calculés :

- True Positive Rate (TPR) ;
- False Positive Rate (FPR).

Les résultats montrent que les performances ne sont pas parfaitement homogènes entre les différents groupes.

Des écarts apparaissent notamment concernant les taux de vrais positifs et de faux positifs selon certaines catégories démographiques.

Ces différences illustrent une problématique classique en intelligence artificielle responsable : un modèle peut présenter d'excellentes performances globales tout en restant moins équitable pour certains groupes de population.

---

## Atténuation des biais

Afin de limiter l'influence des variables sensibles sur les prédictions, une stratégie de mitigation a été mise en œuvre.

Cette approche consiste à retirer certaines variables sensibles du processus d'entraînement du modèle, tout en les conservant uniquement pour l'évaluation de l'équité.

Le modèle a ensuite été réentraîné puis comparé au modèle initial.

Les résultats montrent que :

- les performances prédictives diminuent légèrement ;
- certains écarts observés entre groupes sont réduits ;
- le modèle devient plus équilibré au regard des indicateurs d'équité.

Cette expérience met en évidence un compromis bien connu en Machine Learning responsable : améliorer l'équité d'un modèle peut entraîner une légère diminution de ses performances prédictives.

---

## Enseignements principaux

Les principaux enseignements de ce projet sont les suivants :

- le **Random Forest** constitue le modèle le plus performant pour cette tâche de prédiction ;
- **SHAP** fournit des explications précises permettant de comprendre les décisions du modèle ;
- l'analyse d'équité révèle des différences de comportement entre certains groupes démographiques ;
- la suppression des variables sensibles réduit une partie de ces écarts, mais ne les élimine pas totalement ;
- l'explicabilité et l'équité doivent être considérées conjointement lors du développement de systèmes d'intelligence artificielle destinés à des domaines sensibles.

---

# Veille technologique et algorithmique

L'essor des modèles d'intelligence artificielle dans des domaines sensibles soulève une problématique essentielle : **comment expliquer et justifier les décisions prises par un algorithme ?**

Au cours des dernières années, plusieurs méthodes d'explicabilité ont été développées afin de rendre les modèles de Machine Learning plus transparents. Parmi les plus utilisées figurent **SHAP**, **LIME**, **SAGE** et **DiCE**.

Cette veille présente le principe de chacune de ces approches ainsi que leurs principaux avantages et limites.

---

## SHAP (SHapley Additive exPlanations)

SHAP est une méthode d'explicabilité fondée sur la théorie des jeux coopératifs et plus précisément sur les **valeurs de Shapley**, proposées par Lloyd Shapley en 1953.

L'idée consiste à considérer chaque variable comme un "joueur" participant à une prédiction. La contribution de chaque variable est calculée en estimant son influence moyenne sur le résultat final parmi toutes les combinaisons possibles de variables.

SHAP permet ainsi de répondre à la question :

> **Pourquoi le modèle a-t-il produit cette prédiction ?**

### Avantages

- explications locales et globales ;
- forte base théorique ;
- résultats cohérents ;
- comparaison des variables facilitée ;
- compatible avec de nombreux modèles de Machine Learning.

### Limites

- coût computationnel parfois élevé ;
- sensibilité aux variables fortement corrélées ;
- choix de la valeur de référence (baseline) pouvant influencer les explications ;
- interprétation parfois complexe pour des utilisateurs non spécialistes.

Dans ce projet, SHAP a été utilisé pour expliquer les décisions du modèle **Random Forest**, identifier les variables les plus influentes et analyser les facteurs contribuant aux biais observés.

---

## LIME (Local Interpretable Model-agnostic Explanations)

LIME est une méthode d'explicabilité locale développée en 2016.

Contrairement à SHAP, LIME ne cherche pas à expliquer l'ensemble du modèle mais uniquement une prédiction particulière.

Pour cela, la méthode génère de nombreuses observations artificielles autour de l'individu étudié puis entraîne un modèle linéaire simple permettant d'approcher localement le comportement du modèle complexe.

### Avantages

- rapide à mettre en œuvre ;
- indépendant du modèle utilisé ;
- particulièrement adapté aux explications individuelles.

### Limites

- résultats pouvant varier d'une exécution à l'autre ;
- forte dépendance au voisinage choisi ;
- absence de garantie de cohérence globale ;
- moins robuste que SHAP.

---

## SAGE (Shapley Additive Global Explanations)

SAGE est une extension des valeurs de Shapley appliquée à l'évaluation globale des variables.

Contrairement à SHAP qui explique principalement les prédictions, SAGE mesure directement la contribution de chaque variable aux performances globales du modèle.

Il estime la perte de performance provoquée par la suppression d'une variable.

### Avantages

- mesure globale de l'importance des variables ;
- applicable à différents modèles ;
- interprétation plus stable à l'échelle du modèle.

### Limites

- coût computationnel important ;
- calcul plus long que SHAP ;
- moins adapté à l'explication individuelle.

---

## DiCE (Diverse Counterfactual Explanations)

DiCE adopte une approche totalement différente.

Au lieu d'expliquer pourquoi une décision a été prise, il cherche à répondre à la question :

> **Que faudrait-il modifier pour obtenir une autre décision ?**

Ces explications sont appelées **explications contrefactuelles**.

Par exemple, un individu classé comme présentant un risque élevé de récidive pourrait obtenir une prédiction différente si certaines caractéristiques étaient modifiées.

### Avantages

- très intuitif ;
- particulièrement adapté à l'aide à la décision ;
- facilite la compréhension des décisions algorithmiques.

### Limites

- certaines recommandations peuvent être irréalistes ;
- nécessité de définir quelles variables sont modifiables ;
- coût de calcul parfois élevé.

---

## Comparaison des principales méthodes d'explicabilité

| Méthode | Explication locale | Explication globale | Indépendante du modèle | Explications contrefactuelles |
|---------|:------------------:|:-------------------:|:----------------------:|:-----------------------------:|
| SHAP | ✅ | ✅ | ✅ | ❌ |
| LIME | ✅ | ❌ | ✅ | ❌ |
| SAGE | ❌ | ✅ | ✅ | ❌ |
| DiCE | ✅ | ❌ | ✅ | ✅ |

---

## Pourquoi SHAP a été retenu dans ce projet ?

Plusieurs méthodes auraient pu être utilisées.

Le choix s'est porté sur **SHAP** pour plusieurs raisons :

- il fournit des explications à la fois **locales** et **globales** ;
- il repose sur une base théorique solide (valeurs de Shapley) ;
- il est particulièrement adapté aux modèles d'ensemble tels que le **Random Forest** ;
- il permet d'identifier les variables influençant les décisions individuelles ainsi que le comportement global du modèle ;
- il est aujourd'hui considéré comme l'une des références en matière d'explicabilité des modèles de Machine Learning.

---

# Limites des méthodes d'explicabilité

Les méthodes d'explicabilité constituent aujourd'hui des outils indispensables pour comprendre le fonctionnement des modèles de Machine Learning. Toutefois, aucune d'entre elles n'est parfaite et chacune présente des limites qu'il convient de connaître avant leur utilisation dans des applications sensibles.

## Limites de SHAP

Bien que SHAP soit considéré comme l'une des méthodes les plus robustes d'explicabilité, plusieurs limites doivent être prises en compte.

### Sensibilité à la baseline

Les explications produites par SHAP dépendent d'une **valeur de référence (baseline)** servant de point de comparaison.

Selon le choix de cette baseline, les contributions attribuées aux variables peuvent varier. Deux analyses réalisées avec des références différentes peuvent ainsi conduire à des interprétations légèrement différentes.

### Coût computationnel

Le calcul exact des valeurs de Shapley nécessite d'évaluer toutes les combinaisons possibles de variables.

Ce coût augmente de manière exponentielle avec le nombre de variables. En pratique, SHAP utilise donc des approximations (comme TreeSHAP pour les modèles à base d'arbres) afin de réduire considérablement le temps de calcul.

### Variables corrélées

Lorsque plusieurs variables sont fortement corrélées, SHAP peut répartir leur importance entre elles.

Il devient alors difficile d'identifier laquelle est réellement responsable d'une décision, car les contributions sont partagées entre des variables apportant une information similaire.

### Interprétation

Les graphiques SHAP restent relativement techniques et nécessitent une bonne compréhension des modèles de Machine Learning. Ils peuvent donc être difficiles à interpréter pour des utilisateurs non spécialistes.

---

## Limites de LIME

LIME présente plusieurs limites importantes.

- Les explications sont uniquement locales.
- Les résultats peuvent varier entre deux exécutions.
- La qualité de l'explication dépend fortement des données artificielles générées autour de l'observation étudiée.
- Il n'existe pas de garantie que les explications locales reflètent fidèlement le comportement global du modèle.

---

## Limites de SAGE

SAGE fournit une excellente mesure globale de l'importance des variables mais présente également plusieurs inconvénients.

- Temps de calcul élevé.
- Difficulté d'interprétation au niveau individuel.
- Peu adapté lorsque l'objectif est d'expliquer une décision précise concernant un individu.

---

## Limites de DiCE

Les explications contrefactuelles proposées par DiCE sont particulièrement intuitives mais présentent plusieurs difficultés.

- Certaines modifications proposées peuvent être irréalistes dans la pratique.
- Toutes les variables ne sont pas modifiables (âge, origine, sexe...).
- La génération de contrefactuels plausibles peut devenir coûteuse pour des modèles complexes.

---

## Synthèse

Aucune méthode d'explicabilité ne constitue une solution universelle.

Le choix dépend principalement :

- du type de modèle utilisé ;
- du niveau d'explication souhaité (locale ou globale) ;
- des contraintes de calcul ;
- du public auquel les explications sont destinées.

Dans ce projet, SHAP représente le meilleur compromis entre rigueur théorique, qualité des explications et compatibilité avec le modèle Random Forest.

---

# L'IA explicable et l'AI Act européen

## L'AI Act : vers une intelligence artificielle responsable

L'**AI Act** est le premier règlement européen encadrant le développement et l'utilisation des systèmes d'intelligence artificielle.

Adopté par l'Union européenne en 2024, il vise à garantir que les systèmes d'IA soient :

- sûrs ;
- transparents ;
- explicables ;
- respectueux des droits fondamentaux ;
- non discriminatoires.

Le règlement classe les systèmes d'IA selon leur niveau de risque.

Les applications utilisées dans :

- la justice ;
- la santé ;
- l'éducation ;
- le recrutement ;
- les services publics,

sont considérées comme des **systèmes d'IA à haut risque**.

Le système COMPAS étudié dans ce projet appartient clairement à cette catégorie puisqu'il peut influencer des décisions judiciaires concernant des individus.

---

## L'article 13 : l'obligation de transparence

L'article 13 de l'AI Act impose que les systèmes d'IA à haut risque soient suffisamment transparents pour permettre à leurs utilisateurs de comprendre leur fonctionnement.

Les développeurs doivent notamment fournir :

- une description du fonctionnement du système ;
- les caractéristiques principales influençant les décisions ;
- les limites du modèle ;
- les performances attendues ;
- les risques connus ;
- les conditions d'utilisation appropriées.

L'objectif n'est pas d'expliquer l'ensemble du code source mais de permettre aux utilisateurs humains de comprendre les décisions produites par le système.

---

## Apport des méthodes d'explicabilité

Les méthodes d'explicabilité répondent directement à plusieurs exigences de l'AI Act.

Par exemple :

- SHAP identifie les variables ayant le plus contribué à une décision ;
- LIME explique localement une prédiction particulière ;
- SAGE mesure l'importance globale des variables ;
- DiCE propose des scénarios contrefactuels permettant de comprendre comment une décision pourrait être différente.

Ces outils améliorent considérablement la transparence des modèles de Machine Learning.

---

## Les limites de ces approches

Même si ces méthodes constituent une avancée importante, elles ne suffisent pas à elles seules à satisfaire toutes les exigences de l'AI Act.

En effet :

- une explication n'élimine pas les biais du modèle ;
- un modèle explicable peut rester discriminatoire ;
- certaines explications restent difficiles à comprendre pour un utilisateur non spécialiste ;
- les méthodes d'explicabilité n'offrent aucune garantie juridique sur l'équité des décisions.

L'explicabilité constitue donc un outil d'aide à la transparence, mais ne remplace pas les procédures d'audit, de gouvernance des données et de contrôle humain exigées par le règlement européen.

---

## Application au projet

Dans ce projet, l'utilisation de SHAP a permis :

- d'identifier les variables les plus influentes du modèle Random Forest ;
- d'expliquer les décisions individuelles ;
- de comprendre les facteurs pouvant contribuer aux biais observés.

L'analyse d'équité a ensuite complété cette approche en évaluant les performances du modèle selon plusieurs groupes démographiques.

Cette combinaison entre explicabilité et mesure de l'équité illustre la démarche recommandée par l'AI Act pour les systèmes d'intelligence artificielle à haut risque.

--- 

# Conclusion

Ce projet a permis de mettre en œuvre l'ensemble des étapes d'un projet complet de Data Science, depuis l'exploration des données jusqu'à l'évaluation de l'équité d'un modèle de Machine Learning.

Après une phase de nettoyage, d'analyse exploratoire et de clustering, plusieurs modèles de classification ont été entraînés et comparés. Le modèle **Random Forest** a obtenu les meilleures performances et a été retenu pour les analyses d'explicabilité et d'équité.

L'utilisation de **SHAP** a permis de mieux comprendre les décisions du modèle en identifiant les variables les plus influentes. Les analyses réalisées montrent que les performances globales d'un modèle ne suffisent pas à garantir son utilisation dans un contexte sensible.

L'étude des biais a mis en évidence des différences entre certains groupes de population. Une stratégie de mitigation a permis de réduire une partie de ces écarts, tout en illustrant le compromis classique entre performance prédictive et équité.

Enfin, la veille technologique réalisée autour de SHAP, LIME, SAGE, DiCE, du théorème de Chouldechova et de l'AI Act européen montre que l'intelligence artificielle responsable ne repose pas uniquement sur la performance des modèles, mais également sur leur transparence, leur explicabilité et leur capacité à limiter les discriminations.

Ce projet illustre ainsi l'importance d'intégrer des méthodes d'explicabilité et d'évaluation de l'équité dans le développement des systèmes de Machine Learning destinés à des domaines à fort impact sociétal.

---

# Références

## Jeu de données

- ProPublica. *Machine Bias*. 2016.
- COMPAS Dataset – ProPublica.

## Articles scientifiques

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS.

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME).* KDD.

- Covert, I., Lundberg, S., & Lee, S.-I. (2020). *Understanding Global Feature Contributions with Additive Importance Measures (SAGE).*

- Mothilal, R., Sharma, A., & Tan, C. (2020). *Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations (DiCE).* FAT* Conference.

- Chouldechova, A. (2017). *Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments.*

## Réglementation

- Parlement européen. *Artificial Intelligence Act (AI Act).* 2024.

## Documentation

- SHAP Documentation
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
