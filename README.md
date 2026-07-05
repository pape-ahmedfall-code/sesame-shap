# Projet COMPAS : Analyse prédictive de la récidive et étude de l'équité des modèles

## Mastère Data Mining et Machine Learning
**Étudiant :** FALL Pape Ahmed  
**Année universitaire :** 2025-2026  

---

## 1. Introduction et Objectifs du Projet

Le système **COMPAS** (*Correctional Offender Management Profiling for Alternative Sanctions*) est un outil algorithmique d'aide à la décision utilisé par plusieurs tribunaux aux États-Unis pour évaluer le risque de récidive des prévenus. En 2016, une enquête célèbre de *ProPublica* a mis en lumière de potentiels biais raciaux discriminatoires au sein de cet outil.

L'objectif de ce projet est double :
1. **Modélisation prédictive :** Construire un modèle de Machine Learning performant capable de prédire le risque de récidive générale à partir des données historiques.
2. **Analyse d'Équité (*Fairness*) :** Évaluer de manière critique et quantitative l'équité du score officiel COMPAS ainsi que celle de notre propre modèle vis-à-vis des différents groupes démographiques (notamment l'origine ethnique `race` et le sexe `sex`).

Le projet suit rigoureusement la méthodologie standard du processus de **Data Mining** (CRISP-DM) : Compréhension, Préparation, Exploration, Modélisation, Évaluation et Analyse d'Équité.

---

## 2. Fondements Théoriques et Métriques d'Équité (Fairness)

Dans le cadre de l'évaluation d'un modèle d'aide à la décision de justice, la performance globale (exactitude) ne suffit pas. Il est indispensable d'introduire des concepts mathématiques permettant de mesurer l'équité algorithmique. Trois critères fondamentaux issus de la littérature sont analysés dans ce projet :

### 2.1. L'Indépendance (Parité Démographique / Statistical Parity)
* **Définition :** Un modèle satisfait la parité démographique si la probabilité d'obtenir une prédiction positive (ex: être prédit comme récidiviste) est identique quel que soit le groupe d'appartenance (groupe protégé ou non).
* **Formule :**  
  $$P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = b)$$
  *Où $\hat{Y}$ est la prédiction et $A$ est la variable sensible (ex: origine ethnique).*
* **Limite :** Elle ne prend pas en compte le comportement réel ($Y$) des individus.

### 2.2. La Séparation (Égalité des Chances / Equalized Odds)
* **Définition :** Un modèle satisfait la séparation si le taux de vrais positifs (TPR) **et** le taux de faux positifs (FPR) sont égaux pour tous les groupes. L'égalité restreinte au seul TPR est appelée **Égalité des Chances** (*Equal Opportunity*).
* **Formules :**  
  $$P(\hat{Y} = 1 \mid Y = 1, A = a) = P(\hat{Y} = 1 \mid Y = 1, A = b) \quad \text{(Taux de Vrais Positifs)}$$
  $$P(\hat{Y} = 1 \mid Y = 0, A = a) = P(\hat{Y} = 1 \mid Y = 0, A = b) \quad \text{(Taux de Faux Positifs)}$$
* **Impact :** C'est sur ce critère (notamment le déséquilibre du taux de faux positifs) que COMPAS a été historiquement critiqué.

### 2.3. La Suffisance (Calibration Réciproque / Predictive Parity)
* **Définition :** Un modèle satisfait la suffisance si la valeur prédictive positive (VPP / Précision) est égale entre les différents groupes. Autrement dit, un score de risque donné doit correspondre à la même probabilité réelle de récidive, peu importe le groupe.
* **Formule :**  
  $$P(Y = 1 \mid \hat{Y} = 1, A = a) = P(Y = 1 \mid \hat{Y} = 1, A = b)$$

> **Note Théorique Majeure (Théorème d'Impossibilité) :** La littérature mathématique démontre qu'il est impossible de satisfaire simultanément la *Séparation* (égalité des FPR/TPR) et la *Suffisance* (calibration) dès lors que les taux de base réels de récidive diffèrent entre les groupes dans les données historiques. Le concepteur du modèle doit donc faire un choix éthique et technique.

---

## 3. Exploration et Compréhension des Données (EDA)

### 3.1. Structure du Dataset
Le projet s'appuie sur la version filtrée fournie par ProPublica : `cox-violent-parsed_filt.csv`.
* **Dimensions :** 18 316 observations et 40 variables.
* **Types de variables :** Un mélange de variables numériques (âge, décomptes d'antécédents) et catégorielles (sexe, ethnie, description des infractions).

### 3.2. Analyse Critique et Définition de la Cible
Une analyse minutieuse de la variable `is_recid` a révélé la présence de la valeur `-1` pour 820 individus. 
* **Interprétation :** L'analyse croisée montre que pour toutes ces lignes, les informations de suivi de récidive (`r_charge_degree`, `r_offense_date`, etc.) sont systématiquement manquantes (`NaN`). 
* **Décision :** La valeur `-1` traduit une absence de données de suivi et non un état de non-récidive. Ces observations sont donc **exclues** pour l'entraînement des modèles afin de ne pas biaiser l'apprentissage. La cible finale devient binaire : `0` (non-récidive) et `1` (récidive).

### 3.3. Analyse des Valeurs Manquantes
Le diagnostic des données manquantes montre que :
* Les variables liées à la récidive violente (`violent_recid` à 100%, `vr_offense_date` à 92.6%) comportent un taux de valeurs manquantes très élevé.
* Les variables relatives à la première arrestation (`c_jail_in`, `c_jail_out`) affichent un taux de manquants mineur (~7%), facilement imputable ou gérable par un pipeline de prétraitement.

---

## 4. Préparation des Données et Pipeline de Prétraitement

Afin de garantir une modélisation robuste sans fuite de données (*data leakage*), un pipeline Scikit-Learn automatisé a été mis en place :

1. **Nettoyage :** Suppression des lignes où `is_recid == -1`.
2. **Sélection des Features :** Exclusion des identifiants (`id`, `name`) et des scores COMPAS officiels pour éviter le biais circulaire, et conservation des features prédictives légitimes (caractéristiques démographiques, antécédents criminels `priors_count`, degré de l'accusation actuelle, etc.).
3. **Pipeline Numérique :** Imputation des valeurs manquantes par la médiane et standardisation des variables (`StandardScaler`).
4. **Pipeline Catégoriel :** Imputation par la valeur la plus fréquente (mode) et encodage via `OneHotEncoder`.
5. **Découpage :** Division des données en ensembles d'entraînement et de test ($80\% / 20\%$) avec stratification.

---

## 5. Modélisation et Évaluation des Performances

*(Compléter cette section avec les modèles spécifiques entraînés et les scores obtenus, par exemple :)*
Plusieurs architectures de modèles ont été entraînées et évaluées :
* **Modèles testés :** Régression Logistique, Random Forest, XGBoost.
* **Métriques globales :** Évaluation sur l'ensemble de test à l'aide de l'Accuracy, du F1-Score et de l'AUC-ROC.
* **Résultats :** *(Insérer ici un résumé de vos performances : ex: XGBoost obtient la meilleure AUC-ROC avec X%...)*

---

## 6. Analyse de l'Équité et des Biais (Résultats)

L'évaluation de la Fairness compare le score officiel `decile_score` de COMPAS avec les prédictions de notre modèle personnalisé.

### 6.1. Évaluation du Modèle COMPAS (Constats)
En reproduisant l'analyse de ProPublica sur notre échantillon, nous observons :
* **Un biais sur le Taux de Faux Positifs (FPR) :** La probabilité d'être classé à tort comme "Strict / Risque Élevé" est nettement plus élevée pour les personnes afro-américaines que pour les personnes caucasiennes.
* **Un biais sur le Taux de Faux Négatifs (FNR) :** À l'inverse, les personnes caucasiennes qui récidivent réellement ont une probabilité plus élevée d'avoir été classées à tort comme "Faible Risque".

### 6.2. Évaluation de Notre Modèle et Discussion
* **Comparaison :** *(Expliquer si le modèle entraîné reproduit les biais des données historiques ou s'il parvient à atténuer certaines disparités).*
* **Arbitrage Performance/Équité :** L'analyse démontre la tension réelle entre l'optimisation de la performance globale et le respect de critères d'équité stricts (comme l'Égalité des Chances).

---

## 7. Structure du Répertoire

```text
├── data/
│   └── cox-violent-parsed_filt.csv   # Jeu de données principal filtré
├── Projet_COMPAS.ipynb               # Notebook de travail (Data Mining & ML)
└── README.md                         # Rapport et instructions du projet
