# Projet Knapsack 0/1 - Analyse Comparative d'Algorithmes

**Équipe :** Chaabane, Arman, Bartosz, Ahmed  
**Date :** Décembre 2024  
**Cours :** Analyse d'Algorithmes

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)](LICENSE)

---

## Table des Matières

- [À Propos](#à-propos)
- [Reproduction des Résultats](#reproduction-des-résultats)
- [Installation Détaillée](#installation-détaillée)
- [Guide d'Utilisation Complet](#guide-dutilisation-complet)
- [Structure du Projet](#structure-du-projet)
- [Algorithmes Implémentés](#algorithmes-implémentés)
- [Données et Benchmarks](#données-et-benchmarks)
- [Résultats et Analyses](#résultats-et-analyses)
- [Problèmes Connus et Limitations](#problèmes-connus-et-limitations)

---

## À Propos

Ce projet présente une **analyse comparative exhaustive** de 16 algorithmes pour résoudre le problème du Knapsack 0/1. Notre objectif est de comprendre **quand et pourquoi** utiliser chaque algorithme selon le contexte (taille du problème, contraintes de temps, besoin d'optimalité).

### Caractéristiques Principales

- **16 algorithmes** implémentés de zéro (aucune bibliothèque externe pour les algos)
- **7 types de corrélation** testés pour couvrir différents cas d'usage
- **1029 résultats** de benchmarks sur 85 instances
- **13 tailles différentes** (n = 4 à 10,000 items)
- **Analyses statistiques**
- **Guide de décision pratique** pour choisir l'algorithme optimal
- **Code 100% reproductible** avec seeds fixées

### Notre Contribution

Au-delà de l'implémentation technique, nous avons créé un **guide pratique** montrant **quand et pourquoi** utiliser chaque algorithme. Chaque analyse répond à une question concrète, évitant les statistiques complexes sans utilité.

---

## Reproduction des Résultats

### Reproduction Complète

**Durée estimée :** ~30-60 minutes (selon votre machine)

```bash
# 1. Lancer Jupyter Notebook
jupyter notebook knapsack_project.ipynb

# 2. Exécuter les cellules dans l'ordre :
#    - Cellules 1-4 : Imports et structures de données
#    - Cellules 5-30 : Implémentation des 16 algorithmes
#    - Cellule 42 : Génération des benchmarks (OPTIONNEL - déjà fournis)
#    - Cellule 48 : Exécution des benchmarks (~30 min)
#    - Cellules 50-65 : Analyses et visualisations
```

**Attention Note :** L'exécution complète des benchmarks prend du temps.

---

## Installation Détaillée

### Prérequis

- **Python 3.8 ou supérieur** (testé sur 3.8, 3.9, 3.10, 3.11)
- **pip** (gestionnaire de paquets Python)
- **Jupyter Notebook** ou **JupyterLab**
- **8 GB RAM minimum** (16 GB recommandé pour générer de nouveaux benchmarks)

### Vérifier votre installation Python

```bash
python --version  # Doit afficher Python 3.8 ou supérieur
pip --version     # Doit afficher pip 20.0 ou supérieur
```

### Installation

```bash
# Bibliothèques scientifiques de base
pip install numpy
pip install pandas
pip install scipy

# Visualisation
pip install matplotlib
pip install seaborn

# Machine Learning
pip install scikit-learn

# Jupyter
pip install jupyter
pip install notebook
```

#### 3. Vérifier l'installation

```python
# Ouvrir Python et tester les imports
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
print('Toutes les dépendances sont installées correctement!')
"
```

---

## Guide d'Utilisation Complet

### Étape 1 : Comprendre la Structure du Notebook

Le notebook `knapsack_project.ipynb` est organisé en 8 sections :

```
Section 1 : Configuration et Imports
Section 2 : Structures de Données (Item, Problem, Solution)
Section 3 : Parsing des Benchmarks
Section 4 : Implémentation des 16 Algorithmes
Section 5 : Système de Benchmarking
Section 6 : Génération d'Instances de Test
Section 7 : Visualisations et Analyses
Section 8 : Optimisation des Hyperparamètres (ne fonctionne pas)
```

### Étape 2 : Exécuter les Algorithmes

#### A. Tester un algorithme sur une instance simple

```python
# Créer une instance manuelle
items = [
    Item(0, weight=10, value=60),
    Item(1, weight=20, value=100),
    Item(2, weight=30, value=120)
]
problem = Problem(items, capacity=50)

# Tester différents algorithmes
solution_dp = dynamic_programming(problem)
solution_greedy = greedy_ratio(problem)
solution_genetic = genetic_algorithm(problem, seed=42)

# Comparer les résultats
print(f"DP:      valeur={solution_dp.total_value}, temps={solution_dp.time*1000:.2f}ms")
print(f"Greedy:  valeur={solution_greedy.total_value}, temps={solution_greedy.time*1000:.2f}ms")
print(f"Genetic: valeur={solution_genetic.total_value}, temps={solution_genetic.time*1000:.2f}ms")
```

#### B. Charger et tester une instance de benchmark

```python
# Charger une instance depuis un fichier
problem = parse_benchmark_file('benchmarks/generated/uncorrelated_n100_c5000.txt')

print(f"Instance chargée: n={problem.n}, capacity={problem.capacity}")

# Tester un algorithme
solution = genetic_algorithm(
    problem,
    population_size=100,
    generations=50,
    mutation_rate=0.02,
    crossover_rate=0.8,
    seed=42
)

print(f"Valeur: {solution.total_value}")
print(f"Poids: {solution.total_weight}/{problem.capacity}")
print(f"Temps: {solution.time * 1000:.2f} ms")
print(f"Items sélectionnés: {len(solution.selected_items)}")
```

### Étape 3 : Générer de Nouveaux Benchmarks (OPTIONNEL)



### Étape 4 : Exécuter les Benchmarks Complets

```python
# ATTENTION: Ceci prend ~30-60 minutes
results_df = run_all_benchmarks()

# Les résultats sont automatiquement sauvegardés dans:
# - benchmarks/generated
# Des benchmarks avec résultats connus sont aussi présents dans benchmarks/
```


### Étape 5 : Analyser les Résultats

#### Batch d'instances

Notre générateur de benchmarks a créé **66 instances** réparties stratégiquement selon la taille et le type de corrélation :
```python
# =============================================================================
# GENERATOR OF BENCHMARK
# =============================================================================
# Types: 'uncorrelated', 'weakly_correlated', 'strongly_correlated'
# =============================================================================

# Instances de taille MOYENNE (n = 100-500)
generate_benchmarks(n=100,  capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=6)
generate_benchmarks(n=200,  capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=2)
generate_benchmarks(n=500,  capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=5)

# Instances de GRANDE taille (n = 1000-5000)
generate_benchmarks(n=1000, capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=3)
generate_benchmarks(n=2000, capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=3)
generate_benchmarks(n=5000, capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=2)

# Instance de TRÈS GRANDE taille (n = 10000)
generate_benchmarks(n=10000, capacity=1000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'])

# Exemples commentés pour d'autres configurations possibles:
# generate_benchmarks(n=100, capacity=5000, correlation='uncorrelated')
# generate_benchmarks(n=100, capacity=5000, correlation='strongly_correlated', count=5)
# generate_benchmarks(n=100, capacity=5000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'])
# generate_benchmarks(n=100, capacity=5000, correlation=['uncorrelated', 'strongly_correlated', 'weakly_correlated'], count=5)
```

#### C. Générer les visualisations du notebook

Exécutez simplement les cellules 50-65 du notebook. Elles génèrent :

- Temps d'éxecution
- Heatmap de couverture des tests
- Performance par taille
- Compris temps-qualité
- Régression prédictive
- Comparaisons statistiques
- Optimisation des hyperparamètres (ne fonctionne pas correctement voir le rapport)

---

## Structure du Projet

```
knapsack_project/
│
├── knapsack_project.ipynb          # NOTEBOOK PRINCIPAL
│
└── benchmarks/           
    ├── generated/                   # Instances générées
    │   ├── *.txt
    │   ├── *.txt
    │   ├── *.txt
    │   └── *.txt
    │
    ├── large_scale/                 # Instances grandes tailles
    ├── large_scale_optimum/         # Solutions optimales large_scale
    ├── low_dimension/               # Instances petites tailles
    └── low_dimension_optimum/       # Solutions optimales low_dimension      
```

---

## Algorithmes Implémentés

### 1. Algorithmes Exacts (Optimalité Garantie)

| Algorithme | Complexité Temps | Complexité Espace | Limite Pratique | Implémentation |
|------------|------------------|-------------------|-----------------|----------------|
| **Brute Force** | O(2^n) | O(n) | n ≤ 23 | Cellule 6 |
| **Dynamic Programming** | O(n×C) | O(n×C) | n ≤ 5000, C petit | Cellule 7 |
| **DP Top-Down** | O(n×C) | O(n×C) | n ≤ 5000 | Cellule 8 |
| **Branch and Bound** | O(2^n) ~ O(n log n) | O(n) | n ≤ 500 (variable) | Cellule 9 |

**Quand utiliser :**
- Brute Force : Petites instances (n ≤ 20), vérification
- DP : Instances moyennes (n ≤ 1000) avec capacité modérée
- B&B : Problèmes avec bonne borne supérieure

---

### 2. Algorithmes d'Approximation (Garantie Théorique)

| Algorithme | Complexité | Garantie | Implémentation |
|------------|------------|----------|----------------|
| **FPTAS (ε=0.1)** | O(n²/ε) | ≥ (1-ε)×OPT | Cellule 18 |
| **FPTAS (ε=0.05)** | O(n²/ε) | ≥ (1-ε)×OPT | Cellule 18 |
| **FPTAS Adaptive** | O(n²/ε) | ≥ (1-ε)×OPT | Cellule 19 |

** Attention Limitation connue :** Notre implémentation FPTAS a un bug de scaling qui limite n ≤ 100. Voir section [Problèmes Connus](#problèmes-connus-et-limitations).

**Quand utiliser :**
- Besoin de garantie théorique
- Instances moyennes (n ≤ 500 après correction)
- Compromis qualité/temps ajustable via ε

---

### 3. Heuristiques Gloutonnes (Ultra-Rapides)

| Algorithme | Tri Par | Complexité | Performance | Implémentation |
|------------|---------|------------|-------------|----------------|
| **Greedy Ratio** | value/weight ↓ | O(n log n) | 70-100% selon type | Cellule 10 |
| **Greedy Value** | value ↓ | O(n log n) | 60-95% | Cellule 11 |
| **Greedy Weight** | weight ↑ | O(n log n) | 50-90% | Cellule 12 |
| **Fractional** | ratio ↓ | O(n log n) | Borne supérieure | Cellule 13 |

**Quand utiliser :**
- Contrainte temps stricte (< 1 ms)
- Greedy Ratio : strongly_correlated (quasi-optimal)
- Greedy Value : uncorrelated, inverse_strongly
- Greedy Weight : Éviter sur inverse_strongly (très mauvais)

---

### 4. Métaheuristiques (Grandes Instances)

| Algorithme | Paramètres Clés | Temps | Performance | Implémentation |
|------------|-----------------|-------|-------------|----------------|
| **Genetic Algorithm** | pop=100, gen=50 | 100-500 ms | 85-98% | Cellule 14 |
| **Genetic Adaptive** | Adaptatifs | 100-500 ms | 87-99% (+ stable) | Cellule 15 |
| **Simulated Annealing** | T=1000, α=0.995 | 50-200 ms | 85-97% | Cellule 16 |
| **SA Adaptive** | Adaptatifs | 50-200 ms | 88-98% (+ stable) | Cellule 17 |
| **Randomized** | Glouton + aléa | 5-20 ms | 70-90% | Cellule 20 |

**Quand utiliser :**
- Grandes instances (n > 1000)
- Temps flexible (quelques secondes OK)
- Besoin de stabilité → versions Adaptive

---

## Données et Benchmarks

### Format des Fichiers de Benchmark

Nos fichiers `.txt` suivent ce format standard :

```
100 5000
# n capacity
# Puis n lignes avec : value weight
60 10
100 20
120 30
...
```

### Types de Corrélation Générés

Nous avons généré **4 types différents** pour tester les algorithmes dans divers contextes :

#### 1. **Uncorrelated** (Non-corrélé)
```python
weights = random(1, 100)
values = random(1, 100)  # Indépendants
```
**Usage :** Cas général, pas de structure particulière

---

#### 2. **Strongly Correlated** (Fortement corrélé)
```python
weights = random(1, 100)
values = weights  # Exactement égaux
```
**Usage :** Teste si Greedy Ratio trouve l'optimal (devrait !)

---

#### 3. **Weakly Correlated** (Faiblement corrélé)
```python
weights = random(1, 100)
values = weights + noise(-15, 15)  # Proche mais avec bruit
```
**Usage :** Teste robustesse au bruit

---

#### 4. **Similar Weights** (Poids similaires)
```python
weights = random(47, 53)  # Tous proches de 50
values = random(1, 100)   # Valeurs variées
```
**Usage :** Force Greedy Weight à être médiocre

---


##  Résultats et Analyses

### Analyse 1 : Taux de Solutions Optimales

**Question :** Quel algorithme trouve l'optimal le plus souvent ?

| Algorithme | % Optimal | Gap Moyen | Verdict |
|------------|-----------|-----------|---------|
| Dynamic Programming | 100.0% | 0.00% | ✓ OPTIMAL |
| DP Top-Down | 100.0% | 0.00% | ✓ OPTIMAL |
| Branch and Bound | 100.0% | 0.00% | ✓ OPTIMAL |
| FPTAS (ε=0.05) | 99.2% | 0.15% | ~ QUASI-OPTIMAL |
| FPTAS (ε=0.1) | 95.8% | 0.31% | ~ QUASI-OPTIMAL |
| Greedy Ratio | 78.5% | 2.34% | ○ BON |
| Genetic Adaptive | 12.3% | 4.21% | ○ APPROX |
| Simulated Annealing | 8.7% | 5.12% | ○ APPROX |

---

### Analyse 2 : Limites de Praticabilité (< 5 secondes)

**Question :** Jusqu'à quelle taille puis-je utiliser chaque algorithme ?

| Algorithme | Taille Max (n) | Temps à Max | Complexité Confirmée |
|------------|----------------|-------------|----------------------|
| Brute Force | 23 | 4.8s | ✓ O(2^n) |
| Branch and Bound | 500 | Variable | ✓ Élagage dépendant |
| Dynamic Programming | 5000 | Dépend C | ✓ O(n×C) |
| FPTAS | 100 | 2.2s (BUG) | ✗ Devrait être plus |
| Gloutons | 10000+ | < 1s | ✓ O(n log n) |
| Métaheuristiques | 10000+ | Ajustable | ✓ Scalable |

---

### Analyse 3 : Performance des Gloutons par Type

**Question :** Quel glouton choisir selon le type de problème ?

**Résultats :**

| Type de Corrélation | Meilleur Glouton | Gap | Pire Glouton | Gap |
|---------------------|------------------|-----|--------------|-----|
| **Strongly Correlated** | Greedy Ratio | 0.12% ✓ | Greedy Weight | 8.45% |
| **Uncorrelated** | Greedy Value | 3.78% | Greedy Weight | 9.21% |
| **Weakly Correlated** | Greedy Ratio | 1.89% | Greedy Weight | 7.56% |
| **Similar Weights** | Greedy Value/Ratio | 4.12% | Greedy Weight | 12.34% |

** Remarques:**
- Greedy Ratio **quasi-optimal** sur strongly_correlated (0.12% gap)
- Le **critère de tri** est crucial selon la structure des données

---

### Analyse 4 : Arbre de Décision Pratique

**Question :** Quel algorithme choisir dans mon contexte ?

```
┌─ Besoin d'OPTIMALITÉ GARANTIE ?
│
├─ OUI → Ai-je n×C < 10 millions ?
│        │
│        ├─ OUI → DYNAMIC PROGRAMMING
│        │        ✓ Optimal garanti
│        │        ✓ O(n×C) prévisible
│        │        ✗ Limité par mémoire
│        │
│        └─ NON → BRANCH AND BOUND
│                 ✓ Optimal garanti
│                 ~ Temps variable (élagage)
│                 ✗ Peut être lent
│
└─ NON → Quelle est ma CONTRAINTE principale ?
         │
         ├─ TEMPS STRICT (<1ms)
         │  │
         │  └─ Quel TYPE de problème ?
         │     ├─ strongly_correlated → GREEDY RATIO ✓ Quasi-optimal
         │     ├─ uncorrelated → GREEDY VALUE
         │     ├
         │     └─ autre → GREEDY RATIO (par défaut)
         │
         ├─ QUALITÉ IMPORTANTE (quelques secondes OK)
         │  │
         │  ├─ n < 200 → FPTAS (ε=0.05)
         │  │            ✓ Garantie (1-ε)×OPT
         │  │            ✓ Temps polynomial
         │  │
         │  └─ n ≥ 200 → MÉTAHEURISTIQUE
         │               ├─ Besoin STABILITÉ → Genetic/SA Adaptive
         │               └─ Performance max → Genetic Algorithm
         │
         └─ GRANDE INSTANCE (n > 1000)
            │
            └─ SIMULATED ANNEALING ou GENETIC ALGORITHM
               ✓ Seuls à passer l'échelle
               ~ Temps ajustable
               ~ Qualité non garantie mais bonne (85-98%)
```

**Tableau récapitulatif :**

| Critère | DP | B&B | Greedy | FPTAS | Genetic | SA |
|---------|----|----|--------|-------|---------|-----|
| Optimal | ✓ | ✓ | ✗ | ~ | ✗ | ✗ |
| Rapide (<1ms) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Scalable (n>1000) | ✗ | ✗ | ✓ | ~ | ✓ | ✓ |
| Stable | ✓ | ✓ | ✓ | ✓ | ~ | ~ |
| Mémoire OK | ✗ | ✓ | ✓ | ~ | ✓ | ✓ |

---

## Problèmes Connus

### FPTAS - Dysfonctionnement au-delà de n=100

**Symptômes observés :**
- FPTAS ne fonctionne pas pour n > 100
- Temps d'exécution anormalement élevés :
  - n=100, ε=0.05 : 2222 ms (vs 21 ms pour DP)
  - n=100, ε=0.1 : 1087 ms (vs 21 ms pour DP)
- Ratio: FPTAS est 100× plus lent que DP alors qu'il devrait être comparable !

**Cause identifiée :**

L'erreur provient de la formule de scaling dans la cellule 18 :

```python
# NOTRE CODE (INCORRECT):
K = (epsilon * v_max) / n

# Exemple: n=200, v_max=1000, ε=0.1
# K = (0.1 × 1000) / 200 = 0.5  ← K trop petit!

# Résultat:
# scaled_value = floor(500 / 0.5) = 1000  ← 2x plus grand!
# V_scaled = Σ scaled_values ≈ 200,000   ← Explosion!
# Tableau DP: n × V_scaled = 200 × 200,000 = 40M cellules
```

**Solution proposée :**

```python
# FORMULE CORRECTE:
K = max(1, (epsilon * v_max) / (2 * n))

# Ou ajuster epsilon pour grandes instances:
if n > 100:
    epsilon_adjusted = epsilon * (n / 100)
    K = max(1, (epsilon_adjusted * v_max) / n)
```

**Impact sur les résultats :**
- Heatmap de couverture : cellules FPTAS vides pour n > 100
- Graphiques de performance : FPTAS absents des grandes tailles

**Statut : **Identifié et documenté** dans le rapport (section 5.5). Non corrigé dans le code pour préserver l'authenticité des résultats présentés.

---


**"Il n'y a pas de meilleur algorithme universel - le contexte détermine le choix optimal."**
