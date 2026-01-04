# PROJET : Résolution Parallèle des Règles de Golomb
## Cahier des Charges pour Développement Itératif

---

## 🎯 OBJECTIF GLOBAL

Développer un solveur de règles de Golomb optimales en suivant une démarche itérative :
1. **Phase Séquentielle** : Plusieurs versions, tests, améliorations progressives
2. **Phase Parallèle** : MPI avec communication hypercube
3. **Analyse** : Graphiques comparatifs, benchmarks, documentation

**IMPORTANT** : À chaque étape, générer des résultats mesurables et des visualisations.

---

## 📋 RAPPEL : QU'EST-CE QU'UNE RÈGLE DE GOLOMB ?

Une règle de Golomb d'ordre `n` est un ensemble de `n` entiers `{0, a₂, a₃, ..., aₙ}` où toutes les différences `|aᵢ - aⱼ|` sont distinctes.

**Exemple G4** : `[0, 1, 3, 6]` → distances {1, 2, 3, 5, 6} toutes différentes, longueur = 6

**Solutions optimales de référence** :
| Ordre | Longueur optimale | Exemple |
|-------|-------------------|---------|
| 4 | 6 | [0, 1, 3, 6] |
| 5 | 11 | [0, 1, 3, 7, 11] |
| 6 | 17 | [0, 1, 3, 7, 12, 17] |
| 7 | 25 | [0, 1, 3, 7, 12, 20, 25] |
| 8 | 34 | [0, 1, 3, 6, 10, 14, 26, 34] |
| 9 | 44 | [0, 1, 3, 7, 12, 20, 30, 40, 44] |
| 10 | 55 | [0, 1, 3, 7, 12, 20, 30, 40, 49, 55] |
| 11 | 72 | [0, 1, 3, 7, 12, 20, 30, 44, 58, 68, 72] |

---

## 🗂️ STRUCTURE DU PROJET

```
golomb_project/
├── src/
│   ├── sequential/
│   │   ├── v1_bruteforce.cpp       # Version 1 : Force brute
│   │   ├── v2_backtracking.cpp     # Version 2 : Backtracking
│   │   ├── v3_branch_bound.cpp     # Version 3 : Branch & Bound
│   │   ├── v4_optimized.cpp        # Version 4 : Optimisations avancées
│   │   └── v5_final_seq.cpp        # Version 5 : Séquentiel final
│   ├── parallel/
│   │   ├── v1_basic_mpi.cpp        # Version 1 : MPI basique
│   │   ├── v2_hypercube.cpp        # Version 2 : Communication hypercube
│   │   └── v3_optimized_mpi.cpp    # Version 3 : MPI optimisé
│   ├── common/
│   │   ├── golomb.hpp              # Structures communes
│   │   ├── validation.cpp          # Validation des règles
│   │   └── timing.cpp              # Mesure de temps
│   └── visualization/
│       └── generate_plots.py       # Génération des graphiques
├── tests/
│   ├── test_correctness.cpp        # Tests de validité
│   └── run_benchmarks.sh           # Script de benchmarks
├── results/
│   ├── sequential/                 # Résultats séquentiels
│   ├── parallel/                   # Résultats parallèles
│   └── plots/                      # Graphiques générés
├── docs/
│   └── journal.md                  # Journal de bord des améliorations
├── Makefile
└── README.md
```

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 : DÉVELOPPEMENT SÉQUENTIEL
# ═══════════════════════════════════════════════════════════════════

## ÉTAPE 1.1 : Version Brute Force (v1_bruteforce.cpp)

### Objectif
Créer une première version naïve qui fonctionne, même si elle est lente.

### Spécifications
- **Algorithme** : Énumérer toutes les combinaisons de n positions parmi [0, L_max]
- **Vérification** : Pour chaque combinaison, vérifier si toutes les différences sont distinctes
- **Limite** : Doit fonctionner pour G4, G5 (G6 sera probablement trop lent)

### Pseudo-code
```
fonction bruteforce(n, L_max):
    meilleure_longueur = +∞
    meilleure_regle = null
    
    pour chaque combinaison C de n éléments dans [0, L_max]:
        si C[0] == 0:  # Première marque toujours à 0
            si est_regle_valide(C):
                si C[n-1] < meilleure_longueur:
                    meilleure_longueur = C[n-1]
                    meilleure_regle = C
    
    retourner meilleure_regle

fonction est_regle_valide(regle):
    differences = ensemble vide
    pour i de 0 à n-1:
        pour j de i+1 à n-1:
            d = regle[j] - regle[i]
            si d dans differences:
                retourner faux
            ajouter d à differences
    retourner vrai
```

### Fichiers à créer
1. `src/common/golomb.hpp` - Structures de base
2. `src/sequential/v1_bruteforce.cpp` - Algorithme brute force
3. `src/common/validation.cpp` - Fonction de validation

### Tests requis
- [ ] G4 → doit trouver longueur 6
- [ ] G5 → doit trouver longueur 11
- [ ] Mesurer le temps pour G4, G5, G6

### Métriques à collecter
```
Ordre | Temps (ms) | Noeuds explorés | Solution trouvée
------|------------|-----------------|------------------
4     | ?          | ?               | [0, 1, 3, 6]
5     | ?          | ?               | [0, 1, 3, 7, 11]
6     | ?          | ?               | [0, 1, 3, 7, 12, 17]
```

### Livrable
- Code fonctionnel
- Tableau de résultats
- **Analyse** : Pourquoi c'est lent ? Quelles améliorations possibles ?

---

## ÉTAPE 1.2 : Version Backtracking (v2_backtracking.cpp)

### Objectif
Améliorer en construisant la règle marque par marque, avec abandon précoce.

### Améliorations par rapport à v1
- Construction incrémentale (pas de combinaisons pré-générées)
- Abandon dès qu'une différence est dupliquée
- Pas besoin de générer des combinaisons invalides

### Pseudo-code
```
fonction backtracking(n):
    meilleure = {longueur: +∞, regle: null}
    regle = [0]  # Première marque à 0
    differences = ensemble vide
    recherche(regle, differences, 1, meilleure)
    retourner meilleure.regle

fonction recherche(regle, differences, profondeur, meilleure):
    si profondeur == n:
        si regle[n-1] < meilleure.longueur:
            meilleure.longueur = regle[n-1]
            meilleure.regle = copie(regle)
        retourner
    
    pour pos de regle[profondeur-1]+1 à une_limite_raisonnable:
        nouvelles_diff = []
        valide = vrai
        
        pour i de 0 à profondeur-1:
            d = pos - regle[i]
            si d dans differences ou d dans nouvelles_diff:
                valide = faux
                break
            ajouter d à nouvelles_diff
        
        si valide:
            regle[profondeur] = pos
            recherche(regle, differences ∪ nouvelles_diff, profondeur+1, meilleure)
```

### Tests requis
- [ ] G4, G5, G6, G7 → vérifier les solutions optimales
- [ ] Comparer les temps avec v1

### Métriques à collecter
```
Ordre | Temps v1 | Temps v2 | Speedup | Noeuds explorés
------|----------|----------|---------|----------------
4     | ?        | ?        | ?       | ?
5     | ?        | ?        | ?       | ?
6     | ?        | ?        | ?       | ?
7     | N/A      | ?        | -       | ?
```

### Livrable
- Code fonctionnel
- Tableau comparatif v1 vs v2
- **Graphique** : Temps d'exécution v1 vs v2 pour G4-G7

---

## ÉTAPE 1.3 : Version Branch & Bound (v3_branch_bound.cpp)

### Objectif
Ajouter une borne supérieure pour élaguer les branches inutiles.

### Améliorations par rapport à v2
- **Bound** : Si on a trouvé une solution de longueur L*, abandonner toute branche où pos >= L*
- **Pruning** : Calcul d'une borne inférieure sur la longueur minimale restante

### Pseudo-code
```
fonction branch_and_bound(n):
    meilleure = {longueur: +∞, regle: null}
    regle = [0]
    differences = ensemble vide
    recherche_bb(regle, differences, 1, meilleure)
    retourner meilleure.regle

fonction recherche_bb(regle, differences, profondeur, meilleure):
    si profondeur == n:
        si regle[n-1] < meilleure.longueur:
            meilleure.longueur = regle[n-1]
            meilleure.regle = copie(regle)
        retourner
    
    # BORNE : positions restantes minimum = (n - profondeur)
    # Donc on ne peut pas dépasser meilleure.longueur - (n - profondeur - 1)
    max_pos = meilleure.longueur - (n - profondeur - 1)
    
    pour pos de regle[profondeur-1]+1 à max_pos:
        # ... même logique que backtracking
        si valide:
            recherche_bb(...)
```

### Optimisation supplémentaire : Borne initiale
- Utiliser une heuristique pour trouver une solution initiale (ex: construction gloutonne)
- Cela permet d'élaguer plus tôt

### Tests requis
- [ ] G4-G8 → solutions optimales
- [ ] G9-G10 si temps raisonnable

### Métriques à collecter
```
Ordre | Temps v2 | Temps v3 | Speedup | Noeuds v2 | Noeuds v3 | Ratio élagage
------|----------|----------|---------|-----------|-----------|-------------
5     | ?        | ?        | ?       | ?         | ?         | ?
6     | ?        | ?        | ?       | ?         | ?         | ?
7     | ?        | ?        | ?       | ?         | ?         | ?
8     | ?        | ?        | ?       | ?         | ?         | ?
```

### Livrable
- Code fonctionnel
- Tableau comparatif
- **Graphique** : Évolution du nombre de nœuds explorés v1 vs v2 vs v3

---

## ÉTAPE 1.4 : Version Optimisée (v4_optimized.cpp)

### Objectif
Appliquer toutes les optimisations connues.

### Optimisations à implémenter

#### 1. Exploitation de la symétrie
```
Si [0, a₂, ..., aₙ] est une règle, alors [0, aₙ-aₙ₋₁, ..., aₙ-a₂, aₙ] l'est aussi.
→ Imposer a₂ < aₙ - aₙ₋₁ pour diviser l'espace par 2.
En pratique : a₂ ≤ (aₙ / 2) ou imposer que la 2ème marque < dernière différence
```

#### 2. Structure de données optimisée pour les différences
```cpp
// Au lieu de std::set ou std::unordered_set
// Utiliser un bitset (accès O(1) garanti)
std::bitset<MAX_LENGTH> differences;

// Test : differences.test(d)
// Ajout : differences.set(d)
```

#### 3. Borne initiale avec heuristique
```
Heuristique gloutonne :
- Commencer avec [0, 1]
- À chaque étape, ajouter la plus petite position valide
- Utiliser cette solution comme borne initiale
```

#### 4. Propagation de contraintes (optionnel, avancé)
```
Avant de tester une position, calculer les positions qui seraient
forcément invalides et les éliminer d'avance.
```

### Tests requis
- [ ] G4-G10 → solutions optimales
- [ ] G11 si possible en temps raisonnable (<30 min)

### Métriques à collecter
```
Ordre | v3 (ms)  | v4 (ms)  | Speedup | Noeuds v4
------|----------|----------|---------|----------
7     | ?        | ?        | ?       | ?
8     | ?        | ?        | ?       | ?
9     | ?        | ?        | ?       | ?
10    | ?        | ?        | ?       | ?
```

### Livrable
- Code final séquentiel optimisé
- **Graphique comparatif** : Temps d'exécution v1 → v4 pour G4-G10
- **Graphique** : Nombre de nœuds explorés par version
- **Analyse** : Quelle optimisation a le plus d'impact ?

---

## ÉTAPE 1.5 : Analyse et Visualisation Séquentielle

### Objectif
Générer tous les graphiques et analyses pour la phase séquentielle.

### Graphiques à produire

#### 1. Évolution des temps d'exécution
```python
# Graphique : Temps (log scale) vs Ordre de Golomb
# Courbes : v1, v2, v3, v4
# Format : PNG, 1200x800
```

#### 2. Évolution du nombre de nœuds
```python
# Graphique : Nœuds explorés (log scale) vs Ordre
# Courbes : v2, v3, v4 (v1 n'explore pas de "nœuds")
```

#### 3. Speedup des optimisations
```python
# Graphique en barres : Speedup de chaque version par rapport à v1
# Pour chaque ordre G5, G6, G7, G8
```

#### 4. Répartition du temps (profiling)
```python
# Camembert ou barres empilées :
# - Temps de calcul des différences
# - Temps de vérification
# - Temps de gestion mémoire
```

### Script de visualisation
Créer `src/visualization/generate_plots.py` qui :
- Lit les fichiers CSV de résultats
- Génère tous les graphiques
- Sauvegarde dans `results/plots/`

### Livrable
- Tous les graphiques en PNG
- Fichier `results/sequential/summary.csv` avec toutes les données
- Document `docs/analyse_sequentielle.md` expliquant les résultats

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 2 : DÉVELOPPEMENT PARALLÈLE
# ═══════════════════════════════════════════════════════════════════

## ÉTAPE 2.1 : Version MPI Basique (v1_basic_mpi.cpp)

### Objectif
Première parallélisation fonctionnelle avec distribution statique.

### Principe
1. Le processus maître (P0) génère les premiers niveaux de l'arbre
2. Distribution des sous-arbres aux workers
3. Chaque worker explore son sous-arbre avec l'algo v4
4. Collecte des résultats à la fin

### Pseudo-code
```
si rank == 0:  # Maître
    sous_arbres = generer_sous_arbres(n, profondeur=2)
    distribuer(sous_arbres, aux workers)
    
    meilleure_globale = +∞
    pour chaque worker:
        recevoir(resultat)
        si resultat.longueur < meilleure_globale:
            meilleure_globale = resultat.longueur
sinon:  # Worker
    mon_sous_arbre = recevoir(du maître)
    resultat = explorer(mon_sous_arbre)
    envoyer(resultat, au maître)
```

### Génération des sous-arbres
```
Pour n=10, générer tous les préfixes de longueur 3 :
[0, 1, 3], [0, 1, 4], [0, 1, 5], ...
[0, 2, 4], [0, 2, 5], ...
Chaque préfixe définit un sous-arbre à explorer.
```

### Tests requis
- [ ] Résultat correct pour G8, G9 avec 4 processus
- [ ] Comparer temps avec version séquentielle v4

### Problèmes attendus
- **Déséquilibre de charge** : Certains sous-arbres sont beaucoup plus grands
- **Pas de mise à jour du bound** : Les workers utilisent des bornes obsolètes

### Métriques à collecter
```
Ordre | Procs | Temps seq | Temps par | Speedup | Efficacité
------|-------|-----------|-----------|---------|----------
8     | 4     | ?         | ?         | ?       | ?
9     | 4     | ?         | ?         | ?       | ?
9     | 8     | ?         | ?         | ?       | ?
```

### Livrable
- Code MPI fonctionnel
- Tableau de résultats
- **Analyse** : Pourquoi le speedup n'est pas idéal ?

---

## ÉTAPE 2.2 : Communication Hypercube (v2_hypercube.cpp)

### Objectif
Implémenter la communication du bound global via topologie hypercube.

### Principe
Quand un processus trouve une meilleure solution :
1. Il met à jour son bound local
2. Il diffuse le nouveau bound à tous via l'hypercube
3. Les autres processus mettent à jour leur bound et élaguent

### Hypercube : Rappel
```
Pour p = 2^d processus :
- Chaque processus a d voisins
- Voisin i du processus r = r XOR 2^i
- Broadcast en d = log2(p) étapes

Exemple p=8 (d=3):
P0 (000) voisins: P1 (001), P2 (010), P4 (100)
P5 (101) voisins: P4 (100), P7 (111), P1 (001)
```

### Implémentation du Broadcast Hypercube
```cpp
void hypercube_broadcast(int& value, int source, int rank, int p) {
    int d = log2(p);  // Dimension
    int virtual_rank = rank ^ source;  // Renumérotation
    
    for (int i = d - 1; i >= 0; i--) {
        int mask = 1 << i;
        int partner = virtual_rank ^ mask;
        int real_partner = partner ^ source;
        
        if (virtual_rank < mask) {
            // J'ai la donnée, j'envoie
            if (real_partner < p) {
                MPI_Send(&value, 1, MPI_INT, real_partner, TAG_BOUND, MPI_COMM_WORLD);
            }
        } else if (virtual_rank < 2 * mask) {
            // Je reçois
            MPI_Recv(&value, 1, MPI_INT, real_partner, TAG_BOUND, MPI_COMM_WORLD, &status);
        }
    }
}
```

### Communication Asynchrone
```cpp
// Dans la boucle de recherche, vérifier périodiquement
if (iterations % CHECK_INTERVAL == 0) {
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
        int new_bound;
        MPI_Recv(&new_bound, 1, MPI_INT, status.MPI_SOURCE, TAG_BOUND, MPI_COMM_WORLD, &status);
        if (new_bound < local_bound) {
            local_bound = new_bound;
            // Optionnel : propager aux voisins hypercube
        }
    }
}
```

### Tests requis
- [ ] Broadcast hypercube fonctionne (test unitaire)
- [ ] G9, G10 avec 8, 16 processus
- [ ] Comparer avec v1_basic_mpi (sans communication du bound)

### Métriques à collecter
```
Ordre | Procs | v1_basic | v2_hypercube | Speedup comm | Nb broadcasts
------|-------|----------|--------------|--------------|---------------
9     | 8     | ?        | ?            | ?            | ?
10    | 8     | ?        | ?            | ?            | ?
10    | 16    | ?        | ?            | ?            | ?
```

### Livrable
- Code avec communication hypercube
- Comparaison avec/sans communication du bound
- **Graphique** : Impact de la communication sur le speedup

---

## ÉTAPE 2.3 : Version MPI Optimisée (v3_optimized_mpi.cpp)

### Objectif
Optimiser l'équilibrage de charge et les communications.

### Optimisations possibles

#### 1. Distribution à grain plus fin
```
Au lieu de distribuer des sous-arbres de profondeur 2,
distribuer des sous-arbres de profondeur 3 ou 4.
→ Plus de tâches, meilleur équilibrage.
```

#### 2. Work stealing (avancé)
```
Quand un processus finit son travail :
1. Il demande du travail à un voisin hypercube
2. Le voisin peut partager une partie de son sous-arbre
```

#### 3. Réduction finale optimisée
```cpp
void hypercube_reduce_min(GolombRuler& local, GolombRuler& global, int rank, int p) {
    int d = log2(p);
    
    for (int i = 0; i < d; i++) {
        int partner = rank ^ (1 << i);
        
        if (rank < partner) {
            // Recevoir et comparer
            GolombRuler partner_best;
            MPI_Recv(&partner_best, ...);
            if (partner_best.length < local.length) {
                local = partner_best;
            }
        } else {
            // Envoyer et terminer
            MPI_Send(&local, ...);
            break;
        }
    }
    global = local;  // Valide sur P0
}
```

### Tests requis
- [ ] G10, G11 avec 16, 32, 64 processus
- [ ] Mesurer le déséquilibre de charge (temps max vs temps moyen)

### Métriques à collecter
```
Ordre | Procs | Temps | Speedup | Efficacité | Temps max/min worker
------|-------|-------|---------|------------|---------------------
10    | 16    | ?     | ?       | ?          | ?
10    | 32    | ?     | ?       | ?          | ?
11    | 32    | ?     | ?       | ?          | ?
11    | 64    | ?     | ?       | ?          | ?
```

### Livrable
- Code MPI final optimisé
- Analyse du déséquilibre de charge
- **Graphique** : Scalabilité (Speedup vs nombre de processus)

---

## ÉTAPE 2.4 : Benchmarks sur Romeo

### Objectif
Tests à grande échelle sur le supercalculateur.

### Configurations à tester
```
| Test | Ordre | Processus | Nœuds | Objectif |
|------|-------|-----------|-------|----------|
| T1   | 10    | 16        | 1     | Baseline |
| T2   | 10    | 64        | 4     | Scalabilité |
| T3   | 11    | 64        | 4     | Test limites |
| T4   | 11    | 256       | 16    | Grande échelle |
| T5   | 12    | 256       | 16    | Si temps permet |
```

### Script SLURM
```bash
#!/bin/bash
#SBATCH --job-name=golomb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --output=golomb_%j.out

module load openmpi

ORDER=$1
PROCS=$2

echo "=== Golomb G$ORDER avec $PROCS processus ==="
echo "Début: $(date)"

mpirun -np $PROCS ./golomb_parallel $ORDER

echo "Fin: $(date)"
```

### Métriques à collecter
- Temps total
- Speedup vs séquentiel
- Efficacité parallèle
- Temps de communication vs calcul
- Distribution des temps par worker

---

## ÉTAPE 2.5 : Analyse et Visualisation Parallèle

### Graphiques à produire

#### 1. Courbe de Speedup
```python
# X: Nombre de processus (1, 4, 8, 16, 32, 64)
# Y: Speedup
# Courbes: G9, G10, G11
# Ligne pointillée: Speedup idéal (linéaire)
```

#### 2. Efficacité parallèle
```python
# X: Nombre de processus
# Y: Efficacité (Speedup / p)
# Courbes: G9, G10, G11
```

#### 3. Déséquilibre de charge
```python
# Boxplot: Distribution des temps de calcul par worker
# Pour différentes configurations
```

#### 4. Comparaison séquentiel vs parallèle
```python
# Barres groupées: Temps pour G10, G11
# Groupes: Séquentiel, 8 procs, 16 procs, 64 procs
```

#### 5. Impact de la communication hypercube
```python
# Barres: Temps avec vs sans broadcast du bound
# Pour G10 avec 16 processus
```

---

# ═══════════════════════════════════════════════════════════════════
# PHASE 3 : DOCUMENTATION ET RAPPORT
# ═══════════════════════════════════════════════════════════════════

## Journal de Bord (docs/journal.md)

Format pour chaque étape :
```markdown
## [Date] - Étape X.Y : Nom de l'étape

### Objectif
Qu'est-ce que je voulais faire ?

### Approche initiale
Comment j'ai commencé ?

### Problèmes rencontrés
- Problème 1 : description
  - Analyse : pourquoi ça ne marchait pas
  - Solution : ce que j'ai fait pour corriger

### Résultats obtenus
- Métriques
- Observations

### Améliorations identifiées
Ce que je ferai dans l'étape suivante.
```

## Rapport Final

Structure suggérée :
1. Introduction et contexte
2. Définition du problème
3. Phase séquentielle
   - Évolution des versions
   - Analyse des performances
4. Phase parallèle
   - Stratégie de parallélisation
   - Communication hypercube
   - Résultats sur Romeo
5. Analyse comparative
6. Conclusion et perspectives

---

# ═══════════════════════════════════════════════════════════════════
# INSTRUCTIONS POUR CLAUDE CODE
# ═══════════════════════════════════════════════════════════════════

## Workflow à suivre

1. **Toujours commencer par créer la structure de dossiers**
2. **Une étape à la fois** : Ne pas passer à l'étape suivante avant d'avoir :
   - Code fonctionnel
   - Tests passés
   - Métriques collectées
3. **Sauvegarder les résultats** dans `results/` au format CSV
4. **Documenter** chaque étape dans `docs/journal.md`

## Format des fichiers de résultats

### results/sequential/vX_results.csv
```csv
order,time_ms,nodes_explored,solution,length
4,0.5,24,"[0,1,3,6]",6
5,3.2,156,"[0,1,3,7,11]",11
```

### results/parallel/benchmark_results.csv
```csv
order,procs,time_ms,speedup,efficiency,solution
10,16,5420,8.3,0.52,"[0,1,3,7,12,20,30,40,49,55]"
```

## Commandes de compilation

```bash
# Séquentiel
g++ -O3 -std=c++17 -o golomb_seq src/sequential/vX_*.cpp src/common/*.cpp

# Parallèle
mpicxx -O3 -std=c++17 -o golomb_par src/parallel/vX_*.cpp src/common/*.cpp
```

## Commandes de test

```bash
# Test séquentiel
./golomb_seq 7

# Test parallèle local
mpirun -np 4 ./golomb_par 9

# Soumission Romeo
sbatch scripts/run_parallel.slurm
```

---

# FIN DU CAHIER DES CHARGES
