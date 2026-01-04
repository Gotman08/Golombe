# Journal de Développement - Projet Golomb Ruler

## Résumé du Projet

Développement d'un solveur de règles de Golomb optimales avec:
- 5 versions séquentielles progressives
- 3 versions MPI parallèles avec communication hypercube

---

## Phase 1: Développement Séquentiel

### Étape 1.1: Version Brute Force (v1)

**Objectif**: Créer une version naïve fonctionnelle

**Approche**: Énumération de toutes les combinaisons C(L_max, n-1)

**Résultats**:
| Ordre | Temps | Nœuds | Solution |
|-------|-------|-------|----------|
| G4 | 0.04ms | 220 | [0,1,4,6] |
| G5 | 1.39ms | 7,315 | [0,1,4,9,11] |
| G6 | 83ms | 278,256 | [0,1,4,10,12,17] |
| G7 | 5,144ms | 15,890,700 | [0,1,4,10,18,23,25] |

**Analyse**: Complexité exponentielle, G8+ impraticable

---

### Étape 1.2: Backtracking (v2)

**Objectif**: Construction incrémentale avec abandon précoce

**Améliorations**: Abandon dès détection d'une différence dupliquée

**Résultats** (comparaison avec v1):
| Ordre | v1 | v2 | Speedup |
|-------|-----|-----|---------|
| G5 | 1.39ms | 0.88ms | 1.6x |
| G6 | 83ms | 23ms | 3.6x |
| G7 | 5,144ms | 916ms | 5.6x |
| G8 | N/A | 29,237ms | - |

**Analyse**: Réduction significative des nœuds explorés (~9x pour G7)

---

### Étape 1.3: Branch & Bound (v3)

**Objectif**: Élagage basé sur la borne supérieure

**Améliorations**:
- Heuristique gloutonne pour borne initiale
- Pruning si position + remaining >= best

**Résultats**:
| Ordre | v2 | v3 | Speedup | Ratio élagage |
|-------|-----|-----|---------|---------------|
| G7 | 916ms | 2.53ms | 362x | 33% |
| G8 | 29,237ms | 24ms | 1,218x | 34% |
| G9 | N/A | 254ms | - | 35% |

**Analyse**: Amélioration drastique grâce à l'élagage

---

### Étape 1.4: Optimisée (v4)

**Objectif**: Optimisations avancées

**Améliorations**:
1. **Bitset** pour différences (O(1) au lieu de O(log n))
2. **Symétrie**: marks[1] <= best/2 (divise espace par 2)
3. Allocation stack au lieu de heap

**Résultats**:
| Ordre | v3 | v4 | Speedup |
|-------|-----|-----|---------|
| G8 | 24ms | 2.4ms | 10x |
| G9 | 254ms | 19ms | 13x |
| G10 | N/A | 180ms | - |
| G11 | N/A | 3,600ms | - |

**Analyse**: Symétrie et bitset contribuent chacun ~2-3x

---

### Étape 1.5: Version Finale (v5)

**Objectif**: Production-ready avec CLI complète

**Fonctionnalités**:
- Mode benchmark (--benchmark)
- Export CSV (--csv)
- Mode verbose

**Performances finales séquentielles**:
| Ordre | Temps | Nœuds | Longueur |
|-------|-------|-------|----------|
| G4 | <0.01ms | 9 | 6 |
| G5 | <0.01ms | 66 | 11 |
| G6 | 0.02ms | 464 | 17 |
| G7 | 0.22ms | 4,341 | 25 |
| G8 | 2.54ms | 37,209 | 34 |
| G9 | 19.64ms | 303,757 | 44 |
| G10 | 177ms | 2,466,476 | 55 |
| G11 | 3,600ms | 43,152,641 | 72 |

---

## Phase 2: Développement Parallèle MPI

### Étape 2.1: MPI Basique (v1)

**Architecture**: Master/Worker avec distribution statique round-robin

**Implémentation**:
- Master génère sous-arbres (profondeur 3)
- Distribution aux workers
- Collecte des résultats

**Problèmes identifiés**:
- Déséquilibre de charge
- Pas de partage du bound entre workers

---

### Étape 2.2: Hypercube (v2)

**Objectif**: Communication du bound via topologie hypercube

**Implémentation**:
- Voisin i du processus r = r XOR 2^i
- Broadcast non-bloquant du nouveau bound
- Vérification périodique avec MPI_Iprobe

**Amélioration**: ~1.7x plus rapide que v1 grâce au partage du bound

---

### Étape 2.3: Optimisée (v3)

**Optimisations**:
- Distribution dynamique (workers demandent du travail)
- Profondeur de préfixe adaptative
- Export CSV avec speedup/efficacité

**Résultats parallèles** (vs séquentiel v5):
| Ordre | Procs | Temps | Speedup | Efficacité |
|-------|-------|-------|---------|------------|
| G8 | 4 | 1.16ms | 1.79x | 45% |
| G9 | 4 | 13.04ms | 1.48x | 37% |

**Analyse**:
- Overhead MPI significatif pour petits problèmes
- Meilleur potentiel pour G10+ sur cluster

---

## Graphiques Générés

1. `sequential_times.png` - Évolution temps v1→v5
2. `nodes_explored.png` - Réduction des nœuds
3. `speedup_vs_v1.png` - Accélération par version
4. `pruning_ratio.png` - Efficacité de l'élagage
5. `summary_table.png` - Tableau récapitulatif

---

## Conclusions

### Optimisations les plus impactantes

1. **Branch & Bound** (v3): 1000x+ d'amélioration
2. **Symétrie** (v4): 2x d'amélioration
3. **Bitset** (v4): 2-3x d'amélioration
4. **Hypercube** (MPI v2): 1.7x vs MPI basique

### Leçons apprises

- L'algorithme séquentiel bien optimisé est crucial
- La parallélisation MPI a un overhead non négligeable
- Le partage du bound améliore significativement l'élagage distribué
- Les problèmes plus grands (G10+) bénéficieraient plus du parallélisme

### Perspectives

- Tester sur Romeo avec 64+ processus pour G11-G12
- Implémenter work stealing complet
- Explorer OpenMP pour parallélisme intra-nœud
