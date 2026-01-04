# Mémoire - Observations et Problèmes Rencontrés

## Observation 1 : Explosion Combinatoire pour G17/G18

### Date : 4 janvier 2026

### Contexte
Lors de l'exécution des benchmarks G17 et G18 sur le cluster Romeo, nous avons observé des temps d'exécution très longs (>11 heures) sans indication de progression.

### Données Observées

| Ordre | Processus MPI | Prefix Depth | Subtrees Générés | Bound Initial | Optimal Connu |
|-------|---------------|--------------|------------------|---------------|---------------|
| G17   | 64            | 5            | 113,632,070      | 251           | 163           |
| G18   | 128           | 6            | 4,795,472,286    | 251           | 174           |

### Analyse du Problème

#### 1. Nombre de Subtrees Exponentiel
- G17 → G18 : multiplication par **42x** du nombre de subtrees
- Chaque subtree représente un préfixe partiel qui doit être complété
- Pour G17 (depth=5) : chaque subtree nécessite d'explorer les 12 marques restantes
- Équivalent à résoudre ~113M instances de sous-problèmes G12

#### 2. Bound Initial Trop Éloigné de l'Optimal
```
Bound greedy : 251
Optimal G17  : 163 (écart de 88)
Optimal G18  : 174 (écart de 77)
```

**Impact** :
- Au début de l'exécution, le bound élevé ne permet pas d'élaguer efficacement
- Beaucoup de branches inutiles sont explorées
- Le bound s'améliore progressivement mais lentement

#### 3. Absence d'Indicateur de Progression
- Le code MPI v3 n'affiche aucune progression pendant l'exécution
- Impossible de savoir :
  - Combien de subtrees ont été traités
  - Quelle est la valeur actuelle du meilleur bound
  - Estimation du temps restant

### Solutions Proposées

#### Solution 1 : Ajouter un Output de Progression
```cpp
// Dans la boucle principale du master
if (completedSubtrees % 10000 == 0) {
    double progress = (double)completedSubtrees / totalSubtrees * 100;
    std::cout << "[Progress] " << progress << "% - "
              << "Best bound: " << globalBest << " - "
              << "Subtrees: " << completedSubtrees << "/" << totalSubtrees
              << std::endl;
}
```

#### Solution 2 : Utiliser un Meilleur Bound Initial
- Utiliser les valeurs optimales connues comme bound initial
- Pour les ordres inconnus, utiliser une heuristique plus sophistiquée
- Implémenter un pré-calcul rapide pour trouver une bonne solution initiale

#### Solution 3 : Adapter la Profondeur du Prefix
```
Recommandation :
- G10-G12 : depth = 4-5
- G13-G15 : depth = 5-6
- G16-G18 : depth = 6-7
- G19+    : depth = 7-8
```

Objectif : maintenir le nombre de subtrees entre 1M et 100M pour un bon équilibre charge/overhead.

#### Solution 4 : Version Hybride MPI+OpenMP
La version hybride (v4) permet de :
- Réduire le nombre de processus MPI (moins de communication)
- Utiliser OpenMP pour paralléliser l'exploration intra-subtree
- Meilleure utilisation des ressources par nœud

### Leçons Apprises

1. **Toujours ajouter des indicateurs de progression** pour les calculs longs
2. **Tester à petite échelle** avant de lancer des jobs de plusieurs heures
3. **Documenter les bounds optimaux connus** et les utiliser comme référence
4. **Adapter les paramètres** (depth, processus) en fonction de l'ordre

### Impact sur le Projet

Cette observation justifie :
- Le développement de la version hybride MPI+OpenMP (v4)
- L'importance des optimisations algorithmiques vs. plus de ressources
- La nécessité d'un meilleur monitoring des jobs HPC

---

## Observation 2 : Benchmark Comparatif Toutes Versions (Romeo)

### Date : 4 janvier 2026

### Contexte
Benchmark complet sur le cluster Romeo comparant toutes les versions du solveur :
- Séquentielles : v1-v6
- MPI : v1, v2, v3, v5 (work stealing)
- Hybride : v4 (MPI+OpenMP)

### Configuration
- Cluster : Romeo (Université de Reims)
- Partition : short (16 tasks, 1 node)
- Architecture : x64cpu (AMD EPYC)

### Résultats Séquentiels (G7-G10)

| Order | v1 (Brute) | v2 (Back) | v3 (B&B) | v4 (Opt) | v5 (Final) | v6 (1T) | v6 (4T) |
|-------|------------|-----------|----------|----------|------------|---------|---------|
| G7    | 4845 ms    | 0.27 ms   | 0.19 ms  | 0.20 ms  | 0.20 ms    | 17 ms   | 15 ms   |
| G8    | -          | -         | 1.85 ms  | 1.85 ms  | 1.83 ms    | 20 ms   | 7 ms    |
| G9    | -          | -         | 16.8 ms  | 16.8 ms  | 16.8 ms    | 9.6 ms  | 8.9 ms  |
| G10   | -          | -         | 151 ms   | 151 ms   | 154 ms     | **24 ms** | 37 ms |

**Observation** : v6 avec 1 thread est **6x plus rapide** que v3-v5 sur G10 grâce à l'optimisation AVX2.

### Résultats MPI : v3 vs v5 (Work Stealing)

#### G9
| Procs | v3 (ms) | v5 (ms) | Ratio | Gagnant |
|-------|---------|---------|-------|---------|
| 2     | 41.6    | 76.4    | 1.8x  | v3      |
| 4     | 20.0    | 40.9    | 2.0x  | v3      |
| 8     | 20.9    | 22.9    | 1.1x  | ~égal   |
| **16**| 23.3    | **15.5**| 0.67x | **v5**  |

#### G10
| Procs | v3 (ms) | v5 (ms) | Ratio |
|-------|---------|---------|-------|
| 2     | 206     | 873     | 4.2x  |
| 4     | 82      | 475     | 5.8x  |
| 8     | 62      | 259     | 4.2x  |
| 16    | 49      | 155     | 3.2x  |

#### G11
| Procs | v3 (ms) | v5 (ms) | Ratio | Gagnant |
|-------|---------|---------|-------|---------|
| 2     | 4800    | 11692   | 2.4x  | v3      |
| 4     | 1952    | 5726    | 2.9x  | v3      |
| 8     | 1666    | 2871    | 1.7x  | v3      |
| **16**| 1547    | **1479**| 0.96x | **v5**  |

#### G12
| Procs | v3 (ms) | v5 (ms) | Ratio |
|-------|---------|---------|-------|
| 16    | 5836    | 20124   | 3.4x  |

### Résultats Hybride MPI+OpenMP (v4)

#### G10
| Configuration | Temps (ms) |
|---------------|------------|
| 2 ranks × 8 threads | 453 |
| 4 ranks × 4 threads | 220 |
| 8 ranks × 2 threads | **70** |

#### G11
| Configuration | Temps (ms) |
|---------------|------------|
| 2 ranks × 8 threads | 14117 |
| 4 ranks × 4 threads | 4219 |
| 8 ranks × 2 threads | **1905** |

### Analyse

#### Pourquoi v5 explore plus de noeuds ?
Le Work Stealing utilise une communication **asynchrone** pour les mises à jour de bornes :
- v3 : le master broadcast immédiatement chaque amélioration
- v5 : les workers reçoivent les updates lors des checks périodiques

Résultat : v5 explore 3-5x plus de noeuds car certains workers continuent avec un bound obsolète.

| Version | Noeuds explorés (G11, 16 procs) |
|---------|--------------------------------|
| v3      | ~44M                           |
| v5      | ~130M                          |

#### Quand v5 devient avantageux ?
- **16+ processus** : l'overhead du master dans v3 devient un goulot d'étranglement
- **G9 avec 16 procs** : v5 gagne (15ms vs 23ms)
- **G11 avec 16 procs** : v5 gagne légèrement (1479ms vs 1547ms)

#### Performance de v4 Hybride
La configuration **8 MPI × 2 OpenMP** est optimale :
- Réduit la communication MPI
- Bonne utilisation du cache partagé
- Comparable à v3 pure MPI

### Conclusion

| Cas d'usage | Version recommandée |
|-------------|---------------------|
| Petits problèmes (G7-G9) | v6 séquentiel (AVX2) |
| G10-G11, 2-8 procs | v3 (MPI optimized) |
| G10-G11, 16+ procs | v5 (Work Stealing) |
| Multi-noeuds cluster | v4 (Hybride MPI+OpenMP) |

### Fichier CSV des résultats
```
results/comparison_251140.log
```

---

## Observation 3 : [À compléter lors de prochaines observations]

