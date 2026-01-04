---
title: "Description  matérielle des ressources ROMEO"
source: "https://romeo.univ-reims.fr/documentation/ressources/description_matérielle_des_ressources"
scraped_at: "2026-01-04 02:40:52"
---

# Description matérielle des ressources ROMEO

Le Centre de Calcul Régional ROMEO héberge de nombreuses ressources ayants des objectifs et usages différents : un supercalculateur, un cluster 'Mesonet', un cluster de visualisation, un cloud calcul, ainsi que des ressources destinées à souternir ses activités de veille technologique (emulateur quantique, QGX, clusters de rendu, ...)

Voici une description des ressources principales que vous pouvez utiliser à ROMEO.

## Supercalculateur ROMEO2025[â](#supercalculateur-romeo2025 "Lien direct vers Supercalculateur ROMEO2025")

Le supercalculateur ROMEO "2025" est le fruit d'un partenariat technologique entre l'URCA et la société EVIDEN-BULL.
Il est composé de plusieurs modules, de calcul, de stockage, de veille technologique et de service.
Ce supercalculateur s'est classé **122e au classement international TOP 500** de Novembre 2024 avec 9.86 PFlops, et **2nd au Green500** avec une efficacité énergétique hors du commun de 70.912 GFlops/watts.

- **Module Calcul Accéléré**

  - 58 serveurs Grace Hopper de type ARM+GPU CG4
    - 4 processeurs GH200 Nvidia Grace Hopper Superchip par serveur
    - Une communication CPU-GPU 7 fois plus rapide que le PCIe classique
    - Une interconnexion NVSWITCH entre les Superchips.
    - 288 coeurs ARM et 1,9 To de mémoire LPDDR5 par serveur
    - 4 GPU H100 et 384 Go de mémoire HBM3 par serveur
    - Un port Infiniband ND200 à 200Gb/s par GPU.
  - interconnexion Infiniband 800 Gb/s par serveur
  - un total de 16700 coeurs, 232 GPU H100 et 133 To de mémoire
- **Module Calcul Scalaire CPU**

  - 44 serveurs de type AMD64
    - 2 processeurs AMD EPYC 9654 de 96 cÅurs.
      - Totalisant 192 cÅurs par serveur.
    - 1152 Go de mémoire (4 serveurs 'Fat' avec 1536 Go de mémoire)
    - Réseau 100Gb/s Ethernet
  - un total de 8448 coeurs CPU X86, et de 53 To de mémoire
- 4 serveurs de login de type AMD64
- **Module de stockage GPFS**

  - Stockage hiérarchique HCC et Flash
  - Total de 2,8 Po
  - 90 Gb/s en écriture et 180 Gbs en lecture
  - Sauvegarde et archivage sur bande, 3,2 Po
- **Module de calcul quantique**

  - Qaptiva 804
  - Emulation, compilation, lancement de calcul sur infrastructures matérielle
  - description précise par ailleurs (à venir)
- **Module Cloud Calcul et Données**

  - description précise par ailleurs (à venir)
- **Cluster de visualisation**

  - description précise par ailleurs (à venir)

Le supercalculateur possédant deux architectures distinctes, il possède également deux environnements logiciels distincts. Il conviendra de charger l'un ou l'autre pour accéder a l'environnement souhaité, tel que décris dans la page "Charger ses logiciels".

## Cluster Mesonet "Juliet"[â](#cluster-mesonet-juliet "Lien direct vers Cluster Mesonet \"Juliet\"")

- 3 serveurs de type AMD64
  - 1 processeur double socket AMD EPYC 7662 de 56 cÅurs par socket.
    - Totalisant 112 cÅurs par serveur.
  - 2To de mémoire.
  - 8 GPU Nvidia A100 avec 80 Go de mémoire par GPU.
  - Un interconnexion NVSWITCH entre les GPU.
- interconnexion Infiniband 400 Gb/s par serveur
- Un stockage de type NFS de 164 To partagés entre home et projets.

## Photographies des ressources[â](#photographies-des-ressources "Lien direct vers Photographies des ressources")

Vous pouvez utiliser ces photos des ressources ROMEO comme illustration si vous le souhaitez

<Photographies non disponibles - A venir>