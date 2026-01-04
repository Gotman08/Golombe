---
title: "Lancer un calcul sur le supercalculateur"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/lancer_un_calcul"
scraped_at: "2026-01-04 02:41:06"
---

# Lancer un calcul sur le supercalculateur

Le supercalculateur ROMEO utilise ce qu'on appelle un Ordonnanceur de tâches. Ce système gère les taches qui sont exécutées sur le supercalculateur, leur priorité, ressources, consommation, etc.
Nous utilisons l'outil SLURM dont la documentation complète se trouve à cette a cette adresse : <https://slurm.schedmd.com/archive/slurm-23.11.6/man_index.html>
Vous trouverez d'autres ressources ici :

- Une rapide introduction à SLURM : <https://computing.llnl.gov/tutorials/slurm/slurm.pdf>
- Les tutoriels proposés par l'éditeur du logiciel : <https://www.schedmd.com/publications/>
- Le manuel des commandes SLURM : <https://slurm.schedmd.com/archive/slurm-23.11.6/man_index.html>

## Si vous utilisiez ROMEO 2018[â](#si-vous-utilisiez-romeo-2018 "Lien direct vers Si vous utilisiez ROMEO 2018")

Globalement cela fonctionne de la même manière, mais quelques changements sont à noter.
Par exemple :

- Il est maintenant nécessaire de préciser l'identifiant du projet pour lequel un job est lancé pour qu'il soit accepté.
- Il est maintenant nécessaire de préciser la mémoire à réserver pour un job.
- Le nombre de serveur par partition et leur répartition. Le choix de la partition est automatique si non précisée.
- Il faut préciser dans son job une commande permettant de choisir l'environnement X86\_64 ou aarch64 de Romeo 2025 pour charger l'environnement souhaité.
- Il faut préciser une option a Slurm pour faire tourner votre job sur la partie vectorielle (CPU, sans GPU, architecture x86\_64) ou accélérée (CPU, avec GPU, architecture aarch64).

## Principales commandes[â](#principales-commandes "Lien direct vers Principales commandes")

`squeue`
Cette commande permet de consulter les jobs actuellement sur le supercalculateur, qu'ils soient en attente ou en train de calculer.

Si vous souhaitez ne voir que les jobs qui vous appartiennent, vous pouvez utiliser l'option `--me`.

Par exemple :

squeue --me
```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)  279     short interact  fberini  R       0:02      4 romeo-c[023-026]
```

Nous voyons ici plusieurs informations:

- le numéro de job.
- la partition du job (expliqué plus bas).
- son nom.
- l'identifiant de la personne a qui il appartient.
- son statut.
- le temps d'exécution.
- puis la liste des serveurs sur lesquels il a été lancé.

Dans le cas d'un job qui ne s'est pas lancé :

```
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)  280     short interact  fberini PD       0:00      4 (PartitionTimeLimit)
```

Les informations suivantes changent :

- le statut du job, ici en PD (Pending), signifiant qu'il est en attente.
- le temps d'exécution reste à zéro.
- à la place de la liste des serveurs, nous trouvons l'erreur qui donne la raison de son attente.

---

`sinfo`
Cette commande affiche l'état actuel des différentes partitions (expliquées plus bas), et sur chacune d'elles combien de serveur sont actuellement utilisés complètement, partiellement, ou libres.

---

`scontrol show job 1234`
Cette commande permet d'avoir tous les détails d'un job donné, encore en file d'attente ou en train de calculer.

---

`sbatch`
Cette commande permet d'envoyer un fichier de soumission Slurm (voir chapitre [Ecrire un fichier de soumission](ressources/romeo_2025/ecrire_un_fichier_de_soumission.md)) afin de lancer un calcul sur le supercalculateur.

---

`scancel 1234`
Cette commande permet d'annuler un job en attente, ou d'arrêter un job en cours de calcul.

---

`srun`
Cette commande permet de soumettre une commande en temps que job à Slurm sans fichier de soumission. On doit donc lui passer toutes les informations obligatoires que l'on trouve dans un fichier de soumission mais sous la forme de paramètres de la commande.
Utilisé dans un fichier de soumission, elle est capable de remplacer la commande `mpirun` en configurant automatiquement celle ci avec les paramètres qu'elle obtient du job directement.

---

`salloc`
Cette commande permet de créer un calcul dit "Interactif". Cela va créer un job Slurm qui une fois passé en Running vous donne accès aux serveurs de calcul et ressources qui lui ont été alloués. Vous avez alors la possibilité de vous connecter sur ces ressources pour y exécuter des calculs de manière interactive via votre terminal.

## Le fichier de soumission[â](#le-fichier-de-soumission "Lien direct vers Le fichier de soumission")

Pour lancer un job, il convient d'écrire un script qui contient deux parties :

- Les instructions slurm sont écrites au début du fichier de soumission, et commencent par *#SBATCH*. Elles décrivent les besoins de votre job
- Les commandes à exécuter au démarrage du job
  - Il convient ensuite de soumettre le script à l'aide de la commande sbatch
    Le contenu de ce fichier sera expliqué plus en détail dans le chapitre suivant.

## Les ressources de calcul[â](#les-ressources-de-calcul "Lien direct vers Les ressources de calcul")

Pour soumettre un job de calcul, il y a un certain nombre d'informations à spécifier, afin que le supercalculateur sache ce qui est nécessaire pour votre job, elles doivent être spécifiées dans le fichier de soumission (voir chapitre suivant) :

| Information | Obligatoire | Option | Valeur par défaut | Exemple de valeur | Description |
| --- | --- | --- | --- | --- | --- |
| *Projet* | *Oui* | *`--account`* |  | R00001 | Permet d'indiquer dans le cadre de quel projet ce calcul va être crée et compatibilisé. |
| *Durée du job* | *Oui* | *`--time`* |  | 1-02:03:04 (pour un jour, deux heures, trois minutes et 4 secondes) | Le temps maximum que votre job prendra. Si votre job n'est pas terminé après cette durée, il sera tué automatiquement. Plus la durée demandée est longue moins le job est prioritaire. |
| *Mémoire* | *Oui* | *`--mem`* |  | 1G | La mémoire RAM maximum par serveur que votre job utilisera. Si votre job en consomme d'avantage, il sera tué automatiquement. Plus la quantité demandée est grande moins le job est prioritaire. |
| *Architecture* | *Oui* | *`--constraint`* |  | armgpu | constraint="armgpu" ou constraint="x64cpu" |
| Partition |  | `--partition` | Automatique |  | La partition dans laquelle le job va fonctionner |
| Nombre de serveurs |  | `--nodes` | 1 |  | Le nombre de serveurs souhaités |
| Nombre de taches |  | `--ntasks` | 1 (par serveur) |  | Le nombre de taches souhaitées (associés aux processus lourds, type MPI) |
| Nombre de coeurs |  | `--cpus-per-task` | 1 (par tache) |  | Le nombre de coeurs souhaités (associés aux processus légers, type threads OpenMP) |
| GPUs par serveur |  | `--gpus-per-node` | Aucun | 4 (pour 4 GPU par noeuds) | Permet de demander des GPU pour votre job, ce nombre de GPU sur chaque nÅud alloué (uniquement sur architecture armgpu) |
| GPUs par job |  | `--gpus` | Aucun | 4 (pour 4 GPU par job) | Permet de demander des GPU pour votre job, repartis sur les nÅuds alloués (uniquement sur architecture armgpu) |
| GPUs par tache |  | `--gpus-per-task` | Aucun | 4 (pour 4 GPU par tache) | Permet de demander des GPU pour votre job, ce nombre de GPU par tache MPI reparties sur les nÅuds alloués (uniquement sur architecture armgpu) |
| Réservation à utiliser |  | `--reservation` | Aucune |  | Si vous avez accès à une réservation, pour que votre job accède aux ressources reservées il faudra la préciser dans le job |

> Note
>
> Si vous souhaitez utiliser Spack afin d'accéder aux logiciels installés sur ROMEO, il vous faut tout d'abord lancer l'une de ces deux commandes pour charger le bon environnement :
>
> - `romeo_load_x64cpu_env` : Pour charger l'environnement x86\_64
> - `romeo_load_armgpu_env` : Pour charger l'environnement aarch64
>
> Vous pouvez ajouter ces commandes dans votre fichier de soumission pour ensuite charger des logiciels via Spack (voir [Charger ses logiciels](ressources/romeo_2025/charger_ses_logiciels.md)).
>
> Plus de détails sur ces commandes sont présents dans la partie "Charger ses logiciels" de cette documentation.

> ⚠️ **Warning**
>
> attention

Veillez à ne pas surcharger votre fichier `.bashrc` avec des commandes lourdes, et notamment ne mettez pas les commandes de chargement d'environnement ci-dessus dans votre fichier `.bashrc`.

### Partition[â](#partition "Lien direct vers Partition")

Les partitions sont une forme de "file d'attente et d'exécution".
Chaque partition ne peu accueillir que des jobs ayant une durée maximum qui leur soit adaptée.
Par exemple un job de durée maximum de 90 min pourra se placer dans la partition "long", "short", mais pas "instant"
Chaque partition influe sur la priorité des jobs qu'elle contient, plus la partition autorise des jobs longs, moins les jobs qui y sont placés seront prioritaires.

Les partition suivantes sont actuellement disponibles :

| Partition | Durée maximum | Priorité | Serveurs utilisables par la partition |
| --- | --- | --- | --- |
| instant | 1h00 | très forte | 42 ( romeo-c[001-040,101-104],romeo-a[001-058] ) |
| short | 24h00 | forte | 41 ( romeo-c[001-039,101-104],romeo-a[001-056] ) |
| long | 30 jours | faible | 24 ( romeo-c[001-020,101-104],romeo-a[001-040] ) |

Si vous n'indiquez pas de partition, celle correspondant le mieux à la durée maximum demandée pour votre job sera sélectionnée.

### Limites[â](#limites "Lien direct vers Limites")

Afin de permettre l'accès aux ressources à tous, une limite d'utilisateur en nombre de cÅurs est définie pour chaque utilisateur.

Pour interroger votre limite :

```
sacctmgr show Association where user=$USER
```

## Soumission intéractive[â](#soumission-intéractive "Lien direct vers Soumission intéractive")

> ⚠️ **Warning**
>
> attention

Cette méthode peut être utile mais n'est pas la méthode recommandée par le centre de calcul ROMEO.

De plus, un job intéractif se comportant comme n'importe quel autre job, il est possible qu'il ne se lance pas immédiatement, et qu'il s'ajoute à la file d'attente. N'hésitez pas à utiliser `scancel` pour retirerun job de la file d'attente si vous ne voulez plus qu'il se lance une fois les ressources disponibles.

Si vous souhaitez lancer des calculs "en direct" sur un serveur de calcul ROMEO, par exemple pour faire des taches lourdes en lecture et/ou écriture disque, il est possible de créer ce qu'on appelle un "Job Interactif".

Les commandes de base à utiliser sont les suivantes :

```
salloc -t 1:00:00 --account=monprojet --constraint="x64cpu" --mem=1Gsrun --pty bash
```

Cette commande va vous créer un job d'une heure, sur un serveur, la seconde va vous connecter directement sur ce serveur, où vous pourrez exécuter vos commandes dans votre terminal au fur et à mesure.
Si vous avez réservés plusieurs serveurs, la liste des serveurs alloués sera affichée après la première commande. Vous pouvez utiliser la commande `squeue --me` ou lire la variable d'environnement "SLURM\_NODELIST" pour connaitre les serveurs qui vous sont alloués, vous pouvez les utiliser et vous y connecter via la commande `srun --pty bash` qui vous connectera au premier des noeuds aloués.

Par exemple :

```
[fberini@romeo1 ~]$ salloc --nodes=3 --time=1:00:00 --account=romeoadmin --constraint="x64cpu" --mem=1Gsalloc: Granted job allocation 287salloc: Nodes romeo-c[023-024,040] are ready for job[fberini@romeo1 ~]$ srun --pty bash[fberini@romeo-c023 ~]$
```

Dans cet exemple, je demande 3 serveurs, pendant une 1 heure, avec mon projet 'romeoadmin', et 1Go de mémoire par serveur.
Slurm m'a donc crée un numéro 287, et placé en attente.
Dès que les ressources furent disponibles, potentiellement immédiatement mais ce n'est pas garantis, il m'a affiché les serveurs qui m'ont été alloués (ici les 23, 24 et 40).
Avec la seconde commande je me suis retrouvé connecté au serveur 23.