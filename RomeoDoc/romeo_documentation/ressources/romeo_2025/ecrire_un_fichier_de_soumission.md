---
title: "Ecrire un fichier de soumission"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/ecrire_un_fichier_de_soumission"
scraped_at: "2026-01-04 02:40:58"
---

# Ecrire un fichier de soumission

Pour lancer un job, il convient d'écrire un script qui contient deux parties :

- Partie 1 : Les instructions Slurm sont écrites au début du fichier de soumission, et commencent par *#SBATCH*. Elles décrivent les besoins de votre job.
- Partie 2 : Les commandes à exécuter au démarrage du job.

## Partie 1[​](#partie-1 "Lien direct vers Partie 1")

La toute première ligne du fichier doit indiquer :
`#!/usr/bin/env bash`

Suite à cela il faut ajouter les commandes SBATCH lus par Slurm.
Ces options sur les mêmes que celles indiquées dans la page 'Lancer un calcul'. Voici pour rappel les options obligatoires à préciser :

| Information | Obligatoire | Option | Valeur par défaut | Description |
| --- | --- | --- | --- | --- |
| *Projet* | *Oui* | *`--account`* |  | Permet d'indiquer dans le cadre de quel projet ce calcul va être crée et compatibilisé. |
| *Durée du job* | *Oui* | *`--time`* |  | Le temps maximum que votre job prendra. Si votre job n'est pas terminé après cette durée, il sera tué automatiquement. Plus la durée demandée est longue moins le job est prioritaire. |
| *Mémoire* | *Oui* | *`--mem`* |  | La mémoire RAM maximum par serveur que votre job utilisera. Si votre job en consomme d'avantage, il sera tué automatiquement. Plus la quantité demandée est grande moins le job est prioritaire. |
| *Architecture* | *Oui* | *`--constraint`* |  | constraint="armgpu" ou constraint="x64cpu" |

Ces options dans le fichier de soumission seront à préfixer par `#SBATCH`

## Partie 2[​](#partie-2 "Lien direct vers Partie 2")

Ici vous allez ajouter les instruction de votre job.

Pour commencer, si votre job utilise des programmes disponibles via Spack, il convient d'activer la version de Spack compatible avec l'architecture sur laquelle ce job doit fonctionner.

Si votre job doit utiliser le Spack pour l'architecture X86\_64 sans GPU, ajoutez : `romeo_load_x64cpu_env`
Si votre job doit utiliser le Spack pour l'architecture Aarch64, une architecture ARM avec GPU, ajoutez : `romeo_load_armgpu_env`
Si vous n'avez pas besoin de Spack, n'ajoutez rien.
Afin d'éviter les soucis, ils conviendra de ne jamais activer ces deux commandes en même temps dans un même job.

## Exemples de fichier de soumission[​](#exemples-de-fichier-de-soumission "Lien direct vers Exemples de fichier de soumission")


- x64cpu sans OpenMPI
- armgpu sans OpenMPI
- x64cpu avec OpenMPI

Voici un exemple de fichier de soumission d'un job simple sur x64cpu :

monfichierdesoumission.txt
```
#!/usr/bin/env bash#SBATCH --account="projet0041"#SBATCH --time=1-02:30:00#SBATCH --mem=5G#SBATCH --constraint=x64cpu#SBATCH --nodes=2#SBATCH --cpus-per-task=8#SBATCH --job-name "Le nom de mon job"#SBATCH --comment "Un commentaire pour mon job"#SBATCH --error=job.%J.err#SBATCH --output=job.%J.outromeo_load_x64cpu_envspack load python@3.8.19 %aoccmkdir /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/calcul.py /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/input.data /scratch_p/$USER/$SLURM_JOBID/cd /scratch_p/$USER/$SLURM_JOBID/python3 calcul.py input.data
```

Dans cette exemple, nous retrouvons :

- Le 'shebang' (`#!/usr/bin/env bash`) indiquant a linux que ce script est écrit en Bash
- Les 4 options obligatoires décrites plus haut
  - `#SBATCH --account="projet0041"` : Pour créer un job dans le cadre du projet "projet0041".
  - `#SBATCH --time=1-02:30:00` : Pour créer un job de 1 jour 2 heures et 30 min maximum.
  - `#SBATCH --mem=5G` : Pour créer un job avec 5 Go de mémoire vive par serveur.
  - `#SBATCH --constraint=x64cpu` : Pour créer un job sur des serveurs d'architecture x86\_64.
- Les options suivantes sont optionnelles et expliquées dans la page "Lancer un calcul"
  - `#SBATCH --nodes=2` : Nous demandons 2 serveurs.
  - `#SBATCH --cpus-per-task=8` : Nous demandons 8 CPU par tache, et donc ici par serveur.
  - `#SBATCH --job-name "Le nom de mon job"`: Nous nommons notre job.
  - `#SBATCH --comment "Un commentaire pour mon job"`: Nous décrivons notre job.
  - `#SBATCH --error=job.%J.err`: Nous changeons le nom par défaut du fichier de sortie d'erreur de notre job.
  - `#SBATCH --output=job.%J.out`: Nous changeons le nom par défaut du fichier de sortie standard de notre job.
- `romeo_load_x64cpu_env`: Je charge le Spack adapté a l'environnement X86\_64 sans GPU.
- `spack load python@3.8.19 %aocc`: Je charge le module python 3.8.19 compilé avec aocc.
- `/scratch_p/$USER/$SLURM_JOBID/` : Je créée le dossier où le job va travailler.
- `cp -r $SLURM_SUBMIT_DIR/calcul.py /scratch_p/$USER/$SLURM_JOBID/`: Je copie mon script python dans un dossier portant le numéro du job comme nom, dans l'espace Scratch.
- `cp -r $SLURM_SUBMIT_DIR/input.data /scratch_p/$USER/$SLURM_JOBID/`: Je copie mon le fichier d'input pour mon script python dans un dossier portant le numéro du job comme nom, dans l'espace Scratch.
- `cd /scratch_p/$USER/$SLURM_JOBID/` je me déplace dans le dossier où j'ai copié les fichiers précédents.
- `python3 calcul.py input.data`: Lancement du script python.

Voici un exemple de fichier de soumission d'un job simple sur armgpu :

monfichierdesoumission.txt
```
#!/usr/bin/env bash#SBATCH --account="projet0041"#SBATCH --time=1-02:30:00#SBATCH --mem=5G#SBATCH --constraint=armgpu#SBATCH --nodes=2#SBATCH --cpus-per-task=8#SBATCH --gpus-per-node=4#SBATCH --job-name "Le nom de mon job"#SBATCH --comment "Un commentaire pour mon job"#SBATCH --error=job.%J.err#SBATCH --output=job.%J.outromeo_load_armgpu_envspack load python@3.11.9mkdir /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/calcul.py /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/input.data /scratch_p/$USER/$SLURM_JOBID/cd /scratch_p/$USER/$SLURM_JOBID/python3 calcul.py input.data
```

Dans cette exemple, nous retrouvons en plus de ce que nous avions dans l'exemple x64cpu :

- `#SBATCH --constraint=armgpu` : Pour créer un job sur des serveurs d'architecture ARM64.
- `#SBATCH --gpus-per-node=4` : Nous demandons 4 GPU par serveur.
- `spack load python@3.11.9` : Le python 3.8.19 compilé avec aocc n'existe pas sur l'architecture ARM64, nous changeons donc un autre Python, ici python 3.11.9.

Je créée ce fichier de soumission, ici un exemple de fichier de soumission utilisant OpenMPI :

monfichierdesoumission.txt
```
#!/usr/bin/env bash#SBATCH --account="projet0041"#SBATCH --time=1-02:30:00#SBATCH --mem=5G#SBATCH --constraint=x64cpu#SBATCH --nodes=2#SBATCH --cpus-per-task=8#SBATCH --job-name "Le nom de mon job"#SBATCH --comment "Un commentaire pour mon job"#SBATCH --error=job.%J.err#SBATCH --output=job.%J.outromeo_load_x64cpu_envspack load py-mpi4py ^python@3.8.19 ^openmpi@4.1.7  %aoccmkdir /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/calcul.py /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/input.data /scratch_p/$USER/$SLURM_JOBID/cd /scratch_p/$USER/$SLURM_JOBID/srun python3 calcul.py input.data
```

Dans cette exemple, nous retrouvons en plus de ce que nous avions dans l'exemple x64cpu :

- `spack load py-mpi4py ^python@3.8.19 ^openmpi@4.1.7`: Je charge le module python "mpi4py" adapté à python 3.8.19 et Openmpi 4.1.7. Ici Python et Openmpi sont des dépendances de mpi4py et serons donc chargés également.
- `srun python3 calcul.py input.data`: Lancement du script python, ici je le lance en mode MPI via srun. Pour plus d'explications sur le lancement de jobs MPI, vous pouvez consulter la page "Utiliser OpenMPI",

Vous pouvez ensuite créer le job et le soumettre au supercalculateur en utilisant la commande sbatch, de cette façon :

```
sbatch monfichierdesoumission.txt
```