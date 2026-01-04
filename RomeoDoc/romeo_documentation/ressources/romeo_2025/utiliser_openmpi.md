---
title: "Utiliser OpenMPI"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/utiliser_openmpi"
scraped_at: "2026-01-04 02:41:08"
---

# Utiliser OpenMPI

Le supercalculateur ROMEO est équipé de deux réseaux haut débit et faible latence de technologie Ethernet 100Gb et Infiniband pour respectivement les parties scalaire et accélérée du supercalculateur.

## Exemple de fichier MPI[​](#exemple-de-fichier-mpi "Lien direct vers Exemple de fichier MPI")

`testMpi.c`

```
#include <mpi.h>#include <stdio.h>#include <stdlib.h>#define SERVER_NODE 0#include <unistd.h>int main(int argc, char ** argv) {    int rang, world_size;    char buffer[80];    char hn[80];    MPI_Init(&argc, &argv);    MPI_Comm_rank(MPI_COMM_WORLD,&rang);    MPI_Comm_size(MPI_COMM_WORLD,&world_size);    int i = gethostname(hn, 80);    if (rang==SERVER_NODE)   {        /* code du maitre */        sprintf(buffer, "echo %s 'Master %d'",hn,rang);        system(buffer);    }  else  {        /* code de l'esclave */        sprintf(buffer, "echo %s 'Slave %d'",hn,rang);        system(buffer);    }    MPI_Finalize();    return 0;}
```

## Utiliser OpenMPI sur la partie scalaire[​](#utiliser-openmpi-sur-la-partie-scalaire "Lien direct vers Utiliser OpenMPI sur la partie scalaire")

Il y a actuellement 2 version de OpenMPI disponibles, compilées chacun avec AMD AOCC et GNU GCC.

`spack find openmpi`

```
-- linux-rhel9-zen4 / aocc@5.0.0 --------------------------------openmpi@4.1.7  openmpi@5.0.5-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------openmpi@4.1.7  openmpi@5.0.5
```

*Une version optimisée est fournie par le constructeur, mais n'est pas disponible actuellement à cause de problèmes de compatibilité avec nos environnements.*

Pour compiler avec ces versions de OpenMPI :

Charger tout d'abord l'environnement Spack dédié à l'architecture X86\_64.
Puis utilisez Spack pour charger la version souhaitée. Pensez donc à spécifier explicitement la version voulue comme dans l'exemple suivant :

```
romeo_load_x64cpu_envspack load openmpi@4.1.7 %aocc
```

Vous pouvez maintenant utiliser ce OpenMPI :

- Pour compiler :

```
mpicc testMpi.c -o testMpi.out
```

- Pour calculer, je peux l'utiliser dans un fichier de soumission. Il conviendra alors de préfixer l'exécutable de votre job utilisant par `srun`, comme cet exemple :

```
#!/usr/bin/env bash#SBATCH --account="projet0041" #Il faut changer cette valeur par un projet dont vous faites partie.#SBATCH --time=0-00:10:00#SBATCH --mem=1G#SBATCH --constraint=x64cpu#SBATCH --nodes=4#SBATCH --cpus-per-task=1#SBATCH --job-name "Le nom de mon job MPI"#SBATCH --comment "Un commentaire pour mon job"#SBATCH --error=job.%J.err#SBATCH --output=job.%J.outromeo_load_x64cpu_envspack load openmpi@4.1.7 %aoccmkdir /scratch_p/$USER/$SLURM_JOBID/cd /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/testMpi.c /scratch_p/$USER/$SLURM_JOBID/mpicc testMpi.c -o testMpi.outsrun testMpi.out
```

Sans le `srun`, Slurm et OpenMPI ne saurons pas communiquer entre eux, et l'échange de données nécessaire pour faire fonctionner votre job sur plusieurs serveurs n'aura pas lieu.

Vous trouverez alors dans le dossier où vous étiez a la soumission deux fichiers, job.XXXX.err et job.XXXX.out (où XXXX est le numéro du job). Le .err contiendra les erreurs éventuelles, et le .out la sortie de l'exécution de votre job. Dans /scratch\_p/\_votreUser\_/XXXX/ se trouverons le fichier .c, le fichier .out compilé, et pour un job qui aurait écrit des fichiers (ce n'est pas le cas) ici, les fichiers en question.

## Utiliser OpenMPI sur la partie accélérée[​](#utiliser-openmpi-sur-la-partie-accélérée "Lien direct vers Utiliser OpenMPI sur la partie accélérée")

Il y a actuellement 1 version de MPI disponibles, via NVHPC+HPCX.

`spack find nvhpc`

```
-- linux-rhel9-neoverse_v2 / gcc@11.4.1 -------------------------nvhpc@24.11==> 1 installed package
```

Pour compiler avec ces versions de MPI :

Charger tout d'abord l'environnement Spack dédié à l'architecture Aarch64.
Puis utilisez Spack pour charger la version souhaitée. Pensez donc à spécifier explicitement la version voulue comme dans l'exemple suivant :

```
romeo_load_armgpu_envspack load nvhpc@24.11
```

Vous pouvez maintenant utiliser ce MPI :

- Pour compiler :

```
mpicc testMpi.c -o testMpi.out
```

- Pour calculer, je peux l'utiliser dans un fichier de soumission. Il conviendra alors de préfixer l'exécutable de votre job utilisant par `srun`, comme cet exemple :

```
#!/usr/bin/env bash#SBATCH --account="projet0041" #Il faut changer cette valeur par un projet dont vous faites partie.#SBATCH --time=0-00:10:00#SBATCH --mem=1G#SBATCH --constraint=armgpu#SBATCH --nodes=4#SBATCH --cpus-per-task=1#SBATCH --job-name "Le nom de mon job MPI"#SBATCH --comment "Un commentaire pour mon job"#SBATCH --error=job.%J.err#SBATCH --output=job.%J.outromeo_load_armgpu_envspack load nvhpc@24.11mkdir /scratch_p/$USER/$SLURM_JOBID/cd /scratch_p/$USER/$SLURM_JOBID/cp $SLURM_SUBMIT_DIR/testMpi.c /scratch_p/$USER/$SLURM_JOBID/mpicc testMpi.c -o testMpi.outsrun testMpi.out
```

Sans le `srun`, Slurm et MPI ne saurons pas communiquer entre eux, et l'échange de données nécessaire pour faire fonctionner votre job sur plusieurs serveurs n'aura pas lieu.

Vous trouverez alors dans le dossier où vous étiez a la soumission deux fichiers, job.XXXX.err et job.XXXX.out (où XXXX est le numéro du job). Le .err contiendra les erreurs éventuelles, et le .out la sortie de l'exécution de votre job. Dans /scratch\_p/\_votreUser\_/XXXX/ se trouverons le fichier .c, le fichier .out compilé, et pour un job qui aurait écrit des fichiers (ce n'est pas le cas) ici, les fichiers en question.