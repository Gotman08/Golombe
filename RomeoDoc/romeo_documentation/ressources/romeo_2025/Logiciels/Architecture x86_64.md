---
title: "Architecture x86_64"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/Logiciels/Architecture x86_64"
scraped_at: "2026-01-04 02:41:13"
---

# Architecture x86\_64

Logiciels sur l'architecture x86\_64 sans GPU

Liste en cours de rédaction

## Gaussian[​](#gaussian "Lien direct vers Gaussian")

L'utilisation de Gaussian 16 est soumise à autorisation. Il n'est pour le moment disponible que sur architecture x64 sans GPU.
Pour demander un accès à Gaussian, créez un ticket sur le système de ticket ROMEO dans la catégorie 'Autre demande technique'.

**Gaussian étant un peu particulier à utiliser, il n'a pas été intégré à Spack pour le moment, cela viendra dans un second temps, cette documentation sera mise à jour en conséquence**

Pour utiliser Gaussian 16, voici un exemple de fichier de soumission Slurm :

gaussian16\_slurm\_examplefile.sh
```
#!/usr/bin/env bash#SBATCH --account="mon_code_projet"#SBATCH --time=1-02:30:00#SBATCH --mem=16G#SBATCH --constraint=x64cpu#SBATCH --nodes=1#SBATCH --cpus-per-task=16#SBATCH --job-name "GaussianTest"romeo_load_x64cpu_envspack load pgi@18.10#Je définie ou se trouve Gaussian dans mon environnementexport PATH=/apps/2025/manual_install/gaussian/g16/:$PATHexport g16root=/apps/2025/manual_install/gaussian/#Je dis a Gaussian quel dossier utiliser pour ses fichiers temporaires de travail, ici dans un dossier portant le numéro du job dans notre scratch_pexport GAUSS_SCRDIR="/scratch_p/$USER/Gaussian/$SLURM_JOB_ID"#J'ajoute Gaussian dans mon environnementsource /apps/2025/manual_install/gaussian/g16/bsd/g16.profile#Je créer le dossier de travail temporaire définie plus hautmkdir -p $GAUSS_SCRDIR#Je donne le fichier d'Inputinputfile="Test-Water.gjf"#Je lance le calcul, en redirigeant la sortie de gaussian dans un fichier dédiéecho "========================"echo "Lancement du calcul, output dans gaussian.output.$SLURM_JOB_ID.txt :"g16 < $inputfile > gaussian.output.$SLURM_JOB_ID.txt#J'affiche le contenu du dossier temporaire, normalement redevenu videecho "========================"echo "Verification de le dossier temporaire est bien vide apres le job :"ls -l $GAUSS_SCRDIR#Je supprime le contenu du dossier temporaire, ne fonctionne que si il est vide, sinon ne supprime pasecho "========================"echo "Suppression du dossier temporaire si il est vide :"rmdir $GAUSS_SCRDIR
```