---
title: "Espaces de stockage"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/espaces_de_stockage"
scraped_at: "2026-01-04 02:41:09"
---

# Espaces de stockage

Le supercalculateur ROMEO dispose de plusieurs espaces disques adaptés au différents besoins, ils sont décrits ci-dessous.
La bonne utilisation de ces espaces permet de garantir la disponibilité des ressources à l'ensemble de la communauté.
Il est de la responsabilité de chaque utilisateur de se conformer aux règles d'utilisations telles que décrites dans la charte afin de garantir le fonctionnement des ressources pour l'ensemble des utilisateurs.

> ⚠️ **Warning**
>
> attention

Attention : Les données placées sur ROMEO ne sont pas sauvegardées, toute suppression de votre part est définitive et immediate et sans retour possible.

Veuillez à sauvegarder vos données importantes. Pour cela vous pouvez par exemple utiliser le service de NAS Virtuel SSDS (<https://numerique.univ-reims.fr/documentation/documentation-publique/recherche/ssds/>).

De notre coté les données sont dupliquée en temps réel entre plusieurs serveurs et espaces disques, afin de preserver les données en cas de panne materielle, mais cela ne permet pas une récupération des données en cas de suppression de celle ci des espaces de stockage suite à une action.

## Les espaces de stockage[â](#les-espaces-de-stockage "Lien direct vers Les espaces de stockage")

Il y a sur ROMEO 2025 les stockages suivants :

- /apps
- /home
- /project
- /scratch\_p
  Ces espaces de stockage, contrairement à l'ancien ROMEO, sont maintenant sur une seule et même architecture rapide. Toutefois ceux ci ont été configurés et optimisés de manières différentes.

Voici une description plus en détail de ces différents stockages, et de ce qu'ils contiennent :

#### /apps[â](#apps "Lien direct vers /apps")

Cet espace de stockage vous est accessible en lecture mais pas en écriture, c'est l'espace où vous trouverez tous les logiciels, programmes, scripts, etc que le centre de Calcul ROMEO met à disposition, notamment via l'outil Spack (expliqué plus tard dans les chapitres [Charger ses logiciels](ressources/romeo_2025/charger_ses_logiciels.md) et [Installer un logiciel](ressources/romeo_2025/installer_un_logiciel.md) ).

#### /home[â](#home "Lien direct vers /home")

Cet espace de stockage contient tous les répertoires individuels home des utilisateurs.
Vous y trouverez donc votre repertoire sous la forme `/home/votrelogin`.
Ce repertoire utilise la partie simplement rapide du stockage, et est conçu pour vos données de travail actuellement non utilisées par vos jobs, mais qui sont encore utiles pour vos calculs.
Il n'est pas prévu pour du stockage à long terme de données peu ou pas utilisées.

#### /project[â](#project "Lien direct vers /project")

Cet espace contient tous les répertoires de partage des projets qui en ont fait la demande.
Vous y trouverez donc les repertoires projet sous la forme `/project/project_id`.
Ce repertoire utilise la partie simplement rapide du stockage, et est conçu pour vos données de travail actuellement non utilisées par vos jobs et partagées entre les différents utilisateurs appartement à un projet.

#### /scratch\_p[â](#scratch_p "Lien direct vers /scratch_p")

Cet espace de stockage contient tous les répertoires individuels scratch des utilisateurs.
Vous y trouverez donc votre répertoire sous la forme `/scratch_p/votrelogin`.
Ce répertoire utilise à la fois la partie simplement rapide du stockage, et la partie très rapide.
Les données qui y sont déposées sont placées dans l'espace très rapide, puis au fur et à mesure que l'espace très rapide se remplit, les données les plus anciennes seront replacées (de manière transparente) sur l'espace simplement rapide.
Cet espace est conçu pour vos données de travail utilisées par vos jobs en cours ou en file d'attente.
Il n'est pas prévu pour du stockage à long terme ni à moyen terme de données peu ou pas utilisées.

## Quotas[â](#quotas "Lien direct vers Quotas")

### Les quotas par défaut[â](#les-quotas-par-défaut "Lien direct vers Les quotas par défaut")

Il existe deux valeurs pour comprendre le système de quota :

- Le quota **Hard** : Ce quota est strict et ne peux pas être dépassé, toute opération dépassant ce quota échouera.
- Le quota **Soft** : Ce quota peut être dépassé, toute opération dépassant ce quota continue de fonctionner. Mais cela vous affichera un avertissement puis déclenchera une période dite de 'grace' de 7 jours pendant laquelle vous pourrez continuer a écrire des données jusqu'à la limite de quota Hard. Si au bout de cette période vous n'êtes pas repassés sous la limite de quota Soft, alors celle ci sera considérée comme une limite de quota Hard jusqu'à ce que vous repassiez votre utilisation de l'espace sous la limite Soft et vous ne pourrez donc plus écrire sur le stockage.

Sur Romeo, les quotas par défaut sont les suivants :

| Espace de stockage | Quota utilisateur | Quota groupe (groupe utilisateur ou groupe projet) |
| --- | --- | --- |
| /apps | Sauf exception, pas de possibilité d'écriture | Sauf exception, pas de possibilité d'écriture |
| /home | Soft: 15 Go, Hard: 20 Go | Pas de quota (pas de limite par ce moyen) |
| /project | Pas de quota (pas de limite par ce moyen) | Groupe Projet = Soft: 15 Go, Hard: 20 Go. Groupe Utilisateur = pas de possibilité d'écriture |
| /scratch\_p | Soft: 15 Go, Hard: 20 Go | Pas de quota (pas de limite par ce moyen) |

### Comment fonctionnent les quotas ? Pourquoi est ce que j'ai une erreur de quota dépassé alors que le quota du projet n'est pas atteint ?[â](#comment-fonctionnent-les-quotas--pourquoi-est-ce-que-jai-une-erreur-de-quota-dépassé-alors-que-le-quota-du-projet-nest-pas-atteint- "Lien direct vers Comment fonctionnent les quotas ? Pourquoi est ce que j'ai une erreur de quota dépassé alors que le quota du projet n'est pas atteint ?")

Tous les espaces de stockage sont limité par quota, un quota peu élevé par défaut.
Sur un système Linux, comme utilisé sur ROMEO, il y a un concept de groupe et d'utilisateur. Un utilisateur appartient à un groupe par défaut, et un fichier ou dossier appartient à un utilisateur et à un groupe de manière obligatoire.

Dans /home et /scrach\_p, le quota est lié à l'utilisateur, et non à son groupe. Tous les fichiers que vous créez en votre nom serons comptabilisés de cette façon.

Dans /projet c'est different, cet espace est configuré pour que les fichiers qui y sont placés appartiennent à leur utilisateur mais pas à son groupe personnel, ils appartiennent au groupe du projet. De cette façon quand on y dépose des fichiers.
Ils comptent dans le quota du projet et non le quota personnel, nous n'avons dont pas mis de quota utilisateur, mais avons bloqués à quelques Go les données qui peuvent être déposé avec un groupe personnel, car ils doivent être déposés comme appartenant au groupe du projet.

Si vous utilisez une méthode pour copier vos données qui ignorant nos configurations pour que les données déposées appartiennent automatiquement au groupe du projet, elles appartiennent donc a votre groupe personne qui est très bridé, et vous aurez une erreur de quota rapidement.

Si vous êtes dans cette situation, vous pouvez :

- 1 - Si le quota du groupe de projet n'est pas suffisant, faire une demande d'augmentation de quota via un ticket dédié dans la bonne catégorie en précisant l'espace dont vous avez besoin et en le justifiant.
  Une fois le changement de quota confirmé par le support ROMEO ou si le quota est déjà suffisant, vous pouvez passez à l'étape 2.
- 2 - Changer le groupe des fichiers déjà copié avec la commande "chgrp -R PROJECTCODE /home/PROJECTCODE/"
- 3 - Rendre à ces dossiers la configuration d'application automatique du groupe approprié avec la commande "chmod g+s /home/PROJECTCODE/"
- 4 - Continuer le transfert de vos données mais en faisant attention à le faire d'une façon qui n'écrase pas la configuration de projets de ROMEO (nommée setgid).

### Demander une augmentation de quota[â](#demander-une-augmentation-de-quota "Lien direct vers Demander une augmentation de quota")

Vous pouvez demander l'augmentation de vos quota sur un espace de stockage en créant un ticket ROMEO avec les infos suivantes :

- Quel utilisateur / groupe / projet
- Quel espace de stockage
- Quelle quantité de stockage supplémentaire demandée
- A quel besoin correspond cette augmentation de quota.

Le processus de traitement des demandes de quota est encore en cours de création, en attendant les règles et conditions en cours de l'augmentation d'un quota vous seront précisées dans la réponse à votre ticket.

### Consulter son quota et son utilisation des espaces de stockage[â](#consulter-son-quota-et-son-utilisation-des-espaces-de-stockage "Lien direct vers Consulter son quota et son utilisation des espaces de stockage")

La commande de base pour cela est la suivante :

```
mmlsquota gpfs
```

Pour avoir les valeurs affichée en To, Go, Mo etc vous pouvez ajouter une option. Cela donne :

```
mmlsquota --block-size auto gpfs
```

Voici un exemple de valeur que vous pouvez obtenir :

```
                         Block Limits                                               |     File LimitsFilesystem Fileset    type         blocks      quota      limit   in_doubt    grace |    files   quota    limit in_doubt    grace  Remarksgpfs       home       USR          829.4G         1T     1.465T          0     none |   363010 1000000  1500000        0     none romeo.romeo.univ-reims.frgpfs       scratch    USR          3.286G        15G        20G          0     none |       96 1000000  1500000        0     none romeo.romeo.univ-reims.fr
```

Voici les informations que vous y trouvez :

- Bloc 'Block Limits' : Affiche les quota en terme d'espace de fichiers utilisés
  - Colonnes
    - Filesystem : Le système de fichier. Ici GPFS.
    - Fileset : L'espace de stockage, home ou scratch, peut être plus.
    - type : Vous pouvez ignorer cette valeur
    - blocks : L'espace que vous utiliser sur le stockage
    - quota : Votre quota 'block' soft
    - limit : Votre quota 'block' hard
    - in\_doubt : La quantité de donnée encore a analyser pour votre compte par le système de fichier. Compte dans l'espace utilisé sur le stockage en attendant.
    - grace : Le temps de 'grace time' qu'il vous reste, ou 'none' si vous n'avez pas de grace time en cours.
- Bloc 'File Limits' : Affiche les quotas en terme de nombre de fichiers et dossiers utilisés
  - Colonnes
    - files : Le nombre de fichier que vous utiliser sur le stockage
    - quota : Votre quota 'file' soft
    - limit : Votre quota 'file' hard
    - in\_doubt : Le nombre de fichier encore a analyser pour votre compte par le système de fichier. Compte dans le lnombre de fichier utilisé sur le stockage en attendant.
    - grace : Le temps de 'grace time' qu'il vous reste, ou 'none' si vous n'avez pas de grace time en cours.
    - Remarks: Vous pouvez ignorer cette valeur.

Pour consulter le quota d'un projet, vous pouvez utiliser la commande suivante (en remplaçant R00000 par le code du projet) :

```
mmlsquota --block-size auto -g R00000 gpfs
```

## Snapshot[â](#snapshot "Lien direct vers Snapshot")

Documentation à venir - Fonction encore non disponible.

## Stockage sur Bande[â](#stockage-sur-bande "Lien direct vers Stockage sur Bande")

Stockage sur bande disponible prochainement.