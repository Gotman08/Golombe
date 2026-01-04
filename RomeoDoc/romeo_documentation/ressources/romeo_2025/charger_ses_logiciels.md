---
title: "Charger ses logiciels"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/charger_ses_logiciels"
scraped_at: "2026-01-04 02:41:09"
---

# Charger ses logiciels

Le chargement des logiciels sur ROMEO se fait maintenant via un outil nommé "Spack". Sa documentation est disponible via ce lien : [Documentation Spack](https://spack.readthedocs.io)

## Charger l'environnement correspondant à l'architecture souhaitée[â](#charger-lenvironnement-correspondant-à-larchitecture-souhaitée "Lien direct vers Charger l'environnement correspondant à l'architecture souhaitée")

> ⚠️ **Attention**
>
> info

Spack à été mise à jour en 1.0.1 sur ROMEO. Il s'agit d'une version majeure changeant pas mal de chose. Cette documentation peut donc avoir quelques inexactitudes avec cette nouvelle version mais sera corrigée dès que possible.

> ⚠️ **Warning**
>
> attention

Par défaut, nous n'aurez accès à aucun logiciel, pas même à la commande `spack`.

Il existe sur Romeo 2025 deux architectures distinctes :

- `x86-64` : Cette architecture CPU classique est utilisée par les coeurs de calcul AMD de la partie ROMEO ne possédant pas de GPU.
- `aarch64` : Cette architecture CPU de type ARM est utilisée par les coeurs de calcul Nvidia de la partie de ROMEO possédant des GPU.

Ces deux architectures sont différentes, et la plupart des programmes compilés pour l'une, ne fonctionnent pas sur l'autre.

Nous avons donc séparés les installations logiciel en deux environnements bien distincts correspondants à ces deux architectures.

Pour charger ces environnements, vous devez utiliser la commande correspondante :

- `romeo_load_x64cpu_env` : Pour charger l'environnement x86\_64
- `romeo_load_armgpu_env` : Pour charger l'environnement aarch64

**Une fois une de ces commandes lancée**, vous aurez accès à la commande `spack` correspondant au bon environnement, avec les logiciels compatibles avec cet environnement.
Si aucune de ces commandes n'est exécutée, aucun Spack ne sera disponible.

> ⚠️ **Warning**
>
> attention

Les serveurs de login de Romeo 2025 étant basés sur la même architecture x86\_64 que les serveurs de calcul ayant des coeurs AMD, il est possible de réaliser des compilations et installations Spack sur ces derniers pour des logiciels à destination de la partie x86\_64 de Romeo 2025.
**Toutefois** pour l'environnement ARM aarch64, la compilation, installation, ou execution d'un programme de ou dans cet environnement ne peut pas être réalisée sur un noeud de login, il conviendra alors pour cela de réaliser ces opérations par l'intermédiaire d'un job sur la partition armgpu afin qu'elle s'effectue sur des CPU ayant une architecture compatibles.
Ce job peut être un job interactif, pour plus de facilité, comme décrit dans la partie Slurm ("Lancer un calcul") de cette documentation.

## Lister les logiciels disponibles sur ROMEO[â](#lister-les-logiciels-disponibles-sur-romeo "Lien direct vers Lister les logiciels disponibles sur ROMEO")

::â ï¸::
Le listage des logiciels disponibles sur ROMEO de manière générale peut être consulté depuis les noeuds de login pour les deux architectures (en chargeant leur environnement respectif), toutefois les programmes installés dans votre home ne peuvent être consultés que depuis un noeud ayant l'architecture correspondante à leur installation (donc vous ne verrez les programmes installés par vous même pour armgpu que depuis un noeud de calcul armgpu (réservé avec un salloc par exemple (voir la page de la documentation dédiée )))
:::

Vous pouvez utiliser la commande `spack find`.

Cette commande va vous lister tous les logiciels disponibles avec leurs versions.
Par exemple (de nombreux autres logiciels sont disponibles depuis la rédaction de cet exemple):

spack find
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------aocc@4.2.0                          curl@8.4.0          gmake@4.4.1     pkgconf@2.2.0berkeley-db@18.1.40                 diffutils@3.10      ncurses@6.5     readline@8.2blender@4.2.3                       gcc-runtime@11.4.1  nghttp2@1.63.0  unzip@6.0bzip2@1.0.8                         gdbm@1.23           openssl@3.3.1   zlib-ng@2.2.1ca-certificates-mozilla@2023-05-30  glibc@2.34          perl@5.40.0==> 19 installed packages
```

Cette liste contient tout ce qui est installé, regroupés par compilateurs utilisé. Cette liste inclue toutes les dépendances installées automatiquement.

---

Pour n'avoir que les logiciels installés suite à une demande explicite vous pouvez faire :

spack find --explicit
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------aocc@4.2.0  blender@4.2.3  curl@8.4.0  unzip@6.0==> 4 installed packages
```

---

Si vous souhaitez rechercher un logiciel spécifique, vous pouvez ajouter une recherche à la commande find :

spack find blender
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------blender@4.2.3==> 1 installed package
```

---

Ces logiciels peuvent être installés plusieurs fois avec une même version, mais avec des options différentes. Pour voir les differences vous pouvez utiliser l'option `--variants` :

spack find --variants --explicit
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------aocc@4.2.0~license-agreed build_system=genericblender@4.2.3 build_system=genericcurl@8.4.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=opensslunzip@6.0 build_system=makefile patches=f6f6236==> 4 installed packages
```

Cette commande affiche sur chaque ligne une version d'un logiciel et les options de cette variante.

- Le `~option` indiquent que cette option est désactivée.
- Le `+option` indiquent que cette option est activée.
- Le `option=valeur` indiquent que cette option est définie à 'valeur'.

Dans l'exemple ci dessus nous constatons donc par exemple que `curl@8.4.0` est compilé sans l'option `ldap`, mais avec l'option `nghttp2`, et avec le protocole `tls` configuré en `openssl`.

Cette facon d'indiquer les options est importante et sera utilisée plus tard pour selectionner précisement quel variante de logiciel nous souhaitons charger. Ces options peuvent également être utilisées avec la commande `spack find` pour rechercher des variants spécifiques d'un logiciel, ainsi que pour de très nombreuses commandes spack.

Par exemple :

spack find --variants curl@8.4
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------curl@8.4.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=opensslcurl@8.4.0~gssapi~ldap~libidn2~librtmp+libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl
```

Ici nous constatons que deux versions de curl sont disponibles, une avec l'option libssh, et une sans.

En utilisant spack find de cette manière nous pouvons rechercher uniquement les versions de curl ayant cette option activée :

spack find --variants curl@8.4 +libssh
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------curl@8.4.0~gssapi~ldap~libidn2~librtmp+libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl==> 1 installed package
```

> ⚠️ **Attention**
>
> info

Pour de multiples raisons que nous ne détaillerons pas ici, il peut arriver que certains logiciels soient installés plusieurs fois, dans une même version, et avec les mêmes options

Si vous souhaitez réellement utiliser toujours exactement le même variant, et non un variant identique, il est possible d'afficher le 'hash' de chaque variant et de l'utiliser pour charger cette version très spécifiquement :

spack find --long curl@8.4
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------gty4hz3 curl@8.4.0  zpngtsn curl@8.4.0==> 2 installed packages
```

Cette command va afficher un 'hash' unique devant chaque logiciel. Vous pouvez alors utiliser ce hash pour charger spécifiquement cette version:

`spack load /gty4hz3`

Si une version ou une variante d'une version d'un logiciel n'est pas instalée, vous pouvez apprendre comment l'obtenir dans le chapitre suivant, "Installer un logiciel".

---

Vous souhaitez savoir où est installé un logiciel ? Vous pouvez utiliser l'option :

spack find --paths curl
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------curl@8.4.0  /apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/curl-8.4.0-gty4hz3btysty5kulxmhgnas2tolnvzdcurl@8.4.0  /apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/curl-8.4.0-zpngtsnipryswojzkpecl3wbpda54j3e
```

Toutes ces options peuvent être mélangées en fonction de vos besoins, par exemple :

spack find --paths --variants --long curl
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------gty4hz3 curl@8.4.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl  /apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/curl-8.4.0-gty4hz3btysty5kulxmhgnas2tolnvzdzpngtsn curl@8.4.0~gssapi~ldap~libidn2~librtmp+libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl  /apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/curl-8.4.0-zpngtsnipryswojzkpecl3wbpda54j3e==> 2 installed packages
```

## Charger les logiciels disponibles sur ROMEO[â](#charger-les-logiciels-disponibles-sur-romeo "Lien direct vers Charger les logiciels disponibles sur ROMEO")

Pour charger un logiciel, il faut utiliser la commande `spack load`. Par exemple :

`spack load blender`

Toutefois si le load que vous demandez n'est pas assez explicite, Spack va vous demander d'être plus spécifique. Par exemple si plusieurs variants sont disponibles quand vous souhaitez charger le logiciel curl :

spack load curl
```
==> Error: curl matches multiple packages.  Matching packages:    gty4hz3 curl@8.4.0%gcc@11.4.1 arch=linux-rhel9-zen4    zpngtsn curl@8.4.0%gcc@11.4.1 arch=linux-rhel9-zen4    a7b3ulo curl@8.8.0%gcc@11.4.1 arch=linux-rhel9-zen4  Use a more specific spec (e.g., prepend '/' to the hash).
```

Il conviendra alors de regarder les différentes version et variants existant, avec les commandes expliquées si dessus, et de préciser à spack ce que nous souhaitons :

`spack load curl@8.4.0 +libssh`

Si la commande s'execute sans rien indiquer, c'est que tout s'est bien passé. Nous pouvons vérifier cela en listant les logiciels chargés :

spack find --loaded
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------curl@8.4.0==> 1 loaded package
```

Nous pouvons vérifier également que la commande curl maintenant disponible est bien celle demandée :

which curl
```
/apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/curl-8.4.0-zpngtsnipryswojzkpecl3wbpda54j3e/bin/curl
```

## Utiliser dans un job un logiciel disponible via Spack[â](#utiliser-dans-un-job-un-logiciel-disponible-via-spack "Lien direct vers Utiliser dans un job un logiciel disponible via Spack")

Afin que ce logiciel soit disponible dans votre job de calcul, il conviendra, tout comme avec les modules sur l'ancien supercalculateur, de préciser dans votre fichier de soumission les commandes Spack à charger pour votre job.

## Optimisation des logiciels disponibles[â](#optimisation-des-logiciels-disponibles "Lien direct vers Optimisation des logiciels disponibles")

Les logiciels compilés par ROMEO sont installés par défaut avec un niveau d'optimisation de niveau 2 (type O2), les optimisations plus agressives en "O3" pouvant parfois causer des soucis et augmentant drastiquement les tailles des exécutables obtenus.
Si vous souhaitez une optimisation différente, vous pouvez installer une version du logiciel dans votre home via spack en précisant ces options. Plus détails dans le chapitre suivant sur ce sujet.

---

Si vous avez besoin d'un logiciel et qu'il n'est pas disponible, ou pas dans une version ou variante correspondant à votre besoin, vous pouvez apprendre dans le chapitre suivant comment obtenir sur ROMEO les logiciels dont vous avez besoin.