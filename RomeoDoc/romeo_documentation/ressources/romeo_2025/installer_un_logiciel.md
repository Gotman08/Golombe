---
title: "Installer un logiciel"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/installer_un_logiciel"
scraped_at: "2026-01-04 02:41:07"
---

# Installer un logiciel

Sur cette version 2025 de ROMEO il est maintenant possible d'installer par vous même de nombreux logiciels dans de nombreuses versions sur le supercalculateur.

---

La méthode d'obtention d'un logiciel sur ROMEO est maintenant celle d'une installation autonome par l'utilisateur, via la méthode décrite si dessous. Quelques exceptions existent, voici un tableau les résumant :

: Tableau bientot disponible :

---

> ⚠️ **Attention**
>
> info

Spack à été mise à jour en 1.0.1 sur ROMEO. Il s'agit d'une version majeure changeant pas mal de chose. Cette documentation peut donc avoir quelques inexactitudes avec cette nouvelle version mais sera corrigée dès que possible.

Avant d'installer un logiciel vous pouvez vérifier qu'il n'est pas déjà disponible avec la commande `spack find` décrite dans le chapitre précédant.
Si vous souhaitez installer un logiciel qui sera à priori utilisé par de nombreux utilisateurs, ou que vous avez besoin que plusieurs utilisateurs utilise exactement la même installation d'un logiciel, vous pouvez faire une demande d'installation de logiciel à l'équipe technique ROMEO via le système de ticket (lien disponible sur le portail).

> ⚠️ **Warning**
>
> attention

Avant d'installer un logiciel, il conviendra de se placer sur un serveur ayant la bonne architecture (x86\_64 ou aarch64), puis de lancer la commande correspondante pour charger le bon environnement :

- `romeo_load_x64cpu_env` : Pour charger l'environnement x86\_64
- `romeo_load_armgpu_env` : Pour charger l'environnement aarch64
  Plus de détails sont disponibles dans la partie "Charger ses logiciels" de cette documentation.

## Lister les logiciels disponibles pour l'installation sur ROMEO[â](#lister-les-logiciels-disponibles-pour-linstallation-sur-romeo "Lien direct vers Lister les logiciels disponibles pour l'installation sur ROMEO")

La commande `spack list` va vous afficher la liste complète des logiciels que vous pouvez installer.

spack list
```
3dtk                                            py-flatbuffers3proxy                                          py-flatten-dict7zip                                            py-flawfinderabacus                                          py-flaxabduco                                          py-flexmockabi-compliance-checker                          py-flexxabi-dumper                                      py-flitabinit                                          py-flit-coreabseil-cpp                                      py-flit-scmabyss                                           py-flufl-lockaccfft                                          py-fluiddynacct                                            py-fluidfftaccumulo                                        py-fluidfft-builderace                                             py-fluidfft-fftwacfl                                            py-fluidfft-fftwmpiack                                             py-fluidfft-mpi-with-fftwacl                                             py-fluidfft-p3dfftacpica-tools                                    py-fluidfft-pfftacpid                                           py-fluidsimactiveharmony                                   py-fluidsim-coreactivemq                                        py-flyeacts                                            py-fn-pyacts-algebra-plugins                            py-foliumacts-dd4hep                                     py-fonttoolsactsvg                                          py-fordadditivefoam                                    py-formatizeraddrwatch                                       py-formulaicadept                                           py-fortlsadept-utils                                     py-fortran-language-server[...]
```

Vous pouvez rechercher dans cette liste, par exemple pour gromacs :

spack list gromacs
```
gromacs  gromacs-chain-coordinate  gromacs-swaxs  py-biobb-gromacs==> 4 packages
```

Vous pouvez également rechercher des mots clés directement dans la description des logiciels :

(cette commande est longue à exécuter)

spack list --search-description gromacs
```
ermod    gromacs-chain-coordinate  py-biobb-gromacs  py-mdanalysis  py-pyedrgromacs  gromacs-swaxs             py-gmxapi         py-panedr      py-simpletraj==> 10 packages
```

## Connaitre les versions et options disponibles pour un logiciel[â](#connaitre-les-versions-et-options-disponibles-pour-un-logiciel "Lien direct vers Connaitre les versions et options disponibles pour un logiciel")

Il est possible de connaitre toutes les versions et options qui sont à votre disposition pour chaque logiciels pour les installer. Par exemple pour gromacs :

spack info gromacs
```
CMakePackage:   gromacsDescription:    GROMACS is a molecular dynamics package primarily designed for    simulations of proteins, lipids and nucleic acids. It was originally    developed in the Biophysical Chemistry department of University of    Groningen, and is now maintained by contributors in universities and    research centers across the world. GROMACS is one of the fastest and    most popular software packages available and can run on CPUs as well as    GPUs. It is free, open source released under the GNU Lesser General    Public License. Before the version 4.6, GROMACS was released under the    GNU General Public License.Homepage: https://www.gromacs.orgPreferred version:    2024.3    https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gzSafe versions:    main      [git] https://gitlab.com/gromacs/gromacs.git on branch main    2024.3    https://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz    2024.2    https://ftp.gromacs.org/gromacs/gromacs-2024.2.tar.gz    2024.1    https://ftp.gromacs.org/gromacs/gromacs-2024.1.tar.gz    2024      https://ftp.gromacs.org/gromacs/gromacs-2024.tar.gz    2023.5    https://ftp.gromacs.org/gromacs/gromacs-2023.5.tar.gz    2023.4    https://ftp.gromacs.org/gromacs/gromacs-2023.4.tar.gz    2023.3    https://ftp.gromacs.org/gromacs/gromacs-2023.3.tar.gz    2023.2    https://ftp.gromacs.org/gromacs/gromacs-2023.2.tar.gz    2023.1    https://ftp.gromacs.org/gromacs/gromacs-2023.1.tar.gz    2023      https://ftp.gromacs.org/gromacs/gromacs-2023.tar.gz    2022.6    https://ftp.gromacs.org/gromacs/gromacs-2022.6.tar.gz    2022.5    https://ftp.gromacs.org/gromacs/gromacs-2022.5.tar.gz    2022.4    https://ftp.gromacs.org/gromacs/gromacs-2022.4.tar.gz    2022.3    https://ftp.gromacs.org/gromacs/gromacs-2022.3.tar.gz    2022.2    https://ftp.gromacs.org/gromacs/gromacs-2022.2.tar.gz    2022.1    https://ftp.gromacs.org/gromacs/gromacs-2022.1.tar.gz    2022      https://ftp.gromacs.org/gromacs/gromacs-2022.tar.gz    2021.7    https://ftp.gromacs.org/gromacs/gromacs-2021.7.tar.gz    2021.6    https://ftp.gromacs.org/gromacs/gromacs-2021.6.tar.gz    2021.5    https://ftp.gromacs.org/gromacs/gromacs-2021.5.tar.gz    2021.4    https://ftp.gromacs.org/gromacs/gromacs-2021.4.tar.gz    2021.3    https://ftp.gromacs.org/gromacs/gromacs-2021.3.tar.gz    2021.2    https://ftp.gromacs.org/gromacs/gromacs-2021.2.tar.gz    2021.1    https://ftp.gromacs.org/gromacs/gromacs-2021.1.tar.gz    2021      https://ftp.gromacs.org/gromacs/gromacs-2021.tar.gz    2020.7    https://ftp.gromacs.org/gromacs/gromacs-2020.7.tar.gz    2020.6    https://ftp.gromacs.org/gromacs/gromacs-2020.6.tar.gz    2020.5    https://ftp.gromacs.org/gromacs/gromacs-2020.5.tar.gz    2020.4    https://ftp.gromacs.org/gromacs/gromacs-2020.4.tar.gz    2020.3    https://ftp.gromacs.org/gromacs/gromacs-2020.3.tar.gz    2020.2    https://ftp.gromacs.org/gromacs/gromacs-2020.2.tar.gz    2020.1    https://ftp.gromacs.org/gromacs/gromacs-2020.1.tar.gz    2020      https://ftp.gromacs.org/gromacs/gromacs-2020.tar.gz    2019.6    https://ftp.gromacs.org/gromacs/gromacs-2019.6.tar.gz    2019.5    https://ftp.gromacs.org/gromacs/gromacs-2019.5.tar.gz    2019.4    https://ftp.gromacs.org/gromacs/gromacs-2019.4.tar.gz    2019.3    https://ftp.gromacs.org/gromacs/gromacs-2019.3.tar.gz    2019.2    https://ftp.gromacs.org/gromacs/gromacs-2019.2.tar.gz    2019.1    https://ftp.gromacs.org/gromacs/gromacs-2019.1.tar.gz    2019      https://ftp.gromacs.org/gromacs/gromacs-2019.tar.gz    2018.8    https://ftp.gromacs.org/gromacs/gromacs-2018.8.tar.gz    2018.5    https://ftp.gromacs.org/gromacs/gromacs-2018.5.tar.gz    2018.4    https://ftp.gromacs.org/gromacs/gromacs-2018.4.tar.gz    2018.3    https://ftp.gromacs.org/gromacs/gromacs-2018.3.tar.gz    2018.2    https://ftp.gromacs.org/gromacs/gromacs-2018.2.tar.gz    2018.1    https://ftp.gromacs.org/gromacs/gromacs-2018.1.tar.gz    2018      https://ftp.gromacs.org/gromacs/gromacs-2018.tar.gz    2016.6    https://ftp.gromacs.org/gromacs/gromacs-2016.6.tar.gz    2016.5    https://ftp.gromacs.org/gromacs/gromacs-2016.5.tar.gz    2016.4    https://ftp.gromacs.org/gromacs/gromacs-2016.4.tar.gz    2016.3    https://ftp.gromacs.org/gromacs/gromacs-2016.3.tar.gz    5.1.5     https://ftp.gromacs.org/gromacs/gromacs-5.1.5.tar.gz    5.1.4     https://ftp.gromacs.org/gromacs/gromacs-5.1.4.tar.gz    5.1.2     https://ftp.gromacs.org/gromacs/gromacs-5.1.2.tar.gz    4.6.7     https://ftp.gromacs.org/gromacs/gromacs-4.6.7.tar.gz    4.5.5     https://ftp.gromacs.org/gromacs/gromacs-4.5.5.tar.gzDeprecated versions:    master    [git] https://gitlab.com/gromacs/gromacs.git on branch mainVariants:    build_system [cmake]                     cmake        Build systems supported by the package    build_type [Release]                     Debug, MinSizeRel, Profile, Reference, RelWithAssert,                                             RelWithDebInfo, Release        The build type to build    cp2k [false]                             false, true        CP2K QM/MM interface integration    cuda [false]                             false, true        Build with CUDA    cycle_subcounters [false]                false, true        Enables cycle subcounters    double [false]                           false, true        Produces a double precision version of the executables    hwloc [true]                             false, true        Use the hwloc portable hardware locality library    intel_provided_gcc [false]               false, true        Use this if Intel compiler is installed through spack.The g++ location is written to icp{c,x}.cfg    mdrun_only [false]                       false, true        Enables the build of a cut-down version of libgromacs and/or the mdrun program    mpi [true]                               false, true        Activate MPI support (disable for Thread-MPI support)    nosuffix [false]                         false, true        Disable default suffixes    opencl [false]                           false, true        Enable OpenCL support    openmp [true]                            false, true        Enables OpenMP at configure time    openmp_max_threads [none]                none        Max number of OpenMP threads    relaxed_double_precision [false]         false, true        GMX_RELAXED_DOUBLE_PRECISION, use only for Fujitsu PRIMEHPC    shared [true]                            false, true        Enables the build of shared libraries    when +cuda      cuda_arch [none]                       none, 10, 11, 12, 13, 20, 21, 30, 32, 35, 37, 50, 52, 53, 60, 61,                                             62, 70, 72, 75, 80, 86, 87, 89, 90, 90a          CUDA architecture    when build_system=cmake      generator [make]                       none          the build system generator to use    when build_system=cmake ^cmake@3.9:      ipo [false]                            false, true          CMake interprocedural optimization    when @2022:+cuda+mpi      cufftmp [false]                        false, true          Enable multi-GPU FFT support with cuFFTMp    when @2021:+mpi+sycl      heffte [false]                         false, true          Enable multi-GPU FFT support with HeFFTe    when @2021:      nblib [true]                           false, true          Build and install the NB-LIB C++ API for GROMACS      sycl [false]                           false, true          Enable SYCL support    when @2022:+sycl      intel-data-center-gpu-max [false]      false, true          Enable support for Intel Data Center GPU Max    when @2019:      gmxapi [true]                          false, true          Build and install the gmxlib python API for GROMACS    when @2024:+cuda+mpi      nvshmem [false]                        false, true          Enable NVSHMEM support for Nvidia GPUs    when arch=None-None-neoverse_v1:,neoverse_v2:      sve [true]                             false, true          Enable SVE on aarch64 if available    when arch=None-None-a64fx      sve [true]                             false, true          Enable SVE on aarch64 if available    when @2016.5:2016.6,2018.4,2018.6,2018.8,2019.2,2019.4,2019.6,2020.2,2020.4:2020.7,=2021,2021.4:2021.7,2022.3,2022.5,=2023      plumed [false]                         false, true          Enable PLUMED supportBuild Dependencies:    blas   cp2k  fftw-api  gmake   hwloc   mpi    nvhpc      plumed    cmake  cuda  gcc       heffte  lapack  ninja  pkgconfig  syclLink Dependencies:    blas  cp2k  cuda  fftw-api  gcc  heffte  hwloc  lapack  mpi  nvhpc  plumed  syclRun Dependencies:    NoneLicenses:    GPL-2.0-or-later@:4.5    LGPL-2.1-or-later@4.6:
```

Il y a beaucoup d'informations retournée par cette commande, les parties qui nous interesses sont :

- Description
  - Une description du logiciel
- Homepage
  - L'url du site internet du logiciel
- Preferred version
  - La version du logiciel qui est conseillée et sera installée si une demande d'installation ne précise pas en quelle version
- Safe versions
  - Toutes les autres version stables du logiciel qui peuvent être installées, a condition de demander la version explicitement lors de l'installation.
- Deprecated versions
  - Des versions installables mais non recommandées. Une option particulière est nécéssaire pour installer ces versions.
- Variants
  - Toutes les options et variantes possible à l'installation.

Pour ce qui concerne les variantes, cette liste affiche quelle est l'option par défaut de cette option si vous ne la precisez pas (activée : true, désactivée : false, ou une valeur autre), ainsi qu'une courte description. Il est également indiqué toutes les valeurs que ces options peuvent prendre.

Par exemple :

```
shared [true]                            false, true        Enables the build of shared libraries
```

Ici il est indiqué que l'option 'shared' sera activée par défaut, qu'elle peut être désactivée, et a quoi elle sert.

Un autre exemple :

```
when +cuda      cuda_arch [none]                       none, 10, 11, 12, 13, 20, 21, 30,                                             32, 35, 37, 50, 52, 53, 60, 61,                                             62, 70, 72, 75, 80, 86, 87, 89,                                             90, 90a          CUDA architecture
```

Ici cette option est indiquée comme valide uniquement si l'option `cuda` est activée. Elle est pas défaut à `none` et peut être reglée sur de multiples valeurs.

## Choisir un compilateur[â](#choisir-un-compilateur "Lien direct vers Choisir un compilateur")

Il est possible mais pas obligatoire de choisir un compilateur quand nous installons un logiciel.
Pour connaitre la liste des compilateurs disponibles vous pouvez faire :

spack compilers
```
==> Available compilers-- aocc rhel9-x86_64 --------------------------------------------aocc@4.2.0-- gcc rhel9-x86_64 ---------------------------------------------gcc@11.4.1
```

Quand vous installez un logiciel, Spack choisira le compilateur compatible qu'il jugera le plus adapté (en suivant ce qu'a définit le créateur du package du logiciel installé). Mais d'autres compilateurs sont utilisables, nous verrons comment préciser le compilateur à utiliser, un peu plus bas.

## Lancer l'installation[â](#lancer-linstallation "Lien direct vers Lancer l'installation")

Imaginons maintenant que nous souhaitons installer curl dans sa version 8.7.1, avec une version incluant libssh, et une version sans :

`spack install curl@8.7.1 +libssh`

Spack va alors installer et configurer toutes les dépendances nécéssaires pour ce logiciel, puis le logiciel.

Une fois terminé il indique :

```
==> curl: Successfully installed curl-8.7.1-toswmnbqvylrxvrf3v7vb23un2jrwnto  Stage: 1.57s.  Autoreconf: 0.00s.  Configure: 27.26s.  Build: 13.02s.  Install: 2.90s.  Post-install: 0.25s.  Total: 45.74s[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/curl-8.7.1-toswmnbqvylrxvrf3v7vb23un2jrwnto
```

Pour installer la version sans libssh :

`spack install curl@8.7.1 -libssh` (ou `spack install curl@8.7.1` étant donné que l'option libssh est à false par défaut)

```
==> curl: Successfully installed curl-8.7.1-7sft7ozg4d3qyyesbnf2wojhgjjr4yga  Stage: 1.37s.  Autoreconf: 0.00s.  Configure: 23.05s.  Build: 7.48s.  Install: 2.70s.  Post-install: 0.22s.  Total: 35.17s[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/curl-8.7.1-7sft7ozg4d3qyyesbnf2wojhgjjr4yga
```

Si vous demandez à installer une version et variante d'un logiciel qui est déjà disponible, Spack vous indiquera un [+] devant chaque étape, sans relancer d'étape d'installation :

```
[+] /usr (external glibc-2.34-xydkni6ohoz32trfuw3af23mgh4gzekh)[+] /apps/2025/spack_install/linux-rhel9-zen4/gcc-11.4.1/gcc-runtime-11.4.1-gyz4y5hug4ulenjop7ie3gyqjy6xpe62[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/nghttp2-1.63.0-2xii4wyk6lzolehjwlrhz4thm2753gxz[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/zlib-ng-2.2.1-25dszjwt2ffkfurjanzasse75stbsjb2[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/openssl-3.3.1-uvtay55szpuhp555uawoedgtnsp4c3rs[+] /home/fberini/.spack/userspace-installed/linux-rhel9-zen4/gcc-11.4.1/curl-8.7.1-7sft7ozg4d3qyyesbnf2wojhgjjr4yga
```

Vous remarquerez peut être dans les retours des commandes précédantes que ces installations ne se passent pas dans le /apps de Romeo.
En effet, quand vous installez un logiciel ce dernier s'installe dans un repertoire 'userspace-installed' dans votre home. Ce repertoire est a l'intérieur d'un repertoire caché nommé '.spack' contenant d'éventuelles configurations personnalisées, et le cache de vos installations.

Vous pouvez nettoyez ce cache en faisant : `spack clean`

Suite aux deux installations des commandes précédantes, nous pouvons constater que les deux variantes de curl 8.7.1 sont bien disponibles, l'une avec l'option libssh, et une sans :

spack find --variants curl@8.7.1
```
-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------curl@8.7.1~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=opensslcurl@8.7.1~gssapi~ldap~libidn2~librtmp+libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl==> 2 installed packages
```

Ces programmes peuvent maintenant être utilisés sur votre compte. Mais uniquement depuis votre compte.

Pour qu'un autre utilisateur ai accès à une version identique du logiciel, il faudra lui communiquer la ligne d'installation exacte utilisée par votre installation, en incluant bien les versions, compilateurs (voir plus bas) et options, afin qu'il l'installe dans son home.

Si de nombreux utilisateurs sont succeptibles d'utiliser ce logiciel, il est possible de demander via un ticket au support ROMEO pour qu'il soit installé et donc mis à disposition de manière globale dans les programmes disponibles via Spack. Il faudra là bien aussi préciser dans le ticket toutes les versions et options souhaitées.

---

Il est possible de tester ce que Spack va faire avant de lancer l'installation, par exemple si j'utilise la commande `spec` au lieu de `install`, je peux voir ce qui sera utilisé, ce qui sera installé (avec un `-` ), et ce qui est déjà installé (avec un `[-]`) :

spack spec -t curl@8.7.1 %aocc@4.2.0 ^perl %aocc -threads
```
Input spec-------------------------------- -   curl@8.7.1%aocc@4.2.0 -       ^perl%aocc~threadsConcretized-------------------------------- -   curl@8.7.1%aocc@4.2.0~gssapi~ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-rhel9-zen4[e]      ^glibc@2.34%gcc@11.4.1 build_system=autotools arch=linux-rhel9-zen4[^]      ^gmake@4.4.1%gcc@11.4.1~guile build_system=generic arch=linux-rhel9-zen4[^]          ^gcc-runtime@11.4.1%gcc@11.4.1 build_system=generic arch=linux-rhel9-zen4[+]      ^nghttp2@1.63.0%gcc@11.4.1 build_system=autotools arch=linux-rhel9-zen4[+]          ^diffutils@3.10%gcc@11.4.1 build_system=autotools arch=linux-rhel9-zen4[+]      ^openssl@3.3.1%gcc@11.4.1~docs+shared build_system=generic certs=mozilla arch=linux-rhel9-zen4[+]          ^ca-certificates-mozilla@2023-05-30%gcc@11.4.1 build_system=generic arch=linux-rhel9-zen4[+]          ^perl@5.40.0%gcc@11.4.1+cpanm+opcode+open+shared+threads build_system=generic arch=linux-rhel9-zen4 -       ^perl@5.40.0%aocc@4.2.0+cpanm+opcode+open+shared~threads build_system=generic arch=linux-rhel9-zen4[+]          ^berkeley-db@18.1.40%gcc@11.4.1+cxx~docs+stl build_system=autotools patches=26090f4,b231fcc arch=linux-rhel9-zen4[+]          ^bzip2@1.0.8%gcc@11.4.1~debug~pic+shared build_system=generic arch=linux-rhel9-zen4[+]          ^gdbm@1.23%gcc@11.4.1 build_system=autotools arch=linux-rhel9-zen4[+]              ^readline@8.2%gcc@11.4.1 build_system=autotools patches=bbf97f1 arch=linux-rhel9-zen4[+]                  ^ncurses@6.5%gcc@11.4.1~symlinks+termlib abi=none build_system=autotools patches=7a351bc arch=linux-rhel9-zen4[+]      ^pkgconf@2.2.0%gcc@11.4.1 build_system=autotools arch=linux-rhel9-zen4[+]      ^zlib-ng@2.2.1%gcc@11.4.1+compat+new_strategies+opt+pic+shared build_system=autotools arch=linux-rhel9-zen4
```

## Autres options[â](#autres-options "Lien direct vers Autres options")

Il existe de nombreuses options et possibilités de personnalisation avec Spack. Il est par exemple possible de demander un compilateur en particulier le symbole `%`, ou préciser des options sur ses dépendances en mettant un `^` devant le nom de la dépendance, et les options et compilateur de celle ci derrière. Les dépendances d'un logiciel sont indiquées dans la commande `spack info` (voir chapitre précédant).

Par exemple je peux demander une installation de curl 8.7.1 avec le compilateur aocc, mais dont sa dépendance 'perl' est compilée avec GCC sans l'option threads :

`spack install curl@8.7.1 %aocc@4.2.0 ^perl %gcc@11.4.1 -threads`

Nous pouvons alors voir cette version disponible dans la catégorie `linux-rhel9-zen4 / aocc@4.2.0`:

```
spack find --explicit-- linux-rhel9-zen4 / aocc@4.2.0 --------------------------------curl@8.7.1-- linux-rhel9-zen4 / gcc@11.4.1 --------------------------------aocc@4.2.0  blender@4.2.3  curl@8.4.0  curl@8.4.0  curl@8.7.1  curl@8.7.1  gromacs@2024.3  unzip@6.0==> 9 installed packages
```

Cette version peut ensuite être chargée pour être utilisée, soit en utilisant le hash (voir chapitre précédant), soit en précisant les options souhaitées précisement:

`spack load curl@8.7.1 %aocc@4.2.0 ^perl %gcc@11.4.1 -threads`

Pour ce type de manipulations, nous vous conseillons de lire la documentation officielle de Spack et de ne les manipuler que si vous savez ce que vous faites.

## Optimisations[â](#optimisations "Lien direct vers Optimisations")

Il est possible de changer l'optimisation par défaut utiliser sur l'installation Spack de ROMEO.
Pour cela vous pouvez ajouter a votre commande `spack install` des options modifiant les flags de compilation.
Par exemple pour activer le niveau d'optimisation 3 (agressif) : `spack install zlib cflags=-O3 cxxflags=-O3 cppflags=-O3 fflags=-O3`
Pour plus d'informations, vous pouvez consulter la documentation officielle de Spack.