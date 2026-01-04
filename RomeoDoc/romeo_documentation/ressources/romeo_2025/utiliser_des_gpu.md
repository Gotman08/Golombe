---
title: "Utiliser des GPU"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/utiliser_des_gpu"
scraped_at: "2026-01-04 02:40:56"
---

# Utiliser des GPU

Les GPU ne sont disponibles que sur les serveurs de calcul en architecture Aarch64.
Il y a quatre GH200 sur chaque serveur de calcul Aarch64 avec chacun 96Gio de VRAM.

## Demander des GPU à Slurm[​](#demander-des-gpu-à-slurm "Lien direct vers Demander des GPU à Slurm")

Pour avoir accès à des GPU il conviendra d'utiliser l'option `--gpus-per-node=` en indiquant combien de GPU par serveur vous souhaitez, par exemple :

Pour demander 1 GPU, via salloc, j'ajouterais l'option `--gpus-per-node=1` dans ma commande

Pour demander 4 GPU par serveur, via un fichier de soumission, j'ajouterais dans mon fichier de soumission la ligne : `#SBATCH --gpus-per-node=4`

## Accéder aux GPU réservés, dans votre job.[​](#accéder-aux-gpu-réservés-dans-votre-job "Lien direct vers Accéder aux GPU réservés, dans votre job.")

Dans un job soumis via un fichier de soumission, les GPU visibles par votre job seront automatiquement et uniquement ceux qui vous sont réservés, vous n'avez rien à faire.

Si vous utilisez une soumission interactive, il y a deux cas possibles, une fois votre commande `salloc` lancée et votre job interactif en running :

- Vous utilisez la commande `srun --pty bash -i`, cela va vous connecter au premier serveur qui vous ai alloué, en préservant votre environnement. Dans ce cas cela fonctionne comme pour le fichier de soumission, il n'y a rien de plus à faire vous ne verrez sur ce serveur que les GPU qui vous sont alloués.
- Vous vous connectez sur un serveur où tourne un de vos jobs à l'aide de la commande `ssh`, dans ce cas l'environnement Slurm n'est pas défini et vous pouvez voir tous les GPU du serveur. Mais attention ceux ci sont peut être réservées tout en partie par un autre job d'un autre utilisateur ! **Donc vous ne devez en aucun cas utiliser un GPU via ce moyen sans être certain qu'il vous ai alloué.**
  - Si vous connaissez les `id` des GPU que vous pouvez utiliser (allant de 0 à 3 sur ROMEO 2025), vous pouvez alors définir la variable d'environnement suivante pour dire a vos programmes de les utiliser :
    - `export CUDA_VISIBLE_DEVICES=0,2`
    - Par exemple, pour utiliser les GPU 0 et 2.

## Accéder à CUDA[​](#accéder-à-cuda "Lien direct vers Accéder à CUDA")

Vous pouvez charger cuda à l'aide de Spack. Pour rechercher les versions de CUDA disponibles et comment charger le package cuda souhaité, vous pouvez consulter la page de documentation 'Charger ses logiciels'.
Pensez à charger le bon environnement logiciel à l'aide de la commande `romeo_load_armgpu_env` comme expliqué dans la page 'Charger ses logiciels'.