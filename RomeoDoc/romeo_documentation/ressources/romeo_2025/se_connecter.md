---
title: "Se connecter à ROMEO 2025"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/se_connecter"
scraped_at: "2026-01-04 02:40:58"
---

# Se connecter à ROMEO 2025

## Prérequis[​](#prérequis "Lien direct vers Prérequis")

Vous avez besoin d'un compte ROMEO valide, d'une clé publique SSH sur le portail, ainsi que d'attendre que ces derniers soient déployés sur le supercalculateur.

Vous pouvez trouver les étapes pour obtenir un compte [ici](creation_compte.md).
Les personnels non-académiques peuvent prendre contact avec Florence Draux ([florence.draux@univ-reims.fr](mailto:florence.draux@univ-reims.fr))

## Comment se connecter à ROMEO 2025[​](#comment-se-connecter-à-romeo-2025 "Lien direct vers Comment se connecter à ROMEO 2025")

La connexion s'effectue en SSH vers un serveur de login.
Les quatre serveurs de logins disponibles sont :

- romeo1.univ-reims.fr
- romeo2.univ-reims.fr
- romeo3.univ-reims.fr
- romeo4.univ-reims.fr

Par exemple :
`ssh nomUtilisateur@romeo1.univ-reims.fr`

Vous pouvez trouver votre nom d'utilisateur pour le supercalculateur Romeo 2025 via le nom de compte indiqué dans la page principale « Tableau de bord » du portail.

Si vous utilisez un fichier de configuration vous devez indiquer l'endroit où se trouve votre clé SSH. Cette clef SSH doit avoir été renseignée sur le portail ROMEO (voir [Gérer ses clefs SSH](ressources/connexion_ssh.md)) :

```
Host=romeo1Hostname=romeo1.univ-reims.frIdentityFile=[CheminVersLeFichierDeClefSSH]IdentitiesOnly=yesUser=[UserName]
```