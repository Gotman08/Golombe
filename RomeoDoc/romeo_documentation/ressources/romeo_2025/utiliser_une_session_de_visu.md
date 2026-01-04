---
title: "Utiliser une session de visualisation"
source: "https://romeo.univ-reims.fr/documentation/ressources/romeo_2025/utiliser_une_session_de_visu"
scraped_at: "2026-01-04 02:40:59"
---

# Utiliser une session de visualisation

Le service de visualisation est tout juste disponible et possiblement encore sujet à modification, tout comme cette documentation.

> ⚠️ **Warning**
>
> attention

La visualisation n'est disponible que sur les noeuds de login romeo2 et romeo3. Seul romeo2 est actuellement ouvert aux utilisateurs, romeo3 étant réservé pour terminé la configuration finale de la visualisation avecv GPU.

> ⚠️ **Warning**
>
> attention

Les GPU ne sont pas encore disponibles sur les noeuds de visu, mais le serons sous peu.

# Préparer son environnement ROMEO la première fois.

**A faire qu'une seule fois par compte.**

Dans un terminal connecté à ROMEO , entrez :

```
/apps/romeotools/romeovisu.sh prepare
```

Il va vous demander si vous souhaitez lancer vncpasswd, sauf si vous savez avoir déjà un mot de passe vnc, répondez oui en entrant o. Choisissez alors un mot de pass VNC

## Lancer une session de visu[â](#lancer-une-session-de-visu "Lien direct vers Lancer une session de visu")

Dans un terminal connecté à ROMEO , entrez :

```
/apps/romeotools/romeovisu.sh start
```

Dans un autre terminé **non connecté** à ROMEO, entrez la commande SSH fournie lors de l'execution du script ci dessus.
Cette commande semblera bloquée, c'est normal ! Tant qu'elle reste active le pont SSH que nous allons utiliser fonctionnera. Il suffira quand on le souhaite de l'arrêter ou de ferme le terminal pour détruire le pont.

Installez et démarrez un client VNC, et connectez vous à l'url : `localhost:5988`
Il va vous demander votre mot de passe VNC, choisis précédemment.

Si vous ne possédez pas de client VNC, nous vous conseillons TigerVNC : <https://tigervnc.org/> Nos tests ont été réalisés avec la version 1.15.0 pour Windows : <https://sourceforge.net/projects/tigervnc/files/stable/1.15.0>
Vous trouverez dans ce lien

- un installateur exe Windows 64 bits : [tigervnc64-1.15.0.exe](https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/tigervnc64-1.15.0.exe/download "Click to download tigervnc64-1.15.0.exe")
- un installateur exe Windows 32 bits : [tigervnc-1.15.0.exe](https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/tigervnc-1.15.0.exe/download "Click to download tigervnc-1.15.0.exe")
- une version Windows 64 bits portable : [vncviewer64-1.15.0.exe](https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/vncviewer64-1.15.0.exe/download "Click to download vncviewer64-1.15.0.exe")
- une version Windows 32 bits portable : [vncviewer-1.15.0.exe](https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/vncviewer-1.15.0.exe/download "Click to download vncviewer-1.15.0.exe")
- un installateur Mac : [TigerVNC-1.15.0.dmg](https://sourceforge.net/projects/tigervnc/files/stable/1.15.0/TigerVNC-1.15.0.dmg/download "Click to download TigerVNC-1.15.0.dmg")
- Des dossiers où se trouve différentes version pour Linux RHEL et Ubuntu.

Une fois connecté, vous arrivez sur un bureau virtuel vous permettant de lancer des applications.
Cet espace tourne sur un serveur de login, il est donc interdit d'y réaliser des calculs scientifiques, et doit être utilisé purement pour de la visualisation.

Pour utiliser un GPU, préfixez votre commande par 'vglrun' (pas encore disponible)

## Eteindre manuellement une session de visu[â](#eteindre-manuellement-une-session-de-visu "Lien direct vers Eteindre manuellement une session de visu")

Vous pouvez faire la commande **vncserver -list**, elle va alors vous donner la liste de vos sessions VNC avec leur numéro, faite alors la commande **vncserver -kill** X où X est le numéro de la session à terminer.