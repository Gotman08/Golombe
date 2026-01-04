---
title: "Oratio"
source: "https://romeo.univ-reims.fr/documentation/services/Oratio/"
scraped_at: "2026-01-04 02:41:05"
---

# Oratio

> ⚠️ **Warning**
>
> attention

ATTENTION Le projet IlaaS et le service Oratio sont encore en cours de finalisation.

Des éléments sont susceptibles de changer. Il est important de noter que pour le moment :

- Des fonctionnalités peuvent changer, évoluer ou cesser de fonctionner sans préavis.
- Des modèles peuvent être ajoutés ou supprimés selon les besoins du projet.
- Le système peut ne pas encore supporter une forte charge de demandes simultanées.
- La stabilité complète ne sera atteinte qu'à la fin de la mise en place du projet.
- Cette API ne doit PAS être utilisée pour des applications en production pour le moment Utilisez pour le moment cet outil en connaissance de cause, uniquement pour des expérimentations, des tests ou des projets de recherche.

**Oratio (GUI)** est l'interface web OpenWebUI installée sur l'infrastructure **ROMEO**.
Elle permet aux membres de l'URCA d'utiliser des modèles LLM **sans écrire de code**, directement depuis un navigateur.

> **â ï¸ Accès GUI** : Oratio GUI est accessible **exclusivement aux utilisateurs possédant une adresse e-mail URCA valide**.

---

## L'interface web (oratio.univ-reims.fr)[â](#linterface-web-oratiouniv-reimsfr "Lien direct vers L'interface web (oratio.univ-reims.fr)")

C'est l'interface *visuelle* et utilisable par les utilisateurs URCA.
**Oratio GUI n'appartient pas au projet ILaaS**, mais peut utiliser les ressources du réseau ILaaS.

#### Pourquoi le nom "Oratio" ?[â](#pourquoi-le-nom-oratio- "Lien direct vers Pourquoi le nom "Oratio" ?")

Oratio est un mot latin signifiant « discours » ou « parole ».
Il évoque également **Horatio**, compagnon et témoin de Hamlet dans la pièce éponyme de Shakespeare, faisant écho à **ROMEO**.

## L'API[â](#lapi "Lien direct vers L'API")

L'API Oratio est une API de type 'OpenAI Compatible' servie par le programme LiteLLM.
Via cette API vous pouvez connecter vos applications aux modèles IA du Centre de calcul ROMEO et/ou du réseau IlaaS via une clé unique et personnelle valable 1 an.

# ILaaS - Réseau national

Le but du projet ILaaS est de fournir à l'Enseignement Supérieur Français des moyens techniques pour mettre en Åuvre l'Intelligence Artificielle en répondant aux défis importants de soutenabilité, de résilience, de confiance et de sobriété numérique.
Il vise à offrir une infrastructure d'inférence mutualisée et fiable servant les besoins essentiels de l'ESR, en visant l'équilibre budgétaire, la qualité de service et la sécurité des données.

Le réseau repose sur un **dispatcher fédéré** redirigeant intelligemment les requêtes vers les différents serveurs d'inférence des participants au projet IlaaS.

Il est joignable par API uniquement. Plus d'informations sur <https://www.ilaas.fr/>

## Architecture Simplifiée[â](#architecture-simplifiée "Lien direct vers Architecture Simplifiée")

```
flowchart LR    Client["Applications / scripts URCA"] --> LiteLLM["LiteLLM - Outil de gestion des clés API"]    ClientExt["Applications / scripts extérieurs"] --> DispatcherILaaS    LiteLLM -->|Modèles locaux| OratioROMEO["Oratio - Serveur LLM<br/>(ROMEO)"]    LiteLLM -->|Modèles ILaaS| DispatcherILaaS["Dispatcher ILaaS"]    DispatcherILaaS --> |Accède aux modèles IlaaS| AutresNoeudsILaaS["Autres serveurs partenaires ILaaS"]        DispatcherILaaS --> |Accède aux modèles ROMEO| OratioROMEO    OratioROMEO --> |Accède aux modèles IlaaS| DispatcherILaaS
```