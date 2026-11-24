# Validation — jeu de référence (« golden master »)

*[Retour au README](../README.fr.md) · [Méthodes](METHODES.md) · [Développement](DEVELOPMENT.md)*

Ce document décrit comment on vérifie que les résultats scientifiques restent
identiques d'une version à l'autre, en particulier pendant la migration de
l'interface CustomTkinter vers PySide6. Il tient aussi le registre des
différences acceptées.

## Principe

Les tests unitaires vérifient chaque algorithme sur des signaux synthétiques.
Le jeu de référence vérifie autre chose : qu'un enregistrement donne, **de
bout en bout**, les mêmes nombres qu'avec la version de référence (6.2.0).

`tests/regression/golden.py` rejoue sans fenêtre le code de l'application :
les méthodes des contrôleurs appelées par les boutons (Detect Peaks, Analyze,
Epochs, Sliding…) tournent sur une application factice qui ne contient que
l'état (`ecg/ui/state.py`) et aucun widget. Les paramètres sont ceux de
l'application au démarrage. Le code scientifique exécuté est donc celui de
l'application, sans modification.

## Ce qui est calculé et comparé

| Étape | Scénarios |
|---|---|
| Import | signal, canal retenu, liste des canaux, fs lue dans le fichier |
| Préparation + détection | Auto sans filtre (démarrage), Auto filtré, Auto filtré + notch 50 Hz, SG + Derivative, Wavelet, Envelope Max : signal préparé (échantillonné), candidats, proéminences, pics acceptés / rejetés, seuil, polarité, texte des filtres |
| Qualité | score 0–100 avant et après l'analyse |
| Revue des artefacts | candidats proposés (fenêtre 11 battements, écart 20 %) |
| Analyze | RR, FC, HRV temporelle, template de battement, corrélation par battement |
| Fréquentiel, non linéaire | tables complètes (plafond d'entropie 2 000 RR) |
| Intervalles | N = 1 et N = 32 (défaut), bornes strictes, QTc Mitchell, template par défaut |
| Époques | 60 s ; métriques par défaut et toutes les métriques |
| Fenêtre glissante | 30 s / pas 5 s ; métriques par défaut et toutes les métriques |
| Événements | classification, résumé des battements prématurés, entrée PR du délai AV |
| Autonomique | CSI / CPI (paramètres par défaut) et LF / HF par SPWVD |
| Édition des pics | 3 pics exclus et 1 ajouté à la main, puis reconstruction |
| Correction automatique des artefacts | Analyze, fréquentiel, non linéaire, époques, glissante, événements, édition |
| Fenêtre d'analyse 20–120 s | Analyze, fréquentiel, époques, glissante |

Ne sont pas couverts : le détecteur ML (il dépend du modèle entraîné sur
chaque poste), les exports Excel / PDF / Prism (comparés à la phase 4), les
sessions, Compare segments et Event-locked (ils demandent des annotations).

## Tolérance

- Entiers, booléens, textes (indices des pics, comptages, libellés) : **égalité
  exacte**.
- Flottants : |a − b| ≤ 1e-12 + 1e-9 × |b|, NaN égal à NaN.
- Valeurs exprimées en unités du signal normalisé (signal préparé,
  proéminences, seuil, amplitudes, template de battement) : |a − b| ≤ 1e-7 +
  1e-9 × |b|. Le signal est un z-score (écart-type 1), donc 1e-7 est une
  erreur absolue de 1e-7 écart-type. Près d'un passage par zéro, l'erreur
  relative d'un échantillon n'a pas de sens.

**Pourquoi cette tolérance sur le signal.** Avec exactement les mêmes versions
de bibliothèques que sous Linux, le signal filtré diffère sous Windows sur des
échantillons isolés, et l'écart dépend de la machine de CI :

| Date | Exécution | Écart maximal observé |
|---|---|---|
| 24/11/2026 | PR #2, runner Windows | ~1e-11 |
| 24/11/2026 | PR #4, autre runner Windows | 1,3e-9 |

Les passe-haut très bas (0,5 Hz d'`ecg_clean`, borne basse du passe-bande à
2 kHz) sont des filtres mal conditionnés : ils amplifient les différences
d'arrondi du processeur. Les pics, la HRV, les intervalles et toutes les autres
sorties restaient identiques. Une première tolérance de 1e-9, calibrée sur une
seule machine, s'est révélée trop serrée ; elle est fixée à 1e-7. Une vraie
modification reste détectée : normaliser avec ddof = 1 (écart d'environ 2e-6)
change toujours 75 sorties, dont 52 en unités du signal.

**Versions des bibliothèques.** La référence enregistre les versions de numpy,
scipy, pandas, neurokit2, scikit-learn et PyWavelets. Si l'une d'elles diffère
lors de la vérification, une différence est signalée comme « échec attendu »
(xfail), avec les versions et les valeurs changées : c'est le cas du job CI
« newest deps ». Avec les mêmes versions, toute différence est un échec.

La tolérance absorbe les effets d'ordre de sommation d'une autre bibliothèque
BLAS ou d'un autre processeur. Un algorithme simplement déplacé donne les
mêmes flottants au bit près sur la même machine.

Un signal par échantillon (7,2 millions de valeurs par heure à 2 kHz) est
stocké sur un pas régulier d'au plus 10 000 points, avec sa longueur, sa
moyenne, son écart-type, son minimum et son maximum.

**Sensibilité vérifiée** (24/11/2026, enregistrement synthétique) : passer le
seuil pNN de 6 à 7 ms change 11 valeurs ; normaliser avec l'écart-type
d'échantillon (ddof = 1 au lieu de 0, soit un écart relatif d'environ 2e-6)
en change 75, de la détection aux intervalles. Deux exécutions successives
sans modification donnent zéro différence.

## Utilisation

Enregistrement synthétique : contrôlé à chaque exécution de la suite de tests
(`tests/regression/test_golden_master.py`), contre la référence versionnée
dans `tests/regression/reference/`. Le fichier
`tests/fixtures/synthetic_mouse_ecg.mat` (150 s à 2 kHz, battements
prématurés, doublet, pause, bruit) est décrit dans
`tests/regression/synthetic_recording.py`.

Enregistrements réels : ils restent **hors du dépôt**.

```bash
# Une fois, avec la version à préserver
python tools/golden_master.py record D:/ECG/248.mat D:/ECG/247.mat --out D:/ECG/reference

# Ensuite, à chaque étape (code de sortie 1 en cas de différence)
python tools/golden_master.py check D:/ECG/248.mat D:/ECG/247.mat --ref D:/ECG/reference

# Ou dans pytest : dossier contenant les .mat et leur sous-dossier reference/
ECG_GOLDEN_DIR=D:/ECG python -m pytest tests/regression
```

Chaque référence garde l'empreinte SHA-256 du fichier source, la version et
le commit de l'application, les versions des bibliothèques et la durée de
chaque étape.

## Quand une différence est voulue

Une correction scientifique (par exemple la coupure basse du filtre) change
la référence. Dans ce cas :

1. la correction fait l'objet d'un changement séparé, expliqué et accepté ;
2. la sortie de `check` est recopiée ci-dessous ;
3. la référence est enregistrée de nouveau ;
4. la tolérance n'est jamais élargie pour faire passer un écart.

## Registre des différences acceptées

| Date | Version | Changement | Valeurs touchées | Justification |
|---|---|---|---|---|
| — | 6.2.0 | Référence initiale | — | — |

## Changements de bibliothèques observés (non acceptés)

| Date | Bibliothèques | Valeurs touchées |
|---|---|---|
| 24/11/2026 | neurokit2 0.2.13 → 0.3.0 (avec numpy 2.5.5, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1) | `HRV_TINN` 0 → 46,875 ms ; `HRV_PSS` 0,908 → 0,824 (et 0,905 → 0,819 après correction des artefacts) ; signal filtré à environ 5e-10 près |

Ne pas relever `neurokit2` dans `packaging/constraints.txt` sans examiner ces
changements : TINN et PSS figurent dans les exports.
