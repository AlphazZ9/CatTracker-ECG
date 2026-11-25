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
la structure `FilterParams` (détecteur Auto, sans filtrage, seuil 0,5) et les
valeurs initiales des champs de l'interface. Le menu Method de l'interface Tk
démarre, lui, sur SG + Derivative : ce réglage est le scénario « SG + Derivative ». Le code scientifique exécuté est donc celui de
l'application, sans modification.

## Ce qui est calculé et comparé

| Étape | Scénarios |
|---|---|
| Import | signal, canal retenu, liste des canaux, fs lue dans le fichier |
| Préparation + détection | Auto sans filtre (défauts de `FilterParams`), Auto filtré, Auto filtré + notch 50 Hz, SG + Derivative, Wavelet, Envelope Max : signal préparé (échantillonné), candidats, proéminences, pics acceptés / rejetés, seuil, polarité, texte des filtres |
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
| Zones bruitées (40–45,5 s et 95–96,5 s) | Analyze, fréquentiel, non linéaire, époques, glissante, événements, autonomique |

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
| 24/11/2026 | 6.2.0 | Scénario « zones bruitées » ajouté | 538 valeurs nouvelles ; les 1 666 existantes inchangées | Référence enregistrée de nouveau avec le code de la version 6.2.0 (commit `df556ec`, avant toute extraction), et non avec le code en cours de migration |

## Changements de bibliothèques observés (non acceptés)

| Date | Bibliothèques | Valeurs touchées |
|---|---|---|
| 24/11/2026 | neurokit2 0.2.13 → 0.3.0 (avec numpy 2.5.5, scipy 1.18.1, pandas 3.1.1, scikit-learn 1.9.1) | `HRV_TINN` 0 → 46,875 ms ; `HRV_PSS` 0,908 → 0,824 (et 0,905 → 0,819 après correction des artefacts) ; signal filtré à environ 5e-10 près |

Ne pas relever `neurokit2` dans `packaging/constraints.txt` sans examiner ces
changements : TINN et PSS figurent dans les exports.

## Phase 4 — interface Qt contre interface Tk

Le jeu de référence vérifie les calculs. La phase 4 vérifie en plus que
**l'interface Qt exporte exactement ce qu'exporte l'interface Tk**, fichier
par fichier, sur le même enregistrement.

### Méthode
- `tools/interface_outputs.py` exécute **le même scénario** dans une interface. Chaque interface tourne dans son propre processus et son propre dossier personnel (aucun réglage, session ou modèle de l'utilisateur n'est lu). Le scénario passe par les chemins de code de chaque interface :
  - ouvrir l'enregistrement ;
  - Method = Auto, Detect Peaks ;
  - une zone bruitée et une annotation ;
  - Analyze (cœur, puis HRV fréquentielle et non linéaire) ;
  - Measure intervals ;
  - Epochs, Sliding, Autonomic ;
  - Classify events.
- Il écrit ensuite les sorties :
  - le classeur Excel (`write_excel`) ;
  - le CSV des RR (`export_rr_csv`) ;
  - le texte du rapport et les lignes de provenance de l'onglet Report ;
  - l'état que Save Session enregistre (`collect_session_state`).
- `tools/compare_interfaces.py` compare ces sorties :
  - **classeur** : cellule par cellule, mêmes feuilles et mêmes valeurs, nombres égaux au dernier bit. Seule exception : une cellule d'horodatage d'export doit être un horodatage des deux côtés ;
  - **CSV des RR** : octet par octet ;
  - **rapport et provenance** : ligne par ligne ;
  - **session** : clé par clé, hors heures d'enregistrement et de calcul.
- **Sessions croisées** : la session enregistrée par chaque interface est restaurée dans les deux, puis les sorties sont comparées. L'état restauré doit aussi être égal à l'état enregistré.

Pour l'interface Tk, les fils de calcul sont exécutés au moment de leur
lancement, sur le fil principal. Certaines installations de Python (Tcl 9)
ne transmettent pas leurs appels `after()` à la boucle Tk. Le code des
calculs reste le même ; seul le moment où il tourne change.

### Résultat sur l'enregistrement synthétique (`tests/fixtures/synthetic_mouse_ecg.mat`)

| Sortie | Comparé | Résultat |
|---|---|---|
| Classeur Excel | 284 031 cellules | identique (1 cellule d'horodatage) |
| CSV des RR | 1 301 lignes | identique |
| Texte du rapport | 158 lignes | identique |
| Provenance (onglet Report) | 38 lignes | identique |
| État de session | 34 clés | identique (hors heures) |
| Session Tk restaurée dans Tk et dans Qt | 234 510 cellules + rapport + CSV + état | identique ; état restauré = état enregistré |
| Session Qt restaurée dans Tk et dans Qt | idem | identique ; état restauré = état enregistré |

Test automatique (jusqu'à la version 6.2.0) : `tests/validation/test_tk_vs_qt.py`, retiré avec l'interface Tk (voir plus bas).

### Écarts trouvés par cette comparaison, puis corrigés dans l'interface Qt
1. **Suivi de fraîcheur absent.** Il manquait les états « up to date / out of date » du rapport et le `freshness_basis` / `computed_at` des sessions. L'interface Qt utilise maintenant le `FreshnessService` de Tk, avec les mêmes horodatages.
2. **`sg_window_ms` des sessions enregistré comme nombre.** Tk ne le relit que sous forme de texte, donc une session Qt aurait perdu cette fenêtre dans Tk. Les champs `sex`, `project_name` et `show_raw` manquaient aussi.
3. **Métriques LW absentes de Sliding** (CCD, SD1 (LW), SD2 (LW)).
4. **Mode du Time course non suivi** (`time_course_mode` des sessions).

### Sur tes propres enregistrements
`tools/compare_interfaces.py` n'existe que tant que l'interface Tk existe : il
faut le lancer depuis la branche `phase-5/distribution` (ou toute version
6.x), pas depuis la version 7.
```bash
git switch phase-5/distribution
python tools/compare_interfaces.py chemin/vers/fichier1.mat chemin/vers/fichier2.mat
```
Le code de sortie vaut 0 si tout est identique ; sinon les différences sont listées. Sous Linux sans écran, le lancement passe automatiquement par `xvfb-run`.

### Pas encore couvert par la comparaison
- Compare segments, Group et Event-locked : ils sont vérifiés contre les fonctions du cœur appelées avec les paramètres de Tk (`tests/qt`), mais pas encore dans ce scénario de bout en bout.
- Les exports PDF et ZIP de figures contiennent des dates et du rendu : ils ne sont pas comparés octet par octet.

## Phase 6 — retrait de l'interface Tk (version 7.0.0)

L'interface CustomTkinter (`ecg/ui`) est retirée ; `python -m ecg` ouvre
l'interface Qt.

**Ce qui est garanti, et comment :**
- Le code des contrôleurs et des figures que l'interface Qt utilisait (analyse,
  figures, exports, sessions, fraîcheur, Event-locked…) est **déplacé tel quel**
  dans `ecg/app`. Seuls changent :
  - ses imports Tk ;
  - les boîtes de dialogue, qui passent par `ecg/app/dialogs.py` ;
  - deux parties purement Tk, les cartes d'événements et la hauteur du tableau
    Rhythm, remplacées par les rappels que l'interface Qt utilisait déjà.
- Les fonctions sans interface des modules Tk supprimés sont **copiées octet
  pour octet** par un script dans `ecg/app/*_helpers.py`, `context_fields.py`,
  `text_tables.py`, `jobs.py` et `wave_landmarks.py`. Exemples : configurations
  de Compare segments, lignes du tableau de groupe, tableau des intervalles.
- Le jeu de référence (`tests/regression`) passe toujours, avec la même
  tolérance : il pilote désormais les contrôleurs de `ecg/app`.
- `tests/core/test_core_ui_free.py` vérifie qu'aucun module de `ecg/core`,
  `ecg/services`, `ecg/figures` et `ecg/app` n'importe Qt ou Tk, et que plus
  rien n'importe `ecg.ui`.

**Fonctions qui n'existaient que dans Tk, portées dans Qt avant le retrait :**
- Check agreement (le calcul déplacé dans `ecg/services/agreement_service.py`) ;
- la calibration ks / kp ;
- la suggestion d'annotations à partir des canaux du fichier ;
- la fiche d'enregistrement (sujet, sexe, projet, notes) ;
- les fichiers récents et Cohort Trends ;
- dans Compare segments : les configurations relatives au stimulus, Prism
  (nouveau fichier ou modèle rempli), les figures, Add to group table, le
  Poincaré de groupe et le réglage gardé dans les sessions ;
- Group > From saved sessions et ses exports Prism ;
- Recompute Frequency / Non-linear et le plafond d'entropie ;
- les marqueurs de pics et Review ;
- Clear Edits, Go, la fenêtre d'analyse et la bande RR / HR / Quality ;
- dans Events : Previous / Next et Edit peaks here ;
- les bandeaux « out of date » avec Recompute, et l'état de chaque onglet ;
- dans Morphology : les trois étapes et le tableau des intervalles ;
- le clair / sombre et la taille du texte, gardés d'un lancement à l'autre ;
- les raccourcis clavier ;
- Reset to mouse ECG defaults et Clear session cache.

**Tests retirés :** ceux de `tests/ui` qui pilotaient des widgets ou des mixins
Tk (fenêtres, onglets, barre d'onglets…). Les tests de logique de ces fichiers
sont gardés, avec leurs imports redirigés vers `ecg/app`. Les mêmes fonctions
sont testées dans l'interface Qt (`tests/qt`).

**Corrigé au passage dans l'interface Qt :**
- Les temps des segments étaient affichés avec 6 chiffres significatifs :
  12345,678 devenait 12345,7. Ils gardent maintenant la milliseconde, comme
  dans Tk.
- La fenêtre Qt n'avait pas d'icône.
- Copy for Excel (Events) ne trouvait pas le texte à copier.

