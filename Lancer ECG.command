#!/bin/bash
# ===========================================================================
#  Lancer ECG.command -- lanceur "un clic" de Tom ECG Analysis (macOS)
#
#  Double-cliquez sur ce fichier dans le Finder. Au premier lancement
#  (connexion Internet requise), il installe l'outil "uv", un Python 3.12
#  prive et les bibliotheques de l'application. Les lancements suivants
#  demarrent directement l'application, meme hors ligne.
#
#  PREMIER LANCEMENT - avertissement de securite macOS (Gatekeeper) :
#    macOS 15 (Sequoia) ou plus recent : double-cliquez une fois, cliquez
#    sur "OK", puis ouvrez Reglages Systeme > Confidentialite et securite,
#    descendez et cliquez sur "Ouvrir quand meme" a cote de
#    "Lancer ECG.command", puis confirmez avec votre mot de passe.
#    macOS 14 ou plus ancien : clic droit (ou Ctrl+clic) sur
#    "Lancer ECG.command" > Ouvrir, puis confirmez "Ouvrir".
#    A faire une seule fois.
#    Si cela ne suffit pas, ouvrez le Terminal et tapez :
#      xattr -d com.apple.quarantine "/chemin/vers/Lancer ECG.command"
#    (glissez le fichier dans la fenetre du Terminal pour ecrire le chemin).
#    Si le message dit que le fichier n'est pas executable :
#      chmod +x "/chemin/vers/Lancer ECG.command"
#
#  Environnement prive : ~/Library/Application Support/TomECG (Python 3.12 +
#  bibliotheques ; supprimez ce dossier pour tout reinstaller). uv garde aussi
#  un cache dans ~/.cache/uv ("uv cache clean" le vide). Variables optionnelles :
#    TOM_ECG_HOME            autre dossier pour l'environnement
#    UV_PYTHON_INSTALL_DIR   autre dossier pour Python (defaut TOM_ECG_HOME/python)
#    UV_CACHE_DIR            autre dossier pour le cache de uv
#  Mode diagnostic (Terminal) :  "./Lancer ECG.command" debug
#
#  NOTE : ce script n'a pas pu etre teste sur un Mac (ecrit sous Windows).
# ===========================================================================
set -u

fail() {
    echo
    echo "  ERREUR : $*"
    echo
    read -r -p "  Appuyez sur Entree pour fermer cette fenetre..." _ || true
    exit 1
}

# Se placer dans le dossier du script (le Finder lance depuis ~).
cd "$(dirname "$0")" || fail "impossible d'acceder au dossier du lanceur."

if [ ! -f "ecg_app.py" ] || [ ! -f "requirements.txt" ] || [ ! -f "packaging/constraints.txt" ]; then
    fail "ce fichier doit rester dans le dossier de l'application
  (a cote de ecg_app.py et requirements.txt). Si vous avez telecharge le
  ZIP depuis GitHub, decompressez-le entierement puis relancez."
fi

TOM_ECG_HOME="${TOM_ECG_HOME:-$HOME/Library/Application Support/TomECG}"
# Python 3.12 prive, range lui aussi dans le dossier TomECG.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$TOM_ECG_HOME/python}"
VENV="$TOM_ECG_HOME/venv"
PY="$VENV/bin/python"
STAMP="$TOM_ECG_HOME/requirements.sha256"
LOG="$TOM_ECG_HOME/dernier_lancement.log"
# Incrementer LAYOUT force une reinstallation chez tout le monde.
LAYOUT="py312-v3"
# uv prive dans TomECG : version connue, PATH de l'utilisateur non modifie
# (un vieux uv deja installe ne connait pas toutes les options ci-dessous).
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$TOM_ECG_HOME/uv}"
export UV_NO_MODIFY_PATH=1

HASH="$(shasum -a 256 requirements.txt 2>/dev/null | awk '{print $1}')"
HASH2="$(shasum -a 256 packaging/constraints.txt 2>/dev/null | awk '{print $1}')"
[ -n "$HASH" ] && [ -n "$HASH2" ] || fail "impossible de lire requirements.txt ou packaging/constraints.txt (shasum)."
WANT="$LAYOUT-$HASH-$HASH2"
HAVE=""
[ -f "$STAMP" ] && HAVE="$(head -n 1 "$STAMP")"

find_uv() {
    # Uniquement le uv prive (installe par ce lanceur dans UV_INSTALL_DIR).
    UV="$UV_INSTALL_DIR/uv"
    [ -x "$UV" ] && return 0
    UV=""
    return 1
}

install_env() {
    echo
    echo "  ============================================================"
    echo "   Tom ECG Analysis - installation / mise a jour"
    echo "  ============================================================"
    echo "   Une connexion Internet est necessaire pour cette etape."
    echo "   Premiere installation : quelques minutes (environ 400 Mo)."
    echo "   Dossier utilise : $TOM_ECG_HOME"
    echo "  ------------------------------------------------------------"
    echo
    mkdir -p "$TOM_ECG_HOME" || fail "impossible de creer le dossier $TOM_ECG_HOME"

    if find_uv; then
        echo "  [1/4] Outil \"uv\" deja present."
    else
        echo "  [1/4] Installation de l'outil \"uv\" (gestionnaire Python)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        find_uv || fail "l'installation de \"uv\" a echoue.
  Verifiez la connexion Internet (proxy / pare-feu ?) puis relancez.
  Installation manuelle possible : https://docs.astral.sh/uv/"
    fi
    echo "        $UV"
    echo

    echo "  [2/4] Installation de Python 3.12 (prive, n'affecte pas le systeme)..."
    # --no-bin : ne pas ajouter de commande python3.12 dans ~/.local/bin.
    "$UV" python install 3.12 --no-bin \
        || fail "le telechargement de Python 3.12 a echoue. Verifiez la connexion Internet."
    echo

    echo "  [3/4] Preparation de l'environnement et des bibliotheques..."
    # Environnement recree a chaque mise a jour (rapide : uv garde un cache).
    # --managed-python : jamais le Python d'Apple (Tk 8.5) ni celui du systeme.
    "$UV" venv --clear --managed-python --python 3.12 "$VENV" \
        || fail "creation de l'environnement impossible dans $VENV.
  Supprimez ce dossier puis relancez."
    # -c : les versions testees (packaging/constraints.txt), pas les plus recentes.
    "$UV" pip install --python "$PY" -r requirements.txt -c packaging/constraints.txt \
        || fail "l'installation des bibliotheques a echoue (voir ci-dessus).
  Le plus souvent : connexion Internet coupee ou bloquee. Relancez ; si
  l'erreur persiste, supprimez le dossier $TOM_ECG_HOME et reessayez."
    echo

    echo "  [4/4] Verification de l'application..."
    "$PY" -c "import tkinter, customtkinter, neurokit2, ecg.ui.app" \
        || fail "les bibliotheques sont installees mais l'application ne se
  charge pas (voir le message Python ci-dessus). Envoyez une capture de
  cette fenetre au responsable de l'application."

    printf '%s\n' "$WANT" > "$STAMP"
    echo
    echo "  Installation terminee. Demarrage de l'application..."
    echo
}

if [ ! -x "$PY" ] || [ "$HAVE" != "$WANT" ]; then
    install_env
fi

if [ "${1:-}" = "debug" ]; then
    echo "  Mode diagnostic : erreurs affichees ci-dessous."
    "$PY" ecg_app.py
    echo
    read -r -p "  Application fermee. Appuyez sur Entree..." _ || true
    exit 0
fi

# Demarrage detache : l'application continue si la fenetre Terminal est
# fermee. Les erreurs Python eventuelles vont dans $LOG.
nohup "$PY" ecg_app.py < /dev/null > "$LOG" 2>&1 &
disown 2>/dev/null || true
sleep 2
if ! kill -0 $! 2>/dev/null; then
    tail -n 30 "$LOG" 2>/dev/null
    fail "l'application s'est arretee au demarrage (voir le journal ci-dessus :
  $LOG)."
fi
echo "  Application demarree. Vous pouvez fermer cette fenetre Terminal."
exit 0
