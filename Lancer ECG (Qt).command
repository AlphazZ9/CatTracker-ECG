#!/bin/bash
# Lance l'interface Qt (PySide6) de Tom ECG Analysis, avec le meme Python
# prive et les memes bibliotheques que "Lancer ECG.command".
export ECG_INTERFACE=qt
exec "$(cd "$(dirname "$0")" && pwd)/Lancer ECG.command" "$@"
