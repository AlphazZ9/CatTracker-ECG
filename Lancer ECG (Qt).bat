@echo off
rem Lance l'interface Qt (PySide6) de Tom ECG Analysis, avec le meme
rem Python prive et les memes bibliotheques que "Lancer ECG.bat".
set "ECG_INTERFACE=qt"
call "%~dp0Lancer ECG.bat" %*
