@echo off
rem ===========================================================================
rem  Lancer ECG.bat -- lanceur "un clic" de Tom ECG Analysis (Windows)
rem
rem  Double-cliquez sur ce fichier. Au premier lancement (connexion Internet
rem  requise), il installe l'outil "uv", un Python 3.12 prive et les
rem  bibliotheques de l'application. Les lancements suivants demarrent
rem  directement l'application, meme hors ligne.
rem
rem  Environnement prive : %LOCALAPPDATA%\TomECG  (Python 3.12 + bibliotheques ;
rem  supprimez ce dossier pour tout reinstaller). uv garde aussi un cache de
rem  telechargement dans %LOCALAPPDATA%\uv\cache ("uv cache clean" le vide).
rem  Variables optionnelles (tests / utilisateurs avances) :
rem    TOM_ECG_HOME            autre dossier pour l'environnement
rem    UV_PYTHON_INSTALL_DIR   autre dossier pour Python (defaut TOM_ECG_HOME\python)
rem    UV_CACHE_DIR            autre dossier pour le cache de uv
rem  Mode diagnostic : lancez  "Lancer ECG.bat" debug  depuis une invite de
rem  commandes pour garder la console et voir les erreurs Python.
rem ===========================================================================
setlocal EnableExtensions DisableDelayedExpansion
title Tom ECG Analysis
cd /d "%~dp0"

if not exist "ecg_app.py" goto :err_layout
if not exist "requirements.txt" goto :err_layout
if not exist "packaging\constraints.txt" goto :err_layout

if not defined TOM_ECG_HOME set "TOM_ECG_HOME=%LOCALAPPDATA%\TomECG"
set "VENV=%TOM_ECG_HOME%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"
set "STAMP=%TOM_ECG_HOME%\requirements.sha256"
set "LOG=%TOM_ECG_HOME%\dernier_lancement.log"
rem Python 3.12 prive, range lui aussi dans le dossier TomECG.
if not defined UV_PYTHON_INSTALL_DIR set "UV_PYTHON_INSTALL_DIR=%TOM_ECG_HOME%\python"
rem Incrementer LAYOUT force une reinstallation chez tout le monde.
set "LAYOUT=py312-v4"
rem uv prive dans TomECG : version connue, PATH de l'utilisateur non modifie
rem (un vieux uv deja installe ne connait pas toutes les options ci-dessous).
if not defined UV_INSTALL_DIR set "UV_INSTALL_DIR=%TOM_ECG_HOME%\uv"
set "UV_NO_MODIFY_PATH=1"

rem --- Empreinte de requirements.txt et des versions figees ----------------------
set "HASH="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "requirements.txt" SHA256 2^>nul') do (
    if not defined HASH set "HASH=%%H"
)
if not defined HASH goto :err_hash
set "HASH=%HASH: =%"
set "HASH2="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "packaging\constraints.txt" SHA256 2^>nul') do (
    if not defined HASH2 set "HASH2=%%H"
)
if not defined HASH2 goto :err_hash
set "HASH2=%HASH2: =%"
set "WANT=%LAYOUT%-%HASH%-%HASH2%"

set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"
if not exist "%PYW%" goto :install
if not "%HAVE%"=="%WANT%" goto :install
call :find_base
if not defined BASEW goto :install
goto :launch

rem ============================================================================
:install
echo.
echo  ============================================================
echo   Tom ECG Analysis - installation / mise a jour
echo  ============================================================
echo   Une connexion Internet est necessaire pour cette etape.
echo   Premiere installation : quelques minutes (environ 400 Mo).
echo   Dossier utilise : %TOM_ECG_HOME%
echo  ------------------------------------------------------------
echo.
if not exist "%TOM_ECG_HOME%" mkdir "%TOM_ECG_HOME%" 2>nul
if not exist "%TOM_ECG_HOME%" goto :err_mkdir

call :find_uv
if defined UV goto :have_uv
echo  [1/4] Installation de l'outil "uv" (gestionnaire Python)...
powershell -NoProfile -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
call :find_uv
if not defined UV goto :err_uv
goto :uv_ok
:have_uv
echo  [1/4] Outil "uv" deja present.
:uv_ok
echo        %UV%
echo.

echo  [2/4] Installation de Python 3.12 (prive, n'affecte pas le systeme)...
rem --no-bin / --no-registry : ne rien ajouter au PATH ni au registre Windows.
"%UV%" python install 3.12 --no-bin --no-registry
if errorlevel 1 goto :err_python
echo.

echo  [3/4] Preparation de l'environnement et des bibliotheques...
rem Environnement recree a chaque mise a jour (rapide : uv garde un cache).
rem --managed-python : jamais un Python du systeme, toujours celui de uv.
"%UV%" venv --clear --managed-python --python 3.12 "%VENV%"
if errorlevel 1 goto :err_venv
rem -c : les versions testees (packaging\constraints.txt), pas les plus recentes.
"%UV%" pip install --python "%PY%" -r requirements.txt -c packaging\constraints.txt
if errorlevel 1 goto :err_deps
echo.

echo  [4/4] Verification de l'application...
"%PY%" -c "import PySide6, pyqtgraph, neurokit2, ecg.qt.main_window"
if errorlevel 1 goto :err_check
call :find_base
if not defined BASEW goto :err_check

> "%STAMP%" echo %WANT%
echo.
echo  Installation terminee. Demarrage de l'application...
echo.

rem ============================================================================
:launch
if /i "%~1"=="debug" goto :launch_debug
rem Sans fenetre console : on lance directement le pythonw.exe du Python prive
rem en lui designant l'environnement via __PYVENV_LAUNCHER__ (c'est ce que fait
rem le lanceur du venv, mais celui de uv demarre python.exe, qui ouvre une
rem console, et les processus de calcul en ouvriraient d'autres).
rem Les messages d'erreur Python eventuels vont dans le journal %LOG%.
set "__PYVENV_LAUNCHER__=%PYW%"
start "" /b "%BASEW%" "%~dp0ecg_app.py" > "%LOG%" 2>&1
if not errorlevel 1 goto :launched
rem Journal verrouille (application deja ouverte ?) : lancer sans journal.
start "" "%BASEW%" "%~dp0ecg_app.py"
if errorlevel 1 goto :err_launch
:launched
endlocal
exit /b 0

:launch_debug
echo  Mode diagnostic : la console reste ouverte, erreurs affichees ci-dessous.
"%PY%" "%~dp0ecg_app.py"
echo.
echo  Application fermee (code %errorlevel%).
pause
endlocal
exit /b 0

rem ============================================================================
rem  BASEW = pythonw.exe du Python 3.12 prive (dossier "cpython-3.12-windows-*",
rem  lien stable gere par uv, insensible aux mises a jour 3.12.x).
:find_base
set "BASEW="
for /d %%D in ("%UV_PYTHON_INSTALL_DIR%\cpython-3.12-windows-*") do (
    if not defined BASEW if exist "%%~fD\pythonw.exe" set "BASEW=%%~fD\pythonw.exe"
)
exit /b 0

rem ============================================================================
:find_uv
rem Uniquement le uv prive (installe par ce lanceur dans UV_INSTALL_DIR).
set "UV="
if exist "%UV_INSTALL_DIR%\uv.exe" set "UV=%UV_INSTALL_DIR%\uv.exe"
exit /b 0

rem ============================================================================
rem  Messages d'erreur (la fenetre reste ouverte grace a "pause")
rem ============================================================================
:err_layout
echo.
echo  ERREUR : ce fichier doit rester dans le dossier de l'application
echo  (a cote de ecg_app.py, requirements.txt et du dossier packaging).
echo  Si vous avez telecharge
echo  le ZIP depuis GitHub, decompressez-le entierement d'abord
echo  (clic droit ^> Extraire tout), puis relancez depuis le dossier extrait.
goto :fail

:err_hash
echo.
echo  ERREUR : impossible de lire requirements.txt ou packaging\constraints.txt
echo  (commande certutil).
goto :fail

:err_mkdir
echo.
echo  ERREUR : impossible de creer le dossier "%TOM_ECG_HOME%".
goto :fail

:err_uv
echo.
echo  ERREUR : l'installation de "uv" a echoue.
echo  Verifiez la connexion Internet (proxy / pare-feu de l'institut ?)
echo  puis relancez. Installation manuelle possible : https://docs.astral.sh/uv/
goto :fail

:err_python
echo.
echo  ERREUR : le telechargement de Python 3.12 a echoue.
echo  Verifiez la connexion Internet puis relancez ce fichier.
goto :fail

:err_venv
echo.
echo  ERREUR : creation de l'environnement impossible dans
echo  "%VENV%". Supprimez ce dossier puis relancez.
goto :fail

:err_deps
echo.
echo  ERREUR : l'installation des bibliotheques a echoue (voir messages
echo  ci-dessus). Le plus souvent : connexion Internet coupee ou bloquee.
echo  Relancez ce fichier ; si l'erreur persiste, supprimez le dossier
echo  "%TOM_ECG_HOME%" et reessayez.
goto :fail

:err_check
echo.
echo  ERREUR : les bibliotheques sont installees mais l'application ne
echo  se charge pas (voir le message Python ci-dessus). Envoyez une
echo  capture de cette fenetre au responsable de l'application.
goto :fail

:err_launch
echo.
echo  ERREUR : impossible de demarrer l'application. Supprimez le dossier
echo  "%TOM_ECG_HOME%" puis relancez ce fichier.
goto :fail

:fail
echo.
pause
endlocal
exit /b 1
