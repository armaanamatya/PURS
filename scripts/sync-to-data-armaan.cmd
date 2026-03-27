@echo off
setlocal
REM Run from repo root OR from this folder. Uses PowerShell wrapper (WSL / Git Bash + rsync).
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync-to-data-armaan.ps1" %*
