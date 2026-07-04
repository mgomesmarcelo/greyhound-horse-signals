@echo off
REM Arquivo de execucao do gerador de dicas BFB
REM Configure isso no Task Scheduler do Windows Server

cd /d "%~dp0"
python gerar_dicas_bfb.py
