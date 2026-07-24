@echo off
setlocal
cd /d "%~dp0"
"C:\Users\kjy26\miniconda3\envs\KMAP\python.exe" -m streamlit run app.py --server.port 8503 --server.address localhost
