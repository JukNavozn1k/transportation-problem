@echo off
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)
streamlit run main.py
pause