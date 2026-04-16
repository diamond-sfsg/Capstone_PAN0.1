@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%purpose_driven_scoring"
set "BATCH_SCRIPT=%PROJECT_DIR%\src\batch_score.py"
set "SUMMARY_FILE=%PROJECT_DIR%\data\outputs\batch\all_companies_summary.csv"
set "OPENAI_FLAG="

if not "%OPENAI_API_KEY%"=="" (
    set "OPENAI_FLAG=--enable-openai-judge"
)

echo Running purpose-driven scoring pipeline...
python "%BATCH_SCRIPT%" %OPENAI_FLAG%
if errorlevel 1 (
    echo.
    echo Scoring failed. Please review the terminal output above.
    pause
    exit /b 1
)

echo.
echo Scoring complete.

if exist "%SUMMARY_FILE%" (
    echo Opening summary file...
    start "" "%SUMMARY_FILE%"
) else (
    echo Summary file was not found at:
    echo %SUMMARY_FILE%
)

pause
