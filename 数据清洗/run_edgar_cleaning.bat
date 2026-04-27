@echo off
setlocal

cd /d "%~dp0"

echo ========================================
echo EDGAR data cleaning
echo ========================================
echo.

if not exist "data\10Ks\edgar" (
    echo [ERROR] Cannot find data\10Ks\edgar
    echo Please put EDGAR files under data\10Ks\edgar first.
    echo.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python or add it to PATH.
    echo.
    pause
    exit /b 1
)

echo Input:
echo   data\10Ks\edgar
echo Output:
echo   output\cleaned\edgar_by_type
echo.

python scripts\clean_edgar_purpose_text.py --input data\10Ks\edgar --output output\cleaned\edgar_by_type --progress

if errorlevel 1 (
    echo.
    echo [ERROR] Cleaning failed.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Cleaning completed
echo ========================================
echo.

echo Output summary:
for /d %%D in ("output\cleaned\edgar_by_type\*") do (
    for /f %%C in ('dir /b /s /a-d "%%D" 2^>nul ^| find /c /v ""') do echo   %%~nxD: %%C files
)

echo.
echo Open this folder to view results:
echo   output\cleaned\edgar_by_type
echo.
pause
