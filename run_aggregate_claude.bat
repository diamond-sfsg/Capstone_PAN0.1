@echo off
setlocal EnableExtensions

REM Usage:
REM   run_aggregate_claude.bat [input_csv] [phase_output_root] [aggregate_output_dir] [claude_model]
REM
REM Example:
REM   run_aggregate_claude.bat data\test\test1_rand\unified_chunks_final_v4.csv outputs\aggregate_test1_rand_claude outout\aggregate\test1_rand_claude

set "INPUT_CSV=%~1"
if "%INPUT_CSV%"=="" set "INPUT_CSV=data\clean_2.0\unified_chunks_final_v4.csv"

set "PHASE_OUTPUT_ROOT=%~2"
if "%PHASE_OUTPUT_ROOT%"=="" set "PHASE_OUTPUT_ROOT=outputs\aggregate_claude"

set "AGGREGATE_OUTPUT_DIR=%~3"
if "%AGGREGATE_OUTPUT_DIR%"=="" set "AGGREGATE_OUTPUT_DIR=outout\aggregate\claude"

set "MODEL=%~4"
if "%MODEL%"=="" set "MODEL=%CLAUDE_MODEL%"
if "%MODEL%"=="" set "MODEL=claude-opus-4-1-20250805"

echo Input CSV: %INPUT_CSV%
echo Phase output root: %PHASE_OUTPUT_ROOT%
echo Aggregate output dir: %AGGREGATE_OUTPUT_DIR%
echo Claude model: %MODEL%
echo.

python src\run_aggregate_test.py ^
  --run-phases ^
  --resume ^
  --input "%INPUT_CSV%" ^
  --phase-output-root "%PHASE_OUTPUT_ROOT%" ^
  --output-dir "%AGGREGATE_OUTPUT_DIR%" ^
  --llm-model "%MODEL%"

if errorlevel 1 (
  echo.
  echo Run failed. Re-run this same .bat command to resume from the saved company_phase progress file.
  exit /b %errorlevel%
)

echo.
echo Done.
echo Summary: %AGGREGATE_OUTPUT_DIR%\Summary.csv
echo Company summary: %AGGREGATE_OUTPUT_DIR%\aggregate_company_summary.csv

endlocal
