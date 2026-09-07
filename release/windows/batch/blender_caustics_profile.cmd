@echo off
setlocal
set "CYCLESPLUS_CAUSTICS_PROFILE=1"
set "CYCLES_DEBUG_PER_KERNEL_PERFORMANCE="
set "CYCLESPLUS_CAUSTICS_PROFILE_DIR=%TEMP%\Blender_Caustics_Profile"
if not exist "%CYCLESPLUS_CAUSTICS_PROFILE_DIR%" mkdir "%CYCLESPLUS_CAUSTICS_PROFILE_DIR%"
if not exist "%CYCLESPLUS_CAUSTICS_PROFILE_DIR%" (
  echo Cannot create profiling folder.
  pause
  exit /b 1
)
if not exist "%~dp0blender.exe" (
  echo Run this launcher from the installed Blender folder.
  pause
  exit /b 1
)
echo Caustics profiling enabled. Logs: %CYCLESPLUS_CAUSTICS_PROFILE_DIR%
echo Close Blender normally after reproducing the slowdown.
"%~dp0blender.exe" %* > "%CYCLESPLUS_CAUSTICS_PROFILE_DIR%\console_%RANDOM%_%RANDOM%.log" 2>&1
set "BLENDER_EXIT_CODE=%ERRORLEVEL%"
explorer "%CYCLESPLUS_CAUSTICS_PROFILE_DIR%"
exit /b %BLENDER_EXIT_CODE%
