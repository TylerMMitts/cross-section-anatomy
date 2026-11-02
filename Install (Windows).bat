@echo off
setlocal EnableDelayedExpansion
rem Double-click once. Installs everything into this folder (.uv\ and .venv\): no admin rights,
rem nothing elsewhere on the computer. Delete the folder to uninstall. Takes ~5-10 min (downloads ~1.5 GB;
rem ~3 GB with an NVIDIA graphics card). Put this folder on the computer's own drive (C:): on a USB drive or
rem exFAT disk the install takes hours (thousands of small files) and the app starts slowly.
cd /d "%~dp0"
if not exist ".uv\uv.exe" (
  echo Getting the installer ^(uv^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%CD%\.uv'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
)
set UV_PYTHON_INSTALL_DIR=%CD%\.uv\python
set UV_CACHE_DIR=%CD%\.uv\cache
rem PyPI's Windows PyTorch has no CUDA, so an NVIDIA card sat unused (tens of minutes per image instead of seconds):
rem "auto" picks the build for this computer's graphics driver (CPU build when there is no NVIDIA card)
set UV_TORCH_BACKEND=auto
echo Installing Python and the packages...
.uv\uv.exe venv --python 3.14 --allow-existing .venv
if errorlevel 1 (
  rem exFAT / FAT32 drives (most USB disks) cannot hold the link uv makes after downloading Python: use the
  rem downloaded Python directly
  for /d %%d in (.uv\python\cpython-3.14*) do if exist "%%d\python.exe" set "CSA_PY=%%d\python.exe"
  if not defined CSA_PY goto :fail
  .uv\uv.exe venv --python "!CSA_PY!" --allow-existing .venv || goto :fail
)
.uv\uv.exe pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail
.venv\Scripts\python.exe -c "import napari, cellpose, pipeline, review, torch; print('Graphics card:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (outlining runs on the processor: slow)'); print('All set. Double-click Open app (Windows).bat to start.')" || goto :fail
copy /y requirements.txt .venv\installed_requirements.txt >nul
pause
exit /b 0
:fail
echo.
echo Something went wrong. Send Jagdeep a photo of this window.
pause
