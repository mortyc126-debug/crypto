@echo off
setlocal

REM ===== Adjust these paths for your system =====
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set MSVC_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64

REM ===== Build =====
echo Building ErgoMiner v0.7...

"%CUDA_PATH%\bin\nvcc.exe" ^
    -O3 ^
    -arch=sm_86 ^
    -std=c++17 ^
    --allow-unsupported-compiler ^
    -ccbin "%MSVC_PATH%\cl.exe" ^
    -Xcompiler "/O2 /W1" ^
    main.cu ^
    -o ergominer.exe ^
    -lws2_32

if %ERRORLEVEL% == 0 (
    echo.
    echo Build SUCCESS: ergominer.exe
    echo.
    echo Fixes in v0.7:
    echo   - nonce byte order fixed: now hashed as big-endian ^(matches pool^)
    echo   - GPU batch size 114688 ^(was 14336^) for ~4x hashrate
    echo.
    echo Usage: ergominer.exe
    echo Pool: erg.2miners.com:8888
) else (
    echo.
    echo Build FAILED
)
