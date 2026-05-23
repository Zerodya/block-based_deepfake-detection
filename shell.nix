{ pkgs ? import <nixpkgs> {} }:

# Shell to run the project in NixOS with AMD ROCm

let
  rocmLibs = with pkgs.rocmPackages; [
    clr
    rocblas
    miopen
    rocm-runtime
    rocm-device-libs
    rocfft
    rocrand
    hipblas
    rccl
  ];
in
(pkgs.buildFHSEnv {
  name = "ml-project-env";
  targetPkgs = pkgs: with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    git
    gcc
    gnumake
    cmake
    ninja
    pkg-config
    stdenv.cc.cc.lib
    zlib
    zstd
    numactl
    openblas
    lapack
    libGL
    glib
    freetype
    libpng
    libjpeg
    libtiff
    libx11
    libxext
    libxrender
    libice
    libsm
    tk
  ] ++ rocmLibs;

  profile = ''
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/rocm6.2.4"
    export HSA_OVERRIDE_GFX_VERSION=10.3.0

    if [ ! -d .venv ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate

    if ! python -c "import IPython" 2>/dev/null; then
      echo "[shell.nix] Installing missing IPython into venv..."
      pip install -q ipython
    fi

    export LD_LIBRARY_PATH=/usr/lib:/lib:$LD_LIBRARY_PATH
  '';

  runScript = "bash";
}).env