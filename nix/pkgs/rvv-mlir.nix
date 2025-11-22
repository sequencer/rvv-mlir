{
  stdenv,
  lib,
  cmake,
  coreutils,
  python3,
  git,
  fetchFromGitHub,
  ninja,
  lit,
  gitUpdater,
  callPackage,
  rvv-llvm,
}:

let
  pythonEnv = python3.withPackages (ps: [ ps.psutil ]);
in
stdenv.mkDerivation rec {
  name = "rvv-mlir";
  src = ../..;

  requiredSystemFeatures = [ "big-parallel" ];

  nativeBuildInputs = [
    cmake
    ninja
    git
    pythonEnv
  ];
  buildInputs = [
    rvv-llvm
    lit
  ];

  cmakeFlags = [
    "-DMLIR_DIR=${rvv-llvm}/lib/cmake/mlir"
    # LLVM_EXTERNAL_LIT is executed by python3, the wrapped bash script will not work
    "-DLLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped"
  ];

  doCheck = false;

  passthru = {
    llvm = rvv-llvm;
  };

  meta = {
    description = "RVV Compiler";
    homepage = "https://github.com/sequencer/rvv-mlir";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [
      sequencer
    ];
    platforms = lib.platforms.all;
  };
}
