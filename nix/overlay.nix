final: prev:

rec {
  circt-llvm = final.callPackage "${final.path}/pkgs/by-name/ci/circt/circt-llvm.nix" { };

  rvv-llvm = circt-llvm.dev;

  rvv-mlir = final.callPackage ./pkgs/rvv-mlir.nix { };
}
