set(LLVM_TARGET_DEFINITIONS RVV.td)

mlir_tablegen(RVVEnums.h.inc -gen-enum-decls)
mlir_tablegen(RVVEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(RVVOps.h.inc -gen-op-decls)
mlir_tablegen(RVVOps.cpp.inc -gen-op-defs)
mlir_tablegen(RVVAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=rvv)
mlir_tablegen(RVVAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=rvv)

set(LLVM_TARGET_DEFINITIONS RVVInterfaces.td)
mlir_tablegen(RVVInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(RVVInterfaces.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS RVV.td)

mlir_tablegen(RVVTypes.h.inc -gen-typedef-decls -typedefs-dialect=rvv)
mlir_tablegen(RVVTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=rvv)
mlir_tablegen(RVVDialect.h.inc -gen-dialect-decls -dialect=rvv)
mlir_tablegen(RVVDialect.cpp.inc -gen-dialect-defs -dialect=rvv)

add_public_tablegen_target(RVVIncGen)

set(LLVM_TARGET_DEFINITIONS RVVPasses.td)
mlir_tablegen(RVVPasses.h.inc -gen-pass-decls -name RVV)
add_public_tablegen_target(RVVPassesIncGen)
