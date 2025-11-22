#ifndef RVV_DIALECT
#define RVV_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def RVV_Dialect : Dialect {
  let name = "rvv";
  let summary = "RISC-V Vector Extension dialect";
  let description = [{
    This dialect represents RISC-V Vector Extension operations.
  }];
  let cppNamespace = "::mlir::rvv";
  
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // RVV_DIALECT
