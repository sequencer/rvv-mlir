#ifndef RVV_RVVOPS_H
#define RVV_RVVOPS_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "RVV/RVVAttributes.h"
#include "RVV/RVVDialect.h"
#include "RVV/RVVInterfaces.h"
#include "RVV/RVVTypes.h"

namespace mlir {
namespace rvv {
// Workaround for generated code using unqualified RVV_FRM_DYN
constexpr FRM RVV_FRM_DYN = FRM::DYN;
} // namespace rvv
} // namespace mlir

#define GET_OP_CLASSES
#include "RVV/RVVOps.h.inc"

#endif // RVV_RVVOPS_H
