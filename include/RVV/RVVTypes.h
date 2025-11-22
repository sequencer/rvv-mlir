#ifndef RVV_RVVTYPES_H
#define RVV_RVVTYPES_H

#include "RVV/RVVDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "RVV/RVVTypes.h.inc"

namespace mlir {
namespace rvv {

/// Get the element width in bits for a scalar type (i32, i64, f32, f64)
/// Returns 0 if the type is not a valid RVV scalar type.
unsigned getScalarElementWidth(Type scalarType);

} // namespace rvv
} // namespace mlir

#endif // RVV_RVVTYPES_H
