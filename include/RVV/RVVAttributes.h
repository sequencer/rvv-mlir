#ifndef RVV_RVVATTRIBUTES_H
#define RVV_RVVATTRIBUTES_H

#include "RVV/RVVDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "RVV/RVVAttributes.h.inc"

#endif // RVV_RVVATTRIBUTES_H
