#ifndef RVV_OPS
#define RVV_OPS

include "RVV/RVVDialect.td"
include "RVV/RVVInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//
//
// Design Principle: Variables as Operator Values, Variants as Attributes
//
// SSA Values (Runtime Dynamic):
//   - Vector operands (op1, op2, src): Can vary per execution
//   - Mask operands: Runtime-computed predicates
//   - Vector length (vl): Changes dynamically in strip-mining loops
//   - Scalar operands: Runtime values for .vx variants
//   - Pointers, strides, offsets: Memory addressing parameters
//
// Attributes (Compile-Time Constants):
//   - tail_policy: Determines tail behavior (agnostic/undisturbed)
//   - mask_policy: Determines mask behavior (agnostic/undisturbed)
//   - frm: Rounding mode for FP ops (affects instruction encoding)
//   - sew, lmul: Type configuration for setvl operations
//   - nf: Number of fields for segment operations
//   - kind: Selects between different reduction operations
//
// This separation enables:
//   1. Proper SSA dataflow analysis (vl dependencies explicit)
//   2. Correct instruction selection (attributes map to intrinsic names)
//   3. Policy relaxation optimizations (tu -> ta when safe)
//
//===----------------------------------------------------------------------===//
class RVV_Op<string mnemonic, list<Trait> traits = []> :
    Op<RVV_Dialect, mnemonic, traits> {
  
  let extraClassDeclaration = [{
    std::string getIntrinsicName();
  }];
}
include "RVVIntegerOps.td"
include "RVVFloatOps.td"
include "RVVMemoryOps.td"


//===----------------------------------------------------------------------===//
// Configuration-Setting Instructions (Section 6)
//===----------------------------------------------------------------------===//

def RVV_SetvlOp : RVV_Op<"setvl", [DeclareOpInterfaceMethods<RVVVLProducerInterface>]> {
  let summary = "Set vector length and type";
  let description = [{
    Configures the vtype CSR and returns the active vector length (vl).
    Maps to the vsetvl instruction.
    
    The AVL (application vector length) is an SSA value (runtime dynamic),
    while SEW, LMUL, and policies are attributes (compile-time constants)
    that determine the vtype encoding.
    
    Example:
    ```mlir
    %vl = rvv.setvl %avl { sew = 32, lmul = m2, ta = true, ma = true }
    ```
  }];
  let arguments = (ins
    I64:$avl,                  // Application vector length (requested elements)
    I32Attr:$sew,              // Selected element width (8, 16, 32, 64)
    RVV_LMULAttr:$lmul,        // Length multiplier (mf8, mf4, mf2, m1, m2, m4, m8)
    BoolAttr:$ta,              // Tail agnostic policy bit
    BoolAttr:$ma               // Mask agnostic policy bit
  );
  let results = (outs I64:$new_vl);
  let assemblyFormat = "$avl attr-dict `:` type($new_vl)";
}

def RVV_SetvlMaxOp : RVV_Op<"setvl_max", [DeclareOpInterfaceMethods<RVVVLProducerInterface>]> {
  let summary = "Set vector length to VLMAX";
  let description = [{
    Configures the vtype CSR and sets vl to VLMAX (maximum vector length).
    Maps to vsetvlmax intrinsics.
    
    Example:
    ```mlir
    %vl = rvv.setvl_max { sew = 32, lmul = m1 }
    ```
  }];
  let arguments = (ins
    I32Attr:$sew,              // Selected element width
    RVV_LMULAttr:$lmul         // Length multiplier
  );
  let results = (outs I64:$new_vl);
  let assemblyFormat = "attr-dict `:` type($new_vl)";
}

#endif // RVV_OPS
