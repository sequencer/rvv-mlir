# RVV MLIR Dialect: Operand vs Attribute Design

## Design Principle

**Variables as Operator Values, Variants as Attributes**

This document clarifies what should be SSA value operands versus compile-time attributes in the RVV dialect.

## SSA Value Operands (Runtime Dynamic)

These are **values that can change at runtime** and participate in SSA dataflow:

### Vector Data
- `merge` - Passthru/merge operand for inactive elements
- `op1`, `op2`, `src` - Source vector operands
- `index` - Index vectors for gather/scatter
- `value` - Data to store

### Predicates & Configuration
- `mask` - Mask vector (runtime-computed predicate)
- `vl` - Active vector length (changes per loop iteration)

### Scalars & Addresses
- `scalar` - Scalar operands for `.vx` variants
- `ptr` - Memory pointers
- `stride` - Byte stride for strided access
- `offset` - Slide offset
- `avl` - Application vector length for `setvl`

### Why These Are Operands
```mlir
// Example: vl changes dynamically in strip-mining loop
scf.while (%rem = %N, %vl_prev = %vl_init) {
  %vl = rvv.setvl %rem { sew=32, lmul=m1, ta=true, ma=true }
  //            ↑ Runtime value, not known at compile time
  %v1 = rvv.load %ptr, %undef, %all_ones, %vl { ... }
  //                                        ↑ SSA dataflow dependency
  ...
}
```

## Compile-Time Attributes (Static Configuration)

These are **constants known at IR construction time** that affect instruction selection:

### Policy Configuration
- `policy` : `RVV_PolicyAttr` - Tail/Mask undisturbed settings
  - Maps to intrinsic suffix: `none` → no suffix, `tu` → `_tu`, `tumu` → `_tumu`
- `masked` : `BoolAttr` - Whether masking is enabled
  - Maps to intrinsic variant: `false` → `vadd.vv`, `true` → `vadd.vv.m`

### Floating-Point Configuration
- `frm` : `RVV_FRMAttr` - Rounding mode
  - Values: `rne`, `rtz`, `rdn`, `rup`, `rmm`, `dyn`
  - Affects instruction encoding or selects `_rm` intrinsic variant

### Vector Type Configuration (setvl)
- `sew` : `I32Attr` - Selected element width (8, 16, 32, 64)
- `lmul` : `RVV_LMULAttr` - Length multiplier (mf8, mf4, mf2, m1, m2, m4, m8)
- `ta` : `BoolAttr` - Tail agnostic bit
- `ma` : `BoolAttr` - Mask agnostic bit

### Operation Variants
- `kind` : `I32Attr` - Reduction operation type
  - Selects between `vredsum`, `vredmin`, `vredmax`, etc.
- `nf` : `I32Attr` - Number of fields for segment operations
- `ordered` : `BoolAttr` - Ordered vs unordered (FP reductions, indexed access)

### Why These Are Attributes

RVV C intrinsics encode these in the **function name**, not as runtime parameters:

```c
// Policy and masking are in the NAME, not arguments
vint32m1_t vadd_vv_i32m1(vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t vadd_vv_i32m1_tu(vint32m1_t merge, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t vadd_vv_i32m1_m(vbool32_t mask, vint32m1_t op1, vint32m1_t op2, size_t vl);
vint32m1_t vadd_vv_i32m1_tumu(vbool32_t mask, vint32m1_t merge, ...);
```

The MLIR operation:
```mlir
%res = rvv.add %merge, %op1, %op2, %mask, %vl 
       { policy = #rvv.policy<tumu>, masked = true }
```

Lowers to:
```llvm
call @llvm.riscv.vadd.nxv2i32.tumu(...)  // policy encoded in intrinsic name
```

## Lowering Strategy

### Attribute → Intrinsic Name Selection

| MLIR Attribute | Intrinsic Selection |
|---|---|
| `policy = none, masked = false` | `vadd.vv` |
| `policy = tu, masked = false` | `vadd.vv.tu` |
| `policy = none, masked = true` | `vadd.vv.m` |
| `policy = tumu, masked = true` | `vadd.vv.tumu` |

### SSA Value → Intrinsic Argument

```mlir
%res = rvv.add %merge, %op1, %op2, %mask, %vl { policy = tumu, masked = true }
```
↓
```llvm
%res = call <vscale x 2 x i32> @llvm.riscv.vadd.mask.nxv2i32.tumu(
  <vscale x 2 x i32> %merge,   ← SSA value operand
  <vscale x 2 x i32> %op1,     ← SSA value operand
  <vscale x 2 x i32> %op2,     ← SSA value operand
  <vscale x 2 x i1> %mask,     ← SSA value operand
  i64 %vl                       ← SSA value operand
)
```

## Optimization Implications

This design enables key transformations:

### 1. Policy Relaxation
```mlir
// Original (conservative)
%res = rvv.add %old, %a, %b, %mask, %vl { policy = tu }
                ↑ Creates dependency on %old

// After analysis: tail elements unused
%res = rvv.add %undef, %a, %b, %mask, %vl { policy = none }
                ↑ No dependency, better register allocation
```

### 2. VL Dataflow Analysis
```mlir
%vl1 = rvv.setvl %avl { ... }
%v1 = rvv.add ..., %vl1 { ... }
%v2 = rvv.mul ..., %vl1 { ... }  // Compiler sees %vl1 is identical
// → Backend can elide redundant vsetvli instructions
```

### 3. Constant Folding
```mlir
// sew, lmul are attributes → can be folded at compile time
%vl = rvv.setvl %c_128 { sew = 32, lmul = m1, ... }
// → If AVL is constant, vl can be computed as constant
```

## Summary Table

| Category | Examples | Representation | Rationale |
|----------|----------|----------------|-----------|
| **Vector Data** | merge, op1, op2, src | SSA Value | Runtime values, dataflow dependencies |
| **Runtime Config** | vl, mask, avl | SSA Value | Changes dynamically (e.g., per loop iter) |
| **Memory Params** | ptr, stride, offset | SSA Value | Runtime-computed addresses |
| **Policy Bits** | policy, masked, ta, ma | Attribute | Selects intrinsic variant by name |
| **Type Config** | sew, lmul | Attribute | Part of vtype encoding |
| **Op Variants** | kind, nf, ordered | Attribute | Selects different operations/intrinsics |
| **FP Config** | frm | Attribute | Affects instruction encoding |

## References

- **Rational Design Document**: `doc/rational.typ` - Section "Operational Semantics and Attribute Strategy"
- **RISC-V Vector Intrinsic Spec**: Documents function naming conventions
- **LLVM RVV Backend**: `llvm/include/llvm/IR/IntrinsicsRISCV.td` - Intrinsic definitions
