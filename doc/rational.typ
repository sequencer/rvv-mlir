#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
)
#set text(
  font: "New Computer Modern",
  size: 11pt,
)

= Rational Design Specification for a RISC-V Vector (RVV) MLIR Dialect: Intrinsic Alignment and Operator-Attribute Semantics

== Executive Summary

The emergence of the RISC-V Vector (RVV) extension, particularly the ratified version 1.0, represents a paradigm shift in the architecture of parallel computing systems. Unlike the Single Instruction, Multiple Data (SIMD) extensions that have dominated the landscape for decades—such as Intel's AVX or ARM's NEON—the RISC-V Vector extension adopts a Vector Length Agnostic (VLA) philosophy. This architectural decision decouples the software instruction stream from the physical vector register length (VLEN) of the hardware, allowing a single binary to scale performance across a diverse ecosystem ranging from embedded microcontrollers to massive high-performance computing (HPC) clusters. However, this flexibility introduces profound challenges for compiler infrastructure, particularly in the mapping of high-level loop abstractions to the specific, state-dependent instructions required by the hardware.

This report presents an exhaustive rational design for a specialized MLIR (Multi-Level Intermediate Representation) dialect: the `rvv` dialect. The primary design mandate is to create a representation that is strictly aligned with the RVV C Intrinsic API v1.0, providing a fidelity of representation that generic vector dialects currently lack. While high-level dialects like `vector` or `linalg` offer target-neutral abstractions, they fail to capture the critical architectural nuances of RVV, such as Register Grouping (LMUL), dynamic tail policies, and the precise semantics of mask layouts. The proposed `rvv` dialect bridges this semantic gap by adhering to a strict design philosophy: variables are operator values, and variants are attributes.

The analysis underpinning this design draws comprehensively from the RVV Intrinsic Specification and LLVM backend implementations. Key features of the proposed design include:

- *Explicit Type System*: A custom `!rvv.vector` type that encodes both element type and LMUL, resolving the ambiguity of generic scalable vectors.
- *SSA-Based State Management*: The transformation of implicit architectural state—specifically the active vector length (`vl`) and the vector type register (`vtype`)—into explicit Single Static Assignment (SSA) values (`%vl`), enabling robust dataflow analysis.
- *Attribute-Driven Semantics*: The encapsulation of functional variants (Tail Undisturbed vs. Agnostic, Masking) into static attributes, preventing opcode explosion while preserving the granular control required for manual optimization.

By strictly mirroring the C intrinsic API while leveraging the verification and transformation power of MLIR, this design provides a stable, optimizing intermediate layer. It enables the compiler to reason about "strip-mining" loops, redundant `vsetvli` elimination, and register pressure management with a level of precision that is unattainable in higher-level, architecture-agnostic representations.

== Architectural Context and the Semantic Gap

To appreciate the necessity of a dedicated `rvv` dialect, one must first dissect the architectural uniqueness of RISC-V Vectors and the inadequacies of existing generic representations in capturing them.

=== The Vector Length Agnostic (VLA) Paradigm

Traditional SIMD architectures are characterized by fixed-width registers. In the x86 AVX-512 instruction set, for instance, a vector register is architecturally defined as 512 bits wide. Compilers targeting this architecture perform "pack-based" vectorization, generating instructions that operate on exactly 16 32-bit integers at a time. If the hardware changes (e.g., down to AVX2's 256 bits), the software must be recompiled.

RISC-V fundamentally rejects this rigidity. The VLEN is a hardware implementation parameter, unknown at compile time. It may be 128 bits, 512 bits, or even 4096 bits. The software interacts with the vector unit not by assuming a width, but by requesting an "Application Vector Length" (AVL). The hardware responds via the `vsetvli` instruction, which returns the number of elements (`vl`) it can process in a single cycle or strip.

This mechanism creates a dynamic state dependency that permeates the entire instruction stream. Every arithmetic operation, memory access, or permutation is implicitly predicated on the current value of the `vl` register. In generic IRs, this dependency is often implicit or modeled as a side effect, which severely hinders optimization. If a compiler cannot prove that `vl` remains constant between two operations, it cannot safely reorder them. The proposed design addresses this by making `vl` an explicit operand for every vector instruction, reifying the dependency into the SSA graph.

=== Register Grouping (LMUL): The Scalability Multiplier

Perhaps the most distinctive feature of RVV is the Length Multiplier (LMUL). This feature allows the software to group multiple physical vector registers (e.g., v1, v2, v3, v4) to act as a single logical register, or conversely, to use only a fraction of a register.

- *LMUL > 1 (Grouping)*: Increases the effective vector length ($"VLMAX" = ("VLEN")/("SEW") times "LMUL"$). This is crucial for amortization of instruction fetch and decode overheads, effectively mimicking a longer vector machine.
- *LMUL < 1 (Fractional)*: Allows a single register to hold multiple small vectors, which is vital for reducing register pressure when operating on mixed-width types (e.g., widening multiplication).

Generic MLIR scalable vectors (`vector<xi32>`) struggle to represent this. They model scalability via a generic `vscale` term, but they lack the semantics to distinguish between a vector that occupies 1 register (LMUL=1) and one that occupies 2 registers (LMUL=2). This distinction is not merely a backend detail; it fundamentally dictates the legality of operations. For instance, a widening operation from LMUL=8 is illegal because the result would require LMUL=16, which exceeds the architectural limit. The `rvv` dialect must strictly enforce these type constraints at the IR level.

=== The Challenge of Masking and Tails

RVV v1.0 introduces a sophisticated model for handling "inactive" elements—those that are masked off or lie beyond the active vector length (`vl`). The hardware behavior is controlled by the `vtype` register's policy bits: `vta` (Vector Tail Agnostic) and `vma` (Vector Mask Agnostic).

- *Tail Undisturbed (tu)*: The destination register retains its previous values in positions $i >= "vl"$. This allows for the accumulation of results across multiple iterations (e.g., in a fault-only-first load loop) but creates a dependency on the previous register value.
- *Tail Agnostic (ta)*: The hardware may overwrite tail elements with 1s or leave them undefined. This breaks the dependency chain, allowing the register allocator more freedom to reuse registers.

Existing compiler infrastructures often default to "Undisturbed" for safety, or abstract the concept entirely. However, high-performance tuning requires explicit control. Using `tu` when `ta` would suffice can prevent instruction scheduling optimizations and increase register pressure due to the extended liveness of the destination register. The proposed dialect must therefore expose these policies as first-class attributes.

== Type System Rational Design

The foundation of any robust MLIR dialect is its type system. For the `rvv` dialect, the type system must be isomorphic to the data types defined in the C Intrinsic API, ensuring that all constraints regarding Element Width (SEW) and Register Grouping (LMUL) are preserved.

=== The !rvv.vector Type

We introduce a custom parameterized type, `!rvv.vector`, which explicitly encodes the two dimensions of RISC-V vector storage: the data type and the register utilization.

*Syntax*: `!rvv.vector< element_type, lmul >`

- `element_type`: This parameter accepts standard MLIR integer and floating-point types (`i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`, `f64`). It corresponds to the SEW (Selected Element Width) concept in the ISA.
- `lmul`: This parameter is an enumeration attribute representing the LMUL. The valid values are derived directly from the intrinsic specification: `mf8` (1/8), `mf4` (1/4), `mf2` (1/2), `m1` (1), `m2` (2), `m4` (4), `m8` (8).

*Rationale for Deviation from Standard vector Type*: The standard MLIR `vector<[N]xT>` type is insufficient because it conflates the concept of "scalable number of elements" with "physical register usage." Consider a hardware implementation where VLEN = 128 bits.

- `vector<xi32>`: 4 scalable elements of 32-bits. At VLEN=128, this occupies exactly 128 bits (1 register).
- `vector<xi32>`: 8 scalable elements. At VLEN=128, this occupies 256 bits (2 registers).

While this seems mappable, the mapping depends on VLEN. If the compiler compiles for an unknown VLEN, it cannot know if `vector<xi32>` represents LMUL=1 (on 128-bit hardware) or LMUL=0.5 (on 256-bit hardware). The RVV ISA requires the LMUL to be fixed at compile time to generate the correct `vsetvli` instruction encoding. Therefore, the type must carry the LMUL semantic explicitly.

#figure(
  table(
    columns: 5,
    align: (left, center, center, left, left),
    table.header(
      [*C Intrinsic Type*], [*Element Type (SEW)*], [*Grouping (LMUL)*], [*Proposed RVV Dialect Type*], [*Constraints*]
    ),
    [`vint32m1_t`], [i32], [m1], [`!rvv.vector<i32, m1>`], [Standard mapping.],
    [`vfloat64m4_t`], [f64], [m4], [`!rvv.vector<f64, m4>`], [Consumes 4 registers.],
    [`vint8mf8_t`], [i8], [mf8], [`!rvv.vector<i8, mf8>`], [1/8th register usage.],
    [`vfloat16m2_t`], [f16], [m2], [`!rvv.vector<f16, m2>`], [Requires zvfh extension.],
  ),
  caption: [Mapping C Intrinsics to RVV Dialect Types],
)

=== The !rvv.mask Type

Mask types in RVV require special handling. Unlike some SIMD architectures that use a full register width for boolean vectors (e.g., 0 or -1 in a 32-bit lane), RVV uses a dense bit-packed format. A single vector register can hold masks for elements of varying widths, leading to the concept of "Mask Layout".

The C API defines mask types based on the ratio of the element size to the LMUL ($n = "SEW"/"LMUL"$). This ensures that a mask register has enough bits to cover the data vector.

*Proposed Type*: `!rvv.mask< layout >`

- `layout`: An enum representing the $n$ value from `vbooln_t`. Values: `n1`, `n2`, `n4`, `n8`, `n16`, `n32`, `n64`.

*Compatibility Verification*: A critical role of the dialect verifier is ensuring that a mask used in an operation is compatible with the data vector.

*Rule*: For an operation on type `!rvv.vector<T, M>`, the required mask type is `!rvv.mask<L>` where $L = ("SizeOf"(T))/("Value"(M))$.

*Example*: For `!rvv.vector<i32, m2>`:
- SEW = 32.
- LMUL = 2.
- Ratio = 32 / 2 = 16.
- Required Mask: `!rvv.mask<n16>` (equivalent to `vbool16_t`).

If a user attempts to apply a `!rvv.mask<n1>` (full density) to this operation, the verifier must flag it as an error, as the mask would be too "sparse" or "dense" relative to the data elements.

=== Lowering Logic: From RVV to LLVM IR

While `!rvv.vector` is the frontend type, it must eventually be lowered to the LLVM IR scalable vector type. This lowering process utilizes the target-specific `RVVBitsPerBlock` (usually 64) to determine the `vscale` multiplier.

The formula for determining the number of elements $k$ in `<vscale x k x T>` is:

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, left),
    table.header(
      [*Dialect Type*], [*SEW*], [*LMUL*], [*Calculation*], [*LLVM IR Type*]
    ),
    [`!rvv.vector<i64, m1>`], [64], [1], [$(1 times 64)/64 = 1$], [`<vscale x 1 x i64>`],
    [`!rvv.vector<i32, m2>`], [32], [2], [$(2 times 64)/32 = 4$], [`<vscale x 4 x i32>`],
    [`!rvv.vector<i8, mf2>`], [8], [0.5], [$(0.5 times 64)/8 = 4$], [`<vscale x 4 x i8>`],
    [`!rvv.vector<f64, m8>`], [64], [8], [$(8 times 64)/64 = 8$], [`<vscale x 8 x double>`],
  ),
  caption: [Lowering Mappings (Assuming RVVBitsPerBlock=64)],
)

This mapping ensures that the backend receives the correct type hints to perform register allocation, specifically aligning the generic `vscale` concept with the rigid LMUL structure of RISC-V.

== Operational Semantics and Attribute Strategy

The central design challenge of the `rvv` dialect is reconciling the stateful, variant-rich nature of the instruction set with the stateless, explicit requirements of MLIR's SSA form. The user query explicitly mandates that variables be operator values and differences be defined by attributes.

=== The Operator-Attribute Split

To avoid opcode explosion—where we might have `rvv.add_tu`, `rvv.add_ta`, `rvv.add_m`, `rvv.add_tum`—we define a canonical operation structure that uses attributes to modify semantics.

*The Canonical Op Structure*: Every computational operation generally adheres to the following signature pattern:

```mlir
%result = rvv.operation 
    %source_operands,  // The Input Vectors (Variables)
    %passthru,         // Optional: The Passthru Vector (Variable)
    %mask_operand,     // Optional: The Mask Vector (Variable)
    %vl_operand        // The Active Vector Length (Variable)
    { attributes }     // Static Configuration
```

=== The Policy Attributes: tail_policy and mask_policy

The behavior of tail and masked-off elements is controlled by two orthogonal attributes, mapping directly to the `vta` and `vma` bits in the `vtype` register.

- *Attribute Name*: `tail_policy`
- *Type*: `rvv::TailPolicyAttr` (Enumeration)
- *Values*:
  - `agnostic` (Default): Tail elements are overwritten with 1s (or undefined). Corresponds to `ta`.
  - `undisturbed`: Tail elements retain their previous values. Corresponds to `tu`.

- *Attribute Name*: `mask_policy`
- *Type*: `rvv::MaskPolicyAttr` (Enumeration)
- *Values*:
  - `agnostic` (Default): Masked-off elements are overwritten with 1s (or undefined). Corresponds to `ma`.
  - `undisturbed`: Masked-off elements retain their previous values. Corresponds to `mu`.

=== The Six Policy Variants

The combination of these two attributes and the presence of optional operands (`mask`, `passthru`) defines exactly 6 distinct semantic variants, covering the full spectrum of the C Intrinsic API.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, left),
    table.header(
      [*Variant*], [*Mask Operand*], [*Passthru Operand*], [*Tail Policy*], [*Mask Policy*], [*Semantics*]
    ),
    [`base`], [Absent], [Absent], [`agnostic`], [`agnostic`], [Unmasked, Tail Agnostic],
    [`_m`], [Present], [Absent], [`agnostic`], [`agnostic`], [Masked, Tail Agnostic, Mask Agnostic],
    [`_tu`], [Absent], [Present], [`undisturbed`], [`agnostic`], [Unmasked, Tail Undisturbed],
    [`_mu`], [Present], [Present], [`agnostic`], [`undisturbed`], [Masked, Tail Agnostic, Mask Undisturbed],
    [`_tum`], [Present], [Present], [`undisturbed`], [`agnostic`], [Masked, Tail Undisturbed, Mask Agnostic],
    [`_tumu`], [Present], [Present], [`undisturbed`], [`undisturbed`], [Masked, Tail Undisturbed, Mask Undisturbed],
  ),
  caption: [Mapping of Policy Variants to Operands and Attributes],
)

*Constraints*:
- *Passthru Requirement*: If `tail_policy` is `undisturbed` OR `mask_policy` is `undisturbed`, the `passthru` operand *must* be defined. This operand provides the values for the undisturbed elements.
- *Mask Requirement*: If `mask_policy` is `undisturbed`, the `mask` operand *must* be defined.
- *Mask Implication*: If the `mask` operand is present, the operation is treated as masked.

=== The Rounding Mode Attribute: frm

Floating-point operations in RISC-V can specify a rounding mode statically or use the dynamic `frm` register. Snippet highlights that newer intrinsic versions support explicit rounding modes.

- *Attribute Name*: `frm`
- *Type*: `rvv::FRMAttr` (Enumeration)
- *Values*: `rne` (nearest-even), `rtz` (towards zero), `rdn` (down), `rup` (up), `rmm` (max magnitude), `dyn` (dynamic/default).

This attribute is essential for strictly preserving numerical semantics in ML/AI workloads (e.g., quantization) where rounding behavior is non-negotiable.

== Instruction Category Analysis and Design

This section provides a detailed mapping of specific intrinsic categories to the dialect, analyzing edge cases and operand configurations.

=== Integer Arithmetic Operations

- *Scope*: Addition, Subtraction, Multiplication, Division, Bitwise Logic.
- *Representative Op*: `rvv.add`

*Design Specification*:

```tablegen
def RVV_AddOp : RVV_Op<"add", [Pure]> {
  let arguments = (ins
    RVV_VectorType:$op1,
    RVV_VectorType:$op2,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    I64:$vl,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorType:$res);
}
```

*Usage Examples*:

Unmasked, Tail Agnostic (C: `vadd_vv_i32m1(v1, v2, vl)`):
```mlir
%res = rvv.add %v1, %v2, %vl 
       { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }
```
*Analysis*: The backend emits `vadd.vv`.

Masked, Tail Undisturbed (C: `vadd_vv_i32m1_tum(mask, old_dest, v1, v2, vl)`):
```mlir
%res = rvv.add %v1, %v2, %old_dest, %mask, %vl 
       { tail_policy = #rvv.tail_policy<undisturbed>, mask_policy = #rvv.mask_policy<agnostic> }
```
*Analysis*: The backend ensures that `%res` matches the destination register where the mask is 0 or index > vl.

*Scalar-Vector Variants (.vx)*: We define distinct operations for scalar inputs, e.g., `rvv.add.vx`.
*Reasoning*: While MLIR supports overloading, the RISC-V assembly explicitly distinguishes `vadd.vv` and `vadd.vx`. Separating them at the dialect level simplifies the lowering logic and allows for tighter type verification (ensuring the scalar type matches the vector element type).

=== Widening and Narrowing Operations

- *Scope*: `vwadd` (Widening Add), `vnsra` (Narrowing Shift Right).
- *Complexity*: Type transition rules.

*Design Specification*: The verification logic for these ops must enforcing the $2 times$ rule. For `rvv.wadd`:
- Input: `!rvv.vector<i32, m1>`
- Output: `!rvv.vector<i64, m2>`
- Constraint: The output LMUL must be exactly $2 times$ the input LMUL.
- Edge Case: If Input LMUL is `m8`, the operation is invalid (Result `m16` does not exist). The verifier must catch this.

=== Memory Operations: Loads and Stores

Memory operations are the interface between the vector unit and the system. They support diverse access patterns.

==== Unit-Stride Operations

- *Op*: `rvv.load`
- *Attributes*: `nf` (Number of Fields): Default 1. Used for segment loads.

*Segment Load Support (vlseg)*: Snippet mentions "tuple type segment load/store". When `nf > 1`, the return type of the operation changes. It cannot be a single `!rvv.vector`. It must be a generic list of results or a custom tuple type.

*Design Decision*: Use multiple results.
```mlir
%v1, %v2 = rvv.load.segment %ptr,... { nf=2 } : (!llvm.ptr,...) -> (!rvv.vector<...>,!rvv.vector<...>)
```
*Lowering*: This maps to intrinsics that return structure types (e.g., `vint32m1x2_t`). The backend handles the register allocation constraint that these registers must be contiguous (e.g., v8, v9).

==== Strided and Indexed Operations

- *Ops*: `rvv.load.stride`, `rvv.load.index`
- *Operands*:
  - `stride`: A scalar byte offset.
  - `index`: A vector of offsets (`vluxei`, `vloxei`).

*Constraint*: The index vector has its own LMUL/SEW. The spec defines "Effective Element Width" (EEW). The compiler must verify that the index vector layout is valid for the given data vector layout (e.g., indices must not exceed valid LMUL range).

==== Fault-Only-First Loads (vleff)

Snippet alludes to advanced load behaviors. `vleff` is used for vectorized `strlen` or speculative execution. It updates the `vl` based on memory protection faults.

- *Op*: `rvv.load.ff`
- *Results*: Returns two values: the data vector and the new `vl`.

```mlir
%data, %new_vl = rvv.load.ff %ptr, %original_vl...
```

*Analysis*: This is a rare case where an instruction modifies the vector length. Modeling this explicit return value is critical for the subsequent loop logic, as the loop must process only `%new_vl` elements.

=== Permutation and Gather (vrgather)

- *Scope*: Shuffles, slides, gathers.
- *Op*: `rvv.gather`, `rvv.slideup`, `rvv.slidedown`.

*Design Rule*: `rvv.slideup` must always have a `tail_policy=undisturbed` behavior implicitly. The dialect enforces this to ensure the lower elements are defined.

=== Reductions

- *Scope*: `vredsum`, `vfredusum` (unordered), `vfredosum` (ordered).
- *Op*: `rvv.reduce`
- *Attributes*:
  - `kind`: `{add, min, max, and, or, xor}`.
  - `ordered`: Bool (for FP).

*Operand Structure*:
```mlir
%scalar_res = rvv.reduce %vector_src, %scalar_init, %vl 
              { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }
```

- `%vector_src`: The vector to reduce.
- `%scalar_init`: The value to start accumulation with.
- `%vl`: Active vector length.
- Optional `%mask` and `%passthru` operands follow the standard policy rules.

*Note on "Scalar" Types*: In RVV, scalars in reductions are actually vector registers where only element 0 is active. The dialect uses `!rvv.vector` with LMUL=m1 for these scalar operands to match the ABI.

== Control Flow and State Management: The rvv.setvl Design

The management of `vsetvli` is the single most important optimization enabled by this dialect.

=== The rvv.setvl Operation

We model the configuration instruction as a value-producing operation.

```mlir
%vl = rvv.setvl %avl, %vtype_context { 
    sew = 32, 
    lmul = m2, 
    tail_agnostic = true, 
    mask_agnostic = true 
}
```

- *Input*: `%avl` (Application Vector Length). This is the "requested" number of items (N).
- *Output*: `%vl` (Active Vector Length). This is the actual number of items the hardware will process ($min(N, "VLMAX")$).

=== Strip-Mining Loop Structure

Snippet describes the "Strip-Mining" approach. The `rvv` dialect facilitates this structure naturally.

*Example Dialect Pseudocode (Vector Add Loop)*:
```mlir
func @vec_add(%N: i64, %base_A:!llvm.ptr, %base_B:!llvm.ptr) {
  // Loop while N > 0
  scf.while (%rem_N = %N, %ptr_A = %base_A, %ptr_B = %base_B) : (i64,!llvm.ptr,!llvm.ptr) ->... {
    // 1. Calculate VL for this iteration
    %vl = rvv.setvl %rem_N { sew=32, lmul=m1,... }

    // 2. Load Data (using %vl)
    %v_A = rvv.load %ptr_A, %vl { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }
    %v_B = rvv.load %ptr_B, %vl { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }

    // 3. Compute (using %vl)
    %v_res = rvv.add %v_A, %v_B, %vl { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }

    // 4. Store (using %vl)
    rvv.store %v_res, %ptr_A, %vl { tail_policy = #rvv.tail_policy<agnostic>, mask_policy = #rvv.mask_policy<agnostic> }

    // 5. Pointer Arithmetic
    %vl_i64 = arith.extui %vl : i64
    %new_ptr_A = llvm.getelementptr %ptr_A[%vl_i64]...
    %new_ptr_B = llvm.getelementptr %ptr_B[%vl_i64]...

    // 6. Update Remaining Count
    %new_rem_N = arith.subi %rem_N, %vl_i64
    %cond = arith.cmpi sgt, %new_rem_N, 0
    scf.condition(%cond) %new_rem_N, %new_ptr_A, %new_ptr_B
  } do {
    ^bb0(%a: i64, %b:!llvm.ptr, %c:!llvm.ptr):
      scf.yield %a, %b, %c
  }
}
```

*Analysis*: By making `%vl` an explicit value passed to `rvv.load`, `rvv.add`, and `rvv.store`, the compiler constructs a complete Def-Use chain.

*Optimization*: If the compiler sees two adjacent loops using the same SEW/LMUL configuration, it can analyze the `%vl` production. If the `rvv.setvl` inputs are identical, Common Subexpression Elimination (CSE) can merge them.

*VSETVLI Insertion Pass*: A later pass in the backend (machine level) is usually responsible for inserting `vsetvli`. However, having `rvv.setvl` in the IR allows the mid-level optimizer to hoist these configuration steps out of inner loops if they are loop-invariant (e.g., setting fixed length for a constant-sized matrix operation).

== Lowering and Optimization

=== Optimization: Policy Relaxation

Snippet and note that many intrinsics default to `tu` (Tail Undisturbed). This is safe but slow.

*Pass*: `RVVPolicyRelaxation`
*Logic*: Traverse the Use-Def chain of an `rvv` operation result.
1. Does any user read the tail elements? (e.g., is it a `vslideup` or a reduction that explicitly uses the tail?)
2. If NO: The operation is writing to a fresh register or memory.
*Action*: Change attribute `tail_policy` from `undisturbed` to `agnostic`.
*Benefit*: Removes the RAW dependency on the destination register, allowing the register allocator to use any free register for the result.

=== Lowering to LLVM Intrinsics

The final stage is converting `rvv` dialect ops to `llvm.call` targeting RISC-V intrinsics.

*Intrinsic Signature Matching*: The converter must synthesize the intrinsic name based on the attributes.
- Input: `rvv.add... { tail_policy=undisturbed, mask_policy=undisturbed }` on type `!rvv.vector<i32, m1>`
- Target: `llvm.riscv.vadd.nxv2i32.i32` (or similar, version dependent).

*ABI Handling*: Snippet defines the calling convention. When lowering functions that take `!rvv.vector` arguments, the converter must attach the `riscv_vector_cc` calling convention attribute to the LLVM function definition. This ensures that arguments are passed in v8 etc., rather than on the stack.
