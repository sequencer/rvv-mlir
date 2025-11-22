#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
)
#set text(
  font: "New Computer Modern",
  size: 11pt,
)
= Structured Mapping of RVV Intrinsics to MLIR Operations

This appendix provides a structured mapping of RVV 1.0 C intrinsics to the proposed MLIR `rvv` dialect. This mapping is rigorously organized according to the instruction categories defined in the LLVM frontend definition file `clang/include/clang/Basic/riscv_vector.td`. This file serves as the authoritative source for intrinsic classification in the Clang compiler.

== General Mapping Principles

- *Variables as Operator Values*: All dynamic values (operands, masks, vector lengths) are explicit SSA values.
- *Variants as Attributes*: Differences in behavior (Masking, Tail Policy, Mask Policy) are captured via attributes (`masked`, `policy`), not distinct opcodes.
- *Structure*: The tables below follow the section headers found in `riscv_vector.td`.

== Configuration-Setting Instructions (Section 6 in .td)

These instructions manage the `vtype` and `vl` state. In MLIR, these are value-producing operations that return the active vector length.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Set vl/vtype], [`vsetvl_e32m2(avl)`], [`rvv.setvl`], [`sew`, `lmul`, `ta`, `ma`],
    [Set vl to VLMAX], [`vsetvlmax_e32m2()`], [`rvv.setvl_max`], [`sew`, `lmul`],
  ),
  caption: [Configuration Instructions],
)

*Example MLIR*:
```mlir
%vl = rvv.setvl %avl { sew=32, lmul=m2, ta=true, ma=true }
```

== Vector Loads and Stores (Section 7 in .td)

This category covers data movement between memory and vector registers. It includes unit-stride, strided, indexed, and segment variants.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Unit-Stride Load], [`vle32_v_i32m1`, `vlm_v`], [`rvv.load`], [`policy`, `masked`],
    [Unit-Stride Store], [`vse32_v_i32m1`, `vsm_v`], [`rvv.store`], [`masked`],
    [Strided Load], [`vlse32_v_i32m1`], [`rvv.load.stride`], [`policy`, `masked`],
    [Strided Store], [`vsse32_v_i32m1`], [`rvv.store.stride`], [`masked`],
    [Indexed Load (Gather)], [`vluxei32_v_i32m1`], [`rvv.load.index`], [`policy`, `masked`],
    [Ordered Indexed Load], [`vloxei32_v_i32m1`], [`rvv.load.index`], [`policy`, `masked`, `ordered`],
    [Indexed Store (Scatter)], [`vsuxei32_v_i32m1`], [`rvv.store.index`], [`masked`],
    [Ordered Indexed Store], [`vsoxei32_v_i32m1`], [`rvv.store.index`], [`masked`, `ordered`],
    [Unit-Stride Fault-Only-First], [`vle32ff_v_i32m1`], [`rvv.load.ff`], [`policy`, `masked`],
    [Segment Load], [`vlseg2e32_v_i32m1`], [`rvv.load.segment`], [`nf` (num fields), `masked`, `policy`],
    [Segment Store], [`vsseg2e32_v_i32m1`], [`rvv.store.segment`], [`nf` (num fields), `masked`],
  ),
  caption: [Load/Store Instructions],
)

*Example MLIR (Indexed Load, Masked, Tail Undisturbed)*:
```mlir
%res = rvv.load.index %merge, %base_ptr, %index_vec, %mask, %vl 
       { policy=tu, masked=true }
```

== Vector Integer Arithmetic Instructions (Section 11 in .td)

This section covers standard arithmetic. The `riscv_vector.td` file typically groups these into "Single-Width", "Widening", and "Narrowing" classes.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Single-Width Add/Sub], [`vadd_vv`, `vsub_vx`], [`rvv.add`, `rvv.sub`], [`policy`, `masked`],
    [Single-Width Bitwise], [`vand_vv`, `vxor_vv`], [`rvv.and`, `rvv.xor`], [`policy`, `masked`],
    [Single-Width Shift], [`vsll_vv`, `vsra_vx`], [`rvv.shl`, `rvv.ashr`], [`policy`, `masked`],
    [Narrowing Shift], [`vnsra_wv`], [`rvv.nsra`], [`policy`, `masked`],
    [Integer Compare], [`vmseq_vv`, `vmslt_vx`], [`rvv.icmp`], [`predicate`, `masked`],
    [Integer Min/Max], [`vmin_vv`, `vmaxu_vx`], [`rvv.min`, `rvv.max`], [`policy`, `masked`],
    [Single-Width Multiply], [`vmul_vv`, `vmulh_vv`], [`rvv.mul`, `rvv.mulh`], [`policy`, `masked`],
    [Integer Divide], [`vdiv_vv`, `vrem_vv`], [`rvv.div`, `rvv.rem`], [`policy`, `masked`],
    [Widening Add/Sub], [`vwadd_vv`, `vwsub_vx`], [`rvv.wadd`, `rvv.wsub`], [`policy`, `masked`],
    [Widening Multiply], [`vwmul_vv`], [`rvv.wmul`], [`policy`, `masked`],
    [Single-Width MAC], [`vmacc_vv`, `vnmsac_vv`], [`rvv.macc`, `rvv.nmsac`], [`policy`, `masked`],
    [Widening MAC], [`vwmacc_vv`], [`rvv.wmacc`], [`policy`, `masked`],
    [Integer Merge], [`vmerge_vvm`], [`rvv.merge`], [N/A],
    [Integer Move], [`vmv_v_v`, `vmv_v_x`], [`rvv.mv`], [N/A],
    [Integer Extension], [`vsext_vf2`], [`rvv.sext`], [`policy`, `masked`],
    [Carry/Borrow], [`vmadc_vvm`, `vmsbc_vvm`], [`rvv.adc`, `rvv.sbc`], [`masked` (Return is mask)],
  ),
  caption: [Integer Arithmetic Instructions],
)

*Example MLIR (Widening Add)*:
```mlir
%res = rvv.wadd %undef, %op1, %op2, %all_ones, %vl { policy=none }
```

== Vector Fixed-Point Arithmetic Instructions (Section 12 in .td)

Includes saturating arithmetic and averaging operations tailored for fixed-point DSP algorithms.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Single-Width Saturating], [`vsadd_vv`], [`rvv.sadd`], [`policy`, `masked`],
    [Single-Width Averaging], [`vaadd_vv`], [`rvv.aadd`], [`policy`, `masked`, `rounding`],
    [Fractional Multiply], [`vsmul_vv`], [`rvv.smul`], [`policy`, `masked`, `rounding`],
    [Scaling Shift], [`vssrl_vv`], [`rvv.ssrl`], [`policy`, `masked`, `rounding`],
    [Narrowing Clip], [`vnclip_wv`], [`rvv.nclip`], [`policy`, `masked`, `rounding`],
  ),
  caption: [Fixed-Point Arithmetic Instructions],
)

== Vector Floating-Point Instructions (Section 13 in .td)

Standard IEEE-754 operations. Includes support for `frm` (floating-point rounding mode) attribute where applicable.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Single-Width Arith], [`vfadd_vv`, `vfmul_vf`], [`rvv.fadd`, `rvv.fmul`], [`policy`, `masked`, `frm`],
    [Widening Arith], [`vfwadd_vv`], [`rvv.fwadd`], [`policy`, `masked`, `frm`],
    [Widening Multiply], [`vfwmul_vv`], [`rvv.fwmul`], [`policy`, `masked`, `frm`],
    [Fused Multiply-Add], [`vfmacc_vv`], [`rvv.fma`], [`policy`, `masked`, `frm`],
    [Widening FMA], [`vfwmacc_vv`], [`rvv.fwma`], [`policy`, `masked`, `frm`],
    [Square Root], [`vfsqrt_v`], [`rvv.fsqrt`], [`policy`, `masked`],
    [Reciprocal/Rsqrt Est], [`vfrec7_v`, `vfrsqrt7_v`], [`rvv.frec7`, `rvv.frsqrt7`], [`policy`, `masked`],
    [Min/Max], [`vfmin_vv`, `vfmax_vv`], [`rvv.fmin`, `rvv.fmax`], [`policy`, `masked`],
    [Sign-Injection], [`vfsgnj_vv`, `vfsgnjn_vv`], [`rvv.fsgnj`, `rvv.fsgnjn`], [`policy`, `masked`],
    [Comparison], [`vmfeq_vv`, `vmfgt_vf`], [`rvv.fcmp`], [`predicate` (eq, gt...), `masked`],
    [Classify], [`vfclass_v`], [`rvv.fclass`], [`policy`, `masked`],
    [Merge], [`vfmerge_vfm`], [`rvv.fmerge`], [N/A],
    [Move], [`vfmv_v_f`], [`rvv.fmv`], [N/A],
    [Type Convert], [`vfcvt_x_f_v`, `vfwcvt_f_f_v`], [`rvv.fcvt`, `rvv.fwcvt`], [`policy`, `masked`, `frm`],
  ),
  caption: [Floating-Point Arithmetic Instructions],
)

== Vector Reduction Instructions (Section 14 in .td)

Reductions consume a vector and produce a "scalar" (vector with 1 active element).

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Single-Width Integer], [`vredsum_vs`], [`rvv.reduce.int`], [`kind`=add, `policy`],
    [Widening Integer], [`vwredsum_vs`], [`rvv.wreduce.int`], [`kind`=add, `policy`],
    [Single-Width Float], [`vfredusum_vs`], [`rvv.reduce.fp`], [`kind`=add, `ordered`=false],
    [Ordered Float], [`vfredosum_vs`], [`rvv.reduce.fp`], [`kind`=add, `ordered`=true],
    [Widening Float], [`vfwredusum_vs`], [`rvv.wreduce.fp`], [`kind`=add, `ordered`=false],
  ),
  caption: [Reduction Instructions],
)

== Vector Mask Instructions (Section 15 in .td)

Operations that act specifically on mask registers.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Mask Logical], [`vmand_mm`, `vmnor_mm`], [`rvv.mask.and`, `rvv.mask.nor`], [N/A],
    [Count Population], [`vcpop_m`], [`rvv.mask.popcount`], [N/A],
    [Find First Set], [`vfirst_m`], [`rvv.mask.first`], [N/A],
    [Set-Before-First], [`vmsbf_m`], [`rvv.mask.sbf`], [`masked`],
    [Set-Including-First], [`vmsif_m`], [`rvv.mask.sif`], [`masked`],
    [Set-Only-First], [`vmsof_m`], [`rvv.mask.sof`], [`masked`],
    [Iota], [`viota_m`], [`rvv.iota`], [`policy`, `masked`],
    [Element Index], [`vid_v`], [`rvv.id`], [`policy`, `masked`],
  ),
  caption: [Mask Instructions],
)

== Vector Permutation Instructions (Section 16 in .td)

Data movement within or between registers.

#figure(
  table(
    columns: 4,
    align: (left, left, left, left),
    table.header(
      [*RISC-V Intrinsic Category*], [*C Intrinsic Example*], [*MLIR Dialect Operation*], [*Attributes*]
    ),
    [Integer Scalar Move], [`vmv_s_x`, `vmv_x_s`], [`rvv.mv.s.x`, `rvv.mv.x.s`], [N/A],
    [Float Scalar Move], [`vfmv_s_f`, `vfmv_f_s`], [`rvv.fmv.s.f`, `rvv.fmv.f.s`], [N/A],
    [Vector Slide], [`vslideup_vx`, `vslide1down_vx`], [`rvv.slideup`, `rvv.slidedown`], [`policy`, `masked`],
    [Vector Gather], [`vrgather_vv`, `vrgatherei16_vv`], [`rvv.gather`], [`policy`, `masked`],
    [Vector Compress], [`vcompress_vm`], [`rvv.compress`], [`policy`],
  ),
  caption: [Permutation Instructions],
)
