def VLoadOp : RVV_Op<"load", [
  DeclareOpInterfaceMethods<RVVPolicyInterface>, 
  AttrSizedOperandSegments
  ]> {
  let summary = "Unit/Const-Stride vector load";
  let description = [{
    Loads a vector or mask from memory with unit stride (stride=1) or constant stride.
    This operation maps to the RVV unit-stride and strided load intrinsics.

    Arguments:
    - `ptr`: Base address pointer to load from
    - `vl`: Vector length - number of elements to load
    - `stride`: Memory stride in bytes between consecutive elements (default: 1 for unit-stride)
    - `nf`: Number of fields for segment loads (default: 1 for non-segmented loads)
    - `fault_only_first`: Enable fault-only-first mode for unit-stride loads (default: false)
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are loaded
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)

    Results:
    - `vd`: Loaded vector or mask value

    Lowering Rules:
    - For unit-stride (stride=1, nf=1): lowers to `vle<width>_v_<type>` or `vlm_v_<type>` for masks
    - For strided (stride!=1, nf=1): lowers to `vlse<width>_v_<type>`
    - For segment (nf>1): lowers to `vlseg<nf>e<width>_v_<type>` or `vlsseg<nf>e<width>_v_<type>`
    - For fault-only-first: lowers to `vle<width>ff_v_<type>`
    - Policy suffixes `_tu`, `_ta`, `_mu`, `_ma` are appended based on tail/mask policies
    - Masked variants append `_m` and include mask operand

    Example:
    ```mlir
    %result = rvv.load %ptr, %vl : (!llvm.ptr, i64) -> !rvv.vector<i32, m1>
    %strided = rvv.load %ptr, %vl {stride = 4} : (!llvm.ptr, i64) -> !rvv.vector<f32, m2>
    %masked = rvv.load %ptr, %vl, %mask : (!llvm.ptr, i64, !rvv.mask<n1>) -> !rvv.vector<i64, m4>
    ```
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    I64:$vl,
    DefaultValuedAttr<I32Attr, "1">:$stride,
    DefaultValuedAttr<I32Attr, "1">:$nf,
    DefaultValuedAttr<BoolAttr, "false">:$fault_only_first,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorOrMaskType:$vd);
}

def VStoreOp : RVV_Op<"store", [
  ]> {
  let summary = "Unit/Const-Stride vector store";
  let description = [{
    Stores a vector or mask to memory with unit stride (stride=1) or constant stride.
    This operation maps to the RVV unit-stride and strided store intrinsics.

    Arguments:
    - `value`: Vector or mask value to store
    - `ptr`: Base address pointer to store to
    - `vl`: Vector length - number of elements to store
    - `stride`: Memory stride in bytes between consecutive elements (default: 1 for unit-stride)
    - `nf`: Number of fields for segment stores (default: 1 for non-segmented stores)
    - `mask`: Optional mask to control which elements are stored

    Lowering Rules:
    - For unit-stride (stride=1, nf=1): lowers to `vse<width>_v_<type>` or `vsm_v_<type>` for masks
    - For strided (stride!=1, nf=1): lowers to `vsse<width>_v_<type>`
    - For segment (nf>1): lowers to `vsseg<nf>e<width>_v_<type>` or `vssseg<nf>e<width>_v_<type>`
    - Masked variants append `_m` and include mask operand

    Example:
    ```mlir
    rvv.store %value, %ptr, %vl : !rvv.vector<i32, m1>, !llvm.ptr, i64
    rvv.store %value, %ptr, %vl {stride = 4} : !rvv.vector<f32, m2>, !llvm.ptr, i64
    rvv.store %value, %ptr, %vl, %mask : !rvv.vector<i64, m4>, !llvm.ptr, i64, !rvv.mask<n1>
    ```
  }];
  let arguments = (ins
    RVV_VectorOrMaskType:$value,
    LLVM_AnyPointer:$ptr,
    I64:$vl,
    DefaultValuedAttr<I32Attr, "1">:$stride,
    DefaultValuedAttr<I32Attr, "1">:$nf,
    Optional<RVV_MaskType>:$mask
  );
}

def VLoadIndexOp : RVV_Op<"load.index", [DeclareOpInterfaceMethods<RVVPolicyInterface>, AttrSizedOperandSegments]> {
  let summary = "Indexed vector load (gather)";
  let description = [{
    Performs an indexed (gather) load from memory using a vector of indices.
    Each element is loaded from address `ptr + index[i] * sizeof(element)`.
    This operation maps to the RVV indexed load intrinsics.

    Arguments:
    - `ptr`: Base address pointer to load from
    - `index`: Vector of indices (element type determines index width)
    - `vl`: Vector length - number of elements to load
    - `nf`: Number of fields for segment indexed loads (default: 1 for non-segmented loads)
    - `ordered`: Whether to use ordered indexed load (default: false for unordered)
    - `passthru`: Optional vector value for inactive elements (policy-dependent)
    - `mask`: Optional mask to control which elements are loaded
    - `tail_policy`: Policy for tail elements (tu = tail undisturbed, ta = tail agnostic)
    - `mask_policy`: Policy for masked-off elements (mu = mask undisturbed, ma = mask agnostic)

    Results:
    - `vd`: Loaded vector or mask value

    Lowering Rules:
    - For unordered (ordered=false, nf=1): lowers to `vluxei<index_width>_v_<type>`
    - For ordered (ordered=true, nf=1): lowers to `vloxei<index_width>_v_<type>`
    - For segment (nf>1): lowers to `vluxseg<nf>ei<index_width>_v_<type>` or `vloxseg<nf>ei<index_width>_v_<type>`
    - `<index_width>` is determined by the index vector element type (8, 16, 32, or 64)
    - Policy suffixes `_tu`, `_ta`, `_mu`, `_ma` are appended based on tail/mask policies
    - Masked variants append `_m` and include mask operand

    Example:
    ```mlir
    %result = rvv.load.index %ptr, %indices, %vl : (!llvm.ptr, !rvv.vector<i32, m1>, i64) -> !rvv.vector<f32, m1>
    %ordered = rvv.load.index %ptr, %indices, %vl {ordered = true} : (!llvm.ptr, !rvv.vector<i64, m2>, i64) -> !rvv.vector<f64, m2>
    ```
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    RVV_VectorType:$index,
    I64:$vl,
    DefaultValuedAttr<I32Attr, "1">:$nf,
    DefaultValuedAttr<BoolAttr, "false">:$ordered,
    Optional<RVV_VectorType>:$passthru,
    Optional<RVV_MaskType>:$mask,
    RVV_TailPolicyAttr:$tail_policy,
    RVV_MaskPolicyAttr:$mask_policy
  );
  let results = (outs RVV_VectorOrMaskType:$vd);
}

def VStoreIndexOp : RVV_Op<"store.index", []> {
  let summary = "Indexed vector store (scatter)";
  let description = [{
    Performs an indexed (scatter) store to memory using a vector of indices.
    Each element is stored to address `ptr + index[i] * sizeof(element)`.
    This operation maps to the RVV indexed store intrinsics.

    Arguments:
    - `value`: Vector or mask value to store
    - `ptr`: Base address pointer to store to
    - `index`: Vector of indices (element type determines index width)
    - `vl`: Vector length - number of elements to store
    - `nf`: Number of fields for segment indexed stores (default: 1 for non-segmented stores)
    - `ordered`: Whether to use ordered indexed store (default: false for unordered)
    - `mask`: Optional mask to control which elements are stored

    Lowering Rules:
    - For unordered (ordered=false, nf=1): lowers to `vsuxei<index_width>_v_<type>`
    - For ordered (ordered=true, nf=1): lowers to `vsoxei<index_width>_v_<type>`
    - For segment (nf>1): lowers to `vsuxseg<nf>ei<index_width>_v_<type>` or `vsoxseg<nf>ei<index_width>_v_<type>`
    - `<index_width>` is determined by the index vector element type (8, 16, 32, or 64)
    - Masked variants append `_m` and include mask operand

    Example:
    ```mlir
    rvv.store.index %value, %ptr, %indices, %vl : !rvv.vector<f32, m1>, !llvm.ptr, !rvv.vector<i32, m1>, i64
    rvv.store.index %value, %ptr, %indices, %vl {ordered = true} : !rvv.vector<f64, m2>, !llvm.ptr, !rvv.vector<i64, m2>, i64
    ```
  }];
  let arguments = (ins
    RVV_VectorOrMaskType:$value,
    LLVM_AnyPointer:$ptr,
    RVV_VectorType:$index,
    I64:$vl,
    DefaultValuedAttr<I32Attr, "1">:$nf,
    DefaultValuedAttr<BoolAttr, "false">:$ordered,
    Optional<RVV_MaskType>:$mask
  );
}
