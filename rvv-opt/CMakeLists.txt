get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(rvv-opt
  rvv-opt.cpp
  
  DEPENDS
  RVVIncGen
)

llvm_update_compile_flags(rvv-opt)

target_link_libraries(rvv-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRRVV
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
)

install(TARGETS rvv-opt RUNTIME DESTINATION bin)
