configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Provide extra substitutions for tool directories

set(RVV_OPT_TEST_DEPENDS
        FileCheck count not
        rvv-opt
        )

add_lit_testsuite(check-rvv-opt "Running the rvv-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RVV_OPT_TEST_DEPENDS}
        )
set_target_properties(check-rvv-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(RVV_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RVV_OPT_TEST_DEPENDS})
