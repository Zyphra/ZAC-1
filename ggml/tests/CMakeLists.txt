# Streamed (block 3) == buffered: iSTFT overlap-add / FIR mixer holdback and flush.
add_test(NAME zac-seam COMMAND zac-cli --seam fixtures/dummy-zac.gguf
    fixtures/dummy-zac-codes.npy --block 3 --cpu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(zac-seam PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# Early partial-frame decode: missing residual codebooks are left out (dummy-zac.gguf carries res.{i}.bias).
add_executable(zac-stream stream.cpp)
target_link_libraries(zac-stream PRIVATE zac)
add_test(NAME zac-stream COMMAND zac-stream ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/dummy-zac.gguf)
set_tests_properties(zac-stream PROPERTIES PASS_REGULAR_EXPRESSION "PASS")

# The real semantic teacher's reference-state lifecycle (optional: needs a w2vbert_ssl GGUF).
add_executable(ssl-reference-state ssl-reference-state.cpp)
target_link_libraries(ssl-reference-state PRIVATE zac)
set(ZAC_TEST_SSL_TEACHER "" CACHE FILEPATH "Semantic teacher GGUF for the optional real-weight lifecycle regression")
if (ZAC_TEST_SSL_TEACHER)
    add_test(NAME ssl-reference-state COMMAND ssl-reference-state ${ZAC_TEST_SSL_TEACHER})
endif()
