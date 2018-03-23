cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_test LANGUAGES CXX CUDA)

include(CTest)

add_library(smartgame STATIC
  smartgame/*.cpp
  smartgmae/*.h
  )

add_executable(gamma_go_main gamma_go_main.cu)

set_target_properties(gamma_go_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(gamma_go_main PRIVATE smartgame)






endif()
