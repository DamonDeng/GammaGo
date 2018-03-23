cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_test LANGUAGES CXX CUDA)

include(CTest)

file(GLOB SGSources "smartgame/*.cpp")
file(GLOB SGHeaders "smartgame/*.h")

file(GLOB GTPSources "gtpengine/*.cpp")
file(GLOB GTPHeaders "gtpengine/*.h")

file(GLOB GOSources "go/*.cpp")
file(GLOB GOHeaders "go/*.h")



add_library(smartgame STATIC
  ${SGSources}
  ${SGHeaders}
  )

add_library(gtpengine STATIC
  ${GTPSources}
  ${GTPHeaders}
  )

add_library(go STATIC
  ${GOSources}
  ${GOHeaders}
  )



add_executable(gamma_go_main gamma_go_main.cu)

set_target_properties(gamma_go_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_link_libraries(gamma_go_main PRIVATE smartgame)
target_include_directories (smartgame PUBLIC gtpengine)

target_link_libraries(gamma_go_main PRIVATE gtpengine)

target_link_libraries(gamma_go_main PRIVATE go)
target_include_directories (go PUBLIC smartgame go)







