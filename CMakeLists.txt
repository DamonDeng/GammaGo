cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_test LANGUAGES CXX CUDA)

include(CTest)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.58.0 COMPONENTS filesystem thread system date_time program_options) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 

    file(GLOB SGSources "smartgame/*.cpp")
    file(GLOB SGHeaders "smartgame/*.h")
    
    file(GLOB GTPSources "gtpengine/*.cpp")
    file(GLOB GTPHeaders "gtpengine/*.h")
    
    file(GLOB GOSources "go/*.cpp")
    file(GLOB GOHeaders "go/*.h")
    file(GLOB GOCUDAs "go/*.cu")
     
    
    
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
      ${GOCUDAs}
      )
    
    set_target_properties(go PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
     
    
    add_executable(gamma_go_main gamma_go_main.cu)
    set_target_properties(gamma_go_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories (gamma_go_main PUBLIC go smartgame gtpengine)
    
    
    
    
    target_link_libraries(gamma_go_main PRIVATE smartgame)
    target_link_libraries(gamma_go_main PRIVATE gtpengine)
    target_link_libraries(gamma_go_main PRIVATE go)
    target_link_libraries(gamma_go_main PRIVATE ${Boost_LIBRARIES})
    
    target_link_libraries(smartgame PRIVATE ${Boost_LIBRARIES})

    target_link_libraries(gtpengine PRIVATE smartgame)

    target_link_libraries(go PRIVATE smartgame)


     
    target_include_directories (smartgame PUBLIC gtpengine)
    target_include_directories (go PUBLIC smartgame gtpengine /usr/local/cuda-9.0/include/)
      
    
    
    
    
endif()



