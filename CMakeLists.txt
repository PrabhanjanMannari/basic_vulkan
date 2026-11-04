cmake_minimum_required(VERSION 3.20) 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED) 

set(NAME HowToVulkan)
project(${NAME})

add_executable(vulkan_app
    src/main.cpp
)

target_include_directories(vulkan_app PRIVATE  
    inc 
)

add_subdirectory(libs/tinyobjloader)

target_link_libraries(vulkan_app PRIVATE 
    Vulkan::Vulkan
    glfw
    tinyobjloader
)
