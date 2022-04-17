cmake_minimum_required(VERSION 3.5)
project(mesh)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
add_executable(mesh mesh.cpp)

target_link_libraries (mesh ${PCL_LIBRARIES})

install(TARGETS mesh RUNTIME DESTINATION bin)
