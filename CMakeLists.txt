#Do this command 
#$ sudo cmake CMakeLists.txt
#$ sudo make
#$ sudo ./olle_CNN
cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-g -Wall -O3")
project(olle_CNN)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCES main.cpp c_func.c cpp_func2.cpp)
add_executable(olle_CNN ${SOURCES})
target_link_libraries(olle_CNN ${OpenCV_LIBS})
