cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

project(cudasgm)

if(UNIX)
  set(BOOST_PYTHONLIB python36)
else()
  set(BOOST_PYTHONLIB python3)
endif()

find_package(PythonInterp 3.2 REQUIRED)
find_package(PythonLibs 3.2 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS ${BOOST_PYTHONLIB})
find_package( OpenCV 3 REQUIRED )

set(CMAKE_CXX_FLAGS ${CMAKE_CSS_FLAGS} "-O3  -Wno-deprecated -fPIC -DLINUX")

find_package(CUDA "9.0" REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

cuda_add_library(cuda_lib SHARED)
python_add_module(sgm_gpu main.cu)

target_include_directories(sgm_gpu PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(sgm_gpu cuda_lib ${Boost_LIBRARIES} png ${PYTHON_LIBRARIES} boost_numpy36 ${OpenCV_LIBS})
