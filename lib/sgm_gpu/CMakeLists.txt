cmake_minimum_required(VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")

project(cudasgm)


if(UNIX)
  set(BOOST_PYTHONLIB python-py36)
else()
  set(BOOST_PYTHONLIB python3)
endif()

find_package(PythonInterp 3.2 REQUIRED)
find_package(PythonLibs 3.2 REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS ${BOOST_PYTHONLIB})
find_package( OpenCV 3 REQUIRED )

#get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
#find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})

#if (${UNIX})
#set (EXTRA_OPTIONS "-O3  -Wno-deprecated -fPIC -DLINUX")

set(CMAKE_CXX_FLAGS ${CMAKE_CSS_FLAGS} "-O3  -Wno-deprecated -fPIC -DLINUX")
# endif (${UNIX})


# NOTE: This doesn't work with xcode generators
#enable_language("CUDA")

find_package(CUDA "9.0" REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
# enable device lambda support
# xxxnsubtil: not usable yet
#list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

# set optimization flags
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

#set(CUDA_NVCC_FLAGS "-o3")
#set(CUDA_NVCC_FLAGS "-arch=sm_70")

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_70")

# add nvcc flags to generate code for all enabled architectures
#string(REPLACE "," ";" GPU_ARCHITECTURE ${GPU_ARCHITECTURE})
#foreach(ARCH IN LISTS GPU_ARCHITECTURE)
#    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
#    set(CUDA_NVCC_FLAGS
#        ${CUDA_NVCC_FLAGS}
#        --generate-code arch=${COMPUTE},code=${ARCH})
#endforeach(ARCH)

#option(CUDA_EXAMPLE_USE_CC_EXT "Use CC extensions for CUDA files" OFF)

#set(cpu_source_files cpu.cc)

#FILE(GLOB PROFILE_SRC_FILES "src/*.cu")

#if(CUDA_EXAMPLE_USE_CC_EXT)
#  set(gpu_source_files src/gpu.cu.cc)
#else()
#set(gpu_source_files ${PROFILE_SRC_FILES})
#set(gpu_source_files src/gpu.cu)
#endif()

#FILE(GLOB PROFILE_SRC_FILES "src/*.cu")

set_source_files_properties(${gpu_source_files} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
cuda_add_library(cuda_lib SHARED ${gpu_source_files})
#add_executable(cuda_exe main.cpp)
python_add_module(sgm_gpu main.cu)
#target_link_libraries(cuda_exe cuda_lib)
#target_compile_definitions(cuda_lib PUBLIC ${Boost_LIBRARIES} png ${PYTHON_LIBRARIES} boost_numpy3 ${OpenCV_LIBS} USE_CUDA=1)

target_include_directories(sgm_gpu PUBLIC ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
target_link_libraries(sgm_gpu cuda_lib ${Boost_LIBRARIES} png ${PYTHON_LIBRARIES} boost_numpy3 ${OpenCV_LIBS})
