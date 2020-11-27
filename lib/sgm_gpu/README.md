# cmake_cuda_test

Simple test to illustrate `cuda_add_library()` failure with Xcode generator using non-standard `*.cc` extensions in combinatino wtih `set_source_files_properties(${gpu_source_files} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)`.  Using the `*.cu` extensions works fine.


### `*.cu` extensions work:
Full log [here](https://github.com/headupinclouds/cmake_cuda_test/issues/1)
```
mkdir -p _builds/xcode2
cmake -GXcode -H. -B_builds/xcode2 -DCMAKE_VERBOSE_MAKEFILE=ON -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-9.0 -DCUDA_EXAMPLE_USE_CC_EXT=OFF && cd _builds/xcode2 && cmake --build .
```

```
** BUILD SUCCEEDED **
```


### `*.cc` extensions don't:
Full log [here](https://github.com/headupinclouds/cmake_cuda_test/issues/2)
```
mkdir -p _builds/xcode
cmake -GXcode -H. -B_builds/xcode -DCMAKE_VERBOSE_MAKEFILE=ON -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-9.0 -DCUDA_EXAMPLE_USE_CC_EXT=ON && cd _builds/xcode && cmake --build .
```

```
/cmake_cuda_test/gpu.cu:6:47: error: use of undeclared identifier '__CUDACC_VER__'
    std::cout << "CUDA Compiled version: " << __CUDACC_VER__ << std::endl;
                                              ^
/cmake_cuda_test/gpu.cu:9:5: error: use of undeclared identifier 'cudaRuntimeGetVersion'
    cudaRuntimeGetVersion(&runtime_ver);
    ^
/cmake_cuda_test/gpu.cu:13:5: error: use of undeclared identifier 'cudaDriverGetVersion'
    cudaDriverGetVersion(&driver_ver);

```

