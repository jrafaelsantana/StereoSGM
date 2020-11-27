file(REMOVE_RECURSE
  "libcuda_lib.pdb"
  "libcuda_lib.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cuda_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
