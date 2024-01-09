from torch.utils.cpp_extension import load
arap_cuda= load(name="arap_cuda",
          sources=["arap_cuda.cpp",
                   "arap_cuda_kernel.cu"],build_directory='./lib/',verbose=True)
