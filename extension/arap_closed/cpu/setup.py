from torch.utils.cpp_extension import load
arap_cpu = load(name="arap_cpu", sources=["arap_cpu.cpp"], build_directory='./lib/',verbose=True)
