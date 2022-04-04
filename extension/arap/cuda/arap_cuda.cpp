#include <torch/extension.h>
//#include "eigen/Eigen/Core"
//#include "eigen/Eigen/SVD"
//#include "../eigen/Eigen/Eigen"

//using namespace Eigen;
using namespace std;
// CUDA forward declarations
void ArapKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* cellEnergies,
    float* rotations);

void ArapGradKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* rotations,
    float* gradxyz);

void arap_forward_cuda(
    const at::Tensor xyz, 
    const at::Tensor xyzP, 
    const at::Tensor neighborList, 
    const at::Tensor numNeighbors, 
    const at::Tensor accnumNeighbors, 
    const at::Tensor weightMatrix,
    const at::Tensor cellEnergies,
    const at::Tensor rotations)
{
    const int batchsize = xyz.size(0);
    const int n = xyz.size(1);
    const float* xyz_data = xyz.data<float>();
    const float* xyzP_data = xyzP.data<float>();
    int* neighborList_data = neighborList.data<int>();
    int* numNeighbors_data = numNeighbors.data<int>();
    int* accnumNeighbors_data = accnumNeighbors.data<int>();
    float* weightMatrix_data = weightMatrix.data<float>();
    const int blocklength_data = weightMatrix.size(1); //blocklength.data<int>();
    float* cellEnergies_data = cellEnergies.data<float>();
    float* rotations_data = rotations.data<float>();
    ArapKernelLauncher(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,accnumNeighbors_data,weightMatrix_data,blocklength_data,cellEnergies_data,rotations_data);
}

void arap_backward_cuda(
    const at::Tensor xyz, 
    const at::Tensor xyzP, 
    const at::Tensor neighborList, 
    const at::Tensor numNeighbors, 
    const at::Tensor accnumNeighbors, 
    const at::Tensor weightMatrix,
    const at::Tensor rotations,
    const at::Tensor gradxyz)
{
    const int batchsize = xyz.size(0);
    const int n = xyz.size(1);

    const float* xyz_data = xyz.data<float>();
    const float* xyzP_data = xyzP.data<float>();
    int* neighborList_data = neighborList.data<int>();
    int* numNeighbors_data = numNeighbors.data<int>();
    int* accnumNeighbors_data = accnumNeighbors.data<int>();
    float* weightMatrix_data = weightMatrix.data<float>();
    const int blocklength_data = weightMatrix.size(1); //blocklength.data<int>();
    float* rotations_data = rotations.data<float>();
    float* gradxyz_data = gradxyz.data<float>();

    ArapGradKernelLauncher(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,accnumNeighbors_data,weightMatrix_data,blocklength_data,rotations_data,gradxyz_data);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &arap_forward_cuda, "Arap forward (CUDA)");
    m.def("backward", &arap_backward_cuda, "Arap backward (CUDA)");
}
