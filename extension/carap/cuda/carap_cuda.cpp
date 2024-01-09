#include <torch/extension.h>
//#include "eigen/Eigen/Core"
//#include "eigen/Eigen/SVD"
//#include "../eigen/Eigen/Eigen"

//using namespace Eigen;
using namespace std;
// CUDA forward declarations
void CArapKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* alphas,
    float* areas,
    float* cellEnergies,
    float* rotations);

void CArapGradKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* rotations,
    float* gradxyz);

void carap_forward_cuda(
    const at::Tensor xyz, 
    const at::Tensor xyzP, 
    const at::Tensor neighborList, 
    const at::Tensor numNeighbors, 
    const at::Tensor weightMatrix,
    const at::Tensor alphas,
    const at::Tensor areas,
    const at::Tensor cellEnergies,
    const at::Tensor rotations)
{
    const int batchsize = xyz.size(0);
    const int n = xyz.size(1);
    const float* xyz_data = xyz.data<float>();
    const float* xyzP_data = xyzP.data<float>();
    int* neighborList_data = neighborList.data<int>();
    int* numNeighbors_data = numNeighbors.data<int>();
    float* weightMatrix_data = weightMatrix.data<float>();
    float* alphas_data = alphas.data<float>();
    float* areas_data = areas.data<float>();
    float* cellEnergies_data = cellEnergies.data<float>();
    float* rotations_data = rotations.data<float>();
    CArapKernelLauncher(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,weightMatrix_data,alphas_data,areas_data,cellEnergies_data,rotations_data);
}

void carap_backward_cuda(
    const at::Tensor xyz, 
    const at::Tensor xyzP, 
    const at::Tensor neighborList, 
    const at::Tensor numNeighbors, 
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
    float* weightMatrix_data = weightMatrix.data<float>();
    float* rotations_data = rotations.data<float>();
    float* gradxyz_data = gradxyz.data<float>();

    CArapGradKernelLauncher(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,weightMatrix_data,rotations_data,gradxyz_data);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &carap_forward_cuda, "CArap forward (CUDA)");
    m.def("backward", &carap_backward_cuda, "CArap backward (CUDA)");
}
