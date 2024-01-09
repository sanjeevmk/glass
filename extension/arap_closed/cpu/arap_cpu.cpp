#include <torch/torch.h>
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/SVD"
#include "../eigen/Eigen/Eigen"

using namespace Eigen;
using namespace std;

void arapEnergyCompute(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* cellEnergies,
    float* rotations)
{
    for(int bind=0;bind<b;bind++){
        for(int indi=0;indi<n;indi++){
            int index = bind*n*3 + indi*3;
            int neighIndex = bind*n + indi;
            int neighListIndex = bind*n*n + indi*n;
            int weightMatrixIndex = bind*n*n + indi*n;
            int nNeighbors = numNeighbors[neighIndex];
            float x1 = xyz[index] ; float y1 = xyz[index+1] ; float z1 = xyz[index+2] ;
            float x1P,y1P,z1P;
            x1P = xyzP[index] ; y1P = xyzP[index+1] ; z1P = xyzP[index+2] ;

            MatrixXf Pi(3,nNeighbors);
            MatrixXf PiP(nNeighbors,3);
            MatrixXf wij = MatrixXf::Zero(nNeighbors,nNeighbors);
            MatrixXf Ri(3,3);
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                int index2 = bind*n*3 + neighborPointIndex*3;
                float x2 = xyz[index2] ; float y2 = xyz[index2+1] ; float z2 = xyz[index2+2] ; 
                float x2P,y2P,z2P;
                x2P = xyzP[index2] ; y2P = xyzP[index2+1] ; z2P = xyzP[index2+2] ; 
                Pi(0,nIndex) = x1-x2 ; Pi(1,nIndex) = y1-y2 ; Pi(2,nIndex) = z1-z2 ;

                wij(nIndex,nIndex) =  weightMatrix[weightMatrixIndex+neighborPointIndex];
                PiP(nIndex,0) = x1P - x2P ; PiP(nIndex,1) = y1P - y2P ; PiP(nIndex,2) = z1P - z2P ;
            }
            MatrixXf Si = Pi*wij*PiP;
            JacobiSVD<MatrixXf> svd(Si,ComputeFullU|ComputeFullV);
            MatrixXf U = svd.matrixU();
            MatrixXf V = svd.matrixV();
            Ri = V*U.transpose();
            if(Ri.determinant()<0){
                U.col(2)*=-1;
                Ri = V*U.transpose();
            }
            int rotIndex = bind*n*9 + indi*9;
            rotations[rotIndex] = Ri(0,0) ; rotations[rotIndex+1] = Ri(0,1) ; rotations[rotIndex+2] = Ri(0,2) ;
            rotations[rotIndex+3] = Ri(1,0) ; rotations[rotIndex+4] = Ri(1,1) ; rotations[rotIndex+5] = Ri(1,2) ;
            rotations[rotIndex+6] = Ri(2,0) ; rotations[rotIndex+7] = Ri(2,1) ; rotations[rotIndex+8] = Ri(2,2) ;
            float cellEnergyX = 0.0;
            float cellEnergyY = 0.0;
            float cellEnergyZ = 0.0;
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                MatrixXf RiPi =  Ri*Pi.col(nIndex) ;
                MatrixXf energyNeigh = (PiP.row(nIndex) - RiPi.transpose());
                MatrixXf energyNeigh_pow2 = energyNeigh.array().square();
                MatrixXf weightedEnergyNeigh_pow2 = wij(nIndex,nIndex)*energyNeigh_pow2;
                cellEnergyX += weightedEnergyNeigh_pow2(0,0);
                cellEnergyY += weightedEnergyNeigh_pow2(0,1);
                cellEnergyZ += weightedEnergyNeigh_pow2(0,2);
            }
            int energyIndex = bind*n*3 + indi*3;
            cellEnergies[energyIndex] = cellEnergyX;
            cellEnergies[energyIndex+1] = cellEnergyY;
            cellEnergies[energyIndex+2] = cellEnergyZ;
        }
    }
}

void arapGradCompute(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* rotations,
    float* gradxyz)
{
    for(int bind=0;bind<b;bind++){
        for(int indi=0;indi<n;indi++){
            int gradIndex = bind*n*3 + indi*3;
            int index = bind*n*3 + indi*3;
            int neighIndex = bind*n + indi;
            int neighListIndex = bind*n*n + indi*n;
            int weightMatrixIndex = bind*n*n + indi*n;
            int nNeighbors = numNeighbors[neighIndex];
            float x1 = xyz[index] ; float y1 = xyz[index+1] ; float z1 = xyz[index+2] ;
            float x1P,y1P,z1P;
            x1P = xyzP[index] ; y1P = xyzP[index+1] ; z1P = xyzP[index+2] ;

            MatrixXf Pi(3,1);
            MatrixXf PiP(3,1);
            MatrixXf Ri(3,3);
            MatrixXf Rj(3,3);
            int rotIndex = bind*n*9 + indi*9;
            Ri(0,0) = rotations[rotIndex] ;  Ri(0,1) = rotations[rotIndex+1] ; Ri(0,2) = rotations[rotIndex+2] ;
            Ri(1,0) = rotations[rotIndex+3] ; Ri(1,1) = rotations[rotIndex+4] ; Ri(1,2) = rotations[rotIndex+5] ;
            Ri(2,0) = rotations[rotIndex+6] ; Ri(2,1) = rotations[rotIndex+7] ; Ri(2,2) = rotations[rotIndex+8] ;

            float gradIX = 0.0;
            float gradIY = 0.0;
            float gradIZ = 0.0;
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                int index2 = bind*n*3 + neighborPointIndex*3;
                float x2 = xyz[index2] ; float y2 = xyz[index2+1] ; float z2 = xyz[index2+2] ; 
                float x2P,y2P,z2P;
                x2P = xyzP[index2] ; y2P = xyzP[index2+1] ; z2P = xyzP[index2+2] ; 
                Pi(0,0) = x1-x2 ; Pi(1,0) = y1-y2 ; Pi(2,0) = z1-z2 ;
                PiP(0,0) = x1P - x2P ; PiP(1,0) = y1P - y2P ; PiP(2,0) = z1P - z2P ;
                float wij =  weightMatrix[weightMatrixIndex+neighborPointIndex];

                int neighRotIndex = bind*n*9 + neighborPointIndex*9;
                Rj(0,0) = rotations[neighRotIndex] ;  Rj(0,1) = rotations[neighRotIndex+1] ; Rj(0,2) = rotations[neighRotIndex+2] ;
                Rj(1,0) = rotations[neighRotIndex+3] ; Rj(1,1) = rotations[neighRotIndex+4] ; Rj(1,2) = rotations[neighRotIndex+5] ;
                Rj(2,0) = rotations[neighRotIndex+6] ; Rj(2,1) = rotations[neighRotIndex+7] ; Rj(2,2) = rotations[neighRotIndex+8] ;

                MatrixXf gradj = 4*wij*(PiP - (0.5*(Ri+Rj))*Pi);
                gradIX += gradj(0,0);
                gradIY += gradj(1,0);
                gradIZ += gradj(2,0);
            }
            gradxyz[gradIndex] = gradIX;
            gradxyz[gradIndex+1] = gradIY;
            gradxyz[gradIndex+2] = gradIZ;
        }
    }
}
void arap_forward(
    const at::Tensor xyz, 
    const at::Tensor xyzP, 
    const at::Tensor neighborList, 
    const at::Tensor numNeighbors, 
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
    float* weightMatrix_data = weightMatrix.data<float>();
    float* cellEnergies_data = cellEnergies.data<float>();
    float* rotations_data = rotations.data<float>();

    arapEnergyCompute(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,weightMatrix_data,cellEnergies_data,rotations_data);
}

void arap_backward(
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
    arapGradCompute(batchsize, n, xyz_data, xyzP_data,neighborList_data,numNeighbors_data,weightMatrix_data,rotations_data,gradxyz_data);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &arap_forward, "Arap forward");
    m.def("backward", &arap_backward, "Arap backward");
}
