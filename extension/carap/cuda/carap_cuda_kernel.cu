#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/SVD"
#include "../eigen/Eigen/Eigen"
#include "svd3_cuda.h"
using namespace Eigen;

__global__
void RotationKernel2D(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* alphas,
    float* areas,
    float* rotations)
{
	for(int bind=blockIdx.x;bind<b;bind+=gridDim.x){
        float alpha = alphas[bind];
        float area = areas[bind];
        for(int indi=threadIdx.x+blockIdx.y*blockDim.x;indi<n;indi+=blockDim.x*gridDim.y){
        //for(int indi=blockDim.x*blockIdx.x+threadIdx.x;indi<n;indi+=blockDim.x*gridDim.x){
            int index = bind*n*3 + indi*3;
            int neighIndex = bind*n + indi;
            int neighListIndex = bind*n*n + indi*n;
            int weightMatrixIndex = bind*n*n + indi*n;
            int nNeighbors = numNeighbors[neighIndex];
            float x1 = xyz[index] ; float y1 = xyz[index+1] ; float z1 = xyz[index+2] ;
            float x1P,y1P,z1P;
            x1P = xyzP[index] ; y1P = xyzP[index+1] ; z1P = xyzP[index+2];

            MatrixXf Pi(3,nNeighbors);
            MatrixXf PiP(nNeighbors,3);
            MatrixXf wij = MatrixXf::Zero(nNeighbors,nNeighbors);
            MatrixXf Ri(3,3);
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                int index2 = bind*n*3 + neighborPointIndex*3;
                float x2 = xyz[index2] ; float y2 = xyz[index2+1] ; float z2 = xyz[index2+2];
                float x2P,y2P,z2P;
                x2P = xyzP[index2] ; y2P = xyzP[index2+1] ; z2P = xyzP[index2+2] ;
                Pi(0,nIndex) = x1-x2 ; Pi(1,nIndex) = y1-y2 ; Pi(2,nIndex) = z1-z2 ;
                wij(nIndex,nIndex) =  weightMatrix[weightMatrixIndex+neighborPointIndex];
                PiP(nIndex,0) = x1P - x2P ; PiP(nIndex,1) = y1P - y2P ; PiP(nIndex,2) = z1P - z2P ;
            }
            MatrixXf Si = 2*Pi*wij*PiP;
            MatrixXf SiNeighbors = MatrixXf::Zero(3,3);
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                SiNeighbors += (4*alpha*area*wij(nIndex,nIndex)*MatrixXf::Identity(3,3));
            }
            Si += SiNeighbors;
            float u00,u01,u02,u10,u11,u12,u20,u21,u22;
            float v00,v01,v02,v10,v11,v12,v20,v21,v22;
            float s0,s1,s2;
            svd(Si(0,0),Si(0,1),Si(0,2),Si(1,0),Si(1,1),Si(1,2),Si(2,0),Si(2,1),Si(2,2),u00,u01,u02,u10,u11,u12,u20,u21,u22,s0,s1,s2,v00,v01,v02,v10,v11,v12,v20,v21,v22);
			MatrixXf U(3,3);
            MatrixXf V(3,3);
            U(0,0) = u00; U(0,1) = u01 ; U(0,2) = u02;
            U(1,0) = u10; U(1,1) = u11 ; U(1,2) = u12;
            U(2,0) = u20; U(2,1) = u21 ; U(2,2) = u22;
            V(0,0) = v00; V(0,1) = v01 ; V(0,2) = v02;
            V(1,0) = v10; V(1,1) = v11 ; V(1,2) = v12;
            V(2,0) = v20; V(2,1) = v21 ; V(2,2) = v22;
            Ri = V*U.transpose();
            float determinant = Ri(0,0)*(Ri(1,1)*Ri(2,2) - Ri(1,2)*Ri(2,1)) - Ri(0,1)*(Ri(1,0)*Ri(2,2) - Ri(1,2)*Ri(2,0)) + Ri(0,2)*(Ri(1,0)*Ri(2,1) - Ri(1,1)*Ri(2,0)) ;
            if(determinant<0){
                U.col(1)*=-1;
                Ri = V*U.transpose();
            }
            int rotIndex = bind*n*9 + indi*9;
            rotations[rotIndex] = Ri(0,0) ; rotations[rotIndex+1] = Ri(0,1) ; rotations[rotIndex+2] = Ri(0,2) ;
            rotations[rotIndex+3] = Ri(1,0) ; rotations[rotIndex+4] = Ri(1,1) ; rotations[rotIndex+5] = Ri(1,2) ;
            rotations[rotIndex+6] = Ri(2,0) ; rotations[rotIndex+7] = Ri(2,1) ; rotations[rotIndex+8] = Ri(2,2) ;
        }
    }
}

__global__
void CArapKernel2D(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* alphas,
    float* areas,
    float* cellEnergies,
    float* rotations)
{
	for(int bind=blockIdx.x;bind<b;bind+=gridDim.x){
        float alpha = alphas[bind];
        float area = areas[bind];
        for(int indi=threadIdx.x+blockIdx.y*blockDim.x;indi<n;indi+=blockDim.x*gridDim.y){
        //for(int indi=blockDim.x*blockIdx.x+threadIdx.x;indi<n;indi+=blockDim.x*gridDim.x){
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
            MatrixXf Ri(3,3);
            int rotIndex = bind*n*9 + indi*9;
            Ri(0,0) = rotations[rotIndex] ;  Ri(0,1) = rotations[rotIndex+1] ; Ri(0,2) = rotations[rotIndex+2] ;
            Ri(1,0) = rotations[rotIndex+3] ; Ri(1,1) = rotations[rotIndex+4] ; Ri(1,2) = rotations[rotIndex+5] ;
            Ri(2,0) = rotations[rotIndex+6] ; Ri(2,1) = rotations[rotIndex+7] ; Ri(2,2) = rotations[rotIndex+8] ;
            float cellEnergyX = 0.0;
            float cellEnergyY = 0.0;
            float cellEnergyZ = 0.0;
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                MatrixXf Rj(3,3);
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                MatrixXf RiPi =  Ri*Pi.col(nIndex) ;
                MatrixXf energyNeigh = (PiP.row(nIndex) - RiPi.transpose());
                MatrixXf energyNeigh_pow2 = energyNeigh.array().square();
                MatrixXf weightedEnergyNeigh_pow2 = wij(nIndex,nIndex)*energyNeigh_pow2;
                cellEnergyX += weightedEnergyNeigh_pow2(0,0);
                cellEnergyY += weightedEnergyNeigh_pow2(0,1);

                int neighRotIndex = bind*n*9 + neighborPointIndex*9;
                Rj(0,0) = rotations[neighRotIndex] ;  Rj(0,1) = rotations[neighRotIndex+1] ; Rj(0,2) = rotations[neighRotIndex+2] ;
                Rj(1,0) = rotations[neighRotIndex+3] ; Rj(1,1) = rotations[neighRotIndex+4] ; Rj(1,2) = rotations[neighRotIndex+5] ;
                Rj(2,0) = rotations[neighRotIndex+6] ; Rj(2,1) = rotations[neighRotIndex+7] ; Rj(2,2) = rotations[neighRotIndex+8] ;

                float bending = alpha*area*wij(nIndex,nIndex)*((Ri-Rj).norm());
                cellEnergyX += (bending/3.0);
                cellEnergyY += (bending/3.0);
                cellEnergyZ += (bending/3.0);
            }
            int energyIndex = bind*n*3 + indi*3;
            cellEnergies[energyIndex] = cellEnergyX;
            cellEnergies[energyIndex+1] = cellEnergyY;
            cellEnergies[energyIndex+2] = cellEnergyZ;
        }
    }
}

__global__
void CArapGradKernel2D(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* rotations,
    float* gradxyz)
{
	for(int bind=blockIdx.x;bind<b;bind+=gridDim.x){
        for(int indi=threadIdx.x+blockIdx.y*blockDim.x;indi<n;indi+=blockDim.x*gridDim.y){
        //for(int indi=blockDim.x*blockIdx.x+threadIdx.x;indi<n;indi+=blockDim.x*gridDim.x){
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

                MatrixXf gradj = 4*wij*(PiP - 0.5*(Ri+Rj)*Pi);
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
    float* rotations)
{
	//ArapKernel<<<64,512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,cellEnergies,rotations);
	//ArapKernel2D<<<64,512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,cellEnergies,rotations);
    RotationKernel2D<<<dim3(32,16,1),512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,alphas,areas,rotations);
    cudaDeviceSynchronize();
	CArapKernel2D<<<dim3(32,16,1),512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,alphas,areas,cellEnergies,rotations);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in carap : %s\n", cudaGetErrorString(err));
}

void CArapGradKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    const float* weightMatrix,
    float* rotations,
	float* gradxyz)
{
	//ArapGradKernel<<<64,512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,rotations,gradxyz);
	//ArapGradKernel2D<<<64,512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,rotations,gradxyz);
	CArapGradKernel2D<<<dim3(32,16,1),512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,rotations,gradxyz);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in carap grad computation: %s\n", cudaGetErrorString(err));
}
