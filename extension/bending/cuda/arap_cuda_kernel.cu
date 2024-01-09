#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/SVD"
#include "../eigen/Eigen/Eigen"
#include "svd3_cuda.h"
using namespace Eigen;

__global__
void ArapRotationKernel(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* rotations,
    float* updated_rotations)
{
	for(int bind=blockIdx.x;bind<b;bind+=gridDim.x){
        for(int indi=threadIdx.x+blockIdx.y*blockDim.x;indi<n;indi+=blockDim.x*gridDim.y){
        //for(int indi=blockDim.x*blockIdx.x+threadIdx.x;indi<n;indi+=blockDim.x*gridDim.x){
            int gradIndex = bind*n*3 + indi*3;
            int index = bind*n*3 + indi*3;

            int neighIndex = bind*n + indi;
            int nNeighbors = numNeighbors[neighIndex];

            int accneighborsIndex = bind*n + indi;
            int numAccNeighbors = accnumNeighbors[accneighborsIndex];

            int neighListIndex = bind*blocklength + numAccNeighbors;
            int weightMatrixIndex = bind*blocklength + numAccNeighbors;

            float x1 = xyz[index] ; float y1 = xyz[index+1] ; float z1 = xyz[index+2] ;
            float x1P,y1P,z1P;
            x1P = xyzP[index] ; y1P = xyzP[index+1] ; z1P = xyzP[index+2] ;
            MatrixXf Pi(3,nNeighbors);
            MatrixXf PiP(nNeighbors,3);
            MatrixXf wij = MatrixXf::Zero(nNeighbors,nNeighbors);
            MatrixXf Ri(3,3);
            MatrixXf Rj(3,3);

            MatrixXf neighborRotTerms = MatrixXf::Zero(3,3);
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                int index2 = bind*n*3 + neighborPointIndex*3;
                float x2 = xyz[index2] ; float y2 = xyz[index2+1] ; float z2 = xyz[index2+2] ; 
                float x2P,y2P,z2P;
                x2P = xyzP[index2] ; y2P = xyzP[index2+1] ; z2P = xyzP[index2+2] ; 
                Pi(0,nIndex) = x1-x2 ; Pi(1,nIndex) = y1-y2 ; Pi(2,nIndex) = z1-z2 ;
                wij(nIndex,nIndex) =  weightMatrix[weightMatrixIndex+nIndex];
                PiP(nIndex,0) = x1P - x2P ; PiP(nIndex,1) = y1P - y2P ; PiP(nIndex,2) = z1P - z2P ;

                int neighRotIndex = bind*n*9 + neighborPointIndex*9;
                Rj(0,0) = rotations[neighRotIndex] ;  Rj(0,1) = rotations[neighRotIndex+1] ; Rj(0,2) = rotations[neighRotIndex+2] ;
                Rj(1,0) = rotations[neighRotIndex+3] ; Rj(1,1) = rotations[neighRotIndex+4] ; Rj(1,2) = rotations[neighRotIndex+5] ;
                Rj(2,0) = rotations[neighRotIndex+6] ; Rj(2,1) = rotations[neighRotIndex+7] ; Rj(2,2) = rotations[neighRotIndex+8] ;

                MatrixXf _neighRotTerm = 2.0 * wij(nIndex,nIndex) * Rj.transpose();
                neighborRotTerms += _neighRotTerm;
            }

            MatrixXf Si = ((Pi*wij)*PiP) + neighborRotTerms;
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
                U.col(2)*=-1;
                Ri = V*U.transpose();
            }
            int rotIndex = bind*n*9 + indi*9;
            updated_rotations[rotIndex] = Ri(0,0) ; updated_rotations[rotIndex+1] = Ri(0,1) ; updated_rotations[rotIndex+2] = Ri(0,2) ;
            updated_rotations[rotIndex+3] = Ri(1,0) ; updated_rotations[rotIndex+4] = Ri(1,1) ; updated_rotations[rotIndex+5] = Ri(1,2) ;
            updated_rotations[rotIndex+6] = Ri(2,0) ; updated_rotations[rotIndex+7] = Ri(2,1) ; updated_rotations[rotIndex+8] = Ri(2,2) ;
        }
    }
}

void ArapRotationKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* rotations,
	float* updated_rotations)
{
	ArapRotationKernel<<<dim3(b,(n+7)/8),8>>>(b, n, xyz, xyzP,neighborList,numNeighbors,accnumNeighbors,weightMatrix,blocklength,rotations,updated_rotations);
	//ArapGradKernel2D<<<dim3(32,16,1),512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,rotations,gradxyz);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in arap grad computation: %s\n", cudaGetErrorString(err));
}
