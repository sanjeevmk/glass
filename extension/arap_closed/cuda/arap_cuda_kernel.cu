#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h> #include "../eigen/Eigen/Core"
#include "../eigen/Eigen/SVD"
#include "../eigen/Eigen/Eigen"
#include "svd3_cuda.h"
using namespace Eigen;

__global__
void ArapClosedKernel(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* rotations,
    float* rhs)
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
            MatrixXf Pi(3,1);
            MatrixXf PiP(3,1);
            MatrixXf Ri(3,3);
            MatrixXf Rj(3,3);
            int rotIndex = bind*n*9 + indi*9;
            Ri(0,0) = rotations[rotIndex] ;  Ri(0,1) = rotations[rotIndex+1] ; Ri(0,2) = rotations[rotIndex+2] ;
            Ri(1,0) = rotations[rotIndex+3] ; Ri(1,1) = rotations[rotIndex+4] ; Ri(1,2) = rotations[rotIndex+5] ;
            Ri(2,0) = rotations[rotIndex+6] ; Ri(2,1) = rotations[rotIndex+7] ; Ri(2,2) = rotations[rotIndex+8] ;

            float rhsX = 0.0;
            float rhsY = 0.0;
            float rhsZ = 0.0;
            for(int nIndex=0;nIndex<nNeighbors;nIndex++){
		if(neighListIndex+nIndex>b*blocklength){
			return;
		}
                int neighborPointIndex = neighborList[neighListIndex+nIndex];
                int index2 = bind*n*3 + neighborPointIndex*3;
                float x2 = xyz[index2] ; float y2 = xyz[index2+1] ; float z2 = xyz[index2+2] ; 
                float x2P,y2P,z2P;
                x2P = xyzP[index2] ; y2P = xyzP[index2+1] ; z2P = xyzP[index2+2] ; 
                Pi(0,0) = x1-x2 ; Pi(1,0) = y1-y2 ; Pi(2,0) = z1-z2 ;
                PiP(0,0) = x1P - x2P ; PiP(1,0) = y1P - y2P ; PiP(2,0) = z1P - z2P ;
                float wij =  weightMatrix[weightMatrixIndex+nIndex];
		
                int neighRotIndex = bind*n*9 + neighborPointIndex*9;
                Rj(0,0) = rotations[neighRotIndex] ;  Rj(0,1) = rotations[neighRotIndex+1] ; Rj(0,2) = rotations[neighRotIndex+2] ;
                Rj(1,0) = rotations[neighRotIndex+3] ; Rj(1,1) = rotations[neighRotIndex+4] ; Rj(1,2) = rotations[neighRotIndex+5] ;
                Rj(2,0) = rotations[neighRotIndex+6] ; Rj(2,1) = rotations[neighRotIndex+7] ; Rj(2,2) = rotations[neighRotIndex+8] ;

                MatrixXf rhs_ni = wij*0.5*((Ri+Rj)*Pi);
                rhsX += rhs_ni(0,0);
                rhsY += rhs_ni(1,0);
                rhsZ += rhs_ni(2,0);
            }
            rhs[gradIndex] = rhsX;
            rhs[gradIndex+1] = rhsY;
            rhs[gradIndex+2] = rhsZ;
        }
    }
}

void ArapClosedKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const float* xyzP,
    int* neighborList,
    int* numNeighbors,
    int* accnumNeighbors,
    const float* weightMatrix,
    const int blocklength,
    float* rotations,
	float* rhs)
{
	ArapClosedKernel<<<dim3(b,(n+7)/8),8>>>(b, n, xyz, xyzP,neighborList,numNeighbors,accnumNeighbors,weightMatrix,blocklength,rotations,rhs);
	//ArapGradKernel2D<<<dim3(32,16,1),512>>>(b, n, xyz, xyzP,neighborList,numNeighbors,weightMatrix,rotations,gradxyz);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in arap grad computation: %s\n", cudaGetErrorString(err));
}
