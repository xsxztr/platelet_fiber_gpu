/*
 sem_kernel.cu
 
 GPU kernel functions.

 Author: Scott Christley <schristley@mac.com>
 
 Copyright (C) 2010 Scott Christley
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met: 1. Redistributions of source code must retain the above
 copyright notice, this list of conditions and the following
 disclaimer.  2. Redistributions in binary form must reproduce the
 above copyright notice, this list of conditions and the following
 disclaimer in the documentation and/or other materials provided with
 the distribution.  3. The name of the author may not be used to
 endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


 */
// parameters

#ifndef SEM_KERNEL_CU
#define SEM_KERNEL_CU

#include <stdlib.h>
#include <curand_kernel.h>
//#include "LB_kernel.cu"
// boundary conditions
#define PERIODIC_BOUNDARY 0
//#define PERIODIC_BOUNDARY 1

//#define HARD_Z 0
#define HARD_Z 1

// For 128 and 250 cells
#define BOUNDARY_X 100.0
#define BOUNDARY_Y 100.0
#define BOUNDARY_Z 10.0

// Parameters for Myxo code

#define NUMOFRECEPTORS model_Parameters[0]
#define SEED model_Parameters[1]
#define k_spring  model_Parameters[2] // unit: nN/um
#define Kf0 model_Parameters[3]
#define Kr0 model_Parameters[4]
#define SHEAR_RATE model_Parameters[5]
#define INIT_RHO model_Parameters[6]
#define L0 model_Parameters[7]
#define T0 model_Parameters[8]
#define M0 model_Parameters[9]
#define F0 model_Parameters[10]
#define UMAX model_Parameters[11]
#define OMEGA model_Parameters[13]
#define P0 model_Parameters[14]
#define CL model_Parameters[15]//dimensionless lattice speed
#define UNIT_FACTOR model_Parameters[16]
#define Ks model_Parameters[17]
#define Kv model_Parameters[18]
#define Kbend model_Parameters[19]

#define NO_NB 1 // turn off non-bonded for debugging
#define BONDTHRESHOLD 0.078 //bond length of GPIBalpha-vWF 
#define Kon0 1e-5 
#define Koff0 3.21
#define KbT 4.05
#define SigmaoverkBT 3.7365e9
#define k_receptvWF 10.0 //unit:nN/um
#define Em 2.5e1      //unit:kPa
//#define SlimeDir 1 // set the slime at one end for now.
//
////////////////////////////////////////////////////////////////////////////////////
//


#define norm(dx, dy, dz) sqrt( ( dx )*( dx ) + ( dy )*( dy ) + ( dz )*( dz ) )
__constant__ double I[3][3] = {{1, 0, 0},{0, 1, 0},{0, 0, 1}};
__constant__ double Ni[6][6] = {{2, 2, 4, -3, -3, 1},{2, 0, 0, -1, 0, 0},
                                {0, 2, 0, 0, -1, 0}, {-4, 0, -4, 4, 0, 0},
                                {0, -4, -4, 0, 4, 0},{0, 0, 4, 0, 0, 0}
                               };
__constant__ double xi_eta[6][2] = { {0, 0}, {1, 0}, {0, 1},
                                     {0.5, 0}, {0, 0.5}, {0.5, 0.5}
                                   };
__device__ double dN_dxi(double xi, double eta, double a, double c, double d){
  double result = 2 * a * xi + c * eta + d;
  return result;
}
__device__ double dN_deta(double xi, double eta, double b, double c, double e){
  double result = 2 * b * eta + c * xi + e;
  return result;
}
__device__ double trace(double matrix[3][3]){
  double result = matrix[0][0] + matrix[1][1] + matrix[2][2];
  return result;
}
__device__ double dot(double a[3], double b[3]){
  double result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return result;
}
__device__ double norm_line(double a[3], double b[3]){
  double result = norm(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  return result;
}
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
     assumed = old;
     old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

  typedef struct _sem_GPUgrids {
    int maxCells;
    int numOfCells;
    int maxElements;
    int SurfElem;
    int newnodeNum;
    int numReceptorsPerElem;
    int totalBond;
    float dt;
    int *numOfElements;
    int *elementType;
    double *bondLengths;
    int *triElem;
    float *receptor_r1;
    float *receptor_r2;
    cudaExtent iextent;
    cudaExtent dextent;
    cudaPitchedPtr receptBond;
    cudaPitchedPtr randNum;
//double    double *rho;
    double *X_Ref;
    double *X;
    double *V_X;
   // float *PreV_X;
    double *F_X;
    double *Y_Ref;
    double *Y;
    double *RY;
    double *V_Y;
   // float *PreV_Y;
    double *F_Y;
    double *Z_Ref;
    double *Z;
    double *V_Z;
   // float *PreV_Z;
    double *F_Z;
    int *node_share_Elem;
    int *node_nbrElemNum;
    int *node_nbrNodes;
    double *N;
    double *nelem;
    double *n;
    double *A;
    double *S;
    double *S0;
    double S0_all;
    double S_all;
    double V;
    double V0;
    double *tau;
    double *Kapa;
    double *km;
    double *K;
    double *q;
    double *Laplace_km;
    size_t pitch;
    size_t bondpitch;
//    float *cellCenterX;
 //   float *cellCenterY;
 //   float *cellCenterZ;
    void *devImage; //pointer to the image of the grids on device
    curandState *devState;
  } sem_GPUgrids;

///////////////////////////////////////////////////////////
/// gpu code for fiber added by Shixin Xu 2015/11/18
////////////////////////////////////////////////////////
typedef  struct _sem_GPUfiber{
	int dt;// time step
	int Number_of_Node; // total number of node;
	int Number_of_Bond; // total number of connection bond 
	int Max_Number_Connect; // max number of conncection to one node

	int *Node_Number; // serial number of each node
	int *Node_connnet; // the serial numeber of node that connect to current node
	double *Fsize_connect; // the fiber thickness of the node that connect to cuurrent node 
	int    *Node_type; // the type of node depending on the connection number 
	int    *Number_of_connect; // number of neibour node 
	double *Fsize; // fiber thickness
	double *Bond_Length; // length of bond 
	
/////////// x coodinate information: node, Velocity, Force, initial Node
	double *Node_x; 
	double *V_x;
	double *F_x;
	double *Nodein_x;
/////////// y direction////////////////////////////////////////////////	
	double *Node_y;
	double *V_y;
	double *F_y;
	double *Nodein_y;
/////////// z direction //////////////////////////////////////////////	
	double *Node_z;
	double *V_z;
	double *F_z;
	double *Nodein_z;
	
	size_t pitch;
	size_t bondpitch;

} sem_GPUfiber;



////////////////////////////////////////////////////////////////////////////////////
//

__device__ int stepL1 = 0, stepR1 = 0;
//This is were 2-body and 3-body and non-bonded forces are calculated and summed up
 
__device__ long holdrand = 1L;

__device__ unsigned int get_random()
{
    
    return (((holdrand = holdrand * 214013L + 2531011L) >> 16) & 0x7fff);
}
 
__device__ void gaussian_elimination(double** gauss, int m, int n)
{
   int i = 0, j = 0, k, l;
   
   while (i < m && j < n-1){
        int maxi = i;
        for (k = i + 1; k < m; k++){
            if (abs(*((double*)gauss + k * n + j)) > abs(*((double*)gauss + maxi * n + j))){
               maxi = k;
            }
        }
        if (*((double*)gauss + maxi * n + j) != 0 ){
           double buff,initial;
           for (k = j; k < n; k++){
               buff = *((double*)gauss + i * n + k);
               *((double*)gauss + i * n + k) = *((double*)gauss + maxi * n + k);
               *((double*)gauss + maxi * n + k) = buff;
               if(k == j) initial = *((double*)gauss + i * n + j);
               *((double*)gauss + i * n + k) /= initial; 
           }
           for (k = i + 1; k < m; k++){
               initial = *((double*)gauss + k * n + j);
               for (l = j; l < n; l++){
                    *((double*)gauss + k * n + l) -= initial  * (*((double*)gauss + i * n + l)); 
               }
           }
           i++;
        }
        j++;
       /*if (idx == 0){
          for (int k = 0; k < 3; k++)
              printf("%f %f %f %f\n",*((double*)gauss + k * n), *((double*)gauss + k * n + 1), 
                     *((double*)gauss + k * n + 2), *((double*)gauss + k * n + 3));
       }*/
   }
}

__device__ void solve_linear_eqns(double A[3][3], double *A1, double *A2, double *A3, double *A4, double *A5, double *A6){
   int i, j;
   double gauss[3][4];
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           gauss[0][j] = A1[j];
           gauss[1][j] = A3[j];
           gauss[2][j] = A5[j]; 
       } 
       gauss[0][3] = A2[i];
       gauss[1][3] = A4[i];
       gauss[2][3] = A6[i];
       /*if (idx == 0){
          for (int k = 0; k < 3; k++)
              printf("%f %f %f %f\n",gauss[k][0], gauss[k][1], gauss[k][2], gauss[k][3]);
       }*/
       gaussian_elimination((double**)gauss, 3, 4);
       /*if (idx == 0){
          for (int k = 0; k < 3; k++)
              printf("%f %f %f %f\n",gauss[k][0], gauss[k][1], gauss[k][2], gauss[k][3]);
       }*/
       A[i][2] = gauss[2][3];
       A[i][1] = gauss[1][3] - A[i][2] * gauss[1][2];
       A[i][0] = gauss[0][3] - A[i][2] * gauss[0][2] - A[i][1] * gauss[0][1];
   } 
}
__global__ void setup_RNG_kernel(curandState *state) {
   int id = threadIdx.x + blockIdx.x * blockDim.x   ; // 65;
   /* Each thread gets same seed, a different sequence number,
    no offset */

   curand_init((int)SEED, id, 0, &state[id]);
}

__global__ void
sem_Force_kernel(void* g)
{
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
   int cellNum = 0;
   int elemNum = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (cellNum >= grids->maxCells) return;
   if (elemNum >= grids->numOfElements[cellNum]) return;

// Calculate new forces for updated coordinates
   size_t pitch = grids->pitch;
   size_t bondpitch = grids->bondpitch;
   double *X = grids->X;
   double *Y = grids->RY;
   double *Z = grids->Z;
   double *F_X = grids->F_X;
   double *F_Y = grids->F_Y;
   double *F_Z = grids->F_Z;
   double *bondLengths = grids->bondLengths;
  
   double dx1 = 0.0;
   double dy1 = 0.0;
   double dz1 = 0.0;
   double weight = 0.0;  
   double fconst = 0.0;
   double r1 = 0.0;
   int i, j, k;
  // double fx , fy, fz;

// New 2-body bonded using BondType Array
// Bonded interaction Calculations

// Every Element in Cell check for bonds with other SCE 

//This Naive scheme does not use opposite reaction to add force to both SCEs, each element determines it owns force from all bonded interaction
   for (i = 0; i < grids->maxElements; i++){
    int neighborSCE = i;
    double *row = (double*)((char*)bondLengths + elemNum * bondpitch);  
    double myBond = row[neighborSCE];

    
    if (myBond > 1e-6){
    
       double *row1 = (double*)((char*)X + neighborSCE * pitch);
       double *row2 = (double*)((char*)X + elemNum * pitch);
       dx1 = row1[cellNum] - row2[cellNum];
        
       row1 = (double*)((char*)Y + neighborSCE * pitch);
       row2 = (double*)((char*)Y + elemNum * pitch);
       dy1 = row1[cellNum] - row2[cellNum];

       row1 = (double*)((char*)Z + neighborSCE * pitch);
       row2 = (double*)((char*)Z + elemNum * pitch);
       dz1 = row1[cellNum] - row2[cellNum];

       r1 = norm(dx1, dy1, dz1);
       weight = r1 - myBond;
       fconst = k_spring * weight / r1; //unit: nN
       //float myBond3 = myBond * myBond * myBond;
       double myBond3 = 1.0;
       // Force acting on this element due to bond is opposite to the force acting on the next element
       row = (double*)((char*)F_X + elemNum * pitch);
       row[cellNum] +=  fconst * dx1 / myBond3 /F0;
    //   fx = fconst * dx1 /myBond3;//row[cellNum];
       row = (double*)((char*)F_Y + elemNum * pitch);
       row[cellNum] +=  fconst * dy1 / myBond3 /F0;
     //  fy = fconst * dy1 / myBond3;//row[cellNum];
       row = (double*)((char*)F_Z + elemNum * pitch);
       row[cellNum] +=  fconst * dz1 / myBond3 /F0;
      // fz = fconst * dz1 / myBond3;//row[cellNum];
       /*if (elemNum == 0){
            printf("i = %d, fx = %.16e, fy = %.16e , fz = %.16e\n", i, fx, fy, fz);  
       }*/
    }
    
   }
   /*if (elemNum == 0){
      stepR1 ++;
      printf("R: step = %d, Fx = %.16e, Fy = %.16e , Fz = %.16e\n", stepR1, fx, fy, fz);  
   }*/
  /* if (elemNum == 52){
      stepL1 ++;
      printf("L: step = %d, Fx = %.16e, Fy = %.16e , Fz = %.16e\n", stepL1, fx, fy, fz);  
   }*/
   
// end of bonded interaction

//Calculate the forces due to interaction with wall
   __syncthreads();
   if (elemNum == 0) {
      double dh, dh_min;
      double *row; 
      int elem_idx = 0;
      dh_min = Z[cellNum];
      for (i = 1; i < grids->maxElements;i++){
          row = (double*)((char*)Z + i * pitch);
          dh = row[cellNum];
          if (dh < dh_min){
             dh_min = dh;
             elem_idx = i;
          }
      }
      
      if (dh_min <= 0.02){
         double rep_Force = 5e-4*2000*exp(-2000*dh_min)/(1-exp(-2000*dh_min));//repulsive force from the wall unit: N
        // printf("rep_Force = %e\n", rep_Force);
         row = (double*)((char*)F_Z + elem_idx * pitch);
         row[cellNum] += rep_Force/F0;
      }
   }

//Non-bonded Interactions between two cells: short range repulsive force calculation. See Mody and King, Biophysical Journal, 2008:95(5)
    
   if (NO_NB) return;
   
   double dist; 
   double dx,dy,dz,dr,Force;
   
   __shared__ int minCellsElements[103];
   __shared__ double minCellsElementsDist[103];
   
   for (j = cellNum + 1; j < grids->maxCells; ++j) {
	 
     double dist_min;
     int RepulsiveElement = 0;

     for (k = 0; k < grids->numOfElements[j]; ++k) {
       
        double *row1 = (double*)((char*)X + elemNum * pitch);
        double *row2 = (double*)((char*)X + k * pitch);
        dx = row1[cellNum] - row2[j];
        row1 = (double*)((char*)Y + elemNum * pitch);
        row2 = (double*)((char*)Y + k * pitch);
        dy = row1[cellNum] - row2[j];
        row1 = (double*)((char*)Z + elemNum * pitch);
        row2 = (double*)((char*)Z + k * pitch);
        dz = row1[cellNum] - row2[j];
        dist = norm(dx, dy, dz);
	if (k == 0) dist_min = dist;
        if (dist < dist_min) {         
           dist_min = dist;
           RepulsiveElement = k;      
        }
     }
	minCellsElements[elemNum] = RepulsiveElement;
        minCellsElementsDist[elemNum] = dist_min;
        __syncthreads();
        
        if (elemNum == 0){
           dist_min = 0;
           RepulsiveElement = 0;
           for (i = 0; i < grids->maxElements; ++i){
               if (i == 0) dist_min = minCellsElementsDist[i];
               if (minCellsElementsDist[i] < dist_min){
                  dist_min = minCellsElementsDist[i];
                  RepulsiveElement = i;
               }
           }
           int cellNum1, elemNum1, cellNum2, elemNum2;
           cellNum1 = cellNum;
           elemNum1 = RepulsiveElement;
           cellNum2 = j;
           elemNum2 = minCellsElements[RepulsiveElement];
          
           double *row1 = (double*)((char*)X + elemNum1 * pitch);
           double *row2 = (double*)((char*)X + elemNum2 * pitch);
           dx = row1[cellNum1] - row2[cellNum2];
           row1 = (double*)((char*)Y + elemNum1 * pitch);
           row2 = (double*)((char*)Y + elemNum2 * pitch);
           dy = row1[cellNum1] - row2[cellNum2];
           row1 = (double*)((char*)Z + elemNum1 * pitch);
           row2 = (double*)((char*)Z + elemNum2 * pitch);
           dz = row1[cellNum1] - row2[cellNum2];
           dr = norm(dx, dy, dz);
           Force = 0.5*2000*exp(-2000*abs(dr))/(1-exp(-2000*abs(dr)));//unit: nN
           row1 = (double*)((char*)F_X + elemNum1 * pitch);
           row2 = (double*)((char*)F_X + elemNum2 * pitch);
           row1[cellNum1] += Force*dx/dr;
           row2[cellNum2] -= Force*dx/dr;
           row1 = (double*)((char*)F_Y + elemNum1 * pitch);
           row2 = (double*)((char*)F_Y + elemNum2 * pitch);
           row1[cellNum1] += Force*dy/dr;
           row2[cellNum2] -= Force*dy/dr;
           row1 = (double*)((char*)F_Z + elemNum1 * pitch);
           row2 = (double*)((char*)F_Z + elemNum2 * pitch);
           row1[cellNum1] += Force*dz/dr;
           row2[cellNum2] -= Force*dz/dr;
	}  
   }


// End of NON-Bonded Interaction
}

/*__global__ void
sem_surface_tension_kernel(void* g){
   
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (triElemIdx >= grids->SurfElem) return;
   
   int node[3];
   double node_coord[3][3];
   double node_coord_ref[3][3];
   int *triElem = grids->triElem;   
   double *X_Ref = grids->X_Ref;
   double *Y_Ref = grids->Y_Ref;
   double *Z_Ref = grids->Z_Ref;
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *F_X = grids->F_X;
   double *F_Y = grids->F_Y;
   double *F_Z = grids->F_Z;
   double *rho = grids->rho;
   size_t pitch = grids->pitch;
   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i;
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)X_Ref + node[i] * pitch);
        node_coord_ref[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Y_Ref + node[i] * pitch);
        node_coord_ref[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
        row1 = (double*)((char*)Z_Ref + node[i] * pitch);
        node_coord_ref[i][2] = row1[0];
   }
   double e[3][3], E[3][3], e13[3], E13[3], sum_ref1 = 0.0, sum_ref2 = 0.0, sum1 = 0.0, sum2 = 0.0;
   for (i = 0; i < 3; i++){
       e[0][i] = node_coord_ref[1][i] - node_coord_ref[0][i];
       e13[i] = node_coord_ref[2][i] - node_coord_ref[0][i];
       sum_ref1 += e[0][i] * e[0][i];
       sum_ref2 += e13[i] * e13[i]; 
       E[0][i] = node_coord[1][i] - node_coord[0][i];
       E13[i] = node_coord[2][i] - node_coord[0][i];       
       sum1 += E[0][i] * E[0][i];
       sum2 += E13[i] * E13[i];        
   }  
   for (i = 0; i < 3; i++){
       e[0][i] /= sqrt(sum_ref1);
       e13[i] /= sqrt(sum_ref2);
       E[0][i] /= sqrt(sum1);
       E13[i] /= sqrt(sum2);
   }
   e[2][0] = e[0][1] * e13[2] - e[0][2] * e13[1];
   e[2][1] = e[0][2] * e13[0] - e[0][0] * e13[2];
   e[2][2] = e[0][0] * e13[1] - e[0][1] * e13[0];
   e[1][0] = e[2][1] * e[0][2] - e[2][2] * e[0][1];
   e[1][1] = e[2][2] * e[0][0] - e[2][0] * e[0][2];
   e[1][2] = e[2][0] * e[0][1] - e[2][1] * e[0][0];
   E[2][0] = E[0][1] * E13[2] - E[0][2] * E13[1];
   E[2][1] = E[0][2] * E13[0] - E[0][0] * E13[2];
   E[2][2] = E[0][0] * E13[1] - E[0][1] * E13[0];
   E[1][0] = E[2][1] * E[0][2] - E[2][2] * E[0][1];
   E[1][1] = E[2][2] * E[0][0] - E[2][0] * E[0][2];
   E[1][2] = E[2][0] * E[0][1] - E[2][1] * E[0][0];
   
  // int j;
  // if (triElemIdx == 0) {
  //     for (i = 0 ; i < 3; i++){
  //         for (j = 0; j < 3; j++){
  //             printf("e[%d][%d] = %e, E[%d][%d] = %e\n", i, j, e[i][j], i, j, E[i][j]);
  //         }
  //     }
  // }

   double xl[3][3], XL[3][3];
   for (i = 0; i < 3; i++){
       xl[0][i] = 0.0;
       XL[0][i] = 0.0;
       xl[1][i] = e[i][0] * (node_coord_ref[1][0] - node_coord_ref[0][0]) 
                + e[i][1] * (node_coord_ref[1][1] - node_coord_ref[0][1])
                + e[i][2] * (node_coord_ref[1][2] - node_coord_ref[0][2]); 
       XL[1][i] = E[i][0] * (node_coord[1][0] - node_coord[0][0]) 
                + E[i][1] * (node_coord[1][1] - node_coord[0][1])
                + E[i][2] * (node_coord[1][2] - node_coord[0][2]); 
       xl[2][i] = e[i][0] * (node_coord_ref[2][0] - node_coord_ref[0][0]) 
                + e[i][1] * (node_coord_ref[2][1] - node_coord_ref[0][1])
                + e[i][2] * (node_coord_ref[2][2] - node_coord_ref[0][2]); 
       XL[2][i] = E[i][0] * (node_coord[2][0] - node_coord[0][0]) 
                + E[i][1] * (node_coord[2][1] - node_coord[0][1])
                + E[i][2] * (node_coord[2][2] - node_coord[0][2]);       
   }
   
   double ul[3], vl[3], a[3], b[3], c[3], L[3], A[3], B[3];
   for (i = 0; i < 3; i++){ 
       ul[i]= XL[i][0] - xl[i][0];
       vl[i]= XL[i][1] - xl[i][1];
      // if (triElemIdx == 0 && i == 2) printf(" XL = %e, xl = %e\n", XL[i][0], xl[i][0]);       
   }
   a[0] = xl[1][1] - xl[2][1];
   b[0] = xl[2][0] - xl[1][0];
   c[0] = xl[1][0] * xl[2][1] - xl[2][0] * xl[1][1]; 
   a[1] = xl[2][1] - xl[0][1];
   b[1] = xl[0][0] - xl[2][0];
   c[1] = xl[2][0] * xl[0][1] - xl[0][0] * xl[2][1];
   a[2] = xl[0][1] - xl[1][1];
   b[2] = xl[1][0] - xl[0][0];
   c[2] = xl[0][0] * xl[1][1] - xl[1][0] * xl[0][1];
   for (i = 0; i < 3; i++){
       L[i] = a[i] * xl[i][0] + b[i] * xl[i][1] + c[i];
       A[i] = a[i]/L[i];
       B[i] = b[i]/L[i];
   }
   
   double G11, G22, G12, uA, uB, vA, vB;
   uA = ul[0] * A[0] + ul[1] * A[1] + ul[2] * A[2];
   uB = ul[0] * B[0] + ul[1] * B[1] + ul[2] * B[2];
   vA = vl[0] * A[0] + vl[1] * A[1] + vl[2] * A[2];
   vB = vl[0] * B[0] + vl[1] * B[1] + vl[2] * B[2];
   G11 = 1 + 2 * uA + uA * uA + vA * vA;
   G22 = 1 + 2 * vB + vB * vB + uB * uB;
   G12 = uB + uB * uA + vA + vB * vA;
  // if (triElemIdx == 0) printf("G11 = %e, G22 = %e, G12 = %e\n", G11, G22, G12);
   double Fx[3], Fy[3], Fz[3];
   if ((G11 == G22) && G12 == 0){
       for (i = 0; i < 3; i++){
           Fx[i] = 0;
           Fy[i] = 0;
           Fz[i] = 0;
           // if (triElemIdx == 0) printf("Fx[%d] = %e\n", i, FxL[i]);
    //       row1 = (double*)((char*)F_X + node[i] * pitch);
    //       atomicAdd(&(row1[0]), Fx[i]/F0);
    //       row1 = (double*)((char*)F_Y + node[i] * pitch);
    //       atomicAdd(&(row1[0]), Fy[i]/F0);
    //       row1 = (double*)((char*)F_Z + node[i] * pitch);
    //       atomicAdd(&(row1[0]), Fz[i]/F0);
       }
   }
   else{
       double lambda[3], FxL[3], FyL[3], FzL[3], lambda1U[3], lambda2U[3], lambda1V[3], lambda2V[3];
       lambda[0] = sqrt(0.5 * (G11 + G22 + sqrt((G11 - G22) * (G11 - G22) + 4 * G12 * G12)));
       lambda[1] = sqrt(0.5 * (G11 + G22 - sqrt((G11 - G22) * (G11 - G22) + 4 * G12 * G12)));
   //if (triElemIdx == 0) printf("lambda1 = %e, lambda2 = %e\n", lambda[0], lambda[1]);
       double L1, L2, L3, Pe, Ae, Ve;
       L1 = sqrt((xl[0][0] - xl[1][0]) * (xl[0][0] - xl[1][0]) + (xl[0][1] - xl[1][1]) * (xl[0][1] - xl[1][1]) 
          + (xl[0][2] - xl[1][2]) * (xl[0][2] - xl[1][2]));
       L2 = sqrt((xl[1][0] - xl[2][0]) * (xl[1][0] - xl[2][0])  + (xl[1][1] - xl[2][1]) * (xl[1][1] - xl[2][1]) 
          + (xl[1][2] - xl[2][2]) * (xl[1][2] - xl[2][2]));
       L3 = sqrt((xl[2][0] - xl[0][0]) * (xl[2][0] - xl[0][0]) + (xl[2][1] - xl[0][1]) * (xl[2][1] - xl[0][1]) 
          + (xl[2][2] - xl[0][2]) * (xl[2][2] - xl[0][2]));
       Pe = 0.5 * (L1 + L2 + L3);
       Ae = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));
       Ve = 0.5 * Ae;
     
       double rho_node[3], rho_mean = 0;
       for (i = 0; i < 3; i++){
           row1 = (double*)((char*)rho + node[i] * pitch);
           rho_node[i] = row1[0];
           rho_mean += rho_node[i];
       }
       rho_mean /= 3.0;
       double P = rho_mean * CL * CL/3.0;

       for (i = 0; i < 3; i++){
           lambda1U[i] = 0.25/lambda[0] * (2 * A[i] + 2 * A[i] * uA + 2 * B[i] * uB + 1/sqrt((G11 - G22) * (G11 - G22) 
                       + 4 * G12 * G12) * ((G11 - G22)*(2 * A[i] + 2 * A[i] * uA -  2 * B[i] * uB) + 4 * G12 * (B[i] 
                       + B[i] * uA + A[i] * uB)));
           lambda2U[i] = 0.25/lambda[1] * (2 * A[i] + 2 * A[i] * uA + 2 * B[i] * uB - 1/sqrt((G11 - G22) * (G11 - G22) 
                       + 4 * G12 * G12) * ((G11 - G22)*(2 * A[i] + 2 * A[i] * uA -  2 * B[i] * uB) + 4 * G12 * (B[i]
                       + B[i] * uA + A[i] * uB)));
           lambda1V[i] = 0.25/lambda[0] * (2 * A[i] * vA + 2 * B[i] + 2 * B[i] * vB + 1/sqrt((G11 - G22) * (G11 - G22) 
                       + 4 * G12 * G12) * ((G11 - G22)*(2 * A[i] * vA - 2 * B[i] -  2 * B[i] * vB) + 4 * G12 * (A[i]
                       + B[i] * vA + A[i] * vB)));
           lambda2V[i] = 0.25/lambda[1] * (2 * A[i] * vA + 2 * B[i] + 2 * B[i] * vB - 1/sqrt((G11 - G22) * (G11 - G22) 
                       + 4 * G12 * G12) * ((G11 - G22)*(2 * A[i] * vA - 2 * B[i] -  2 * B[i] * vB) + 4 * G12 * (A[i]
                       + B[i] * vA + A[i] * vB)));
       //if (triElemIdx == 0) printf("lambda1U[%d] = %e\n", i, lambda1U[i]);
           FxL[i] = Ve * (Em/3.0 * (lambda[0] - 1.0/ (lambda[1] * lambda[1])/ (lambda[0] * lambda[0] * lambda[0])) * lambda1U[i] +
                    Em/3.0 * (lambda[1] - 1.0/ (lambda[0] * lambda[0])/ (lambda[1] * lambda[1] * lambda[1])) * lambda2U[i]);
           FyL[i] = Ve * (Em/3.0 * (lambda[0] - 1.0/ (lambda[1] * lambda[1])/ (lambda[0] * lambda[0] * lambda[0])) * lambda1V[i] +
                    Em/3.0 * (lambda[1] - 1.0/ (lambda[0] * lambda[0])/ (lambda[1] * lambda[1] * lambda[1])) * lambda2V[i]);
           FzL[i] = - P0 * P * Ae/3.0;
          // printf("FzL = %e\n", FzL[i]);
           
        }
       
       for (i = 0; i < 3; i++){
           Fx[i] = E[0][0] * FxL[i] + E[1][0] * FyL[i] + E[2][0] * FzL[i];
           Fy[i] = E[0][1] * FxL[i] + E[1][1] * FyL[i] + E[2][1] * FzL[i];
           Fz[i] = E[0][2] * FxL[i] + E[1][2] * FyL[i] + E[2][2] * FzL[i];
          // node[i] = row[i];
           // if (triElemIdx == 0) printf("Fx[%d] = %e\n", i, FxL[i]);
           row1 = (double*)((char*)F_X + node[i] * pitch);
           atomicAdd(&(row1[0]), Fx[i]/F0);
           row1 = (double*)((char*)F_Y + node[i] * pitch);
           atomicAdd(&(row1[0]), Fy[i]/F0);
           row1 = (double*)((char*)F_Z + node[i] * pitch);
           atomicAdd(&(row1[0]), Fz[i]/F0);
       }
   }      
}*/

__global__ void
sem_calculate_A_kernel(void* g){
   
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (triElemIdx >= grids->SurfElem) return;
   //if (triElemIdx == 0) printf("S = %e\n", grids->S_all); 
   int node[6];
   double node_coord[6][3], node_coord_ref[6][3];
   int *triElem = grids->triElem;   
   double *X_Ref = grids->X_Ref;
   double *Y_Ref = grids->Y_Ref;
   double *Z_Ref = grids->Z_Ref;
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
  // double *N = grids->N;
   double *n = grids->n;
   double *nelem = grids->nelem;
   double *S = grids->S;
   //double *A = grids->A;
   size_t pitch = grids->pitch;
   
   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i, j;// k, l;
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        node[i+3] = row[i+3];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)X_Ref + node[i] * pitch);
        node_coord_ref[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Y_Ref + node[i] * pitch);
        node_coord_ref[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
        row1 = (double*)((char*)Z_Ref + node[i] * pitch);
        node_coord_ref[i][2] = row1[0];
   }
  // if (triElemIdx == 1993)
  //    printf("%d %d %d %d %d %d\n", node[0], node[1], node[2], node[3], node[4], node[5]);
   double coord_center[3];
   for (i = 0; i < 3; i++){
       node_coord_ref[3][i] = 0.5 * (node_coord_ref[0][i] + node_coord_ref[1][i]);
       node_coord_ref[4][i] = 0.5 * (node_coord_ref[0][i] + node_coord_ref[2][i]);
       node_coord_ref[5][i] = 0.5 * (node_coord_ref[1][i] + node_coord_ref[2][i]);
       node_coord[3][i] = 0.5 * (node_coord[0][i] + node_coord[1][i]);
       node_coord[4][i] = 0.5 * (node_coord[0][i] + node_coord[2][i]);
       node_coord[5][i] = 0.5 * (node_coord[1][i] + node_coord[2][i]);
       coord_center[i] = 1/3.0 * (node_coord[0][i] + node_coord[1][i] + node_coord[2][i]);
   }
   // calculate normal vector n
   double n_elem[3], e12[3], e13[3], norm_n;
   for (i = 0; i < 3; i++){
       e12[i] = node_coord[1][i] - node_coord[0][i];
       e13[i] = node_coord[2][i] - node_coord[0][i];
   }
  // double L1 = norm(e12[0],e12[1],e12[2]);
  // double L2 = norm(e13[0],e13[1],e13[2]);
  // double L3 = norm(e23[0],e23[1],e23[2]);
  // double Pe = 0.5 * (L1 + L2 + L3);
  // double Se = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));
  // S[triElemIdx] = Se; 
  // atomicAdd(&(grids->S_all), Se);
   
   n_elem[0] = e12[1] * e13[2] - e12[2] * e13[1];
   n_elem[1] = e12[2] * e13[0] - e12[0] * e13[2];
   n_elem[2] = e12[0] * e13[1] - e12[1] * e13[0];
   norm_n = norm(n_elem[0], n_elem[1], n_elem[2]);
   //if (triElemIdx == 0) printf("%e %e %e\n", n_elem[0], n_elem[1], n_elem[2]);
   n_elem[0] /= -norm_n;    
   n_elem[1] /= -norm_n;    
   n_elem[2] /= -norm_n;
   
   S[triElemIdx] = 0.5 * norm_n; 
   atomicAdd(&(grids->S_all), S[triElemIdx]);
   double V = 1/3.0 * (coord_center[0] * n_elem[0] + coord_center[1] * n_elem[1]
                    + coord_center[2] * n_elem[2]) * S[triElemIdx];
   atomicAdd(&(grids->V), V);   
   
   row1 = (double*)((char*)nelem + triElemIdx * pitch);   
   for (i = 0; i < 3; i++){
       row1[i] = n_elem[i];
   }    
   for (i = 0; i < 6; i++){
       row1 = (double*)((char*)n + node[i] * pitch);
       for (j = 0; j < 3; j++){
         // row1[j] = n_elem[j];
         atomicAdd(&(row1[j]), n_elem[j]);           
       }
   }
   
   /*double A_elem[3][3], A1[3], A2[3], A3[3], A4[3], A5[3], A6[3];
   
   for (i = 0; i < 6; i++){
       for(j = 0; j < 3; j++){
           A1[j] = node_coord_ref[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                 + node_coord_ref[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                 + node_coord_ref[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                 + node_coord_ref[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                 + node_coord_ref[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                 + node_coord_ref[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
           A2[j] = node_coord[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                 + node_coord[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                 + node_coord[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                 + node_coord[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                 + node_coord[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                 + node_coord[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
    //       if (triElemIdx == 0){
      //       printf("A1[%d] = %f, A2[%d] = %f\n", j, A1[j],j,A2[j]); 
        //   }
           A3[j] = node_coord_ref[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                 + node_coord_ref[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                 + node_coord_ref[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                 + node_coord_ref[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                 + node_coord_ref[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                 + node_coord_ref[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
           A4[j] = node_coord[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                 + node_coord[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                 + node_coord[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                 + node_coord[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                 + node_coord[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                 + node_coord[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
          // if (triElemIdx == 0){
            // printf("A3[%d] = %f, A4[%d] = %f\n", j, A3[j],j,A4[j]); 
          // }
           row1 = (double*)((char*)N + node[i] * pitch);
           A5[j] = row1[j];
           A6[j] = 0;             
           //if (triElemIdx == 0){
          //     printf("A5[%d] = %e, A6[%d] = %f\n", j, A5[j],j,A6[j]); 
          // }
       }
       solve_linear_eqns(A_elem, A1, A2, A3, A4, A5, A6);
       row1 = (double*)((char*)A + node[i] * pitch);
       for (k = 0; k < 3; k++){
           for(l = 0; l < 3; l++){
          //  if(triElemIdx == 0) printf("A_elem[%d][%d] = %f\n", k, l, A_elem[k][l]);
              atomicAdd(&(row1[k * 3 + l]), A_elem[k][l]);
           }
       }                    
   }*/      
}

__global__ void
sem_surface_tension_kernel(void* g){
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (nodeIdx >= grids->newnodeNum) return;
   //double* A = grids->A;
   double* n = grids->n;
   int* node_nbr = grids->node_nbrElemNum;
  // double *tau = grids->tau;
   size_t pitch = grids->pitch;
   int i; 
   double *row = (double*)((char*)n + nodeIdx * pitch);
  // double n_elem[3];
   //if (nodeIdx == 4670) printf("numElem = %d\n", node_nbr[nodeIdx]);
   for (i = 0; i < 3; i++){
       row[i] /= node_nbr[nodeIdx];
   }
   double norm_n = norm(row[0],row[1],row[2]);
   for (i = 0; i < 3; i++){
       row[i] /= norm_n;
      // n_elem[i] = row[i];
      // if (nodeIdx == 0) 
      //    printf("n_elem[%d] = %e\n", i, n_elem[i]);
   }
  // if (nodeIdx < 14)
  //     printf("%d  %e %e %e %e\n",nodeIdx, n_elem[0], n_elem[1], n_elem[2], norm(n_elem[0], n_elem[1], n_elem[2]));
   /*row = (double*)((char*)A + nodeIdx * pitch);
   double A_elem[3][3];   
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
          // if (nodeIdx == 0) printf("%d,A[%d][%d] = %f\n", node_nbr[nodeIdx], i, j, row[i*3 + j]);
           row[i*3 + j] /= node_nbr[nodeIdx];
           A_elem[i][j] = row[i*3 +j];
          // if (nodeIdx == 0) printf("A[%d][%d] = %f\n", A_elem[i][j]);
           
       }
   }  
   double nn[3][3], P[3][3], B[3][3] = {0}, B2[3][3] = {0};
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           nn[i][j] = n_elem[i] * n_elem[j];
           P[i][j] = I[i][j] - nn[i][j];
           for (k = 0; k < 3; k++){
               B[i][j] += A_elem[i][k] * A_elem[j][k];
           }
          // if (nodeIdx == 0){ 
          //     printf("A[%d][%d] = %e\n", i,j,A_elem[i][j]);
            //   printf("P[%d][%d] = %e\n", i,j,P[i][j]);
          // }            
       }
   }
   for (i = 0; i < 3; i++){
       for(j = 0; j < 3; j++){
          for (k = 0; k < 3; k++){
              B2[i][j] += B[i][k] * B[j][k];
          }          
       }
   }
   double Lambda1;
   Lambda1 = 0.5 * log(0.5 * (trace(B) * trace(B) - trace(B2)));
  // if (nodeIdx == 0) 
   //   printf("Lambda1 = %e\n", Lambda1);
   //Lambda2 = 0.5 * trace(B) - 1;
   row = (double*)((char*)tau + nodeIdx * pitch);
   for (i = 0; i < 3; i++){
       for(j = 0; j < 3; j++){
          row[i * 3 + j] = exp(-Lambda1) * Em * 0.5 / 3.0 * (-exp(-2 * Lambda1) * P[i][j] + B[i][j]);
          //if (nodeIdx == 0) printf("tau[%d][%d] = %e\n", i,j,row[i*3+j]);
       }
   }*/
      
}

__global__ void
sem_calculate_Kapa_kernel(void* g){
   
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (triElemIdx >= grids->SurfElem) return;
   
   int node[6];
   double node_coord[6][3];
   int *triElem = grids->triElem;   
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *Kapa = grids->Kapa;
   double *n = grids->n;
   size_t pitch = grids->pitch;
   
   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i, j, k, l;
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        node[i+3] = row[i+3];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
   }
   for (i = 0; i < 3; i++){
       node_coord[3][i] = 0.5 * (node_coord[0][i] + node_coord[1][i]);
       node_coord[4][i] = 0.5 * (node_coord[0][i] + node_coord[2][i]);
       node_coord[5][i] = 0.5 * (node_coord[1][i] + node_coord[2][i]); 
   }
   // calculate normal vector n
   double n_elem[6][3];       
   for (i = 0; i < 6; i++){
       row1 = (double*)((char*)n + node[i] * pitch);
       for (j = 0; j < 3; j++){
           n_elem[i][j] = row1[j];
       }
     // if (triElemIdx == 0)
     //     printf("%d %e %e %e\n", node[i],n_elem[i][0], n_elem[i][1], n_elem[i][2]);           
       
   }
   
   double Kapa_elem[3][3], A1[3], A2[3], A3[3], A4[3], A5[3], A6[3];
   
   for (i = 0; i < 6; i++){
       for(j = 0; j < 3; j++){
           A1[j] = node_coord[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                 + node_coord[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                 + node_coord[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                 + node_coord[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                 + node_coord[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                 + node_coord[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
           A2[j] = n_elem[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                 + n_elem[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                 + n_elem[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                 + n_elem[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                 + n_elem[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                 + n_elem[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
           A3[j] = node_coord[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                 + node_coord[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                 + node_coord[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                 + node_coord[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                 + node_coord[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                 + node_coord[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
           A4[j] = n_elem[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                 + n_elem[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                 + n_elem[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                 + n_elem[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                 + n_elem[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                 + n_elem[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
           row1 = (double*)((char*)n + node[i] * pitch);
           A5[j] = row1[j];
           A6[j] = 0; 
          // if (triElemIdx == 0 && i == 0){
          //    printf("%e %e %e %e %e %e\n",A1[j],A3[j],A5[j],A2[j],A4[j],A6[j]);
          // }            
       }
       solve_linear_eqns(Kapa_elem, A1, A2, A3, A4, A5, A6);
      /* if (triElemIdx == 1993){
          double km = 0.5 * trace(Kapa_elem);
          if (abs(km - 1) > 0.1){
             printf("elemId = %d, node = %d, km = %e\n",triElemIdx, i, km);
          }
       }*/
       row1 = (double*)((char*)Kapa + node[i] * pitch);
       for (k = 0; k < 3; k++){
           for(l = 0; l < 3; l++){
              atomicAdd(&(row1[k * 3 + l]),Kapa_elem[l][k]);
           }
          // if (triElemIdx > 7 && triElemIdx < 12){
          //    printf("%d %d %e %e %e\n", triElemIdx, node[i], Kapa_elem[0][k], Kapa_elem[1][k], Kapa_elem[2][k]);
          // }
       }
       //if (triElemIdx > 7 && triElemIdx < 12){
       //       printf("%d %d \n", triElemIdx, node[i]);
       //}                           
   }      
}

__global__ void
sem_calculate_m_kernel(void* g){
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (nodeIdx >= grids->newnodeNum) return;
  // double* n = grids->n;
   double* Kapa = grids->Kapa;
   double* km = grids->km;
   int* node_nbr = grids->node_nbrElemNum;
   size_t pitch = grids->pitch;
   int i,j; 
   double *row;
   /*double *row = (double*)((char*)n + nodeIdx * pitch);
   double n_elem[3];
   for (i = 0; i < 3; i++){
       n_elem[i] = row[i];
   }*/
   row = (double*)((char*)Kapa + nodeIdx * pitch);
   double Kapa_elem[3][3];
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           row[i*3 + j] /= node_nbr[nodeIdx];
           Kapa_elem[i][j] = row[i*3 +j];
       }
     //  if (nodeIdx == 6) 
     //      printf("%d %e %e %e\n", nodeIdx, Kapa_elem[i][0], Kapa_elem[i][1], Kapa_elem[i][2]);
   }
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           row[i*3 + j] = 0.5 * (Kapa_elem[i][j] + Kapa_elem[j][i]);
       }
   }
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           Kapa_elem[i][j] = row[i*3 + j];
       }
       //if (nodeIdx == 34) 
       //    printf("%d %e %e %e\n", nodeIdx, Kapa_elem[i][0], Kapa_elem[i][1], Kapa_elem[i][2]);
   }
   row = (double*)((char*)km + nodeIdx * pitch);
   //printf("km = %e\n", row[0]);  
  // if (row[0] == 0){
   row[0] = 0.5 * trace(Kapa_elem);   
     // if (km - 1 > 0.1)
     // printf("nodeId = %d, km = %e\n",nodeIdx,km);
     printf("nodeId = %d, km = %e\n",nodeIdx,row[0]);
  // }
   //double kmR = row[0];
   /*double nn[3][3], P[3][3];
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           nn[i][j] = n_elem[i] * n_elem[j];
           P[i][j] = I[i][j] - nn[i][j];            
       }
   }*/
  /* row = (double*)((char*)Kapa + nodeIdx * pitch);
   for (i = 0; i < 3; i++){
       for(j = 0; j < 3; j++){
         // if (nodeIdx == 1) printf("%e  %e\n",Kapa_elem[i][j],P[i][j]);
          row[i*3 + j] = Em * 0.5 * 0.5 * 0.5 / (12 * (1 - 0.5 * 0.5))
                         * (row[i*3 + j]);// - kmR * P[i][j]);                         
         // row[i*3 + j] = Em * 0.5 * 0.5 * 0.5 / (12 * (1 - 0.5 * 0.5))
                        // * (trace(Kapa_elem) - kmR) * P[i][j];                         
       }
       //if (nodeIdx == 6) 
       //    printf("%d %e %e %e\n", nodeIdx, row[i*3], row[i*3+1], row[i*3+2]);       
   }*/
   
}
__global__ void
sem_calculate_K_kernel(void* g){
   
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (triElemIdx >= grids->SurfElem) return;
   
   int node[6];
   double node_coord[6][3];
   int *triElem = grids->triElem;   
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *Kapa = grids->Kapa;
   double *n = grids->n;
   //double *nelem = grids->nelem;
   double *K = grids->K;
   size_t pitch = grids->pitch;
   
   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i, j, k, l;
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        node[i+3] = row[i+3];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
   }
   for (i = 0; i < 3; i++){
       node_coord[3][i] = 0.5 * (node_coord[0][i] + node_coord[1][i]);
       node_coord[4][i] = 0.5 * (node_coord[0][i] + node_coord[2][i]);
       node_coord[5][i] = 0.5 * (node_coord[1][i] + node_coord[2][i]); 
   }
   // calculate normal vector n
   /*double n_elem[6][3];       
   for (i = 0; i < 6; i++){
       row1 = (double*)((char*)n + node[i] * pitch);
       for (j = 0; j < 3; j++){
           n_elem[i][j] = row1[j];           
       }
   }*/
   double Kapa_elem[6][3]; 
   
   double K_elem[3][3], A1[3], A2[3], A3[3], A4[3], A5[3], A6[3];
   
   for (i = 0; i < 6; i++){
       for (l = 0; l < 3; l++){
           for (j = 0; j < 6; j++){
               row1 = (double*)((char*)Kapa + node[j] * pitch);
               for (k = 0; k < 3; k++){
                   Kapa_elem[j][k] = row1[k * 3 + l]; 
               }
           }
           
           for (j = 0; j < 3; j++){
               A1[j] = node_coord[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                     + node_coord[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                     + node_coord[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                     + node_coord[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                     + node_coord[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                     + node_coord[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
               A2[j] = Kapa_elem[0][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[0][0], Ni[0][2], Ni[0][3])
                     + Kapa_elem[1][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[1][0], Ni[1][2], Ni[1][3])
                     + Kapa_elem[2][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[2][0], Ni[2][2], Ni[2][3])
                     + Kapa_elem[3][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[3][0], Ni[3][2], Ni[3][3]) 
                     + Kapa_elem[4][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[4][0], Ni[4][2], Ni[4][3]) 
                     + Kapa_elem[5][j] * dN_dxi(xi_eta[i][0], xi_eta[i][1], Ni[5][0], Ni[5][2], Ni[5][3]);
               A3[j] = node_coord[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                     + node_coord[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                     + node_coord[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                     + node_coord[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                     + node_coord[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                     + node_coord[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
               A4[j] = Kapa_elem[0][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[0][1], Ni[0][2], Ni[0][4])
                     + Kapa_elem[1][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[1][1], Ni[1][2], Ni[1][4])
                     + Kapa_elem[2][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[2][1], Ni[2][2], Ni[2][4])
                     + Kapa_elem[3][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[3][1], Ni[3][2], Ni[3][4]) 
                     + Kapa_elem[4][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[4][1], Ni[4][2], Ni[4][4]) 
                     + Kapa_elem[5][j] * dN_deta(xi_eta[i][0], xi_eta[i][1], Ni[5][1], Ni[5][2], Ni[5][4]);
                row1 = (double*)((char*)n + node[i] * pitch);
                A5[j] = row1[j];
                A6[j] = 0;             
           }
          /* if (node[i] == 0 && triElemIdx == 0 && l == 0){
              printf("%e %e %e\n", A1[0], A1[1], A1[2]);
              printf("%e %e %e\n", A3[0], A3[1], A3[2]);
              printf("%e %e %e\n", A5[0], A5[1], A5[2]);
              printf("%e %e %e\n", A2[0], A2[1], A2[2]);
              printf("%e %e %e\n", A4[0], A4[1], A4[2]);
              printf("%e %e %e\n", A6[0], A6[1], A6[2]);
           }*/
           solve_linear_eqns(K_elem, A1, A2, A3, A4, A5, A6);
           /*if (node[i] == 0 && triElemIdx == 0 && l == 0){
              for (j = 0; j < 3; j++){
                  printf("%e %e %e\n", K_elem[0][j], K_elem[1][j], K_elem[2][j]);
              }
           }*/
           row1 = (double*)((char*)K + node[i] * pitch);
           for (k = 0; k < 3; k++){
               for (j = 0; j < 3; j++){
                   atomicAdd(&(row1[j * 9 + k * 3 + l]),K_elem[k][j]);
               }
           }
           /*for (j = 0; j < 3; j++){    
              if (node[i] == 0 && triElemIdx == 1)
                  printf("%e %e %e\n", K_elem[0][j], K_elem[1][j], K_elem[2][j]);               
           }*/              
        }                    
  }      
}

__global__ void
sem_calculate_q_kernel(void* g){
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (nodeIdx >= grids->newnodeNum) return;
   double* n = grids->n;
   double* K = grids->K;
   double* q = grids->q;
   int* node_nbr = grids->node_nbrElemNum;
   size_t pitch = grids->pitch;
   int i,j,k; 
   
   double *row = (double*)((char*)n + nodeIdx * pitch);
   double n_elem[3];
   for (i = 0; i < 3; i++){
       n_elem[i] = row[i];
   }
   double nn[3][3],P[3][3];
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           nn[i][j] = n_elem[i] * n_elem[j];
           P[i][j] = I[i][j] - nn[i][j];         
       }
   }
   row = (double*)((char*)K + nodeIdx * pitch);
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           for (k = 0; k < 3; k++){
              row[i*9 + j*3 + k] /= node_nbr[nodeIdx];
           }
       }
   }
   /*for (k = 0; k < 3; k++){
       for (i = 0; i < 3; i++){
           if (nodeIdx == 2)
              printf("%d %e %e %e\n", nodeIdx, row[i*9 + k], row[i*9 + 3 + k], row[i*9 + 6 + k]);
       }
   }*/
   double K_elem[3][3], traceK[3];
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           for (k = 0; k < 3; k++){
              K_elem[j][k] = row[j*9 + k*3 + i];
           }
       }
       traceK[i] = trace(K_elem); 
   }
   row = (double*)((char*)q + nodeIdx * pitch);   
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           row[j] += traceK[i] * P[i][j];
       }
   }
   //if (nodeIdx > 13) 
   //    printf("%d %e %e %e %e\n", nodeIdx, row[0], row[1], row[2], norm(row[0], row[1], row[2])); 
      // printf("%d %e %e %e %e\n", nodeIdx, traceK[0], traceK[1], traceK[2], norm(traceK[0], traceK[1], traceK[2])); 
}

__global__ void
sem_bending_force_kernel(void* g){
   
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
  // int cellNum = blockIdx.x;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
  // if (cellNum >= grids->maxCells) return;
   if (triElemIdx >= grids->SurfElem) return;
   
   int node[6];
   double node_coord[6][3];
   int *triElem = grids->triElem;   
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *F_X = grids->F_X;
   double *F_Y = grids->F_Y;
   double *F_Z = grids->F_Z;
   //double *tau = grids->tau;
   double *n = grids->n;
  // double *nelem = grids->nelem;
   double *S = grids->S;
   double *q = grids->q;
   size_t pitch = grids->pitch;
   
   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i, j;
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        node[i+3] = row[i+3];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
   }
   for (i = 0; i < 3; i++){
       node_coord[3][i] = 0.5 * (node_coord[0][i] + node_coord[1][i]);
       node_coord[4][i] = 0.5 * (node_coord[0][i] + node_coord[2][i]);
       node_coord[5][i] = 0.5 * (node_coord[1][i] + node_coord[2][i]); 
   }
   // calculate normal vector n
   double n_elem[6][3];       
   for (i = 0; i < 6; i++){
       row1 = (double*)((char*)n + node[i] * pitch);
      // row1 = (double*)((char*)nelem + triElemIdx * pitch);
       for (j = 0; j < 3; j++){
           n_elem[i][j] = row1[j];           
       }
   } 
   double b[6][6] = {0}, f[6][6] = {0}, t[6][6], norm_t[3]; 
   for (i = 0; i < 3; i++){
       t[0][i] = (node_coord[0][i] - node_coord[1][i]);
       t[1][i] = (node_coord[1][i] - node_coord[2][i]);
       t[2][i] = (node_coord[2][i] - node_coord[0][i]);
       t[3][i] = t[0][i];
       t[4][i] = t[2][i];
       t[5][i] = t[1][i];         
   }  
   for (i = 0; i < 3; i++){
       norm_t[i] = norm(t[i][0], t[i][1], t[i][2]);
       t[i][0] /= norm_t[i];
       t[i][1] /= norm_t[i];
       t[i][2] /= norm_t[i];
   }
   for (i = 0; i < 3; i++){
       t[3][i] /= norm_t[0];
       t[4][i] /= norm_t[2];
       t[5][i] /= norm_t[1];
   }
   /*if (triElemIdx == 0){
      for (i = 0; i < 6; i++){
          for (j = 0; j < 3; j++)
             printf("n[%d][%d] = %f\n",i,j,n_elem[i][j]);
      }
   } */
   double norm_b[6][2]; 
   for (i = 0; i < 6; i++){
       b[i][0] = t[i][1] * n_elem[i][2] - t[i][2] * n_elem[i][1];
       b[i][1] = t[i][2] * n_elem[i][0] - t[i][0] * n_elem[i][2];
       b[i][2] = t[i][0] * n_elem[i][1] - t[i][1] * n_elem[i][0];
       norm_b[i][0] = norm(b[i][0], b[i][1], b[i][2]);
       b[i][0] /= norm_b[i][0];
       b[i][1] /= norm_b[i][0];
       b[i][2] /= norm_b[i][0];       
   }   
   b[0][3] = t[2][1] * n_elem[0][2] - t[2][2] * n_elem[0][1];
   b[0][4] = t[2][2] * n_elem[0][0] - t[2][0] * n_elem[0][2];
   b[0][5] = t[2][0] * n_elem[0][1] - t[2][1] * n_elem[0][0];
   b[1][3] = t[0][1] * n_elem[1][2] - t[0][2] * n_elem[1][1];
   b[1][4] = t[0][2] * n_elem[1][0] - t[0][0] * n_elem[1][2];
   b[1][5] = t[0][0] * n_elem[1][1] - t[0][1] * n_elem[1][0];
   b[2][3] = t[1][1] * n_elem[2][2] - t[1][2] * n_elem[2][1];
   b[2][4] = t[1][2] * n_elem[2][0] - t[1][0] * n_elem[2][2];
   b[2][5] = t[1][0] * n_elem[2][1] - t[1][1] * n_elem[2][0]; 
   for (i = 0; i < 3; i++){
       norm_b[i][1] = norm(b[i][3], b[i][4], b[i][5]);
       b[i][3] /= norm_b[i][1];
       b[i][4] /= norm_b[i][1];
       b[i][5] /= norm_b[i][1];               
   }
      
   /*for (i = 0; i < 6; i++){
       row1 = (double*)((char*)tau + node[i] * pitch);       
       for (j = 0; j < 3; j++){
           for (k = 0; k < 3; k++){
               f[i][j] += b[i][k] * row1[k * 3 + j];
               if (i < 3){
                  f[i][j+3] += b[i][k+3] * row1[k * 3 + j];
               } 
           }    
       //if (triElemIdx == 0) printf("f[%d] = %e\n", j, f[i][j]);
       }
   }*/
   double L1 = norm_t[0];
   double L2 = norm_t[1];
   double L3 = norm_t[2];
   double Pe = 0.5 * (L1 + L2 + L3);
   double Ae = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));
   for (i = 0; i < 6; i++){
       row1 = (double*)((char*)q + node[i] * pitch);       
       for (j = 0; j < 3; j++){
               f[i][j] += (b[i][0] * row1[0] + b[i][1] * row1[1]
                           + b[i][2] * row1[2]) * n_elem[i][j];
               if (i < 3){
                  f[i][j+3] += (b[i][3] * row1[0] + b[i][4] * row1[1]
                            + b[i][5] * row1[2]) * n_elem[i][j];                  
               } 
           } 
        if(triElemIdx == 0) {
          //  printf("%d n_elem[%d] = %e %e %e\n", node[i], i, n_elem[i][0],n_elem[i][1],n_elem[i][2]);  
          //printf("%d b = %e %e %e %e %e %e\n",node[i], b[i][0],b[i][1],b[i][2],b[i][3],b[i][4],b[i][5]);
         //printf("%d q = %e %e %e\n", node[i], row1[0],row1[1],row1[2]);                  
         //printf("%d f = %e %e %e %e %e %e, A = %e\n",node[i],f[i][0], f[i][1], f[i][2], f[i][3], f[i][4], f[i][5],Ae);
        }   
   }
   
   /*double L1 = norm_t[0];
   double L2 = norm_t[1];
   double L3 = norm_t[2];
   double Pe = 0.5 * (L1 + L2 + L3);
   double Ae = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));*/
   double F[3], Fx[3], Fy[3], Fz[3];
   //double F1[3], F2[3], F3[3];
   for (i = 0; i < 3; i++){
       F[i] = 0.5 * ((1/3.0 *(f[0][i] + f[1][i+3]) + 4.0/3.0 * f[3][i]) * L1
              + (1/3.0 * (f[1][i] + f[2][i+3]) + 4.0/3.0 * f[5][i]) * L2
              + (1/3.0 * (f[2][i] + f[0][i+3]) + 4.0/3.0 * f[4][i]) * L3);
     //  F1[i] = 0.5 * (1/3.0 *(f[0][i] + f[1][i+3]) + 4.0/3.0 * f[3][i]) * L1;
     //  F2[i] = 0.5 * (1/3.0 * (f[1][i] + f[2][i+3]) + 4.0/3.0 * f[5][i]) * L2;
     //  F3[i] = 0.5 * (1/3.0 * (f[2][i] + f[0][i+3]) + 4.0/3.0 * f[4][i]) * L3;
     /*F[i] = (-L1 * ((1/3.0 * Ni[0][0] + 0.5 * Ni[0][3] + Ni[0][5]) * f[0][i]
              + (1/3.0 * Ni[1][0] + 0.5 * Ni[1][3] + Ni[1][5]) * f[1][i+3]
              + (1/3.0 * Ni[3][0] + 0.5 * Ni[3][3] + Ni[3][5]) * f[3][i])
            + L2 * (f[1][i] * (1/3.0 * (Ni[1][0] + Ni[1][1] - Ni[1][2]) + 
              0.5 * (- 2 * Ni[1][1] + Ni[1][2] + Ni[1][3] - Ni[1][4]) 
              + Ni[1][1] + Ni[1][4] + Ni[1][5]) 
              + f[2][i] * (1/3.0 * (Ni[2][0] + Ni[2][1] - Ni[2][2]) +
              0.5 * (- 2 * Ni[2][1] + Ni[2][2] + Ni[2][3] - Ni[2][4])
              + Ni[2][1] + Ni[2][4] + Ni[2][5])
              + f[5][i+3] * (1/3.0 * (Ni[5][0] + Ni[5][1] - Ni[5][2]) +
              0.5 * (- 2 * Ni[5][1] + Ni[5][2] + Ni[5][3] - Ni[5][4])
              + Ni[5][1] + Ni[5][4] + Ni[5][5]))
            + L3 * (f[0][i+3] * (1/3.0 * Ni[0][1] + 0.5 * Ni[0][4] + Ni[0][5]) 
              + f[2][i] * (1/3.0 * Ni[2][1] + 0.5 * Ni[2][4] + Ni[2][5]) 
              + f[4][i] * (1/3.0 * Ni[4][1] + 0.5 * Ni[4][4] + Ni[4][5])));*/  
   }  
   /*if (triElemIdx == 0){
         printf(" %d F = %e %e %e\n", triElemIdx, F[0], F[1], F[2]);
         printf("F1 = %e %e %e\n",F1[0], F1[1], F1[2]);
         printf("F2 = %e %e %e\n",F2[0], F2[1], F2[2]);
         printf("F3 = %e %e %e\n",F3[0], F3[1], F3[2]);          
   }*/   
   for (i = 0; i < 3; i++){
       Fx[i] = F[0]/S[node[i]];
       Fy[i] = F[1]/S[node[i]];
       Fz[i] = F[2]/S[node[i]];
       row1 = (double*)((char*)F_X + node[i] * pitch);
       atomicAdd(&(row1[0]), Fx[i]/F0);
       row1 = (double*)((char*)F_Y + node[i] * pitch);
       atomicAdd(&(row1[0]), Fy[i]/F0);
       row1 = (double*)((char*)F_Z + node[i] * pitch);
       atomicAdd(&(row1[0]), Fz[i]/F0);
      // if (node[i] == 6)
     //     printf("%d %e %e %e\n",triElemIdx,Fx[i],Fy[i],Fz[i]);
   }   
}

__global__ void
sem_Laplace_Kapa_kernel(void*g){
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
   int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (nodeIdx >= grids->maxElements) return;
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *F_X = grids->F_X;
   double *F_Y = grids->F_Y;
   double *F_Z = grids->F_Z;
   double* km = grids->km;
   double *n = grids->n;
   //int* node_share_Elem = grids->node_share_Elem;
   int* node_nbr = grids->node_nbrElemNum;
   int* node_nbrNodes = grids->node_nbrNodes;
  // int *triElem = grids->triElem;   
   size_t pitch = grids->pitch;
   int i,j,k;
   int* row;
   double* row1;
   double xi[3], xj[3], x1[3], x2[3], ni[3];
   row1 = (double*)((char*)X + nodeIdx * pitch);
   xi[0] = row1[0]; 
   row1 = (double*)((char*)Y + nodeIdx * pitch);
   xi[1] = row1[0]; 
   row1 = (double*)((char*)Z + nodeIdx * pitch);
   xi[2] = row1[0];
   row1 = (double*)((char*)n + nodeIdx * pitch);
   ni[0] = row1[0];
   ni[1] = row1[1];
   ni[2] = row1[2];
   double Ai = 0, sum = 0;
   double ri1[3], r1j[3], rij[3], ri2[3], r2j[3]; 
  // double ci1[3], c1j[3], 
   double cij[3], ci2[3], c2j[3]; 
   double ai, a1, a2, aj, ce[3];  
   for (i = 0; i < node_nbr[nodeIdx]; i++){
       row = (int*)((char*)node_nbrNodes + nodeIdx * pitch);
       row1 = (double*)((char*)X + row[i] * pitch);
       xj[0] = row1[0]; 
       row1 = (double*)((char*)Y + row[i] * pitch);
       xj[1] = row1[0]; 
       row1 = (double*)((char*)Z + row[i] * pitch);
       xj[2] = row1[0];
       if (i == 0) j = node_nbr[nodeIdx] - 1;           
       else j = i-1;
       row1 = (double*)((char*)X + row[j] * pitch);
       x1[0] = row1[0]; 
       row1 = (double*)((char*)Y + row[j] * pitch);
       x1[1] = row1[0]; 
       row1 = (double*)((char*)Z + row[j] * pitch);
       x1[2] = row1[0];
       if (i == (node_nbr[nodeIdx] -1)) j = 0;
       else j = i + 1;
       row1 = (double*)((char*)X + row[j] * pitch);
       x2[0] = row1[0]; 
       row1 = (double*)((char*)Y + row[j] * pitch);
       x2[1] = row1[0]; 
       row1 = (double*)((char*)Z + row[j] * pitch);
       x2[2] = row1[0];
       for (j = 0; j < 3; j++){
           ri1[j] = xi[j] - x1[j];
           r1j[j] = x1[j] - xj[j]; 
           rij[j] = xi[j] - xj[j];
           ri2[j] = xi[j] - x2[j];
           r2j[j] = x2[j] - xj[j];
    //       ci1[j] = 0.5 * (xi[j] + x1[j]);
    //       c1j[j] = 0.5 * (x1[j] + xj[j]);
           cij[j] = 0.5 * (xi[j] + xj[j]);
           ci2[j] = 0.5 * (xi[j] + x2[j]);
           c2j[j] = 0.5 * (x2[j] + xj[j]);
       }
       ai = dot(rij, ri2);
       a2 = - dot(ri2, r2j);
       aj = dot(rij, r2j);
       a1 = -dot(r1j, ri1);
       if (ai < 0) {
          ce[0] = c2j[0];
          ce[1] = c2j[1];
          ce[2] = c2j[2];
       }
       else if (a2 < 0) {
          ce[0] = cij[0];
          ce[1] = cij[1];
          ce[2] = cij[2];          
       }
       else if (aj < 0) {
          ce[0] = ci2[0];
          ce[1] = ci2[1];
          ce[2] = ci2[2];          
       }
       else{
          double gauss[3][4] = {0};
          for (k = 0; k < 3; k++){
             gauss[0][k] = rij[k];
             gauss[1][k] = ri2[k];
             gauss[0][3] += cij[k] * rij[k];
             gauss[1][3] += ci2[k] * ri2[k];
          } 
          gauss[2][0] = (xj[1] - xi[1]) * (x2[2] - xi[2]) - (xj[2] - xi[2]) * (x2[1] - xi[1]);
          gauss[2][1] = (xj[2] - xi[2]) * (x2[0] - xi[0]) - (xj[0] - xi[0]) * (x2[2] - xi[2]);
          gauss[2][2] = (xj[0] - xi[0]) * (x2[1] - xi[1]) - (xj[1] - xi[1]) * (x2[0] - xi[0]);
          gauss[2][3] = gauss[2][0] * xi[0] + gauss[2][1] * xi[1] + gauss[2][2] * xi[2];
          gaussian_elimination((double**)gauss, 3, 4);          
          ce[2] = gauss[2][3];
          ce[1] = gauss[1][3] - ce[2] * gauss[1][2];
          ce[0] = gauss[0][3] - ce[2] * gauss[0][2] - ce[1] * gauss[0][1];
       } 
       double L1 = norm_line(xi,cij);
       double L2 = norm_line(ce,cij);
       double L3 = norm_line(xi,ce);
       double Pe = 0.5 * (L1 + L2 + L3);
       double Ae = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));
       Ai += Ae;      
       L1 = norm_line(xi,ci2);
       L2 = norm_line(ce,ci2);
       Pe = 0.5 * (L1 + L2 + L3);
       Ae = sqrt(Pe * (Pe - L1) * (Pe - L2)* (Pe - L3));
       Ai += Ae;
       double alpha = 1/tan(acos(a1/(norm_line(xi,x1) * norm_line(x1,xj))));
       double beta =  1/tan(acos(a2/(norm_line(xi,x2) * norm_line(x2,xj))));
       row1 = (double*)((char*)km + nodeIdx * pitch);
       double kmi = row1[0];
       row1 = (double*)((char*)km + row[i] * pitch);
       double kmj = row1[0];       
       sum += 0.5 * (alpha + beta) * (kmj - kmi);
       //if (nodeIdx == 0) 
          printf("%e %e %e %e %e sum = %e\n", alpha, beta, kmj, kmi, Ae, sum); 
   }
   
   row1 = (double*)((char*)F_X + nodeIdx * pitch);
   row1[0] += Kbend * 1/Ai * sum * ni[0]/F0;
  // if (nodeIdx == 0) printf("%e %e %e\n", Ai, sum, 1/Ai*sum);
   row1 = (double*)((char*)F_Y + nodeIdx * pitch);
   row1[0] += Kbend * 1/Ai * sum * ni[1]/F0;
   row1 = (double*)((char*)F_Z + nodeIdx * pitch);
   row1[0] += Kbend * 1/Ai * sum * ni[2]/F0;
}

__global__ void
sem_ForceAreaVol_kernel(void* g){
   sem_GPUgrids *grids = (sem_GPUgrids *)g;
   int triElemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (triElemIdx >= grids->SurfElem) return;

   int node[6];
   double node_coord[6][3];
   int *triElem = grids->triElem;
   double *X = grids->X;
   double *Y = grids->Y;
   double *Z = grids->Z;
   double *F_X = grids->F_X;
   double *F_Y = grids->F_Y;
   double *F_Z = grids->F_Z;
   double *nelem = grids->nelem;
   double *S0 = grids->S0;
   double S0_all = grids->S0_all;
   double S_all = grids->S_all;
   double *S = grids->S;
   double V0 = grids->V0;
   double V = grids->V;
   //double *nelem = grids->nelem;
   size_t pitch = grids->pitch;

   int *row = (int*)((char*)triElem + triElemIdx * pitch);
   double *row1;
   int i, j;
   double S0e = S0[triElemIdx];
   double Se = S[triElemIdx];
   
   for (i = 0; i < 3; i++){
        node[i] = row[i];
        node[i+3] = row[i+3];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];
   }
   
   double L[3][3], coord_center[3], n_elem[3];
   row1 = (double*)((char*)nelem + triElemIdx * pitch);
   for (i = 0; i < 3; i++){
       L[0][i] = node_coord[1][i] - node_coord[2][i];
       L[1][i] = node_coord[2][i] - node_coord[0][i];
       L[2][i] = node_coord[0][i] - node_coord[1][i];
       coord_center[i] = 1/3.0 * (node_coord[0][i] + node_coord[1][i] + node_coord[2][i]);
       n_elem[i] = row1[i];
   }
   /*n_elem[0][0] = L[2][1] * L[1][2] - L[2][2] * L[1][1];
   n_elem[0][1] = L[2][2] * L[1][0] - L[2][0] * L[1][2];
   n_elem[0][2] = L[2][0] * L[1][1] - L[2][1] * L[1][0];
   n_elem[1][0] = L[0][1] * L[2][2] - L[0][2] * L[2][1];
   n_elem[1][1] = L[0][2] * L[2][0] - L[0][0] * L[2][2];
   n_elem[1][2] = L[0][0] * L[2][1] - L[0][1] * L[2][0];
   n_elem[2][0] = L[1][1] * L[0][2] - L[1][2] * L[0][1];
   n_elem[2][1] = L[1][2] * L[0][0] - L[1][0] * L[0][2];
   n_elem[2][2] = L[1][0] * L[0][1] - L[1][1] * L[0][0];*/
   
   /*if (triElemIdx == 0) {
      printf("%e %e %e\n", n_elem[0][0], n_elem[0][1], n_elem[0][2]);
      printf("%e %e %e\n", n_elem[1][0], n_elem[1][1], n_elem[1][2]);
      printf("%e %e %e\n", n_elem[2][0], n_elem[2][1], n_elem[2][2]); 
   }*/
   //double V = 1/3.0 * (coord_center[0] * n_elem[0] + coord_center[1] * n_elem[1]
   //                + coord_center[2] * n_elem[2]) * S[triElemIdx];
   double dSdx[3][3], dVdx[3][3], F[3];
   for (i = 0; i < 3; i++){
       dSdx[i][0] = -0.5 * (L[i][1] * n_elem[2] - L[i][2] * n_elem[1]);
       dSdx[i][1] = -0.5 * (L[i][2] * n_elem[0] - L[i][0] * n_elem[2]);
       dSdx[i][2] = -0.5 * (L[i][0] * n_elem[1] - L[i][1] * n_elem[0]);
       for (j = 0; j < 3; j++){
           F[j] = - Ks * ((Se - S0e)/S0e +(S_all - S0_all)/S0_all)* dSdx[i][j];
       }
   //    if (triElemIdx == 0) printf("%e %e %e\n", dSdx[i][0], dSdx[i][1], dSdx[i][2]);
      // if (triElemIdx == 0) printf("S = %e, S0 = %e\n", S_all, S0_all);
       row1 = (double*)((char*)F_X + node[i] * pitch);
       atomicAdd(&(row1[0]), F[0]/F0);
       row1 = (double*)((char*)F_Y + node[i] * pitch);
       atomicAdd(&(row1[0]), F[1]/F0);
       row1 = (double*)((char*)F_Z + node[i] * pitch);
       atomicAdd(&(row1[0]), F[2]/F0);       
   }
   dVdx[0][0] = - L[1][1]*L[0][2]/3.0 + L[1][2]*L[0][1]/3.0 - L[0][2] * coord_center[1] + L[0][1] * coord_center[2];
   dVdx[0][1] = - L[1][2]*L[0][0]/3.0 + L[1][0]*L[0][2]/3.0 + L[0][2] * coord_center[0] - L[0][0] * coord_center[2];
   dVdx[0][2] = - L[1][0]*L[0][1]/3.0 + L[1][1]*L[0][0]/3.0 - L[0][1] * coord_center[0] + L[0][0] * coord_center[1];
   dVdx[1][0] = - L[1][1]*L[0][2]/3.0 + L[1][2]*L[0][1]/3.0 - L[1][2] * coord_center[1] + L[1][1] * coord_center[2];
   dVdx[1][1] = - L[1][2]*L[0][0]/3.0 + L[1][0]*L[0][2]/3.0 + L[1][2] * coord_center[0] - L[1][0] * coord_center[2];
   dVdx[1][2] = - L[1][0]*L[0][1]/3.0 + L[1][1]*L[0][0]/3.0 - L[1][1] * coord_center[0] + L[1][0] * coord_center[1];
   dVdx[2][0] = - L[1][1]*L[0][2]/3.0 + L[1][2]*L[0][1]/3.0 - L[2][2] * coord_center[1] + L[2][1] * coord_center[2];
   dVdx[2][1] = - L[1][2]*L[0][0]/3.0 + L[1][0]*L[0][2]/3.0 + L[2][2] * coord_center[0] - L[2][0] * coord_center[2];
   dVdx[2][2] = - L[1][0]*L[0][1]/3.0 + L[1][1]*L[0][0]/3.0 - L[2][1] * coord_center[0] + L[2][0] * coord_center[1];
   //if (triElemIdx == 0) printf("%e %e %e %e\n", S[triElemIdx], S0[triElemIdx], V, V0[triElemIdx]);
   for (i = 0; i < 3; i++){
       for (j = 0; j < 3; j++){
           F[j]= - Kv *(V - V0)/V0 * dVdx[i][j]/6.0;
       }
       row1 = (double*)((char*)F_X + node[i] * pitch);
       atomicAdd(&(row1[0]), F[0]/F0);
       row1 = (double*)((char*)F_Y + node[i] * pitch);
       atomicAdd(&(row1[0]), F[1]/F0);
       row1 = (double*)((char*)F_Z + node[i] * pitch);
       atomicAdd(&(row1[0]), F[2]/F0);       
       //if (triElemIdx == 0) printf("V = %e, V0 = %e\n", V, V0);
       //if (triElemIdx == 0) printf("%e %e %e\n", F[0], F[1], F[2]);
   }         
}
__global__ void
sem_platelet_wall_kernel(void* g, void* g_SEM){
    fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
    sem_GPUgrids *grids_SEM = (sem_GPUgrids *)g_SEM;

    int triElemIdx = blockIdx.x;
    int receptorIdx = threadIdx.x;
    if (triElemIdx >= grids_SEM->SurfElem) return;
    if (receptorIdx >= grids_SEM->numReceptorsPerElem) return;
    
    int node[3];
    double node_coord[3][3];
    int numReceptors = grids_SEM->numReceptorsPerElem;
    //int SurfElem = grids_SEM->SurfElem;
    int totalBond = grids_SEM->totalBond;
    int avaiBond = NUMOFRECEPTORS - totalBond;
    int *triElem = grids_SEM->triElem;
    float *receptor_r1 = grids_SEM->receptor_r1;
    float *receptor_r2 = grids_SEM->receptor_r2;
    size_t pitch = grids_SEM->pitch;
    double *X = grids_SEM->X;
    double *Y = grids_SEM->Y;
    double *Z = grids_SEM->Z;
    double *F_X = grids_SEM->F_X;
    double *F_Y = grids_SEM->F_Y;
    double *F_Z = grids_SEM->F_Z;
   
    size_t receptPitch = grids_SEM->receptBond.pitch;
    size_t receptSlicePitch = numReceptors * receptPitch;
    char *receptBondPtr = (char*)grids_SEM->receptBond.ptr;
    char *receptSlice;
    int *receptBond;
    
    size_t randNumPitch = grids_SEM->randNum.pitch;
    size_t randNumSlicePitch = numReceptors * randNumPitch;
    char *randNumPtr = (char*)grids_SEM->randNum.ptr;
    char *randNumSlice;
    double *randNum;

    int *row = (int*)((char*)triElem + triElemIdx * pitch);
    double *row1;
    int i,j,k;
    for (i = 0; i < 3; i++){
        node[i] = row[i];
        row1 = (double*)((char*)X + node[i] * pitch);
        node_coord[i][0] = row1[0];
        row1 = (double*)((char*)Y + node[i] * pitch);
        node_coord[i][1] = row1[0];
        row1 = (double*)((char*)Z + node[i] * pitch);
        node_coord[i][2] = row1[0];    
    }
    
    int recept_vWF[2] = {-1, -1};
    float r1, r2;
    double recept_X, recept_Y, recept_Z;
    float *row2 = (float*)((char*)receptor_r1 + triElemIdx * pitch);
    r1 = row2[receptorIdx]; 
   // r1 = sqrt(r1);    
    row2 = (float*)((char*)receptor_r2 + triElemIdx * pitch);
    r2 = row2[receptorIdx];  
    
    recept_Z = (1-r1) * node_coord[0][2] + r1 * (1-r2) * node_coord[1][2] + r1 * r2 * node_coord[2][2];   
    int bond = 0;
    receptSlice = receptBondPtr + 2 * receptSlicePitch;
    receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
    if (receptBond[triElemIdx] > -1) {
       bond = receptBond[triElemIdx];
    }
    
    if (recept_Z > BONDTHRESHOLD && bond == 0) return; 
    
    for (i = 0; i < 2; i++){
        receptSlice = receptBondPtr + i * receptSlicePitch;
        receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
        recept_vWF[i] = receptBond[triElemIdx];
    }
         
   // curandState *state = grids_SEM->devState;
    int id = threadIdx.x + blockIdx.x * blockDim.x;
   
    size_t vWFpitch = grids->vWFbond.pitch;
    size_t vWFSlicePitch = grids->height * vWFpitch;
    char *devPtr = (char*)grids->vWFbond.ptr;
    char *slice;
    int *vWF;
    
   // curandState localState = state[id];    
    double dt = grids->dt, dx = grids->dx, dx1, dy1, dz1, dr1, weight1, fconst = 0.0;
    double fX, fY, fZ;
    double aon = 0.0, aoff = 0.0, randnum1 = 0.0, randnum2 = 0.0; 
    recept_X = (1-r1) * node_coord[0][0] + r1 * (1-r2) * node_coord[1][0] + r1 * r2 * node_coord[2][0];   
    recept_Y = (1-r1) * node_coord[0][1] + r1 * (1-r2) * node_coord[1][1] + r1 * r2 * node_coord[2][1];
    if (bond){                
       i = recept_vWF[0]; 
       j = recept_vWF[1];
    }
    else{
       i = (int)(recept_X/dx + 0.5);
       j = (int)(recept_Y/dx + 0.5);          
    }
    dx1 = recept_X - i * dx;
    dy1 = recept_Y - j * dx;
    dz1 = recept_Z;
    dr1 = norm(dx1, dy1, dz1);
    weight1 = abs(BONDTHRESHOLD - dr1);
    fconst = k_receptvWF * weight1;
    
    double Kf,Kr;
   // printf("Kf0 = %e\n",Kf0);
    Kf = Kf0 * exp(fconst * (0.00071 - 0.5 * weight1)/KbT)* avaiBond; 
   // Kf = Kf0 * 1e4; 
   // printf("Kf = %e\n",Kf);
    Kr = Kr0 * exp(0.00071 * fconst /KbT)* avaiBond; 
    aon = 1 - exp(- Kf * dt);
    aoff = 1 - exp(- Kr * dt);
    randNumSlice = randNumPtr;
    randNum = (double*) (randNumSlice + receptorIdx * randNumPitch);
    randnum1 = randNum[triElemIdx];                                        
    randNumSlice = randNumPtr + randNumSlicePitch;
    randNum = (double*) (randNumSlice + receptorIdx * randNumPitch);
    randnum2 = randNum[triElemIdx];                                        
    
   // a = aon + aoff;
   // randnum = curand_uniform(&localState);
   //if (triElemIdx == 95) printf("%e\n",randnum);
   // prob = randnum * a;
    
    //if (bond == 1) printf("aon = %e, aoff = %e, randnum1 = %e\n, randnum2 = %e\n", aon, aoff, randnum1, randnum2);
    
    int m;
    if (bond == 0){
       if (randnum1 < aon){
          i = (int)(recept_X/dx + 0.5);
          j = (int)(recept_Y/dx + 0.5);
         // printf("i = %d, j = %d\n",i,j);
         // printf("randnum = %e, aon = %e\n", randnum, aon);
          for (k = 0; k < 4; k++){
              slice = devPtr + k * vWFSlicePitch;
              vWF = (int*) (slice + j * vWFpitch);
             // printf("vWF[i] = %d\n", vWF[i]);              
              int old = atomicCAS(&(vWF[i]), -1, id);
          //    printf("old = %d\n", old);
              if (old == -1) {
                 bond++;
                 randnum1 = -1; 
                 receptSlice = receptBondPtr;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = i;                                        
                 receptSlice = receptBondPtr + receptSlicePitch;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = j;                          
                 break; 
              }              
          }
      }
     // randnum = curand_uniform(&localState);
      if (randnum2 < aon){
          i = (int)(recept_X/dx + 0.5);
          j = (int)(recept_Y/dx + 0.5);
          // printf("i = %d, j = %d\n",i,j);
          for (k = 0; k < 4; k++){
              slice = devPtr + k * vWFSlicePitch;
              vWF = (int*) (slice + j * vWFpitch);
             // printf("vWF[i] = %d\n", vWF[i]);              
              int old = atomicCAS(&(vWF[i]), -1, id);
             //    printf("old = %d\n", old);
              if (old == -1) {
                 bond++;
                 randnum2 = -1;
                 receptSlice = receptBondPtr;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = i;                                        
                 receptSlice = receptBondPtr + receptSlicePitch;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = j;                          
                 break; 
              }              
          }
      }
       
    }
    else if (bond == 1){
       if (randnum1 < aoff){
          for (k = 0; k < 4; k++){
              slice = devPtr + k * vWFSlicePitch;
              vWF = (int*) (slice + j * vWFpitch);
              int old = atomicCAS(&vWF[i], id, -1);
              if (old == id) {
                 bond--;
                 randnum1 = -1;
                 receptSlice = receptBondPtr;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                                        
                 receptSlice = receptBondPtr + receptSlicePitch;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                          
                 break; 
              }             
   //           printf("randnum = %e, aoff = %e\n",randnum1,aoff); 
          }
          //randnum = curand_uniform(&localState);
          if (randnum2 < aon && recept_Z < BONDTHRESHOLD){
              i = (int)(recept_X/dx + 0.5);
              j = (int)(recept_Y/dx + 0.5);
              // printf("i = %d, j = %d\n",i,j);
              for (k = 0; k < 4; k++){
                  slice = devPtr + k * vWFSlicePitch;
                  vWF = (int*) (slice + j * vWFpitch);
             // printf("vWF[i] = %d\n", vWF[i]);              
                  int old = atomicCAS(&(vWF[i]), -1, id);
             //    printf("old = %d\n", old);
                  if (old == -1) {
                     randnum2 = -1;
                     bond++;
                     receptSlice = receptBondPtr;
                     receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                     receptBond[triElemIdx] = i;                                        
                     receptSlice = receptBondPtr + receptSlicePitch;
                     receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                     receptBond[triElemIdx] = j;                          
                     break; 
                  }              
              }
           }
                    
       }
       else{
           //randnum = curand_uniform(&localState);
           if (randnum2 < aoff && recept_Z < BONDTHRESHOLD){
              for (k = 0; k < 4; k++){
                  slice = devPtr + k * vWFSlicePitch;
                  vWF = (int*) (slice + j * vWFpitch);
                  int old = atomicCAS(&vWF[i], id, -1);
                  if (old == id) {
                     bond--;
                     randnum2 = -1;
                     receptSlice = receptBondPtr;
                     receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                     receptBond[triElemIdx] = -1;                                        
                     receptSlice = receptBondPtr + receptSlicePitch;
                     receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                     receptBond[triElemIdx] = -1;                          
                     break; 
                  }                 
               }
          
           }
       }  
   }
   else if (bond == 2){
       if (randnum1 < aoff){
          for (k = 0; k < 4; k++){
              slice = devPtr + k * vWFSlicePitch;
              vWF = (int*) (slice + j * vWFpitch);
              int old = atomicCAS(&vWF[i], id, -1);
              if (old == id) {
                 bond--;
                 randnum1 = -1;
                 receptSlice = receptBondPtr;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                                        
                 receptSlice = receptBondPtr + receptSlicePitch;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                          
                 break; 
              }              
          }
       }
       //randnum = curand_uniform(&localState);
       if (randnum2 < aoff){
          for (k = 0; k < 4; k++){
              slice = devPtr + k * vWFSlicePitch;
              vWF = (int*) (slice + j * vWFpitch);
              int old = atomicCAS(&vWF[i], id, -1);
              if (old == id) {
                 bond--;
                 randnum2 = -1;
                 receptSlice = receptBondPtr;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                                        
                 receptSlice = receptBondPtr + receptSlicePitch;
                 receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
                 receptBond[triElemIdx] = -1;                          
                 break; 
              }              
          }
       }                           
   }
   
   if (bond){
      // printf("bond = %d\n", bond);                
       dx1 = recept_X - i * dx;
       dy1 = recept_Y - j * dx;
       dz1 = recept_Z;
       dr1 = norm(dx1, dy1, dz1);
       weight1 = BONDTHRESHOLD - dr1;
       fconst = k_receptvWF * weight1 * bond / dr1;
       fX = fconst * dx1;
       fY = fconst * dy1;
       fZ = fconst * dz1;
       double dr[3];
       for (m = 0; m < 3; m++){
           dx1 = recept_X - node_coord[m][0];
           dy1 = recept_Y - node_coord[m][1];
           dz1 = recept_Z - node_coord[m][2];
           dr[m] = norm(dx1,dy1,dz1);
       }
       double sum = 1.0/dr[0] + 1.0/dr[1] + 1.0/dr[2];
       for (m = 0; m < 3; m++){
           row1 = (double*)((char*)F_X + node[m] * pitch);
           row1[0] += fX * 1.0/dr[m]/sum / F0;
           row1 = (double*)((char*)F_Y + node[m] * pitch);
           row1[0] += fY * 1.0/dr[m]/sum / F0;
           row1 = (double*)((char*)F_Z + node[m] * pitch);
           row1[0] += fZ * 1.0/dr[m]/sum / F0;                 
       }
       receptSlice = receptBondPtr + 2 * receptSlicePitch;
       receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
       receptBond[triElemIdx] = bond;       
       atomicAdd(&(grids_SEM->totalBond), bond);
   }
   else{
        receptSlice = receptBondPtr + 2 * receptSlicePitch;
        receptBond = (int*) (receptSlice + receptorIdx * receptPitch);
        receptBond[triElemIdx] = -1;     
   }
  // state[id] = localState;
   randNumSlice = randNumPtr;
   randNum = (double*) (randNumSlice + receptorIdx * randNumPitch);
   randNum[triElemIdx] = -1;                                        
   randNumSlice = randNumPtr + randNumSlicePitch;
   randNum = (double*) (randNumSlice + receptorIdx * randNumPitch);
   randNum[triElemIdx] = -1;                                                    
}



__global__ void
sem_update_position_kernel(void* g, void* g_SEM)
{
   fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
   sem_GPUgrids *grids_SEM = (sem_GPUgrids *)g_SEM;
   int cellNum = 0;
   int elemNum = blockIdx.x * blockDim.x + threadIdx.x;

   if (cellNum >= grids_SEM->numOfCells) return;
   if (elemNum >= grids_SEM->numOfElements[cellNum]) return;
   
   double dx = grids->dx;
   int height = grids->height;
   double boundary = height * dx;
   double *X = grids_SEM->X;
   double *Y = grids_SEM->Y;
   double *RY = grids_SEM->RY;
   double *Z = grids_SEM->Z;
   double *V_X = grids_SEM->V_X;
   double *V_Y = grids_SEM->V_Y;
   double *V_Z = grids_SEM->V_Z;
   //double *PreV_X = grids->PreV_X;
   //float *PreV_Y = grids->PreV_Y;
   //float *PreV_Z = grids->PreV_Z;
  // double rand_num;
   float dt = grids_SEM->dt/T0;
   size_t pitch = grids_SEM->pitch;
   double *row1 = (double*)((char*)X + elemNum * pitch);
   double *row2 = (double*)((char*)V_X + elemNum * pitch);
  // double *row3 = (double*)((char*)PreV_X + elemNum * pitch);
  // row1[cellNum] += row3[cellNum] * dt + 0.5* (row2[cellNum] - row3[cellNum]) * dt;
  // rand_num = (double)get_random()/(double)32767*elemNum/179;
  // printf("rand = %f\n",rand_num);
  // row1[cellNum] += row2[cellNum] * dt + 1e-3* rand_num;
   row1[cellNum] += row2[cellNum] * dt;
  // row3[cellNum] = row2[cellNum];
   row1 = (double*)((char*)Y + elemNum * pitch);
   row2 = (double*)((char*)V_Y + elemNum * pitch);
  /* if (elemNum == 0){
      printf("vy = %.16e\n",row2[0]);
   }*/
   double *row3 = (double*)((char*)RY + elemNum * pitch);
 //  row1[cellNum] += row3[cellNum] * dt + 0.5* (row2[cellNum] - row3[cellNum]) * dt;
  // row1[cellNum] += row2[cellNum] * dt + 1e-3 * rand_num;
   row1[cellNum] += row2[cellNum] * dt;
   if (row1[cellNum] > boundary){
      row1[cellNum] -= boundary;
   }
  // row3[cellNum] += row2[cellNum] * dt + 1e-3 * rand_num;
   row3[cellNum] += row2[cellNum] * dt;
   row1 = (double*)((char*)Z + elemNum * pitch);
   row2 = (double*)((char*)V_Z + elemNum * pitch);
  // row3 = (double*)((char*)PreV_Z + elemNum * pitch);
   //row1[cellNum] += row3[cellNum] * dt + 0.5* (row2[cellNum] - row3[cellNum]) * dt;
  // row1[cellNum] += row2[cellNum] * dt + 1e-3 * rand_num;
   row1[cellNum] += row2[cellNum] * dt;
  // if (elemNum == 0){
  //    printf("vz = %.16e\n", row2[0]);
  // }
  // row3[cellNum] = row2[cellNum];
   
 }

 #endif
