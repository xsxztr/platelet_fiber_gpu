/*
 semLB.cu
 
 Main functions for running the GPU kernels.

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

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <curand_kernel.h>
#include <gsl/gsl_rng.h>


// Constant GPU array for holding parameters
__constant__ float model_Parameters[100];

extern "C" {
  #define X_a 64
  #define Y_a 1
  #include "GPUDefines.h"

  #include <stdio.h>
  //#include <BioSwarm/GPUDefines.h>

typedef struct _fluid_GPUgrids {
	float dt;
    	float dx;
    	int width;
    	int height;
    	int depth;
    	size_t pitch;
    	cudaExtent iextent; //int extent
    	cudaExtent fextent; //float extent
    	cudaExtent dextent; //double extent
   	cudaPitchedPtr fIN[19];
    	cudaPitchedPtr fOUT[19];
    	cudaPitchedPtr ux;
    	cudaPitchedPtr uy;
    	cudaPitchedPtr uz;
    	cudaPitchedPtr obst;
    	cudaPitchedPtr rho;
    	cudaPitchedPtr Fx;  //External force
    	cudaPitchedPtr Fy;
    	cudaPitchedPtr Fz;
    	cudaPitchedPtr vWFbond;
    	void *deviceStruct;
  } fluid_GPUgrids;


  #include "sem_kernel.cu"
  #include "LB_kernel.cu"

 

void cudacheck(const char *message)
{
    	cudaError_t error = cudaGetLastError();
    	if (error!=cudaSuccess)
    	{
      		printf("cudaERROR: %s : %i (%s)\n", message, error, cudaGetErrorString(error));
      		exit(EXIT_FAILURE);
    	}
}
  
void pitchcheck(size_t pitch_test, size_t pitch, int id)
{
    	if (pitch_test != pitch)
	{ 
        	printf("pitch is not match at %d.\n", id);
         	exit(1); 
       	}
    
}


////////// GPU for Fluid ////////////////////////////////////////////////////////////////////////
   ///BEGIN: FLUID ALLOC AND INITIAL ///

  // Allocation
void *fluid_allocGPUKernel(void *model, float dt, float dx, int width, int height, int depth)
{
    	fluid_GPUgrids *grids = (fluid_GPUgrids *)malloc(sizeof(fluid_GPUgrids));
    	int i;

    // Save parameters
    	grids->dt = dt;
    	grids->dx = dx;
    	grids->width = width;
    	grids->height = height;
    	grids->depth = depth;

    // Allocate device memory
    	grids->iextent = make_cudaExtent(grids->width*sizeof(int), grids->height, 4);
    	grids->fextent = make_cudaExtent(grids->width*sizeof(float), grids->height, grids->depth);
    	grids->dextent = make_cudaExtent(grids->width*sizeof(double), grids->height, grids->depth);
    	for (i = 0; i < 19; ++i) 
	{
      		cudaMalloc3D(&(grids->fIN[i]), grids->dextent);
      		cudacheck("FIN alloc");
      //printf("%d\n", grids->fIN[i].pitch);
      		cudaMalloc3D(&(grids->fOUT[i]), grids->dextent);
      		cudacheck("FOUT alloc");
      //printf("%d\n", grids->fIN[i].pitch);
    	}
    
    	cudaMalloc3D(&(grids->vWFbond), grids->iextent);
    	cudaMalloc3D(&(grids->ux), grids->dextent);
    	cudaMalloc3D(&(grids->uy), grids->dextent);
   	cudaMalloc3D(&(grids->uz), grids->dextent);
    	cudaMalloc3D(&(grids->obst), grids->fextent);
    	cudaMalloc3D(&(grids->rho), grids->dextent);
    	cudaMalloc3D(&(grids->Fx), grids->dextent);
    	cudaMalloc3D(&(grids->Fy), grids->dextent);
    	cudaMalloc3D(&(grids->Fz), grids->dextent);

    	cudaMalloc(&(grids->deviceStruct), sizeof(fluid_GPUgrids));
    	cudacheck("Fluid alloc");

    	return grids;
}

  // Initialization
void fluid_initGPUKernel(void *model, void *g, int aFlag, float *hostParameters, void **fIN, void **fOUT, void *ux,
			   void *uy, void *uz, void *rho, void *obstacle, void *Fx, void *Fy, void *Fz, void *vWFbond)
{
    	fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
    	int i;

    	if (aFlag) {
      // Copy host memory to device memory
     // cudaMemcpyToSymbol(model_Parameters, hostParameters, 16 * sizeof(float), 0, cudaMemcpyHostToDevice);
     // cudacheck("constant");

      		cudaMemcpy3DParms p3d = {0};
      		cudaError_t error;
      		p3d.extent = grids->dextent;
      		p3d.kind = cudaMemcpyHostToDevice;

      		for (i = 0; i < 19; ++i) {
			p3d.srcPtr = make_cudaPitchedPtr(fIN[i], grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
			p3d.dstPtr = grids->fIN[i];
			error = cudaMemcpy3D(&p3d);
		if (error) printf("cudaERROR in: %s\n", cudaGetErrorString(error));

		p3d.srcPtr = make_cudaPitchedPtr(fOUT[i], grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
		p3d.dstPtr = grids->fOUT[i];
		error = cudaMemcpy3D(&p3d);
		if (error) printf("cudaERROR out %d: %s\n", i, cudaGetErrorString(error));
 		}
      
    //  error = cudaMemset3D(grids->vWFbond, -1, grids->iextent);
    //  if (error) printf("cudaERROR Memset vWFbond: %s\n", cudaGetErrorString(error));
    
      		p3d.srcPtr = make_cudaPitchedPtr(ux, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->ux;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR ux: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = make_cudaPitchedPtr(uy, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->uy;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR uy: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = make_cudaPitchedPtr(uz, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->uz;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR uz: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = make_cudaPitchedPtr(rho, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->rho;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR rho: %s\n", cudaGetErrorString(error));
      
      		p3d.srcPtr = make_cudaPitchedPtr(Fx, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->Fx;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fx: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = make_cudaPitchedPtr(Fy, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->Fy;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fy: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = make_cudaPitchedPtr(Fz, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		p3d.dstPtr = grids->Fz;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fz: %s\n", cudaGetErrorString(error));
      
      		p3d.extent = grids->fextent;
      		p3d.srcPtr = make_cudaPitchedPtr(obstacle, grids->width*sizeof(float), grids->width*sizeof(float), grids->height);
      		p3d.dstPtr = grids->obst;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR obstacle: %s\n", cudaGetErrorString(error));
      
      		p3d.extent = grids->iextent;
      		p3d.srcPtr = make_cudaPitchedPtr(vWFbond, grids->width*sizeof(int), grids->width*sizeof(int), grids->height);
      		p3d.dstPtr = grids->vWFbond;
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR vWFbond: %s\n", cudaGetErrorString(error));

      		cudaMemcpy(grids->deviceStruct, grids, sizeof(fluid_GPUgrids), cudaMemcpyHostToDevice);
      		cudacheck("deviceStruct");

   	 } else {

      // Copy result to host memory
     		cudaMemcpy3DParms p3d = {0};
      		cudaError_t error;
      		p3d.extent = grids->dextent;
      		p3d.kind = cudaMemcpyDeviceToHost;

      		for (i = 0; i < 19; ++i) {
	      		p3d.srcPtr = grids->fIN[i];
	      		p3d.dstPtr = make_cudaPitchedPtr(fIN[i], grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
	      		error = cudaMemcpy3D(&p3d);
	      		if (error) printf("cudaERROR in 0: %s\n", cudaGetErrorString(error));

	     	 	p3d.srcPtr = grids->fOUT[i];
	      		p3d.dstPtr = make_cudaPitchedPtr(fOUT[i], grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
	      		error = cudaMemcpy3D(&p3d);
	      		if (error) printf("cudaERROR out 0: %d %s\n", i, cudaGetErrorString(error));
      		}

      		p3d.srcPtr = grids->ux;
      		p3d.dstPtr = make_cudaPitchedPtr(ux, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR ux 0: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = grids->uy;
      		p3d.dstPtr = make_cudaPitchedPtr(uy, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR uy 0: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = grids->uz;
      		p3d.dstPtr = make_cudaPitchedPtr(uz, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR uz 0: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = grids->rho;
      		p3d.dstPtr = make_cudaPitchedPtr(rho, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR rho 0: %s\n", cudaGetErrorString(error)); 

      		p3d.srcPtr = grids->Fx;
      		p3d.dstPtr = make_cudaPitchedPtr(Fx, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fx 0: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = grids->Fy;
      		p3d.dstPtr = make_cudaPitchedPtr(Fy, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fy 0: %s\n", cudaGetErrorString(error));

      		p3d.srcPtr = grids->Fz;
      		p3d.dstPtr = make_cudaPitchedPtr(Fz, grids->width*sizeof(double), grids->width*sizeof(double), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR Fz 0: %s\n", cudaGetErrorString(error));
      
      		p3d.extent = grids->fextent;           
      		p3d.srcPtr = grids->obst;
      		p3d.dstPtr = make_cudaPitchedPtr(obstacle, grids->width*sizeof(float), grids->width*sizeof(float), grids->height);
      		error = cudaMemcpy3D(&p3d);
      		if (error) printf("cudaERROR obstacle 0: %s\n", cudaGetErrorString(error));

    	}
}

  ///END: FLUID ALLOC AND INITIAL ///


 // Execution fluid step(s)
void fluid_invokeGPUKernel(void *model, void *g, void *g_SEM, double *randNum, gsl_rng *r, int timeSteps)
{
    	int aBank, t, i, j, k;
    	fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
    	sem_GPUgrids *grids_SEM = (sem_GPUgrids *)g_SEM;
    	int SurfElem = grids_SEM->SurfElem;
    	int numReceptorsPerElem = grids_SEM->numReceptorsPerElem;
    // Z blocksPerGrid has to be == 1 so break into banks
    	int bankDepth = 16;
    	int zBanks = (grids->depth + bankDepth - 1) / bankDepth;
    	dim3 threadsPerBlock3D(32, 1, bankDepth);
    	dim3 blocksPerGrid3D((grids->width + threadsPerBlock3D.x - 1) / threadsPerBlock3D.x,
			 (grids->height + threadsPerBlock3D.y - 1) / threadsPerBlock3D.y, 1);
    
    	int threadsPerBlockSEM = 32;
    	int blocksPerGridSEM = (grids_SEM->maxElements + threadsPerBlockSEM - 1)/threadsPerBlockSEM;
    	int blocksPerGridRecept = grids_SEM->SurfElem;
    	int threadsPerBlockRecept = grids_SEM->numReceptorsPerElem;
    	cudaError_t error;
   	cudaMemcpy3DParms p3d = {0};
    	p3d.extent = grids_SEM->dextent;
    
    	for (t = 0; t < timeSteps; ++t) {
	  
	  	error = cudaMemset3D(grids->Fx, 0, grids->dextent);//set fluid grid external force to zero.
	  	if (error) printf("cudaERROR Memset Fx: %s\n", cudaGetErrorString(error));

	  	error = cudaMemset3D(grids->Fy, 0, grids->dextent);
	  	if (error) printf("cudaERROR Memset Fy: %s\n", cudaGetErrorString(error));

	  	error = cudaMemset3D(grids->Fz, 0, grids->dextent);
	  	if (error) printf("cudaERROR Memset Fz: %s\n", cudaGetErrorString(error));

	 // error = cudaMemset2D(grids_SEM->rho, grids_SEM->pitch, 0, grids_SEM->maxCells * sizeof(double), grids_SEM->maxElements);
	 // if (error) printf("cudaERROR Memset Rho: %s\n", cudaGetErrorString(error));
	  
          	error = cudaMemset2D(grids_SEM->V_X, grids_SEM->pitch, 0, grids_SEM->maxCells * sizeof(double), grids_SEM->maxElements);//set SEM velocity to zero.
	  	if (error) printf("cudaERROR Memset Vx: %s\n", cudaGetErrorString(error));

	  	error = cudaMemset2D(grids_SEM->V_Y, grids_SEM->pitch, 0, grids_SEM->maxCells * sizeof(double), grids_SEM->maxElements);
	  	if (error) printf("cudaERROR Memset Vy: %s\n", cudaGetErrorString(error));

	  	error = cudaMemset2D(grids_SEM->V_Z, grids_SEM->pitch, 0, grids_SEM->maxCells * sizeof(double), grids_SEM->maxElements);
	  	if (error) printf("cudaERROR Memset Vz: %s\n", cudaGetErrorString(error));
    
          	p3d.kind = cudaMemcpyDeviceToHost;          
          	p3d.srcPtr = grids_SEM->randNum;
          	p3d.dstPtr = make_cudaPitchedPtr(randNum, SurfElem*sizeof(double),SurfElem*sizeof(double), numReceptorsPerElem);
          	error = cudaMemcpy3D(&p3d);
          	if (error) printf("cudaERROR randNum to host: %s\n", cudaGetErrorString(error));
          	for (i = 0; i < SurfElem; i++){
              		for (j = 0; j < numReceptorsPerElem; j++){
                  		for (k = 0; k < 2; k++){
                      			if (*(randNum + k * SurfElem * numReceptorsPerElem + j * SurfElem + i) < 0)
                         			*(randNum + k * SurfElem * numReceptorsPerElem + j * SurfElem + i) = gsl_rng_uniform(r);
                  		}
              		}
          	} 
          	p3d.kind = cudaMemcpyHostToDevice;
          	p3d.srcPtr = p3d.dstPtr;
          	p3d.dstPtr = grids_SEM->randNum;
          	error = cudaMemcpy3D(&p3d);
          	if (error) printf("cudaERROR randNum to device: %s\n", cudaGetErrorString(error));          
	   
          	sem_platelet_wall_kernel<<<blocksPerGridRecept, threadsPerBlockRecept>>>(grids->deviceStruct, grids_SEM->devImage);
         	cudacheck("platelet_wall"); 
          	for (aBank = 0; aBank < zBanks; ++aBank) {
          		fluid3d_force_distribute_kernel<<< blocksPerGrid3D, threadsPerBlock3D>>>(grids->deviceStruct, grids_SEM->devImage, aBank);
          		cudacheck("force_distribute");
          	}
	  
      		for (aBank = 0; aBank < zBanks; ++aBank) {
        		fluid3d_collision_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        		cudacheck("collision");
      		}

      		for (aBank = 0; aBank < zBanks; ++aBank) {
        		fluid3d_stream_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        		cudacheck("stream");
      		}
	
     /* for (aBank = 0; aBank < zBanks; ++aBank) {
        fluid3d_obst_bounce_back_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        cudacheck("obst_bounce");
      }*/

     /* for (aBank = 0; aBank < zBanks; ++aBank) {
        fluid3d_obst_stream_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        cudacheck("obst_stream");
      }*/
      		for (aBank = 0; aBank < zBanks; ++aBank) {
        		fluid3d_noslip_boundary_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        		cudacheck("no_slip");
      		}
      
      		for (aBank = 0; aBank < zBanks; ++aBank) {
        		fluid3d_moving_plate_boundary_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        		cudacheck("moving_plate");
      		}
      
   /*   for (aBank = 0; aBank < zBanks; ++aBank) {
        fluid3d_edge_corner_boundary_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        cudacheck("edge_corner");
      }*/
      

     /* for (aBank = 0; aBank < zBanks; ++aBank) {
        fluid3d_yzplane_boundary_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        cudacheck("yzplane");
      }*/
      
      //// DO NOT CALL fluid3d_in_out_flow_boundary_kernel before fluid3d_obst_bounce_back_kernel() 
      //// and fluid3d_obst_stream_kernel(). There are ad hoc fix in fluid3d_in_out_flow_boundary_kernel 
      //// for fluid nodes on inflow/outflow plane directly linked to solid wall nodes.
     /* for (aBank = 0; aBank < zBanks; ++aBank) {
        fluid3d_in_out_flow_boundary_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        cudacheck("flow_boundary");
      }*/
      		for (aBank = 0; aBank < zBanks; ++aBank) {
        		fluid3d_velocity_density_kernel<<< blocksPerGrid3D, threadsPerBlock3D >>>(grids->deviceStruct, aBank);
        		cudacheck("velocity_density");
      		}
      
       		for (aBank = 0; aBank < zBanks; ++aBank) {
         		fluid3d_velocity_distribute_kernel<<< blocksPerGrid3D, threadsPerBlock3D>>>(grids->deviceStruct, grids_SEM->devImage, aBank);
         		cudacheck("velocity_distribute");
      	 	}

	  	sem_update_position_kernel<<<blocksPerGridSEM,threadsPerBlockSEM>>>(grids->deviceStruct, grids_SEM->devImage);
	  	cudacheck("update_position");

	}
    	cudaMemcpy(grids_SEM, grids_SEM->devImage, sizeof(sem_GPUgrids), cudaMemcpyDeviceToHost);
    	cudacheck("sem devImage to host"); 
     
}

  //Calculate force acting on elements
void sem_invokeGPUKernel_Force(void *model, void *g, int timeSteps, int* done, int* totalT , gsl_rng *r, double gama)
  //  add force if to pass back to io
{
   	int t;
   	sem_GPUgrids *grids = (sem_GPUgrids *)g;
   	double randNum;
 
   // printf("randNum = %e, gama = %e, totalBond = %d\n", randNum, gama, grids->totalBond); 
   // randNum = gsl_rng_uniform(r);
   // printf("randNum = %e, gama = %e, totalBond = %d\n", randNum, gama, grids->totalBond); 
   	if ((grids->totalBond == 0) && ((randNum = gsl_rng_uniform(r)) < gama)){  
   // if ((grids->totalBond == 0) && ((randNum) < gama)){  
      	printf("randNum = %e, gama = %e\n", randNum, gama); 
      	*done = 1;
   	} 
   	else {        
     		int threadsPerBlock = 32;
     		int threadsPerBlock_FEM = 32;
     		int threadsPerBlock_node = 32; 
     		int blocksPerGrid = (grids->maxElements + threadsPerBlock - 1)/threadsPerBlock;
     		int blocksPerGrid_FEM = (grids->SurfElem + threadsPerBlock_FEM - 1)/threadsPerBlock_FEM;
     		int blocksPerGrid_node =(grids->newnodeNum + threadsPerBlock_node - 1)/threadsPerBlock_node;
     		cudaError_t error;
     		(*totalT)++;
 
     		for (t = 0; t < timeSteps; ++t) {
          		grids->S_all = 0;
          		grids->V = 0;
         	 	grids->totalBond = 0;
         // printf("S = %e\n", grids->S_all);
          		error = cudaMemset(grids->S, 0, grids->SurfElem * sizeof(double));//set SEM velocity to zero.
	  		if (error) printf("cudaERROR Memset S: %s\n", cudaGetErrorString(error));
          
          		error = cudaMemset2D(grids->F_X, grids->pitch, 0, grids->maxCells * sizeof(double), grids->maxElements);//set SEM velocity to zero.
	  		if (error) printf("cudaERROR Memset F_x: %s\n", cudaGetErrorString(error));

	  		error = cudaMemset2D(grids->F_Y, grids->pitch, 0, grids->maxCells * sizeof(double), grids->maxElements);
	  		if (error) printf("cudaERROR Memset F_y: %s\n", cudaGetErrorString(error));

	  		error = cudaMemset2D(grids->F_Z, grids->pitch, 0, grids->maxCells * sizeof(double), grids->maxElements);
	  		if (error) printf("cudaERROR Memset F_z: %s\n", cudaGetErrorString(error));
	  
          		error = cudaMemset2D(grids->n, grids->pitch, 0, 3 * sizeof(double), grids->newnodeNum);
	  		if (error) printf("cudaERROR Memset n: %s\n", cudaGetErrorString(error));
          
          		error = cudaMemset2D(grids->nelem, grids->pitch, 0, 3 * sizeof(double), grids->SurfElem);
	  		if (error) printf("cudaERROR Memset nelem: %s\n", cudaGetErrorString(error));
          
          		error = cudaMemset2D(grids->q, grids->pitch, 0, 3 * sizeof(double), grids->newnodeNum);
	  		if (error) printf("cudaERROR Memset q: %s\n", cudaGetErrorString(error));
          
          		error = cudaMemset2D(grids->A, grids->pitch, 0, 9 * sizeof(double), grids->newnodeNum);
          		if (error) printf("cudaERROR Memset A: %s\n", cudaGetErrorString(error));
          
          		error = cudaMemset2D(grids->tau, grids->pitch, 0, 9 * sizeof(double), grids->newnodeNum);
          		if (error) printf("cudaERROR Memset tau: %s\n", cudaGetErrorString(error));

          		error = cudaMemset2D(grids->Kapa, grids->pitch, 0, 9 * sizeof(double), grids->newnodeNum);
          		if (error) printf("cudaERROR Memset Kapa: %s\n", cudaGetErrorString(error));

	  		error = cudaMemset2D(grids->K, grids->pitch, 0, 27 * sizeof(double), grids->newnodeNum);
	  		if (error) printf("cudaERROR Memset K: %s\n", cudaGetErrorString(error));
	  
          		error = cudaMemset2D(grids->Laplace_km, grids->pitch, 0, sizeof(double), grids->maxElements);
	  		if (error) printf("cudaERROR Memset Laplace_km: %s\n", cudaGetErrorString(error));

          		cudaMemcpy(grids->devImage, grids, sizeof(sem_GPUgrids), cudaMemcpyHostToDevice);
          		cudacheck("devImage"); 
        
          		sem_Force_kernel<<< blocksPerGrid, threadsPerBlock >>>(grids->devImage);
          		cudacheck("sem_Force");
          		sem_calculate_A_kernel<<<blocksPerGrid_FEM, threadsPerBlock_FEM>>>(grids->devImage);
          		cudacheck("calculate_A");        
          		sem_surface_tension_kernel<<< blocksPerGrid_node, threadsPerBlock_node >>>(grids->devImage);
          		cudacheck("sem_surface_tension");
          		sem_calculate_Kapa_kernel<<<blocksPerGrid_FEM, threadsPerBlock_FEM>>>(grids->devImage);        
          		sem_calculate_m_kernel<<< blocksPerGrid_node, threadsPerBlock_node >>>(grids->devImage);
        //sem_calculate_K_kernel<<<blocksPerGrid_FEM, threadsPerBlock_FEM>>>(grids->devImage);        
        //sem_calculate_q_kernel<<< blocksPerGrid_node, threadsPerBlock_node >>>(grids->devImage);
        //sem_bending_force_kernel<<<blocksPerGrid_FEM, threadsPerBlock_FEM>>>(grids->devImage);               
        //cudacheck("bending_force");
          		sem_ForceAreaVol_kernel<<<blocksPerGrid_FEM, threadsPerBlock_FEM>>>(grids->devImage);
          		sem_Laplace_Kapa_kernel<<< blocksPerGrid, threadsPerBlock >>>(grids->devImage);
          		cudacheck("sem_Laplace_Kapa");         
          
          //cudaMemcpy(grids, grids->devImage, sizeof(sem_GPUgrids), cudaMemcpyDeviceToHost);
          //cudacheck("devImage to host"); 
		}  
     	}

} //// END: sem_invokeGPUKernel_Force() 

// Release
void fluid_releaseGPUKernel(void *model, void *g)
{
	fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
       int i;
       for (i = 0; i < 19; ++i) {
         cudaFree(&(grids->fIN[i]));
         cudaFree(&(grids->fOUT[i]));
       }
       cudaFree(&(grids->ux));
       cudaFree(&(grids->uy));
       cudaFree(&(grids->uz));
       cudaFree(&(grids->obst));
       cudaFree(&(grids->rho));
       cudaFree(&(grids->Fx));
       cudaFree(&(grids->Fy));
       cudaFree(&(grids->Fz));
       cudaFree(&(grids->vWFbond));
       cudaFree(grids->deviceStruct);
       free(grids);
       cudaThreadExit();
}

///////////End of GPU Fluid ///////////////////////////////////////////////////////////////////


////////// Begin of the GPU for Fiber Network //////////////////////////////////////////////////
////////// Fibre alloc and initial///////
// alloc 
void *fiber_allocGPUKernel(void *model, int maxNodes, int maxLinks,
                           int max_N_conn_at_Node,double dt, double *hostParameters)
{
	tmp_fiber_GPUgrids *fgrids=(tmp_fiber_GPUgrids *)malloc(sizeof(tmp_fiber_GPUgrids));
	fgrids->maxNodes = maxNodes;
	fgrids->maxLinks = maxLinks;
	fgrids->dt = dt;
	fgrids->max_N_conn_at_Node = max_N_conn_at_Node;

        //  allocate memory for fiber nodes //////
        cudaMalloc((void**)&(fgrids->X), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->Y), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->Z), maxNodes * sizeof(double));	

        cudaMalloc((void**)&(fgrids->X0), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->Y0), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->Z0), maxNodes * sizeof(double));	

        cudaMalloc((void**)&(fgrids->V_X), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->V_Y), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->V_Z), maxNodes * sizeof(double));	

        cudaMalloc((void**)&(fgrids->F_X), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->F_Y), maxNodes * sizeof(double));	
        cudaMalloc((void**)&(fgrids->F_Z), maxNodes * sizeof(double));	

        cudaMalloc((void**)&(fgrids->NodeType), maxNodes * sizeof(int));	
        cudaMalloc((void**)&(fgrids->N_Conn_at_Node), maxNodes * sizeof(int));	


        //  allocate memory for fiber link //////
        cudaMallocPitch((void**)&(fgrids->Link_at_Node), &(fgrids->pitchLink_at_Node), max_N_conn_at_Node*sizeof(int), maxNodes); 
        cudaMallocPitch((void**)&(fgrids->lAdjVer), &(fgrids->pitchlAdjVer), 2*sizeof(int),  maxLinks); 

        cudaMalloc((void**)&(fgrids->linkLengths), maxLinks * sizeof(double));	
        cudaMalloc((void**)&(fgrids->linkLengths0), maxLinks * sizeof(double));	
        cudaMalloc((void**)&(fgrids->linkThick), maxLinks * sizeof(double));	

        return fgrids;
}


	


//// INITIAL ///////////////////////////////////////////////////////////////////
  
void fiber_init_GPUKernel(void *model, void *fg, void *NodeType, void *N_Conn_at_Node, 
                           void *Link_at_Node, void *lAdjVer, void *linkLengths,void *linkLengths0, 
                           void *linkThick, void *X,void *V_X,void *X0,  void *F_X, 
                           void *Y,void *Y0, void *V_Y, void *F_Y, void *Z,void *Z0, void *V_Z, void *F_Z )
{
     	tmp_fiber_GPUgrids *fgrids = (tmp_fiber_GPUgrids *)fg;
  
    // memory copy for node
   	cudaMemcpy(fgrids->N_Conn_at_Node, N_Conn_at_Node, fgrids->maxNodes * sizeof(int), cudaMemcpyHostToDevice);
    	cudacheck("N_Conn_at_Node");

   	cudaMemcpy(fgrids->NodeType, NodeType, fgrids->maxNodes * sizeof(int), cudaMemcpyHostToDevice);
    	cudacheck("NodeType");
  	cudaMemcpy(fgrids->X, X, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("X");
  	cudaMemcpy(fgrids->Y, Y, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("Y");
   	cudaMemcpy(fgrids->Z, Z, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("Z");
   
  	cudaMemcpy(fgrids->X0, X0, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("X0");
   	cudaMemcpy(fgrids->Y0, Y0, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("Y0");
   	cudaMemcpy(fgrids->Z0, Z0, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("Z0");

   	cudaMemcpy(fgrids->V_X, V_X, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("V_X");
   	cudaMemcpy(fgrids->V_Y, V_Y, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("V_Y");
   	cudaMemcpy(fgrids->V_Z, V_Z, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("V_Z");
  
   	cudaMemcpy(fgrids->F_X, F_X, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("F_X");
   	cudaMemcpy(fgrids->F_Y, F_Y, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("F_Y");
   	cudaMemcpy(fgrids->F_Z, F_Z, fgrids->maxNodes * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("F_Z");
  // memory copy for the link 

   	cudaMemcpy2D(fgrids->Link_at_Node, fgrids->pitchLink_at_Node, Link_at_Node, fgrids->max_N_conn_at_Node * sizeof(int), fgrids->max_N_conn_at_Node * sizeof(int), fgrids->maxNodes, cudaMemcpyHostToDevice);
   	cudacheck("Link_at_Node");
   	cudaMemcpy2D(fgrids->lAdjVer, fgrids->pitchlAdjVer, lAdjVer, 2 * sizeof(int), 2 * sizeof(int), fgrids->maxLinks, cudaMemcpyHostToDevice);
   	cudacheck("lAdjVer");

   	cudaMemcpy(fgrids->linkLengths, linkLengths, fgrids->maxLinks * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("linkLengths");
   	cudaMemcpy(fgrids->linkLengths0, linkLengths0, fgrids->maxLinks * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("linkLengths0");
   	cudaMemcpy(fgrids->linkThick, linkThick, fgrids->maxLinks * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("linkThick");
  
  
}


// COMPUTE THE FORCE ///////////

void fiber_copy_GPUKernel(void *model, void *fg,  void *X,void *V_X,void *X0,  void *F_X, 
                           void *Y,void *Y0, void *V_Y, void *F_Y, void *Z,void *Z0, void *V_Z, void *F_Z )
{
     	tmp_fiber_GPUgrids *fgrids = (tmp_fiber_GPUgrids *)fg;

	cudaMemcpy(X,fgrids->X, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
  	cudaMemcpy(Y,fgrids->Y, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
   	cudaMemcpy(Z,fgrids->Z, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
    	
	cudaMemcpy(V_X,fgrids->V_X, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
  	cudaMemcpy(V_Y,fgrids->V_Y, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
   	cudaMemcpy(V_Z,fgrids->V_Z, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(F_X,fgrids->F_X, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
  	cudaMemcpy(F_Y,fgrids->F_Y, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
   	cudaMemcpy(F_Z,fgrids->F_Z, fgrids->maxNodes * sizeof(double), cudaMemcpyDeviceToHost);
	




}
// memory copy from device to host 














// memory release 

void fiber_release_GPUKernel(void *model, void *fg)
{	
	tmp_fiber_GPUgrids *fgrids = (tmp_fiber_GPUgrids *)fg;

    	
    	cudaFree(fgrids->NodeType);
    	cudaFree(fgrids->N_Conn_at_Node);
    	cudaFree(fgrids->Link_at_Node);
    	cudaFree(fgrids->lAdjVer);
    	cudaFree(fgrids->linkLengths);
    	cudaFree(fgrids->linkLengths0);
    	cudaFree(fgrids->linkThick);
    
    	cudaFree(fgrids->X);
    	cudaFree(fgrids->V_X);
    	cudaFree(fgrids->F_X);
   
  	cudaFree(fgrids->Y);
    	cudaFree(fgrids->V_Y);
    	cudaFree(fgrids->F_Y);
   
	cudaFree(fgrids->Z);
   	cudaFree(fgrids->V_Z);
    	cudaFree(fgrids->F_Z);

    //cudaFree(grids->cellCenterZ);
    free(fgrids);
    cudaThreadExit();
}



//////////////////////////END of GPU for the Fiber Network /////////////////////////////////////




/////////////////////////BEGIN OF GPU for the Platelets ///////////////////////////////////////

  ///BEGIN: SEM ALLOC AND INITIAL ///
  // Allocation
void *sem_allocGPUKernel(void *model, int maxCells, int maxElements, int SurfElem, 
                           int newnode, int numReceptorsPerElem, 
                           float dt, double S0_all, float *hostParameters)
{
   	sem_GPUgrids *grids = (sem_GPUgrids *)malloc(sizeof(sem_GPUgrids));

   // Save parameters
   	grids->maxCells = maxCells;
   	grids->maxElements = maxElements;
   	grids->dt = dt;
   	grids->SurfElem = SurfElem;
	grids->newnodeNum = newnode;
   	grids->numReceptorsPerElem = numReceptorsPerElem;
   	grids->S0_all = S0_all;
   	grids->iextent = make_cudaExtent(grids->SurfElem*sizeof(int), grids->numReceptorsPerElem, 3);
   	grids->dextent = make_cudaExtent(grids->SurfElem*sizeof(double), grids->numReceptorsPerElem, 2);
   
  // grids->ReversalPeriod = ReversalPeriod;
   // Allocate device memory
   	cudaMalloc((void**)&(grids->devImage),sizeof(sem_GPUgrids));
   	size_t pitch_test;
   // cells and elements
   	cudaMalloc((void**)&(grids->numOfElements), maxCells * sizeof(int));
   	cudaMalloc((void**)&(grids->node_nbrElemNum), newnode * sizeof(int));
   	cudaMalloc((void**)&(grids->S0), SurfElem * sizeof(double));
   	cudaMalloc((void**)&(grids->S), SurfElem * sizeof(double));
   
   	cudaMallocPitch((void**)&(grids->elementType), &(grids->pitch), maxCells * sizeof(int), maxElements);
   	pitch_test = grids->pitch;
   	cudaMallocPitch((void**)&(grids->triElem), &(grids->pitch), 6 * sizeof(int), SurfElem); 
   	pitchcheck(pitch_test, grids->pitch, 1);
   	cudaMallocPitch((void**)&(grids->receptor_r1), &(grids->pitch), numReceptorsPerElem * sizeof(float), SurfElem); 
   	pitchcheck(pitch_test, grids->pitch, 2);
   	cudaMallocPitch((void**)&(grids->receptor_r2), &(grids->pitch), numReceptorsPerElem * sizeof(float), SurfElem);
   	pitchcheck(pitch_test, grids->pitch, 3);
   //cudaMallocPitch((void**)&(grids->rho), &(grids->pitch), maxCells * sizeof(double), maxElements);
   //pitchcheck(pitch_test, grids->pitch);
   	cudaMallocPitch((void**)&(grids->X_Ref), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 4);
   	cudaMallocPitch((void**)&(grids->X), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 5);
   	cudaMallocPitch((void**)&(grids->V_X), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 6);
   	cudaMallocPitch((void**)&(grids->F_X), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 7);
   	cudaMallocPitch((void**)&(grids->Y_Ref), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 8);
   	cudaMallocPitch((void**)&(grids->Y), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 9);
   	cudaMallocPitch((void**)&(grids->V_Y), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 10);
   	cudaMallocPitch((void**)&(grids->RY), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 11);
   	cudaMallocPitch((void**)&(grids->F_Y), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 12);
   	cudaMallocPitch((void**)&(grids->Z_Ref), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 13);
   	cudaMallocPitch((void**)&(grids->Z), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 14);
   	cudaMallocPitch((void**)&(grids->V_Z), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 15);
   	cudaMallocPitch((void**)&(grids->F_Z), &(grids->pitch), maxCells * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 16);
   	cudaMallocPitch((void**)&(grids->node_share_Elem), &(grids->pitch), 10 * sizeof(int), newnode);
   	pitchcheck(pitch_test, grids->pitch, 17);
   	cudaMallocPitch((void**)&(grids->N), &(grids->pitch), 3 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 18);
   	cudaMallocPitch((void**)&(grids->n), &(grids->pitch), 3 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 19);
   	cudaMallocPitch((void**)&(grids->q), &(grids->pitch), 3 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 20);
   	cudaMallocPitch((void**)&(grids->A), &(grids->pitch), 9 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 21);
   	cudaMallocPitch((void**)&(grids->tau), &(grids->pitch), 9 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch,22);
   	cudaMallocPitch((void**)&(grids->Kapa), &(grids->pitch), 9 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 23);
   	cudaMallocPitch((void**)&(grids->km), &(grids->pitch),sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 24);
   	cudaMallocPitch((void**)&(grids->K), &(grids->pitch), 27 * sizeof(double), newnode);
   	pitchcheck(pitch_test, grids->pitch, 25); 
   	cudaMallocPitch((void**)&(grids->nelem), &(grids->pitch), 3 * sizeof(double), SurfElem);
   	pitchcheck(pitch_test, grids->pitch, 26); 
   	cudaMallocPitch((void**)&(grids->Laplace_km), &(grids->pitch),sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 27);
   	cudaMallocPitch((void**)&(grids->node_nbrNodes), &(grids->pitch), 10 * sizeof(int), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 28);
   	cudaMallocPitch((void**)&(grids->bondLengths), &(grids->bondpitch), maxElements * sizeof(double), maxElements);
   	pitchcheck(pitch_test, grids->pitch, 29);

   	cudaMalloc3D(&(grids->receptBond), grids->iextent);
   	cudaMalloc3D(&(grids->randNum), grids->dextent);
   	cudacheck("sem_alloc");
   

  // Reversal Clock Values of cells
//   cudaMalloc((void**)&(grids->ClockValue), maxCells * sizeof(int));
 //  cudaMalloc((void**)&(grids->SlimeDir), maxCells * sizeof(int));

   // cell centers
  // cudaMalloc((void**)&(grids->cellCenterX), maxCells * sizeof(float));
  // cudaMalloc((void**)&(grids->cellCenterY), maxCells * sizeof(float));
  // cudaMalloc((void**)&(grids->cellCenterZ), maxCells * sizeof(float));

   // copy parameters
   	cudaMemcpyToSymbol(model_Parameters, hostParameters, 100 * sizeof(float), 0, cudaMemcpyHostToDevice);
 
   //  Allocate Memory for RNG states
   /* Allocate space for prng states on device */
   	cudaMalloc((void **)&(grids->devState), SurfElem * numReceptorsPerElem *sizeof(curandState));

   	return grids;
}


// Initialization
void sem_initGPUKernel(void *model, void *g, int numOfCells, int *numOfElements, int SurfElem, int numReceptorsPerElem,
                          void *hostX_Ref, void *hostY_Ref, void *hostZ_Ref,
                          void *hostX, void *hostY, void *hostRY, void *hostZ, void *hostVX, void *hostVY, void *hostVZ,
                          void *hostFX, void *hostFY, void *hostFZ, void *hostType, void *hostBonds, 
                          void *triElem, void *receptor_r1, void *receptor_r2, void *node_share_Elem, void *N,
                          void * node_nbrElemNum, void * node_nbr_nodes, void *S0, double V0, void *receptBond, void *randNum)//transfer function  // add hostV_X...hostF_X
{


    	sem_GPUgrids *grids = (sem_GPUgrids *)g;

   // Begin RNG Stuff
   /* Setup prng states */
   // printf("SurfElem = %d, numReceptors = %d\n", SurfElem, numReceptorsPerElem);
     // setup_RNG_kernel<<<SurfElem, numReceptorsPerElem>>>(grids->devState);
   // RNG finished
   // cudacheck("setup_RNG_kernel");
    	grids->numOfCells = numOfCells;
    	grids->S_all = 0;
    	grids->V0 = V0;
    	grids->V = 0;
    	grids->totalBond = 1;
   // grids->SurfElem = SurfElem;
   // grids->numReceptorsPerElem = numReceptorsPerElem;

    // Copy host memory to device memory
    	cudaMemcpy(grids->numOfElements, numOfElements, grids->maxCells * sizeof(int), cudaMemcpyHostToDevice);
    	cudacheck("numOfElements"); 
    	cudaMemcpy(grids->node_nbrElemNum, node_nbrElemNum, grids->newnodeNum * sizeof(int), cudaMemcpyHostToDevice);
    	cudacheck("Neighbor Elements Number"); 
    	cudaMemcpy(grids->S0, S0, grids->SurfElem * sizeof(double), cudaMemcpyHostToDevice);
    	cudacheck("Area"); 
   // cudaMemcpy(grids->ClockValue, ClockValue, grids->maxCells * sizeof(int), cudaMemcpyHostToDevice);
   // cudaMemcpy(grids->SlimeDir, SlimeDir, grids->maxCells * sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(grids->devImage, grids, sizeof(sem_GPUgrids), cudaMemcpyHostToDevice);
   	cudacheck("devImage"); 

    	cudaMemcpy2D(grids->elementType, grids->pitch, hostType, grids->maxCells * sizeof(int), grids->maxCells * sizeof(int), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("elementType"); 
    	cudaMemcpy2D(grids->bondLengths, grids->bondpitch, hostBonds, grids->maxElements * sizeof(double), grids->maxElements * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("bondLengths"); 
    	cudaMemcpy2D(grids->triElem, grids->pitch, triElem, 6 * sizeof(int), 6 * sizeof(int), grids->SurfElem, cudaMemcpyHostToDevice);
   	cudacheck("triElem"); 
    	cudaMemcpy2D(grids->receptor_r1, grids->pitch, receptor_r1, numReceptorsPerElem * sizeof(float), numReceptorsPerElem * sizeof(float), SurfElem, cudaMemcpyHostToDevice);
   	cudacheck("receptor_r1"); 
    	cudaMemcpy2D(grids->receptor_r2, grids->pitch, receptor_r2, numReceptorsPerElem * sizeof(float), numReceptorsPerElem * sizeof(float), SurfElem, cudaMemcpyHostToDevice);
   	cudacheck("receptor_r2"); 
    	cudaMemcpy2D(grids->X_Ref, grids->pitch, hostX_Ref, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("X_Ref"); 
    	cudaMemcpy2D(grids->Y_Ref, grids->pitch, hostY_Ref, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("Y_Ref"); 
    	cudaMemcpy2D(grids->Z_Ref, grids->pitch, hostZ_Ref, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("Z_Ref"); 
    	cudaMemcpy2D(grids->X, grids->pitch, hostX, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("X"); 
    	cudaMemcpy2D(grids->Y, grids->pitch, hostY, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("Y"); 
    	cudaMemcpy2D(grids->RY, grids->pitch, hostY, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("RY"); 
    	cudaMemcpy2D(grids->Z, grids->pitch, hostZ, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("Z"); 
   // added Memcpy for force and velocity
    	cudaMemcpy2D(grids->F_X, grids->pitch, hostFX, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("FX"); 
    	cudaMemcpy2D(grids->F_Y, grids->pitch, hostFY, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("FY"); 
    	cudaMemcpy2D(grids->F_Z, grids->pitch, hostFZ, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   	cudacheck("FZ"); 
    	cudaMemcpy2D(grids->V_X, grids->pitch, hostVX, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
    	cudacheck("VX"); 
    	cudaMemcpy2D(grids->V_Y, grids->pitch, hostVY, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
    	cudacheck("VY"); 
    	cudaMemcpy2D(grids->V_Z, grids->pitch, hostVZ, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
    	cudacheck("VZ");
    	cudaMemcpy2D(grids->node_share_Elem, grids->pitch, node_share_Elem, 10 * sizeof(int), 10 * sizeof(int), grids->newnodeNum, cudaMemcpyHostToDevice);
    	cudacheck("node_share_Elem");
    	cudaMemcpy2D(grids->node_nbrNodes, grids->pitch, node_nbr_nodes, 10 * sizeof(int), 10 * sizeof(int), grids->maxElements, cudaMemcpyHostToDevice);
    	cudacheck("node_nbr_nodes");
    	cudaMemcpy2D(grids->N, grids->pitch, N, 3 * sizeof(double), 3 * sizeof(double), grids->newnodeNum, cudaMemcpyHostToDevice);
    	cudacheck("N");
   // cudaMemcpy2D(grids->PreV_X, grids->pitch, hostPreVX, grids->maxCells * sizeof(double), grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyHostToDevice);
   // cudaMemcpy2D(grids->PreV_Y, grids->pitch, hostPreVY, grids->maxCells * sizeof(float), grids->maxCells * sizeof(float), grids->maxElements, cudaMemcpyHostToDevice);
   // cudaMemcpy2D(grids->PreV_Z, grids->pitch, hostPreVZ, grids->maxCells * sizeof(float), grids->maxCells * sizeof(float), grids->maxElements, cudaMemcpyHostToDevice);
   //add cudaMec.. for VX, VY, VZ, FX, FY, FZ
  // cudaMemset2D((void**)&(grids->receptBond), grids->pitch, 0,  numReceptorsPerElem * sizeof(float), SurfElem); 
   // cudaError_t error = cudaMemset3D(grids->receptBond, -1, grids->iextent);
   // if (error) printf("cudaERROR Memset receptBond: %s\n", cudaGetErrorString(error));
    
    	cudaError_t error = cudaMemset2D(grids->km, grids->pitch, 0, grids->maxCells * sizeof(double), grids->newnodeNum);
    	if (error) printf("cudaERROR Memset km: %s\n", cudaGetErrorString(error));
    
    	cudaMemcpy3DParms p3d = {0};
    	p3d.extent = grids->iextent;
    	p3d.kind = cudaMemcpyHostToDevice;
      
    	p3d.srcPtr = make_cudaPitchedPtr(receptBond, grids->SurfElem*sizeof(int), grids->SurfElem*sizeof(int), grids->numReceptorsPerElem);
    	p3d.dstPtr = grids->receptBond;
    	error = cudaMemcpy3D(&p3d);
    	if (error) printf("cudaERROR receptBond: %s\n", cudaGetErrorString(error));
    
    	p3d.extent = grids->dextent;
    	p3d.kind = cudaMemcpyHostToDevice;
      
    	p3d.srcPtr = make_cudaPitchedPtr(randNum, grids->SurfElem*sizeof(double), grids->SurfElem*sizeof(double), grids->numReceptorsPerElem);
    	p3d.dstPtr = grids->randNum;
    	error = cudaMemcpy3D(&p3d);
    	if (error) printf("cudaERROR randNum: %s\n", cudaGetErrorString(error));
      
    	cudacheck("sem_init"); 
}
///END SEM ALLOC AND INITIAL ///

 

void sem_copyGPUKernel(void *model, void *g, void *hostX, void *hostY, void *hostRY, void *hostZ, 
                          void *hostVX, void *hostVY, void *hostVZ, 
                          void *hostFX, void *hostFY, void *hostFZ, int timeSteps)
{
    // Copy result to host memory
    	sem_GPUgrids *grids = (sem_GPUgrids *)g;
    	cudaMemcpy2D(hostX, grids->maxCells * sizeof(double), grids->X, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostY, grids->maxCells * sizeof(double), grids->Y, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostRY, grids->maxCells * sizeof(double), grids->RY, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostZ, grids->maxCells * sizeof(double), grids->Z, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);

    	cudaMemcpy2D(hostVX, grids->maxCells * sizeof(double), grids->V_X, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostVY, grids->maxCells * sizeof(double), grids->V_Y, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostVZ, grids->maxCells * sizeof(double), grids->V_Z, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);

    	cudaMemcpy2D(hostFX, grids->maxCells * sizeof(double), grids->F_X, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostFY, grids->maxCells * sizeof(double), grids->F_Y, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);
    	cudaMemcpy2D(hostFZ, grids->maxCells * sizeof(double), grids->F_Z, grids->pitch, grids->maxCells * sizeof(double), grids->maxElements, cudaMemcpyDeviceToHost);

//    cudaMemcpy(ClockValue, grids->ClockValue, grids->maxCells * sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(SlimeDir  , grids->SlimeDir  , grids->maxCells * sizeof(int), cudaMemcpyDeviceToHost);
}


   // Release
   
void sem_releaseGPUKernel(void *model, void *g)
{
    	sem_GPUgrids *grids = (sem_GPUgrids *)g;
    	cudaFree(grids->numOfElements);
    	cudaFree(grids->elementType);
    	cudaFree(grids->bondLengths);
    	cudaFree(grids->triElem);
    	cudaFree(grids->receptor_r1);
    	cudaFree(grids->receptor_r2);
    	cudaFree(&(grids->receptBond));
    
    	cudaFree(grids->X);
    	cudaFree(grids->V_X);
   // cudaFree(grids->PreV_X);
    	cudaFree(grids->F_X);
    	cudaFree(grids->Y);
    	cudaFree(grids->V_Y);
    	cudaFree(grids->RY);
    	cudaFree(grids->F_Y);
    	cudaFree(grids->Z);
    	cudaFree(grids->V_Z);
   // cudaFree(grids->PreV_Z);
    	cudaFree(grids->F_Z);
    	cudaFree(grids->devImage);
    	cudaFree(grids->node_share_Elem);
    	cudaFree(grids->N);
    //cudaFree(grids->cellCenterZ);
    	free(grids);
    	cudaThreadExit();
}

} /// END::: extern "C" {

//////////////////////////////////End of GPU for the Platelets///////////////////////////////////
