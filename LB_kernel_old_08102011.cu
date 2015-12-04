/*
 lb_kernel.cu
 
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

// obst_r = ly/10+1;  % radius of the cylinder
#define OBST_R fluid_parameters[0]
// maximum velocity of Poiseuille inflow
#define UMAX fluid_parameters[1]
// vertical lid velocity 
#define VLID fluid_parameters[2]
// Reynolds number 
#define RE fluid_parameters[3]
// kinematic viscosity 
#define NU fluid_parameters[4]
// relaxation parameter 
#define OMEGA fluid_parameters[5]
// fluid density
#define FLUID_RHO fluid_parameters[6]

__constant__ float t[] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
__constant__ float cx[] = {0, 1, 0, 0, -1,  0,  0,  1, -1, -1,  1, 1, -1, -1,  1, 0,  0,  0,  0};
__constant__ float cy[] = {0, 0, 1, 0,  0, -1,  0,  1,  1, -1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
__constant__ float cz[] = {0, 0, 0, 1,  0,  0, -1,  0,  0,  0,  0, 1,  1, -1, -1, 1,  1, -1, -1};
__constant__ int   bb[] = {0, 4, 5, 6, 1, 2, 3, 9, 10, 7, 8, 13, 14, 11, 12, 17, 18, 15, 16};

//// TMP
// float              newcx[] = {0, 1, 0, 0, -1,  0,  0,  1, -1, -1,  1, 1, -1, -1,  1, 0,  0,  0,  0};

__device__ float
get3d_value(void *devPtr, size_t pitch, size_t slicePitch, int i, int j, int k)
{
  char *slice = (char *)devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  return row[i];
}

__global__ void
fluid3d_boundary_kernel(void *g, int aBank)
{
  //fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
   

  // check thread in boundary
  if (i >= fluid_grid->width) return;
  if (j >= fluid_grid->height) return;
  if (k >= fluid_grid->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  float ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0;
  // float tmp_ux, tmp_uy, tmp_uz, tmp_rho;
  float Cs = fluid_grid->dx/fluid_grid->dt;

  for (d = 0; d < 19; ++d) {
    rho += get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
    // ux += cx[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
    // uy += cy[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
    // uz += cz[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
    ux += cx[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uy += cy[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uz += cz[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
  }
  char *devPtr = (char *)fluid_grid->rho.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  row[i] = rho;

  devPtr = (char *)fluid_grid->ux.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = ux / rho;
  //row[i] = ux;

  devPtr = (char *)fluid_grid->uy.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = uy / rho;
  //row[i] = uy;

  devPtr = (char *)fluid_grid->uz.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = uz / rho;
  //row[i] = uz;

#if 1
  if (k == 0) {

     int use_tmp_fix = 0;
     if(use_tmp_fix == 1)
     {
         /****
         float LY = ((float)fluid_grid->height) - 2.0;
         float LX = ((float)fluid_grid->width) - 2.0;
         float y_phys = ((float)j) - 0.5;
         float x_phys = ((float)i) - 0.5;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         tmp_ux = row[i];

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         tmp_uy = row[i];

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
         tmp_uz = row[i];

         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 1.0;
         tmp_rho = row[i];

         for (d = 0; d < 19; ++d) 
         {
             devPtr = (char *)fluid_grid->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = tmp_rho*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs)); 
         }
         ***/
         //// TEMP::: 08052011. simple fix. copy flow velocity and density from [i][j][k-1] to [i][j][k];
         ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0;
         for (d = 0; d < 19; ++d) {
             rho += get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k+1);
             ux += cx[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k+1)*Cs;
             uy += cy[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k+1)*Cs;
             uz += cz[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k+1)*Cs;
         }
         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = rho;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = ux/rho;

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = uy/rho;

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = uz/rho;
         //// END TEMP

         //// TEMP::: 08052011. simple fix. copy particle density function from [i][j][k-1] to [i][j][k];
         for (d = 0; d < 19; ++d) {
             devPtr = (char *)fluid_grid->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             row[i] = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k+1);
         }
         //// END::: TEMP::: 08052011. simple fix. copy particle density function from [i][j][k-1] to [i][j][k];
     } /// END::: if(use_tmp_fix == 1)
     else
     {
         // Implementation of on-site velocity boundary conditions for D2Q19 lattice Boltzmann simulations
         // M. Hecht, J. Harting. J. of Stat. Mech.: Theory and Experiment. 2010

         // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
         // % Inlet: Poiseuille profile
         //
         float LY = ((float)fluid_grid->height) - 2.0;
         float LX = ((float)fluid_grid->width) - 2.0;
         float y_phys = ((float)j) - 0.5;
         float x_phys = ((float)i) - 0.5;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);

         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = 1.0 / (1.0 - get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k))
           * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
    	  + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
	  + 2.0 * (get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)));
         ***/
         ///// NEW 08052011, with units
         row[i] = 1.0 / (1.0 - get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/Cs)
           * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)));
         /////END::: NEW 08052011, with units

         // % MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
         //
         /*** Original Scott's implementation, non-dimensionalized
         float Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
        			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
    			+ get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k);

         float Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k);
         ****/ 
         /////NEW 08052011, with units
         float Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)/Cs;

         float Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)/Cs;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[3].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k);
         ***/ 
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/Cs;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[11].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) - Nzx;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/Cs 
               - Nzx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)fluid_grid->fIN[12].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) + Nzx;
         ***/ 
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/Cs 
             + Nzx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[15].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) 
              - Nzy;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/Cs 
              - Nzy;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[16].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) + Nzy;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/Cs 
              + Nzy;
         /////END::: NEW 08052011, with units
         }
   } /// end :::: if(k == 0)

   if ((k == (fluid_grid->depth - 1))
       && (i > 0) && (i < (fluid_grid->width - 1))
       && (j > 0) && (j < (fluid_grid->height - 1))) {

     // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
     // % Outlet: Constant pressure
     //
     /*** original implementation
     devPtr = (char *)fluid_grid->rho.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 1.0;

     devPtr = (char *)fluid_grid->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)fluid_grid->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)fluid_grid->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = -1.0 + 1.0 / get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
	  + 2.0 * (get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)));
     ***/

         /*** Impose flow as inlet side
         float LY = ((float)fluid_grid->height) - 2.0;
         float LX = ((float)fluid_grid->width) - 2.0;
         float y_phys = ((float)j) - 0.5;
         float x_phys = ((float)i) - 0.5;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         tmp_ux = row[i];

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         tmp_uy = row[i];

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
         tmp_uz = row[i];

         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 1.0;
         tmp_rho = row[i];

         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)fluid_grid->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = tmp_rho*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
         ***/
     //// TEMP::: 08052011. simple fix. copy flow velocity and density from [i][j][k-1] to [i][j][k];
     ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0;
     for (d = 0; d < 19; ++d) {
         rho += get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k-1);
         ux += cx[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k-1)*Cs;
         uy += cy[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k-1)*Cs;
         uz += cz[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k-1)*Cs;
     }
     devPtr = (char *)fluid_grid->rho.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = rho;

     devPtr = (char *)fluid_grid->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = ux/rho;

     devPtr = (char *)fluid_grid->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = uy/rho;

     devPtr = (char *)fluid_grid->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = uz/rho;
     //// END TEMP

     //// TEMP::: 08052011. simple fix. copy particle density function from [i][j][k-1] to [i][j][k];
     for (d = 0; d < 19; ++d) {
         devPtr = (char *)fluid_grid->fIN[d].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k-1);
     }
     //// END::: TEMP::: 08052011. simple fix. copy particle density function from [i][j][k-1] to [i][j][k];

     // % MICROSCOPIC BOUNDARY CONDITIONS: OUTLET (Zou/He BC)
     //
     // fIn(4,out,col) = fIn(2,out,col) - 2/3*rho(:,out,col).*ux(:,out,col); 
     // fIn(8,out,col) = fIn(6,out,col) + 1/2*(fIn(3,out,col)-fIn(5,out,col)) ... 
     //                                 - 1/2*rho(:,out,col).*uy(:,out,col) ...
     //                                 - 1/6*rho(:,out,col).*ux(:,out,col); 
     // fIn(7,out,col) = fIn(9,out,col) + 1/2*(fIn(5,out,col)-fIn(3,out,col)) ... 
     //                                 + 1/2*rho(:,out,col).*uy(:,out,col) ...
     //                                 - 1/6*rho(:,out,col).*ux(:,out,col); 
     /////////////////////////////////////////////////////////////////////////////////////////////
     // 08052011: 1. There seems to be some dimensionality issue with the pressure boundary condition.
     //           2. If the inflow side is the velocity boundary condition (the velocity is specified),
     //              the outflow side should be using the stress-free boundary condition. 
     //              See: Lattice Boltzmann Simulations of Droplet Behaviour in Microfluidic Devices
     //              Haihu Liu and Yonghao Zhang.                

     /*** Original implementation
     float Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k);
     float Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k);

     devPtr = (char *)fluid_grid->fIN[6].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k);

     devPtr = (char *)fluid_grid->fIN[13].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) + Nzx;

     devPtr = (char *)fluid_grid->fIN[14].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) - Nzx;

     devPtr = (char *)fluid_grid->fIN[17].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) + Nzy;

     devPtr = (char *)fluid_grid->fIN[18].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) - Nzy;
     ***/
   }
#endif
}

__global__ void
fluid3d_in_out_flow_boundary_kernel(void *g, int aBank)
{
  //fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
   
  // check thread in boundary
  if (i >= fluid_grid->width) return;
  if (j >= fluid_grid->height) return;
  if (k >= fluid_grid->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  char *devPtr = (char *)fluid_grid->rho.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);

  // float ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0; 
  float tmp_ux, tmp_uy, tmp_uz, cu;
  float Cs = fluid_grid->dx/fluid_grid->dt;
  float Nzx = 0.0, Nzy = 0.0, LY, LX, y_phys, x_phys;

#if 1
  //// Inflow boundary condition at depth = 0.
  if ((k == 0) && (i > 0) && (i < (fluid_grid->width - 1))
       && (j > 0) && (j < (fluid_grid->height - 1)) ) 
  {
     // Implementation of on-site velocity boundary conditions for D2Q19 lattice Boltzmann simulations
     // M. Hecht, J. Harting. J. of Stat. Mech.: Theory and Experiment. 2010
     // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
     // % Inlet: Poiseuille profile
     /// due to singularity for nodes which are directly linked to solid boundary nodes. Special treatment is needed.
     /// In fact, need to follow "M. Hecht, J. Harting." and "Y.T. Feng, K. Han and D.R.J. Owen. 
     /// Coupled lattice Boltzmann method and discrete element modeling of particle transport in 
     /// turbulent fluid flows: Computational issue. Int. J.  Numer. Meth. Engng. 72:1111-1134, 2007" to derive the boundary condition. 
     /// Here, we try a simple fix: use initial condition to compute f again.
     LY = ((float)fluid_grid->height) - 2.0;
     LX = ((float)fluid_grid->width) - 2.0;
     y_phys = ((float)j) - 0.5;
     x_phys = ((float)i) - 0.5;

     devPtr = (char *)fluid_grid->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_ux = 0.0;

     devPtr = (char *)fluid_grid->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_uy = 0.0;

     devPtr = (char *)fluid_grid->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
     // row[i] = UMAX;
     tmp_uz = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);

     if( (j == 1) || (j == fluid_grid->height-2) || 
         (i == 1) || (i == fluid_grid->width-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = FLUID_RHO;
         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)fluid_grid->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = FLUID_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
     }
     else
     {
         /***
         LY = ((float)fluid_grid->height) - 2.0;
         LX = ((float)fluid_grid->width) - 2.0;
         y_phys = ((float)j) - 0.5;
         x_phys = ((float)i) - 0.5;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         // row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
         row[i] = UMAX;
         ****/ 

         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);

         ///// NEW 08052011, with units.
         ///// part of fIN entering the domain is not correct at this point because of fluid3d_stream_kernel_2()
         ///// assumes periodicity on boundary conditions. 
         ///// On inflow side (z = 0) in Z (depth) direction: This means e11, e15,e3, e16,e12 need to be updated.
         row[i] = 1.0 / (1.0 - get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/Cs)
           * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)));
         /////END::: NEW 08052011, with units

         // % MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
         //
         /////NEW 08052011, with units
         Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)/Cs;

         Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)/Cs;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[3].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k);
         ***/ 
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[6].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/(Cs);
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[11].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) - Nzx;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[13].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
               - Nzx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)fluid_grid->fIN[12].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) + Nzx;
         ***/ 
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[14].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
             + Nzx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[15].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) 
              - Nzy;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              - Nzy;
         /////END::: NEW 08052011, with units

         devPtr = (char *)fluid_grid->fIN[16].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         /*** Original Scott's implementation, non-dimensionalized
         row[i] = get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) + Nzy;
         ***/
         /////NEW 08052011, with units
         row[i] = get3d_value(fluid_grid->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              + Nzy;
         /////END::: NEW 08052011, with units
     }
  } /// end :::: if(k == 0)

  //// Outflow boundary condition at depth = grid->depth-1.
  if ((k == (fluid_grid->depth - 1))
       && (i > 0) && (i < (fluid_grid->width - 1))
       && (j > 0) && (j < (fluid_grid->height - 1))) {

     // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
     // % Outlet: Constant pressure
     //
     /*** original implementation
     devPtr = (char *)fluid_grid->rho.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 1.0;

     devPtr = (char *)fluid_grid->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)fluid_grid->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)fluid_grid->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = -1.0 + 1.0 / get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
	  + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
	  + 2.0 * (get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
		   + get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)));
     ***/
     /*** Outflow: Impose flow profile as inlet side ***/
         // Implementation of on-site velocity boundary conditions for D2Q19 lattice Boltzmann simulations
         // M. Hecht, J. Harting. J. of Stat. Mech.: Theory and Experiment. 2010

     LY = ((float)fluid_grid->height) - 2.0;
     LX = ((float)fluid_grid->width) - 2.0;
     y_phys = ((float)j) - 0.5;
     x_phys = ((float)i) - 0.5;

     devPtr = (char *)fluid_grid->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_ux = 0.0;

     devPtr = (char *)fluid_grid->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_uy = 0.0;

     devPtr = (char *)fluid_grid->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
     // row[i] = UMAX;
     tmp_uz = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);

     if( (j == 1) || (j == fluid_grid->height-2) ||
         (i == 1) || (i == fluid_grid->width-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = FLUID_RHO;

         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)fluid_grid->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = FLUID_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
     }
     else 
     {
         /***
         LY = ((float)fluid_grid->height) - 2.0;
         LX = ((float)fluid_grid->width) - 2.0;
         y_phys = ((float)j) - 0.5;
         x_phys = ((float)i) - 0.5;

         devPtr = (char *)fluid_grid->ux.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         // tmp_ux = row[i];

         devPtr = (char *)fluid_grid->uy.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = 0.0;
         // tmp_uy = row[i];

         devPtr = (char *)fluid_grid->uz.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         // row[i] = 16.0 * UMAX / (LY * LY * LX * LX) * (y_phys * LY - y_phys * y_phys) * (x_phys * LX - x_phys * x_phys);
         row[i] = UMAX;
         // tmp_uz = row[i];
         ****/

         devPtr = (char *)fluid_grid->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);

         row[i] = 1.0 / (1.0 + get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/Cs)
           * (get3d_value(fluid_grid->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)));

         ///////////////////////////////////////////
         Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) *
                         get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)/Cs;

         Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) *
                         get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)/Cs;
         //////////////////////
         devPtr = (char *)fluid_grid->fIN[6].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
           - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) *
                     get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k)/(Cs);

         /////////////////////
         devPtr = (char *)fluid_grid->fIN[13].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (-1.0*get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/(Cs)
               + Nzx;

         /////////////////////
         devPtr = (char *)fluid_grid->fIN[14].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (-1.0*get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k))/(Cs)
             - Nzx;

         /////////////////////
         devPtr = (char *)fluid_grid->fIN[17].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (-1.0*get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/(Cs)
              + Nzy;

         /////////////////////
         devPtr = (char *)fluid_grid->fIN[18].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
           * (-1.0*get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k))/(Cs)
              - Nzy;
     }

     // % MICROSCOPIC BOUNDARY CONDITIONS: OUTLET (Zou/He BC)
     //
     // fIn(4,out,col) = fIn(2,out,col) - 2/3*rho(:,out,col).*ux(:,out,col); 
     // fIn(8,out,col) = fIn(6,out,col) + 1/2*(fIn(3,out,col)-fIn(5,out,col)) ... 
     //                                 - 1/2*rho(:,out,col).*uy(:,out,col) ...
     //                                 - 1/6*rho(:,out,col).*ux(:,out,col); 
     // fIn(7,out,col) = fIn(9,out,col) + 1/2*(fIn(5,out,col)-fIn(3,out,col)) ... 
     //                                 + 1/2*rho(:,out,col).*uy(:,out,col) ...
     //                                 - 1/6*rho(:,out,col).*ux(:,out,col); 
     /////////////////////////////////////////////////////////////////////////////////////////////
     // 08052011: 1. There seems to be some dimensionality issue with the pressure boundary condition.
     //           2. If the inflow side is the velocity boundary condition (the velocity is specified),
     //              the outflow side should be using the stress-free boundary condition. 
     //              See: Lattice Boltzmann Simulations of Droplet Behaviour in Microfluidic Devices
     //              Haihu Liu and Yonghao Zhang.                

     /*** Original implementation
     float Nzx = 0.5 * (get3d_value(fluid_grid->fIN[1].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[4].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k))
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k);
     float Nzy = 0.5 * (get3d_value(fluid_grid->fIN[2].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[7].ptr, pitch, slicePitch, i, j, k)
			+ get3d_value(fluid_grid->fIN[8].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[5].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[9].ptr, pitch, slicePitch, i, j, k)
			- get3d_value(fluid_grid->fIN[10].ptr, pitch, slicePitch, i, j, k))
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k);

     devPtr = (char *)fluid_grid->fIN[6].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[3].ptr, pitch, slicePitch, i, j, k)
       - 1.0 / 3.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k) * get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k);

     devPtr = (char *)fluid_grid->fIN[13].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[11].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) + Nzx;

     devPtr = (char *)fluid_grid->fIN[14].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[12].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k)) - Nzx;

     devPtr = (char *)fluid_grid->fIN[17].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[15].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) + Nzy;

     devPtr = (char *)fluid_grid->fIN[18].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = get3d_value(fluid_grid->fIN[16].ptr, pitch, slicePitch, i, j, k)
       + 1.0 / 6.0 * get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k)
       * (-get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k)) - Nzy;
     ***/
  } /// END:::: if ((k == (fluid_grid->depth - 1))
#endif
}



__global__ void
fluid3d_collision_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  float Cs = fluid_grid->dx/fluid_grid->dt, Cs2, Cs4;
  Cs2 = Cs*Cs;
  Cs4 = Cs2*Cs2;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  float ux = get3d_value(fluid_grid->ux.ptr, pitch, slicePitch, i, j, k);
  float uy = get3d_value(fluid_grid->uy.ptr, pitch, slicePitch, i, j, k);
  float uz = get3d_value(fluid_grid->uz.ptr, pitch, slicePitch, i, j, k);
  float rho = get3d_value(fluid_grid->rho.ptr, pitch, slicePitch, i, j, k);

  for (d = 0; d < 19; ++d) {
     // float cu = 3.0 * (cx[d] * ux + cy[d] * uy + cz[d] * uz);
     float cu = 3.0 * (cx[d] * ux *Cs + cy[d] * uy  *Cs + cz[d] * uz *Cs);
     // float fEQ = rho * t[d] * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
     float fEQ = rho * t[d] * (1.0 + cu/(Cs2) + 0.5 * cu * cu/(Cs4) - 1.5 * (ux * ux + uy * uy + uz * uz)/(Cs2));
     float fIN = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);

     char *devPtr = (char *)fluid_grid->fOUT[d].ptr;
     char *slice = devPtr + k * slicePitch;
     float *row = (float *)(slice + j * pitch);
     // row[i] = fIN - (1.0/OMEGA) * (fIN - fEQ);
     row[i] = fIN - (fluid_grid->dt/OMEGA) * (fIN - fEQ);
  }

#if 1
  // MICROSCOPIC BOUNDARY CONDITIONS: OBSTACLES (bounce-back)
  //
  char *devPtr = (char *)fluid_grid->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i]) {
    for (d = 0; d < 19; ++d) {
      devPtr = (char *)fluid_grid->fOUT[d].ptr;
      slice = devPtr + k * slicePitch;
      row = (float *)(slice + j * pitch);
      row[i] = get3d_value(fluid_grid->fIN[bb[d]].ptr, pitch, slicePitch, i, j, k);
    }
  }
#endif
}

///// fluid3d_obst_bounce_back_kernel() do bounce-back (reverse states only)
///// for all solid nodes. 
///// For solid nodes, the bounce-back states are saved in
///// both fIN and fOUT. fOUT will be used in fluid3d_obst_stream_kernel()
///// to stream the state from solid nodes back to the connected fluid nodes. 
///// fluid3d_obst_bounce_back_kernel() and fluid3d_obst_stream_kernel()
///// together accomplish the no-slip wall boundary condition by bounce-back rule.
__global__ void
fluid3d_obst_bounce_back_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  // float tmpf[19];

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  // NOT OBSTACLES, return
  char *devPtr = (char *)fluid_grid->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i] < 0.5) return;

  // MICROSCOPIC BOUNDARY CONDITIONS: OBSTACLES (Half-Way bounce-back)
  //
  /*****
  for (d = 0; d < 19; ++d) 
  {
      // devPtr = (char *)fluid_grid->fOUT[d].ptr;
      // slice = devPtr + k * slicePitch;
      // row = (float *)(slice + j * pitch);
      // row[i] = get3d_value(fluid_grid->fIN[bb[d]].ptr, pitch, slicePitch, i, j, k);
      ///// tmpf[d] = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
  }

  for (d = 0; d < 19; ++d)
  {
      devPtr = (char *)fluid_grid->fIN[d].ptr;
      slice = devPtr + k * slicePitch;
      row = (float *)(slice + j * pitch);
      row[i] = tmpf[bb[d]];
  }

  /// Copy to fOUT for the streaming step for obstacles.
  for (d = 0; d < 19; ++d)
  {
      char *devPtrout = (char *)fluid_grid->fOUT[d].ptr;
      char *sliceout = devPtrout + k * slicePitch;
      float *rowout = (float *)(sliceout + j * pitch);

      // devPtr = (char *)fluid_grid->fIN[d].ptr;
      // slice = devPtr + k * slicePitch;
      // row = (float *)(slice + j * pitch);
      // rowout[i] = row[i];
      rowout[i] = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
  }
  /// END: Copy to fOUT for the streaming step for obstacles.
  ***/

  //// switch distribution function at solid nodes.
  //// Call fluid3d_obst_stream_kernel() afterwards to bounce-back.
  for (d = 0; d < 19; ++d)
  {
      devPtr = (char *)fluid_grid->fOUT[d].ptr;
      slice = devPtr + k * slicePitch;
      row = (float *)(slice + j * pitch);
      row[i] = get3d_value(fluid_grid->fIN[bb[d]].ptr, pitch, slicePitch, i, j, k); 
  }
}


///// fluid3d_collision_kernel_2()
///// does collision step for all fluid nodes. 
///// save to fOUT.
__global__ void
fluid3d_collision_kernel_2(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  float Cs = fluid_grid->dx/fluid_grid->dt, Cs2, Cs4, cu;
  float ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0;
  Cs2 = Cs*Cs;
  Cs4 = Cs2*Cs2;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  // OBSTACLES, return
  char *devPtr = (char *)fluid_grid->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i] > 0.5) return;

  //// calculate rho and ux, uy, uz
  for (d = 0; d < 19; ++d) {
    rho += get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);
    ux += cx[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uy += cy[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uz += cz[d] * get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
  }

  ux = ux/rho;
  uy = uy/rho;
  uz = uz/rho;

  devPtr = (char *)fluid_grid->rho.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = rho;

  devPtr = (char *)fluid_grid->ux.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = ux;
  // ux = row[i];

  devPtr = (char *)fluid_grid->uy.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = uy;
  // uy = row[i];

  devPtr = (char *)fluid_grid->uz.ptr;
  slice = devPtr + k * slicePitch;
  row = (float *)(slice + j * pitch);
  row[i] = uz;
  // uz = row[i];

  for (d = 0; d < 19; ++d) {
     cu = 3.0 * (cx[d] * ux *Cs + cy[d] * uy  *Cs + cz[d] * uz *Cs);
     float fEQ = rho * t[d] * (1.0 + cu/(Cs2) + 0.5 * cu * cu/(Cs4) - 1.5 * (ux * ux + uy * uy + uz * uz)/(Cs2));
     float fIN = get3d_value(fluid_grid->fIN[d].ptr, pitch, slicePitch, i, j, k);

     char *devPtr = (char *)fluid_grid->fOUT[d].ptr;
     char *slice = devPtr + k * slicePitch;
     float *row = (float *)(slice + j * pitch);
     row[i] = fIN - (1.0/OMEGA) * (fIN - fEQ);
     // row[i] = fIN - (fluid_grid->dt/OMEGA) * (fIN - fEQ);
  }
}

__global__ void
fluid3d_stream_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  int si, sj, sk, d;

  for (d = 0; d < 19; ++d) {
    si = i - (int)cx[d]; sj = j - (int)cy[d]; sk = k - (int)cz[d];
#if 0
    if (si < 0) si = fluid_grid->width - 1;
    if (sj < 0) sj = fluid_grid->height - 1;
    if (sk < 0) sk = fluid_grid->depth - 1;
    if (si == fluid_grid->width) si = 0;
    if (sj == fluid_grid->height) sj = 0;
    if (sk == fluid_grid->depth) sk = 0;
#else
    if (si < 0) continue;
    if (sj < 0) continue;
    if (sk < 0) continue;
    if (si == fluid_grid->width) continue;
    if (sj == fluid_grid->height) continue;
    if (sk == fluid_grid->depth) continue;
#endif

    char *devPtr = (char *)fluid_grid->fIN[d].ptr;
    char *slice = devPtr + k * slicePitch;
    float *row = (float *)(slice + j * pitch);
    row[i] = get3d_value(fluid_grid->fOUT[d].ptr, pitch, slicePitch, si, sj, sk);
   }
}

__global__ void
fluid3d_obst_stream_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;

  // int lx = grids->width;
  // int ly = grids->height;
  // int lz = grids->depth;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  int si, sj, sk, d;

  /*** Old implementation, current [i][j][k] receives from neighors. ****/
  // Go through all nodes. Find fluid nodes which are linked to solid boundary nodes.
  //// IMPORTANT: check if (si,sj,sk) is a solid boundary node.
  ////            If not, skip it.
  ////  This assumes that we use inflow-outflow boundary condition in depth direction 
  ////  and wall boundary conditions at the rest boundaries, which are specified by 
  ////   fluid_grid->obst.
  for (d = 0; d < 19; ++d) 
  {
    si = i - (int)cx[d]; sj = j - (int)cy[d]; sk = k - (int)cz[d];

    if (si < 0) continue;
    if (sj < 0) continue;
    if (sk < 0) continue;
    if (si >= fluid_grid->width) continue;
    if (sj >= fluid_grid->height) continue;
    if (sk >= fluid_grid->depth) continue;

    char *devPtr = (char *)fluid_grid->obst.ptr;
    char *slice = devPtr + sk * slicePitch;
    float *row = (float *)(slice + sj * pitch);
    if (row[si] < 0.5) continue;

    devPtr = (char *)fluid_grid->fIN[d].ptr;
    slice = devPtr + k * slicePitch;
    row = (float *)(slice + j * pitch);
    // Using fOUT[bb[d]] is wrong, has been reversed already in fluid3d_obst_bounce_back_kernel. 
    // row[i] = get3d_value(fluid_grid->fOUT[bb[d]].ptr, pitch, slicePitch, si, sj, sk); 
    row[i] = get3d_value(fluid_grid->fOUT[d].ptr, pitch, slicePitch, si, sj, sk);
  }
}

///// fluid3d_stream_kernel_2() only do streaming
///// for fluid nodes. 
///// Also, periodicity is assumed in x, y, and z directions. 
///// When using inflow/outflow boundary condition at z-direction, this makes
///// fIN for nodes in inflow/outflow plane incorrect.
///// Results are save to fIN.
__global__ void
fluid3d_stream_kernel_2(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = fluid_grid->rho.pitch;
  size_t slicePitch = pitch * fluid_grid->height;

  // OBSTACLES needs to receive from fluid nodes as well.
  /***
  char *devPtr = (char *)fluid_grid->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i] > 0.5) return;
  ***/

  int si, sj, sk, d;

  /*** current [i][j][k] receives from neighors. ****/
  for (d = 0; d < 19; ++d) 
  {
    si = i - (int)cx[d]; sj = j - (int)cy[d]; sk = k - (int)cz[d];

    //// NOTE: Periodicity is assumed here. 
    //// This is not matter for most cases considered. Specific 
    ////  boundary condition functions will be called after to set proper 
    ////  boundary conditions. 
    if (si < 0) si = fluid_grid->width - 1;
    if (sj < 0) sj = fluid_grid->height - 1;
    if (sk < 0) sk = fluid_grid->depth - 1;
    if (si == fluid_grid->width) si = 0;
    if (sj == fluid_grid->height) sj = 0;
    if (sk == fluid_grid->depth) sk = 0;

    char *devPtr = (char *)fluid_grid->fIN[d].ptr;
    char *slice = devPtr + k * slicePitch;
    float *row = (float *)(slice + j * pitch);
    row[i] = get3d_value(fluid_grid->fOUT[d].ptr, pitch, slicePitch, si, sj, sk);
  }
}

