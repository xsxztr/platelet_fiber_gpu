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
#ifndef LB_KERNEL_CU
#define LB_KERNEL_CU 
#include <math.h>
#include "sem_kernel.cu" 

__constant__ double t[] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
			  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
__constant__ double cx[] = {0, 1,-1, 0, 0, 0, 0, 1, 1, 1, 1,-1,-1,-1,-1, 0, 0, 0, 0};
__constant__ double cy[] = {0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1};
__constant__ double cz[] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1};
__constant__ int   bb[] = {0, 2, 1, 4, 3, 6, 5, 12, 11, 14, 13, 8, 7, 10, 9, 18, 17, 16, 15};

//// TMP
// float              newcx[] = {0, 1, 0, 0, -1,  0,  0,  1, -1, -1,  1, 1, -1, -1,  1, 0,  0,  0,  0};
__device__ int stepL = 0, stepR = 0;
__device__ float
get3d_value_float(void *devPtr, size_t pitch, size_t slicePitch, int i, int j, int k)
{
  char *slice = (char *)devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  return row[i];
}
__device__ double
get3d_value(void *devPtr, size_t pitch, size_t slicePitch, int i, int j, int k)
{
  char *slice = (char *)devPtr + k * slicePitch;
  double *row = (double *)(slice + j * pitch);
  return row[i];
}

/*__device__ double atomicAdd(double* address, double val) { 
  unsigned long long int* address_as_ull = (unsigned long long int*)address; 
  unsigned long long int old = *address_as_ull, assumed; 
  do { 
     assumed = old; 
     old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); 
  } while (assumed != old); 
  return __longlong_as_double(old); 
}*/


/*__global__ void
fluid3d_in_out_flow_boundary_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
   
  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  char *devPtr = (char *)grids->rho.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
 
  float tmp_ux, tmp_uy, tmp_uz, cu;
  float Cs = grids->dx/grids->dt;
  float Nyx = 0.0, Nyz = 0.0;

  //// Inflow boundary condition at height = 0.
  if ((j == 0) && (k > 0) && (k < (grids->depth - 1))
       && (i > 0) && (i < (grids->width - 1)) ) 
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

     devPtr = (char *)grids->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_ux = 0.0;

     devPtr = (char *)grids->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = SHEAR_RATE* k * (grids->dx);
     tmp_uy = SHEAR_RATE* k * (grids->dx);

     devPtr = (char *)grids->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0;
     tmp_uz = 0;

     if( (i == 1) || (i == grids->width-2) || 
         (k == 1) || (k == grids->depth-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = INIT_RHO;
         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)grids->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = INIT_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
         
     }
     else
     {

         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);

         ///// NEW 08052011, with units.
         ///// part of fIN entering the domain is not correct at this point because of fluid3d_stream_kernel_2()
         ///// assumes periodicity on boundary conditions. 
         ///// On inflow side (z = 0) in Z (depth) direction: This means e11, e15,e3, e16,e12 need to be updated.
         row[i] = 1.0 / (1.0 - get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/Cs)
           * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[5].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[6].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[17].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[18].ptr, pitch, slicePitch, i, j, k)));
         /////END::: NEW 08052011, with units

         // % MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
         //
         /////NEW 08052011, with units
         
         Nyx = 0.5 * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k)/Cs;

         Nyz = 0.5 * (get3d_value(grids->fIN[5].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[6].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k)/Cs;
 

         devPtr = (char *)grids->fIN[3].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/(Cs);
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[7].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
               - Nyx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)grids->fIN[11].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
             + Nyx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[15].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              - Nyz;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[16].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              + Nyz;
         /////END::: NEW 08052011, with units
     }
  } /// end :::: if(k == 0)

  //// Outflow boundary condition at depth = grid->depth-1.
  if ((j == (grids->height - 1))
       && (k > 0) && (k < (grids->depth - 1))
       && (i > 0) && (i < (grids->width - 1))) {

     // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
     // % Outlet: Constant pressure
     //
    
         // Implementation of on-site velocity boundary conditions for D2Q19 lattice Boltzmann simulations
         // M. Hecht, J. Harting. J. of Stat. Mech.: Theory and Experiment. 2010

     devPtr = (char *)grids->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;
     tmp_ux = 0.0;

     devPtr = (char *)grids->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = SHEAR_RATE* k * (grids->dx);;
     tmp_uy = SHEAR_RATE* k * (grids->dx);;

     devPtr = (char *)grids->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0;
     tmp_uz = 0;

     if( (i == 1) || (i == grids->width-2) ||
         (k == 1) || (k == grids->depth-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = INIT_RHO;

         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)grids->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = INIT_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
        
     }
     else 
     {

         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);

         row[i] = 1.0 / (1.0 + get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/Cs)
           * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[5].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[6].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[15].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[16].ptr, pitch, slicePitch, i, j, k)));

         
         devPtr = (char *)grids->fIN[4].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) *
                     get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/(Cs);

         /////////////////////
         devPtr = (char *)grids->fIN[12].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
               + Nyx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)grids->fIN[8].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/(Cs) 
             - Nyx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[18].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[15].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              + Nyz;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[17].ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[16].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k))/(Cs) 
              - Nyz;
     }
                

     
  } /// END:::: if ((k == (grids->depth - 1))
}*/


__global__ void
fluid3d_noslip_boundary_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
 // int d;
   
  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  char *devPtr = (char *)grids->rho.ptr;
  char *slice = devPtr + k * slicePitch;
  double *row = (double *)(slice + j * pitch);
 
  double C = grids->dx/grids->dt/UMAX;
  double Nzx = 0.0, Nzy = 0.0;

  //// Inflow boundary condition at height = 0.
  if ((k == 0)/* && (i > 0) && (i < grids->width - 1) && (j > 0) && (j < grids->height - 1)*/) 
  {
     // Implementation of on-site velocity boundary conditions for D2Q19 lattice Boltzmann simulations
     // M. Hecht, J. Harting. J. of Stat. Mech.: Theory and Experiment. 2010
     // % MACROSCOPIC (DIRICHLET) BOUNDARY CONDITIONS
     /// due to singularity for nodes which are directly linked to solid boundary nodes. Special treatment is needed.
     /// In fact, need to follow "M. Hecht, J. Harting." and "Y.T. Feng, K. Han and D.R.J. Owen. 
     /// Coupled lattice Boltzmann method and discrete element modeling of particle transport in 
     /// turbulent fluid flows: Computational issue. Int. J.  Numer. Meth. Engng. 72:1111-1134, 2007" to derive the boundary condition. 
     /// Here, we try a simple fix: use initial condition to compute f again.

     devPtr = (char *)grids->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)grids->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)grids->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = 0.0;

   /*  if( (j == 1) || (j == grids->height-2) || 
         (i == 1) || (i == grids->width-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = FLUID_RHO;
         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)grids->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = FLUID_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
         
     }
     else*/
    // {

         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);

         ///// NEW 08052011, with units.
         ///// part of fIN entering the domain is not correct at this point because of fluid3d_stream_kernel_2()
         ///// assumes periodicity on boundary conditions. 
         ///// On inflow side (z = 0) in Z (depth) direction: This means e11, e15,e3, e16,e12 need to be updated.
         row[i] = 1.0 / (1.0 - get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k)/C)
           * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(grids->fIN[6].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[16].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[18].ptr, pitch, slicePitch, i, j, k)));
         /////END::: NEW 08052011, with units

         // % MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
         //
         /////NEW 08052011, with units
         
         Nzx = 0.5 * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k)/C;

         Nzy = 0.5 * (get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/C;
 

         devPtr = (char *)grids->fIN[5].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[6].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k)/C;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[9].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/C 
               - Nzx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)grids->fIN[13].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/C 
             + Nzx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[15].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[18].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k))/C 
              - Nzy;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[17].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[16].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k))/C 
              + Nzy;
         /////END::: NEW 08052011, with units
     //}
 /* double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0, dt = grids->dt, f[19];
  for (int d = 0; d < 19; ++d) {
    rho += get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
    ux += cx[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uy += cy[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    uz += cz[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k)*Cs;
    if (i == 40 && j == 40 && d == 5){
       double a = get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
       printf("bd fIN[%d] = %.16e\n", d, a);
    }
   // f[d] = get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
  }
   
    if (i == 40 && j == 40){
        printf("Boundary fIN[5] = %.16e, fIN[6] = %.16e\n", f[5],f[6]);
    } 
   if (i == 40 && j == 40 && k == 1){
     printf("rho = %.16e, ux = %.16e, uy = %.16e, uz = %.16e\n", rho, ux, uy, uz);
  }
  
 // float fdt = 0.5 * get3d_value(grids->Fx.ptr, pitch, slicePitch, i, j, k) * dt;
  ux = (ux + 0.5 * get3d_value(grids->Fx.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  uy = (uy + 0.5 * get3d_value(grids->Fy.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  uz = (uz + 0.5 * get3d_value(grids->Fz.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  
  if (i == 40 && j == 40){
     printf("rho = %.16e, ux = %.16e, uy = %.16e, uz = %.16e\n", rho, ux, uy, uz);
  }*/
  } /// end :::: if(k == 0)
}

__global__ void
fluid3d_moving_plate_boundary_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
 // int d;
   
  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  char *devPtr = (char *)grids->rho.ptr;
  char *slice = devPtr + k * slicePitch;
  double *row = (double *)(slice + j * pitch);
 
  float C = grids->dx/grids->dt/UMAX;
  float Nzx = 0.0, Nzy = 0.0;

  //// Inflow boundary condition at height = 0.
  if ((k == grids->depth - 1 )/* && (i > 0) && (i < grids->width - 1) && (j > 0) && (j < grids->height - 1)*/) 
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

     devPtr = (char *)grids->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)grids->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = SHEAR_RATE* k * (grids->dx)/UMAX;

     devPtr = (char *)grids->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (double *)(slice + j * pitch);
     row[i] = 0.0;

   /*  if( (j == 1) || (j == grids->height-2) || 
         (i == 1) || (i == grids->width-2)
       )
     {
         //// TMP fix 08092011. Simply reinitialize
         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (float *)(slice + j * pitch);
         row[i] = FLUID_RHO;
         for (d = 0; d < 19; ++d)
         {
             devPtr = (char *)grids->fIN[d].ptr;
             slice = devPtr + k * slicePitch;
             row = (float *)(slice + j * pitch);
             cu = 3.0 * (cx[d] * tmp_ux + cy[d] * tmp_uy + cz[d] * tmp_uz)*Cs;
             row[i] = FLUID_RHO*t[d]*( 1.0 + cu/(Cs*Cs) + 0.5*cu*cu/(Cs*Cs*Cs*Cs) -
                                  1.5* (tmp_ux * tmp_ux + tmp_uy * tmp_uy + tmp_uz*tmp_uz)/(Cs*Cs));
         }
         
     }
     else*/
     {

         devPtr = (char *)grids->rho.ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);

         ///// NEW 08052011, with units.
         ///// part of fIN entering the domain is not correct at this point because of fluid3d_stream_kernel_2()
         ///// assumes periodicity on boundary conditions. 
         ///// On inflow side (z = 0) in Z (depth) direction: This means e11, e15,e3, e16,e12 need to be updated.
         row[i] = 1.0 / (1.0 + get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k)/C)
           * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
              + get3d_value(grids->fIN[0].ptr, pitch, slicePitch, i, j, k)
              + 2.0 * (get3d_value(grids->fIN[5].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[15].ptr, pitch, slicePitch, i, j, k)
                       + get3d_value(grids->fIN[17].ptr, pitch, slicePitch, i, j, k)));
         /////END::: NEW 08052011, with units

         // % MICROSCOPIC BOUNDARY CONDITIONS: INLET (Zou/He BC)
         //
         /////NEW 08052011, with units
         
         Nzx = 0.5 * (get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k)/C;

         Nzy = 0.5 * (get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                        + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                        - get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k))
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                         get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k)/C;
 

         devPtr = (char *)grids->fIN[6].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[5].ptr, pitch, slicePitch, i, j, k)
           - 1.0 / 3.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k) * 
                     get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k)/C;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[10].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/C 
               - Nzx;
         /////END::: NEW 08052011, with units
    
         devPtr = (char *)grids->fIN[14].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
        
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k))/C 
             + Nzx;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[16].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[17].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) + get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k))/C 
              - Nzy;
         /////END::: NEW 08052011, with units

         devPtr = (char *)grids->fIN[18].ptr;
         slice = devPtr + k * slicePitch;
         row = (double *)(slice + j * pitch);
         
         /////NEW 08052011, with units
         row[i] = get3d_value(grids->fIN[15].ptr, pitch, slicePitch, i, j, k)
           + 1.0 / 6.0 * get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k)
           * (- get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k) - get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k))/C 
              + Nzy;
         /////END::: NEW 08052011, with units
     }
  } /// end :::: if(i == 0)
}

/*__global__ void
fluid3d_edge_corner_boundary_kernel(void *g, int aBank){
  
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
   
  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  char *devPtr;
  char *slice;
  float *row;
  //const float SHEAR_RATE = 1e4;
  float Nyzx = 0.0, Nxzy = 0.0;
  
  if (k == grids->depth - 1){
     devPtr = (char *)grids->ux.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0;

     devPtr = (char *)grids->uy.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = SHEAR_RATE * k * (grids->dx);

     devPtr = (char *)grids->uz.ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 0.0 ;
  
//Edges
      if ((j == 0) && (i > 0) && (i < grids->width - 1)){

     
          int bounce[19] = {0, 1, 2, 4, 4, 5, 5, 12, 8, 9, 13, 8, 12, 13, 9, 15, 17, 17, 18};
                         //{0, 1, 2, 3, 4, 5, 6,  7, 8, 9, 10,11, 12, 13, 14,15, 16, 17, 18}
          for (d = 0; d < 19; ++d){
              devPtr = (char *)grids->fIN[d].ptr;
              slice = devPtr + k * slicePitch;
              row = (float *)(slice + j * pitch);
              row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
           }
     
          float f1 = get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k); 
          float f2 = get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k); 
     Nyzx = 0.25 * (f1 - f2); 
     
     devPtr = (char *)grids->fIN[7].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nyzx;
     
     devPtr = (char *)grids->fIN[11].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nyzx;

     devPtr = (char *)grids->fIN[10].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nyzx;
     
     devPtr = (char *)grids->fIN[14].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nyzx;

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/22.0 * (get3d_value(grids->fIN[16].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[17].ptr, pitch, slicePitch, i, j, k));      
  }
  
  
  if ((j == grids->height - 1) && (i > 0) && (i < grids->width - 1)){
     
      int bounce[19] = {0, 1, 2, 3, 3, 5, 5, 7, 11, 9, 13, 11, 7, 13, 9, 15, 16, 17, 15};
      for (d = 0; d < 19; ++d){
          devPtr = (char *)grids->fIN[d].ptr;
          slice = devPtr + k * slicePitch;
          row = (float *)(slice + j * pitch);
          row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }
     
     float f1 = get3d_value(grids->fIN[1].ptr, pitch, slicePitch, i, j, k); 
     float f2 = get3d_value(grids->fIN[2].ptr, pitch, slicePitch, i, j, k); 
     Nyzx = 0.25 * (f1 - f2); 
     
     devPtr = (char *)grids->fIN[8].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nyzx;
     
     devPtr = (char *)grids->fIN[12].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nyzx;

     devPtr = (char *)grids->fIN[10].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nyzx;
     
     devPtr = (char *)grids->fIN[14].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nyzx;

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/22.0 * (get3d_value(grids->fIN[15].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[18].ptr, pitch, slicePitch, i, j, k));      
  
  }
  
  if ((i == 0) && (j > 0) && (j < grids->height - 1)){
    
     int bounce[19] = {0, 2, 2, 3, 4, 5, 5, 12, 11, 9, 13, 11, 12, 13, 14, 15, 17, 17, 15};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }
     
     float f1 = get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k); 
     float f2 = get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k); 
     Nxzy = 0.25 * (f1 - f2); 
     
     devPtr = (char *)grids->fIN[7].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nxzy;
     
     devPtr = (char *)grids->fIN[8].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nxzy;

     devPtr = (char *)grids->fIN[16].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nxzy;
     
     devPtr = (char *)grids->fIN[18].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nxzy;

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/22.0 * (get3d_value(grids->fIN[9].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[14].ptr, pitch, slicePitch, i, j, k));      
  
  }
  
  if ((i == grids->width - 1) && (j > 0) && (j < grids->height - 1)){
   
     int bounce[19] = {0, 1, 1, 3, 4, 5, 5, 7, 8, 9, 10, 8, 7, 13, 9, 15, 17, 17, 15};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }
     
     float f1 = get3d_value(grids->fIN[3].ptr, pitch, slicePitch, i, j, k); 
     float f2 = get3d_value(grids->fIN[4].ptr, pitch, slicePitch, i, j, k); 
     Nxzy = 0.25 * (f1 - f2); 
     
     devPtr = (char *)grids->fIN[11].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nxzy;
     
     devPtr = (char *)grids->fIN[12].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nxzy;

     devPtr = (char *)grids->fIN[16].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] += Nxzy;
     
     devPtr = (char *)grids->fIN[18].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] -= Nxzy;

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/22.0 * (get3d_value(grids->fIN[10].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[13].ptr, pitch, slicePitch, i, j, k));      
  
  }
  
  // corners
  if ((i == 0) && (j == 0)){
     int bounce[19] = {0, 2, 2, 4, 4, 5, 5, 12, 8, 9, 13, 11, 12, 13, 14, 15, 17, 17, 18};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/18.0 * (get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k));      
  
  }
  
  if ((i == 0) && (j == grids->height - 1)){
     int bounce[19] = {0, 2, 2, 3, 3, 5, 5, 7, 11, 9, 13, 11, 12, 13, 14, 15, 16, 17, 15};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/18.0 * (get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k));      
  
  }

  if ((i == grids->width - 1) && (j == 0)){
     int bounce[19] = {0, 1, 1, 4, 4, 5, 5, 7, 8, 9, 10, 8, 12, 13, 9, 15, 17, 17, 18};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/18.0 * (get3d_value(grids->fIN[7].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[12].ptr, pitch, slicePitch, i, j, k));      
  
  }
  
  
  if ((i == grids->width - 1) && (j == grids->height - 1)){
     int bounce[19] = {0, 1, 1, 3, 3, 5, 5, 7, 8, 9, 10, 11, 7, 13, 9, 15, 16, 17, 15};
     for (d = 0; d < 19; ++d){
       devPtr = (char *)grids->fIN[d].ptr;
       slice = devPtr + k * slicePitch;
       row = (float *)(slice + j * pitch);
       row[i] = get3d_value(grids->fIN[bounce[d]].ptr, pitch, slicePitch, i, j, k); 
     }

     devPtr = (char *)grids->fIN[0].ptr;
     slice = devPtr + k * slicePitch;
     row = (float *)(slice + j * pitch);
     row[i] = 12.0/18.0 * (get3d_value(grids->fIN[8].ptr, pitch, slicePitch, i, j, k)
                           + get3d_value(grids->fIN[11].ptr, pitch, slicePitch, i, j, k));      
  
   }
  }//if (k == grids->depth - 1)
}*/
///// fluid3d_obst_bounce_back_kernel() do bounce-back (reverse states only)
///// for all solid nodes. 
///// For solid nodes, the bounce-back states are saved in
///// both fIN and fOUT. fOUT will be used in fluid3d_obst_stream_kernel()
///// to stream the state from solid nodes back to the connected fluid nodes. 
///// fluid3d_obst_bounce_back_kernel() and fluid3d_obst_stream_kernel()
///// together accomplish the no-slip wall boundary condition by bounce-back rule.
/*__global__ void
fluid3d_obst_bounce_back_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  // NOT OBSTACLES, return
  char *devPtr = (char *)grids->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i] < 0.5) return;

  // MICROSCOPIC BOUNDARY CONDITIONS: OBSTACLES (Half-Way bounce-back)
  //

  //// switch distribution function at solid nodes.
  //// Call fluid3d_obst_stream_kernel() afterwards to bounce-back.
  for (d = 0; d < 19; ++d)
  {
      devPtr = (char *)grids->fOUT[d].ptr;
      slice = devPtr + k * slicePitch;
      row = (float *)(slice + j * pitch);
      row[i] = get3d_value(grids->fIN[bb[d]].ptr, pitch, slicePitch, i, j, k); 
  }
}*/

__global__ void
fluid3d_velocity_density_kernel(void *g, int aBank){
  
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  float dx = grids->dx, dt = grids->dt;
  double C = dx/dt/UMAX;
  double ux = 0.0, uy = 0.0, uz = 0.0, rho = 0.0;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;
  size_t pitchf = grids->obst.pitch;
  size_t slicePitchf = pitchf * grids->height;

  // OBSTACLES, return
  char *devPtr = (char *)grids->obst.ptr;
  char *slice = devPtr + k * slicePitchf;
  float *rowf = (float *)(slice + j * pitchf);
  if (rowf[i] > 0.5) return;
  
// double f[19];
  //// calculate rho and ux, uy, uz
  //double fx, fy, fz;
  for (d = 0; d < 19; ++d) {
    rho += get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
    ux += cx[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
    uy += cy[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
    uz += cz[d] * get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
   /* if (i == 40 && j == 20 && k == 60){
       double a = get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
       printf("fIN[%d] = %.16e\n", d, a);
    }*/
  // f[d] = get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
  }
 /* if (i == 40 && j == 200 && k == 40){
        printf("fIN[5] = %.16e, fIN[6] = %.16e\n", f[5],f[6]);
    }*/ 
 /* if (i == 40 && j == 20 && k == 60){
     printf("rho = %.16e, ux = %.16e, uy = %.16e, uz = %.16e\n", rho, ux * Cs, uy * Cs, uz * Cs);
  }*/
  
  //fx = UNIT_FACTOR * 0.5 * get3d_value(grids->Fx.ptr, pitch, slicePitch, i, j, k) * dt;
  //fy = UNIT_FACTOR * 0.5 * get3d_value(grids->Fy.ptr, pitch, slicePitch, i, j, k) * dt;
  //fz = UNIT_FACTOR * 0.5 * get3d_value(grids->Fz.ptr, pitch, slicePitch, i, j, k) * dt;
  double F_over_RHO_U = F0/INIT_RHO/UMAX * UNIT_FACTOR; 
  ux = (ux * C + 0.5 * F_over_RHO_U * get3d_value(grids->Fx.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  uy = (uy * C + 0.5 * F_over_RHO_U * get3d_value(grids->Fy.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  uz = (uz * C + 0.5 * F_over_RHO_U * get3d_value(grids->Fz.ptr, pitch, slicePitch, i, j, k) * dt) /rho;
  
  /*if (i == 40 && j == 200 && k == 7){
     stepL ++;
     printf("L:step = %d, ux = %.16e, uy = %.16e, uz = %.16e\n", stepL, ux, uy, uz);
   //  printf("fx = %.16e, fy = %.16e, fz = %.16e\n", fx, fy, fz);
  }*/
 /* if (i == 80 && j == 2 && k == 65){
     stepR ++;
     printf("R:step = %d, ux = %.16e, uy = %.16e, uz = %.16e\n", stepR, ux, uy, uz);
  }*/
  
  /*ux = ux * Cs /rho;
  uy = uy * Cs /rho;
  uz = uz * Cs /rho;*/
  devPtr = (char *)grids->rho.ptr;
  slice = devPtr + k * slicePitch;
  double *row = (double *)(slice + j * pitch);
  row[i] = rho;

  devPtr = (char *)grids->ux.ptr;
  slice = devPtr + k * slicePitch;
  row = (double *)(slice + j * pitch);
  row[i] = ux;
  

  devPtr = (char *)grids->uy.ptr;
  slice = devPtr + k * slicePitch;
  row = (double *)(slice + j * pitch);
  row[i] = uy;

  devPtr = (char *)grids->uz.ptr;
  slice = devPtr + k * slicePitch;
  row = (double *)(slice + j * pitch);
  row[i] = uz;
  
}
///// fluid3d_collision_kernel()
///// does collision step for all fluid nodes. 
///// save to fOUT.
__global__ void
fluid3d_collision_kernel(void *g, int aBank)
{
  fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = aBank * blockDim.z + threadIdx.z;
  int d;
  double C = grids->dx/grids->dt/UMAX;
  double rho = 0.0, cu = 0.0;
  double F[3], u[3], e[3], uF[3][3], ee[3][3];
  double C2 = C*C;
  double C4 = C2*C2;

  // check thread in boundary
  if (i >= grids->width) return;
  if (j >= grids->height) return;
  if (k >= grids->depth) return;

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;
  //size_t pitchf = grids->obst.pitch;
  //size_t slicePitchf = pitchf * grids->height;

  // OBSTACLES, return
  /*char *devPtr = (char *)grids->obst.ptr;
  char *slice = devPtr + k * slicePitch;
  float *row = (float *)(slice + j * pitch);
  if (row[i] > 0.5) return;*/
   
  rho = get3d_value(grids->rho.ptr, pitch, slicePitch, i, j, k);
  u[0] = get3d_value(grids->ux.ptr, pitch, slicePitch, i, j, k);
  u[1] = get3d_value(grids->uy.ptr, pitch, slicePitch, i, j, k);
  u[2] = get3d_value(grids->uz.ptr, pitch, slicePitch, i, j, k);
  /*if (i == 37 && j == 3 && k == 10){
     printf("ux = %.16e, uy = %.16e, uz = %.16e\n", u[0], u[1], u[2]);
   //  printf("fx = %.16e, fy = %.16e, fz = %.16e\n", fx, fy, fz);
  }*/

  char *devPtr = (char *)grids->Fx.ptr;
  char *slice = devPtr + k * slicePitch;
  double *row = (double *)(slice + j * pitch);
  F[0] = row[i];

  devPtr = (char *)grids->Fy.ptr;
  slice = devPtr + k * slicePitch;
  row = (double *)(slice + j * pitch);
  F[1] = row[i];

  devPtr = (char *)grids->Fz.ptr;
  slice = devPtr + k * slicePitch;
  row = (double *)(slice + j * pitch);
  F[2] = row[i];
  
  for (int l = 0; l < 3; ++l){
      for (int m = 0; m < 3; ++m){
           uF[l][m] = u[l] * F[m] + F[l] * u[m];
          } 
  }

  for (d = 0; d < 19; ++d) {
      e[0] = cx[d] * C; 
      e[1] = cy[d] * C; 
      e[2] = cz[d] * C;
      double sum = 0; 
      cu = 3.0 * (e[0] * u[0] + e[1] * u[1] + e[2] * u[2]);
      for (int l = 0; l < 3; ++l)
           for (int m = 0; m < 3; ++m){
               if (l == m) ee[l][m] = e[l] * e[m] - C2/3.0;
               else ee[l][m] = e[l] * e[m]; 
      } 
      
      for (int l = 0; l < 3; ++l){
           for (int m = 0; m < 3; ++m){
               sum += uF[l][m] * ee[m][l];  
           } 
      }

     double fEQ = rho * t[d] * (1.0 + cu/(C2) + 0.5 * cu * cu/(C4) - 1.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])/(C2));
     double fIN = get3d_value(grids->fIN[d].ptr, pitch, slicePitch, i, j, k);
    /* if(i == 0 && j == 0 && k == 2){
         printf("rho = %.16e, fEQ = %.16e, fIN[%d] = %.16e\n", rho, fEQ, d, fIN);
     }*/
     char *devPtr = (char *)grids->fOUT[d].ptr;
     char *slice = devPtr + k * slicePitch;
     double *row = (double *)(slice + j * pitch);
    // double fdt = t[d] * (1.0 - 0.5/OMEGA) * (3.0 * ((cx[d] * Cs - ux) * Fx + 
    //                    (cy[d] * Cs - uy)* Fy + (cz[d] * Cs - uz) * Fz) + 
    //                    9.0 * (cx[d] * ux + cy[d] * uy + cz[d] * uz) * (cx[d] * Fx + cy[d] * Fy + 
    //                    cz[d] * Fz))/ Cs2 * (grids->dt) * UNIT_FACTOR;// pico gram/micron^3
     double fdt = t[d] * (1.0 - 0.5/OMEGA) * (3.0 * (e[0] * F[0] + e[1] * F[1] + e[2] * F[2])/C2 + 
                        4.5 * sum /C4) * (grids->dt);
    // double fdt = 1.5 * t[d] * (Fx * cx[d] + Fy * cy[d] + Fz * cz[d]) * Cs * (grids->dt);
     row[i] = fIN - (1.0/OMEGA) * (fIN - fEQ) + F0 * fdt/INIT_RHO/UMAX * UNIT_FACTOR;
    /*if(i == 40 && j == 20 && k == 60 && d == 0){
          printf("fdt = %.16e\n", fdt);
       // printf("fIN[%d] = %.16e,fOUT[%d] = %.16e\n", d, fIN, d, row[i]);
   //     printf("Fx = %.16e, Fy = %.16e, Fz = %.16e\n", Fx,Fy,Fz);
    }*/
    
  }
}

///// fluid3d_stream_kernel() only do streaming
///// for fluid nodes. 
///// Also, periodicity is assumed in x, y, and z directions. 
///// When using inflow/outflow boundary condition at z-direction, this makes
///// fIN for nodes in inflow/outflow plane incorrect.
///// Results are save to fIN.
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

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  // OBSTACLES needs to receive from fluid nodes as well.
  /***
  char *devPtr = (char *)grids->obst.ptr;
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
    if (si < 0) si = grids->width - 1;
    if (sj < 0) sj = grids->height - 1;
    if (sk < 0) sk = grids->depth - 1;
    if (si == grids->width) si = 0;
    if (sj == grids->height) sj = 0;
    if (sk == grids->depth) sk = 0;

    char *devPtr = (char *)grids->fIN[d].ptr;
    char *slice = devPtr + k * slicePitch;
    double *row = (double *)(slice + j * pitch);
    row[i] = get3d_value(grids->fOUT[d].ptr, pitch, slicePitch, si, sj, sk);
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

  size_t pitch = grids->rho.pitch;
  size_t slicePitch = pitch * grids->height;

  int si, sj, sk, d;

  /*** [i][j][k] receives from neighors. ****/
  // Go through all nodes. Find fluid nodes which are linked to solid boundary nodes.
  //// IMPORTANT: check if (si,sj,sk) is a solid boundary node.
  ////            If not, skip it.
  ////  This assumes that we use inflow-outflow boundary condition in depth direction 
  ////  and wall boundary conditions at the rest boundaries, which are specified by 
  ////   grids->obst.
  for (d = 0; d < 19; ++d) 
  {
    si = i - (int)cx[d]; sj = j - (int)cy[d]; sk = k - (int)cz[d];

    if (si < 0) continue;
    if (sj < 0) continue;
    if (sk < 0) continue;
    if (si >= grids->width) continue;
    if (sj >= grids->height) continue;
    if (sk >= grids->depth) continue;

    char *devPtr = (char *)grids->obst.ptr;
    char *slice = devPtr + sk * slicePitch;
    float *row = (float *)(slice + sj * pitch);
    if (row[si] < 0.5) continue;

    devPtr = (char *)grids->fIN[d].ptr;
    slice = devPtr + k * slicePitch;
    row = (float *)(slice + j * pitch);
    // Using fOUT[bb[d]] is wrong, has been reversed already in fluid3d_obst_bounce_back_kernel. 
    // row[i] = get3d_value(grids->fOUT[bb[d]].ptr, pitch, slicePitch, si, sj, sk); 
    row[i] = get3d_value(grids->fOUT[d].ptr, pitch, slicePitch, si, sj, sk);
  }
}


//Calculate distributed force on LB grids

__global__ void
fluid3d_force_distribute_kernel(void *g, void *g_SEM, int aBank){
 
 fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
 sem_GPUgrids *grids_SEM = (sem_GPUgrids *)g_SEM;

 int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;
 int k = aBank * blockDim.z + threadIdx.z;

 // check thread in boundary
 if (i >= grids->width) return;
 if (j >= grids->height) return;
 if (k >= grids->depth) return;
 
 size_t pitch = grids->rho.pitch;
 size_t slicePitch = pitch * grids->height;
 
 float dx = grids->dx;
 double threshold = 2 * dx; 
 double delta;
 double x, y, z;
 int numOfCells = grids_SEM->numOfCells;
 int* numOfElements = grids_SEM->numOfElements;
 double *sem_X = grids_SEM->X;
 double *sem_Y = grids_SEM->Y;
 double *sem_Z = grids_SEM->Z;
 double *sem_Fx = grids_SEM->F_X;
 double *sem_Fy = grids_SEM->F_Y;
 double *sem_Fz = grids_SEM->F_Z;
 size_t pitch_sem = grids_SEM->pitch;

 char *devPtr = (char *)grids->Fx.ptr;
 char *slice = devPtr + k * slicePitch;
 double *Fx = (double *)(slice + j * pitch);

 devPtr = (char *)grids->Fy.ptr;
 slice = devPtr + k * slicePitch;
 double *Fy = (double *)(slice + j * pitch);

 devPtr = (char *)grids->Fz.ptr;
 slice = devPtr + k * slicePitch;
 double *Fz = (double *)(slice + j * pitch);

 for (int l = 0; l < numOfCells; ++l)
   for (int m = 0; m < numOfElements[l]; ++m)
     {
        double *row = (double*)((char*)sem_X + m * pitch_sem);
        x = row[l];
        row = (double*)((char*)sem_Y + m * pitch_sem);
        y = row[l];
        row = (double*)((char*)sem_Z + m * pitch_sem);
        z = row[l];

	if (abs(x- i * dx) >= threshold) continue;
	if (abs(y- j * dx) >= threshold) continue;
	if (abs(z- k * dx) >= threshold) continue;

	delta = (0.25 * (1.0 + cos(0.5*PI*(x - i * dx)/dx))) * (0.25 * (1.0 + cos(0.5*PI*(y - j * dx)/dx))) * 
                (0.25 * (1.0 + cos(0.5*PI*(z - k * dx)/dx))); 
        
        row = (double*)((char*)sem_Fx + m * pitch_sem);
	Fx[i] += row[l] * delta;
        row = (double*)((char*)sem_Fy + m * pitch_sem);
	Fy[i] += row[l] * delta;
        row = (double*)((char*)sem_Fz + m * pitch_sem);
	Fz[i] += row[l] * delta;

     }
 //Periodic distribute
 if (j < 2 || j > (grids->height - 2)){
    if (j < 2) j = j + grids->height;
    else j = j - grids->height;
    
    for (int l = 0; l < numOfCells; ++l)
        for (int m = 0; m < numOfElements[l]; ++m)
        {
           double *row = (double*)((char*)sem_X + m * pitch_sem);
           x = row[l];
           row = (double*)((char*)sem_Y + m * pitch_sem);
           y = row[l];
           row = (double*)((char*)sem_Z + m * pitch_sem);
           z = row[l];

	   if (abs(x- i * dx) >= threshold) continue;
	   if (abs(y- j * dx) >= threshold) continue;
	   if (abs(z- k * dx) >= threshold) continue;

	   delta = (0.25 * (1.0 + cos(0.5*PI*(x - i * dx)/dx))) * (0.25 * (1.0 + cos(0.5*PI*(y - j * dx)/dx))) * 
                (0.25 * (1.0 + cos(0.5*PI*(z - k * dx)/dx))); 
        
           row = (double*)((char*)sem_Fx + m * pitch_sem);
	   Fx[i] += row[l] * delta;
           row = (double*)((char*)sem_Fy + m * pitch_sem);
	   Fy[i] += row[l] * delta;
           row = (double*)((char*)sem_Fz + m * pitch_sem);
	   Fz[i] += row[l] * delta;
       }    
 }
 

}//fluid3d_force_distribut_kernel

//Calculate distributed velocity on SEM

__global__ void
fluid3d_velocity_distribute_kernel(void *g, void *g_SEM, int aBank){
 
 fluid_GPUgrids *grids = (fluid_GPUgrids *)g;
 sem_GPUgrids *grids_SEM = (sem_GPUgrids *)g_SEM;

 int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;
 int k = aBank * blockDim.z + threadIdx.z;

 // check thread in boundary
 if (i >= grids->width) return;
 if (j >= grids->height) return;
 if (k >= grids->depth) return;
 
 size_t pitch = grids->rho.pitch;
 size_t slicePitch = pitch * (grids->height);
 
 double dx = grids->dx, x, y, z;
 double threshold = 2 * dx; 
 double delta;
 int numOfCells = grids_SEM->numOfCells;
 int *numOfElements = grids_SEM->numOfElements;
 double *sem_X = grids_SEM->X;
 double *sem_Y = grids_SEM->Y;
 double *sem_Z = grids_SEM->Z;
 double *sem_Vx = grids_SEM->V_X;
 double *sem_Vy = grids_SEM->V_Y;
 double *sem_Vz = grids_SEM->V_Z;
 size_t pitch_sem = grids_SEM->pitch;

 char *devPtr = (char *)grids->ux.ptr;
 char *slice = devPtr + k * slicePitch;
 double *ux = (double *)(slice + j * pitch);

 devPtr = (char *)grids->uy.ptr;
 slice = devPtr + k * slicePitch;
 double *uy = (double *)(slice + j * pitch);

 devPtr = (char *)grids->uz.ptr;
 slice = devPtr + k * slicePitch;
 double *uz = (double *)(slice + j * pitch);
 
 devPtr = (char *)grids->rho.ptr;
 slice = devPtr + k * slicePitch;
 //double *rho = (double *)(slice + j * pitch);

 for (int l = 0; l < numOfCells; ++l)
   for (int m = 0; m < numOfElements[l]; ++m)
     {
        double *row = (double*)((char*)sem_X + m * pitch_sem);
        x = row[l];
        row = (double*)((char*)sem_Y + m * pitch_sem);
        y = row[l];
        row = (double*)((char*)sem_Z + m * pitch_sem);
        z = row[l];

	if (abs(x- i * dx) >= threshold) continue;
	if (abs(y- j * dx) >= threshold) continue;
	if (abs(z- k * dx) >= threshold) continue;
	delta = (0.25 * (1.0 + cos(0.5*PI*(x - i * dx)/dx))) * (0.25 * (1.0 + cos(0.5*PI*(y - j * dx)/dx))) * 
                (0.25 * (1.0 + cos(0.5*PI*(z - k * dx)/dx))); 
        /*if (i == 40 && j == 20 && k == 60 && m == 0){
           printf("delta = %.16e, uy = %.16e\n", delta, uy[i]);
        }*/
 //       row = (double*)((char*)sem_rho + m * pitch_sem);
   //     atomicAdd(&(row[l]), rho[i] * delta);
                
        row = (double*)((char*)sem_Vx + m * pitch_sem);
        atomicAdd(&(row[l]), ux[i] * delta);
        /*if (m == 0){
           printf("i = %d, j = %d, k = %d, Vx = %.16e\n", i, j, k,row[l]);
        }*/
        row = (double*)((char*)sem_Vy + m * pitch_sem);
        atomicAdd(&(row[l]), uy[i] * delta);
       // if (m == 0){
           //printf("i = %d, j = %d, k = %d, uy = %.16e, delta = %e, um = %e\n",i,j,k,uy[i],delta,row[l]);
         //  printf("%e\n", uy[i] * delta);
       // }
        row = (double*)((char*)sem_Vz + m * pitch_sem);
        atomicAdd(&(row[l]), uz[i] * delta);
        //if (m == 0){
        //   printf("i = %d, j = %d, k = %d, uy = %.16e, um = %e\n",i,j,k,uy[i],row[l]);
       // }
     }
 // Periodic distribute
 if (j < 2 || j > (grids->height - 2)){
    
    if (j < 2) j = j + grids->height;
    else j = j - grids->height;
    
    for (int l = 0; l < numOfCells; ++l)
        for (int m = 0; m < numOfElements[l]; ++m)
        {
            double *row = (double*)((char*)sem_X + m * pitch_sem);
            x = row[l];
            row = (double*)((char*)sem_Y + m * pitch_sem);
            y = row[l];
            row = (double*)((char*)sem_Z + m * pitch_sem);
            z = row[l];

	    if (abs(x- i * dx) >= threshold) continue;
	    if (abs(y- j * dx) >= threshold) continue;
	    if (abs(z- k * dx) >= threshold) continue;
	    delta = (0.25 * (1.0 + cos(0.5*PI*(x - i * dx)/dx))) * (0.25 * (1.0 + cos(0.5*PI*(y - j * dx)/dx))) * 
                    (0.25 * (1.0 + cos(0.5*PI*(z - k * dx)/dx))); 
     //       row = (double*)((char*)sem_rho + m * pitch_sem);
       //     atomicAdd(&(row[l]), rho[i] * delta);
            row = (double*)((char*)sem_Vx + m * pitch_sem);
            atomicAdd(&(row[l]), ux[i] * delta);
            row = (double*)((char*)sem_Vy + m * pitch_sem);
            atomicAdd(&(row[l]), uy[i] * delta);
            row = (double*)((char*)sem_Vz + m * pitch_sem);
            atomicAdd(&(row[l]), uz[i] * delta);
     }
 }
  

}//fluid3d_velocity_distribute_kernel

#endif
