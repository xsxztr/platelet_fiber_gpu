/*
 GPUDefines.h
 
 Definitions for GPU code.
 
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
#if !defined(_GPUDefines_H)
#define _GPUDefines_H

#define PI 3.141592653589793

#if defined(__cplusplus)
#define NO                 false
#define FUNCTION_FAILED    false
#define YES                true
#define FUNCTION_SUCCEEDED true
#else /* defined(__cplusplus) */
enum _bool { false              = 0,
                NO                 = false,
                FUNCTION_FAILED    = false,
                true               = 1,
                YES                = true,
                FUNCTION_SUCCEEDED = true};
typedef enum _bool bool;
#endif /* defined(__cplusplus) */

typedef struct _RK2grids {
  int speciesCount;
  void **speciesGrid;
  void **speciesF1;
  void **speciesF2;
  void **speciesDiff;
} RK2grids;

typedef struct _RModelRK2_GPUptrs {
  int numOfODEs;
  int numParams;
  float dt;
  float EPS;
  int *localEpsilonCheck;
  int *epsilonCheck;
  int speciesCount;
  float *speciesData;
  float *speciesF1;
  float *speciesF2;
  size_t speciesPitch;
  float *parameters;
  size_t paramPitch;
} RModelRK2_GPUptrs;

typedef struct _ModelGPUData {
  void *data;
  float *parameters;
  int numModels;
  int numParams;
  void *gpuPtrs;
  void *gpuFunctions;
} ModelGPUData;

struct _RECT_GRID {
        double lL[3];       /* Lower corner of rectangle containing grid */
        double lU[3];       /* Upper corner of rectangle containing grid */
        double h[3];       /* Average grid spacings in the grid         */
        int   gmax[3];    /* Number of grid blocks                     */
        int   dim;        /* Dimension of Grid */

                /* Specifications for virtual domains and variable grids */
        double GL[3];      /* Lower corner of global grid */
        double GU[3];      /* Upper corner of global grid */
        double VL[3];      /* Lower corner of virtual domain */
        double VU[3];      /* Upper corner of virtual domain */
        int   lbuf[3];    /* Lower buffer zone width */
        int   ubuf[3];    /* Upper buffer zone width */
                /* Specifications for variable mesh grids */

        // double *edges[3];        /* Coordinate cell edges */
        // double *centers[3];      /* Coordinate cell centers */
        // double *dh[3];           /* Coordindate cell widths */
        // double *glstore;         /* Storage for edges, centers and dh arrays */
        // int    variable_mesh[3]; /* YES for variable dh in ith direction */
};
typedef struct _RECT_GRID RECT_GRID;
#define cell_index(p,i,gr)      irint(floor(((p)-(gr)->lL[i])/(gr)->h[i]))
#define cell_center(indx,i,gr)  ((gr)->lL[i] + ((indx) + 0.5)*(gr)->h[i])
#define lattice_crds(indx,i,gr)  ((gr)->lL[i] + ((indx) )*(gr)->h[i])


// typedef for GPU functions

// ReactionModel, static structure
typedef	void* (allocRModelGPUFunction)(void*, int, int, int, float, float);
typedef void (initRModelGPUFunction)(void*, void*, float*, float*);
typedef void (invokeRModelGPUFunction)(void*, void*, float*, int);
typedef void (releaseRModelGPUFunction)(void*, void*);

typedef struct _RModelGPUFunctions {
    allocRModelGPUFunction *allocGPUKernel;
    initRModelGPUFunction *initGPUKernel;
    invokeRModelGPUFunction *invokeGPUKernel;
    releaseRModelGPUFunction *releaseGPUKernel;
} RModelGPUFunctions;

// ReactionModel, dynamic structure
typedef	void* (allocRModelDynamicGPUFunction)(void*, int, int, int, float, float);
typedef void (initRModelDynamicGPUFunction)(void*, void*, float*, float*, float*, float*, float*, float*, float*, int*, int*, int*, int*);
typedef void (invokeRModelDynamicGPUFunction)(void*, void*, float*, int);
typedef void (releaseRModelDynamicGPUFunction)(void*, void*);

typedef struct _RModelDynamicGPUFunctions {
  allocRModelDynamicGPUFunction *allocGPUKernel;
  initRModelDynamicGPUFunction *initGPUKernel;
  invokeRModelDynamicGPUFunction *invokeGPUKernel;
  releaseRModelDynamicGPUFunction *releaseGPUKernel;
} RModelDynamicGPUFunctions;

// ReactionDiffusionModel
typedef	void* (allocRDModelGPUFunction)(void*, float, float, float*, int, int);
typedef void (initRDModelGPUFunction)(void*, void*, float*, RK2grids*);
typedef void (invokeRDModelGPUFunction)(void*, void*, RK2grids*, int);
typedef void (releaseRDModelGPUFunction)(void*, void*);

typedef struct _RK2functions {
    allocRDModelGPUFunction *allocGPUKernel;
    initRDModelGPUFunction *initGPUKernel;
    invokeRDModelGPUFunction *invokeGPUKernel;
    releaseRDModelGPUFunction *releaseGPUKernel;
} RK2functions;

// Hill function
// General form
// f(X) = a + (b - a) / (1 + (X / c)^h)
#ifndef HILL
#define HILL(X, A, B, C, H) ((A) + ((B) - (A)) / (1 + pow((X) / (C), H)))
#endif

#define     sqr(x)       ((x)*(x))
#define     cub(x)       ((x)*(x)*(x))
#define     quar(x)      ((x)*(x)*(x)*(x))

#ifdef __cplusplus
extern "C" { 
#endif // #ifdef __cplusplus
  // structure to hold fiber network data together
  typedef struct _tmp_fiber_GPUgrids {
  int maxNodes;            // Nodes index starts at "0", and ends at maxNodes-1.
  int maxLinks;
  int max_N_conn_at_Node;  // Max. number of links at a node
  double dt;
  int   *NodeType;         // [Nodes], node type: fixed = 1, otherwise = 0
  int   *N_Conn_at_Node;      // [Nodes], number of connections at each node.
  int   *Link_at_Node;     // [Nodes][max_node_N_connect], the links connected to the  node.
  int   *lAdjVer;          // [LNum][2]  index of two nodes/vertices forming this link
  double *linkLengths;      // [LNum]   distance between  2 nodes forming  link
  double *linkLengths0;     // [LNum]   initial distance between  2 nodes forming  link
  double *linkThick;        // [LNum]   thickness of each  link
  double *X;                // array of x coordinate for all nodes
  double *X0;		   // array of x at initial 	
  double *V_X;              // x-component of velocity of nodes. 
  double *F_X;              // x-component of FSI force at nodes.
  double *Y;                // array of y coordinate for all nodes
  double *Y0;
  double *V_Y;
  double *F_Y;
  double *Z;                // array of z coordinate for all nodes
  double *Z0;
  double *V_Z;
  double *F_Z;
  size_t pitchCellsNodes;// pitch for all x, y, z, Fx, Fy, Fz arrays  [numNodes]
  size_t pitchlAdjVer;  // pitch for 2d arrays for Link associatation [numLinks][2]
  // size_t pitchElemAdj;  // pitch for 2d arrays for Elem Associations  [numElements][3]
  // size_t pitchElemNorm;  // pitch for 2d arrays for Elem Associations  [numElements][3]
 size_t pitchLink_at_Node; // pitch for 2d arrays for Link_at_Node [Nodes][max_node_N_connect],
 //  curandState *devState;
  } tmp_fiber_GPUgrids;
#ifdef __cplusplus
}
#endif // #ifdef __cplusplus


////////// global functions
///// gpuSEM_LB.c /////
///// semLB.cu /////
///// C function callable by C++ code.
#ifdef __cplusplus
extern "C" { 
#endif // #ifdef __cplusplus
extern void  *fiber_allocGPUKernel(void*,int,int,int,double, double*);
extern void    *sem_allocGPUKernel(void *model, int maxCells, int maxElements, int SurfElem, int newnode,int numReceptorsPerElem,
                                   float dt, double S0_all, float *hostSEMparameters);
extern void    sem_initGPUKernel(void *model, void *g, int numOfCells, int *numOfElements, int SurfElem,
                                 int numReceptorsPerElem, void *hostX_Ref, void *hostY_Ref, void *hostZ_Ref,
                                 void *hostX, void *hostY, void *hostRY, void *hostZ,
                                 void *hostVX, void *hostVY,void *hostVZ,void *hostFX, void *hostFY, void *hostFZ,
                                 void *hostType, void *hostBonds, void *triElem, void *receptor_r1, void *receptor_r2,
                                 void *node_share_Elem, void *N, void *node_nbrElemNum, void *node_nbr_nodes,
                                 void *S0, double V0, void *receptBond, void *randNum);
#ifdef __cplusplus
}
#endif // #ifdef __cplusplus

///// fiber.cpp /////
//// C++ function that is callable by C code.
#ifdef __cplusplus
extern "C" { 
#endif // #ifdef __cplusplus
extern tmp_fiber_GPUgrids  *init_fibrin_network(double dt); 
// void  *fiber_allocGPUKernel(void *model, int maxNodes, int maxLinks, float dt);
extern void  cpu_init_fiber_LB(tmp_fiber_GPUgrids*,double*, void **fIN, void **fOUT, 
                 void *ux, void *uy, void *uz, void *rho, void *obstacle,RECT_GRID*,double);
extern void  fiber_LB_Fsi_force(tmp_fiber_GPUgrids*,double*, void **fIN, void **fOUT, 
                 void *ux, void *uy, void *uz, void *rho, void *obstacle,RECT_GRID*,double);
extern int   Flow_vel_at_position(double*,RECT_GRID*,void*,void*,void*,void*,void*,double*,double*);
extern int   rect_in_which(double*,int*,RECT_GRID*,int);

#if !defined(sun) || (defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || defined(_HPUX_SOURCE) || defined(cray) || (defined(__GNUC__) && !defined(linux))
extern int   irint(double);
#endif /* !defined(sun) || (defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || defined(_HPUX_SOURCE)  || defined(cray) || (defined(__GNUC__) && !defined(linux)) */

#ifdef __cplusplus
}
#endif // #ifdef __cplusplus
///// END:  fiber.cpp /////


#endif /// #if !defined(_GPUDefines_H)
