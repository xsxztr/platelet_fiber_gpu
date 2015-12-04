/*
 gpu_SEM_LB.c
 
 Main functions for running simulation.

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


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include <gsl/gsl_rng.h>
//#include <iostream>
//#include <fstream>
//#include <BioSwarm/MultiScale3DGrid.h>
#include "GPUDefines.h"

#define RAND_NUM ((double)rand() / (double)RAND_MAX )
/// BEGIN: DECLARE LOCAL variables
static RECT_GRID     *lattice_gr = NULL;
/// END: DECLARE LOCAL variables

/// BEGIN: DECLARE GPU KERNELS ///

// The SEM GPU kernels (functions)
/**
extern void    *sem_allocGPUKernel(void *model, int maxCells, int maxElements, int SurfElem, int newnode,int numReceptorsPerElem, 
                                   float dt, double S0_all, float *hostSEMparameters);
extern void    sem_initGPUKernel(void *model, void *g, int numOfCells, int *numOfElements, int SurfElem, 
                                 int numReceptorsPerElem, void *hostX_Ref, void *hostY_Ref, void *hostZ_Ref,
                                 void *hostX, void *hostY, void *hostRY, void *hostZ, 
                                 void *hostVX, void *hostVY,void *hostVZ,void *hostFX, void *hostFY, void *hostFZ,  
                                 void *hostType, void *hostBonds, void *triElem, void *receptor_r1, void *receptor_r2,
                                 void *node_share_Elem, void *N, void *node_nbrElemNum, void *node_nbr_nodes,
                                 void *S0, double V0, void *receptBond, void *randNum);
**/
//extern void sem_invokeGPUKernel(void *model, void *g, void *hostX, void *hostY, void *hostZ, int timeSteps);
extern void    sem_invokeGPUKernel_Force(void *model, void *g, int timeSteps, int* done, int* totalT, gsl_rng *r, double gama);

extern void    skin_invokeGPUKernel(void *model, void *g, void *hostX, void *hostY, void *hostZ,
                                 void *, void *, void *, void *, void *, int timeSteps);
extern void    sem_copyGPUKernel(void *model, void *g, void *hostX, void *hostY, void *hostRY, void *hostZ, 
                                 void *hostVX, void *hostVY, void *hostVZ,
                                 void *hostFX, void *hostFY, void *hostFZ, int timeSteps);

extern void    sem_releaseGPUKernel(void *model, void *g);

// GPU Kernels for LB_fluid
void    *fluid_allocGPUKernel(void *model, float dt, float dx, int width, int height, int depth);
void    fluid_initGPUKernel(void *model, void *g, int aFlag, float *hostLBparameters, void **fIN, void **fOUT, void *ux,
                         void *uy, void *uz, void *rho, void *obstacle, void *Fx, void *Fy, void *Fz, void *vWFbond);
void    fluid_invokeGPUKernel(void *model, void *g, void *g_SEM, double *randNum, gsl_rng *r, int timeSteps);
void    fluid_releaseGPUKernel(void *model, void *g);
/// END: DECLARE GPU KERNELS ///

/// Local function 
static void  runGPUSimulation(int);
static void  outputSave(int t, int nx, int ny, int nz, void *rho, int write_rho,
               void *ux, void *uy, void *uz, int write_vel, char *directory, char *filename);
static void  writeVTK(int t, int nx, int ny, int nz,
               void *rho, int write_rho, void *ux, void *uy, void *uz, int write_vel, char *directory, char *filename);
/// END: Local function 

//float hostParameters[100];

// spatial domain
// For 128 and 250 cells
#define BOUNDARY_X 100.0
#define BOUNDARY_Y 100.0

#define BOUNDARY_Z 10.0
#define Z_RANGE 0.5
#define Z_OFFSET 0.5

#define INTER_DIST 0.6

// species index
#define NOTCH_species 0
#define DELTA_species 1
#define BOUND_species 2
#define BMA_species 3
#define OVOL1_species 4
#define OVOL2_species 5
#define CMYC_species 6

#define Cell_Width 0.5
#define Cell_Length 5.0 


// Generate normally distributed random number
double normal_dist_rand(double mean, double sd)
{
  double fac, radius, v1, v2;
  double rd1Value, rd2Value;

  do {
    rd1Value = RAND_NUM;
    rd2Value = RAND_NUM;
    v1 = (2.0 * rd1Value) - 1.0;
    v2 = (2.0 * rd2Value) - 1.0;
    radius = v1*v1 + v2*v2; 
  } while (radius >= 1.0);
  fac = sqrt (-2.0 * log (radius) / radius);
  return v2 * fac * sd + mean;    // use fixed params
}

// Readers for input
//
// This Parser looks for the parameter file specified by the Path ParName.
// The format of the parameter file should be as follows
//
// <KEYWORD>   value
// <KEYWORD>   value



void ParReader( char ParName[], float *pdt ,float *psimTime , int *pOutputFreq, int *pMaxCells, 
              int *pMaxElements, int * pSurfElem, char *pPDBName, char *pPSFName, char *pNEUName, float *pdx, float *HostParam )
{


    FILE *parIn;
    parIn = fopen(ParName , "r");

    if (parIn == NULL)
       {
          printf("Parameter File was not open");
          exit(1);
       }
  char PDBFileName[25];
  char PSFFileName[25];
  char NEUFileName[50];
  char KEYWORD[20];
  float *p2dt = malloc(sizeof(float));
  float *p2dx = malloc(sizeof(float));
  int OutFreq, MaxCells, MaxElems, SurfElems, Seed;
  float SimTime;
/////Read in Path to PDB
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "PDB_PATH") ){
       fscanf(parIn, "%s"  , pPDBName);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "PDB_PATH");
         exit(1);
       }
/////Read in Path to PSF
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "PSF_PATH") ){
       fscanf(parIn, "%s"  , pPSFName);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "PSF_PATH");
         exit(1);
       }
///Read in Path to Neu
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "NEU_PATH") ){
       fscanf(parIn, "%s"  , pNEUName);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "PSF_PATH");
         exit(1);
       }
  
///Read in dx
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "DELTA_X") ){
       fscanf(parIn, "%f" , p2dx);
       *pdx = *p2dx;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "DELTA_X");
         exit(1);
       }

///Read in dt
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "DELTA_T") ){
       fscanf(parIn, "%f" , p2dt);
       *pdt = *p2dt;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "DELTA_T");
         exit(1);
       }
  
///Read in Simulation time (total in units of time:  eg.  200 -> is 200 us, if dt = .1, this will need 2000 simulation Steps)
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "SIM_TIME") ){
        fscanf(parIn, "%f", &SimTime);
        *psimTime = SimTime;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "SIM_TIME");
         exit(1);
       }

///Read in Output Frequency in simulation Steps. Eg if SIM_TIME=200, dt=.01, (TotalSimSteps=20000),
/// outputFreq=200 -> gives 100 frames (outputs)
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "OUT_FREQ") ){
       fscanf(parIn, "%d", &OutFreq);
       *pOutputFreq = OutFreq;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "OUT_FREQ");
         exit(1);
       }
// Read In Max cells
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "MAX_CELL") ){

       fscanf(parIn, "%d", &MaxCells);
       *pMaxCells = MaxCells;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "MAX_CELL");
         exit(1);
       }
// Read In number of SCE per cell
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "MAX_ELEM") ){
       fscanf(parIn, "%d", &MaxElems);
       *pMaxElements = MaxElems;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "MAX_ELEM");
         exit(1);
       }
// Read In number of surface triangle elements
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "SURF_ELEM") ){
       fscanf(parIn, "%d", &SurfElems);
       *pSurfElem = SurfElems;
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "SURF_ELEM");
         exit(1);
       }

// NOW WE FILL IN THE HOSTPARAMETER array which contains parameters for force calculations
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "RECEPTOR") ){
       fscanf(parIn, "%f", & HostParam[0]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "RECEPTOR");
         exit(1);
       }

  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "RANDOM_SEED") ){
       fscanf(parIn, "%f", &HostParam[1]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "RANDOM_SEED");
         exit(1);
       }

  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "K_SPRING") ){
       fscanf(parIn, "%f", & HostParam[2]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "K_SPRING");
         exit(1);
       }
  //Kf0 will be HostParam[3]
  
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "Kr0") ){
       fscanf(parIn, "%f", & HostParam[4]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "Kr0");
         exit(1);
       }
  
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "SHEAR_RATE") ){
       fscanf(parIn, "%f", & HostParam[5]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "SHEAR_RATE");
         exit(1);
       }  
  
  fscanf(parIn, "%s"  , KEYWORD);
    if (!strcmp(KEYWORD, "INIT_RHO") ){
       fscanf(parIn, "%f", & HostParam[6]);
       }
      else
       {
         printf("Parameter Read Failed for %s \n", "INIT_RHO");
         exit(1);
       }  
}

void NEUReader(char *NEUFileP, double *X, double *Y, double *Z, double *bondLengths, int *trielem){
     FILE *neuIn;
     char NEUFilename[512];
     int NumSCE, NumTriElem;
     strcpy(NEUFilename, NEUFileP);
     neuIn = fopen(NEUFileP, "r");
     if (neuIn == NULL){
        printf("File was not open");
        exit(1);
     }     
     fscanf(neuIn,"%d",&NumSCE);
     fscanf(neuIn,"%d",&NumTriElem);
     int SCE_idx = 0, TriE_idx, trash, elemidx;
     double iX, iY, iZ;
    /* *(X + SCE_idx) = 0.0;
     *(Y + SCE_idx) = 0.0;
     *(Z + SCE_idx) = 0.0;*/
     int i;
     for (i = 0; i < NumSCE; i++){
         fscanf(neuIn,"%d",&SCE_idx);
         SCE_idx--;
         fscanf(neuIn,"%lf",&iX);
         fscanf(neuIn,"%lf",&iY);
         fscanf(neuIn,"%lf",&iZ);
        // if (SCE_idx == 407) printf("%e, %e, %e\n", iX , iY, iZ);
         *(X + SCE_idx) = iX;
         *(Y + SCE_idx) = iY;
         *(Z + SCE_idx) = iZ;            
     }
     for (i = 0; i < NumTriElem; i++){
         fscanf(neuIn,"%d",&TriE_idx);
         TriE_idx--;
         fscanf(neuIn,"%d",&trash);
         fscanf(neuIn,"%d",&trash);
         fscanf(neuIn,"%d",&elemidx);
         //if (i == 0) printf("%d, %d\n", TriE_idx, elemidx);
         *(trielem + TriE_idx * 6) = (elemidx - 1);          
         fscanf(neuIn,"%d",&elemidx);
         *(trielem + TriE_idx * 6 + 1) = (elemidx - 1);          
         fscanf(neuIn,"%d",&elemidx);
         *(trielem + TriE_idx * 6 + 2) = (elemidx - 1);          
     }
     //printf("%d %d %d\n",*trielem, *(trielem+1), *(trielem+2));
    // NumSCE++; //include Centroid 
     double dx,dy,dz,dr;
   /*  for (i = 1; i < NumSCE; i++){         
         dx = *X - *(X + i);
         dy = *Y - *(Y + i);
         dz = *Z - *(Z + i);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + NumSCE * i)= dr;
         *(bondLengths + i)= dr;        
     }*/
     
     int elem1, elem2, elem3;     
     for (i = 0; i < NumTriElem; i++){
         elem1 = *(trielem + i * 6);
         elem2 = *(trielem + i * 6 + 1);
         elem3 = *(trielem + i * 6 + 2);
         dx = *(X + elem1) - *(X + elem2);
         dy = *(Y + elem1) - *(Y + elem2);
         dz = *(Z + elem1) - *(Z + elem2);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem1 + NumSCE * elem2)= dr;
         *(bondLengths + elem2 + NumSCE * elem1)= dr;        
         dx = *(X + elem2) - *(X + elem3);
         dy = *(Y + elem2) - *(Y + elem3);
         dz = *(Z + elem2) - *(Z + elem3);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem2 + NumSCE * elem3)= dr;
         *(bondLengths + elem3 + NumSCE * elem2)= dr; 
         dx = *(X + elem3) - *(X + elem1);
         dy = *(Y + elem3) - *(Y + elem1);
         dz = *(Z + elem3) - *(Z + elem1);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem3 + NumSCE * elem1)= dr;
         *(bondLengths + elem1 + NumSCE * elem3)= dr;                        
     } 
}

void NEU_Center_Reader(char *NEUFileP, double *X, double *Y, double *Z, double *bondLengths, int *trielem){
     FILE *neuIn;
     char NEUFilename[512];
     int NumSCE, NumTriElem;
     strcpy(NEUFilename, NEUFileP);
     neuIn = fopen(NEUFileP, "r");
     if (neuIn == NULL){
        printf("File was not open");
        exit(1);
     }     
     fscanf(neuIn,"%d",&NumSCE);
     fscanf(neuIn,"%d",&NumTriElem);
     int SCE_idx = 0, TriE_idx, trash, elemidx;
     double iX, iY, iZ;
     *(X + SCE_idx) = 0.0;
     *(Y + SCE_idx) = 0.0;
     *(Z + SCE_idx) = 0.0;
     int i;
     for (i = 0; i < NumSCE; i++){
         fscanf(neuIn,"%d",&SCE_idx);
        // SCE_idx--;
         fscanf(neuIn,"%lf",&iX);
         fscanf(neuIn,"%lf",&iY);
         fscanf(neuIn,"%lf",&iZ);
        // if (SCE_idx == 407) printf("%e, %e, %e\n", iX , iY, iZ);
         *(X + SCE_idx) = iX;
         *(Y + SCE_idx) = iY;
         *(Z + SCE_idx) = iZ;            
     }
     for (i = 0; i < NumTriElem; i++){
         fscanf(neuIn,"%d",&TriE_idx);
         TriE_idx--;
         fscanf(neuIn,"%d",&trash);
         fscanf(neuIn,"%d",&trash);
         fscanf(neuIn,"%d",&elemidx);
         //if (i == 0) printf("%d, %d\n", TriE_idx, elemidx);
         *(trielem + TriE_idx * 6) = (elemidx - 1);          
         fscanf(neuIn,"%d",&elemidx);
         *(trielem + TriE_idx * 6 + 1) = (elemidx - 1);          
         fscanf(neuIn,"%d",&elemidx);
         *(trielem + TriE_idx * 6 + 2) = (elemidx - 1);          
     }
     //printf("%d %d %d\n",*trielem, *(trielem+1), *(trielem+2));
     NumSCE++; //include Centroid 
     double dx,dy,dz,dr;
     for (i = 1; i < NumSCE; i++){         
         dx = *X - *(X + i);
         dy = *Y - *(Y + i);
         dz = *Z - *(Z + i);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + NumSCE * i)= dr;
         *(bondLengths + i)= dr;        
     }
     
     int elem1, elem2, elem3;     
     for (i = 0; i < NumTriElem; i++){
         elem1 = *(trielem + i * 6);
         elem2 = *(trielem + i * 6 + 1);
         elem3 = *(trielem + i * 6 + 2);
         dx = *(X + elem1) - *(X + elem2);
         dy = *(Y + elem1) - *(Y + elem2);
         dz = *(Z + elem1) - *(Z + elem2);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem1 + NumSCE * elem2)= dr;
         *(bondLengths + elem2 + NumSCE * elem1)= dr;        
         dx = *(X + elem2) - *(X + elem3);
         dy = *(Y + elem2) - *(Y + elem3);
         dz = *(Z + elem2) - *(Z + elem3);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem2 + NumSCE * elem3)= dr;
         *(bondLengths + elem3 + NumSCE * elem2)= dr; 
         dx = *(X + elem3) - *(X + elem1);
         dy = *(Y + elem3) - *(Y + elem1);
         dz = *(Z + elem3) - *(Z + elem1);
         dr = sqrt (dx * dx + dy * dy + dz * dz);
         *(bondLengths + elem3 + NumSCE * elem1)= dr;
         *(bondLengths + elem1 + NumSCE * elem3)= dr;                        
     } 
}

void PDBReader(char * PDBFileP, int NumCells, int NumSCE,  double *X, double *Y, double *Z, int *elemType){
    FILE *pdbIn;
    char  PDBFilename[512];   
  //        printf("Reader Called \n");
    strcpy(PDBFilename , PDBFileP);
    pdbIn = fopen(PDBFileP , "r");
    if (pdbIn == NULL)
       {
          printf("File was not open");
          exit(1);
       }
  char type [5], SCE_type [4] , residue [4], trash [2], trash2 [5];
  int  index, cell_ID, ElemInCell;
  double iX, iY, iZ, t1, t2 ;
  int more = 1, Values = 0;
  while(more){
  fscanf(pdbIn, "%s"  , type);
    if (strcmp(type, "ATOM") ){
        //  printf("Stopped after %d calls \n", Values);
       more = 0;
       break;
       }
      fscanf(pdbIn, "%d" , &index);
      fscanf(pdbIn, "%s"  , SCE_type);
      fscanf(pdbIn, "%s"  , residue);
      fscanf(pdbIn, "%s"  , trash);
      fscanf(pdbIn, "%d" , &cell_ID);
      fscanf(pdbIn, "%lf" , &iX);
      fscanf(pdbIn, "%lf" , &iY);
      fscanf(pdbIn, "%lf" , &iZ);
      fscanf(pdbIn, "%lf" , &t1);
      fscanf(pdbIn, "%lf" , &t2);
      fscanf(pdbIn, "%s"  , trash2);
      ElemInCell = index % NumSCE;
       if (!ElemInCell) ElemInCell = NumSCE; 
      ElemInCell--; 
      cell_ID--;
      *(X + cell_ID + NumCells * ElemInCell )=iX;
      *(Y + cell_ID + NumCells * ElemInCell )=iY;
      *(Z + cell_ID + NumCells * ElemInCell )=iZ;
      // assign element type to elementType array
      if (!strcmp(SCE_type, "CA"))
       *(elemType + cell_ID + NumCells * ElemInCell )= 0;
       else if (!strcmp(SCE_type, "C1"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 1;
       else if (!strcmp(SCE_type, "C2"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 2;
       else if (!strcmp(SCE_type, "C3"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 3;
       else if (!strcmp(SCE_type, "C4"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 4;
	   else if (!strcmp(SCE_type, "C5"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 5;
	   else if (!strcmp(SCE_type, "C6"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 6;
	   else if (!strcmp(SCE_type, "C7"))
        *(elemType + cell_ID + NumCells * ElemInCell )= 7;
      else
       {
        // printf("Failed to assigne Element type  %s \n", SCE_type);
        // exit(1);
       }
         
   /*   printf ("Values is %d  \n", Values);
      printf ("type = %s \n ", type );
      printf ("SCE_type = %s \n ", SCE_type );
      printf ("resdiue = %s \n ", residue );
      printf ("trash = %s \n ", trash );
      printf ("Cell_ID = %d \n ", cell_ID );
     printf ("Index = %d \n ", index );
     printf ("ElemIn Cell = %d  and cell_ID = %d \n ", ElemInCell, cell_ID  );
      printf ("iX = %f \n ", *(X + cell_ID + NumCells * ElemInCell) );
      printf ("iY = %f \n ", *(Y + cell_ID + NumCells * ElemInCell) );
      printf ("iZ = %f \n ", *(Z + cell_ID + NumCells * ElemInCell) );
    printf ("t1 = %f \n ", t1 );
      printf ("t2 = %f \n ", t2 );
      printf ("trash2 = %s \n ", trash2 );
  */  Values++; //THis is for debuging purposes only     
    }
  }

void PSFReader(char * PSFFileP, int NumCells, int NumSCE,  double *bondValue, double *X, double *Y, double *Z, int *elemType){
    FILE *psfIn;
    char  PSFFilename[512], checkForm[512], type [512], beforetype [512];   
  //        printf("Reader Called \n");
// this is the bondLength array defined...TODO, add to input parameterfile OR newly formatted PSF
    strcpy(PSFFilename , PSFFileP);
    psfIn = fopen(PSFFileP , "r");
    if (psfIn == NULL)
       {
          printf("File was not open");
          exit(1);
       }
    fscanf(psfIn, "%s"  , checkForm);
    if (strcmp(checkForm, "PSF_New") ){
          printf("PSF File Not Formated correctly was");
          exit(1);
    }

  int  numBonds;
  fscanf(psfIn, "%s", type);
  strcpy(beforetype, type);
  while(strcmp(type, "!NBOND:")){ 
     fscanf(psfIn, "%s"  , type);
     if (!strcmp(type, "!NBOND:") ){
        numBonds = atoi(beforetype);
     }
     else{
        strcpy(beforetype, type);        
     }
  }
  fscanf(psfIn, "%s"  , type);

  int sce1, sce2, ElemInCell1, ElemInCell2, cell_ID1, cell_ID2,/* type1, type2,*/ bondType;
  int i;
  double dx, dy, dz, dr;
  for( i = 0; i < numBonds; i++){
       fscanf(psfIn, "%d"  , &sce1);//sce1 and sce2 are the index of the SCE elements
       fscanf(psfIn, "%d"  , &sce2);//
       ElemInCell1 = sce1 % NumSCE;
         if (!ElemInCell1) ElemInCell1 = NumSCE; // if modulo == 0, then set ElemInCell to last
       ElemInCell1--; // decrement by 1 for index
       cell_ID1 = floor(ElemInCell1 / NumSCE);
      // type1 =  *(elemType + cell_ID + NumCells * ElemInCell1 );
       ElemInCell2 = sce2 % NumSCE;
         if (!ElemInCell2) ElemInCell2 = NumSCE; // if modulo == 0, then set ElemInCell to 10
       ElemInCell2--; // decrement by 1 for index
       cell_ID2 = floor(ElemInCell2 / NumSCE);
      // type2 =  *(elemType + cell_ID + NumCells * ElemInCell2 );
       
       dx = *(X + cell_ID1 + NumCells * ElemInCell1) - *(X + cell_ID2 + NumCells * ElemInCell2);
       dy = *(Y + cell_ID1 + NumCells * ElemInCell1) - *(Y + cell_ID2 + NumCells * ElemInCell2);
       dz = *(Z + cell_ID1 + NumCells * ElemInCell1) - *(Z + cell_ID2 + NumCells * ElemInCell2);
       dr = sqrt (dx * dx + dy * dy + dz * dz);
  
       *(bondValue + ElemInCell1 + NumSCE * ElemInCell2)= dr;
       *(bondValue + ElemInCell2 + NumSCE * ElemInCell1)= dr;
     
  } 

}


static void runGPUSimulation(int seed)
{
  int    i, j;
  int    p, l, k, d;
  double  cu;
  int    *pMaxCells = malloc(sizeof(int));
  int    *pMaxElements = malloc(sizeof(int));
  int    *pSurfElem = malloc(sizeof(int));// Surface Elements Number
  float  *pSimulateTime = malloc(sizeof(float));
  int    *pOutputFreq = malloc(sizeof(int));
  float  *pDt = malloc(sizeof(float));
  float  *pDx = malloc(sizeof(float));
 
  float  *hostSEMparameters = (float *)malloc( 100 * sizeof(float));
  float  *hostLBparameters = malloc(sizeof(float) * 100);

  int    maxCells , numOfCells, maxElements , SurfElem, InitialElements, ReversalPeriod;
  char   *pPDBName = malloc( 25 * sizeof(char));
  pPDBName[0] = 'C';
  pPDBName[1] = '\0';
  char *pPSFName = malloc( 25 * sizeof(char));
  pPSFName[0] = 'C';
  pPSFName[1] = '\0';
  char PDBName[25];
  char *pNEUName = malloc( 50 * sizeof(char));
  pNEUName[0] = 'C';
  pNEUName[1] = '\0';
  char ParName[] = "Input/parameter.dat";

  // execution settings
  float dx, dt, simulateTime;
  int outputFrequency, thisFrame;

  // declare fiber related variables
  tmp_fiber_GPUgrids *fiber_grid;


  /// Reader Parameter FILE and intialize Readers for PDB and PSF ///  
  ParReader(ParName, pDt, pSimulateTime, pOutputFreq,  pMaxCells, pMaxElements, pSurfElem, pPDBName, pPSFName, pNEUName,
            pDx, hostSEMparameters);
  // Assign values from parameter read in to variables 
  maxCells = *pMaxCells;
  maxElements = *pMaxElements;
  numOfCells = maxCells;
  InitialElements = maxElements;
  SurfElem = *pSurfElem;
  dt = *pDt;
  dx = *pDx;
  outputFrequency = *pOutputFreq;
  simulateTime = *pSimulateTime;


  int timeSteps =  simulateTime / dt; //Assigns number of simulations steps from input parameters

  /// BEGIN: HOST MEMORY ALLOCOATION ///

  /// HOST ALLOCATION FOR SEM ///
  // initialize cell position, velocity, force memory locations, as well as clock's and slime direction
  int *numOfElements = (int *)malloc(maxCells * sizeof(int));
  void *elementType = malloc(maxCells * maxElements * sizeof(int));
  int (*etGrid)[maxElements][maxCells] = elementType;
  void *bondLengths = malloc(maxElements * maxElements * sizeof(double));
 // int totElem = maxCells * maxElements;
  double (*bondgrid)[maxElements][maxElements] = bondLengths;

  void *triElem = malloc(SurfElem * 6 * sizeof(int));// node id of the triangle elements
  int (*triangleElem)[SurfElem][6] = triElem;
  for (i = 0; i < SurfElem; i++){
      for (j = 0; j < 6; j++){
          (*triangleElem)[i][j] = -1;
      }
  }
  

  int numReceptorsPerElem = hostSEMparameters[0]/SurfElem/2;
  //printf("%d\n", numReceptorsPerElem);
  void *receptor_r1 = malloc(SurfElem * numReceptorsPerElem * sizeof(float));// parameters of receptors
  float (*receptor_r1_2D)[SurfElem][numReceptorsPerElem] = receptor_r1;
  void *receptor_r2 = malloc(SurfElem * numReceptorsPerElem * sizeof(float));// parameters of receptors
  float (*receptor_r2_2D)[SurfElem][numReceptorsPerElem] = receptor_r2;
  void *randNum = malloc(SurfElem * numReceptorsPerElem * 2 * sizeof(double));// parameters of receptors
  double (*randNum3D)[2][numReceptorsPerElem][SurfElem] = randNum;
  
  const gsl_rng_type * T;
  gsl_rng * r;
  //gsl_rng_env_setup();
  T = gsl_rng_ranlxs2;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, (unsigned long int) seed);     
  
  for (i = 0; i < SurfElem; i++){
      for (j = 0; j < numReceptorsPerElem; j++){
          (*receptor_r1_2D)[i][j] = gsl_rng_uniform(r);
          (*receptor_r2_2D)[i][j] = gsl_rng_uniform(r);
          (*randNum3D)[0][j][i] = gsl_rng_uniform(r);
          (*randNum3D)[1][j][i] = gsl_rng_uniform(r);
          // printf("r1 = %e, r2 = %e\n", (*receptor_r1_2D)[i][j], (*receptor_r2_2D)[i][j]);        
      }
  }

  void *X = malloc(maxCells * maxElements * sizeof(double));
  void *Y = malloc(maxCells * maxElements * sizeof(double));
  void *Z = malloc(maxCells * maxElements * sizeof(double));
  void *RY = malloc(maxCells * maxElements * sizeof(double));// Real Y for periodic boundary
  void *X_Ref = malloc(maxCells * maxElements * sizeof(double));
  void *Y_Ref = malloc(maxCells * maxElements * sizeof(double));
  void *Z_Ref = malloc(maxCells * maxElements * sizeof(double));
  double (*xGrid)[maxElements][maxCells] = X;
  double (*yGrid)[maxElements][maxCells] = Y;
  double (*zGrid)[maxElements][maxCells] = Z;
  double (*ryGrid)[maxElements][maxCells] = RY;
  double (*xRefGrid)[maxElements][maxCells] = X_Ref;
  double (*yRefGrid)[maxElements][maxCells] = Y_Ref;
  double (*zRefGrid)[maxElements][maxCells] = Z_Ref;
//   float (*xGrid)[maxElements][maxCells] = (float (*)[maxElements][maxCells])X;
//   float (*yGrid)[maxElements][maxCells] = (float (*)[maxElements][maxCells])Y;
//   float (*zGrid)[maxElements][maxCells] = (float (*)[maxElements][maxCells])Z;
// add force and velocity arrays
  void *VX = malloc(maxCells * maxElements * sizeof(double));
  void *VY = malloc(maxCells * maxElements * sizeof(double));
  void *VZ = malloc(maxCells * maxElements * sizeof(double));
  double (*xVGrid)[maxElements][maxCells] = VX;
  double (*yVGrid)[maxElements][maxCells] = VY;
  double (*zVGrid)[maxElements][maxCells] = VZ;

  void *FX = malloc(maxCells * maxElements * sizeof(double));
  void *FY = malloc(maxCells * maxElements * sizeof(double));
  void *FZ = malloc(maxCells * maxElements * sizeof(double));
  double (*xFGrid)[maxElements][maxCells] = FX;
  double (*yFGrid)[maxElements][maxCells] = FY;
  double (*zFGrid)[maxElements][maxCells] = FZ;

 // float *centerX = (float *)malloc(maxCells * sizeof(double));
  //float *centerY = (float *)malloc(maxCells * sizeof(double));
  //float *centerZ = (float *)malloc(maxCells * sizeof(double));

  void *numNeighbors = malloc(maxCells * maxElements * sizeof(int));
  int (*nnGrid)[maxElements][maxCells] = numNeighbors;
  
  /// SEM : INIT ///

  // clear values
  for (j = 0; j < numOfCells; ++j) {
    numOfElements[j] = InitialElements;
    //centerX[j] = 0.0;
    //centerY[j] = 0.0;
    //centerZ[j] = 0.0;
    for (i = 0; i < maxElements; ++i)
      (*etGrid)[i][j] = 0;
      
  }
  for (i = 0; i < maxElements; ++i)  
    for (j = 0; j < maxElements; ++j){
        (*bondgrid)[i][j] = 0.0;
    }

  // Get cell positions from PDB file
     NEUReader(pNEUName, X_Ref, Y_Ref, Z_Ref, bondLengths, triElem);
    // NEU_Center_Reader(pNEUName, X_Ref, Y_Ref, Z_Ref, bondLengths, triElem);

  // Read Configuration Files
 //PDBReader(pPDBName, numOfCells, maxElements, X_Ref, Y_Ref, Z_Ref, elementType );
//Offset the cells to the initial positions
  for (i = 0; i < maxCells; ++i)
      for (j = 0; j < numOfElements[i]; ++j){
         /* double temp = (*yGrid)[j][i];
          (*yGrid)[j][i] = (*zGrid)[j][i];
          (*zGrid)[j][i] = temp;*/
          (*xRefGrid)[j][i]  = (*xRefGrid)[j][i] + 8.0;
          (*yRefGrid)[j][i]  = (*yRefGrid)[j][i] + 1.5;
          (*zRefGrid)[j][i]  = (*zRefGrid)[j][i] + 0.32;
          (*xGrid)[j][i] = (*xRefGrid)[j][i];
          (*yGrid)[j][i] = (*yRefGrid)[j][i];
          (*zGrid)[j][i] = (*zRefGrid)[j][i];
          (*ryGrid)[j][i] = (*yGrid)[j][i];
          (*xVGrid)[j][i] = 0.0; 
          (*yVGrid)[j][i] = 0.0; 
          (*zVGrid)[j][i] = 0.0;
         // (*xPreVGrid)[j][i] = 0.0; 
         // (*yPreVGrid)[j][i] = 0.0; 
         // (*zPreVGrid)[j][i] = 0.0;
          (*xFGrid)[j][i] = 0.0; 
          (*yFGrid)[j][i] = 0.0; 
          (*zFGrid)[j][i] = 0.0;
  }
  double Lm = 0;
  int bondNum = 0;
  for (i = 0; i < maxElements; i++){
      for (j = i+1; j < maxElements; j++){
          if ((*bondgrid)[i][j] > 0){
             Lm += (*bondgrid)[i][j];
             bondNum++;
          }
      }
  }
  Lm /= bondNum;
  //printf("Lm = %e\n", Lm);
  int nodeIdx, elemIdx, newnode = maxElements;
  for (i = 0; i < SurfElem; ++i){
      for (j = i + 1; j < SurfElem; ++j){
          int flag1 = 0, flag2 = 0, matchflag = 0;
          for (k = 0; k < 3; k++){
              for (l = 0; l < 3; l++){
                  if ((*triangleElem)[i][k] == (*triangleElem)[j][l]){ 
                     flag1 += k;
                     flag2 += l;
                     matchflag++;
                  }
              }
          }
          if (matchflag == 2){
             switch(flag1){
                   case 0: break;
                   case 1: (*triangleElem)[i][3]  = newnode; break;
                   case 2: (*triangleElem)[i][4]  = newnode; break;
                   case 3: (*triangleElem)[i][5]  = newnode; break;
                   default : printf("error occurs in setting flag1 neighbour elements\n");  
             }
             switch(flag2){
                   case 0: break;
                   case 1: (*triangleElem)[j][3]  = newnode; break;
                   case 2: (*triangleElem)[j][4]  = newnode; break;
                   case 3: (*triangleElem)[j][5]  = newnode; break;
                   default : printf("error occurs in setting flag2 neighbour elements\n");  
             }
             newnode++;                    
          }
      }    
  }
  if (SurfElem == 1) {
     newnode += 3;
     (*triangleElem)[0][3] = 3;
     (*triangleElem)[0][4] = 4; 
     (*triangleElem)[0][5] = 5;       
  } 
  //printf("%d %d %d\n", (*triangleElem)[0][0], (*triangleElem)[0][1],(*triangleElem)[0][2]);
  void *node_share_Elem = malloc(newnode * 10 * sizeof(int));// triangle elements sharing a node
  int (*nodewithElem)[newnode][10] = node_share_Elem;
  for (i = 0; i < newnode; i++){
      for (j = 0; j < 10; j++){
          (*nodewithElem)[i][j] = -1; 
      }
  }
  void *node_nbrElemNum = malloc(newnode * sizeof(int));
  int (*node_nbrElem)[newnode] = node_nbrElemNum;
  for (i = 0; i < newnode; i++){
      (*node_nbrElem)[i] = 0;
  }
   
  void *N = malloc(newnode * 3 * sizeof(double)); //unit normal vectors of undeforme cell
  double (*Ngrid)[newnode][3] = N;
  for (i = 0; i < SurfElem; ++i){
      for (j = 0; j < 6; ++j){
          nodeIdx = (*triangleElem)[i][j];
          if (nodeIdx == -1) break;
          k = 0;
          while (k < 10){
            if (k == 9 && ((*nodewithElem)[nodeIdx][k] != -1)){
               // printf("i = %d, j = %d, k = %d, node = %d\n", i, j, k, nodeIdx);
                printf("The Elem matrix is not large enough\n");
                exit(1);
            }
            if ((*nodewithElem)[nodeIdx][k] == -1){
                 (*nodewithElem)[nodeIdx][k] = i;
                k = 10;
                (*node_nbrElem)[nodeIdx]++;
            }
            k++;
          }
      }
  }
 // for (i = 0; i < (*node_nbrElem)[0]; i++){
 //      printf("%d\n", (*nodewithElem)[22][i]); 
 // }
  void *node_nbr_nodes = malloc(maxElements * 10 * sizeof(int));
  int (*node_nbrNodes)[maxElements][10] = node_nbr_nodes; 
  for (i = 0; i < maxElements; i++){
      for (j = 0; j < 10; j++){
          (*node_nbrNodes)[i][j] = -1;
      }
  } 
  int elem, next_elem,flag;
  for (i = 0; i < maxElements; i++){
      elem = (*nodewithElem)[i][0];
      for (j = 0; j < 3; j++){
          if ((*triangleElem)[elem][j] == i) {
             if (j == 2) nodeIdx = (*triangleElem)[elem][0];
             else nodeIdx = (*triangleElem)[elem][j+1];
            // if (i == 0) printf("%d\n", nodeIdx);
             (*node_nbrNodes)[i][0] = nodeIdx;
          }
      }
      for (j = 1; j < (*node_nbrElem)[i]; j++){ 
          flag = 0;         
          for (k = j; k < (*node_nbrElem)[i]; k++){
              elem = (*nodewithElem)[i][k];
              for (l = 0; l < 3; l++){
                  if ((*triangleElem)[elem][l] == nodeIdx){
                     flag = 1;
                     if (l == 0) nodeIdx = (*triangleElem)[elem][2];
                     else nodeIdx = (*triangleElem)[elem][l-1];
                    // if (i == 0) printf("%d %d\n",j, nodeIdx);
                     (*node_nbrNodes)[i][j] = nodeIdx;
                     break;
                  }
              }
              if (flag) {
                  next_elem = (*nodewithElem)[i][j];
                  (*nodewithElem)[i][j] = (*nodewithElem)[i][k];
                  (*nodewithElem)[i][k] = next_elem;
                  break;              
              }
          }             
      }
   }
 //  for (i = 0; i < (*node_nbrElem)[0]; i++){
 //      printf("%d\n", (*nodewithElem)[22][i]); 
 //  }
 /* for (i = 0; i < newnode; i++){
      printf("elemnum[%d] = %d\n", i, (*node_nbrElem)[i]);      
      for (j = 0; j < 10; j++){          
          //printf("elemidx[%d][%d] = %d\n", i, j, (*nodewithElem)[i][j]);
      }
  }*/ 
  double Coord[3][3], E12[3], E13[3],N_elem[3], normN;
  for (i = 0; i < newnode; i++){
      (*Ngrid)[i][0] = 0;
      (*Ngrid)[i][1] = 0;
      (*Ngrid)[i][2] = 0;
      for (j = 0; j < 10; j++){
         // printf("elemIdx[%d][%d] = %d\n",i,j,(*nodewithElem)[i][j]);
          if ((*nodewithElem)[i][j] == -1) break;
          else{
              elemIdx = (*nodewithElem)[i][j];
             // printf("elemIdx = %d\n", (*nodewithElem)[i][j]);
              for (k = 0; k < 3; k++){
                  nodeIdx = (*triangleElem)[elemIdx][k];
                  Coord[k][0] = (*xRefGrid)[nodeIdx][0];
                  Coord[k][1] = (*yRefGrid)[nodeIdx][0];
                  Coord[k][2] = (*zRefGrid)[nodeIdx][0];                   
              }
             // (*node_nbrElem)[i]++;
             // double sum_ref1 = 0.0, sum_ref2 = 0.0;
              for (k = 0; k < 3; k++){
                  E12[k] = Coord[1][k] - Coord[0][k];
                  E13[k] = Coord[2][k] - Coord[0][k];
                 // sum_ref1 += E12[k] * E12[k];
                 // sum_ref2 += E13[k] * E13[k];
              } 
             // printf("sum1 = %f, sum2 = %f\n", sum_ref1, sum_ref2);
             /* for (k = 0; k < 3; k++){
                  E12[k] /= sqrt(sum_ref1);
                  E13[k] /= sqrt(sum_ref2);
              }*/
              N_elem[0] = E12[1] * E13[2] - E12[2] * E13[1];
              N_elem[1] = E12[2] * E13[0] - E12[0] * E13[2];
              N_elem[2] = E12[0] * E13[1] - E12[1] * E13[0];
              normN = sqrt(N_elem[0] * N_elem[0] + N_elem[1] * N_elem[1] + N_elem[2] * N_elem[2]);
              (*Ngrid)[i][0] += N_elem[0]/normN;
              (*Ngrid)[i][1] += N_elem[1]/normN;
              (*Ngrid)[i][2] += N_elem[2]/normN;
          }          
      }
      (*Ngrid)[i][0] /= (*node_nbrElem)[i];
      (*Ngrid)[i][1] /= (*node_nbrElem)[i];
      (*Ngrid)[i][2] /= (*node_nbrElem)[i];
      normN = sqrt((*Ngrid)[i][0] * (*Ngrid)[i][0] + (*Ngrid)[i][1] * (*Ngrid)[i][1] + (*Ngrid)[i][2] * (*Ngrid)[i][2]);
      (*Ngrid)[i][0] /= -normN;
      (*Ngrid)[i][1] /= -normN;
      (*Ngrid)[i][2] /= -normN;            
      //if (i == 1878) printf("%d,N[%d] = %e %e %e\n", (*node_nbrElem)[i],i, (*Ngrid)[i][0],(*Ngrid)[i][1],(*Ngrid)[i][2]);          
  }
  void *S0 = malloc(SurfElem * sizeof(double)); //unit normal vectors of undeforme cell
  double (*S0grid)[SurfElem] = S0;
  double S0_all = 0;
  double V0 = 0; 
  double Coord_center[3];
  for (i = 0; i < SurfElem; i++){
      for (k = 0; k < 3; k++){
           nodeIdx = (*triangleElem)[i][k];
           Coord[k][0] = (*xRefGrid)[nodeIdx][0];
           Coord[k][1] = (*yRefGrid)[nodeIdx][0];
           Coord[k][2] = (*zRefGrid)[nodeIdx][0];                   
      }
      for (k = 0; k < 3; k++){
           E12[k] = Coord[1][k] - Coord[0][k];
           E13[k] = Coord[2][k] - Coord[0][k];
           Coord_center[k] = 1/3.0 * (Coord[0][k] + Coord[1][k] + Coord[2][k]);
      }
      N_elem[0] = E13[1] * E12[2] - E13[2] * E12[1];
      N_elem[1] = E13[2] * E12[0] - E13[0] * E12[2];
      N_elem[2] = E13[0] * E12[1] - E13[1] * E12[0];
      normN = sqrt(N_elem[0] * N_elem[0] + N_elem[1] * N_elem[1] + N_elem[2] * N_elem[2]);
      N_elem[0] /= normN;
      N_elem[1] /= normN;
      N_elem[2] /= normN;    
      //if (i == 0) printf("%e %e %e\n", N_elem[0], N_elem[1], N_elem[2]);
      (*S0grid)[i] = 0.5 * normN;
      S0_all += (*S0grid)[i]; 
      V0 += 1/3.0 * (Coord_center[0] * N_elem[0] + Coord_center[1] * N_elem[1] 
                   + Coord_center[2] * N_elem[2]) * (*S0grid)[i];
     /* if (i == 0) {
         printf("%e %e %e\n", Coord_center[0] * 3, Coord_center[1] * 3, Coord_center[2] * 3);          
         printf("%e %e %e %e\n", N_elem[0], N_elem[1], N_elem[2], (*S0grid)[i]);
      }*/
      //if ((*V0grid)[i] < 0) printf("%d %e %e\n", i, (*S0grid)[i],(*V0grid)[i]);
  }
  //printf("S0 = %e, V0 = %e\n", S0_all, V0);
  
// Check PDB Reader
 // printf("Reader Done \n");
 //PSFReader(pPSFName, numOfCells,  maxElements,  bondLengths, X, Y, Z, elementType);

//Test bondLength
  //  FILE *bond;
  //  bond = fopen("bondLengths.data","wb+");
  /*for (i = 0; i < maxElements; ++i){
     for (j = 0; j < maxElements; ++j){
         double dx = (*xGrid)[i][0] - (*xGrid)[j][0];
         double dy = (*yGrid)[i][0] - (*yGrid)[j][0];
         double dz = (*zGrid)[i][0] - (*zGrid)[j][0];
         double dist = sqrt (dx * dx + dy * dy + dz * dz);
         if ((*bondgrid)[j][i] > 1e-7 && (dist - (*bondgrid)[j][i]) > 1e-7){
            printf("dist = %e, bondLengths = %e, %e, %d, %d\n", dist, (*bondgrid)[j][i], dist - (*bondgrid)[j][i], i, j);
         }
       // fwrite((&(*bondgrid)[j][i]), sizeof(double), 1, bond);
      }  
  }*/
 //fclose(bond);


/// END : HOST INITIALIZATION ///
  
  /// HOST ALLOCATION FOR LB ///
  // create spatial grids. 
  int lx = 81;       /// width node #, width = 80*dx
  int ly = 301;       /// height node #, height = 80*dx
  int lz = 21;     /// depth node #, depth = 600*dx

  printf("Simulation domain X[%f], Y [%f], Z[%f], dx = %f\n", 
           (lx-1)*dx, (ly-1)*dx, (lz-1)*dx, dx);

  void *receptBond = malloc(SurfElem * numReceptorsPerElem * 3 * sizeof(int));
  int (*gridreceptBond)[3][numReceptorsPerElem][SurfElem] = receptBond;
  for (i = 0; i < SurfElem; i++){
      for (j = 0; j < numReceptorsPerElem; j++){
          for (k = 0; k < 3; k++){
               if (k == 2) (*gridreceptBond)[k][j][i] = 0;
               else (*gridreceptBond)[k][j][i] = -1;
          }
      }
  }
  void *vWFbond = malloc(lx * ly * 4 * sizeof(int));
  int (*gridvWFbond)[4][ly][lx] = vWFbond;
  memset(vWFbond, -1, lx * ly * 4);
  
  int minNode = 0;
  for (i = 0; i < maxCells; ++i){
      for (j = 0; j < numOfElements[i]; ++j){
          if ((*zRefGrid)[j][i] < (*zRefGrid)[minNode][i])
             minNode = j;
      } 
  }
  int minElem = -1;
  for (i = 0; i < SurfElem; i++){
      for (j = 0; j < 3; j++){
         if((*triangleElem)[i][j] == minNode){
           minElem = i;
         }
      }
      if (minElem > -1) break;
  }
  //printf("minElem = %d\n", minElem);
  for (i = 0; i < 3; i++){
       nodeIdx = (*triangleElem)[minElem][0];
       Coord[i][0] = (*xRefGrid)[nodeIdx][0];
       Coord[i][1] = (*yRefGrid)[nodeIdx][0];
       Coord[i][2] = (*zRefGrid)[nodeIdx][0];                   
  }
  double recept_coord[3], r1 = (*receptor_r1_2D)[minElem][0], r2 = (*receptor_r2_2D)[minElem][0];
  for (i = 0; i < 3; i++){
     recept_coord[i] = (1-r1) * Coord[0][i] + r1 * (1-r2) * Coord[1][i] + r1 * r2 * Coord[2][i];
  }  
  i = (int)(recept_coord[0]/dx + 0.5);
  j = (int)(recept_coord[1]/dx + 0.5);
  (*gridreceptBond)[0][0][minElem] = i;
  (*gridreceptBond)[1][0][minElem] = j;
  (*gridreceptBond)[2][0][minElem] = 1;
  (*gridvWFbond)[0][j][i] = minElem * numReceptorsPerElem;
     
  void *fIN[19];
  void *fOUT[19];

  for (i = 0; i < 19; ++i) {
   fIN[i]  = malloc(lx * ly * lz * sizeof(double));
   fOUT[i] = malloc(lx * ly * lz * sizeof(double)); 
  }
  void *ux = malloc(lx * ly * lz * sizeof(double));
  void *uy = malloc(lx * ly * lz * sizeof(double));
  void *uz = malloc(lx * ly * lz * sizeof(double));
  void *rho = malloc(lx * ly * lz * sizeof(double));
  void *obstacle = malloc(lx * ly * lz * sizeof(float));
  void *Fx = malloc(lx * ly * lz * sizeof(double));
  void *Fy = malloc(lx * ly * lz * sizeof(double));
  void *Fz = malloc(lx * ly * lz * sizeof(double));

  double (*gridX)[lz][ly][lx] = ux;
  double (*gridY)[lz][ly][lx] = uy;
  double (*gridZ)[lz][ly][lx] = uz;
  double (*gridR)[lz][ly][lx] = rho;
  float (*gridO)[lz][ly][lx] = obstacle;

  double (*gridIN)[lz][ly][lx];
  double (*gridOUT)[lz][ly][lx];

  double (*gridFx)[lz][ly][lx] = Fx;
  double (*gridFy)[lz][ly][lx] = Fy;
  double (*gridFz)[lz][ly][lx] = Fz;

  /// BEGIN: INITIALIZATION OF VALUES ///

  /// LB : INIT  ///
  double f0[] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
  double cx[] = {0, 1, -1, 0, 0,  0,  0,  1, 1, 1,  1,-1, -1, -1, -1, 0,  0,  0,  0};
  double cy[] = {0, 0, 0, 1, -1,  0,  0,  1,-1, 0,  0, 1, -1,  0,  0, 1,  1, -1, -1};
  double cz[] = {0, 0, 0, 0,  0,  1, -1,  0, 0, 1, -1, 0,  0,  1, -1, 1, -1,  1, -1};

  // % INITIAL CONDITION: Linear shear flow at equilibrium
  // obstacle
  //  INIT. Walls of Flow Channel
  for (i = 0; i < lz; ++i)
    for (j = 0; j < ly; ++j)
      for (k = 0; k < lx; ++k)
      {
         if (i == 0 || i == lz - 1){
            (*gridO)[i][j][k] = 1.0;
            (*gridFx)[i][j][k] = 0.0;
            (*gridFy)[i][j][k] = 0.0;
            (*gridFz)[i][j][k] = 0.0;
             
           }
         else{ 
          (*gridO)[i][j][k] = 0.0;
          (*gridFx)[i][j][k] = 0.0;
          (*gridFy)[i][j][k] = 0.0;
          (*gridFz)[i][j][k] = 0.0;
         }
      }

  // END::: INIT. Walls of Flow Channel



  /// init flow.
  double SHEAR_RATE = hostSEMparameters[5]; //shear rate (sec^-1)
  double UMAX = SHEAR_RATE*(lz-1)*dx;   /// maximum velocity of linear shear flow (micron/sec)
  const double INIT_RHO = hostSEMparameters[6]; /// picogram/micron^3
  hostSEMparameters[3] = 0.0165 * exp(0.00065 * SHEAR_RATE); //Kf0
  double C = dx/dt/UMAX;
  double C2 = C * C;
  
  double L0 = 1;   //L0 = 1 micron
  double t0 = L0/UMAX; //unit: sec
  double M0 = INIT_RHO * L0 * L0 * L0; //unit: picogram
  double F0 = 1;//unit: nN
  double P0 = INIT_RHO * UMAX * UMAX;
  printf("t0 = %e, M0 = %e, F0 = %e\n", t0, M0, F0);

  hostSEMparameters[7] = L0;
  hostSEMparameters[8] = t0;
  hostSEMparameters[9] = M0;
  hostSEMparameters[10] = F0;  
  hostSEMparameters[11] = UMAX;
  hostSEMparameters[12] = 1.2e6/INIT_RHO; //kinematic viscosity  unit: micron^2/sec
  hostSEMparameters[13] = ((3.0 * hostSEMparameters[12]*dt)/(dx*dx)) + 0.5;// relaxation time
  hostSEMparameters[14] = P0; // unit: nN/micron^2
  hostSEMparameters[15] = C;
  hostSEMparameters[16] = 1e12;//UNIT_FACTOR
  double KbT =  1.3806488e-8 * 300; 
  hostSEMparameters[17] = 600 * KbT /2.0/Lm/Lm;
  hostSEMparameters[18] = 600 * KbT /2.0/Lm/Lm/Lm;
  hostSEMparameters[19] = 500 * KbT * 1e-6;
  for (i = 0; i < lz; ++i)
    for (j = 0; j < ly; ++j)
      for (k = 0; k < lx; ++k) 
      { 
            (*gridX)[i][j][k] = 0.0;
            (*gridY)[i][j][k] = SHEAR_RATE * i * dx/UMAX;
            (*gridZ)[i][j][k] = 0.0; //micron/sec
            (*gridR)[i][j][k] = 1.0; /// picogram/micron^3
        
      }

  ///// init. density distribution function
  for (d = 0; d < 19; ++d) {
    gridIN = (void *)fIN[d];
    gridOUT = (void *)fOUT[d];
    for (i = 0; i < lz; ++i)
      for (j = 0; j < ly; ++j)
        for (k = 0; k < lx; ++k) {
          cu = 3.0 * (cx[d] * (*gridX)[i][j][k] + cy[d] * (*gridY)[i][j][k] + cz[d] * (*gridZ)[i][j][k]);
          (*gridIN)[i][j][k] = (*gridR)[i][j][k] * f0[d] * ( 1.0 + cu/C + 0.5*cu*cu/C2 -
                                  1.5* ((*gridX)[i][j][k] * (*gridX)[i][j][k] + 
                               (*gridY)[i][j][k] * (*gridY)[i][j][k] + (*gridZ)[i][j][k] * (*gridZ)[i][j][k])/C2);
          (*gridOUT)[i][j][k] = 0.0;
      }
  }
  
  // Unit factor
  hostLBparameters[0] = 1e12;
  // maximum velocity of linear flow
  hostLBparameters[1] = UMAX;
  // vertical lid velocity 
  hostLBparameters[2] = 0.0;
  // Reynolds number
  hostLBparameters[3] = 100.0;
  // kinematic viscosity 
  // hostLBparameters[4] = ((hostLBparameters[1] * 2.0 * hostLBparameters[0]) / hostLBparameters[3]);
  hostLBparameters[4] = 1.2e6/INIT_RHO;   //// water = 1000 micron^2/milli-sec
  // relaxation parameter  should > 0.5
  // hostLBparameters[5] = (1.0 / (3.0 * hostLBparameters[4] + 0.5));
  hostLBparameters[5] = ((3.0 * hostLBparameters[4]*dt)/(dx*dx)) + 0.5;
  // hostLBparameters[5] = 2.5;
  /// dt = (tau - 0.5)*dx^2/(3*mu); 
  /// Ma = v_max/C << 1; where C = dx/dt;
  /// dt < 0.25 to keep Ma < 0.1
  hostLBparameters[6] = INIT_RHO;

  printf("kinematic viscosity = %e, relaxation parameter = %e\n", hostLBparameters[4], hostLBparameters[5]);
  printf("dx/dt = %f, Ma = %f\n", dx/dt, UMAX/(dx/dt));

  /// add code for simulating fiber network
  fiber_grid = init_fibrin_network(dt); 
  fiber_allocGPUKernel(NULL, fiber_grid->maxNodes, fiber_grid->maxLinks,
                           fiber_grid->max_N_conn_at_Node, dt, NULL); 
  printf("WARNING: exit in  , after init_fibrin_network()\n");
  exit(0);

/// END : HOST INITIALIZATION ///

/// BEGIN:  DEVICE ALLOCATION and INITIALIZATION  ///

  // ALLOC and init SEM on GPU

  void *LBgrids;
  LBgrids = fluid_allocGPUKernel(NULL, dt, dx, lx, ly, lz);
  fluid_initGPUKernel(NULL, LBgrids, 1, hostSEMparameters, fIN, fOUT, ux, uy, uz, rho, obstacle, Fx, Fy, Fz, vWFbond);

  void *SEMgrids = sem_allocGPUKernel(NULL, maxCells, maxElements, SurfElem, newnode, numReceptorsPerElem, dt, S0_all, hostSEMparameters);
  sem_initGPUKernel(NULL, SEMgrids, numOfCells, numOfElements, SurfElem, numReceptorsPerElem, X_Ref, Y_Ref, Z_Ref, 
                    X, Y, RY, Z, VX, VY, VZ, FX, FY, FZ, elementType, bondLengths, 
                    triElem, receptor_r1, receptor_r2, node_share_Elem, N, node_nbrElemNum, node_nbr_nodes,
                    S0, V0, receptBond, randNum);//pass force velocity

/// END:  DEVICE ALLOCATION and INITIALIZATION  ///

/// BEGIN : OUTPUT DATA FILE CREATION ///
  // LB data files
  FILE *uxFile, *uyFile, *uzFile, *fxFile, *fyFile, *fzFile, *rhoFile, *obstFile, *fINFile, *fOUTFile;
  uxFile = fopen("fluid3d_ux.dat", "wb+");
  uyFile = fopen("fluid3d_uy.dat", "wb+");
  uzFile = fopen("fluid3d_uz.dat", "wb+");
 // fxFile = fopen("fluid3d_fx.dat", "wb+");
 // fyFile = fopen("fluid3d_fy.dat", "wb+");
 // fzFile = fopen("fluid3d_fz.dat", "wb+");
 // fINFile = fopen("fluid3d_fIN.dat", "wb+");
 // fOUTFile = fopen("fluid3d_fOUT.dat", "wb+");
 
  rhoFile = fopen("fluid3d_rho.dat", "wb+");
  obstFile = fopen("fluid3d_obst.dat", "wb+");
  
  /*FILE *tecplotFile;
  tecplotFile = fopen("fluid3d_tec_0.dat","wb+");
  fprintf(tecplotFile,"VARIABLES=\"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\"\n");
  fprintf(tecplotFile,"ZONE I = %d, J = %d, K = %d\n", lx, ly, lz);*/
  // SEM data files
  FILE *dataFiles[7];

  dataFiles[0] = fopen("Output/cellsPos.data", "wb+");
  dataFiles[1] = fopen("Output/cellsVel.data", "wb+");
  dataFiles[2] = fopen("Output/cellsForce.data", "wb+");
  dataFiles[3] = fopen("Output/cellsPosASCII.dat", "wb+"); 
  dataFiles[4] = fopen("Output/cellsVelASCII.dat", "wb+"); 
  dataFiles[5] = fopen("Output/cellsForceASCII.dat", "wb+");
  dataFiles[6] = fopen("Output/cellsPauseTime.dat", "a+");
  // debug file
 // FILE *cent_fp[3];
 // cent_fp[0] = fopen("Output/center_line_ux.dat", "wb+");
 // cent_fp[1] = fopen("Output/center_line_uy.dat", "wb+");
 // cent_fp[2] = fopen("Output/center_line_uz.dat", "wb+");

  // WRITE INITIAL DATA LB

  /* for (i = 0; i < lz ; ++i) {
    for (j = 0; j < ly; ++j) {
      for (k = 0; k < lx; ++k) {
      //  [grid dumpGrid: @"ux" toFile: uxFile];
      //  [grid dumpGrid: @"uy" toFile: uyFile];
      //  [grid dumpGrid: @"uz" toFile: uzFile];
      //  [grid dumpGrid: @"rho" toFile: rhoFile];
      //  [grid dumpGrid: @"obstacle" toFile: obstFile];

      fwrite(&((*gridX)[i][j][k]), sizeof(double), 1, uxFile);
      fwrite(&((*gridY)[i][j][k]), sizeof(double), 1, uyFile);
      fwrite(&((*gridZ)[i][j][k]), sizeof(double), 1, uzFile);
   //   fwrite(&((*gridFx)[i][j][k]), sizeof(double), 1, fxFile);
    //  fwrite(&((*gridFy)[i][j][k]), sizeof(double), 1, fyFile);
    //  fwrite(&((*gridFz)[i][j][k]), sizeof(double), 1, fzFile);
    //  fwrite(&((*gridR)[i][j][k]), sizeof(double), 1, rhoFile);
     // fwrite(&((*gridO)[i][j][k]), sizeof(float), 1, obstFile);
      // for (d = 0; d < 19; d++){
        //     gridIN = (void *)fIN[d];
          //   gridOUT = (void *)fOUT[d];
            // fwrite(&((*gridIN)[i][j][k]), sizeof(double), 1, fINFile);
            // fwrite(&((*gridOUT)[i][j][k]), sizeof(double), 1, fOUTFile);
       //  }
      fprintf(tecplotFile,"%d, %d, %d, %e, %e, %e\n", k, j, i, (*gridX)[i][j][k], (*gridY)[i][j][k], (*gridZ)[i][j][k]);
       }
     }
  }
  fclose(tecplotFile);*/
  // WRITE INITIAL DATA SEM
    fprintf(dataFiles[3], "Timestep       ElemID        X         Y         Z\n");
    fprintf(dataFiles[4], "Timestep       ElemID        X         Y         Z\n");
    fprintf(dataFiles[5], "Timestep       ElemID        X         Y         Z\n");
  
  for (j = 0; j < numOfCells; ++j) {
    for (i = 0; i < numOfElements[j]; ++i) {
     //
     // We write an additional z value for the puropse of alignment in the octal dump
     //
     // write initial positions
      fwrite(&((*xGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
      fwrite(&((*ryGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
      fwrite(&((*zGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
      fwrite(&((*zGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
      fprintf(dataFiles[3], "0      %d    %e   %e   %e\n ", 
              i, (*xGrid)[i][j], (*ryGrid)[i][j], (*zGrid)[i][j]);
      //fwrite(&((*etGrid)[i][j]), sizeof(int), 1, dataFiles[0]);

     // write initial velocities
      fwrite(&((*xVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
      fwrite(&((*yVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
      fwrite(&((*zVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
      fwrite(&((*zVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
      fprintf(dataFiles[4], "0      %d    %e   %e   %e\n ", 
              i, (*xVGrid)[i][j], (*yVGrid)[i][j], (*zVGrid)[i][j]);
     // write initial Forces
      fwrite(&((*xFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);
      fwrite(&((*yFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);
      fwrite(&((*zFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);
      fwrite(&((*zFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);
      fprintf(dataFiles[5], "0      %d    %e   %e   %e\n ", 
              i, (*xFGrid)[i][j], (*yFGrid)[i][j], (*zFGrid)[i][j]);
    }
  }
   
  /*for (i = 0; i < 10000; i++){
     double rand_t = RAND_NUM;
     if (rand_t < 1e-3)
        printf(" %d %e\n",i, rand_t);
  }*/
  

/// END : OUTPUT DATA FILE CREATION ///

/////////////////////////////////////
/////////////////////////////////////
//////////// BEGIN : SIMULATION /////
/////////////////////////////////////
/////////////////////////////////////

  /// COPY DEVICE DATA TO HOST DATA ///
  int done = 0;
  int t = 0;
  int totalT = 0;
  int fluidTimeSteps = 1;
  int SEMTimeSteps = 1;
  double gama = SHEAR_RATE * dt;
  //outputSave(totalT, lx, ly, lz, rho, 1, ux, uy, uz, 1, "Output", "LBMflow");

  while (!done) {
    // invoke GPU
  // if (totalT >= 300) outputFrequency = 10;
   //if (totalT >= 330) outputFrequency = 1;
    // if (totalT >= 3880) outputFrequency = 1;
    for(t = 0; t < outputFrequency; t++)
    {
       if (!done){
          sem_invokeGPUKernel_Force(NULL, SEMgrids,SEMTimeSteps, &done, &totalT, r, gama); //Calculate the force acting on elements
          fluid_invokeGPUKernel(NULL, LBgrids, SEMgrids, (double*)randNum, r,  fluidTimeSteps);     //Calculate fluid LB

          exit(0);
       }
       else break;
    }

    // are we done?
    //totalT += outputFrequency;
    //if (totalT >= timeSteps) done = 1;
        /// COPY DEVICE DATA TO HOST DATA ///
    fluid_initGPUKernel(NULL, LBgrids, 0, hostLBparameters, fIN, fOUT, ux, uy, uz, rho, obstacle, Fx, Fy, Fz, vWFbond);
    sem_copyGPUKernel(NULL, SEMgrids, X, Y, RY, Z, VX, VY, VZ, FX, FY, FZ, outputFrequency);

    /// BEGIN : DATA OUTPUT  ///

    // output SEM DATA
    printf("Time Step = %d, time = %e\n", totalT, totalT*dt);
    //    fwrite(&(numOfCells), sizeof(int), 1, dataFiles[0]);
    // new output file formated (x y z)
    thisFrame = totalT / outputFrequency;
    int nCell, nElem;
    for (j = 0; j < numOfCells; ++j) {
        for (i = 0; i < numOfElements[j]; ++i) {
        
            fwrite(&((*xGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
       	   // fwrite(&((*xVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
	   // fwrite(&((*xFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);

            // fprintf(pFile, "%f ", (*xGrid)[i][j]);
	    fwrite(&((*ryGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
	   // fwrite(&((*yVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
	   // fwrite(&((*yFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);

	    fwrite(&((*zGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
	   // fwrite(&((*zVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
	   // fwrite(&((*zFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);

	    //fwrite(&((*etGrid)[i][j]), sizeof(int), 1, dataFiles[0]);
            //  We added the extra output...for alignment purposes in the octal dump
	    fwrite(&((*zGrid)[i][j]), sizeof(double), 1, dataFiles[0]);
	   // fwrite(&((*zVGrid)[i][j]), sizeof(double), 1, dataFiles[1]);
	   // fwrite(&((*zFGrid)[i][j]), sizeof(double), 1, dataFiles[2]);
              fprintf(dataFiles[3], "%d      %d  %e  %e  %e\n ", 
                      totalT, i, (*xGrid)[i][j], (*ryGrid)[i][j], (*zGrid)[i][j]);
              fprintf(dataFiles[4], "%d      %d  %e  %e  %e\n ", 
                      totalT, i, (*xVGrid)[i][j], (*yVGrid)[i][j], (*zVGrid)[i][j]);
              fprintf(dataFiles[5], "%d      %d  %e  %e  %e\n ", 
                      totalT, i, (*xFGrid)[i][j], (*yFGrid)[i][j], (*zFGrid)[i][j]);// (*xFGrid)[i][j]*(*xFGrid)[i][j] + (*yFGrid)[i][j]*(*yFGrid)[i][j] +(*zFGrid)[i][j]*(*zFGrid)[i][j]);
              //if (i == 0) printf ("%e %e %e\n", (*xVGrid)[i][j], (*yVGrid)[i][j], (*zVGrid)[i][j]);
              //if (i == 0) printf ("%e %e %e\n", (*xFGrid)[i][j], (*yFGrid)[i][j], (*zFGrid)[i][j]);
          }
        }

        //// Output data to VTK, visualized by paraview
       // outputSave(totalT, lx, ly, lz, rho, 1, ux, uy, uz, 1, "Output", "LBMflow");
    /* char filename[128] = "fluid3d_tec_";
     sprintf(filename,"fluid3d_tec_%d.dat",totalT);
     tecplotFile = fopen(filename,"wb+");
     fprintf(tecplotFile,"VARIABLES=\"X\" \"Y\" \"Z\" \"U\" \"V\" \"W\"\n");
     fprintf(tecplotFile,"ZONE I = %d, J = %d, K = %d\n", lx, ly, lz);

     for (i = 0; i < lz ; ++i) {
          for (j = 0; j < ly; ++j) {
              for (k = 0; k < lx; ++k) {
           // fwrite(&((*gridX)[i][j][k]), sizeof(double), 1, uxFile);
           // fwrite(&((*gridY)[i][j][k]), sizeof(double), 1, uyFile);
           // fwrite(&((*gridZ)[i][j][k]), sizeof(double), 1, uzFile);
           // fwrite(&((*gridFx)[i][j][k]), sizeof(double), 1, fxFile);
           // fwrite(&((*gridFy)[i][j][k]), sizeof(double), 1, fyFile);
           // fwrite(&((*gridFz)[i][j][k]), sizeof(double), 1, fzFile);
           // fwrite(&((*gridR)[i][j][k]), sizeof(double), 1, rhoFile);
       //     fwrite(&((*gridO)[i][j][k]), sizeof(float), 1, obstFile);
       //        for (d = 0; d < 19; d++){
         //          gridIN = (void *)fIN[d];
         //          gridOUT = (void *)fOUT[d];
         //          fwrite(&((*gridIN)[i][j][k]), sizeof(double), 1, fINFile);
         //          fwrite(&((*gridOUT)[i][j][k]), sizeof(double), 1, fOUTFile);
              fprintf(tecplotFile,"%d, %d, %d, %e, %e, %e\n", k, j, i, 
                     (*gridX)[i][j][k], (*gridY)[i][j][k], (*gridZ)[i][j][k]);
              
                
             }
           }
        }
     fclose(tecplotFile);   */
        
    /// END : DATA OUTPUT  ///

  } //// while (!done){}
  /// END : SIMULTAION ///
  fprintf(dataFiles[6], "%e, %d\n", totalT *dt, seed); 
  
  /// BEGIN: RELEASE DATA AND CLOSE FILES ///
  // release GPU
  gsl_rng_free(r);
  sem_releaseGPUKernel(NULL, SEMgrids);
  fluid_releaseGPUKernel(NULL, LBgrids);

  // write out final data
  fclose(dataFiles[0]);
  fclose(dataFiles[1]);
  fclose(dataFiles[2]);
  fclose(dataFiles[3]);
  fclose(dataFiles[4]);
  fclose(dataFiles[5]);
  fclose(dataFiles[6]);

  fclose(uxFile);
  fclose(uyFile);
  fclose(uzFile);
 // fclose(fxFile);
 // fclose(fyFile);
 // fclose(fzFile);
 // fclose(fINFile);
 // fclose(fOUTFile);
  fclose(rhoFile);
  fclose(obstFile);
 // fclose(cent_fp[0]);
 // fclose(cent_fp[1]);
 // fclose(cent_fp[2]);

  /// END: RELEASE DATA AND CLOSE FILES ///
}

int main( int argc, char** argv) 
{
    int dflag = 0; // used for  option argument for device specification
    int gpuDevice;
    int devNum = 0;
    int c;
    // cudaError_t cudareturn;
    int cudareturn = 0;
    //used to handle options passed to program
    while ((c = getopt (argc, argv, "d:")) != -1)
    {
        switch (c)
        {
        case 'd':
            dflag = 1;
            devNum = atoi(optarg);
        break;
        case '?':
            if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
            return 1;
        default:
            // abort ();
            printf("GPU device not specified using device 0 ");
       }
    }

    cudareturn = cudaSetDevice( devNum );
    printf("cudaSetDevice()=%d, cudareturn = %d\n", devNum, cudareturn); 

    //    if (cudareturn == cudaErrorInvalidDevice)
    if (cudareturn == 11 )
    {
       printf("cudaSetDevice returned  11, invalid device number ");
       exit(11);
    }
    else
    {
       cudaGetDevice( &gpuDevice );
       printf("cudaGetDevice()=%d\n", gpuDevice);
    }
    //printf("Pre Run \n");
    int seed;
    for (seed = 1101; seed <= 1300; seed++)
      runGPUSimulation(seed);
}

static void outputSave(int t, int nx, int ny, int nz, void *rho, int write_rho,
          void *ux, void *uy, void *uz, int write_vel, char *directory, char *filename)
{
    writeVTK(t, nx, ny, nz, rho, write_rho, ux, uy, uz, write_vel, directory, filename);

    //Calculate Mega Lattice Site Update per second MLSU/s
    // Speed = (nx*ny*nz)*(t-step_now)/((clock() - time_now)/CLOCKS_PER_SEC)/1000000.0;
    // step_now = t;
    // time_now = clock();
    // if (mass == 0) printf("t=%d\tSpeed=%f MLUP/s\n", t, Speed);
    // else printf("t=%d\tSpeed=%f MLUP/s mass=%f\n", t, Speed, mass);
}

static void writeVTK(int t, int nx, int ny, int nz, 
            void *rho, int write_rho, void *ux, void *uy, void *uz, int write_vel, char *directory, char *filename)

{
    int x,y,z,dir;
    char dataFileName[255];
    FILE *dataFile;

    float (*gridX)[nz][ny][nx] = ux;
    float (*gridY)[nz][ny][nx] = uy;
    float (*gridZ)[nz][ny][nx] = uz;
    float (*gridR)[nz][ny][nx] = rho;

#ifdef WIN32
    dir = mkdir(directory);
#else
    dir = mkdir(directory,0777);
#endif

    sprintf(dataFileName,"%s/%s_%07d.vti",directory,filename,t);
    dataFile = fopen(dataFileName,"w");
    fprintf(dataFile, "<?xml version=\"1.0\"?>\n");
    fprintf(dataFile, "<!-- gpuLBMflow ND ACMS -->\n");
    fprintf(dataFile, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(dataFile, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",nx-1,ny-1,nz-1);
    fprintf(dataFile, "  <Piece Extent=\"0 %d 0 %d 0 %d\">\n",nx-1,ny-1,nz-1);
    fprintf(dataFile, "    <PointData Scalars=\"scalars\">\n");

    //write density
    if (write_rho != 0)
    {
        fprintf(dataFile, "      <DataArray type=\"Float32\" Name=\"Density\" NumberOfComponents=\"1\" format=\"ascii\">\n");
        for (z=0; z<nz; z++)
        {
            for (y=0; y<ny; y++)
            {
                for (x=0; x<nx; x++)
                {
                    fprintf(dataFile,"%.4e ", (*gridR)[z][y][x]);
                }
                fprintf(dataFile, "\n");
            }
        }
        fprintf(dataFile, "      </DataArray>\n");
    }


    //fprintf(dataFile, "    <PointData Vectors=\"Velocity\">\n");
    //write velocity
    if (write_vel != 0)
    {
        fprintf(dataFile, "      <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (z=0; z<nz; z++)
        {
            for (y=0; y<ny; y++)
            {
                for (x=0; x<nx; x++)
                {
                    fprintf(dataFile,"%.4e ", (*gridX)[z][y][x]);
                    fprintf(dataFile,"%.4e ", (*gridY)[z][y][x]);
                    fprintf(dataFile,"%.4e ", (*gridZ)[z][y][x]);
                }
                fprintf(dataFile, "\n");
            }
        }
        fprintf(dataFile, "      </DataArray>\n");
    }
    //fprintf(dataFile, "    </PointData>\n");

    fprintf(dataFile, "    </PointData>\n");

    fprintf(dataFile, "    <CellData>\n");
    fprintf(dataFile, "    </CellData>\n");
    fprintf(dataFile, "  </Piece>\n");
    fprintf(dataFile, "  </ImageData>\n");

    fprintf(dataFile, "</VTKFile>\n");
    fclose(dataFile);
}
