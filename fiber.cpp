#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "GPUDefines.h"
#include "network.h"

#define  LEN_UNIT    0.1426;   //// The scale factor for fiber length is s = 0.1426 micron/unit.        
#define  F_X_SHIFT   120.0     /// shift fiber data to make it in the domain center
#define  F_Y_SHIFT   0.0       /// shift fiber data to make it in the domain center
#define  F_Z_SHIFT   120.0     /// shift fiber data to make it in the domain center

static  double f0[] = {1.0/3.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
                1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
static  double cx[] = {0, 1, 0, 0, -1,  0,  0,  1, -1, -1,  1, 1, -1, -1,  1, 0,  0,  0,  0};
static  double cy[] = {0, 0, 1, 0,  0, -1,  0,  1,  1, -1, -1, 0,  0,  0,  0, 1, -1, -1,  1};
static  double cz[] = {0, 0, 0, 1,  0,  0, -1,  0,  0,  0,  0, 1,  1, -1, -1, 1,  1, -1, -1};
static  double Cs, Cs2;


static  double  Linear_interp(double, double*);
static  double   Bilinear_interp(double*, double*);
static  double   Trilinear_interp(double*, double*);
static  double   DH1D(double x,double dh);
static  double   wormlike_chain_force(double,double,double,double);

/**
 Linear interpolation formula: 0 <= t <= 1; 
 v[0] is lower side value; v[1] is upper side value;
 Interpolated value V =  t*v[1] + (1.0-t)*v[0];
**/
static double Linear_interp(double t, double *v)
{
    return (t*v[1] + (1.0-t)*v[0]);
}

/** 
 BiLinear interpolation, linear interpolation in 2D
   t[2]  - a 2D point (X,Y), 0 <= t[0] <= 1 and 0 <= t[1] <= 1
   v[4]  - an array of size 4 containg values cockwise around the square starting from bottom left
   performs 3 linear interpolations
**/
static  double Bilinear_interp(double *t, double *v)
{
      double v_tmp[2], inv[2]; 

      inv[0] = v[0]; inv[1] = v[1];
      v_tmp[0] = Linear_interp(t[1], inv);

      inv[0] = v[3]; inv[1] = v[2];
      v_tmp[1] = Linear_interp(t[1], inv);

      return Linear_interp(t[0], v_tmp);
}

/* 
 TriLinear interpolation, linear interpolation in 2D
 t[3]  - a 3D point (X,Y,Z) 0 <= t[0] <= 1; 0 <= t[1] <= 1; 0 <= t[2] <= 1 

 Let x-y plane be horizontal and z-axis be vertical. 
 v[8]  - an array of size 8 containg the values of the 8 corners 
        of a cube defined as two faces: 0-3 face z = 0 (bottom face) 
                                        4-7 face z = 1 (top face)
 When view from top (negative z-direction), 0-3 in clockwise direction around the square(z = 0) starting from bottom left.
 4-7 in clockwise direction around the square(z = 1) starting from bottom left.
 performs 7 linear interpolations
*/
static  double Trilinear_interp(double *t, double *v)
{
      double  v_tmp[2];  

      v_tmp[0] = Bilinear_interp(t, &(v[0]));
      v_tmp[1] = Bilinear_interp(t, &(v[4]));
      return Linear_interp(t[2], v_tmp);
}


static double DH1D(double x,double dh)
{
        double r,r_h;

        r=fabs(x);
        r_h=r/dh;
        /***
        if(r<dh)
            return (3-2*r_h+sqrt(1+4*r_h-4*r_h*r_h))/8.0/dh;
        else
        {
            if(r<2*dh)
                return (5-2*r_h-sqrt(-7+12*r_h-4*r_h*r_h))/8.0/dh;
            else
                return 0.0;
        }
        ***/
        if(r<2.0*dh)  
        {
            (1.0 + cos(PI*r_h/2.0) )/(4.0*dh);
        }
        else
            return 0.0;
}

// revise the init_fibrin_network type from void to tmp_fiber_GUPgrids  by shixin Xu 
//void *init_fibrin_network(double dt)
tmp_fiber_GPUgrids *init_fibrin_network(double dt) 
{
        // char FibrinName[] = "Input/KRM2.txt";
        char    FibrinName[] = "Input/HemoThickInputCor.txt";
        FILE    *FibrinIn;
        int     i, j, k, n_link_sz, total_N_fiber = 0;
        double   f_len, thickness;
        double   lside[3] = {1.0e8, 1.0e8, 1.0e8}, uside[3] = {-1.0e8, -1.0e8, -1.0e8};
        int     max_Nnode_connect = 0, node_N_conn, *tmp_N_Conn_at_Node;

        network myNetwork;
        tmp_fiber_GPUgrids *cpu_fibers; 

        // ~~~~~~ READ and STORE INPUT DATA ~~~~~~~~~~~~~~~~//
        // Be careful, the node index in the data file begins with "1" instead of "0".  
        string line;
        ifstream Input;

        Input.open(FibrinName); //SampleC.txt, KRM2, HemoThickInputCor
        // check
        if(!Input.is_open())
        {
            std::cout << "init_fibrin_network() cannot open file: " << FibrinName  << std::endl;
            exit(0);
        }

        // Temporary storage vector to read data
        vector <int> NConnect;
        vector <int> nodeNumber;

        NConnect.clear();

        myNetwork.linkr.clear(); // node based network's connection information
        myNetwork.fibersize.clear();
        myNetwork.branchp.clear();
        myNetwork.node_N_conn.clear();

        // Read new format file with thickness
        while(getline(Input,line)){

                vector <double> dat; // reading first line
                dat.clear();//clear
                vector < int > Connect; //linking, connection information
                vector < double > Fsize; // fibrin information, thickness, size

                Connect.clear();
                Fsize.clear();

                double value;

                istringstream iss(line); // line in the file

                //read data
                while(iss >> value)
                {
                //cout << value << endl;
                    dat.push_back(value);   // store the whole line in the value, dat
                }

                //get size of data
                const unsigned int dasz = dat.size();

                //record node number
                // "-1" so that node index starts with "0". 
                nodeNumber.push_back(dat[0]-1);  // first component of the line

                // stores coordinates of each node
                const double aa = 1.0;
                myNetwork.putIntoVec(dat[1]/aa,dat[2]/aa,dat[3]/aa);

                // stores number of connections
                NConnect.push_back(dat[4]);
                if(dat[4] > max_Nnode_connect)
                    max_Nnode_connect = (int)(dat[4]);

                // stores list of connecting node
                Connect.push_back(dat[0]-1);

                node_N_conn = 0;
                for(int jj = 5; jj < dat[4]+5; jj++)
                {   // 5 is the first of the connecting node, th column in the file
                    // exclude connecting node if it is the same as node number
                    if( dat[jj] != dat[0]) 
                    {
                        Connect.push_back(dat[jj]-1);
                        node_N_conn++;
                    }
                }
             
                // store number of connections at the node
                myNetwork.node_N_conn.push_back(node_N_conn);

                // store list of connections
                myNetwork.linkr.push_back(Connect);

                // stores fiber thickness for all connections of a particular node
                //Fsize.push_back(dat[0]);
                for(int kk = dat[4] + 5; kk < dasz;kk++)
                { // dat[4] is the number ofconnections, 5 is the first column of connected node
                    const unsigned int te = kk - dat[4];// uniportant..trying
                     //cout << dat[te] << " " << dat[0] <<  " " << kk << endl;
                    if( dat[te]  != dat[0])
                    { //uniportant
                        Fsize.push_back(dat[kk]/1.0); //this is working, elementary thickness info
                    }
                }
                //store fiber thickness
                myNetwork.fibersize.push_back(Fsize); // container of thickness, vector of values
        }
        Input.close();

        std::cout<< "max N of links at nodes = " << max_Nnode_connect << endl;

        // ~~~~~~ SET PARAMETERS and RECORD INITIAL STATE OF NETWORK ~~~~~~~~~~~~~~~~//
        const int     branch_sz = myNetwork.branchp.size();
                           /// Network branch point size = 7111; which is the number of nodes of the network
        /////// Convert data to ones with physical units.    
        /****
        for(i = 0; i < branch_sz; i++)
        {
            cout <<"branch_pt[" << nodeNumber[i] <<"]_crds = [" <<  myNetwork.branchp[i].x  << "," << 
                    myNetwork.branchp[i].y <<","<< myNetwork.branchp[i].z << "]" << endl;
        }
        ***/  

        // store coordinates of node into 1-D vector
        myNetwork.storeCoords(); // stores, #node, x,y,z; need to read about vector class
        
        // get network statistics: average thickness and number of branching points
        myNetwork.getstat();
        
        /// INIT memory of fibrin network structure for GPUs. For Input/KRM2.txt
        const unsigned int sz = myNetwork.node.size();
                           //// Network Node size = 21333 = branch_sz * 3
        int OLink, NLink, linksz;
        OLink = myNetwork.numberoffibers(0);
        linksz = myNetwork.linkr.size();
        cout <<"Network Node size = "<< sz <<" Network branch point size = " << branch_sz <<
                          " Number of fibers(multiple-counted) = " << OLink <<" linker size = " << linksz  << endl;
  
        //myNetwork.getBNodes();
        myNetwork.newvel.clear();
        myNetwork.newvel.resize(sz,0);
     
        cout << "fiber pixel: xmin " << myNetwork.getBCs(0,0)<< "; xmax " << myNetwork.getBCs(1,0) << endl; //three coordinates x,y,z bc microns
        cout << "fiber pixel: ymin " << myNetwork.getBCs(0,1)<< "; ymax " << myNetwork.getBCs(1,1) << endl;
        cout << "fiber pixel: zmin " << myNetwork.getBCs(0,2)<< "; zmax " << myNetwork.getBCs(1,2) << endl;
     
        const double tzs =  myNetwork.getBCs(0,2); //min z
        const double tze = myNetwork.getBCs(1,2);// max z
        const double dz =tze - tzs;//thickness of network in z
        const double Stzs = tzs + 0.1*dz;
        const double Stze = tze - 0.1*dz;
     
        const double txs = myNetwork.getBCs(0,0); //min x
        const double txe = myNetwork.getBCs(1,0); //max x
        const double dx = txe - txs;
        const double Stx = txe - 0.1*dx;
            
        const double tys = myNetwork.getBCs(0,1); //min y
        const double tye = myNetwork.getBCs(1,1); //max y
        const double dy = tye - tys;
            
        const double Stxs = txs + 0.1*dx; // added by Oleg,9.17.2010, for streching in x-direction
        const double Stxe = txe - 0.1*dx;
        const double Stys = tys + 0.1*dy; // added by Oleg,9.17.2010, for streching in y-direction
        const double Stye = tye - 0.1*dy;

        // viscosity of blood 0.04 poise = 0.004 N second/meter squared
        // viscosity of plasma 0.015 poise = 0.0015 N second/meter squared
        double myGamma = 4*1e3; //nano N micro S/micro sq

        //const double radius = myNetwork.fibersize[0][1]/2.0;
        const double radius = 23*1e-3;

        const double mass = 3.14*(radius*radius)*10; // pico gram, dens = 1000 kg /m^3
        //Stokes Equation
        const double gamma = 6*3.14*myGamma*radius/mass;

        // 1 pixel = 0.14 micrometer,
        //const double radius = thickness*0.14/1e6/2;
        // const double radius = thickness/2;

        // loop through links, fiber diameter
        // reference: network::getSpringForce();
        for(i = 0; i < linksz; i++)
        {
            int oindex = myNetwork.linkr[i][0]; 
            n_link_sz = myNetwork.linkr[i].size(); // = actual # of fibers + 1

            // cout<<"Number of links = "<< n_link_sz <<"; oindx = " << oindex << endl;
            // cout<< "onode position =[" << myNetwork.node[3*oindex] << "," << 
            //      myNetwork.node[3*oindex+1] << "," << myNetwork.node[3*oindex+2] <<"]"<< endl; 

            for(int j = 1; j  < n_link_sz; j++) 
            {
                thickness = myNetwork.fibersize[i][j-1];
                k = myNetwork.linkr[i][j];
                // cout <<"connecting to node " << k << endl;
                  
                if(k > oindex)
                {
                    // double rlx = (myNetwork.node[3*k] - myNetwork.node[3*oindex]);
                    // double rly = (myNetwork.node[3*k+1] - myNetwork.node[3*oindex+1]);
                    // double rlz = (myNetwork.node[3*k+2] - myNetwork.node[3*oindex+2]);
                    // f_len = sqrt(rlx*rlx + rly*rly + rlz*rlz);

                    // printf("fiber[%d], len = %f, thinkness = %f\n", total_N_fiber, f_len, thickness);
                    total_N_fiber++;
                }
            }
            // exit(0);
        }
        // printf("last fiber[%d], len = %f, thinkness = %f\n", total_N_fiber, f_len, thickness);

        cpu_fibers = (tmp_fiber_GPUgrids *)malloc(sizeof(tmp_fiber_GPUgrids));
        cpu_fibers->dt = dt;
        cpu_fibers->maxNodes = branch_sz;
        cpu_fibers->max_N_conn_at_Node = max_Nnode_connect; 

        cpu_fibers->NodeType = (int*)malloc(sizeof(int)*branch_sz);
        cpu_fibers->N_Conn_at_Node = (int*)malloc(sizeof(int)*branch_sz);
        tmp_N_Conn_at_Node = (int*)malloc(sizeof(int)*branch_sz);
        cpu_fibers->Link_at_Node = (int*)malloc(sizeof(int)*branch_sz*max_Nnode_connect); 
        for(i = 0; i < branch_sz*max_Nnode_connect; i++)
            cpu_fibers->Link_at_Node[i] = -100;


        cpu_fibers->X = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->Y = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->Z = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->V_X = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->V_Y = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->V_Z = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->F_X = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->F_Y = (double*)malloc(sizeof(double)*branch_sz);
        cpu_fibers->F_Z = (double*)malloc(sizeof(double)*branch_sz);
        /// NEED to switch direction of the fiber network.
        /// The flow domain is: x-width, y-height, z-length(depth)
        /// fiber network switch y and z coordinates since it is thin in original z direction.
        /// This is subject to change, depending on the fiber data.
        for(i = 0; i < branch_sz; i++)
        {
            cpu_fibers->NodeType[i] = 0;
            cpu_fibers->N_Conn_at_Node[i] = myNetwork.node_N_conn[i];

            cpu_fibers->X[i] = (myNetwork.branchp[i].x);
            cpu_fibers->X[i] *= LEN_UNIT;
            cpu_fibers->X[i] += F_X_SHIFT;
            cpu_fibers->Y[i] = (myNetwork.branchp[i].z);
            cpu_fibers->Y[i] *= LEN_UNIT;
            cpu_fibers->Y[i] += F_Y_SHIFT;
            cpu_fibers->Z[i] = (myNetwork.branchp[i].y);
            cpu_fibers->Z[i] *= LEN_UNIT;
            cpu_fibers->Z[i] += F_Z_SHIFT;

            if(cpu_fibers->X[i] < lside[0])
                lside[0] = cpu_fibers->X[i];
            if(cpu_fibers->Y[i] < lside[1])
                lside[1] = cpu_fibers->Y[i];
            if(cpu_fibers->Z[i] < lside[2])
                lside[2] = cpu_fibers->Z[i];

            if(cpu_fibers->X[i] > uside[0])
                uside[0] = cpu_fibers->X[i];
            if(cpu_fibers->Y[i] > uside[1])
                uside[1] = cpu_fibers->Y[i];
            if(cpu_fibers->Z[i] > uside[2])
                uside[2] = cpu_fibers->Z[i];
            // cout<<"node["<<i<<"], x-coord = " << cpu_fibers->X[i] << endl;
            /// fix node at bottom.
            if(cpu_fibers->Y[i] < 1.0e-6)
            { 
                cpu_fibers->NodeType[i] = 1;
                // printf("Node[%d] is fixed\n", i);
            }
           
            /// init number of links at a node
            tmp_N_Conn_at_Node[i] = 0;
        }
     
        cout <<"fiber domain X[" << lside[0] <<","<<uside[0] <<"]; Y[" << lside[1] <<","<<uside[1] << "]; Z["
                  << lside[2] << ","<<uside[2] <<"]"<< endl;

        cpu_fibers->maxLinks = total_N_fiber;
        cpu_fibers->lAdjVer = (int*)malloc(sizeof(int)*total_N_fiber*2);
        cpu_fibers->linkLengths = (double*)malloc(sizeof(double)*total_N_fiber);
        cpu_fibers->linkLengths0 = (double*)malloc(sizeof(double)*total_N_fiber);
        cpu_fibers->linkThick = (double*)malloc(sizeof(double)*total_N_fiber);
        total_N_fiber = 0;
        for(i = 0; i < linksz; i++)
        {
            int oindex = myNetwork.linkr[i][0]; 
            n_link_sz = myNetwork.linkr[i].size(); // = actual # of fibers + 1
            for(int j = 1; j  < n_link_sz; j++) 
            {
                thickness = myNetwork.fibersize[i][j-1];
                k = myNetwork.linkr[i][j];

                /// set links connected to nodes
                if(k > oindex)
                {
                    if(-100 == cpu_fibers->Link_at_Node[k*max_Nnode_connect + tmp_N_Conn_at_Node[k]])
                    {
                        cpu_fibers->Link_at_Node[k*max_Nnode_connect + tmp_N_Conn_at_Node[k]] = total_N_fiber;
                        tmp_N_Conn_at_Node[k]++;    
                    }
                    if(-100 == cpu_fibers->Link_at_Node[oindex*max_Nnode_connect + tmp_N_Conn_at_Node[oindex]])
                    {
                        cpu_fibers->Link_at_Node[oindex*max_Nnode_connect + tmp_N_Conn_at_Node[oindex]] = total_N_fiber;
                        tmp_N_Conn_at_Node[oindex]++;    
                    }
                }

                if(k > oindex)
                {
                    cpu_fibers->lAdjVer[2*total_N_fiber] = oindex;
                    cpu_fibers->lAdjVer[2*total_N_fiber+1] = k;
                    double rlx = (cpu_fibers->X[k] - cpu_fibers->X[oindex]);
                    double rly = (cpu_fibers->Y[k] - cpu_fibers->Y[oindex]);
                    double rlz = (cpu_fibers->Z[k] - cpu_fibers->Z[oindex]);
                    cpu_fibers->linkLengths0[total_N_fiber] 
                                         = cpu_fibers->linkLengths[total_N_fiber] 
                                         = sqrt(rlx*rlx + rly*rly + rlz*rlz);
                    cpu_fibers->linkThick[total_N_fiber] = thickness*LEN_UNIT;
                    total_N_fiber++;
                }
            }
        }

        /// TMP, check fibers with endpoint being a node
        /****
        for(i = 0; i < branch_sz; i++)
        {
            for(j = 0; j < cpu_fibers->N_Conn_at_Node[i]; j++)
            {
                printf("Node[%d] is the end of fiber[%d]; ", i, 
                    cpu_fibers->Link_at_Node[i*max_Nnode_connect + j]);
                k = cpu_fibers->Link_at_Node[i*max_Nnode_connect + j]; 
                printf("fiber[%d] has nodes[%d, %d]\n", k, 
                     cpu_fibers->lAdjVer[2*k], cpu_fibers->lAdjVer[2*k+1]);
            }   
            if(i == 9)
                break;
        }
        ****/
 
        /// Free myNetwork  and other temporary storage.
        NConnect.erase(NConnect.begin(),NConnect.end());
        nodeNumber.erase(nodeNumber.begin(),nodeNumber.end());
        myNetwork.branchp.erase(myNetwork.branchp.begin(),myNetwork.branchp.end());
        myNetwork.node.erase(myNetwork.node.begin(),myNetwork.node.end());
        for(i = 0; i < linksz; i++)
        {
            myNetwork.linkr[i].erase(myNetwork.linkr[i].begin(), myNetwork.linkr[i].end());
            myNetwork.fibersize[i].erase(myNetwork.fibersize[i].begin(), myNetwork.fibersize[i].end());
        }
        myNetwork.linkr.erase(myNetwork.linkr.begin(), myNetwork.linkr.end()); 
        myNetwork.fibersize.erase(myNetwork.fibersize.begin(), myNetwork.fibersize.end()); 
        myNetwork.newvel.erase(myNetwork.newvel.begin(), myNetwork.newvel.end()); 
        myNetwork.node_N_conn.erase(myNetwork.node_N_conn.begin(), myNetwork.node_N_conn.end());
        ///END:::: Free myNetwork  and other temporary storage.

        /// Verify links at node
        for(i = 0; i < branch_sz; i++)
        {
            if(tmp_N_Conn_at_Node[i] != cpu_fibers->N_Conn_at_Node[i])
            {
                printf("ERROR: init_fibrin_network(), node[%d]\n", i);
                printf("number of links at node recovered from connectivity does not"
                       " match node link number\n");
                exit(-1);
            }
        }

        free(tmp_N_Conn_at_Node);
       
        // for(i = 0; i < cpu_fibers->maxLinks; i++)
        // {
        //     printf("fiber[%d], node[%d, %d]\n", i, cpu_fibers->lAdjVer[2*i], cpu_fibers->lAdjVer[2*i+1]);
        // }

        // void *Fibergrids = fiber_allocGPUKernel(NULL,  branch_sz, total_N_fiber, dt);

        // cout <<"Test code, exit in init_fibrin_network()" << endl;
        // exit(0);
        
        // return Fibergrids;
        return cpu_fibers;
} 


void  fiber_LB_Fsi_force(
	tmp_fiber_GPUgrids *cpu_fibers,
        double	           *hostParameters, 
        void               **fIN, 
        void               **fOUT, 
        void               *ux,
        void               *uy, 
        void               *uz, 
        void               *rho, 
        void               *obstacle,
        RECT_GRID          *gr,
        double              dt)
{
        int        i, j, k, ni, ix, iy, iz, l_ic[3], d, li, nindx0, nindx1;
        int        lx = gr->gmax[0];
        int        ly = gr->gmax[1];
        int        lz = gr->gmax[2];
        double      node_crds[3], node_crds1[3], l_lat_crds[3], d_dist[3], nodev[3], flowv[3], lat_rho;
        double      ffsi[3], tmp_val, len, eps = 1.0e-13, fwlc[3];
        int        status, icrds[3];
        double      (*gridO)[lz][ly][lx] = (double (*)[lz][ly][lx])obstacle;
        double      (*gridIN)[lz][ly][lx];
        double      dx, cur_len, ini_len, wf, add_f;
        static double EE = 1e3; // micro N/ micron sq
        double      radius = 23.0*1e-3;
        double      mass = 3.14*(radius*radius)*10.0; // pico gram, dens = 1000 kg /m^3 
        int        max_N_conn_at_Node = cpu_fibers->max_N_conn_at_Node;
        static int first = YES;
        
        if(YES == first)
        {
            first = NO;
            dx = gr->h[0];
            for (d = 0; d < 19; ++d) {
                cx[d] *= dx/dt;
                cy[d] *= dx/dt;
                cz[d] *= dx/dt;
            }
            Cs=dx/dt;
            Cs2=Cs*Cs;
        }

        for(ni = 0; ni < cpu_fibers->maxNodes; ni ++)
        {
            node_crds[0] = cpu_fibers->X[ni];
            node_crds[1] = cpu_fibers->Y[ni];
            node_crds[2] = cpu_fibers->Z[ni];

            for(i = 0; i < 3; i++)
                nodev[i] = cpu_fibers->V_X[i];
            status = Flow_vel_at_position(node_crds,gr,ux,uy,uz,rho,obstacle,flowv,&lat_rho);
            if(status == NO)
            {
                printf("ERROR: fiber_LB_Fsi_force(), node[%d] with crds[%f, %f, %f] outside domain\n",
                      ni, node_crds[0], node_crds[1], node_crds[2]);
                exit(0);
            }
            ffsi[0] = cpu_fibers->F_X[ni] = lat_rho*(flowv[0] - nodev[0])/dt;  
            ffsi[1] = cpu_fibers->F_Y[ni] = lat_rho*(flowv[1] - nodev[1])/dt;           
            ffsi[2] = cpu_fibers->F_Z[ni] = lat_rho*(flowv[2] - nodev[2])/dt;           
            
            rect_in_which(node_crds, icrds, gr, 3);

            for(ix = icrds[0]-3; ix <= icrds[0]+3; ix++)
            {
                l_ic[0] = ix;
                for(iy = icrds[1]-3; iy <= icrds[1]+3; iy++)
                {
                    l_ic[1] = iy;
                    for(iz = icrds[2]-3; iz <= icrds[2]+3; iz++)
                    {
                        l_ic[2] = iz;

                        if(l_ic[0] < -gr->lbuf[0] || l_ic[0] >= gr->ubuf[0] + gr->gmax[0] ||
                           l_ic[1] < -gr->lbuf[1] || l_ic[1] >= gr->ubuf[1] + gr->gmax[1] ||
                           l_ic[2] < -gr->lbuf[2] || l_ic[2] >= gr->ubuf[2] + gr->gmax[2]
                          )
                            continue;

                        // if((*gridO)[l_icrds[2]][l_icrds[1]][l_icrds[0]] > 0.5)
                        //     continue;
                        for(j = 0; j < 3; j++)
                        {
                            l_lat_crds[j] = lattice_crds(l_ic[j],j,gr);
                            d_dist[j] = (node_crds[j] - l_lat_crds[j])/gr->h[j];
                        }
                        tmp_val = DH1D(d_dist[0], gr->h[0])*DH1D(d_dist[1], gr->h[1])*DH1D(d_dist[2], gr->h[2]);
                        if(tmp_val > 0.0)
                        {
                            for(d = 0; d < 19; d++)
                            {
                                gridIN = (double (*)[lz][ly][lx])(fIN[d]);    
                                add_f = tmp_val*(ffsi[0]*cx[d] + ffsi[1]*cy[d] + ffsi[2]*cz[d])*f0[d]*dt*lat_rho/Cs2;
                                (*gridIN)[lz][ly][lx] += add_f;
                            }
                        }
                    }
                }
            }
        } /// END::: for(ni = 0; ni < cpu_fibers->maxNodes; ni ++)

        /// The following algorithm can not be used on GPU.
        /****
        for(li = 0; li < cpu_fibers->maxLinks; li++)
        { 
            nindx0 = cpu_fibers->lAdjVer[2*li];
            nindx1 = cpu_fibers->lAdjVer[2*li + 1];

            node_crds[0] = cpu_fibers->X[nindx0]; //
            node_crds[1] = cpu_fibers->Y[nindx0];
            node_crds[2] = cpu_fibers->Z[nindx0];

            node_crds1[0] = cpu_fibers->X[nindx1];
            node_crds1[1] = cpu_fibers->Y[nindx1];
            node_crds1[2] = cpu_fibers->Z[nindx1];
            ini_len = cpu_fibers->linkLengths0[li];

            cur_len = 0.0;
            for(j = 0; j < 3; j++)
            {
                d_dist[j] = (node_crds[j] - node_crds1[j]);
                cur_len += sqr(d_dist[j]);
            }
            cur_len = sqrt(cur_len);
            if(cur_len < eps)
            {
                for(j = 0; j < 3; j++)
                    fwlc[j] = 0.0;
            }
            else
            {
                wf = wormlike_chain_force(ini_len, cur_len, EE, mass);
                for(j = 0; j < 3; j++)
                    fwlc[j] = wf*d_dist[j]/cur_len;
                cpu_fibers->F_X[nindx0] -= fwlc[0];
                cpu_fibers->F_Y[nindx0] -= fwlc[1];
                cpu_fibers->F_Z[nindx0] -= fwlc[2];

                cpu_fibers->F_X[nindx1] += fwlc[0];
                cpu_fibers->F_Y[nindx1] += fwlc[1];
                cpu_fibers->F_Z[nindx1] += fwlc[2];
            }
            
        }// for(li = 0; li < cpu_fibers->maxLinks; li++)
        ****/
        
        for(ni = 0; ni < cpu_fibers->maxNodes; ni ++)
        {
            for(j = 0; j < cpu_fibers->N_Conn_at_Node[ni]; j++)
            {
                li = cpu_fibers->Link_at_Node[ni*max_N_conn_at_Node + j];

                if(ni == cpu_fibers->lAdjVer[2*li])
                {
                    nindx0 = cpu_fibers->lAdjVer[2*li];
                    nindx1 = cpu_fibers->lAdjVer[2*li + 1];
                }
                else if(ni == cpu_fibers->lAdjVer[2*li + 1])
                {
                    nindx0 = cpu_fibers->lAdjVer[2*li + 1];
                    nindx1 = cpu_fibers->lAdjVer[2*li];
                }
                else
                {
                    printf("ERROR: fiber_LB_Fsi_force(),"
                      " link end points and nodes are not consistent\n");
                    printf("ni = %d, nindx0 = %d, nindx1 = %d\n", ni, nindx0, nindx1);
                    exit(-1);
                }

                node_crds[0] = cpu_fibers->X[nindx0]; //
                node_crds[1] = cpu_fibers->Y[nindx0];
                node_crds[2] = cpu_fibers->Z[nindx0];

                node_crds1[0] = cpu_fibers->X[nindx1];
                node_crds1[1] = cpu_fibers->Y[nindx1];
                node_crds1[2] = cpu_fibers->Z[nindx1];
                ini_len = cpu_fibers->linkLengths0[li];

                cur_len = 0.0;
                for(k = 0; k < 3; k++)
                {
                    d_dist[k] = (node_crds[k] - node_crds1[k]);
                    cur_len += sqr(d_dist[k]);
                }
                cur_len = sqrt(cur_len);
                if(cur_len < eps)
                {
                    for(k = 0; k < 3; k++)
                        fwlc[k] = 0.0;
                }
                else
                {
                    wf = wormlike_chain_force(ini_len, cur_len, EE, mass);
                    for(k = 0; k < 3; k++)
                        fwlc[k] = wf*d_dist[k]/cur_len;
                    cpu_fibers->F_X[nindx0] -= fwlc[0];
                    cpu_fibers->F_Y[nindx0] -= fwlc[1];
                    cpu_fibers->F_Z[nindx0] -= fwlc[2];

                    // cpu_fibers->F_X[nindx1] += fwlc[0];
                    // cpu_fibers->F_Y[nindx1] += fwlc[1];
                    // cpu_fibers->F_Z[nindx1] += fwlc[2];
                }
            }
        } /// END: for(ni = 0; ni < cpu_fibers->maxNodes; ni ++)
 
}

double wormlike_chain_force(
	double  ini_l,
        double  cur_l,
        double  Modulus,
        double  mass)
{
        double               IP;
        static double        r = 23*1e-3;
        static unsigned int Nmol = 1200;
        static double        kB=1.3806503*1e-8; // micro m2 pg micors-2 K-1
        static double        mTemp = 10;
        static double        En = kB*mTemp;
        double               Y = Modulus;
        double               PerL;
        PerL = ini_l;
        double               CC = En/PerL;  // CC = kB*mTemp*Nmol/PerL;
        double               tl, IP1, xi;
       
        tl = 1.0 - (cur_l - ini_l)/ini_l;
        IP1 = 0.25*1.0/(tl*tl) - 0.25 + (cur_l - ini_l)/ini_l;
        IP = -CC*IP1/mass;
        // xi = lz/ol;
        return IP;
}

int   Flow_vel_at_position(
	double              *crds,
        RECT_GRID          *gr,
        void               *ux,
        void               *uy,
        void               *uz,
        void               *rho,
        void               *obstacle,
        double              *vel,
        double              *lat_rho)
{
        int        i, j, k, ix, iy, iz;
        int        lx = gr->gmax[0];
        int        ly = gr->gmax[1];
        int        lz = gr->gmax[2];
        double     *h = gr->h, tmp_val, loc_dens = 0.0;
        int        icrds[3], l_icrds[3];

        double (*gridX)[lz][ly][lx] = (double (*)[lz][ly][lx])ux;
        double (*gridY)[lz][ly][lx] = (double (*)[lz][ly][lx])uy;
        double (*gridZ)[lz][ly][lx] = (double (*)[lz][ly][lx])uz;
        double (*gridR)[lz][ly][lx] = (double (*)[lz][ly][lx])rho;
        double (*gridO)[lz][ly][lx] = (double (*)[lz][ly][lx])obstacle;
        double      l_lat_crds[3], d_dist[3], lat_v[3];

        for(i = 0; i < 3; i++)
        {
            if(crds[i] < gr->lL[i] || crds[i] > gr->lU[i])
                return NO;
        }
 
        memset(vel, 0, sizeof(double)*3); 

        rect_in_which(crds, icrds, gr, 3);

        for(ix = icrds[0]-3; ix <= icrds[0]+3; ix++)
        {
            l_icrds[0] = ix;
            for(iy = icrds[1]-3; iy <= icrds[1]+3; iy++)
            {
                l_icrds[1] = iy;
                for(iz = icrds[2]-3; iz <= icrds[2]+3; iz++)
                {
                    l_icrds[2] = iz;
                    
                    if(l_icrds[0] < -gr->lbuf[0] || l_icrds[0] >= gr->ubuf[0] + gr->gmax[0] ||
                       l_icrds[1] < -gr->lbuf[1] || l_icrds[1] >= gr->ubuf[1] + gr->gmax[1] ||
                       l_icrds[2] < -gr->lbuf[2] || l_icrds[2] >= gr->ubuf[2] + gr->gmax[2]
                      )
                    {
                        continue;
                    }
                    
                    for(j = 0; j < 3; j++)
                    {
                        l_lat_crds[j] = lattice_crds(l_icrds[j],j,gr);
                        d_dist[j] = (crds[j] - l_lat_crds[j])/gr->h[j];
                    }   

                    lat_v[0] = (*gridX)[l_icrds[2]][l_icrds[1]][l_icrds[0]];
                    lat_v[1] = (*gridY)[l_icrds[2]][l_icrds[1]][l_icrds[0]];
                    lat_v[2] = (*gridZ)[l_icrds[2]][l_icrds[1]][l_icrds[0]];

                    if((*gridO)[l_icrds[2]][l_icrds[1]][l_icrds[0]] > 0.5)
                        continue;

                    tmp_val = h[0]*h[1]*h[2]*DH1D(d_dist[0], h[0])*DH1D(d_dist[1], h[1])*DH1D(d_dist[2], h[2]);
                    loc_dens += tmp_val*(*gridR)[l_icrds[2]][l_icrds[1]][l_icrds[0]]; 

                    for(j = 0; j < 3; j++)
                    {
                        vel[j] += lat_v[j]*tmp_val;
                    }
                }
            }
        }

        *lat_rho = loc_dens;
        return YES;
}


/// init. node velocity = fluid velocity
void  cpu_init_fiber_LB(
	tmp_fiber_GPUgrids *cpu_fibers,
        double	           *hostParameters, 
        void               **fIN, 
        void               **fOUT, 
        void               *ux,
        void               *uy, 
        void               *uz, 
        void               *rho, 
        void               *obstacle,
        RECT_GRID          *gr,
        double              dt)
{
        int        i, j, k, ni;
        int        lx = gr->gmax[0]; 
        int        ly = gr->gmax[1]; 
        int        lz = gr->gmax[2]; 
        double      node_crds[3], l_lat_crds[3], d_dist[3];
        int        status, icrds[3];
        double      Lat_Vx[8], Lat_Vy[8], Lat_Vz[8], nodev[3];

        double (*gridX)[lz][ly][lx] = (double (*)[lz][ly][lx])ux;
        double (*gridY)[lz][ly][lx] = (double (*)[lz][ly][lx])uy;
        double (*gridZ)[lz][ly][lx] = (double (*)[lz][ly][lx])uz;
        double (*gridR)[lz][ly][lx] = (double (*)[lz][ly][lx])rho;
        double (*gridO)[lz][ly][lx] = (double (*)[lz][ly][lx])obstacle;
 
        double rho_l = hostParameters[6];
 
        for(ni = 0; ni < cpu_fibers->maxNodes; ni ++)
        {
            node_crds[0] = cpu_fibers->X[ni];
            node_crds[1] = cpu_fibers->Y[ni];
            node_crds[2] = cpu_fibers->Z[ni];

            status = rect_in_which(node_crds, icrds, gr, 3);
            if(status == NO)
            {
                cout<<"WARNING: cpu_init_fiber_LB_coupling() fiber node["<< 
                   node_crds[0] <<","<< node_crds[1] <<","<<node_crds[2]<<"] outside lattice domain"<<endl;  
            }
            else
            {
                for(j = 0; j < 3; j++)
                {
                    l_lat_crds[j] = lattice_crds(icrds[j],j,gr);
                    d_dist[j] = (node_crds[j] - l_lat_crds[j])/gr->h[j];
                }

                Lat_Vx[0] = (*gridX)[icrds[2]][icrds[1]][icrds[0]];
                Lat_Vx[1] = (*gridX)[icrds[2]][icrds[1]+1][icrds[0]];
                Lat_Vx[2] = (*gridX)[icrds[2]][icrds[1]+1][icrds[0]+1];
                Lat_Vx[3] = (*gridX)[icrds[2]][icrds[1]][icrds[0]+1];

                Lat_Vx[4] = (*gridX)[icrds[2]+1][icrds[1]][icrds[0]];
                Lat_Vx[5] = (*gridX)[icrds[2]+1][icrds[1]+1][icrds[0]];
                Lat_Vx[6] = (*gridX)[icrds[2]+1][icrds[1]+1][icrds[0]+1];
                Lat_Vx[7] = (*gridX)[icrds[2]+1][icrds[1]][icrds[0]+1];
                nodev[0] = Trilinear_interp(d_dist, Lat_Vx);

                Lat_Vy[0] = (*gridY)[icrds[2]][icrds[1]][icrds[0]];
                Lat_Vy[1] = (*gridY)[icrds[2]][icrds[1]+1][icrds[0]];
                Lat_Vy[2] = (*gridY)[icrds[2]][icrds[1]+1][icrds[0]+1];
                Lat_Vy[3] = (*gridY)[icrds[2]][icrds[1]][icrds[0]+1];

                Lat_Vy[4] = (*gridY)[icrds[2]+1][icrds[1]][icrds[0]];
                Lat_Vy[5] = (*gridY)[icrds[2]+1][icrds[1]+1][icrds[0]];
                Lat_Vy[6] = (*gridY)[icrds[2]+1][icrds[1]+1][icrds[0]+1];
                Lat_Vy[7] = (*gridY)[icrds[2]+1][icrds[1]][icrds[0]+1];
                nodev[1] = Trilinear_interp(d_dist, Lat_Vy);

                Lat_Vz[0] = (*gridZ)[icrds[2]][icrds[1]][icrds[0]];
                Lat_Vz[1] = (*gridZ)[icrds[2]][icrds[1]+1][icrds[0]];
                Lat_Vz[2] = (*gridZ)[icrds[2]][icrds[1]+1][icrds[0]+1];
                Lat_Vz[3] = (*gridZ)[icrds[2]][icrds[1]][icrds[0]+1];

                Lat_Vz[4] = (*gridZ)[icrds[2]+1][icrds[1]][icrds[0]];
                Lat_Vz[5] = (*gridZ)[icrds[2]+1][icrds[1]+1][icrds[0]];
                Lat_Vz[6] = (*gridZ)[icrds[2]+1][icrds[1]+1][icrds[0]+1];
                Lat_Vz[7] = (*gridZ)[icrds[2]+1][icrds[1]][icrds[0]+1];
                nodev[2] = Trilinear_interp(d_dist, Lat_Vz);

                cpu_fibers->V_X[ni] = nodev[0];
                cpu_fibers->V_Y[ni] = nodev[1];
                cpu_fibers->V_Z[ni] = nodev[2];

                // printf("node crds[%g, %g, %g], low_lattice_crds[%g, %g, %g], lattic_indx[%d, %d, %d]\n",
                //      node_crds[0], node_crds[1], node_crds[2], l_lat_crds[0], l_lat_crds[1], l_lat_crds[2],
                //         icrds[0], icrds[1], icrds[2]);

                // exit(0);
            }
        }
}

int rect_in_which(
        double      *crds,
        int        *icrds,
        RECT_GRID  *grid,
        int        dim)
{
        int      status = YES;
        double   *h = grid->h;
        int      *gmax = grid->gmax;
        int      i;

        for(i = 0; i < dim; i++)
        {
            icrds[i] = cell_index(crds[i],i,grid);
            if (icrds[i] < 0)
            {
                status = NO;
            }
            if(icrds[i] >= gmax[i])
            {
                status = NO;
            }
        }
        return status;
}

#if !defined(sun) || (defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || defined(_HPUX_SOURCE) || defined(cray) || (defined(__GNUC__) && !defined(linux))
int irint(double x)
{
        return (int) rint(x);
}               /*end irint*/
#endif /* !defined(sun) || (defined(__SUNPRO_C) || defined(__SUNPRO_CC)) || defined(_HPUX_SOURCE)  || defined(cray) || (defined(__GNUC__) && !defined(linux)) */

