/*
 * File:   network.cpp
 * Author: eunjungkim
 *
 */
// #include "StdAfx.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include <iterator>
#include <algorithm>
#include <numeric>
#include "network.h"


using namespace std;

network::network() {
}



network::~network() {
}


void network::getstat(){
// Get average thickness, number of branching points, fiber density, etc


    // number of branching points
    unsigned int sz = linkr.size();

	// count number of node whose degree is degree 3 or 4
	unsigned int deg1 = 0;

	// count number of nodes whose degree is less than 2
    unsigned int deg0 = 0;

	// count number of nodes degree greater than 5
    unsigned int deg2 = 0;

    for(int i = 0; i < sz; i++){

        //cout << linkr[i].size() << endl;

        if(linkr[i].size() < 4) deg0++;

        else if(linkr[i].size() >= 4 && linkr[i].size() <=5 ) deg1++;

        else deg2++;
    }

	const double volume = 150*150*10;
	cout << "number of branch point: " <<  deg0 << " " << deg1 << " " << deg2 <<
            " " << deg1 + deg2<< " branch pt density " << ((double)deg1+(double)deg2)/volume << " "
			<< ((double)deg0+(double)deg1+(double)deg2)/volume <<  endl;

    //averaging thickness
    const unsigned int fsz = fibersize.size();

	// average thickness
	double favethick;

	// adding all thickness
    double fallthick;

    unsigned int Count =0;
    fallthick = 0.0;

	for(int k = 0; k < fsz; k++){

		const unsigned int ffsz = fibersize[k].size();

		for(int j = 0; j < ffsz; j++){

			fallthick += fibersize[k][j];

			Count++;

		}

	}

    cout <<  " thickness: " << fallthick << " count: " << Count << endl;


	favethick = fallthick/Count;

	cout << "average thickness: " << favethick << endl;


	// all fiber length

    double rl;
    double flength = 0.0;
    unsigned int CTf = 0;

	for(int k2 = 0; k2 < sz; k2++ ){

        const double linksz = linkr[k2].size();

        for(int j = 1; j  < linksz; j++){

			int k = linkr[k2][j];
            int oindex = linkr[k2][0];

            if(k > oindex){

				double rlx = (node[3*k]-node[3*oindex]);
				double rly = (node[3*k+1]-node[3*oindex+1]);
				double rlz = (node[3*k+2]-node[3*oindex+2]);

				rl = sqrt(rlx*rlx + rly*rly + rlz*rlz);
				//cout << rl*1e6 << endl;

            }

				flength += rl;
			CTf++;


        }
    }

    cout << "fiber length: " << flength << " " << CTf << " average f length " << flength/(float)CTf << " density " << flength/volume<< endl;


    // density (length/volume)
}

void network::getBNodes(){

	const unsigned int sz = size();
	Bnodesx.clear();
	Bnodesz.clear();

	const double tx = getBCs(0,0);
	const double tz = getBCs(1,2);

	//cout << tx << " " << tz << endl;
	for(int i = 0; i < sz; i++){

		if(node[3*i] == tx ) Bnodesx.push_back(i);
	        else Bnodesx.push_back(0);
	}
	for(int k3 = 0; k3 < sz; k3++){

		if(node[3*k3+2] == tz ) Bnodesz.push_back(k3);
	        else Bnodesz.push_back(0);
	}

        const unsigned int szx = Bnodesx.size();
	const unsigned int szz = Bnodesz.size();

	cout << szx << " bnode " << szz << endl;
	for(int k4 = 0; k4 < szx; k4++){
	  	if(node[3*k4] == tx) cout << Bnodesx[k4] << " x  " << node[3*k4] << endl;
	}

	for(int k5 = 0; k5 < szz; k5++){
		if(node[3*k5+2] == tz) cout << Bnodesz[k5] << " z " << node[3*k5+2] << endl;
	}
}



void network::setNodes(const int number, const double xsize,
                    const double ysize, const double zsize){

// Set "number" of random nodes in the domain: xsize x ysize x zsize

    branchp.clear();
    //loop over number of nodes
    for(int i=0; i<number; i++){

        //set coordinate
        coord myCoords;
        myCoords.x = (double)rand() / (double)RAND_MAX * xsize;
        myCoords.y = (double)rand() / (double)RAND_MAX * ysize;
        myCoords.z = (double)rand() / (double)RAND_MAX * zsize;

        branchp.push_back(myCoords);

    }

    //clear links
    linkr.clear();
    linkr.resize(number);

}

void network::definelinks(const double distSq) {
// If distance of two randome nodes is less than distSq, we will define a link between the nodes.

    const int bsize = branchp.size();

    linkr.resize(bsize);

    for (int i=0;i<bsize;i++){
        for(int j=0;j<bsize;j++){

            //branchp.push_back(); //() put input file
            double dist = pow((branchp[i].x - branchp[j].x),2)+
            pow((branchp[i].y - branchp[j].y),2)+
            pow((branchp[i].z - branchp[j].z),2);

            if(dist <= distSq ){
                linkr[i].push_back(j);
            }
        }

    }
}

unsigned int network::size(){
    return branchp.size();
}

double network::getCoords(const unsigned int indx, const unsigned int xyz){
// Get coordinates of nodes

    double out = 0.0;

    if(indx < branchp.size() && xyz < 3){

        switch(xyz){
            case 0: out = branchp[indx].x;
            break;
            case 1: out = branchp[indx].y;
            break;
            case 2: out = branchp[indx].z;
            break;
        }
    }

    return out;
}


void network::putIntoVec(double xc,double yc,double zc){
//Stores coordinates of nodes
// The points in image file is not is micrometer unit.
// To change it, we need to use actual size and number of slides.
// The network size = 146 x 146 x 10 micron cubed

    coord MyC;

	//Store the coordinate in meter scale
    MyC.x = xc;
    MyC.y = yc;
    MyC.z = zc; // 30 slides -- > 10 micrometer

    branchp.push_back(MyC);

}

void network::storeCoords(){
// Store coordinates into a vector node
// node[3*i] = coordinate of (branchp[i].x), etc


    node.clear();

	const int bsize = branchp.size();
    for (int i =0;i<bsize;i++){

        const double xc = getCoords(i,0);
        const double yc = getCoords(i,1);
        const double zc = getCoords(i,2);

	node.push_back(xc);
        node.push_back(yc);
        node.push_back(zc);

    }

}

 void network::getinialPafterF(double strain){
// To prescribe boundary conditions
// Fix nodes on substrate
// Apply force initially in some direction.

	 const int bsize = branchp.size();

      newp.resize(3*bsize);

      for(int j =0; j < bsize;j++){

		  newp[3*j] = node[3*j];
		  newp[3*j+1] = strain*node[3*j+1];
		  newp[3*j+2] = node[3*j+2];
      }

 }


void network::getSpringForce(vector<double>& nodein,int s,const double Modulus,const double mass){


    double fvecx,fvecy,fvecz;
    double ol, rl;
    double factor;

    const unsigned int sz = linkr.size();
    const unsigned int sz2 = nodein.size();

   // cout << "Size node " << sz2 << " nodein " << nodein.size() << endl;

    Force.clear();
    Force.resize(sz2,0);


    for(int i = 0; i < sz; i++ ){

        fvecx = 0.0; fvecy = 0.0; fvecz = 0.0;

	// add force over all connecting nodes
        int linksz = linkr[i].size();


        for(int j = 1; j  < linksz; j++){

            const unsigned int oindex = linkr[i][0];
            const unsigned int k = linkr[i][j];
            const double thickness = fibersize[i][j-1];

            
            // count just once
            if(k > oindex){

                //Rest length == intial length
                double rlx = (node[3*k]-node[3*oindex]);
                double rly = (node[3*k+1]-node[3*oindex+1]);
		double rlz = (node[3*k+2]-node[3*oindex+2]);

		rl = sqrt(rlx*rlx + rly*rly + rlz*rlz);

		// new length
		const double lx = (nodein[3*k]-nodein[3*oindex]);
		const double ly = (nodein[3*k+1]-nodein[3*oindex+1]);
		const double lz = (nodein[3*k+2]-nodein[3*oindex+2]);

		ol = sqrt(lx*lx + ly*ly + lz*lz);

                              
              #if 0
		// spring constant

		// 1 pixel = 0.14 micrometer,
		//const double radius = thickness*0.14/1e6/2;
		const double radius = thickness/2;

               // cout << "radius " << radius << endl;
		const double Y = Modulus;
		const double Area = 3.14*radius*radius;


		// Spring constant = Young's modulus x Area
		if((rl*mass) != 0.0 ) {  factor = -1*Y*Area/(rl*mass); }
                
		if(ol != 0.0){

                    fvecx = factor*lx*(ol-rl)/ol;
                    fvecy = factor*ly*(ol-rl)/ol;
                    fvecz = factor*lz*(ol-rl)/ol;

                }
                #endif
             
                

                                // Nonlinear chain model for individual fiber

                                // WLC chain from J Weisel's Science paper

                                double IP;
                                //const double r = thickness/2.0*0.1;
                                const double r = 23*1e-3;
                                const unsigned int Nmol = 1200;

                                //const double kB=1.3806503*1e-23; //m2 kg s-2 K-1
                                const double kB=1.3806503*1e-8; // micro m2 pg micors-2 K-1
                                const double mTemp = 10;
                                const double En = kB*mTemp;
				const double Y = Modulus;
                                //const double PerL = Y*3.14*r*r*r*r/En;
                                const double PerL = rl;
                                //const double PerL = 8e-4;
                                //const double CC = kB*mTemp*Nmol/PerL;
                                const double CC = En/PerL;
                                const double phi = atan2(ly,lx);
                                //const double phi = atan2(ly+rly,lx+rlx);
                                double xi;
				
		//		if(ol > 1e5) cout << " length " << ol << " old le " << rl << endl;
                                if(rl != 0.0){
                                    const double tl = 1 - (ol-rl)/rl;
                                    const double IP1 = .25*1/(tl*tl) - .25 + (ol-rl)/rl;
                                    IP = -CC*IP1/mass;
                                    xi = lz/ol;
				    //const double temp = (lx+rlx)*(lx+rlx) + (ly +rly)*(ly+rly) + (lz+rlz)*(lz*rlz);
                                    //xi = (rlz+lz)/sqrt(temp);
				   // cout << " rlz, lz and temp " << rlz << " " << lz << " " << temp << endl;
                                }
                                if(ol != 0.0){
                                	fvecx = IP*lx/ol;
                                	fvecy = IP*ly/ol;
                                	fvecz = IP*lz/ol;
				}
				//cout << " xi and cos xi " << xi << " " << cos(xi) << endl;
				//cout << " force x " << fvecx << " f y " << fvecy << "  f z " << fvecz << endl;

                                Force[3*oindex] += fvecx;
				Force[3*oindex+1] += fvecy;
				Force[3*oindex+2] += fvecz;

				Force[3*k] -= fvecx;
				Force[3*k+1] -= fvecy;
				Force[3*k+2] -= fvecz;
		
           }
        }
/*---------------------------------Trying some force 7/13/2010
for (int iex=0;iex<linksz;iex++)
{
	Force[3*iex+1]+=1;

}    
*/

}

	

}



void network::Solve(double dt,double strain,
        const double gamma, int s, const double Modulus,
        const double tzs, const double tze, const double tys, 
		const double tye, const double mass){

    // calcuate force
    getSpringForce(newp,s,Modulus,mass);

    
    
    // with friction
    solverkick(dt/2,gamma);

	// leapfrog
//  solverleapfrog(dt/2.0);

    const unsigned int sz = linkr.size();

    //cout << "Got here " << endl;

    //#if 0
 /*   for(int i = 0; i < sz; i++ ){

		if ( node[3*i+2] >= tzs) //& node[3*i+2] <= tze )
		{

			newp[3*i] = newp[3*i] + newvel[3*i]*dt;
			newp[3*i+1] = newp[3*i+1] + newvel[3*i+1]*dt;
			newp[3*i+2] = newp[3*i+2] + newvel[3*i+2]*dt;

		}
		
    }*/

	//added for xbc by Oleg KIm 9/18/2010
	for(int i = 0; i < sz; i++ ){

		if ( node[3*i+2] >= tys /* & node[3*i+2] <= tye */){

			newp[3*i] = newp[3*i] + newvel[3*i]*dt;
			newp[3*i+1] = newp[3*i+1] + newvel[3*i+1]*dt;
			newp[3*i+2] = newp[3*i+2] + newvel[3*i+2]*dt;

		}
		
    }

    //#endif
	//cout << "Newp[509]:"<<newp[509]<<endl; //test
   // calculate force
	getSpringForce(newp,s,Modulus,mass);

	// Adding external Force on the top
	/*
	for(int i = 0; i < sz; i++ ){

		if ( node[3*i+2] >= 0.5*tze)
		{
			Force[3*i+2] = Force[3*i+2] - 10;
			
		}
		
    }
	*/
//	solverleapfrog(dt/2.0);

   solverkick(dt/2,gamma);

}


void network::UpdateLink( vector< double >& NewPos, const double MaxCoeff){
// If a fiber stretched more than 3 times, remove the node

	const unsigned fsz = fibersize.size();
	const unsigned int sz = linkr.size();

        //nlinkr.resize(sz);
        //nfibersize.resize(fsz);
       
       // cout << " fibersize check " <<  fsz << " " << sz << endl;
	
        for(int i = 0; i < sz; i++ ){

            const double linksz = linkr[i].size();
	
            double rl,ol;
          //  nlinkr[i].push_back(linkr[i][0]);
          //  nfibersize[i].push_back(fibersize[i][0]);


            for(int j = 1; j  < linksz; j++){

                int k = linkr[i][j];
                int oindex = linkr[i][0];

                const double fsize = fibersize[i][j];
             
                // cout << "input size " << fsize << endl;
            
                if(k > oindex){

                    double rlx = (node[3*k] - node[3*oindex]);
                    double rly = (node[3*k+1] - node[3*oindex+1]);
                    double rlz = (node[3*k+2] - node[3*oindex+2]);
               
                    rl = sqrt(rlx*rlx + rly*rly + rlz*rlz);
               
                    const double lx = (NewPos[3*oindex]-NewPos[3*k]);
                    const double ly = (NewPos[3*oindex+1]-NewPos[3*k+1]);
                    const double lz = (NewPos[3*oindex+2]-NewPos[3*k+2]);

                    ol = sqrt(lx*lx + ly*ly + lz*lz);

                }

               // nlinkr[i].push_back(k);

		//nfibersize[i].push_back(fsize);
                
               
                const double MAX_ST = 4.0;
                const double MAX_LINK_FORCE = MaxCoeff*MAX_ST;

                //if(rl != 0.0){

                const double LINK_FORCE = MaxCoeff*fabs(ol/rl);
	
			if(ol > 1e15*rl)
			{
                            //const int ii =
                            linkr[i].erase(linkr[i].begin() + j - 1);
                            fibersize[i].erase(fibersize[i].begin() + j - 1 );
                            //nlinkr[i].push_back(k);

				//  nfibersize[i].push_back(fsize);

			}

               
                
		//}


        }



    }



}




void network::Print(vector<double>& Eqpos,int TIME, ofstream &outFileVis){
//Print file for viewers
    const unsigned int bsize = branchp.size();

    outFileVis << "#Header: number of frames, format (1=old 2=with fiber thickness),"
            " default node size, default fiber size, project name" << endl;

    outFileVis << "1	2	0.1	1e-6	KRM" << endl;

	outFileVis << "#number of nodes " << endl;

	outFileVis <<bsize << endl;

	outFileVis << "#node	#x,y,z	#number of connections	#connections	"
            "#Thickness of fiber (not for format 1)" << endl;


	const unsigned int sz2 = linkr.size();

   // cout << sz2 << endl;

    for(unsigned int i = 0; i < sz2; i++){

        const int oindex = linkr[i][0];
        const unsigned int szlink =linkr[i].size();
        const unsigned int fsize = fibersize[i].size();

		//output node and coordinates

		outFileVis << oindex
                    << " " << Eqpos[3*i]
                    << " " << Eqpos[3*i+1]
                    << " " << Eqpos[3*i+2]
                    << " ";


        outFileVis << szlink - 1;

        for(unsigned int j=1; j < szlink;j++){

            const int k = linkr[i][j];

                outFileVis << " " << k ;

        }

        for(unsigned int kk = 0; kk < fsize ; kk++ ){

            outFileVis << " " << fibersize[i][kk];
        }

        //end line
        outFileVis << endl;
    }

}

void network::PrintEqposition(int TIME,int CA){
	//Print file for viewers

	char name[100];
	//output visualization file: deformed state
    ofstream outFileVis;

	sprintf(name,"Deformed%d_%d.txt",TIME,CA);

	outFileVis.open(name);

	const unsigned int bsize = branchp.size();

    outFileVis << "#Header: number of frames, format (1=old 2=with fiber thickness),"
	" default node size, default fiber size, project name" << endl;

    outFileVis << "1	2	3e-5	1e-6	KRM" << endl;

	outFileVis << "#number of nodes " << endl;

	outFileVis <<bsize << endl;

	outFileVis << "#node	#x,y,z	#number of connections	#connections	"
	"#Thickness of fiber (not for format 1)" << endl;


	//const unsigned int sz = nlinkr.size();



    for(unsigned int i = 0; i < bsize; i++){

        const int oindex = linkr[i][0];
        const unsigned int szlink = linkr[i].size();
        const unsigned int fsize = fibersize[i].size();



		//output node and coordinates
        {

			outFileVis << oindex
			<< " " << newp[3*i]
			<< " " << newp[3*i+1]
			<< " " << newp[3*i+2]
			<< " ";

			//cout << " error " << i << endl;

			outFileVis << szlink - 1;


			for(unsigned int j = 1; j < szlink;j++){

				const int k = linkr[i][j];

				outFileVis << " " << k ;

			}

			for(unsigned int kk = 0; kk < fsize ; kk++ ){

				outFileVis << " " << fibersize[i][kk];

			}

        //end line
        outFileVis << endl;
		}
    }

	//close file
	outFileVis.close();


}


double network::getBCs(const unsigned int bci, const unsigned int xyz){
// Get boundary coordinate

     double out = 0.0;
     vector <double> tempNx;
     vector <double> tempNy;
     vector <double> tempNz;

     tempNx.clear();
     tempNy.clear();
     tempNz.clear();
     const int Nsize = branchp.size();

     for (int i=0;i<Nsize;i++){
         tempNx.push_back(node[3*i]);
         tempNy.push_back(node[3*i+1]);
         tempNz.push_back(node[3*i+2]);
     }

     const double minx =  *(std::min_element( tempNx.begin(), tempNx.end() ));
     const double maxx =  *(std::max_element( tempNx.begin(), tempNx.end() ));
     const double miny =  *(std::min_element( tempNy.begin(), tempNy.end() ));
     const double maxy =  *(std::max_element( tempNy.begin(), tempNy.end() ));
     const double minz =  *(std::min_element( tempNz.begin(), tempNz.end() ));
     const double maxz =  *(std::max_element( tempNz.begin(), tempNz.end() ));

      switch(xyz){
            case 0: if( bci == 0) out = minx; else out=maxx;
            break;
            case 1: if( bci == 0) out = miny; else out=maxy;
            break;
            case 2: if( bci == 0) out = minz; else out=maxz;
            break;
      }


    return out;

 }

// find stress Sij = 1/V sum xi Fj

 double network::calculateStresses(vector<double>& Position, vector<double>& Force){

       double Syy;
	const double xl = getBCs(0,0);
	const double xr = getBCs(1,0);
	const double yl = getBCs(0,1);
	const double yr = getBCs(1,1);
	const double zl = getBCs(0,2);
	const double zr = getBCs(1,2);


            Syy = 0.0;

            const unsigned int sz = size();

            for(int i = 0; i < sz; i++){
               //if(node[3*i] == xl || node[3*i] == xr ||
               // node[3*i+1] == yl ||node[3*i+1] == yr||
               // node[3*i+2] == zl || node[3*i+2] == zr)
               {
		Syy += (Position[3*i+1])*Force[3*i+1];
              
                }

            }


            double volume = 150*150*10;

            cout << "Syy Volume Ave " << Syy << endl;
	double out = Syy/volume;
	return out;
    }


 double network::calculateFStresses(vector<double>& Position, vector<double>& Force,const double txe){

        double FSyy;
        const double xr = getBCs(1,0);

        FSyy = 0.0;

            const unsigned int sz = size();
           //if(node[3*i] >= 140 || node[3*i] <= 10 || node[3*i+1] >= 140 || node[3*i+1] <= 10|| node[3*i+2] >= 9 || node[3*i+2] <= 1)
            for(int i = 0; i < sz; i++){
           	//if(node[3*i] == xr)
			{
				 FSyy += Force[3*i+1];
			}
            }


            //double area = 150*10;
            double area = 150*150*10;


	double out = FSyy/area;
	return out;
    }

 void network::solverkick( double dt, double myGammathick ){

        //constants

	 const double fdt = ( 1.0 - exp(-myGammathick * dt ) ) / myGammathick;
	 const double vdt = exp(-myGammathick*dt);
      
	//cout << " damping my gamma " << myGammathick 
//		<< " scale " << vdt 
//		<< " fdt " << fdt << endl;
	 const unsigned int sz = newp.size();

	 //calculate half step in velocities



        for (int i = 0; i < sz ; i++) {

//	   cout << " newvel 1 " << newvel[i] << endl;

            //scale velocity
            newvel[i] = vdt*newvel[i];

//	   cout << " newvel2  " << newvel[i] << endl;
	    //add scaled forces
				
			
            newvel[i]= newvel[i] + (Force[i])*fdt;
/*	
	if (i ==509) {
	newvel[i] = newvel[i] + 1*fdt;
	cout<<"node #1, vel: "<<newvel[i]<<endl;
	cout<<"node #1, force: "<<Force[i]<<endl;
	} */ //to test 1 N in y direction for node #1

//	if (Force[i]!=0) cout<<i<<" node force"<<Force[i]<<endl;

  //         cout << " newvel[i] " << newvel[i] 
//		<< " fdt*dt" << fdt*dt
//		<< " force " << Force[i] << endl;
        }

    }


 void network::solverleapfrog(float dt){

     const unsigned int sz = newvel.size();

      for (int i = 0; i < sz; i++) {
            newvel[i]= newvel[i] + Force[i]*dt;
     //       cout << " velocity " << newvel[i] << endl;
        }



}


int network::numberoffibers(int NeOld){

	const unsigned int sz = linkr.size();
	unsigned int count;
	count = 0;
	for(int i = 0; i < sz; i++){

		count += linkr[i].size();

	}


	const unsigned int sz2 = linkr.size();
	unsigned int countOld;
	countOld = 0;
	for(int k6 = 0; k6 < sz2; k6++){

		countOld += linkr[k6].size();

	}

	int out;

	if( NeOld == 0) out = countOld;
	else out = count;


	return out;

}


double network::getEqforce(const double tzs, const double tze){


	double out;
	double sum = 0.0;
	double Sx = 0.0;
	double Sy = 0.0;
	double Sz = 0.0;

	const double nsz = size();

       // cout << "size " << nsz << endl;

	//getSpringForce(newp,1);

	for(int i = 0; i < nsz ; i++){

            if(node[3*i+2] >= tze ){
		const double temp = Force[3*i]*Force[3*i] + Force[3*i+1]*Force[3*i+1]
		+ Force[3*i+2]*Force[3*i+2];
                const double SS = sqrt(temp);
                sum += SS;
            }

	}

        out = sum;
	//out = Sx + Sy + Sz;
	return out;
}

double network::getAveDiff(vector<double>& Pos){

	double ol,rl;
	double sum = 0.0;
	int count = 0;
	const unsigned int sz = size();

	for(int i = 0; i < sz; i++ ){

		// add force over all connecting nodes
        //const double linksz = nlinkr[i].size();
        const double linksz = linkr[i].size();
        
        for(int j = 1; j  < linksz; j++){

		//int k = nlinkr[i][j];
                int k = linkr[i][j];
       		int oindex = linkr[i][0];


            	if(k > oindex){


				double rlx = (Pos[3*k]-Pos[3*oindex]);
				double rly = (Pos[3*k+1]-Pos[3*oindex+1]);
				double rlz = (Pos[3*k+2]-Pos[3*oindex+2]);

				rl = sqrt(rlx*rlx + rly*rly + rlz*rlz);


				// new length
				const double lx = (newp[3*oindex]-newp[3*k]);
				const double ly = (newp[3*oindex+1]-newp[3*k+1]);
				const double lz = (newp[3*oindex+2]-newp[3*k+2]);

				ol = sqrt(lx*lx + ly*ly + lz*lz);
				count++;
			}
		}
                if(rl != 0.0 ) {

                        const double ratio = (ol-rl);
			sum += fabs(ratio);
			
                        }
	}

	double out = sum/count;
	return out;
}


double network::getAveStrain(const double txe){

        const unsigned int sz = linkr.size();
        double ol,rl,deltal;
        ol = 0.0; rl = 0.0; deltal = 0.0;

        int ct = 0;
        for(int i = 0; i < sz; i++ ){

                const unsigned int linksz = linkr[i].size();

                for(int j = 1; j  < linksz; j++){

                        int k, oindex;
                        //oindex = linkr[i][0];
                        oindex = linkr[i][0];
                        k = linkr[i][j];

                        if(k > oindex){

                                const double rlx = (node[3*k]-node[3*oindex]);
                                const double rly = (node[3*k+1]-node[3*oindex+1]);
                                const double rlz = (node[3*k+2]-node[3*oindex+2]);


                                const double lx = (newp[3*oindex]-newp[3*k]);
                                const double ly = (newp[3*oindex+1]-newp[3*k+1]);
                                const double lz = (newp[3*oindex+2]-newp[3*k+2]);

                                {
                                        ol = sqrt(lx*lx + ly*ly + lz*lz);
                                        rl = sqrt(rlx*rlx + rly*rly + rlz*rlz);
                                }
                        }
                        ct++;
                if(rl != 0.0)  {
			//if(node[3*i] > txe)
			{
                        const double ratio = (ol-rl)/rl;
			deltal += ratio;
                        //cout << ol << " " << rl  << " " << ol-rl <<  " " << (ol-rl)/rl << " aveST: " << deltal << endl;
			}
                        }
                }
        }
        const double out= deltal/(double)ct;
        return out;
}
