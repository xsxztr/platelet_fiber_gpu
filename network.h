/*
 * File:   network.h
 * Author: eunjungkim
 *

 */

#ifndef _NETWORK_H
#define	_NETWORK_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

class network {

     struct coord{

        coord(){
            x=y=z=0.0;
        }


     double x,y,z; // (bx,by,bz) stores position of branch point
     };

public:
        network();

        virtual ~network();


        // Set random nodes
	void setNodes(const int number, const double xsize,
                    const double ysize, const double zsize);
	 // average strain
        double getAveStrain(const double txe);

	// store nodes in brachp.x,y,z
	void putIntoVec(double xc, double yc,double zc);

	// Define link between two nodes
        void definelinks(const double distSq); // int : numer of branch point

	//get network size
        unsigned int size();

	// Get coordinates of node
        double getCoords(const unsigned int indx, const unsigned int xyz);

	// put position in 1D vector form
	void storeCoords();


	//Calculate force
        void getSpringForce(vector<double>& nodein,int s, const double Modulus,const double mass);

	// Solve ODE
        void Solve(const double dt, double strain, const double gamma, int s,
		const double Modulus, const double tzs, const double tze, const double tys, 
		const double tye, const double mass);

	// Prescribe boundary conditions
        void getinialPafterF(double strain);


	// Print out node, link, thickness for viewers
        void PrintEqposition(int TIME, int CA);

	void Print(vector<double>& Eqpos,int TIME, ofstream &outFileVis);

	// get boundary conditions
        double getBCs(const unsigned int bci, const unsigned int xyz);

	//Calculate stress
        double calculateStresses(vector<double>& Position, vector<double>& Force);
        double calculateFStresses(vector<double>& Position, vector<double>& Force, const double tx);

	// Remove node if fiber length > 3*intial length of the fiber
        void UpdateLink(vector<double>& NewPos,const double mass);

	// Propagate with friction
        void solverkick( double dt, double myGamma);

	// Leapfrog solver
        void solverleapfrog(float dt);

	// Get number of branching points, thickness, density, etc..
        void getstat();

	// Get Eq Force
	double getEqforce(const double tzs, const double tze);

	//Get difference between two positions
	double getAveDiff(vector < double >& Pos);

        // Get nodes of substrate and top
	void getBNodes();

	// copy linkr after removing node

	void copylinkr();

	int numberoffibers(int NeOld);

	void CheckConstraint(vector <double>& PreP);

public:
        // put position
        vector <coord> branchp;

	// put list info
        vector < std::vector <int> > linkr;

        //vector < vector < int > > nlinkr;

        // put fiberthickness of node i and node j
        vector < vector <double> > fibersize;

	//vector < vector <double> > nfibersize;

	// put position in 1D vector form
        vector < double  > node;

	// store coord at boundary
	vector < int > Bnodesx;
	vector < int > Bnodesz;

	// store Force information between node i and node j
        vector <double> Force;

	// new position
        vector <double> newp;

	// new velocity
        vector <double> newvel;

	//Update Node;
	vector <int> Nnode;
     
        // store number of connections of individual nodes
	vector <int> node_N_conn;

private:

};

#endif	/* _RDNETWORK_H */

