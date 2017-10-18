#include <ctime>
#include "aoutput_struct.hpp"

#include "poisson.hpp"

using namespace std;
using namespace amat;
using namespace poisson;

int main(int argc, char* argv[])
{
	string dum, solver, gradscheme, meshfile, solfile;
	double tol;
	int maxiter;
	vector<int> bcflag(4,-1);
	vector<double> bvalue(4,0);

	if(argc < 2) {
		cout << "Insufficient command line arguments. Please give a control file name." << endl;
		return -1;
	}
	
	ifstream fin(argv[1]);
	fin >> dum; fin >> meshfile;
	fin >> dum; fin >> solfile;
	fin >> dum; fin >> gradscheme;
	fin >> dum; fin >> solver;
	fin >> dum; fin >> tol;
	fin >> dum; fin >> maxiter;
	fin >> dum;
	for(int i = 0; i < 4; i++)
		fin >> bcflag[i];
	fin >> dum;
	for(int i = 0; i < 4; i++)
		fin >> bvalue[i];
	fin.close();

	cout << "Initializing mesh and Poisson solver for:\n";
	cout << meshfile << " " << solfile << " " << gradscheme << " " << solver << " " << tol << " " << maxiter << endl;

	Structmesh2d m;
	m.readmesh(meshfile);
	m.preprocess();

	clock_t begin = clock();
	Poisson2d p(&m, gradscheme, solver, bcflag, bvalue, maxiter, tol);
	//p.compute_source();

	p.solve();
	clock_t dura = clock() - begin;
	double duration = double(dura)/CLOCKS_PER_SEC;

	Array2d<double> u = p.getPointSolution();

	string varname = "u";
	Array2d<double>** vecs = nullptr;
	Structdata2d dout(&m, 1, &u, &varname, 0, vecs, &varname, "poisson2d");
	dout.writevtk(solfile);

	string convfile = solfile;
	convfile.erase(convfile.end()-4,convfile.end());
	convfile = convfile + "_conv.dat";
	p.export_convergence_data(convfile);

	cout << "Time taken for grid " << meshfile << " for gradient scheme " << gradscheme << " using " << solver << " is: " << duration << endl;

	cout << endl;
	return 0;
}
