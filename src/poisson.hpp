#ifndef POISSON_H
#define POISSON_H 1

#include <string>
#include <cmath>
#include <vector>

#include "structmesh2d.hpp"
#include "gradientschemes.hpp"
#include "afsolver.hpp"

namespace poisson {

class Poisson2d
{
	Structmesh2d* m;

	/// Number of cells surrounding a cell (4, in case of 2d structured mesh.
	const int nesuel;				

	amat::Array2d<double>* a;
	amat::Array2d<double> u;
	amat::Array2d<double> res;
	amat::Array2d<double> f;		///< Source term
	std::vector<double> resnorms;

	GradientScheme* g;

	AFSolver* solver;

	/**	Contains a flag for each of the 4 boundaries, indicating the type of boundary 
	 * (0 is Dirichlet, 1 is Neumann).
	 * Note that the std::vector has 4 elements:
	 * 0: boundary i = m->gimx()
	 * 1: boundary j = m->gjmx()
	 * 2: boundary i = 1
	 * 3: boundary j = 1
	 */
	std::vector<int> bcflag;
	
	/// Contains boundary values corresponding to the flags in [bcflag](@ref bcflag)
	std::vector<double> bvalue;

	const int maxiter;
	const double tol;

	double xcentre;
	double ycentre;

public:
	Poisson2d(Structmesh2d* mesh, std::string gradientscheme, std::string solver, 
			std::vector<int> bcflags, std::vector<double> bcvalues, 
			int iters, double tolerance);
	~Poisson2d();
	void setDirichletBCs();
	void setBCs();
	double source(double x, double y);
	void compute_source();
	void solve();
	amat::Array2d<double> getSolution() const;
	
	/**	Computes solution at each point (node) by averaging cell data around the point.
	 * For boundary points, fewer cells are used, 
	 * ie, ghost cell data is not used (as it is not calculated at all.
	 */
	amat::Array2d<double> getPointSolution() const;
	void export_convergence_data(std::string mfile) const;
};

}


#endif
