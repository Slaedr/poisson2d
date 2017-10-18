/** Implements classes for some iterative linear solvers.
 */

#ifndef AFSOLVER_H

#include "aarray2d.hpp"
#include "structmesh2d.hpp"

#define AFSOLVER_H 1

namespace poisson {

/**	Base class for classes that implement one update of an approximate factorization (AF) solution
 */
class AFSolver
{
protected:
	Structmesh2d* m;
	amat::Array2d<double>* a;		///< LHS; let's call it A.
	amat::Array2d<double>* res;	///< residual
	amat::Array2d<double>* u;		///< unkown to be solved for
	amat::Array2d<double> du;		///< change vector
public:
	virtual void setup(Structmesh2d* mesh, 
			amat::Array2d<double>* aa, amat::Array2d<double>* residual, amat::Array2d<double>* u);
	virtual ~AFSolver()
	{ }
	virtual void update() = 0;
};

/**	Class implementing the point Jacobi update.
*/
class AFpj : public AFSolver
{
public:
	void update();
};

/** Class implementing ILU update.
*/
class AFilu : public AFSolver
{
	amat::Array2d<double> d;
public:
	void setup(Structmesh2d* mesh, 
			amat::Array2d<double>* aa, amat::Array2d<double>* residual, amat::Array2d<double>* uu);
	void update();
};

}
#endif
