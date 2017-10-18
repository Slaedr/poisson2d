#include "afsolver.hpp"

namespace poisson {

void AFSolver::setup(Structmesh2d* mesh, amat::Array2d<double>* aa, amat::Array2d<double>* residual, amat::Array2d<double>* uu)
{
	m = mesh;
	a = aa;
	res = residual;
	u = uu;
	du.setup(m->gimx()+1, m->gjmx()+1);
	du.zeros();
}

void AFpj::update()
{
	// solve
	// NOTE: we use -res instead of res as we need b-Ax on RHS, not Ax-b, and the latter is what is assumed as input.
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
			du(i,j) = -1.0*res->get(i,j)/a[2](i,j);

	// update
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
			(*u)(i,j) += du(i,j);
}

void AFilu::setup(Structmesh2d* mesh, amat::Array2d<double>* aa, amat::Array2d<double>* residual, amat::Array2d<double>* uu)
{
	m = mesh;
	a = aa;
	res = residual;
	u = uu;
	du.setup(m->gimx()+1, m->gjmx()+1);
	du.zeros();

	// Compute modified diagonal terms
	double termi, termj;
	d.setup(m->gimx(),m->gjmx());
	for(int j = 1; j <= m->gjmx()-1; j++)
		for(int i = 1; i <= m->gimx()-1; i++)
		{
			termi = 0;
			if(i > 1)
				termi = a[0].get(i,j)*a[4].get(i-1,j)/d.get(i-1,j);
			termj = 0;
			if(j > 1)
				termj = a[1].get(i,j)*a[3].get(i,j-1)/d.get(i,j-1);
			d(i,j) = a[2].get(i,j) - termi - termj;
		}
}

void AFilu::update()
{
	// forward Gauss-Seidel sweep
	for(int j = 1; j <= m->gjmx()-1; j++)
		for(int i = 1; i <= m->gimx()-1; i++)
			du(i,j) = (-res->get(i,j) - a[0].get(i,j)*du.get(i-1,j) - a[1].get(i,j)*du.get(i,j-1)) / d.get(i,j);

	// backward Gauss-Seidel sweep
	for(int j = m->gjmx()-1; j >= 1; j--)
		for(int i = m->gimx()-1; i >= 1; i--)
			du(i,j) -= (a[3].get(i,j)*du.get(i,j+1) + a[4].get(i,j)*du.get(i+1,j)) / d.get(i,j);

	// update
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
			(*u)(i,j) += du.get(i,j);
}

}
