#include <fstream>
#include "poisson.hpp"

namespace poisson {

Poisson2d::Poisson2d(Structmesh2d* mesh, std::string gradientscheme, std::string solvers, std::vector<int> bcflags, std::vector<double> bcvalues, int iters, double tolerance)
	: m(mesh), nesuel{4}, g{nullptr}, solver{nullptr},
	bcflag(bcflags), bvalue(bcvalues), maxiter(iters), tol(tolerance)
{
	std::cout << "Poisson2d: Selecting gradient scheme..." << std::endl;
	if(gradientscheme == "thinlayer")
		g = new ThinLayerGradient();
	else if(gradientscheme == "normtan")
		g = new NormTanGradient();
	else
		std::cout << "Poisson2d: Invalid gradient scheme. Please use 'thinlayer' or 'normtan'." << std::endl;

	std::cout << "Poisson2d: selecting solver..." << std::endl;
	if(solvers == "pj")
		solver = new AFpj();
	else if(solvers == "ilu")
		solver = new AFilu();
	else 
		std::cout << "Poisson2d: Invalid linear solver. Please use 'pj' or 'ilu'." << std::endl;

	a = new amat::Array2d<double>[nesuel+1];		// for each element, 4 surrounding elements and the element itself
	u.setup(m->gimx()+1,m->gjmx()+1);
	res.setup(m->gimx()+1,m->gjmx()+1);
	f.setup(m->gimx()+1,m->gjmx()+1);
	for(int i = 0; i < nesuel+1; i++)
		a[i].setup(m->gimx()+1,m->gjmx()+1);

	// set initial solution
	u.zeros();
	
	g->setup(m, &u, &res, a, bcflag, bvalue);
	g->compute_s();

	// compute LHS
	std::cout << "Poisson2d: Computing CV vols" << std::endl;
	g->compute_CV_volumes();
	std::cout << "Poisson2d: Computing LHS coeffs" << std::endl;
	g->compute_lhs();
	compute_source();
	
	solver->setup(m, a, &res, &u);

	std::cout << "Poisson2d: Problem set up done." << std::endl;
}

Poisson2d::~Poisson2d()
{
	delete [] a;
	delete g;
	delete solver;
}

double Poisson2d::source(double x, double y)
{
	return exp(-35.0*((x-xcentre)*(x-xcentre) + (y-ycentre)*(y-ycentre)));
}

void Poisson2d::compute_source()
{
	std::cout << "Poisson2d: compute_source(): Computing source terms." << std::endl;
	xcentre = 0; ycentre = 0;
	double volsum = 0;
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
		{
			volsum += m->gvol(i,j);
			xcentre += m->gxc(i,j)*m->gvol(i,j);
			ycentre += m->gyc(i,j)*m->gvol(i,j);
		}
	xcentre = xcentre/volsum;
	ycentre = ycentre/volsum;

	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
			f(i,j) = source(m->gxc(i,j),m->gyc(i,j))*m->gvol(i,j);
	std::cout << "Poisson2d: source terms computed" << std::endl;
}

/**	Assigns ghost cell values according to boundary values specified in [bvalue](@ref bvalue).
 * bcvalue contains either value of u or value of \f$ \frac{\partial u}{\partial n} \f$ for the 4 faces.
 * If the boundary is set as Dirichlet in [bcflag](@ref bcflag), these are used in the flux computation; otherwise they are ignored.
 */
void Poisson2d::setDirichletBCs()
{
	for(int i = 1; i <= m->gimx()-1; i++)
	{
		u(i,0) = 2*bvalue[3] - u(i,1);
		u(i,m->gjmx()) = 2*bvalue[1] - u(i,m->gjmx()-1);
	}

	for(int j = 1; j <= m->gjmx()-1; j++)
	{
		u(0,j) = 2*bvalue[2] - u(1,j);
		u(m->gimx(),j) = 2*bvalue[0] - u(m->gimx()-1,j);
	}
}

/** Sets ghost cell values for both Dirichlet BCs and Neumann BCs.
 */
void Poisson2d::setBCs()
{
	//std::cout << "Poisson2d: setBCs(): Setting BCs" << std::endl;
	double dn;
	// dn is the perpendicular distance from cell center to boundary face.

	int j = 0;
	if(bcflag[3] > 0)
		for(int i = 1; i <= m->gimx()-2; i++)
		{
			dn = abs((m->gy(i+1,j+1)-m->gy(i,j+1))*m->gxc(i,j+1) - (m->gx(i+1,j+1)-m->gx(i,j+1))*m->gyc(i,j+1) + m->gx(i+1,j+1)*m->gy(i,j+1) - m->gy(i+1,j+1)*m->gx(i,j+1))
				/ sqrt((m->gy(i+1,j+1)-m->gy(i,j+1))*(m->gy(i+1,j+1)-m->gy(i,j+1)) + (m->gx(i+1,j+1)-m->gx(i,j+1))*(m->gx(i+1,j+1)-m->gx(i,j+1)));
			u(i,j) = u(i,j+1) - 2*dn*bvalue[3];
		}
	else
		for(int i = 1; i <= m->gimx()-2; i++)
			u(i,j) = 2*bvalue[3] - u(i,j+1);
	
	j = m->gjmx();
	if(bcflag[1] > 0)
		for(int i = 1; i<=m->gimx()-2; i++)
		{
			dn = abs((m->gy(i+1,j)-m->gy(i,j))*m->gxc(i,j-1) - (m->gx(i+1,j)-m->gx(i,j))*m->gyc(i,j-1) + m->gx(i+1,j)*m->gy(i,j) - m->gy(i+1,j)*m->gx(i,j))
				/ sqrt((m->gy(i+1,j)-m->gy(i,j))*(m->gy(i+1,j)-m->gy(i,j)) + (m->gx(i+1,j)-m->gx(i,j))*(m->gx(i+1,j)-m->gx(i,j)));
			u(i,j) = u(i,j-1) - 2*dn*bvalue[1];
		}
	else
		for(int i = 1; i <= m->gimx()-2; i++)
			u(i,j) = 2*bvalue[3] - u(i,j-1);
	
	int i = 0;
	if(bcflag[2] > 0)
		for(int j = 1; j <= m->gjmx()-2; j++)
		{
			dn = abs((m->gy(i+1,j+1)-m->gy(i+1,j))*m->gxc(i+1,j) - (m->gx(i+1,j+1)-m->gx(i+1,j))*m->gyc(i+1,j) + m->gx(i+1,j+1)*m->gy(i+1,j) - m->gy(i+1,j+1)*m->gx(i+1,j))
				/ sqrt((m->gy(i+1,j+1)-m->gy(i+1,j))*(m->gy(i+1,j+1)-m->gy(i+1,j)) + (m->gx(i+1,j+1)-m->gx(i+1,j))*(m->gx(i+1,j+1)-m->gx(i+1,j)));
			u(i,j) = u(i+1,j) - 2*dn*bvalue[2];
		}
	else
		for(int j = 1; j <= m->gjmx()-2; j++)
			u(i,j) = 2*bvalue[2] - u(i+1,j);
	
	i = m->gimx();
	if(bcflag[0] > 0)
		for(int j = 1; j <= m->gjmx()-2; j++)
		{
			dn = abs((m->gy(i,j+1)-m->gy(i,j))*m->gxc(i-1,j) - (m->gx(i,j+1)-m->gx(i,j))*m->gyc(i-1,j) + m->gx(i,j+1)*m->gy(i,j) - m->gy(i,j+1)*m->gx(i,j))
				/ sqrt((m->gy(i,j+1)-m->gy(i,j))*(m->gy(i,j+1)-m->gy(i,j)) + (m->gx(i,j+1)-m->gx(i,j))*(m->gx(i,j+1)-m->gx(i,j)));
			u(i,j) = u(i-1,j) - 2*dn*bvalue[0];
		}
	else
		for(int j = 1; j <= m->gjmx()-2; j++)
			u(i,j) = 2*bvalue[0] - u(i-1,j);
	
	// Corner ghost cells. Their values are just taken as average of the two neighboring ghost cells. This will hopefully not induce much error.
	u(0,0) = 0.5*(u(0,1)+u(1,0));
	u(m->gimx(),0) = 0.5*(u(m->gimx()-1,0) + u(m->gimx(),1));
	u(0,m->gjmx()) = 0.5*(u(1,m->gjmx()) + u(0,m->gjmx()-1));
	u(m->gimx(),m->gjmx()) = 0.5*(u(m->gimx(),m->gjmx()-1) + u(m->gimx()-1,m->gjmx()));

	//std::cout << "Poisson2d: setBCs(): BCs set." << std::endl;
}

void Poisson2d::solve()
{
	int k = 0;
	double resnorm, resnorm0;
	bool converged = false;

	do {
		res.zeros();

		setBCs();

		g->compute_fluxes();

		//add source term
		for(int i = 1; i < m->gimx()-1; i++)
			for(int j = 1; j < m->gjmx()-1; j++)
				res(i,j) -= f(i,j);

		// get resnorm
		resnorm = 0;
		for(int i = 1; i <= m->gimx()-1; i++)
			for(int j = 1; j < m->gjmx()-1; j++)
				resnorm += res.get(i,j)*res.get(i,j);
		resnorm = sqrt(resnorm);
		if(k == 0) resnorm0 = resnorm;
		resnorms.push_back(resnorm/resnorm0);
		
		if(resnorm/resnorm0 < tol)
		{
			std::cout << "Poisson2d: solve(): Solver converged in " << k << " iterations." << std::endl;
			std::cout << "Poisson2d: solve(): Final area-weighted residual is " << resnorm*(1.0/(m->gimx()-1)) << std::endl;
			converged = true;
			break;
		}

		//update
		solver->update();
		
		if(k%10 == 0)
			std::cout << "Poisson2d: solve(): Iteration " << k << ", residual = " << resnorm/resnorm0 << std::endl;
		k++;
	} while(k < maxiter);

	if(!converged) { 
		std::cout << "Poisson2d: solve(): Solver could not converge!!" << std::endl;
	}
}

amat::Array2d<double> Poisson2d::getSolution() const
{ return u; }

amat::Array2d<double> Poisson2d::getPointSolution() const
{ 
	amat::Array2d<double> upoint(m->gimx()+1,m->gjmx()+1);
	// interior points
	for(int i = 1; i <= m->gimx(); i++)
		for(int j = 1; j <= m->gjmx(); j++)
		{
			upoint(i,j) = (u.get(i,j) + u.get(i-1,j) + u.get(i,j-1) + u.get(i-1,j-1))*0.25;
		}

	return upoint; 
}

void Poisson2d::export_convergence_data(std::string fname) const
{
	std::ofstream fout(fname);
	for(int i = 0; i < static_cast<int>(resnorms.size()); i++)
		fout << i << " " << log10(resnorms[i]) << '\n';
	fout.close();
}

}
