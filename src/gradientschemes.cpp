#include "gradientschemes.hpp"

namespace poisson {

void GradientScheme::setup(Structmesh2d* mesh, amat::Array2d<double>* unknown, amat::Array2d<double>* residual, amat::Array2d<double>* lhs, std::vector<int> bcflags, std::vector<double> bvalues)
{
	m = mesh;
	u = unknown;
	res = residual;
	a = lhs;
	bcflag = bcflags;
	bvalue = bvalues;
	dualvol.resize(2);
	for(int i = 0; i < 2; i++)
		dualvol[i].setup(m->gimx()+1, m->gjmx()+1);
}

void GradientScheme::compute_CV_volumes()
{
	// iterate over cells
	for(int i = 0; i <= m->gimx()-1; i++)
		for(int j = 0; j <= m->gjmx()-1; j++)
			if((i>0 || j>0) && (i < m->gimx() || j < m->gjmx()))
			{
				dualvol[0](i,j) = (m->gvol(i+1,j) + m->gvol(i,j))*0.5;
				dualvol[1](i,j) = (m->gvol(i,j+1) + m->gvol(i,j))*0.5;
			}
	// We can do the above since volumes of all cells, real and ghost, have been computed in Structmesh2d::preprocess.
}

void GradientScheme::compute_s()
{ }

void ThinLayerGradient::compute_lhs()
{
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
		{
			a[0](i,j) = (m->gdel(i-1,j,0)*m->gdel(i-1,j,0) + m->gdel(i-1,j,1)*m->gdel(i-1,j,1))
				/dualvol[0](i-1,j);
			a[1](i,j) = (m->gdel(i,j-1,2)*m->gdel(i,j-1,2) + m->gdel(i,j-1,3)*m->gdel(i,j-1,3))
				/dualvol[1](i,j-1);
			a[3](i,j) = (m->gdel(i,j,2)*m->gdel(i,j,2) + m->gdel(i,j,3)*m->gdel(i,j,3))
				/dualvol[1](i,j);
			a[4](i,j) = (m->gdel(i,j,0)*m->gdel(i,j,0) + m->gdel(i,j,1)*m->gdel(i,j,1))
				/dualvol[0](i,j);
			a[2](i,j) = -1.0 * (a[0](i,j) + a[1](i,j) + a[3](i,j) + a[4](i,j));
		}
}

void ThinLayerGradient::compute_fluxes()
{
	double g,h;
	// First iterate over interior cells. Our real cells start from i=1. i=0 is corresponds to ghost cells.
	for(int i = 2; i <= m->gimx()-2; i++)
		for(int j = 2; j <= m->gjmx()-2; j++)
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}
	
	// Next, now iterate over boundary cells. Apply Neumann BCs.

	// Boundaries j = 1 and j = m->gjmx()
	for(int i = 1; i <= m->gimx()-1; i++)
	{
		// first, j = 1
		int j = 1;
		if(bcflag[3]>=1)	// if there's a Neumann condition specified
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) + bvalue[3];
			(*res)(i,j) += g+h;
		}
		else	// if there's only a Dirichlet BC specified, it'll be taken care of by ghost cell values.
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}

		j = m->gjmx()-1;
		if(bcflag[1]>=1)
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = bvalue[1] - (u->get(i,j)-u->get(i,j-1))*a[0](i,j);
			(*res)(i,j) += g+h;
		}
		else
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}
	}

	// Boundaries i = 1 and i = m->gimx(). We don't want to do the corner cells again.
	for(int j = 2; j <= m->gjmx()-2; j++)
	{
		int i = 1;
		if(bcflag[2]>=1)
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - bvalue[2];
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}
		else
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}

		i = m->gimx()-1;
		if(bcflag[0]>=1)
		{
			g = bvalue[0] - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}
		else
		{
			g = (u->get(i+1,j) - u->get(i,j))*a[4](i,j) - (u->get(i,j)-u->get(i-1,j))*a[0](i,j);
			h = (u->get(i,j+1)-u->get(i,j))*a[3](i,j) - (u->get(i,j)-u->get(i,j-1))*a[1](i,j);
			(*res)(i,j) += g+h;
		}
	}
}

void NormTanGradient::compute_s()
{
	svect.resize(4);
	for(int i = 0; i < 4; i++)
		svect[i].setup(m->gimx()+1, m->gjmx()+1);
	
	dels.resize(2);
	for(int i = 0; i < 2; i++)
		dels[i].setup(m->gimx()+1, m->gjmx()+1);

	/* NOTE: We are not calculating svect and dels for the ghost cells i=imx+1 and j=jmx+1. */
	
	for(int i = 0; i <= m->gimx()-1; i++)
		for(int j = 0; j <= m->gjmx()-1; j++)
		{
			svect[0](i,j) = m->gxc(i+1,j) - m->gxc(i,j);
			svect[1](i,j) = m->gyc(i+1,j) - m->gyc(i,j);
			dels[0](i,j) = sqrt(svect[0](i,j)*svect[0](i,j) + svect[1](i,j)*svect[1](i,j));
			svect[0](i,j) /= dels[0](i,j);
			svect[1](i,j) /= dels[0](i,j);
			
			svect[2](i,j) = m->gxc(i,j+1) - m->gxc(i,j);
			svect[3](i,j) = m->gyc(i,j+1) - m->gyc(i,j);
			dels[1](i,j) = sqrt(svect[2](i,j)*svect[2](i,j) + svect[3](i,j)*svect[3](i,j));
			svect[2](i,j) /= dels[1](i,j);
			svect[3](i,j) /= dels[1](i,j);
		}
}

/** As an approximation, we use the same coefficients as for the Thin Layer gradient model. */
void NormTanGradient::compute_lhs()
{
	std::vector<double> temp(2);
	
	for(int i = 1; i <= m->gimx()-1; i++)
		for(int j = 1; j <= m->gjmx()-1; j++)
		{
			a[0](i,j) = (m->gdel(i-1,j,0)*m->gdel(i-1,j,0) + m->gdel(i-1,j,1)*m->gdel(i-1,j,1))/dualvol[0](i-1,j);
			a[1](i,j) = (m->gdel(i,j-1,2)*m->gdel(i,j-1,2) + m->gdel(i,j-1,3)*m->gdel(i,j-1,3))/dualvol[1](i,j-1);
			a[3](i,j) = (m->gdel(i,j,2)*m->gdel(i,j,2) + m->gdel(i,j,3)*m->gdel(i,j,3))/dualvol[1](i,j);
			a[4](i,j) = (m->gdel(i,j,0)*m->gdel(i,j,0) + m->gdel(i,j,1)*m->gdel(i,j,1))/dualvol[0](i,j);
			a[2](i,j) = -1.0 * (a[0](i,j) + a[1](i,j) + a[3](i,j) + a[4](i,j));
		}
}

void NormTanGradient::compute_fluxes()
{
	amat::Array2d<double> cdelu(5,2);		// to store cell-wise gradient values for each of the 5 cells in a loop iteration.

	// first iterate over interior cells
	for(int i = 2; i <= m->gimx()-2; i++)
		for(int j = 2; j <= m->gjmx()-2; j++)
		{
			for(int d = 0; d < 2; d++)
			{
				// i,j-1
				cdelu(0,d) = 0.5/m->gvol(i,j-1)*( (u->get(i,j-1)+u->get(i+1,j-1))*m->gdel(i,j-1,d) + (u->get(i,j-1)+u->get(i,j))*m->gdel(i,j-1,2+d)
					- (u->get(i,j-1)+u->get(i-1,j-1))*m->gdel(i-1,j-1,d) - (u->get(i,j-1)+u->get(i,j-2))*m->gdel(i,j-2,2+d));
				//i-1,j
				cdelu(1,d) = 0.5/m->gvol(i-1,j)*( (u->get(i-1,j)+u->get(i,j))*m->gdel(i-1,j,d) + (u->get(i-1,j)+u->get(i-1,j+1))*m->gdel(i-1,j,2+d)
					- (u->get(i-1,j)+u->get(i-2,j))*m->gdel(i-2,j,d) - (u->get(i-1,j)+u->get(i-1,j-1))*m->gdel(i-1,j-1,2+d));
				//i,j
				cdelu(2,d) = 0.5/m->gvol(i,j)*( (u->get(i,j)+u->get(i+1,j))*m->gdel(i,j,d) + (u->get(i,j)+u->get(i,j+1))*m->gdel(i,j,2+d)
					- (u->get(i,j)+u->get(i-1,j))*m->gdel(i-1,j,d) - (u->get(i,j)+u->get(i,j-1))*m->gdel(i,j-1,2+d));
				//i+1,j
				cdelu(3,d) = 0.5/m->gvol(i+1,j)*( (u->get(i+1,j)+u->get(i+2,j))*m->gdel(i+1,j,d) + (u->get(i+1,j)+u->get(i+1,j+1))*m->gdel(i+1,j,2+d)
					- (u->get(i+1,j)+u->get(i,j))*m->gdel(i,j,d) - (u->get(i+1,j)+u->get(i+1,j-1))*m->gdel(i+1,j-1,2+d));
				//i,j+1
				cdelu(4,d) = 0.5/m->gvol(i,j+1)*( (u->get(i,j+1)+u->get(i+1,j+1))*m->gdel(i,j+1,d) + (u->get(i,j+1)+u->get(i,j+2))*m->gdel(i,j+1,2+d)
					- (u->get(i,j+1)+u->get(i-1,j+1))*m->gdel(i-1,j+1,d) - (u->get(i,j+1)+u->get(i,j))*m->gdel(i,j,2+d));
			}
			
			// now calculate contributions from the 4 faces
			(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(3,0))*m->gdel(i,j,0)+(cdelu(2,1)+cdelu(3,1))*m->gdel(i,j,1) 
				- ((cdelu(2,0)+cdelu(3,0))*svect[0](i,j)+(cdelu(2,1)+cdelu(3,1))*svect[1](i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1)))
				+ (u->get(i+1,j)-u->get(i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1))/dels[0](i,j);
			(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(1,0))*m->gdel(i-1,j,0)+(cdelu(2,1)+cdelu(1,1))*m->gdel(i-1,j,1) 
				- ((cdelu(2,0)+cdelu(1,0))*svect[0](i-1,j)+(cdelu(2,1)+cdelu(1,1))*svect[1](i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1)))
				+ (u->get(i,j)-u->get(i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1))/dels[0](i-1,j);
			(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(0,0))*m->gdel(i,j-1,2)+(cdelu(2,1)+cdelu(0,1))*m->gdel(i,j-1,3) 
				- ((cdelu(2,0)+cdelu(0,0))*svect[2](i,j-1)+(cdelu(2,1)+cdelu(0,1))*svect[3](i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3)))
				+ (u->get(i,j)-u->get(i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3))/dels[1](i,j-1);
			(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(4,0))*m->gdel(i,j,2)+(cdelu(2,1)+cdelu(4,1))*m->gdel(i,j,3) 
				- ((cdelu(2,0)+cdelu(4,0))*svect[2](i,j)+(cdelu(2,1)+cdelu(4,1))*svect[3](i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3)))
				+ (u->get(i,j+1)-u->get(i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3))/dels[1](i,j);
		}
	
	// boundary cells
	// If it's a homogeneous Neumann boundary, the "second" ghost cell value is the same as the ghost cell value;
	// If it's a Dirichlet boundary, the value is 2*bvalue - u(second interior cell).
	// We take care of corner boundary cells during the treatment of boundaries 1 and 3 (the i=const boundaries).
	// \todo TODO: Implement this second ghost cell value for non-homogeneous Neumann condition.
	
	// boundary 1
	double secondghost;
	double sg1, sg2, sg3, sg4;		// for corner boundary cells
	std::vector<double> arvec(2);		// del for second ghost cells
	std::vector<double> crvec(2);		// extra del required in the 4 corner boundary cells

	if(bcflag[3] >= 1)	// Neumann
	{
		sg1 = u->get(m->gimx()-1,0);
		sg3 = u->get(1,0);
	}
	else {
		sg1 = 2*bvalue[3] - u->get(m->gimx()-1,2);
		sg3 = 2*bvalue[3] - u->get(1,2);
	}
	
	if(bcflag[1] >= 1)	// Neumann
	{
		sg2 = u->get(m->gimx()-1,m->gjmx());
		sg4 = u->get(1,m->gjmx());
	}
	else {
		sg2 = 2*bvalue[1] - u->get(m->gimx()-1,m->gjmx()-2);
		sg4 = 2*bvalue[1] - u->get(1,m->gjmx()-2);
	}

	int i = m->gimx()-1;
	crvec[0] = -(m->gy(i+1,0) - m->gy(i,0));
	crvec[1] = m->gx(i+1,0) - m->gx(i,0);
	for(int j = 1; j <= m->gjmx()-1; j++)
	{
		if(bcflag[0] >= 1)
			secondghost = u->get(i+1,j);
		else
			secondghost = 2*bvalue[0] - u->get(i-1,j);
		
		for(int d = 0; d < 2; d++)
		{
			// i,j-1
			cdelu(0,d) = 0.5/m->gvol(i,j-1)*( (u->get(i,j-1)+u->get(i+1,j-1))*m->gdel(i,j-1,d) + (u->get(i,j-1)+u->get(i,j))*m->gdel(i,j-1,2+d)
				- (u->get(i,j-1)+u->get(i-1,j-1))*m->gdel(i-1,j-1,d) - (u->get(i,j-1)+( j>1 ? u->get(i,j-2):sg1))*( j>1 ? m->gdel(i,j-2,2+d):crvec[d]));
			//i-1,j
			cdelu(1,d) = 0.5/m->gvol(i-1,j)*( (u->get(i-1,j)+u->get(i,j))*m->gdel(i-1,j,d) + (u->get(i-1,j)+u->get(i-1,j+1))*m->gdel(i-1,j,2+d)
				- (u->get(i-1,j)+u->get(i-2,j))*m->gdel(i-2,j,d) - (u->get(i-1,j)+u->get(i-1,j-1))*m->gdel(i-1,j-1,2+d));
			//i,j
			cdelu(2,d) = 0.5/m->gvol(i,j)*( (u->get(i,j)+u->get(i+1,j))*m->gdel(i,j,d) + (u->get(i,j)+u->get(i,j+1))*m->gdel(i,j,2+d)
				- (u->get(i,j)+u->get(i-1,j))*m->gdel(i-1,j,d) - (u->get(i,j)+u->get(i,j-1))*m->gdel(i,j-1,2+d));
			//i+1,j
			cdelu(3,d) = 0.5/m->gvol(i+1,j)*( (u->get(i+1,j)+secondghost)*m->gdel(i+1,j,d) + (u->get(i+1,j)+u->get(i+1,j+1))*m->gdel(i+1,j,2+d)
				- (u->get(i+1,j)+u->get(i,j))*m->gdel(i,j,d) - (u->get(i+1,j)+u->get(i+1,j-1))*m->gdel(i+1,j-1,2+d));
			//i,j+1
			cdelu(4,d) = 0.5/m->gvol(i,j+1)*( (u->get(i,j+1)+u->get(i+1,j+1))*m->gdel(i,j+1,d) + (u->get(i,j+1)+( j < m->gjmx()-1 ? u->get(i,j+2):sg2))*m->gdel(i,j+1,2+d)
				- (u->get(i,j+1)+u->get(i-1,j+1))*m->gdel(i-1,j+1,d) - (u->get(i,j+1)+u->get(i,j))*m->gdel(i,j,2+d));
		}
		
		// now calculate contributions from the 4 faces
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(3,0))*m->gdel(i,j,0)+(cdelu(2,1)+cdelu(3,1))*m->gdel(i,j,1) 
			- ((cdelu(2,0)+cdelu(3,0))*svect[0](i,j)+(cdelu(2,1)+cdelu(3,1))*svect[1](i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1)))
			+ (u->get(i+1,j)-u->get(i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1))/dels[0](i,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(1,0))*m->gdel(i-1,j,0)+(cdelu(2,1)+cdelu(1,1))*m->gdel(i-1,j,1) 
			- ((cdelu(2,0)+cdelu(1,0))*svect[0](i-1,j)+(cdelu(2,1)+cdelu(1,1))*svect[1](i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1)))
			+ (u->get(i,j)-u->get(i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1))/dels[0](i-1,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(0,0))*m->gdel(i,j-1,2)+(cdelu(2,1)+cdelu(0,1))*m->gdel(i,j-1,3) 
			- ((cdelu(2,0)+cdelu(0,0))*svect[2](i,j-1)+(cdelu(2,1)+cdelu(0,1))*svect[3](i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3)))
			+ (u->get(i,j)-u->get(i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3))/dels[1](i,j-1);
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(4,0))*m->gdel(i,j,2)+(cdelu(2,1)+cdelu(4,1))*m->gdel(i,j,3) 
			- ((cdelu(2,0)+cdelu(4,0))*svect[2](i,j)+(cdelu(2,1)+cdelu(4,1))*svect[3](i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3)))
			+ (u->get(i,j+1)-u->get(i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3))/dels[1](i,j);
	}

	// boundary 2
	int j = m->gjmx()-1;
	for(int i = 2; i <= m->gimx()-2; i++)
	{
		if(bcflag[1] >= 1)
			secondghost = u->get(i,j+1);
		else
			secondghost = 2*bvalue[1] - u->get(i,j-1);
		for(int d = 0; d < 2; d++)
		{
			// i,j-1
			cdelu(0,d) = 0.5/m->gvol(i,j-1)*( (u->get(i,j-1)+u->get(i+1,j-1))*m->gdel(i,j-1,d) + (u->get(i,j-1)+u->get(i,j))*m->gdel(i,j-1,2+d)
				- (u->get(i,j-1)+u->get(i-1,j-1))*m->gdel(i-1,j-1,d) - (u->get(i,j-1)+u->get(i,j-2))*m->gdel(i,j-2,2+d));
			//i-1,j
			cdelu(1,d) = 0.5/m->gvol(i-1,j)*( (u->get(i-1,j)+u->get(i,j))*m->gdel(i-1,j,d) + (u->get(i-1,j)+u->get(i-1,j+1))*m->gdel(i-1,j,2+d)
				- (u->get(i-1,j)+u->get(i-2,j))*m->gdel(i-2,j,d) - (u->get(i-1,j)+u->get(i-1,j-1))*m->gdel(i-1,j-1,2+d));
			//i,j
			cdelu(2,d) = 0.5/m->gvol(i,j)*( (u->get(i,j)+u->get(i+1,j))*m->gdel(i,j,d) + (u->get(i,j)+u->get(i,j+1))*m->gdel(i,j,2+d)
				- (u->get(i,j)+u->get(i-1,j))*m->gdel(i-1,j,d) - (u->get(i,j)+u->get(i,j-1))*m->gdel(i,j-1,2+d));
			//i+1,j
			cdelu(3,d) = 0.5/m->gvol(i+1,j)*( (u->get(i+1,j)+u->get(i+2,j))*m->gdel(i+1,j,d) + (u->get(i+1,j)+u->get(i+1,j+1))*m->gdel(i+1,j,2+d)
				- (u->get(i+1,j)+u->get(i,j))*m->gdel(i,j,d) - (u->get(i+1,j)+u->get(i+1,j-1))*m->gdel(i+1,j-1,2+d));
			//i,j+1
			cdelu(4,d) = 0.5/m->gvol(i,j+1)*( (u->get(i,j+1)+u->get(i+1,j+1))*m->gdel(i,j+1,d) + (u->get(i,j+1)+secondghost)*m->gdel(i,j+1,2+d)
				- (u->get(i,j+1)+u->get(i-1,j+1))*m->gdel(i-1,j+1,d) - (u->get(i,j+1)+u->get(i,j))*m->gdel(i,j,2+d));
		}
		
		// now calculate contributions from the 4 faces
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(3,0))*m->gdel(i,j,0)+(cdelu(2,1)+cdelu(3,1))*m->gdel(i,j,1) 
			- ((cdelu(2,0)+cdelu(3,0))*svect[0](i,j)+(cdelu(2,1)+cdelu(3,1))*svect[1](i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1)))
			+ (u->get(i+1,j)-u->get(i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1))/dels[0](i,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(1,0))*m->gdel(i-1,j,0)+(cdelu(2,1)+cdelu(1,1))*m->gdel(i-1,j,1) 
			- ((cdelu(2,0)+cdelu(1,0))*svect[0](i-1,j)+(cdelu(2,1)+cdelu(1,1))*svect[1](i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1)))
			+ (u->get(i,j)-u->get(i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1))/dels[0](i-1,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(0,0))*m->gdel(i,j-1,2)+(cdelu(2,1)+cdelu(0,1))*m->gdel(i,j-1,3) 
			- ((cdelu(2,0)+cdelu(0,0))*svect[2](i,j-1)+(cdelu(2,1)+cdelu(0,1))*svect[3](i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3)))
			+ (u->get(i,j)-u->get(i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3))/dels[1](i,j-1);
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(4,0))*m->gdel(i,j,2)+(cdelu(2,1)+cdelu(4,1))*m->gdel(i,j,3) 
			- ((cdelu(2,0)+cdelu(4,0))*svect[2](i,j)+(cdelu(2,1)+cdelu(4,1))*svect[3](i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3)))
			+ (u->get(i,j+1)-u->get(i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3))/dels[1](i,j);
	}

	// boundary 3
	i = 1;
	crvec[0] = -(m->gy(i+1,0) - m->gy(i,0));
	crvec[1] = m->gx(i+1,0) - m->gx(i,0);
	for(int j = 1; j <= m->gjmx()-1; j++)
	{
		if(bcflag[2] >= 1)
			secondghost = u->get(i-1,j);
		else
			secondghost = 2*bvalue[2] - u->get(i+1,j);
		arvec[0] = m->gy(i-1,j+1) - m->gy(i-1,j);
		arvec[1] = -(m->gx(i-1,j+1) - m->gx(i-1,j));
		for(int d = 0; d < 2; d++)
		{
			// i,j-1
			cdelu(0,d) = 0.5/m->gvol(i,j-1)*( (u->get(i,j-1)+u->get(i+1,j-1))*m->gdel(i,j-1,d) + (u->get(i,j-1)+u->get(i,j))*m->gdel(i,j-1,2+d)
				- (u->get(i,j-1)+u->get(i-1,j-1))*m->gdel(i-1,j-1,d) - (u->get(i,j-1)+(j>1 ? u->get(i,j-2):sg3))*(j>1 ? m->gdel(i,j-2,2+d):crvec[d]));
			//i-1,j
			cdelu(1,d) = 0.5/m->gvol(i-1,j)*( (u->get(i-1,j)+u->get(i,j))*m->gdel(i-1,j,d) + (u->get(i-1,j)+u->get(i-1,j+1))*m->gdel(i-1,j,2+d)
				- (u->get(i-1,j)+secondghost)*arvec[d] - (u->get(i-1,j)+u->get(i-1,j-1))*m->gdel(i-1,j-1,2+d));
			//i,j
			cdelu(2,d) = 0.5/m->gvol(i,j)*( (u->get(i,j)+u->get(i+1,j))*m->gdel(i,j,d) + (u->get(i,j)+u->get(i,j+1))*m->gdel(i,j,2+d)
				- (u->get(i,j)+u->get(i-1,j))*m->gdel(i-1,j,d) - (u->get(i,j)+u->get(i,j-1))*m->gdel(i,j-1,2+d));
			//i+1,j
			cdelu(3,d) = 0.5/m->gvol(i+1,j)*( (u->get(i+1,j)+u->get(i+2,j))*m->gdel(i+1,j,d) + (u->get(i+1,j)+u->get(i+1,j+1))*m->gdel(i+1,j,2+d)
				- (u->get(i+1,j)+u->get(i,j))*m->gdel(i,j,d) - (u->get(i+1,j)+u->get(i+1,j-1))*m->gdel(i+1,j-1,2+d));
			//i,j+1
			cdelu(4,d) = 0.5/m->gvol(i,j+1)*( (u->get(i,j+1)+u->get(i+1,j+1))*m->gdel(i,j+1,d) + (u->get(i,j+1)+(j<m->gjmx()-1 ? u->get(i,j+2):sg4))*m->gdel(i,j+1,2+d)
				- (u->get(i,j+1)+u->get(i-1,j+1))*m->gdel(i-1,j+1,d) - (u->get(i,j+1)+u->get(i,j))*m->gdel(i,j,2+d));
		}
		
		// now calculate contributions from the 4 faces
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(3,0))*m->gdel(i,j,0)+(cdelu(2,1)+cdelu(3,1))*m->gdel(i,j,1) 
			- ((cdelu(2,0)+cdelu(3,0))*svect[0](i,j)+(cdelu(2,1)+cdelu(3,1))*svect[1](i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1)))
			+ (u->get(i+1,j)-u->get(i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1))/dels[0](i,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(1,0))*m->gdel(i-1,j,0)+(cdelu(2,1)+cdelu(1,1))*m->gdel(i-1,j,1) 
			- ((cdelu(2,0)+cdelu(1,0))*svect[0](i-1,j)+(cdelu(2,1)+cdelu(1,1))*svect[1](i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1)))
			+ (u->get(i,j)-u->get(i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1))/dels[0](i-1,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(0,0))*m->gdel(i,j-1,2)+(cdelu(2,1)+cdelu(0,1))*m->gdel(i,j-1,3) 
			- ((cdelu(2,0)+cdelu(0,0))*svect[2](i,j-1)+(cdelu(2,1)+cdelu(0,1))*svect[3](i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3)))
			+ (u->get(i,j)-u->get(i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3))/dels[1](i,j-1);
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(4,0))*m->gdel(i,j,2)+(cdelu(2,1)+cdelu(4,1))*m->gdel(i,j,3) 
			- ((cdelu(2,0)+cdelu(4,0))*svect[2](i,j)+(cdelu(2,1)+cdelu(4,1))*svect[3](i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3)))
			+ (u->get(i,j+1)-u->get(i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3))/dels[1](i,j);
	}

	// boundary 4
	j = 1;
	for(int i = 2; i <= m->gjmx()-2; i++)
	{
		if(bcflag[3] >= 1)
			secondghost = u->get(i,j-1);
		else
			secondghost = 2*bvalue[3] - u->get(i,j+1);
		arvec[0] = -(m->gy(i+1,j-1) - m->gy(i,j-1));
		arvec[1] = m->gx(i+1,j-1) - m->gx(i,j-1);
		for(int d = 0; d < 2; d++)
		{
			// i,j-1
			cdelu(0,d) = 0.5/m->gvol(i,j-1)*( (u->get(i,j-1)+u->get(i+1,j-1))*m->gdel(i,j-1,d) + (u->get(i,j-1)+u->get(i,j))*m->gdel(i,j-1,2+d)
				- (u->get(i,j-1)+u->get(i-1,j-1))*m->gdel(i-1,j-1,d) - (u->get(i,j-1)+secondghost)*arvec[d]);
			//i-1,j
			cdelu(1,d) = 0.5/m->gvol(i-1,j)*( (u->get(i-1,j)+u->get(i,j))*m->gdel(i-1,j,d) + (u->get(i-1,j)+u->get(i-1,j+1))*m->gdel(i-1,j,2+d)
				- (u->get(i-1,j)+u->get(i-2,j))*m->gdel(i-2,j,d) - (u->get(i-1,j)+u->get(i-1,j-1))*m->gdel(i-1,j-1,2+d));
			//i,j
			cdelu(2,d) = 0.5/m->gvol(i,j)*( (u->get(i,j)+u->get(i+1,j))*m->gdel(i,j,d) + (u->get(i,j)+u->get(i,j+1))*m->gdel(i,j,2+d)
				- (u->get(i,j)+u->get(i-1,j))*m->gdel(i-1,j,d) - (u->get(i,j)+u->get(i,j-1))*m->gdel(i,j-1,2+d));
			//i+1,j
			cdelu(3,d) = 0.5/m->gvol(i+1,j)*( (u->get(i+1,j)+u->get(i+2,j))*m->gdel(i+1,j,d) + (u->get(i+1,j)+u->get(i+1,j+1))*m->gdel(i+1,j,2+d)
				- (u->get(i+1,j)+u->get(i,j))*m->gdel(i,j,d) - (u->get(i+1,j)+u->get(i+1,j-1))*m->gdel(i+1,j-1,2+d));
			//i,j+1
			cdelu(4,d) = 0.5/m->gvol(i,j+1)*( (u->get(i,j+1)+u->get(i+1,j+1))*m->gdel(i,j+1,d) + (u->get(i,j+1)+u->get(i,j+2))*m->gdel(i,j+1,2+d)
				- (u->get(i,j+1)+u->get(i-1,j+1))*m->gdel(i-1,j+1,d) - (u->get(i,j+1)+u->get(i,j))*m->gdel(i,j,2+d));
		}
		
		// now calculate contributions from the 4 faces
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(3,0))*m->gdel(i,j,0)+(cdelu(2,1)+cdelu(3,1))*m->gdel(i,j,1) 
			- ((cdelu(2,0)+cdelu(3,0))*svect[0](i,j)+(cdelu(2,1)+cdelu(3,1))*svect[1](i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1)))
			+ (u->get(i+1,j)-u->get(i,j))*(svect[0](i,j)*m->gdel(i,j,0)+svect[1](i,j)*m->gdel(i,j,1))/dels[0](i,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(1,0))*m->gdel(i-1,j,0)+(cdelu(2,1)+cdelu(1,1))*m->gdel(i-1,j,1) 
			- ((cdelu(2,0)+cdelu(1,0))*svect[0](i-1,j)+(cdelu(2,1)+cdelu(1,1))*svect[1](i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1)))
			+ (u->get(i,j)-u->get(i-1,j))*(svect[0](i-1,j)*m->gdel(i-1,j,0)+svect[1](i-1,j)*m->gdel(i-1,j,1))/dels[0](i-1,j);
		(*res)(i,j) -= 0.5*((cdelu(2,0)+cdelu(0,0))*m->gdel(i,j-1,2)+(cdelu(2,1)+cdelu(0,1))*m->gdel(i,j-1,3) 
			- ((cdelu(2,0)+cdelu(0,0))*svect[2](i,j-1)+(cdelu(2,1)+cdelu(0,1))*svect[3](i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3)))
			+ (u->get(i,j)-u->get(i,j-1))*(svect[2](i,j-1)*m->gdel(i,j-1,2)+svect[3](i,j-1)*m->gdel(i,j-1,3))/dels[1](i,j-1);
		(*res)(i,j) += 0.5*((cdelu(2,0)+cdelu(4,0))*m->gdel(i,j,2)+(cdelu(2,1)+cdelu(4,1))*m->gdel(i,j,3) 
			- ((cdelu(2,0)+cdelu(4,0))*svect[2](i,j)+(cdelu(2,1)+cdelu(4,1))*svect[3](i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3)))
			+ (u->get(i,j+1)-u->get(i,j))*(svect[2](i,j)*m->gdel(i,j,2)+svect[3](i,j)*m->gdel(i,j,3))/dels[1](i,j);
	}
}

}
