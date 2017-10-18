#ifndef GRADIENTSCHEMES_H
#define GRADIENTSCHEMES_H 1

#include "aarray2d.hpp"
#include "structmesh2d.hpp"

#include <vector>

namespace poisson {

///	Base class for gradient reconstruction schemes.
class GradientScheme
{
protected:
	Structmesh2d* m;		///< Associated mesh
	amat::Array2d<double>* u;		///< The unknown with which to compute the graient flux
	amat::Array2d<double>* res;	///< The residual (for AF type solution) containing fluxes at each grid cell
	
	/// LHS of AF scheme corresponding to a particular (compact) gradient model.
	/**
	 * - a(i,j,0) corresponds to coeff of u(i-1,j)
	 * - a(i,j,1) corresponds to coeff of u(i,j-1)
	 * - a(i,j,2) corresponds to coeff of u(i,j)
	 * - a(i,j,3) corresponds to coeff of u(i,j+1)
	 * - a(i,j,4) corresponds to coeff of u(i+1,j),
	 * in the row of Ax=b corresponding to the (i,j) cell.
	 */
	amat::Array2d<double>* a;

	///	Contains a flag for each of the 4 boundaries, indicating the type of boundary 
	/// (0 is Dirichlet, 1 is Neumann).
	/**
	 * Note that the vector has 4 elements:
	 * 0: boundary i = m->gimx()
	 * 1: boundary j = m->gjmx()
	 * 2: boundary i = 1
	 * 3: boundary j = 1
	 */
	std::vector<int> bcflag;	
	
	/// Contains boundary values corresponding to each of the 4 boundaries.
	std::vector<double> bvalue;	
	
	/// Volumes of the thin-layer CVs for the gradient. 
	/** Two components: first component for +i face and the other for +j face. 
	 * Arranged like Structmesh2d::del.
	 */
	std::vector<amat::Array2d<double>> dualvol;

public:
	void setup(Structmesh2d* mesh, 
			amat::Array2d<double>* unknown, amat::Array2d<double>* residual, amat::Array2d<double>* lhs, 
			std::vector<int> bcflags, std::vector<double> bvalues);

	virtual ~GradientScheme()
	{ }

	void compute_CV_volumes();		///< Computes dualvol. To be precomputed just once.

	virtual void compute_s();		///< Required for Normal tangent gradient decomposition scheme.
	
	/**	NOTE: compute_fluxes() *increments* the residual res. 
	 * If res contains anything non-zero, contribution by gradients is *added* to it; it is not replaced.
	 */
	virtual void compute_fluxes() = 0;
	
	/// Computes LHS arrays corresponding to a particular gradient scheme.
	/**	Make sure to execcute [calculate_CV_volumes](@ref calculate_CV_volumes) before calling this function.
	*/
	virtual void compute_lhs() = 0;
};

//----------------- end of base class GradientScheme -----------------------------//


///	Thin layer gradient reconstruction scheme. 
/** We consider grad u at a face to be influenced by only the change in normal component of u at the face.
*/
class ThinLayerGradient : public GradientScheme
{
public:
	void compute_fluxes();
	void compute_lhs();
};

///	Computes LHS and residual for diffusion using Normal-tangent gradient decomposition model
/**	Applied properly, the scheme is non-compact and has 13 terms in the LHS for each cell. 
 * However, we implement only the 5 compact terms for the LHS. 
 * But we treat the residual fully. This requires specification of more than one layer of ghost cells. 
 * This is taken care of without introducing another layer;
 * rather, we compute a `ghost state' on-the-fly using the BCs.
 */
class NormTanGradient : public GradientScheme
{
	/** Stores a unit vector in the direction of the line joining the two cells across each face.
	 *  Organized like Structmesh2d::del.
	 */
	std::vector<amat::Array2d<double>> svect;

	/** Magnitude of the corresponding vectors in [svect](@ref svect).
	 * dels[0] refers to magnitude of the vector for +i face 
	 * dels[1] refers to magnitude of the vector for +j face.
	 */
	std::vector<amat::Array2d<double>> dels;

public:
	void compute_s();
	void compute_lhs();
	void compute_fluxes();
};

}

#endif
