/** Provides a class to manage output of mesh data to VTK-type files.
*/
#ifndef AOUTPUT_STRUCT_H
#define AOUTPUT_STRUCT_H 1

#include <string>

#include "aarray2d.hpp"
#include "structmesh2d.hpp"

namespace poisson {

///	Class for managing output of analysis data for simulations on structured 2D grids.
/**	We assume 1-based arrays for all array-quantities.
* Vectors are taken as an array (for each different vector quantity) of arrays (for each component of vector) of matrices over i,j.
*/
class Structdata2d
{
	int ndim;			///< Dimension of the problem
	int nscalars;		///< Number of scalar quantities
	int nvectors;		///< Number of vector quantities
	Structmesh2d* m;
	amat::Array2d<double>* scalars;
	amat::Array2d<double>** vectors;
	std::string* scalarnames;
	std::string* vectornames;
	std::string title;
public:
	Structdata2d(Structmesh2d* mesh, 
			int n_scalars, amat::Array2d<double>* _scalars, std::string* scalar_names, 
			int n_vectors, amat::Array2d<double>** _vectors, std::string* vector_names, 
			std::string title);

	void writevtk(std::string fname);
};

}
#endif
