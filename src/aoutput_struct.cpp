#include <iostream>
#include <fstream>
#include "aoutput_struct.hpp"

namespace poisson {

Structdata2d::Structdata2d(Structmesh2d* mesh, int n_scalars, amat::Array2d<double>* _scalars, std::string* scalar_names, int n_vectors, amat::Array2d<double>** _vectors, std::string* vector_names, std::string title)
{
	ndim = 2;
	m = mesh;
	nscalars = n_scalars;
	nvectors = n_vectors;
	scalars = _scalars;
	vectors = _vectors;
	scalarnames = scalar_names;
	vectornames = vector_names;
}

void Structdata2d::writevtk(std::string fname)
{
	std::cout << "Structdata2d: writevtk(): Writing data to file " << fname << std::endl;
	std::ofstream fout(fname);
	fout << "# vtk DataFile Version 2.0\n";
	fout << title << '\n';
	std::cout << title << '\n';
	fout << "ASCII\n";
	fout << "DATASET STRUCTURED_GRID\n";
	fout << "DIMENSIONS " << m->gimx() << " " << m->gjmx() << " 1\n";
	fout << "POINTS " << m->gimx()*m->gjmx() << " float\n";
	for(int j = 1; j <= m->gjmx(); j++)
		for(int i = 1; i <= m->gimx(); i++)
			fout << m->gx(i,j) << " " << m->gy(i,j) << " " << 0.0 << '\n';
	// Now output data
	if(nscalars > 0 || nvectors > 0)
	{
		fout << "POINT_DATA " << m->gimx()*m->gjmx() << '\n';
		for(int isca = 0; isca < nscalars; isca++)
		{
			fout << "SCALARS " << scalarnames[isca] << " float 1\n";
			fout << "LOOKUP_TABLE default\n";
			for(int j = 1; j <= m->gjmx(); j++)
				for(int i = 1; i <= m->gimx(); i++)
					fout << scalars[isca].get(i,j) << '\n';
		}
		for(int iv = 0; iv < nvectors; iv++)
		{
			fout << "VECTORS " << vectornames[iv] << " float\n";
			for(int j = 1; j <= m->gjmx(); j++)
				for(int i = 1; i <= m->gimx(); i++)
				{
					for(int idim = 0; idim < ndim; idim++)
						fout << vectors[iv][idim].get(i,j) << " ";
					for(int idim = 3-ndim; idim > 0; idim--)
						fout << "0 ";
					fout << '\n';
				}
		}
	}
	//Done.
}

}
