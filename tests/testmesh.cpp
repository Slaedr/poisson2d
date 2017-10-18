#include "structmesh2d.hpp"
#include "aoutput_struct.hpp"

using namespace amat;
using namespace acfd;
using namespace std;

int main()
{
	Structmesh2d m;
	m.readmesh("testmesh");

	Matrix<double>* dum;
	Matrix<double>** dum2;
	string* sdum;
	string title = "project2_mesh";
	Structdata2d d(&m, 0, dum, sdum, 0, dum2, sdum, title);
	d.writevtk("testmesh.vtk");

	cout << endl;
	return 0;
}
