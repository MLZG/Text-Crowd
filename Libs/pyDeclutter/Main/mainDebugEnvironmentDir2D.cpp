#include <Collision/Environment2D.h>

using namespace Collision2D;

int main()
{
  Polygon2D<scalarD>::Vss vss;
  Environment2D<scalarD>::Pss pss;
  vss.push_back(Polygon2D<scalarD>::Vec2T(-1,-1));
  vss.push_back(Polygon2D<scalarD>::Vec2T( 1,-1));
  vss.push_back(Polygon2D<scalarD>::Vec2T( 1, 1));
  vss.push_back(Polygon2D<scalarD>::Vec2T(-1, 1));
  for(sizeType i=0; i<10; i++)
    for(sizeType j=0; j<10; j++) {
      Polygon2D<scalarD> p(vss,100);
      p=p.transform(Polygon2D<scalarD>::Vec2T(i*3,j*3),rand()*M_PI*2/RAND_MAX);
      pss.push_back(p);
    }
  Environment2D<scalarD> env(pss);
  env.writeVTK("env.vtk");
  env.writeDirVTK("dir.vtk",pss,Polygon2D<scalarD>::Vec2T(100,100));
  return 0;
}
