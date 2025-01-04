#include <Collision/Environment2D.h>

using namespace Collision2D;

int main()
{
  std::vector<scalarD> rad;
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
  vss.clear();
  Environment2D<scalarD> env(pss);
  for(sizeType i=0,step=10; i<100; i+=step)
    for(sizeType j=0; j<100; j+=step) {
      vss.push_back(Environment2D<scalarD>::Vec2T(i,j)/3.0);
      rad.push_back(5/3.0);
    }
  env.writeValidSamplesVTK("sample.vtk",vss,rad);
  env.writeVTK("env.vtk");
  return 0;
}
