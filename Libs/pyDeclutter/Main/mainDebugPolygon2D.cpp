#include <Collision/Polygon2D.h>

using namespace Collision2D;

int main()
{
  Polygon2D<scalarD> poly(10,0.5,10.0);
  poly=poly.transform(Polygon2D<scalarD>::Vec2T::Random()*0.1,M_PI/4);
  Polygon2D<scalarD>::Vss ptss(1000);
  std::vector<LineSeg2D<scalarD>> lss(1000);
  for(sizeType i=0; i<(sizeType)lss.size(); i++) {
    lss[i].setRandom(0.5,10.0);
    ptss[i].setRandom();
    ptss[i]*=0.5;
  }
  poly.writeDistanceVTK("polyD.vtk",ptss);
  poly.writeVTK("poly.vtk",true);

  recreate("dirCost");
  std::vector<DirectionalCost2D<scalarD>> dss=poly.buildDirectionalCostBatched(lss);
  for(sizeType i=0; i<(sizeType)dss.size(); i++) {
    dss[i].writeVTK("dirCost/dirCost"+std::to_string(i)+".vtk",lss[i],0.001f,true);
    dss[i].writeVTK("dirCost/dirCostH"+std::to_string(i)+".vtk",lss[i],0.001f,false);
  }
  return 0;
}
