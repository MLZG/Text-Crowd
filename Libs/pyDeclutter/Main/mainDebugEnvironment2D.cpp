#include <Collision/Environment2D.h>

using namespace Collision2D;

int main()
{
  std::vector<Vec2i> iss;
  for(sizeType x=0; x<100; x++)
    for(sizeType y=0; y<100; y++)
      if(x>=20 && x<40 && y>=20 && y<80) {}
      else if(x>=60 && x<80 && y>=20 && y<80) {}
      else if(y>=40 && y<60 && x>=20 && x<80) {}
      else iss.push_back(Vec2i(x,y));
  Environment2D<scalarD> env(iss);
  Environment2D<scalarD>::Vss vss;
  vss.push_back(Environment2D<scalarD>::Vec2T(10,20));
  vss.push_back(Environment2D<scalarD>::Vec2T(10,30));
  vss.push_back(Environment2D<scalarD>::Vec2T(10,40));
  vss.push_back(Environment2D<scalarD>::Vec2T(20,10));
  vss.push_back(Environment2D<scalarD>::Vec2T(30,10));
  vss.push_back(Environment2D<scalarD>::Vec2T(40,10));
  env.setAgent(vss,5);
  env.writeVTK("env.vtk");
  env.writeSensorVTK("sensor1.vtk",Environment2D<scalarD>::Vec2T(10,10),100,64);
  env.writeSensorVTK("sensor2.vtk",Environment2D<scalarD>::Vec2T(50,25),100,64);
  env.writeSensorVTK("sensor3.vtk",Environment2D<scalarD>::Vec2T(10,20),100,64);
  env.writeSensorVTK("sensor4.vtk",Environment2D<scalarD>::Vec2T(10,20),100,64,0);
  return 0;
}
