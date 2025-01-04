#include "VTKWriter.h"

namespace Collision2D
{
//VTK compatible atomic IO
class Endianness
{
public:
  static bool isLittleEndian() {
    union u {
      unsigned long l;
      unsigned char c[sizeof(unsigned long)];
    };
    u dummy;
    dummy.l = 1;
    return dummy.c[0] == 1;
  }
  static void swap2Bytes(unsigned char* &ptr) {
    unsigned char tmp;
    tmp = ptr[0];
    ptr[0] = ptr[1];
    ptr[1] = tmp;
  }
  static void swap4Bytes(unsigned char* &ptr) {
    unsigned char tmp;
    tmp = ptr[0];
    ptr[0] = ptr[3];
    ptr[3] = tmp;
    tmp = ptr[1];
    ptr[1] = ptr[2];
    ptr[2] = tmp;
  }
  static void swap8Bytes(unsigned char* &ptr) {
    unsigned char tmp;
    tmp = ptr[0];
    ptr[0] = ptr[7];
    ptr[7] = tmp;
    tmp = ptr[1];
    ptr[1] = ptr[6];
    ptr[6] = tmp;
    tmp = ptr[2];
    ptr[2] = ptr[5];
    ptr[5] = tmp;
    tmp = ptr[3];
    ptr[3] = ptr[4];
    ptr[4] = tmp;
  }
};
template<class T2>
void byteSwap(T2& val)
{
  sizeType n=sizeof(T2);
  unsigned char *p=(unsigned char*)&val;
  switch( n ) {
  case 1:
    return;
  case 2:
    Endianness::swap2Bytes(p);
    break;
  case 4:
    Endianness::swap4Bytes(p);
    break;
  case 8:
    Endianness::swap8Bytes(p);
    break;
  default:
    break;
  }
}
template <typename T2>
void vtkWriteTpl(std::ostream& oss,T2 val)
{
  if(typeid(val) == typeid(scalarD) || typeid(val) == typeid(double)) {
    //double
    double valD=double(val);
    if(Endianness::isLittleEndian())
      byteSwap(valD);
    oss.write((const char*)&valD,sizeof(double));
  } else if(typeid(val) == typeid(float)) {
    //float
    float valF=float(val);
    if(Endianness::isLittleEndian())
      byteSwap(valF);
    oss.write((const char*)&valF,sizeof(float));
  } else {
    //integer
    int valI=int(val);
    if(Endianness::isLittleEndian())
      byteSwap(valI);
    oss.write((const char*)&valI,sizeof(int));
  }
}
void vtkWrite(std::ostream& oss,scalarF val)
{
  vtkWriteTpl(oss,val);
}
void vtkWrite(std::ostream& oss,scalarD val)
{
  vtkWriteTpl(oss,val);
}
void vtkWrite(std::ostream& oss,int val)
{
  vtkWriteTpl(oss,val);
}
void vtkWrite(std::ostream& oss,char val)
{
  oss.write((const char*)&val,1);
}
void vtkWrite(std::ostream& oss,unsigned char val)
{
  oss.write((const char*)&val,1);
}
}
