#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include "Collision/Polygon2D.h"
#include "Collision/Environment2D.h"
#include <sstream>

typedef scalarD T;
namespace py=pybind11;
using namespace Collision2D;
#define STRINGIFY_OMP(T) #T
PYBIND11_DECLARE_HOLDER_TYPE(T,std::shared_ptr<T>);
#define PYTHON_FUNC(C,NAME) .def(STRINGIFY_OMP(NAME),&C::NAME)
#define PYTHON_MEMBER_READ(C,NAME) .def_readonly(STRINGIFY_OMP(NAME),&C::NAME)
#define PYTHON_MEMBER_READWRITE(C,NAME) .def_readwrite(STRINGIFY_OMP(NAME),&C::NAME,&C::NAME)
#define PYTHON_FIELD_READWRITE(C,NAME) .def_readwrite(STRINGIFY_OMP(NAME),&C::NAME)
#define EIGEN_OP_VEC(T) \
.def("__add__",[](const T& v,const T& w){return T(v+w);},py::is_operator()) \
.def("__sub__",[](const T& v,const T& w){return T(v-w);},py::is_operator()) \
.def("fromList",[](T& v,py::list lst){sizeType id=0;for(auto item:lst)v[id++]=item.cast<T::Scalar>();}) \
.def("toList",[](const T& v){py::list li;for(sizeType i=0;i<v.size();i++)li.append(v[i]);return li;})  \
.def("setZero",[](T& v){v.setZero();})  \
.def("setOnes",[](T& v){v.setOnes();})  \
.def("setConstant",[](T& v,T::Scalar val){v.setConstant(val);})  \
.def("dot",[](const T& v,const T& w){return v.dot(w);}) \
.def("__getitem__",[](const T& v,sizeType i){return v[i];}) \
.def("__setitem__",[](T& v,sizeType i,T::Scalar val){v[i]=val;})   \
.def("__repr__",[](const T& v){std::ostringstream os;os<<"["<<v.transpose()<<"]^T";return os.str();});
PYBIND11_MODULE(pyDeclutter,m)
{
  typedef typename Environment2D<T>::Vec2T Vec2T;
  typedef typename Environment2D<T>::Vec3T Vec3T;
  typedef typename Environment2D<T>::Vssi Vssi;
  typedef typename Environment2D<T>::Vss Vss;
  typedef typename Environment2D<T>::Lss Lss;
  typedef typename Environment2D<T>::Pss Pss;
  typedef BBox<T,2> BBox2D;
  //-----------------------------------------------------------------basic types
  //Vec2
  py::class_<Vec2T>(m,"Vec2")
  .def(py::init<>())
  .def(py::init<T,T>())
  EIGEN_OP_VEC(Vec2T)
  //Vec3
  py::class_<Vec3T>(m,"Vec3")
  .def(py::init<>())
  .def(py::init<T,T,T>())
  EIGEN_OP_VEC(Vec3T)
  //Vec2i
  py::class_<Vec2i>(m,"Vec2i")
  .def(py::init<>())
  .def(py::init<sizeType,sizeType>())
  EIGEN_OP_VEC(Vec2i)
  //-----------------------------------------------------------------BBox<T,2>
  py::class_<BBox2D,std::shared_ptr<BBox2D>>(m,std::string("BBox2D").c_str())
                                          PYTHON_MEMBER_READ(BBox2D,_minC)
                                          PYTHON_MEMBER_READ(BBox2D,_maxC);
  //-----------------------------------------------------------------LineSeg2D
  py::class_<LineSeg2D<T>,std::shared_ptr<LineSeg2D<T>>>(m,std::string("LineSeg2D").c_str())
  .def(py::init<>())
  .def(py::init<Vec2T,Vec2T>())
  PYTHON_FUNC(LineSeg2D<T>,transform)
  PYTHON_FUNC(LineSeg2D<T>,intersectsLine)
  PYTHON_FUNC(LineSeg2D<T>,intersectsCircle)
  PYTHON_FUNC(LineSeg2D<T>,interp)
  PYTHON_FUNC(LineSeg2D<T>,computeBB)
  PYTHON_FUNC(LineSeg2D<T>,negate)
  PYTHON_FUNC(LineSeg2D<T>,setRandom)
  PYTHON_FUNC(LineSeg2D<T>,initialize)
  PYTHON_FUNC(LineSeg2D<T>,a3D)
  PYTHON_FUNC(LineSeg2D<T>,b3D)
  PYTHON_MEMBER_READ(LineSeg2D<T>,_a)
  PYTHON_MEMBER_READ(LineSeg2D<T>,_b)
  PYTHON_MEMBER_READ(LineSeg2D<T>,_dir)
  PYTHON_MEMBER_READ(LineSeg2D<T>,_len)
  PYTHON_MEMBER_READ(LineSeg2D<T>,_invLen);
  //-----------------------------------------------------------------DirectionalCostCase2D
  py::class_<DirectionalCostCase2D<T>,std::shared_ptr<DirectionalCostCase2D<T>>>(m,std::string("DirectionalCostCase2D").c_str())
  .def(py::init<T,T>())
  .def(py::init<T,T,LineSeg2D<T>,Vec2T>())
  .def(py::init<T,T,LineSeg2D<T>,Vec2T,Vec2T>())
  PYTHON_FUNC(DirectionalCostCase2D<T>,getVss)
  PYTHON_FUNC(DirectionalCostCase2D<T>,cost)
  PYTHON_FUNC(DirectionalCostCase2D<T>,color)
  PYTHON_MEMBER_READ(DirectionalCostCase2D<T>,_alpha0)
  PYTHON_MEMBER_READ(DirectionalCostCase2D<T>,_alpha1)
  PYTHON_MEMBER_READ(DirectionalCostCase2D<T>,_a)
  PYTHON_MEMBER_READ(DirectionalCostCase2D<T>,_b)
  PYTHON_MEMBER_READ(DirectionalCostCase2D<T>,_c)
  .def("getType",[](const DirectionalCostCase2D<T>& c) {
    return (int)c._type;
  });
  //-----------------------------------------------------------------DirectionalCost2D
  py::class_<DirectionalCost2D<T>,std::shared_ptr<DirectionalCost2D<T>>>(m,std::string("DirectionalCost2D").c_str())
  PYTHON_FUNC(DirectionalCost2D<T>,writeVTK)
  PYTHON_MEMBER_READ(DirectionalCost2D<T>,_case);
  //-----------------------------------------------------------------Python2D
  py::class_<Polygon2D<T>,std::shared_ptr<Polygon2D<T>>>(m,std::string("Polygon2D").c_str())
  .def(py::init<>())
  .def(py::init<Vss,T>())
  .def(py::init<sizeType,T,T>())
  PYTHON_FUNC(Polygon2D<T>,writeVTK)
  PYTHON_FUNC(Polygon2D<T>,writeDistanceVTK)
  PYTHON_FUNC(Polygon2D<T>,transform)
  PYTHON_FUNC(Polygon2D<T>,computeBB)
  PYTHON_FUNC(Polygon2D<T>,edge)
  PYTHON_FUNC(Polygon2D<T>,vertex)
  PYTHON_FUNC(Polygon2D<T>,vertex3D)
  PYTHON_FUNC(Polygon2D<T>,edgeLen)
  PYTHON_FUNC(Polygon2D<T>,outNormalLen)
  PYTHON_FUNC(Polygon2D<T>,contains)
  PYTHON_FUNC(Polygon2D<T>,initialize)
  PYTHON_FUNC(Polygon2D<T>,nrV)
  PYTHON_FUNC(Polygon2D<T>,containsEdgeDirichlet)
  PYTHON_FUNC(Polygon2D<T>,containsVertexDirichlet)
  PYTHON_FUNC(Polygon2D<T>,buildDirectionalCost)
  PYTHON_FUNC(Polygon2D<T>,buildDirectionalCostBatched)
  PYTHON_FUNC(Polygon2D<T>,distance)
  PYTHON_FUNC(Polygon2D<T>,distanceBatched);
  //-----------------------------------------------------------------Environment2D
  py::class_<Environment2D<T>,std::shared_ptr<Environment2D<T>>>(m,std::string("Environment2D").c_str())
  .def(py::init<Pss>())
  .def(py::init<Lss>())
  .def(py::init<Vssi>())
  PYTHON_FUNC(Environment2D<T>,writeVTK)
  PYTHON_FUNC(Environment2D<T>,ray)
  PYTHON_FUNC(Environment2D<T>,rayE)
  PYTHON_FUNC(Environment2D<T>,rayA)
  PYTHON_FUNC(Environment2D<T>,range)
  PYTHON_FUNC(Environment2D<T>,rangeBatched)
  PYTHON_FUNC(Environment2D<T>,sensorLine)
  PYTHON_FUNC(Environment2D<T>,sensorData)
  PYTHON_FUNC(Environment2D<T>,writeSensorVTK)
  PYTHON_FUNC(Environment2D<T>,dirPolygonBatched)
  PYTHON_FUNC(Environment2D<T>,writeDirVTK)
  PYTHON_FUNC(Environment2D<T>,circle)
  PYTHON_FUNC(Environment2D<T>,validSamples)
  PYTHON_FUNC(Environment2D<T>,writeValidSamplesVTK)
  PYTHON_FUNC(Environment2D<T>,clearAgent)
  PYTHON_FUNC(Environment2D<T>,setAgent);
}
