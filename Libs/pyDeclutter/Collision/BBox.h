#ifndef BBOX_H
#define BBOX_H

#include "Config.h"

namespace Collision2D
{
template <typename T,int DIM=3>
struct BBox {
  static const int dim=DIM;
  typedef Eigen::Matrix<T,DIM,1> PT;
  typedef Eigen::Matrix<T,2,1> PT2;
  EIGEN_DEVICE_FUNC BBox();
  EIGEN_DEVICE_FUNC BBox(const PT& p);
  EIGEN_DEVICE_FUNC BBox(const PT& minC,const PT& maxC);
  EIGEN_DEVICE_FUNC static BBox createMM(const PT& minC,const PT& maxC);
  EIGEN_DEVICE_FUNC static BBox createME(const PT& minC,const PT& extent);
  EIGEN_DEVICE_FUNC static BBox createCE(const PT& center,const PT& extent);
  EIGEN_DEVICE_FUNC BBox getIntersect(const BBox& other) const;
  EIGEN_DEVICE_FUNC BBox getUnion(const BBox& other) const;
  EIGEN_DEVICE_FUNC BBox getUnion(const PT& point) const;
  EIGEN_DEVICE_FUNC BBox getUnion(const PT& ctr,const T& rad) const;
  EIGEN_DEVICE_FUNC void setIntersect(const BBox& other);
  EIGEN_DEVICE_FUNC void setUnion(const BBox& other);
  EIGEN_DEVICE_FUNC void setUnion(const PT& point);
  EIGEN_DEVICE_FUNC void setUnion(const PT& ctr,const T& rad);
  EIGEN_DEVICE_FUNC void setPoints(const PT& a,const PT& b,const PT& c);
  EIGEN_DEVICE_FUNC PT minCorner() const;
  EIGEN_DEVICE_FUNC PT maxCorner() const;
  EIGEN_DEVICE_FUNC void enlargedEps(T eps);
  EIGEN_DEVICE_FUNC BBox enlargeEps(T eps) const;
  EIGEN_DEVICE_FUNC void enlarged(T len,const sizeType d=DIM);
  EIGEN_DEVICE_FUNC BBox enlarge(T len,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC PT lerp(const PT& frac) const;
  EIGEN_DEVICE_FUNC bool empty() const;
  template <int DIM2>
  EIGEN_DEVICE_FUNC bool containDim(const PT& point) const;
  EIGEN_DEVICE_FUNC bool contain(const BBox& other,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC bool contain(const PT& point,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC bool contain(const PT& point,const T& rad,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC void reset();
  EIGEN_DEVICE_FUNC PT getExtent() const;
  EIGEN_DEVICE_FUNC T distTo(const BBox& other,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC T distTo(const PT& pt,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC T distToSqr(const PT& pt,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC PT closestTo(const PT& pt,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC bool intersect(const PT& p,const PT& q,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC bool intersect(const PT& p,const PT& q,T& s,T& t,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC bool intersect(const BBox& other,const sizeType& d=DIM) const;
  EIGEN_DEVICE_FUNC PT2 project(const PT& a,const sizeType d=DIM) const;
  EIGEN_DEVICE_FUNC T perimeter(const sizeType d=DIM) const;
  PT _minC;
  PT _maxC;
};
}

#endif
