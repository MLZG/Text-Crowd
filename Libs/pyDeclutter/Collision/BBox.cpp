#include "BBox.h"

namespace Collision2D
{
//BBox
template <typename T,int DIM>
EIGEN_DEVICE_FUNC BBox<T,DIM>::BBox() {
  reset();
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC BBox<T,DIM>::BBox(const PT& p):_minC(p),_maxC(p) {}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC BBox<T,DIM>::BBox(const PT& minC,const PT& maxC):_minC(minC),_maxC(maxC) {}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::createMM(const PT& minC,const PT& maxC) {
  return BBox(minC,maxC);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::createME(const PT& minC,const PT& extent) {
  return BBox(minC,minC+extent);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::createCE(const PT& center,const PT& extent) {
  return BBox(center-extent,center+extent);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::getIntersect(const BBox& other) const {
  return createMM(compMax(_minC,other._minC),compMin(_maxC,other._maxC));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::getUnion(const BBox& other) const {
  return createMM(compMin(_minC,other._minC),compMax(_maxC,other._maxC));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::getUnion(const PT& point) const {
  return createMM(compMin(_minC,point),compMax(_maxC,point));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::getUnion(const PT& ctr,const T& rad) const {
  return createMM(compMin(_minC,ctr-PT::Constant(rad)),compMax(_maxC,ctr+PT::Constant(rad)));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::setIntersect(const BBox& other) {
  _minC=compMax(_minC,other._minC);
  _maxC=compMin(_maxC,other._maxC);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::setUnion(const BBox& other) {
  _minC=compMin<PT>(_minC,other._minC);
  _maxC=compMax<PT>(_maxC,other._maxC);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::setUnion(const PT& point) {
  _minC=compMin(_minC,point);
  _maxC=compMax(_maxC,point);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::setUnion(const PT& ctr,const T& rad) {
  _minC=compMin(_minC,ctr-PT::Constant(rad));
  _maxC=compMax(_maxC,ctr+PT::Constant(rad));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::setPoints(const PT& a,const PT& b,const PT& c) {
  _minC=compMin(compMin(a,b),c);
  _maxC=compMax(compMax(a,b),c);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT BBox<T,DIM>::minCorner() const {
  return _minC;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT BBox<T,DIM>::maxCorner() const {
  return _maxC;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::enlargedEps(T eps) {
  PT d=(_maxC-_minC)*(eps*0.5f);
  _minC-=d;
  _maxC+=d;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::enlargeEps(T eps) const {
  PT d=(_maxC-_minC)*(eps*0.5f);
  return createMM(_minC-d,_maxC+d);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::enlarged(T len,const sizeType d) {
  for(sizeType i=0; i<d; i++) {
    _minC[i]-=len;
    _maxC[i]+=len;
  }
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::BBox BBox<T,DIM>::enlarge(T len,const sizeType d) const {
  BBox ret=createMM(_minC,_maxC);
  ret.enlarged(len,d);
  return ret;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT BBox<T,DIM>::lerp(const PT& frac) const {
  return (_maxC.array()*frac.array()-_minC.array()*(frac.array()-1.0f)).matrix();
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::empty() const {
  return !compL(_minC,_maxC);
}
template <typename T,int DIM>
template <int DIM2>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::containDim(const PT& point) const {
  for(int i=0; i<DIM2; i++)
    if(_minC[i] > point[i] || _maxC[i] < point[i])
      return false;
  return true;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::contain(const BBox& other,const sizeType d) const {
  for(int i=0; i<d; i++)
    if(_minC[i] > other._minC[i] || _maxC[i] < other._maxC[i])
      return false;
  return true;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::contain(const PT& point,const sizeType d) const {
  for(int i=0; i<d; i++)
    if(_minC[i] > point[i] || _maxC[i] < point[i])
      return false;
  return true;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::contain(const PT& point,const T& rad,const sizeType d) const {
  for(int i=0; i<d; i++)
    if(_minC[i]+rad > point[i] || _maxC[i]-rad < point[i])
      return false;
  return true;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC void BBox<T,DIM>::reset() {
  _minC=PT::Constant( std::numeric_limits<T>::max());
  _maxC=PT::Constant(-std::numeric_limits<T>::max());
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT BBox<T,DIM>::getExtent() const {
  return _maxC-_minC;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC T BBox<T,DIM>::distTo(const BBox& other,const sizeType d) const {
  PT dist=PT::Zero();
  for(sizeType i=0; i<d; i++) {
    if (other._maxC[i] < _minC[i])
      dist[i] = other._maxC[i] - _minC[i];
    else if (other._minC[i] > _maxC[i])
      dist[i] = other._minC[i] - _maxC[i];
  }
  return dist.norm();
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC T BBox<T,DIM>::distTo(const PT& pt,const sizeType d) const {
  return std::sqrt(distToSqr(pt,d));
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC T BBox<T,DIM>::distToSqr(const PT& pt,const sizeType d) const {
  PT dist=PT::Zero();
  for(sizeType i=0; i<d; i++) {
    if (pt[i] < _minC[i])
      dist[i] = pt[i] - _minC[i];
    else if (pt[i] > _maxC[i])
      dist[i] = pt[i] - _maxC[i];
  }
  return dist.squaredNorm();
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT BBox<T,DIM>::closestTo(const PT& pt,const sizeType d) const {
  PT dist(pt);
  for(sizeType i=0; i<d; i++) {
    if (pt[i] < _minC[i])
      dist[i] = _minC[i];
    else if (pt[i] > _maxC[i])
      dist[i] = _maxC[i];
  }
  return dist;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::intersect(const PT& p,const PT& q,const sizeType d) const {
  T s=0, t=1;
  return intersect(p,q,s,t,d);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::intersect(const PT& p,const PT& q,T& s,T& t,const sizeType d) const {
  const T lo=1-5*std::numeric_limits<T>::epsilon();
  const T hi=1+5*std::numeric_limits<T>::epsilon();

  s=0;
  t=1;
  for(sizeType i=0; i<d; ++i) {
    T D=q[i]-p[i];
    if(p[i]<q[i]) {
      T s0=lo*(_minC[i]-p[i])/D, t0=hi*(_maxC[i]-p[i])/D;
      if(s0>s) s=s0;
      if(t0<t) t=t0;
    } else if(p[i]>q[i]) {
      T s0=lo*(_maxC[i]-p[i])/D, t0=hi*(_minC[i]-p[i])/D;
      if(s0>s) s=s0;
      if(t0<t) t=t0;
    } else {
      if(p[i]<_minC[i] || p[i]>_maxC[i])
        return false;
    }

    if(s>t)
      return false;
  }
  return true;
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC bool BBox<T,DIM>::intersect(const BBox& other,const sizeType& d) const {
  for(sizeType i=0; i<d; i++) {
    if(_maxC[i] < other._minC[i] || other._maxC[i] < _minC[i])
      return false;
  }
  return true;
  //return compLE(_minC,other._maxC) && compLE(other._minC,_maxC);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC typename BBox<T,DIM>::PT2 BBox<T,DIM>::project(const PT& a,const sizeType d) const {
  PT ctr=(_minC+_maxC)*0.5f;
  T ctrD=a.dot(ctr);
  T delta=0.0f;
  ctr=_maxC-ctr;
  for(sizeType i=0; i<d; i++)
    delta+=std::abs(ctr[i]*a[i]);
  return PT2(ctrD-delta,ctrD+delta);
}
template <typename T,int DIM>
EIGEN_DEVICE_FUNC T BBox<T,DIM>::perimeter(const sizeType d) const {
  PT ext=getExtent();
  if(d <= 2)
    return ext.sum()*2.0f;
  else {
    ASSERT(d == 3);
    return (ext[0]*ext[1]+ext[1]*ext[2]+ext[0]*ext[2])*2.0f;
  }
}

template struct BBox<scalarF,2>;
template struct BBox<scalarF,3>;
template struct BBox<scalarD,2>;
template struct BBox<scalarD,3>;
}
