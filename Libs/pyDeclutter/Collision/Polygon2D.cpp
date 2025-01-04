#include "Polygon2D.h"
#include "VTKWriter.h"
#include <omp.h>

namespace Collision2D
{
//LineSeg2D
template <typename T>
LineSeg2D<T>::LineSeg2D() {}
template <typename T>
LineSeg2D<T>::LineSeg2D(const Vec2T& a,const Vec2T& b):_a(a),_b(b)
{
  initialize();
}
template <typename T>
LineSeg2D<T> LineSeg2D<T>::transform(const Vec2T& pos,T theta) const
{
  Mat2T R;
  if(theta!=0) {
    R(0,0)=R(1,1)=std::cos(theta);
    R(1,0)=R(0,1)=std::sin(theta);
    R(0,1)*=-1;
  } else {
    R.setIdentity();
  }

  LineSeg2D ret=*this;
  ret._a=R*_a+pos;
  ret._b=R*_b+pos;
  ret._dir=R*_dir;
  return ret;
}
template <typename T>
bool LineSeg2D<T>::intersectsLine(const LineSeg2D& l,T& alpha) const
{
  Mat2T m;
  m.col(0)=_dir;
  m.col(1)=-l._dir;
  Mat2T im=inv2x2(m);
  if(im.isZero())
    return false;
  Vec2T alpha01=im*(l._a-_a);
  alpha=alpha01[0];
  return alpha01[0]>=0 && alpha01[0]<=_len && alpha01[1]>=0 && alpha01[1]<=l._len;
}
template <typename T>
bool LineSeg2D<T>::intersectsCircle(const Vec2T& pos,const T rad,T& alpha) const
{
  Vec2T a2p=_a-pos;
  T a=_dir.squaredNorm();
  T b=_dir.dot(a2p)*2;
  T c=a2p.dot(a2p)-rad*rad;
  T delta=b*b-4*a*c;
  if(delta<=0)
    return false;
  else {
    delta=std::sqrt(delta);
    T alpha0=(-b-delta)/(2*a);
    T alpha1=(-b+delta)/(2*a);
    if(alpha0<0) {
      if(alpha1<0 || alpha1>_len)
        return false;
      else {
        alpha=alpha1;
        return true;
      }
    } else if(alpha0<=_len) {
      alpha=alpha0;
      return true;
    } else return false;
  }
}
template <typename T>
typename LineSeg2D<T>::Vec2T LineSeg2D<T>::interp(T alpha) const
{
  return _a+_dir*alpha;
}
template <typename T>
void LineSeg2D<T>::setRandom(T a,T b)
{
  _a=Vec2T::Random()*a;
  _b=Vec2T::Random()*b;
  initialize();
}
template <typename T>
BBox<T,2> LineSeg2D<T>::computeBB() const
{
  BBox<T,2> bb;
  bb.setUnion(_a);
  bb.setUnion(_b);
  return bb;
}
template <typename T>
LineSeg2D<T> LineSeg2D<T>::negate() const
{
  LineSeg2D<T> ret;
  ret._a=_a;
  ret._b=_a*2-_b;
  ret._dir=-_dir;
  ret._len=_len;
  ret._invLen=_invLen;
  return ret;
}
template <typename T>
void LineSeg2D<T>::initialize()
{
  _dir=(_b-_a).normalized();
  _len=(_b-_a).norm();
  _invLen=1/_len;
}
template <typename T>
typename LineSeg2D<T>::Vec3T LineSeg2D<T>::a3D() const
{
  return to3D(_a);
}
template <typename T>
typename LineSeg2D<T>::Vec3T LineSeg2D<T>::b3D() const
{
  return to3D(_b);
}
//DirectionalCostCase2D
template <typename T>
DirectionalCostCase2D<T>::DirectionalCostCase2D(T alpha0,T alpha1):_alpha0(alpha0),_alpha1(alpha1),_a(0),_b(0),_c(0),_type(INTERIOR) {}
template <typename T>
DirectionalCostCase2D<T>::DirectionalCostCase2D(T alpha0,T alpha1,const LineSeg2D<T>& l,const Vec2T& v0):_alpha0(alpha0),_alpha1(alpha1)
{
  Vec2T a2v=l._a-v0;
  _a=l._dir.dot(l._dir);
  _b=l._dir.dot(a2v)*2;
  _c=a2v.dot(a2v);
  _type=VERTEX;
}
template <typename T>
DirectionalCostCase2D<T>::DirectionalCostCase2D(T alpha0,T alpha1,const LineSeg2D<T>& l,const Vec2T& v0,const Vec2T& n):_alpha0(alpha0),_alpha1(alpha1)
{
  Vec2T a2v=l._a-v0;
  T a2vn=a2v.dot(n);
  T dirn=l._dir.dot(n);
  _a=dirn*dirn;
  _b=dirn*a2vn*2;
  _c=a2vn*a2vn;
  _type=EDGE;
}
template <typename T>
typename DirectionalCostCase2D<T>::V3ss DirectionalCostCase2D<T>::getVss(const LineSeg2D<T>& l,T interval,bool planar) const
{
  Vec3T v;
  V3ss vss;
  sizeType nrSample=std::max<sizeType>(std::ceil((_alpha1-_alpha0)/interval),2);
  for(sizeType i=0; i<nrSample; i++) {
    T alpha=i/(T)(nrSample-1);
    alpha=_alpha0*(1-alpha)+_alpha1*alpha;
    v.template segment<2>(0)=l._a+l._dir*alpha;
    if(planar)
      v[2]=0;
    else v[2]=cost(alpha);
    vss.push_back(v);
  }
  return vss;
}
template <typename T>
T DirectionalCostCase2D<T>::cost(T alpha) const
{
  return _a*alpha*alpha+_b*alpha+_c;
}
template <typename T>
T DirectionalCostCase2D<T>::color() const
{
  if(_type==INTERIOR)
    return 0;
  else if(_type==VERTEX)
    return 1;
  else return 2;
}
//DirectionalCost2D
template <typename T>
void DirectionalCost2D<T>::writeVTK(const std::string& path,const LineSeg2D<T>& l,T interval,bool planar) const
{
  V3ss vss;
  std::vector<T> css,cssc;
  for(sizeType i=0; i<(sizeType)_case.size(); i++) {
    V3ss vssc=_case[i].getVss(l,interval,planar);
    vss.insert(vss.end(),vssc.begin(),vssc.end());
    cssc.assign(vssc.size(),_case[i].color());
    css.insert(css.end(),cssc.begin(),cssc.end());
  }
  VTKWriter<T> os("DirectionalCost",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,0,1),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()-1,0,1),
                 VTKWriter<T>::LINE);
  os.appendCustomPointData("color",css.begin(),css.end());
}
//Polygon2D
template <typename T>
Polygon2D<T>::Polygon2D() {}
template <typename T>
Polygon2D<T>::Polygon2D(const Vss& vss,T infinityNumber):_vss(vss)
{
  initialize(infinityNumber);
}
template <typename T>
Polygon2D<T>::Polygon2D(sizeType n,T range,T infinityNumber)
{
  Vss vss;
  for(sizeType i=0; i<n; i++)
    vss.push_back(Vec2T::Random()*range);

  Vssi ess;
  for(sizeType i=0; i<n; i++)
    for(sizeType j=0; j<n; j++) {
      if(i==j)
        continue;
      Vec2T nor=perp(Vec2T(vss[i]-vss[j])).normalized();
      T minV=infinityNumber;
      T maxV=-infinityNumber;
      for(sizeType k=0; k<n; k++)
        if(k!=i && k!=j) {
          minV=std::min(minV,nor.dot(vss[k]-vss[j]));
          maxV=std::max(maxV,nor.dot(vss[k]-vss[j]));
        }
      if(minV<0 && maxV<0)
        ess.push_back(Vec2i(j,i));
    }
  ASSERT(!ess.empty());

  std::vector<sizeType> iss;
  iss.push_back(ess[0][0]);
  iss.push_back(ess[0][1]);
  while(true) {
    for(sizeType i=0; i<(sizeType)ess.size(); i++)
      if(ess[i][0]==iss.back()) {
        iss.push_back(ess[i][1]);
        break;
      }
    ASSERT(iss.size()<=vss.size()+1);
    if(iss[0]==iss.back()) {
      iss.pop_back();
      break;
    }
  }
  for(sizeType i=0; i<(sizeType)iss.size(); i++)
    _vss.push_back(vss[iss[i]]);
  initialize(infinityNumber);
}
template <typename T>
void Polygon2D<T>::writeVTK(const std::string& path,bool rays) const
{
  std::vector<Vec3T,Eigen::aligned_allocator<Vec3T>> vss;
  std::vector<Vec3T,Eigen::aligned_allocator<Vec3T>> rss;
  for(sizeType i=2; i<(sizeType)_vss.size(); i++) {
    Vec3T v0=vertex3D(0);
    Vec3T v1=vertex3D(i-1);
    Vec3T v2=vertex3D(i-0);
    vss.push_back(v0);
    vss.push_back(v1);
    vss.push_back(v2);
  }
  VTKWriter<T> os("Polygon2D",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,3,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/3,3,0),
                 VTKWriter<T>::TRIANGLE);
  if(rays) {
    for(sizeType i=0; i<(sizeType)_rays.size(); i++) {
      rss.push_back(_rays[i].a3D());
      rss.push_back(_rays[i].b3D());
    }
    os.setRelativeIndex();
    os.appendPoints(rss.begin(),rss.end());
    os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                   typename VTKWriter<T>::template IteratorIndex<Vec3i>(rss.size()/2,2,0),
                   VTKWriter<T>::LINE,true);
  }
}
template <typename T>
void Polygon2D<T>::writeDistanceVTK(const std::string& path,const Vss& ptss) const
{
  V3ss vss;
  std::vector<std::pair<T,Vec2T>> cpss;
  cpss=distanceBatched(ptss);
  for(sizeType i=0; i<(sizeType)ptss.size(); i++) {
    vss.push_back(to3D(ptss[i]));
    vss.push_back(to3D(cpss[i].second));
  }
  VTKWriter<T> os("Polygon2DDistance",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<T>::LINE);
}
template <typename T>
Polygon2D<T> Polygon2D<T>::transform(const Vec2T& pos,T theta) const
{
  Mat2T R;
  if(theta!=0) {
    R(0,0)=R(1,1)=std::cos(theta);
    R(1,0)=R(0,1)=std::sin(theta);
    R(0,1)*=-1;
  } else {
    R.setIdentity();
  }

  Polygon2D ret;
  for(sizeType i=0; i<(sizeType)_vss.size(); i++)
    ret._vss.push_back(R*_vss[i]+pos);
  for(sizeType i=0; i<(sizeType)_nss.size(); i++)
    ret._nss.push_back(R*_nss[i]);
  for(sizeType i=0; i<(sizeType)_rays.size(); i++)
    ret._rays.push_back(_rays[i].transform(pos,theta));
  return ret;
}
template <typename T>
BBox<T,2> Polygon2D<T>::computeBB() const
{
  BBox<T,2> bb;
  for(sizeType i=0; i<(sizeType)_vss.size(); i++)
    bb.setUnion(_vss[i]);
  return bb;
}
//getter
template <typename T>
typename Polygon2D<T>::Vec2T Polygon2D<T>::edge(sizeType i) const
{
  return vertex(i+1)-vertex(i);
}
template <typename T>
typename Polygon2D<T>::Vec2T Polygon2D<T>::vertex(sizeType i) const
{
  while(i<0)
    i+=(sizeType)_vss.size();
  return _vss[i%nrV()];
}
template <typename T>
typename Polygon2D<T>::Vec3T Polygon2D<T>::vertex3D(sizeType i) const
{
  return to3D(vertex(i));
}
template <typename T>
typename Polygon2D<T>::Vec2T Polygon2D<T>::edgeLen(sizeType i,T infinityNumber) const
{
  Vec2T e=edge(i);
  return e.normalized()*infinityNumber;
}
template <typename T>
typename Polygon2D<T>::Vec2T Polygon2D<T>::outNormalLen(sizeType i,T infinityNumber) const
{
  return perp(edge(i)).normalized()*infinityNumber;
}
template <typename T>
bool Polygon2D<T>::contains(const Vec2T& pt) const
{
  for(sizeType i=0; i<(sizeType)_vss.size(); i++)
    if((pt-_vss[i]).dot(_nss[i])>0)
      return false;
  return true;
}
template <typename T>
void Polygon2D<T>::initialize(T infinityNumber)
{
  _nss.resize(_vss.size());
  for(sizeType i=0; i<nrV(); i++) {
    _nss[i]=outNormalLen(i,1);
    _rays.push_back(LineSeg2D<T>(vertex(i),vertex(i+1)));
  }
  for(sizeType i=0; i<nrV(); i++) {
    _rays.push_back(LineSeg2D<T>(vertex(i),vertex(i)+outNormalLen(i-1,infinityNumber)));
    _rays.push_back(LineSeg2D<T>(vertex(i),vertex(i)+outNormalLen(i,infinityNumber)));
  }
}
template <typename T>
sizeType Polygon2D<T>::nrV() const
{
  return (sizeType)_vss.size();
}
//directional cost
template <typename T>
bool Polygon2D<T>::containsEdgeDirichlet(const Vec2T& pt,sizeType id) const
{
  sizeType idR=(id+1)%(sizeType)_vss.size();
  return (pt-_vss[id]).dot(_nss[id])>=0 && (pt-_vss[id]).dot(_rays[id]._dir)>=0 && (pt-_vss[idR]).dot(_rays[id]._dir)<=0;
}
template <typename T>
bool Polygon2D<T>::containsVertexDirichlet(const Vec2T& pt,sizeType id) const
{
  sizeType idL=(id+(sizeType)_vss.size()-1)%(sizeType)_vss.size();
  return (pt-_vss[id]).dot(_rays[id]._dir)<=0 && (pt-_vss[id]).dot(_rays[idL]._dir)>=0;
}
template <typename T>
DirectionalCost2D<T> Polygon2D<T>::buildDirectionalCost(const LineSeg2D<T>& l) const
{
  T a;
  std::vector<T> alpha(1,0.0);
  for(sizeType j=0; j<(sizeType)_rays.size(); j++)
    if(l.intersectsLine(_rays[j],a))
      alpha.push_back(a);
  std::sort(alpha.begin(),alpha.end());
  alpha.push_back(l._len);

  DirectionalCost2D<T> cost;
  for(sizeType j=0; j<(sizeType)alpha.size()-1; j++) {
    Vec2T pt=l.interp((alpha[j]+alpha[j+1])/2);
    if(contains(pt))
      cost._case.push_back(DirectionalCostCase2D<T>(alpha[j],alpha[j+1]));
    else {
      for(sizeType k=0; k<(sizeType)_rays.size(); k++)
        if(containsVertexDirichlet(pt,k)) {
          cost._case.push_back(DirectionalCostCase2D<T>(alpha[j],alpha[j+1],l,_vss[k]));
          break;
        } else if(containsEdgeDirichlet(pt,k)) {
          cost._case.push_back(DirectionalCostCase2D<T>(alpha[j],alpha[j+1],l,_vss[k],_nss[k]));
          break;
        }
    }
  }
  return cost;
}
template <typename T>
std::vector<DirectionalCost2D<T>> Polygon2D<T>::buildDirectionalCostBatched(const Lss& lss) const
{
  std::vector<DirectionalCost2D<T>> costs(lss.size());
  #pragma omp parallel for
  for(sizeType i=0; i<(sizeType)lss.size(); i++)
    costs[i]=buildDirectionalCost(lss[i]);
  return costs;
}
//distance
template <typename T>
std::pair<T,typename Polygon2D<T>::Vec2T> Polygon2D<T>::distance(const Vec2T& pt) const
{
  Vec2T cp=pt;
  if(contains(pt)) {
    cp=pt;
    return std::make_pair(0,cp);
  } else {
    Vec2T cp2;
    T minDist=std::numeric_limits<T>::max(),dist;
    for(sizeType i=0; i<(sizeType)_vss.size(); i++) {
      cp2=_vss[i];
      dist=(pt-cp2).norm();
      if(dist<minDist) {
        minDist=dist;
        cp=cp2;
      }
    }
    for(sizeType i=0; i<(sizeType)_vss.size(); i++) {
      dist=std::min<T>(std::max<T>((pt-_rays[i]._a).dot(_rays[i]._dir),0),_rays[i]._len);
      cp2=_rays[i].interp(dist);
      dist=(pt-cp2).norm();
      if(dist<minDist) {
        minDist=dist;
        cp=cp2;
      }
    }
    return std::make_pair(minDist,cp);
  }
}
template <typename T>
std::vector<std::pair<T,typename Polygon2D<T>::Vec2T>> Polygon2D<T>::distanceBatched(const Vss& ptss) const
{
  std::vector<std::pair<T,typename Polygon2D<T>::Vec2T>> ret;
  ret.resize(ptss.size());
  #pragma omp parallel for
  for(sizeType i=0; i<(sizeType)ptss.size(); i++)
    ret[i]=distance(ptss[i]);
  return ret;
}
//instance
template struct LineSeg2D<scalarF>;
template struct LineSeg2D<scalarD>;
template struct DirectionalCostCase2D<scalarF>;
template struct DirectionalCostCase2D<scalarD>;
template struct DirectionalCost2D<scalarF>;
template struct DirectionalCost2D<scalarD>;
template struct Polygon2D<scalarF>;
template struct Polygon2D<scalarD>;
}
