#include "Environment2D.h"
#include "VTKWriter.h"
#include <omp.h>

namespace Collision2D
{
//Hash
std::size_t Hash::operator()(const Vec2i& h) const
{
  return std::hash<sizeType>()(h[0])+std::hash<sizeType>()(h[1]);
}
//BVHNode
template <typename T>
BVHNode<T>::CompareBB::CompareBB(sizeType dim):_dim(dim) {}
template <typename T>
bool BVHNode<T>::CompareBB::operator()(const BBox<T,2>& a,const BBox<T,2>& b) const
{
  T ad=(a._minC[_dim]+a._maxC[_dim])/2;
  T bd=(b._minC[_dim]+b._maxC[_dim])/2;
  return ad<bd;
}
template <typename T>
bool BVHNode<T>::CompareBB::operator()(const BVHNode<T>& a,const BVHNode<T>& b) const
{
  return operator()(a._bb,b._bb);
}
template <typename T>
sizeType BVHNode<T>::buildBVHTopDown(std::vector<BVHNode<T>>& bvh,sizeType f,sizeType t)
{
  if(t==f+1)
    return f;
  BVHNode<T> node;
  for(sizeType i=f; i<t; i++)
    node._bb.setUnion(bvh[i]._bb);
  if(node._bb.getExtent()[0]>node._bb.getExtent()[1])
    std::sort(bvh.begin()+f,bvh.begin()+t,CompareBB(0));
  else
    std::sort(bvh.begin()+f,bvh.begin()+t,CompareBB(1));
  sizeType a=(f+t)/2;
  node._childL=buildBVHTopDown(bvh,f,a);
  node._childR=buildBVHTopDown(bvh,a,t);
  node._cell=-1;
  bvh.push_back(node);
  return (sizeType)bvh.size()-1;
}
template <typename T>
std::vector<BVHNode<T>> BVHNode<T>::buildBVHTopDown(std::function<BBox<T,2>(sizeType)> f,sizeType n)
{
  std::vector<BVHNode<T>> bvh;
  bvh.resize(n);
  for(sizeType i=0; i<n; i++) {
    bvh[i]._bb=f(i);
    bvh[i]._childL=bvh[i]._childR=-1;
    bvh[i]._cell=i;
  }
  buildBVHTopDown(bvh,0,n);
  return bvh;
}
template <typename T>
std::vector<BVHNode<T>> BVHNode<T>::buildBVHBottomUp(std::function<BBox<T,2>(sizeType)> f,sizeType n)
{
  std::vector<BVHNode<T>> bvh;
  bvh.resize(n);
  std::unordered_set<sizeType> root;
  for(sizeType i=0; i<n; i++) {
    bvh[i]._bb=f(i);
    bvh[i]._childL=bvh[i]._childR=-1;
    bvh[i]._cell=i;
    root.insert(i);
  }
  std::vector<BVHNode<T>> nodeHeap;
  for(sizeType I:root)
    for(sizeType J:root)
      if(I!=J) {
        BVHNode<T> node;
        node._bb=bvh[I]._bb;
        node._bb.setUnion(bvh[J]._bb);
        node._childL=I;
        node._childR=J;
        node._cell=-1;
        nodeHeap.push_back(node);
        std::push_heap(nodeHeap.begin(),nodeHeap.end());
      }
  while(root.size()>1) {
    std::pop_heap(nodeHeap.begin(),nodeHeap.end());
    BVHNode<T> n=nodeHeap.back();
    nodeHeap.pop_back();
    if(root.find(n._childL)!=root.end() && root.find(n._childR)!=root.end()) {
      root.erase(n._childL);
      root.erase(n._childR);
      for(sizeType I:root) {
        BVHNode<T> node;
        node._bb=bvh[I]._bb;
        node._bb.setUnion(n._bb);
        node._childL=I;
        node._childR=bvh.size();
        node._cell=-1;
        nodeHeap.push_back(node);
        std::push_heap(nodeHeap.begin(),nodeHeap.end());
      }
      root.insert(bvh.size());
      bvh.push_back(n);
    }
  }
  return bvh;
}
template <typename T>
std::vector<BVHNode<T>> BVHNode<T>::buildBVH(std::function<BBox<T,2>(sizeType)> f,sizeType n,bool bottomup)
{
  if(bottomup)
    return buildBVHBottomUp(f,n);
  else return buildBVHTopDown(f,n);
}
template <typename T>
bool BVHNode<T>::operator<(const BVHNode<T>& other) const
{
  return _bb.getExtent().sum()<other._bb.getExtent().sum();
}
//Environment2D
template <typename T>
Environment2D<T>::Environment2D(const Pss& pss,bool bottomup)
{
  for(sizeType i=0; i<(sizeType)pss.size(); i++)
    for(sizeType j=0; j<(sizeType)pss[i]._vss.size(); j++) {
      _lss.push_back(pss[i]._rays[j]);
      _liss.push_back(i);
    }
  _bvhl=BVHNode<T>::buildBVH([&](sizeType n) {
    return _lss[n].computeBB().enlarge(0.001f);
  },_lss.size(),bottomup);
}
template <typename T>
Environment2D<T>::Environment2D(const Lss& lss,bool bottomup)
{
  _lss=lss;
  _liss.assign(_lss.size(),-1);
  _bvhl=BVHNode<T>::buildBVH([&](sizeType n) {
    return _lss[n].computeBB().enlarge(0.001f);
  },_lss.size(),bottomup);
}
template <typename T>
Environment2D<T>::Environment2D(const Vssi& iss,bool bottomup)
{
  Vssi vss;
  _iss.insert(iss.begin(),iss.end());
  for(const Vec2i& I:_iss) {
    if(_iss.find(Vec2i(I[0],I[1]-1))==_iss.end())
      _lss.push_back(LineSeg2D<T>(Vec2T(I[0],I[1]),Vec2T(I[0]+1,I[1])));
    if(_iss.find(Vec2i(I[0]+1,I[1]))==_iss.end())
      _lss.push_back(LineSeg2D<T>(Vec2T(I[0]+1,I[1]),Vec2T(I[0]+1,I[1]+1)));
    if(_iss.find(Vec2i(I[0],I[1]+1))==_iss.end())
      _lss.push_back(LineSeg2D<T>(Vec2T(I[0]+1,I[1]+1),Vec2T(I[0],I[1]+1)));
    if(_iss.find(Vec2i(I[0]-1,I[1]))==_iss.end())
      _lss.push_back(LineSeg2D<T>(Vec2T(I[0],I[1]+1),Vec2T(I[0],I[1])));
  }
  _liss.assign(_lss.size(),-1);
  _bvhl=BVHNode<T>::buildBVH([&](sizeType n) {
    return _lss[n].computeBB().enlarge(0.001f);
  },_lss.size(),bottomup);
}
template <typename T>
void Environment2D<T>::writeVTK(const std::string& path,sizeType RES) const
{
  V3ss vss;
  for(sizeType i=0; i<(sizeType)_lss.size(); i++) {
    vss.push_back(to3D(_lss[i]._a));
    vss.push_back(to3D(_lss[i]._b));
  }
  if(_ass.size()>0) {
    for(sizeType i=0; i<(sizeType)_ass.size(); i++) {
      const Vec2T& pos=_ass[i];
      T rad=_bvha[i]._bb.getExtent()[0]/2;
      for(sizeType j=0; j<RES; j++) {
        T a=j*M_PI*2/RES,b=(j+1)*M_PI*2/RES;
        vss.push_back(to3D(Vec2T(std::cos(a)*rad+pos[0],std::sin(a)*rad+pos[1])));
        vss.push_back(to3D(Vec2T(std::cos(b)*rad+pos[0],std::sin(b)*rad+pos[1])));
      }
    }
  }
  VTKWriter<T> os("Environment2D",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<T>::LINE);
}
//range
template <typename T>
std::pair<T,sizeType> Environment2D<T>::ray(const LineSeg2D<T>& l,sizeType idMask) const
{
  sizeType id=-1;
  T alpha=l._len;
  rayE(_bvhl.size()-1,l,alpha,id,idMask);
  rayA(_bvha.size()-1,l,alpha,id,idMask);
  return std::make_pair(alpha,id);
}
template <typename T>
bool Environment2D<T>::rayE(sizeType i,const LineSeg2D<T>& l,T& alpha,sizeType& id,sizeType idMask) const
{
  if(i<0 || i>=(sizeType)_bvhl.size())
    return false;
  const BVHNode<T>& n=_bvhl[i];
  if(n._cell>=0) {
    if(_liss[n._cell]==idMask)
      return false;
    T alphaNew=0;
    if(l.intersectsLine(_lss[n._cell],alphaNew) && alphaNew<alpha) {
      alpha=alphaNew;
      id=_liss[n._cell];
      return true;
    } else return false;
  } else if(n._bb.intersect(l._a,l.interp(alpha),2)) {
    bool iL=rayE(n._childL,l,alpha,id,idMask);
    bool iR=rayE(n._childR,l,alpha,id,idMask);
    return iL||iR;
  } else return false;
}
template <typename T>
bool Environment2D<T>::rayA(sizeType i,const LineSeg2D<T>& l,T& alpha,sizeType& id,sizeType idMask) const
{
  if(i<0 || i>=(sizeType)_bvha.size())
    return false;
  const BVHNode<T>& n=_bvha[i];
  if(n._cell>=0) {
    if(_aiss[n._cell]==idMask)
      return false;
    T alphaNew=0;
    T rad=n._bb.getExtent()[0]/2;
    if(l.intersectsCircle(_ass[n._cell],rad,alphaNew) && alphaNew<alpha) {
      alpha=alphaNew;
      id=_aiss[n._cell];
      return true;
    } else return false;
  } else if(n._bb.intersect(l._a,l.interp(alpha),2)) {
    bool iL=rayA(n._childL,l,alpha,id,idMask);
    bool iR=rayA(n._childR,l,alpha,id,idMask);
    return iL||iR;
  } else return false;
}
template <typename T>
typename std::tuple<T,sizeType,T,sizeType> Environment2D<T>::range(const LineSeg2D<T>& l,bool oneDir,sizeType idMask) const
{
  if(!_iss.empty()) {
    Vec2i id(std::floor(l._a[0]),std::floor(l._a[1]));
    if(_iss.find(id)==_iss.end())
      return std::make_tuple(0,0,0,0);
  }
  if(!oneDir) {
    std::pair<T,sizeType> a=ray(l,idMask);
    std::pair<T,sizeType> b=ray(l.negate(),idMask);
    return std::make_tuple(a.first,a.second,b.first,b.second);
  } else {
    std::pair<T,sizeType> a=ray(l,idMask);
    return std::make_tuple(a.first,a.second,0,0);
  }
}
template <typename T>
typename Environment2D<T>::Tss Environment2D<T>::rangeBatched(const Lss& lss,bool oneDir) const
{
  Tss vss(lss.size());
  #pragma omp parallel for
  for(sizeType i=0; i<(sizeType)lss.size(); i++)
    vss[i]=range(lss[i],oneDir);
  return vss;
}
template <typename T>
typename Environment2D<T>::Lss Environment2D<T>::sensorLine(const Vec2T& pos,T range,sizeType N,sizeType idMask) const
{
  Lss ret(N);
  #pragma omp parallel for
  for(sizeType i=0; i<N; i++) {
    T alpha=i*M_PI*2/N;
    ret[i]._a=pos;
    ret[i]._dir=Vec2T(std::cos(alpha),std::sin(alpha));
    ret[i]._b=pos+ret[i]._dir*range;
    ret[i]._len=range;
    ret[i]._invLen=1/range;

    T l=ray(ret[i],idMask).first;
    ret[i]._b=pos+ret[i]._dir*l;
    ret[i]._len=l;
    ret[i]._invLen=1/l;
  }
  return ret;
}
template <typename T>
std::vector<T> Environment2D<T>::sensorData(const Vec2T& pos,T range,sizeType N,sizeType idMask) const
{
  std::vector<T> ret(N);
  #pragma omp parallel for
  for(sizeType i=0; i<N; i++) {
    T alpha=i*M_PI*2/N;
    LineSeg2D<T> l;
    l._a=pos;
    l._dir=Vec2T(std::cos(alpha),std::sin(alpha));
    l._b=pos+l._dir*range;
    l._len=range;
    l._invLen=1/range;
    ret[i]=ray(l,idMask).first;
  }
  return ret;
}
template <typename T>
void Environment2D<T>::writeSensorVTK(const std::string& path,const Vec2T& pos,T range,sizeType N,sizeType idMask) const
{
  V3ss vss;
  Lss ret=sensorLine(pos,range,N,idMask);
  for(sizeType i=0; i<(sizeType)ret.size(); i++) {
    vss.push_back(to3D(ret[i]._a));
    vss.push_back(to3D(ret[i]._b));
  }
  VTKWriter<T> os("Sensor",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<T>::LINE);
}
//dir polygon
template <typename T>
typename Environment2D<T>::Dss Environment2D<T>::dirPolygonBatched(const Pss& pss,const Vec2T& dir) const
{
  Lss lss;
  std::vector<bool> bss;
  std::vector<sizeType> vss;
  std::vector<sizeType> mss;
  for(sizeType i=0; i<(sizeType)pss.size(); i++) {
    sizeType nrV=(sizeType)pss[i]._vss.size();
    for(sizeType j=0; j<nrV; j++) {
      const Vec2T& n0=pss[i]._nss[(j+nrV-1)%nrV];
      const Vec2T& n1=pss[i]._nss[j];
      if(dir.dot(n0)>0 || dir.dot(n1)>0) {
        lss.push_back(LineSeg2D<T>(pss[i]._vss[j],pss[i]._vss[j]+dir));
        bss.push_back(true);
        vss.push_back(j);
        mss.push_back(i);
      }
      if(dir.dot(n0)<0 || dir.dot(n1)<0) {
        lss.push_back(LineSeg2D<T>(pss[i]._vss[j],pss[i]._vss[j]-dir));
        bss.push_back(false);
        vss.push_back(j);
        mss.push_back(i);
      }
    }
  }
  Dss dss(lss.size());
  #pragma omp parallel for
  for(sizeType i=0; i<(sizeType)lss.size(); i++) {
    std::tuple<T,sizeType,T,sizeType> t=range(lss[i],true,mss[i]);
    dss[i]=std::make_tuple(mss[i],vss[i],(bss[i]?std::get<0>(t):-std::get<0>(t)),std::get<1>(t));
  }
  return dss;
}
template <typename T>
void Environment2D<T>::writeDirVTK(const std::string& path,const Pss& pss,const Vec2T& dir) const
{
  V3ss vss;
  //pid,vid,did,d,poid
  Vec2T dirn=dir.normalized();
  Dss dss=dirPolygonBatched(pss,dir);
  for(sizeType i=0; i<(sizeType)dss.size(); i++) {
    const std::tuple<sizeType,sizeType,T,sizeType>& d=dss[i];
    const Vec2T& v=pss[std::get<0>(d)]._vss[std::get<1>(d)];
    const Vec2T vd=v+dirn*std::get<2>(d);
    vss.push_back(to3D(v));
    vss.push_back(to3D(vd));
  }
  VTKWriter<T> os("Dir",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<T>::LINE);
}
//valid sample
template <typename T>
bool Environment2D<T>::circle(sizeType i,const Vec2T& v,T rad,sizeType idMask) const
{
  if(i<0 || i>=(sizeType)_bvhl.size())
    return false;
  const BVHNode<T>& n=_bvhl[i];
  if(n._cell>=0) {
    if(_liss[n._cell]==idMask)
      return false;
    T alphaNew=0;
    return (_lss[n._cell]._a-v).norm()<rad || _lss[n._cell].intersectsCircle(v,rad,alphaNew);
  } else if(n._bb.distTo(v,2)<rad) {
    if(circle(n._childL,v,rad,idMask))
      return true;
    if(circle(n._childR,v,rad,idMask))
      return true;
    return false;
  } else return false;
}
template <typename T>
std::vector<sizeType> Environment2D<T>::validSamples(const Vss& css,const std::vector<T>& rad) const
{
  std::vector<sizeType> valids(css.size());
  #pragma omp parallel for
  for(sizeType i=0; i<(sizeType)css.size(); i++)
    valids[i]=circle((sizeType)_bvhl.size()-1,css[i],rad[i]);
  return valids;
}
template <typename T>
void Environment2D<T>::writeValidSamplesVTK(const std::string& path,const Vss& css,const std::vector<T>& rad,sizeType RES) const
{
  V3ss vss;
  std::vector<T> data;
  std::vector<sizeType> valid=validSamples(css,rad);
  for(sizeType i=0; i<(sizeType)css.size(); i++) {
    const Vec2T& pos=css[i];
    for(sizeType j=0; j<RES; j++) {
      T a=j*M_PI*2/RES,b=(j+1)*M_PI*2/RES;
      vss.push_back(to3D(Vec2T(std::cos(a)*rad[i]+pos[0],std::sin(a)*rad[i]+pos[1])));
      vss.push_back(to3D(Vec2T(std::cos(b)*rad[i]+pos[0],std::sin(b)*rad[i]+pos[1])));
      data.push_back(valid[i]?1:0);
    }
  }
  VTKWriter<T> os("ValidSample",path,true);
  os.appendPoints(vss.begin(),vss.end());
  os.appendCells(typename VTKWriter<T>::template IteratorIndex<Vec3i>(0,2,0),
                 typename VTKWriter<T>::template IteratorIndex<Vec3i>(vss.size()/2,2,0),
                 VTKWriter<T>::LINE);
  os.appendCustomData("color",data.begin(),data.end());
}
//set agent
template <typename T>
void Environment2D<T>::clearAgent()
{
  _ass.clear();
  _aiss.clear();
  _bvha.clear();
}
template <typename T>
void Environment2D<T>::setAgent(const Vss& vss,T rad)
{
  _ass=vss;
  _aiss.assign(vss.size(),-1);
  for(sizeType i=0; i<(sizeType)vss.size(); i++)
    _aiss[i]=i;
  _bvha=BVHNode<T>::buildBVH([&](sizeType n) {
    return BBox<T,2>(vss[n]-Vec2T::Constant(rad),vss[n]+Vec2T::Constant(rad));
  },vss.size(),false);
}
//instance
template struct Environment2D<scalarF>;
template struct Environment2D<scalarD>;
}
