#ifndef ENVIRONMENT_2D_H
#define ENVIRONMENT_2D_H

#include "Polygon2D.h"

namespace Collision2D
{
struct Hash
{
  std::size_t operator()(const Vec2i& h) const;
};
template <typename T>
struct BVHNode
{
  struct CompareBB
  {
    CompareBB(sizeType dim);
    bool operator()(const BBox<T,2>& a,const BBox<T,2>& b) const;
    bool operator()(const BVHNode<T>& a,const BVHNode<T>& b) const;
    T _dim;
  };
  static sizeType buildBVHTopDown(std::vector<BVHNode<T>>& bvh,sizeType f,sizeType t);
  static std::vector<BVHNode> buildBVHTopDown(std::function<BBox<T,2>(sizeType)> f,sizeType n);
  static std::vector<BVHNode> buildBVHBottomUp(std::function<BBox<T,2>(sizeType)> f,sizeType n);
  static std::vector<BVHNode> buildBVH(std::function<BBox<T,2>(sizeType)> f,sizeType n,bool bottomup);
  bool operator<(const BVHNode<T>& other) const;
  sizeType _childL;
  sizeType _childR;
  sizeType _cell;
  BBox<T,2> _bb;
};
template <typename T>
struct Environment2D
{
  typedef Eigen::Matrix<T,2,1> Vec2T;
  typedef Eigen::Matrix<T,3,1> Vec3T;
  typedef Eigen::Matrix<T,4,1> Vec4T;
  typedef Eigen::Matrix<T,2,2> Mat2T;
  typedef std::vector<Vec2i> Vssi;
  typedef std::vector<Vec2T> Vss;
  typedef std::vector<Vec3T> V3ss;
  typedef std::vector<LineSeg2D<T>> Lss;
  typedef std::vector<Polygon2D<T>> Pss;
  typedef std::vector<std::tuple<T,sizeType,T,sizeType>> Tss;
  typedef std::vector<std::tuple<sizeType,sizeType,T,sizeType>> Dss;//pid,vid,did,d,poid
  Environment2D(const Pss& pss,bool bottomup=false);
  Environment2D(const Lss& lss,bool bottomup=true);
  Environment2D(const Vssi& iss,bool bottomup=true);
  void writeVTK(const std::string& path,sizeType RES=32) const;
  //range
  std::pair<T,sizeType> ray(const LineSeg2D<T>& l,sizeType idMask=-2) const;
  bool rayE(sizeType i,const LineSeg2D<T>& l,T& alpha,sizeType& id,sizeType idMask=-2) const;
  bool rayA(sizeType i,const LineSeg2D<T>& l,T& alpha,sizeType& id,sizeType idMask=-2) const;
  std::tuple<T,sizeType,T,sizeType> range(const LineSeg2D<T>& l,bool oneDir,sizeType idMask=-2) const;
  Tss rangeBatched(const Lss& lss,bool oneDir) const;
  Lss sensorLine(const Vec2T& pos,T range,sizeType N,sizeType idMask=-2) const;
  std::vector<T> sensorData(const Vec2T& pos,T range,sizeType N,sizeType idMask=-2) const;
  void writeSensorVTK(const std::string& path,const Vec2T& pos,T range,sizeType N=64,sizeType idMask=-2) const;
  //dir polygon
  Dss dirPolygonBatched(const Pss& pss,const Vec2T& dir) const;
  void writeDirVTK(const std::string& path,const Pss& pss,const Vec2T& dir) const;
  //valid sample
  bool circle(sizeType i,const Vec2T& v,T rad,sizeType idMask=-2) const;
  std::vector<sizeType> validSamples(const Vss& vss,const std::vector<T>& rad) const;
  void writeValidSamplesVTK(const std::string& path,const Vss& vss,const std::vector<T>& rad,sizeType RES=64) const;
  //set agent
  void clearAgent();
  void setAgent(const Vss& vss,T rad);
  //env
  std::unordered_set<Vec2i,Hash> _iss;
  std::vector<BVHNode<T>> _bvhl;
  std::vector<sizeType> _liss;
  Lss _lss;
  //agent
  std::vector<Vec2T> _ass;
  std::vector<sizeType> _aiss;
  std::vector<BVHNode<T>> _bvha;
};
}

#endif
