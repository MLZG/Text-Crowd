#ifndef POLYGON_2D_H
#define POLYGON_2D_H

#include <unordered_set>
#include "Config.h"
#include "BBox.h"

namespace Collision2D
{
template <typename T>
struct LineSeg2D
{
  typedef Eigen::Matrix<T,2,1> Vec2T;
  typedef Eigen::Matrix<T,3,1> Vec3T;
  typedef Eigen::Matrix<T,2,2> Mat2T;
  LineSeg2D();
  LineSeg2D(const Vec2T& a,const Vec2T& b);
  LineSeg2D transform(const Vec2T& pos,T theta) const;
  bool intersectsLine(const LineSeg2D& l,T& alpha) const;
  bool intersectsCircle(const Vec2T& pos,T rad,T& alpha) const;
  Vec2T interp(T alpha) const;
  BBox<T,2> computeBB() const;
  LineSeg2D negate() const;
  void setRandom(T a,T b);
  void initialize();
  Vec3T a3D() const;
  Vec3T b3D() const;
  Vec2T _a,_b,_dir;
  T _len,_invLen;
};
template <typename T>
struct DirectionalCostCase2D
{
  enum CASE_TYPE
  {
    INTERIOR,
    VERTEX,
    EDGE,
  };
  typedef Eigen::Matrix<T,2,1> Vec2T;
  typedef Eigen::Matrix<T,3,1> Vec3T;
  typedef Eigen::Matrix<T,4,1> Vec4T;
  typedef std::vector<Vec2T> Vss;
  typedef std::vector<Vec3T> V3ss;
  DirectionalCostCase2D(T alpha0,T alpha1);
  DirectionalCostCase2D(T alpha0,T alpha1,const LineSeg2D<T>& l,const Vec2T& v0);
  DirectionalCostCase2D(T alpha0,T alpha1,const LineSeg2D<T>& l,const Vec2T& v0,const Vec2T& n);
  V3ss getVss(const LineSeg2D<T>& l,T interval,bool planar) const;
  T cost(T alpha) const;
  T color() const;
  T _alpha0,_alpha1,_a,_b,_c;
  CASE_TYPE _type;
};
template <typename T>
struct DirectionalCost2D
{
  typedef Eigen::Matrix<T,2,1> Vec2T;
  typedef Eigen::Matrix<T,3,1> Vec3T;
  typedef std::vector<Vec2T> Vss;
  typedef std::vector<Vec3T> V3ss;
  void writeVTK(const std::string& path,const LineSeg2D<T>& l,T interval,bool planar) const;
  std::vector<DirectionalCostCase2D<T>> _case;
};
template <typename T>
struct Polygon2D
{
  typedef Eigen::Matrix<T,2,1> Vec2T;
  typedef Eigen::Matrix<T,3,1> Vec3T;
  typedef Eigen::Matrix<T,4,1> Vec4T;
  typedef Eigen::Matrix<T,2,2> Mat2T;
  typedef std::vector<Vec2i> Vssi;
  typedef std::vector<Vec2T> Vss;
  typedef std::vector<Vec3T> V3ss;
  typedef std::vector<LineSeg2D<T>> Lss;
  Polygon2D();
  Polygon2D(const Vss& vss,T infinityNumber);
  Polygon2D(sizeType n,T range,T infinityNumber);
  void writeVTK(const std::string& path,bool rays=false) const;
  void writeDistanceVTK(const std::string& path,const Vss& ptss) const;
  Polygon2D transform(const Vec2T& pos,T theta) const;
  BBox<T,2> computeBB() const;
  //getter
  Vec2T edge(sizeType i) const;
  Vec2T vertex(sizeType i) const;
  Vec3T vertex3D(sizeType i) const;
  Vec2T edgeLen(sizeType i,T infinityNumber) const;
  Vec2T outNormalLen(sizeType i,T infinityNumber) const;
  bool contains(const Vec2T& pt) const;
  void initialize(T infinityNumber);
  sizeType nrV() const;
  //directional cost
  bool containsEdgeDirichlet(const Vec2T& pt,sizeType id) const;
  bool containsVertexDirichlet(const Vec2T& pt,sizeType id) const;
  DirectionalCost2D<T> buildDirectionalCost(const LineSeg2D<T>& l) const;
  std::vector<DirectionalCost2D<T>> buildDirectionalCostBatched(const Lss& lss) const;
  //distance
  std::pair<T,Vec2T> distance(const Vec2T& pt) const;
  std::vector<std::pair<T,Vec2T>> distanceBatched(const Vss& ptss) const;
  Vss _vss,_nss;
  Lss _rays;
};
}

#endif
