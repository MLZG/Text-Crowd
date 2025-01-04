#ifndef CONFIG_H
#define CONFIG_H

#include <Eigen/Dense>
#include <experimental/filesystem>
typedef double scalarD;
typedef float scalarF;
typedef int sizeType;

typedef Eigen::Matrix<sizeType,2,1> Vec2i;
typedef Eigen::Matrix<sizeType,3,1> Vec3i;

#define ASSERT(x) do{if(!(x)){exit(EXIT_FAILURE);}}while(0);
#define ASSERT_MSG(x,msg) do{if(!(x)){WARNING(msg);exit(EXIT_FAILURE);}}while(0);
#define ASSERT_MSGV(x,fmt,...) do{if(!(x)){WARNINGV(fmt,__VA_ARGS__);exit(EXIT_FAILURE);}}while(0);

#define WARNING(msg) do{printf("[WARNING] \x1B[31m %s \x1B[0m\n",msg);fflush(stdout);}while(0);
#define WARNINGV(fmt,...) do{printf("[WARNING] \x1B[31m " fmt " \x1B[0m\n",__VA_ARGS__);fflush(stdout);}while(0);
#define INFO(msg) do{printf("[INFO] %s \n",msg);fflush(stdout);}while(0);
#define INFOV(fmt,...) do{printf("[INFO] " fmt " \n",__VA_ARGS__);fflush(stdout);}while(0);
#define NOTIFY_MSG(msg) do{printf("[NOTIFY] %s \n",msg);fflush(stdout);}while(0); getchar();
#define NOTIFY_MSGV(fmt,...) do{printf("[NOTIFY] " fmt " \n",__VA_ARGS__);fflush(stdout);}while(0); getchar();

namespace Collision2D
{
template <typename T>
EIGEN_DEVICE_FUNC Eigen::Matrix<T,3,1> to3D(const Eigen::Matrix<T,2,1>& e)
{
  return Eigen::Matrix<T,3,1>(e[0],e[1],0);
}
template <typename T>
EIGEN_DEVICE_FUNC Eigen::Matrix<T,2,1> perp(const Eigen::Matrix<T,2,1>& e)
{
  return Eigen::Matrix<T,2,1>(e[1],-e[0]);
}
template <typename T>
EIGEN_DEVICE_FUNC Eigen::Matrix<T,2,2> inv2x2(const Eigen::Matrix<T,2,2>& m)
{
  Eigen::Matrix<T,2,2> im;
  im(0,0)=m(1,1);
  im(1,1)=m(0,0);
  im(0,1)=-m(0,1);
  im(1,0)=-m(1,0);
  T det=m(0,0)*m(1,1)-m(0,1)*m(1,0);
  if(std::abs(det)<std::numeric_limits<T>::min())
    im.setZero();
  else
    im*=1/det;
  return im;
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC bool compL(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.array().binaryExpr(b.array(),std::less<typename Derived::Scalar>()).all();
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC bool compLE(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.array().binaryExpr(b.array(),std::less_equal<typename Derived::Scalar>()).all();
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC bool compG(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.array().binaryExpr(b.array(),std::greater<typename Derived::Scalar>()).all();
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC bool compGE(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.array().binaryExpr(b.array(),std::greater_equal<typename Derived::Scalar>()).all();
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC Eigen::Matrix<typename Derived::Scalar,Derived::RowsAtCompileTime,Derived::ColsAtCompileTime>
compMax(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.cwiseMax(b);
}
template <typename Derived,typename Derived2>
EIGEN_DEVICE_FUNC Eigen::Matrix<typename Derived::Scalar,Derived::RowsAtCompileTime,Derived::ColsAtCompileTime>
compMin(const Eigen::MatrixBase<Derived>& a,const Eigen::MatrixBase<Derived2>& b)
{
  return a.cwiseMin(b);
}

//filesystem
bool notDigit(char c);
bool lessDirByNumber(std::experimental::filesystem::v1::path A,std::experimental::filesystem::v1::path B);
bool exists(const std::experimental::filesystem::v1::path& path);
void removeDir(const std::experimental::filesystem::v1::path& path);
void create(const std::experimental::filesystem::v1::path& path);
void recreate(const std::experimental::filesystem::v1::path& path);
std::vector<std::experimental::filesystem::v1::path> files(const std::experimental::filesystem::v1::path& path);
std::vector<std::experimental::filesystem::v1::path> directories(const std::experimental::filesystem::v1::path& path);
void sortFilesByNumber(std::vector<std::experimental::filesystem::v1::path>& files);
bool isDir(const std::experimental::filesystem::v1::path& path);
size_t fileSize(const std::experimental::filesystem::v1::path& path);
}

#endif
