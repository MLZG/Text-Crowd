#include "Config.h"

namespace Collision2D
{
//filesystem
bool notDigit(char c)
{
  return !std::isdigit(c);
}
bool lessDirByNumber(std::experimental::filesystem::v1::path A,std::experimental::filesystem::v1::path B)
{
  std::string strA=A.string();
  std::string::iterator itA=std::remove_if(strA.begin(),strA.end(),notDigit);
  strA.erase(itA,strA.end());

  std::string strB=B.string();
  std::string::iterator itB=std::remove_if(strB.begin(),strB.end(),notDigit);
  strB.erase(itB,strB.end());

  sizeType idA,idB;
  std::istringstream(strA) >> idA;
  std::istringstream(strB) >> idB;
  return idA < idB;
}
bool exists(const std::experimental::filesystem::v1::path& path)
{
  return std::experimental::filesystem::v1::exists(path);
}
void removeDir(const std::experimental::filesystem::v1::path& path)
{
  if(std::experimental::filesystem::v1::exists(path))
    try {
      std::experimental::filesystem::v1::remove_all(path);
    } catch(...) {}
}
void create(const std::experimental::filesystem::v1::path& path)
{
  if(!std::experimental::filesystem::v1::exists(path))
    try {
      std::experimental::filesystem::v1::create_directory(path);
    } catch(...) {}
}
void recreate(const std::experimental::filesystem::v1::path& path)
{
  removeDir(path);
  try {
    std::experimental::filesystem::v1::create_directory(path);
  } catch(...) {}
}
std::vector<std::experimental::filesystem::v1::path> files(const std::experimental::filesystem::v1::path& path)
{
  std::vector<std::experimental::filesystem::v1::path> ret;
  for(std::experimental::filesystem::v1::directory_iterator beg(path),end; beg!=end; beg++)
    if(std::experimental::filesystem::v1::is_regular_file(*beg))
      ret.push_back(*beg);
  return ret;
}
std::vector<std::experimental::filesystem::v1::path> directories(const std::experimental::filesystem::v1::path& path)
{
  std::vector<std::experimental::filesystem::v1::path> ret;
  for(std::experimental::filesystem::v1::directory_iterator beg(path),end; beg!=end; beg++)
    if(std::experimental::filesystem::v1::is_directory(*beg))
      ret.push_back(*beg);
  return ret;
}
void sortFilesByNumber(std::vector<std::experimental::filesystem::v1::path>& files)
{
  sort(files.begin(),files.end(),lessDirByNumber);
}
bool isDir(const std::experimental::filesystem::v1::path& path)
{
  return std::experimental::filesystem::v1::is_directory(path);
}
size_t fileSize(const std::experimental::filesystem::v1::path& path)
{
  return (size_t)std::experimental::filesystem::v1::file_size(path);
}
}
