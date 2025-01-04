#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include "Config.h"
#include "BBox.h"
#include <fstream>
#include <unordered_map>

namespace Collision2D
{
//for VTK IO
void vtkWrite(std::ostream& oss,scalarF val);
void vtkWrite(std::ostream& oss,scalarD val);
void vtkWrite(std::ostream& oss,int val);
void vtkWrite(std::ostream& oss,char val);
void vtkWrite(std::ostream& oss,unsigned char val);
template<typename T>
struct VTKWriter {
  enum VTK_DATA_TYPE {
    UNSTRUCTURED_GRID,
    STRUCTURED_POINTS,
  };
  enum VTK_CELL_TYPE {
    POINT=1,
    LINE=3,
    TRIANGLE=5,
    TETRA=10,

    //2D
    PIXEL=8,
    QUAD=9,
    //3D
    VOXEL=11,
    HEX=12,

    POLYLINE=4,
    QUADRATIC_LINE=21,
  };
public:
  struct Data {
    Data():_nr(0) {}
    std::string _str;
    sizeType _nr;
  };
  template <typename ITER>
  struct ValueTraits {
    typedef typename ITER::value_type value_type;
  };
  template <typename POINTED_TYPE>
  struct ValueTraits<POINTED_TYPE*> {
    typedef POINTED_TYPE value_type;
  };
  template <typename ID>
  struct IteratorIndex {
    typedef ID value_type;
    IteratorIndex(const sizeType& id,const sizeType stride,const sizeType& off)
      :_id(id),_stride(stride),_off(off) {}
    void operator++() {
      _id++;
    }
    bool operator!=(const IteratorIndex& other) const {
      return _id < other._id;
    }
    virtual ID operator*() const {
      ID ret;
      for(sizeType i=0; i<ret.size(); i++)ret(i)=(_stride == 0) ? _id+_off*i : _id*_stride+i;
      return ret;
    }
    sizeType _id;
    sizeType _stride;
    sizeType _off;
  };
  template <typename ID>
  struct IteratorRepeat {
    typedef ID value_type;
    IteratorRepeat(const sizeType& id,const ID& val)
      :_id(id),_val(val) {}
    void operator++() {
      _id++;
    }
    bool operator!=(const IteratorRepeat& other) const {
      return _id < other._id;
    }
    virtual ID operator*() const {
      return _val;
    }
    sizeType _id;
    ID _val;
  };
  template <typename ITER>
  struct IteratorAdd {
    typedef typename ValueTraits<ITER>::value_type value_type;
    IteratorAdd(ITER beg0,ITER beg1):_beg0(beg0),_beg1(beg1) {}
    void operator++() {
      _beg0++;
      _beg1++;
    }
    bool operator!=(const IteratorAdd& other) const {
      return _beg0 != other._beg0;
    }
    virtual value_type operator*() const {
      return (*_beg0)+(*_beg1);
    }
    ITER _beg0,_beg1;
  };
  template <typename ITER,typename SCALAR>
  struct IteratorAddMult {
    typedef typename ValueTraits<ITER>::value_type value_type;
    IteratorAddMult(ITER beg0,ITER beg1,SCALAR mult):_beg0(beg0),_beg1(beg1),_mult(mult) {}
    void operator++() {
      _beg0++;
      _beg1++;
    }
    bool operator!=(const IteratorAddMult& other) const {
      return _beg0 != other._beg0;
    }
    virtual value_type operator*() const {
      return (*_beg0)+(*_beg1)*_mult;
    }
    ITER _beg0,_beg1;
    SCALAR _mult;
  };
public:
  VTKWriter(const std::string& name,const std::string& path,bool binary)
    :_os(path.c_str(),binary ? std::ios_base::binary : std::ios_base::out),
     _points(binary ? std::ios_base::binary : std::ios_base::out),
     _cells(binary ? std::ios_base::binary : std::ios_base::out),
     _cellTypes(binary ? std::ios_base::binary : std::ios_base::out),
     _cellDatas(binary ? std::ios_base::binary : std::ios_base::out),
     _nrPoint(0),_nrCell(0),_nrIndex(0),_nrData(0),_vdt(UNSTRUCTURED_GRID),
     _binary(binary) {
    _os << "# vtk DataFile Version 1.0" << std::endl;
    _os << name << std::endl;
    _os << (binary ? "BINARY" : "ASCII") << std::endl;
    _os << "DATASET " << "UNSTRUCTURED_GRID" << std::endl;
  }
  VTKWriter(const std::string& name,const std::string& path,bool binary,const BBox<T>& bb,const Vec3i& nrCell,bool center)
    :_os(path.c_str(),binary ? std::ios_base::binary : std::ios_base::out),
     _points(binary ? std::ios_base::binary : std::ios_base::out),
     _cells(binary ? std::ios_base::binary : std::ios_base::out),
     _cellTypes(binary ? std::ios_base::binary : std::ios_base::out),
     _cellDatas(binary ? std::ios_base::binary : std::ios_base::out),
     _nrPoint(0),_nrCell(0),_nrIndex(0),_nrData(0),_vdt(STRUCTURED_POINTS),
     _binary(binary) {
    typename BBox<T>::PT ext=bb.getExtent();
    typename BBox<T>::PT spacing(ext.x()/nrCell.x(),ext.y()/nrCell.y(),ext.z()/nrCell.z());
    _os << "# vtk DataFile Version 1.0" << std::endl;
    _os << name << std::endl;
    _os << (binary ? "BINARY" : "ASCII") << std::endl;
    _os << "DATASET " << "STRUCTURED_POINTS" << std::endl;
    if(center) {
      typename BBox<T>::PT origin=bb._minC+spacing*0.5f;
      _os << "DIMENSIONS " << nrCell.x() << " " << nrCell.y() << " " << nrCell.z() << std::endl;
      _os << "ORIGIN " << origin.x() << " " << origin.y() << " " << origin.z() << std::endl;
      _os << "SPACING " << spacing.x() << " " << spacing.y() << " " << spacing.z() << std::endl;
    } else {
      typename BBox<T>::PT origin=bb._minC;
      _os << "DIMENSIONS " << (nrCell.x()+1) << " " << (nrCell.y()+1) << " " << (nrCell.z()+1) << std::endl;
      _os << "ORIGIN " << origin.x() << " " << origin.y() << " " << origin.z() << std::endl;
      _os << "SPACING " << spacing.x() << " " << spacing.y() << " " << spacing.z() << std::endl;
    }
  }
  virtual ~VTKWriter() {
    bool first;
    switch(_vdt) {
    case UNSTRUCTURED_GRID:
      _os << "POINTS " << _nrPoint << " " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
      _os << _points.str();
      _os << "CELLS " << _nrCell << " " << _nrIndex << std::endl;
      _os << _cells.str();
      _os << "CELL_TYPES " << _nrCell << std::endl;
      _os << _cellTypes.str();
      first=false;
      for(typename std::unordered_map<std::string,Data>::const_iterator beg=_customData.begin(),end=_customData.end(); beg!=end; beg++) {
        if(!first)
          _os << "CELL_DATA " << beg->second._nr << std::endl;
        first=true;
        _os << beg->second._str << std::endl;
      }
      first=false;
      for(typename std::unordered_map<std::string,Data>::const_iterator beg=_customPointData.begin(),end=_customPointData.end(); beg!=end; beg++) {
        if(!first)
          _os << "POINT_DATA " << beg->second._nr << std::endl;
        first=true;
        _os << beg->second._str << std::endl;
      }
      break;
    case STRUCTURED_POINTS:
      ASSERT(_nrData == 1);
      _os << "POINT_DATA " << _nrPoint << std::endl;
      //write custom data first
      for(typename std::unordered_map<std::string,Data>::const_iterator beg=_customPointData.begin(),end=_customPointData.end(); beg!=end; beg++)
        _os << beg->second._str << std::endl;
      //final write default data
      _os << "SCALARS data " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
      _os << "LOOKUP_TABLE default" << std::endl;
      _os << _cellDatas.str();
      break;
    default:
      ASSERT_MSG(false,"Unsupported!")
    }
  }
  template <typename ITER> VTKWriter& appendPoints(ITER beg,ITER end) {
    typedef typename ValueTraits<ITER>::value_type value_type;
    sizeType nr=0;
    if(_binary) {
      for(; beg != end; ++beg) {
        const value_type& val=*beg;
        unsigned char sz=std::min<unsigned char>(3,val.size()),d=0;
        for(; d<sz; d++)
          vtkWrite(_points,(T)val(d));
        for(; d<3; d++)
          vtkWrite(_points,(T)0);
        nr++;
      }
    } else {
      for(; beg != end; ++beg) {
        const value_type& val=*beg;
        unsigned char sz=std::min<unsigned char>(3,val.size()),d=0;
        for(; d<sz; d++)
          _points << (T)val(d) << " ";
        for(; d<3; d++)
          _points << (T)0 << " ";
        _points << std::endl;
        nr++;
      }
    }
    _nrPoint+=nr;
    return *this;
  }
  template <typename ITER> VTKWriter& appendPixels(ITER beg,ITER end,bool quad,bool relativeIndex=false) {
    typedef typename Eigen::Matrix<sizeType,4,1> IDS;
    typedef typename ValueTraits<ITER>::value_type value_type;
    std::vector<value_type,Eigen::aligned_allocator<value_type> > points;
    std::vector<IDS,Eigen::aligned_allocator<IDS> > cells;
    for(; beg!=end;) {
      IDS ids;
      value_type minC=*beg++;
      value_type maxC=*beg++;
      value_type ext=maxC-minC;

      if(quad) ids << 0,1,3,2;
      else ids << 0,1,2,3;
      ids.array()+=points.size();
      cells.push_back(ids);

      points.push_back(minC+value_type(0.0f   ,0.0f   ,0.0f   ));
      points.push_back(minC+value_type(ext.x(),   0.0f,0.0f   ));
      points.push_back(minC+value_type(0.0f   ,ext.y(),0.0f   ));
      points.push_back(minC+value_type(ext.x(),ext.y(),0.0f   ));
    }
    setRelativeIndex();
    appendPoints(points.begin(),points.end());
    appendCells(cells.begin(),cells.end(),quad ? QUAD : PIXEL,relativeIndex);
    return *this;
  }
  template <typename ITER> VTKWriter& appendVoxels(ITER beg,ITER end,bool hex,bool relativeIndex=false) {
    typedef typename Eigen::Matrix<sizeType,8,1> IDS;
    typedef typename ValueTraits<ITER>::value_type value_type;
    std::vector<value_type,Eigen::aligned_allocator<value_type> > points;
    std::vector<IDS,Eigen::aligned_allocator<IDS> > cells;
    for(; beg!=end;) {
      IDS ids;
      value_type minC=*beg++;
      value_type maxC=*beg++;
      value_type ext=maxC-minC;

      if(hex) ids << 0,1,3,2,4,5,7,6;
      else ids << 0,1,2,3,4,5,6,7;
      ids.array()+=points.size();
      cells.push_back(ids);

      points.push_back(minC+value_type(0.0f   ,0.0f   ,0.0f   ));
      points.push_back(minC+value_type(ext.x(),   0.0f,0.0f   ));
      points.push_back(minC+value_type(0.0f   ,ext.y(),0.0f   ));
      points.push_back(minC+value_type(ext.x(),ext.y(),0.0f   ));

      points.push_back(minC+value_type(0.0f   ,0.0f   ,ext.z()));
      points.push_back(minC+value_type(ext.x(),   0.0f,ext.z()));
      points.push_back(minC+value_type(0.0f   ,ext.y(),ext.z()));
      points.push_back(minC+value_type(ext.x(),ext.y(),ext.z()));
    }
    setRelativeIndex();
    appendPoints(points.begin(),points.end());
    appendCells(cells.begin(),cells.end(),hex ? HEX : VOXEL,relativeIndex);
    return *this;
  }
  template <typename ITER> VTKWriter& appendCells(ITER beg,ITER end,VTK_CELL_TYPE ct,bool relativeIndex=false) {
    if(relativeIndex)
      ASSERT(_relativeCellIndex >= -1)
      int base=relativeIndex ? (int)_relativeCellIndex : 0;

    typedef typename ValueTraits<ITER>::value_type value_type;
    sizeType nr=0;
    sizeType nrIndex=0;
    if(_binary) {
      for(; beg != end; ++beg) {
        const value_type& val=*beg;
        switch(ct) {
        case POINT:
          nrIndex+=2;
          vtkWrite(_cells,1);
          vtkWrite(_cells,base+(int)val(0));
          break;
        case LINE:
          nrIndex+=3;
          vtkWrite(_cells,2);
          vtkWrite(_cells,base+(int)val(0));
          vtkWrite(_cells,base+(int)val(1));
          break;
        case TRIANGLE:
        case QUADRATIC_LINE:
          nrIndex+=4;
          vtkWrite(_cells,3);
          vtkWrite(_cells,base+(int)val(0));
          vtkWrite(_cells,base+(int)val(1));
          vtkWrite(_cells,base+(int)val(2));
          break;
        case TETRA:
        case PIXEL:
        case QUAD:
          nrIndex+=5;
          vtkWrite(_cells,4);
          vtkWrite(_cells,base+(int)val(0));
          vtkWrite(_cells,base+(int)val(1));
          vtkWrite(_cells,base+(int)val(2));
          vtkWrite(_cells,base+(int)val(3));
          break;
        case VOXEL:
        case HEX:
          nrIndex+=9;
          vtkWrite(_cells,8);
          vtkWrite(_cells,base+(int)val(0));
          vtkWrite(_cells,base+(int)val(1));
          vtkWrite(_cells,base+(int)val(2));
          vtkWrite(_cells,base+(int)val(3));
          vtkWrite(_cells,base+(int)val(4));
          vtkWrite(_cells,base+(int)val(5));
          vtkWrite(_cells,base+(int)val(6));
          vtkWrite(_cells,base+(int)val(7));
          break;
        case POLYLINE:
          nrIndex+=val.rows()+1;
          vtkWrite(_cells,(int)val.rows());
          for(sizeType i=0; i<(int)val.rows(); i++)
            vtkWrite(_cells,base+(int)val[i]);
          break;
        }
        vtkWrite(_cellTypes,(int)ct);
        nr++;
      }
    } else {
      for(; beg != end; ++beg) {
        const value_type& val=*beg;
        switch(ct) {
        case POINT:
          nrIndex+=2;
          _cells << "1 " << (base+(int)val(0)) << std::endl;
          break;
        case LINE:
          nrIndex+=3;
          _cells << "2 " << (base+(int)val(0)) << " " << (base+(int)val(1)) << std::endl;
          break;
        case TRIANGLE:
        case QUADRATIC_LINE:
          nrIndex+=4;
          _cells << "3 " << (base+(int)val(0)) << " " << (base+(int)val(1)) << " " << (base+(int)val(2)) << std::endl;
          break;
        case TETRA:
        case PIXEL:
        case QUAD:
          nrIndex+=5;
          _cells << "4 " << (base+(int)val(0)) << " " << (base+(int)val(1)) << " " << (base+(int)val(2)) << " " << (base+(int)val(3)) << std::endl;
          break;
        case VOXEL:
        case HEX:
          nrIndex+=9;
          _cells << "8 " << (base+(int)val(0)) << " " << (base+(int)val(1)) << " " << (base+(int)val(2)) << " " << (base+(int)val(3)) << " "
                 << (base+(int)val(4)) << " " << (base+(int)val(5)) << " " << (base+(int)val(6)) << " " << (base+(int)val(7)) << std::endl;
          break;
        case POLYLINE:
          nrIndex+=val.rows()+1;
          _cells << val.rows() << " ";
          for(sizeType i=0; i<(int)val.rows(); i++)
            _cells << (base+(int)val[i]) << " ";
          _cells << std::endl;
          break;
        }
        _cellTypes << ct << std::endl;
        nr++;
      }
    }
    _nrCell+=nr;
    _nrIndex+=nrIndex;
    return *this;
  }
  template <typename ITER> VTKWriter& appendDatas(const std::string name,ITER beg,ITER end) {
    if(_binary)
      for(; beg != end; ++beg,++_nrPoint)
        vtkWrite(_cellDatas,*beg);
    else
      for(; beg != end; ++beg,++_nrPoint)
        _cellDatas << (T)*beg << std::endl;
    _nrData++;
    return *this;
  }
  template <typename ITER> VTKWriter& appendCustomData(const std::string name,ITER beg,ITER end) {
    std::ostringstream os;
    if(_customData.find(name) == _customData.end()) {
      os << "SCALARS " << name << " " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
      os << "LOOKUP_TABLE default" << std::endl;
    }

    Data& dat=_customData[name];
    if(_binary)
      for(; beg != end; ++beg,++dat._nr)
        vtkWrite(os,(T)*beg);
    else
      for(; beg != end; ++beg,++dat._nr)
        os << (T)*beg << std::endl;
    dat._str+=os.str();
    ASSERT(dat._nr == _nrCell)
    return *this;
  }
  template <typename ITER> VTKWriter& appendCustomPointData(const std::string name,ITER beg,ITER end) {
    std::ostringstream os;
    if(_customPointData.find(name) == _customPointData.end()) {
      //Data& dat=_customPointData[name];
      os << "SCALARS " << name << " " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
      os << "LOOKUP_TABLE default" << std::endl;
    }

    Data& dat=_customPointData[name];
    if(_binary)
      for(; beg != end; ++beg,++dat._nr)
        vtkWrite(os,(T)*beg);
    else
      for(; beg != end; ++beg,++dat._nr)
        os << (T)*beg << std::endl;
    dat._str+=os.str();
    ASSERT(dat._nr == _nrPoint)
    return *this;
  }
  template <typename ITER> VTKWriter& appendCustomVectorData(const std::string name,ITER beg,ITER end) {
    std::ostringstream os;
    if(_customData.find(name) == _customData.end()) {
      os << "VECTORS " << name << " " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
    }

    Data& dat=_customData[name];
    if(_binary)
      for(; beg != end; ++beg,++dat._nr) {
        vtkWrite(os,(T)(*beg)[0]);
        vtkWrite(os,(T)(*beg)[1]);
        vtkWrite(os,(T)(*beg)[2]);
      }
    else
      for(; beg != end; ++beg,++dat._nr)
        os << (T)(*beg)[0] << " " << (T)(*beg)[1] << " " << (T)(*beg)[2] << std::endl;
    dat._str+=os.str();
    ASSERT(dat._nr == _nrCell)
    return *this;
  }
  template <typename ITER> VTKWriter& appendCustomPointVectorData(const std::string name,ITER beg,ITER end) {
    std::ostringstream os;
    if(_customPointData.find(name) == _customPointData.end()) {
      //Data& dat=_customPointData[name];
      os << "VECTORS " << name << " " << (sizeof(T) == sizeof(float) ? "float" : "double") << std::endl;
    }

    Data& dat=_customPointData[name];
    if(_binary)
      for(; beg != end; ++beg,++dat._nr) {
        vtkWrite(os,(T)(*beg)[0]);
        vtkWrite(os,(T)(*beg)[1]);
        vtkWrite(os,(T)(*beg)[2]);
      }
    else
      for(; beg != end; ++beg,++dat._nr)
        os << (T)(*beg)[0] << " " << (T)(*beg)[1] << " " << (T)(*beg)[2] << std::endl;
    dat._str+=os.str();
    ASSERT(dat._nr == _nrPoint)
    return *this;
  }
  template <typename ITER> VTKWriter& appendPointsByAdd(ITER beg0,ITER beg1,ITER end0) {
    appendPoints(IteratorAdd<ITER>(beg0,beg1),IteratorAdd<ITER>(end0,end0));
    return *this;
  }
  template <typename ITER> VTKWriter& appendCustomPointColorData(const std::string name,ITER begC,ITER endC) {
    std::ostringstream os;
    if(_customPointData.find(name) == _customPointData.end()) {
      //Data& dat=_customPointData[name];
      os << "COLOR_SCALARS " << name << " " << 4 << std::endl;
    }

    int sz=(int)(*begC).size();
    Data& dat=_customPointData[name];
    if(_binary) {
      for(; begC != endC; ++begC,++dat._nr) {
        for(int d=0; d<sz; d++)
          vtkWrite(os,(unsigned char)((*begC)[d]*255));
        for(int d=sz; d<4; d++)
          vtkWrite(os,(unsigned char)255);
      }
    } else {
      for(; begC != endC; ++begC,++dat._nr) {
        for(int d=0; d<sz; d++)
          os << (T)(*begC)[d] << " ";
        for(int d=sz; d<4; d++)
          os << (T)1 << " ";
        os << std::endl;
      }
    }
    dat._str+=os.str();
    return *this;
  }
  void setRelativeIndex(sizeType rel=-1) {
    if(rel == -1)
      _relativeCellIndex=_nrPoint;
    else _relativeCellIndex=rel;
  }
private:
  std::ofstream _os;
  std::ostringstream _points,_cells,_cellTypes,_cellDatas;
  std::unordered_map<std::string,Data> _customData,_customPointData;
  sizeType _nrPoint,_nrCell,_nrIndex,_nrData,_relativeCellIndex;
  VTK_DATA_TYPE _vdt;
  bool _binary;
};
}

#endif
