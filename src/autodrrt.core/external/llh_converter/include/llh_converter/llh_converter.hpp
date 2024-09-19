// BSD 3-Clause License
//
// Copyright (c) 2022, Map IV, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef LLH_CONVERTER_HPP
#define LLH_CONVERTER_HPP

#include <string>
#include <map>

#include "llh_converter/height_converter.hpp"

namespace llh_converter
{
enum class MGRSPrecision
{
  KILO_METER_10 = 1,
  KILO_METER_1 = 2,
  METER_100 = 3,
  METER_10 = 4,
  METER_1 = 5,
  MILLI_METER_100 = 6,
  MILLI_METER_10 = 7,
  MILLI_METER_1 = 8,
  MICRO_METER_100 = 9,
};

struct LLHParam
{
  bool use_mgrs;
  int plane_num;
  std::string mgrs_code;
  // MGRSPrecision precision;
  ConvertType height_convert_type;
  GeoidType geoid_type;
};

class LLHConverter
{
public:
  LLHConverter();
  LLHConverter(const std::string& geoid_file);

  void convertDeg2XYZ(const double& lat_deg, const double& lon_deg, const double& h, double& x, double& y, double& z,
                      const LLHParam& param);
  void convertRad2XYZ(const double& lat_rad, const double& lon_rad, const double& h, double& x, double& y, double& z,
                      const LLHParam& param);
  void revertXYZ2Deg(const double& x, const double& y, double& lat_deg, double& lon_deg, const LLHParam& param);
  void revertXYZ2Rad(const double& x, const double& y, double& lat_rad, double& lon_rad, const LLHParam& param);

  void convertMGRS2JPRCS(const double& m_x, const double& m_y, double& j_x, double& j_y, const LLHParam& param);
  void convertJPRCS2MGRS(const double& j_x, const double& j_y, double& m_x, double& m_y, const LLHParam& param);

  void getMapOriginDeg(double& lat_rad, double& lon_rad, const LLHParam& param);
  void getMapOriginRad(double& lat_rad, double& lon_rad, const LLHParam& param);

  std::string getMGRSGridCode()
  {
    return mgrs_code_;
  }
  void setMGRSGridCode(const std::string& mgrs_code)
  {
    mgrs_code_ = mgrs_code;
  }

private:
  double plane_lat_rad_, plane_lon_rad_;
  std::string mgrs_code_;
  std::string origin_x_zone_, origin_y_zone_;

  bool use_origin_zone_ = true;
  bool is_origin_set_ = false;

  // Constant param
  const double F_ = 298.257222101;
  const double a_ = 6378137;
  const double m0_ = 0.9999;
  const int grid_code_size_ = 5;
  MGRSPrecision precision_ = MGRSPrecision::MICRO_METER_100;

  // Object
  HeightConverter height_converter_;

  // Functions
  // Japan Plane Rectangular Coordinate System
  void convRad2JPRCS(const double& lat_rad, const double& lon_rad, double& x, double& y);
  void revJPRCS2Rad(const double& x, const double& y, double& lat_rad, double& lon_rad);
  // MGRS
  void convRad2MGRS(const double& lat_rad, const double& lon_rad, double& x, double& y);
  void revMGRS2Rad(const double& x, const double& y, double& lat_rad, double& lon_rad);

  int checkCrossBoader(const std::string& code_origin, const std::string& code_current, bool is_x);

  void setPlaneNum(int plane_num);

  std::map<std::string, int> mgrs_alphabet_{ { "A", 0 },  { "B", 1 },  { "C", 2 },  { "D", 3 },  { "E", 4 },
                                             { "F", 5 },  { "G", 6 },  { "H", 7 },  { "J", 8 },  { "K", 9 },
                                             { "L", 10 }, { "M", 11 }, { "N", 12 }, { "P", 13 }, { "Q", 14 },
                                             { "R", 15 }, { "S", 16 }, { "T", 17 }, { "U", 18 }, { "V", 19 },
                                             { "W", 20 }, { "X", 21 }, { "Y", 22 }, { "Z", 23 } };
};
}  // namespace llh_converter

#endif
