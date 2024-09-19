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

#ifndef M4_HEIGHT_CONVERTER_HPP
#define M4_HEIGHT_CONVERTER_HPP

#include <string>

#include <GeographicLib/Geoid.hpp>

#include "llh_converter/gsigeo.hpp"

namespace llh_converter
{
enum class GeoidType
{
  EGM2008 = 0,
  GSIGEO2011 = 1,
};

enum ConvertType
{
  ELLIPS2ORTHO = -1,
  NONE = 0,
  ORTHO2ELLIPS = 1,
};

class HeightConverter
{
public:
  double convertHeightRad(const double& lat_rad, const double& lon_rad, const double& h, const ConvertType& type);
  double convertHeightDeg(const double& lat_deg, const double& lon_deg, const double& h, const ConvertType& type);
  void setGeoidType(const GeoidType& geoid_type);
  double getGeoidRad(const double& lat_rad, const double& lon_rad);
  double getGeoidDeg(const double& lat_deg, const double& lon_deg);
  void loadGSIGEOGeoidFile(const std::string& geoid_file);
  void loadGSIGEOGeoidFile();

private:
  // Geoid Type
  GeoidType geoid_type_ = GeoidType::EGM2008;
  // Flag
  bool is_gsigeo_loaded_ = false;

  // Geoid maps
  GeographicLib::Geoid egm2008_{ "egm2008-1" };
  GSIGEO2011 gsigeo2011_;

  double getGeoidEGM2008(const double& lat_rad, const double& lon_rad);
  double getGeoidGSIGEO2011(const double& lat_rad, const double& lon_rad);
};
}  // namespace llh_converter

#endif
