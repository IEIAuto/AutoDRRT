// BSD 3-Clause License
//
// Copyright (c) 2023, Map IV, Inc.
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

#include "llh_converter/meridian_convergence_angle_correction.hpp"
#include "llh_converter/llh_converter.hpp"

#include <iostream>
#include <iomanip>

void test(const double& result, const double& answer)
{
  double diff = std::abs(result - answer);
  double diff_theres = 0.0001;
  if (diff < diff_theres)
  {
    std::cout << "\033[32;1mTEST SUCCESS: |(" << result << ") - (" <<  answer << ")| = " << diff << "< " << diff_theres << "\033[m" << std::endl;
  }
  else
  {
    std::cout << "\033[31;1mTEST FAILED: |(" << result << ") - (" <<  answer << ")| = " << diff << ">= " << diff_theres << "\033[m" << std::endl;
  }
}

void meridian_convergence_angle_correction_test(const double& test_lat, const double& test_lon, const double& answered_angle,
  llh_converter::LLHConverter &llh_converter,  const llh_converter::LLHParam &param
)
{
  llh_converter::LLA lla;
  llh_converter::XYZ xyz;
  lla.latitude = test_lat;
  lla.longitude = test_lon;
  lla.altitude = 30.0;
  // convert lla to xyz
  llh_converter.convertDeg2XYZ(lla.latitude, lla.longitude, lla.altitude, xyz.x, xyz.y, xyz.z, param);

  // get meridian convergence angle
  double mca = llh_converter::getMeridianConvergence(lla, xyz, llh_converter, param);

  std::cout << "-------------------------------------------------------------------------------------" << std::endl;
  std::cout << "Testing LatLon (" << std::setw(6) << test_lat << ", " << std::setw(6) << test_lat << ") ... " << std::endl;
  std::cout << "Calcalated Meridian Convergence Angle (" << mca << ")" << std::endl;
  test(llh_converter::rad2deg(mca), answered_angle);

}

int main(int argc, char** argv)
{
  // Meridian Convergence Angle Correction Test
  llh_converter::LLHConverter llh_converter;
  llh_converter::LLHParam param;
  param.use_mgrs = false;
  param.plane_num = 7;
  param.height_convert_type = llh_converter::ConvertType::NONE;
  param.geoid_type = llh_converter::GeoidType::EGM2008;

  // ref: Conversion to plane rectangular coordinates(in Japanse)
  // https://vldb.gsi.go.jp/sokuchi/surveycalc/surveycalc/bl2xyf.html
  // nagoya city ueda
  double test_lat = 35.141168610, test_lon = 136.989591759;
  double answered_angle = -0.101925000; // [deg]
  meridian_convergence_angle_correction_test(test_lat, test_lon, answered_angle, llh_converter, param);
  // nagoya city ozone
  double test_lat2 = 35.188843433, test_lon2 = 136.943096063;
  double answered_angle2 = -0.128838889; // [deg]
  meridian_convergence_angle_correction_test(test_lat2, test_lon2, answered_angle2, llh_converter, param);

  return 0;
}