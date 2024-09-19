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

#include "llh_converter/gsigeo.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

namespace llh_converter
{
std::vector<std::string> split(std::string input, char delimiter)
{
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;

  while (std::getline(stream, field, delimiter))
  {
    if (field != std::string(" ") && !field.empty() && field != "\n" && field != "\r")
      result.push_back(field);
  }
  return result;
}

GSIGEO2011::GSIGEO2011()
{
}

GSIGEO2011::~GSIGEO2011()
{
}

void GSIGEO2011::loadGeoidMap(const std::string& geoid_file)
{
  std::ifstream ifs(geoid_file);
  if (!ifs)
  {
    std::cerr << "Error: Cannot open Geoid data file: " << geoid_file << std::endl;
    exit(2);
  }

  geoid_map_.resize(row_size_);
  for (int i = 0; i < row_size_; i++)
  {
    geoid_map_[i].resize(column_size_);
  }

  std::string curr_line;
  // Skip header line
  std::getline(ifs, curr_line);

  int i = 0, j = 0;

  while (std::getline(ifs, curr_line))
  {
    std::vector<std::string> str_vec = split(curr_line, ' ');

    for (int k = 0; k < str_vec.size(); k++)
    {
      geoid_map_[i][j] = std::stod(str_vec[k]);
      j++;
    }
    if (j == column_size_)
    {
      j = 0;
      i++;
    }
  }
  is_geoid_loaded_ = true;
}

double GSIGEO2011::getGeoid(const double& lat, const double& lon)
{
  if (!is_geoid_loaded_)
  {
    std::cerr << "Error: Geoid map is not loaded" << std::endl;
    exit(1);
  }
  const double lat_min = 20;
  const double lon_min = 120;
  const double d_lat = 1.0 / 60.0;
  const double d_lon = 1.5 / 60.0;

  const int i_lat = std::floor((lat - lat_min) / d_lat);
  const int i_lon = std::floor((lon - lon_min) / d_lon);
  const int j_lat = i_lat + 1;
  const int j_lon = i_lon + 1;

  // const double t = (((lat - lat_min) / d_lat) - i_lat) / d_lat;
  // const double u = (((lon - lon_min) / d_lon) - i_lon) / d_lon;
  const double t = (lat - (lat_min + i_lat * d_lat)) / d_lat;
  const double u = (lon - (lon_min + i_lon * d_lon)) / d_lon;

  if (i_lat < 0 || i_lat >= row_size_ - 1 || i_lon < 0 || i_lon >= column_size_ - 1)
  {
    std::cerr << "Error: latitude/longitude is out of range (20~50, 120~150)" << std::endl;
    std::cerr << (lat, lon) << std::endl;
    exit(1);
  }

  if (geoid_map_[i_lat][i_lon] == 999 || geoid_map_[i_lat][j_lon] == 999 || geoid_map_[j_lat][i_lon] == 999 ||
      geoid_map_[j_lat][j_lon] == 999)
  {
    std::cerr << "Error: Not supported area" << std::endl;
    exit(1);
  }

  double geoid = (1 - t) * (1 - u) * geoid_map_[i_lat][i_lon] + (1 - t) * u * geoid_map_[i_lat][j_lon] +
                 t * (1 - u) * geoid_map_[j_lat][i_lon] + t * u * geoid_map_[j_lat][j_lon];

  return geoid;
}
}  // namespace llh_converter
