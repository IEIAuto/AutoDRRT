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

namespace llh_converter
{
double getDotNorm(Vector2d a, Vector2d b)
{
    return a.x * b.x + a.y * b.y;
}

double getCrossNorm(Vector2d a, Vector2d b)
{
    return a.x * b.y - a.y * b.x;
}

double getMeridianConvergence(const LLA &lla, const XYZ &xyz, llh_converter::LLHConverter &llhc,  const llh_converter::LLHParam &llhc_param)
{

    // This value has no special meaning.
    const double OFFSET_LATITUDE = 0.01; // 1.11 km
    const double OFFSET_XYZ_Y = 1000.0; // 1 km

    LLA offset_lla = lla;
    XYZ offset_xyz = xyz;

    XYZ xyz_by_offset_lla;

    offset_lla.latitude += OFFSET_LATITUDE; 
    offset_xyz.y += OFFSET_XYZ_Y;

    llhc.convertDeg2XYZ(offset_lla.latitude, offset_lla.longitude, offset_lla.altitude, xyz_by_offset_lla.x,
                         xyz_by_offset_lla.y, xyz_by_offset_lla.z, llhc_param);

    Vector2d offset_converted_vec;
    Vector2d xyz_by_offset_lla_converted_vec;

    offset_converted_vec.x = offset_xyz.x - xyz.x;
    offset_converted_vec.y = offset_xyz.y - xyz.y;
    xyz_by_offset_lla_converted_vec.x = xyz_by_offset_lla.x - xyz.x;
    xyz_by_offset_lla_converted_vec.y = xyz_by_offset_lla.y - xyz.y;

    double dot_norm = getDotNorm(offset_converted_vec, xyz_by_offset_lla_converted_vec);
    double cross_norm = getCrossNorm(offset_converted_vec, xyz_by_offset_lla_converted_vec);

    return std::atan2(cross_norm, dot_norm);
}

}  // namespace llh_converter