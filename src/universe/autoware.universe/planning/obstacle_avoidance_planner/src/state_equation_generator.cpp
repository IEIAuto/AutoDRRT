// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "obstacle_avoidance_planner/state_equation_generator.hpp"

#include "obstacle_avoidance_planner/mpt_optimizer.hpp"

namespace obstacle_avoidance_planner
{
// state equation: x = B u + W (u includes x_0)
// NOTE: Originally, x_t+1 = Ad x_t + Bd u + Wd.
StateEquationGenerator::Matrix StateEquationGenerator::calcMatrix(
  const std::vector<ReferencePoint> & ref_points) const
{
  time_keeper_ptr_->tic(__func__);

  const size_t N_ref = ref_points.size();
  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t D_v = D_x + D_u * (N_ref - 1);

  // matrices for whole state equation
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(D_x * N_ref, D_v);
  Eigen::VectorXd W = Eigen::VectorXd::Zero(D_x * N_ref);

  // matrices for one-step state equation
  Eigen::MatrixXd Ad(D_x, D_x);
  Eigen::MatrixXd Bd(D_x, D_u);
  Eigen::MatrixXd Wd(D_x, 1);

  // calculate one-step state equation considering kinematics N_ref times
  for (size_t i = 0; i < N_ref; ++i) {
    if (i == 0) {
      B.block(0, 0, D_x, D_x) = Eigen::MatrixXd::Identity(D_x, D_x);
      continue;
    }

    const int idx_x_i = i * D_x;
    const int idx_x_i_prev = (i - 1) * D_x;
    const int idx_u_i_prev = (i - 1) * D_u;

    // get discrete kinematics matrix A, B, W
    const auto & p = ref_points.at(i - 1);

    // TODO(murooka) use curvature by stabling optimization
    // Currently, when using curvature, the optimization result is weird with sample_map.
    // vehicle_model_ptr_->calculateStateEquationMatrix(Ad, Bd, Wd, p.curvature,
    // p.delta_arc_length);
    vehicle_model_ptr_->calculateStateEquationMatrix(Ad, Bd, Wd, 0.0, p.delta_arc_length);

    B.block(idx_x_i, 0, D_x, D_x) = Ad * B.block(idx_x_i_prev, 0, D_x, D_x);
    B.block(idx_x_i, D_x + idx_u_i_prev, D_x, D_u) = Bd;

    for (size_t j = 0; j < i - 1; ++j) {
      size_t idx_u_j = j * D_u;
      B.block(idx_x_i, D_x + idx_u_j, D_x, D_u) =
        Ad * B.block(idx_x_i_prev, D_x + idx_u_j, D_x, D_u);
    }

    W.segment(idx_x_i, D_x) = Ad * W.block(idx_x_i_prev, 0, D_x, 1) + Wd;
  }

  Matrix mat;
  mat.B = B;
  mat.W = W;

  time_keeper_ptr_->toc(__func__, "        ");
  return mat;
}

Eigen::VectorXd StateEquationGenerator::predict(
  const StateEquationGenerator::Matrix & mat, const Eigen::VectorXd U) const
{
  return mat.B * U + mat.W;
}
}  // namespace obstacle_avoidance_planner
