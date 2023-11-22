// Copyright 2022 Takagi, Isamu
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

#include "conditions.hpp"
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace
{

template <class T>
using ExprType = std::vector<std::function<bool(T)>>;
using UintType = uint64_t;
using SintType = int64_t;
using RealType = double;
using TextType = std::string;
using UintExpr = ExprType<UintType>;
using SintExpr = ExprType<SintType>;
using RealExpr = ExprType<RealType>;
using TextExpr = ExprType<TextType>;
using Expr = std::variant<UintExpr, SintExpr, RealExpr, TextExpr>;

template <class T, template <class> class F>
void init_term(ExprType<T> & expr, const YAML::Node & yaml, const std::string & op)
{
  if (yaml[op])
  {
    const auto operand = yaml[op].as<T>();
    expr.push_back(std::bind(F<T>(), std::placeholders::_1, operand));
  }
}

template <class T>
void init_expr(ExprType<T> & expr, const YAML::Node & yaml)
{
  // clang-format off
  init_term<T, std::equal_to     >(expr, yaml, "eq");
  init_term<T, std::not_equal_to >(expr, yaml, "ne");
  init_term<T, std::less         >(expr, yaml, "lt");
  init_term<T, std::less_equal   >(expr, yaml, "le");
  init_term<T, std::greater      >(expr, yaml, "gt");
  init_term<T, std::greater_equal>(expr, yaml, "ge");
  // clang-format on
}

template <class T>
bool eval_expr(const ExprType<T> & expr, const T & value)
{
  for (const auto & expr : expr)
  {
    if (!expr(value)) return false;
  }
  return true;
}

YAML::Node take_expr_yaml(YAML::Node & yaml)
{
  YAML::Node result;
  for (const auto & op : {"eq", "ne", "lt", "le", "gt", "ge"})
  {
    if (yaml[op])
    {
      result[op] = yaml[op];
      yaml.remove(op);
    }
  }
  return result;
}

Expr make_expr(const std::string & type)
{
  if (type == "uint") return UintExpr();
  if (type == "sint") return SintExpr();
  if (type == "real") return RealExpr();
  if (type == "text") return TextExpr();
  throw std::invalid_argument("unknown condition type: " + type);
};

class InitExpr
{
public:
  explicit InitExpr(const YAML::Node & yaml) : yaml_(yaml) {}
  void operator()(UintExpr & expr) { return init_expr(expr, yaml_); }
  void operator()(SintExpr & expr) { return init_expr(expr, yaml_); }
  void operator()(RealExpr & expr) { return init_expr(expr, yaml_); }
  void operator()(TextExpr & expr) { return init_expr(expr, yaml_); }

private:
  YAML::Node yaml_;
};

class EvalExpr
{
public:
  explicit EvalExpr(const YAML::Node & yaml) : yaml_(yaml) {}
  bool operator()(const UintExpr & expr) { return eval_expr(expr, yaml_.as<UintType>()); }
  bool operator()(const SintExpr & expr) { return eval_expr(expr, yaml_.as<SintType>()); }
  bool operator()(const RealExpr & expr) { return eval_expr(expr, yaml_.as<RealType>()); }
  bool operator()(const TextExpr & expr) { return eval_expr(expr, yaml_.as<TextType>()); }

private:
  YAML::Node yaml_;
};

}  // namespace

namespace multi_data_monitor
{

struct Condition::Impl
{
  Expr expr;
};

Condition::Condition(const std::string & type, YAML::Node & yaml)
{
  YAML::Node ops = take_expr_yaml(yaml);
  if (!ops.IsNull())
  {
    impl_ = std::make_unique<Impl>();
    impl_->expr = make_expr(type);
    std::visit(InitExpr(ops), impl_->expr);
  }
}

Condition::~Condition()
{
}

bool Condition::eval(const YAML::Node & yaml) const
{
  return impl_ ? std::visit(EvalExpr(yaml), impl_->expr) : true;
}

}  // namespace multi_data_monitor
