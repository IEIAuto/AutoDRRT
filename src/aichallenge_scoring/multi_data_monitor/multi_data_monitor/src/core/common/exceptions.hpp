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

#ifndef CORE__COMMON__EXCEPTIONS_HPP_
#define CORE__COMMON__EXCEPTIONS_HPP_

#include <stdexcept>
#include <string>

namespace multi_data_monitor
{

struct LogicError : public std::logic_error
{
  using std::logic_error::logic_error;
};

struct RuntimeError : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

struct PluginError : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

struct ConfigError : public std::runtime_error
{
  using std::runtime_error::runtime_error;

  static ConfigError LabelConflict(const std::string & label, const std::string & scope)
  {
    return ConfigError(scope + " label '" + label + "' is not unique");
  }
  static ConfigError LabelNotFound(const std::string & label, const std::string & scope)
  {
    return ConfigError(scope + " label '" + label + "' is not found");
  }
};

struct FilePathError : public ConfigError
{
  using ConfigError::ConfigError;
};

struct LabelCirculation : public ConfigError
{
  using ConfigError::ConfigError;
};

struct GraphCirculation : public ConfigError
{
  using ConfigError::ConfigError;
};

struct GraphIsNotTree : public ConfigError
{
  using ConfigError::ConfigError;
};

}  // namespace multi_data_monitor

#endif  // CORE__COMMON__EXCEPTIONS_HPP_
