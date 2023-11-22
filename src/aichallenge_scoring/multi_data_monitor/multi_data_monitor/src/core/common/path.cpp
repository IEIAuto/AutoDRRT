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

#include "path.hpp"
#include "common/exceptions.hpp"
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <string>

namespace multi_data_monitor::path
{

bool starts_with(const std::string & text, const std::string & pattern)
{
  return text.find(pattern) == 0;
}

const std::string remove_scheme(const std::string & path, const std::string & scheme)
{
  return starts_with(path, scheme) ? path.substr(scheme.size()) : std::string();
}

const std::string resolve(const std::string & path)
{
  // file scheme
  {
    const auto body = remove_scheme(path, "file://");
    if (!body.empty())
    {
      return body;
    }
  }

  // package scheme
  try
  {
    const auto body = remove_scheme(path, "package://");
    if (!body.empty())
    {
      const auto pos = body.find("/");
      const auto package_name = body.substr(0, pos);
      const auto package_file = body.substr(pos);
      const auto package_path = ament_index_cpp::get_package_share_directory(package_name);
      return package_path + package_file;
    }
  }
  catch (const ament_index_cpp::PackageNotFoundError & error)
  {
    throw FilePathError("package not found '" + path + "'");
  }

  // unknown scheme
  throw FilePathError("invalid path scheme '" + path + "'");
}

}  // namespace multi_data_monitor::path
