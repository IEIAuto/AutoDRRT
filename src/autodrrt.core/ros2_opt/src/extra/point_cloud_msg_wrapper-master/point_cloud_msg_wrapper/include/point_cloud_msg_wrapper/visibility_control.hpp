// Copyright 2021 Apex.AI, Inc.
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

/// \copyright Copyright 2021 Apex.AI, Inc.
/// All rights reserved.


#ifndef POINT_CLOUD_MSG_WRAPPER__VISIBILITY_CONTROL_HPP_
#define POINT_CLOUD_MSG_WRAPPER__VISIBILITY_CONTROL_HPP_

#if defined(_MSC_VER) && defined(_WIN64)
  #if defined(POINT_CLOUD_MSG_WRAPPER_BUILDING_DLL) || defined(POINT_CLOUD_MSG_WRAPPER_EXPORTS)
    #define POINT_CLOUD_MSG_WRAPPER_PUBLIC __declspec(dllexport)
    #define POINT_CLOUD_MSG_WRAPPER_LOCAL
  #else
    #define POINT_CLOUD_MSG_WRAPPER_PUBLIC __declspec(dllimport)
    #define POINT_CLOUD_MSG_WRAPPER_LOCAL
  #endif
#elif defined(__GNUC__) && defined(__linux__)
  #define POINT_CLOUD_MSG_WRAPPER_PUBLIC __attribute__((visibility("default")))
  #define POINT_CLOUD_MSG_WRAPPER_LOCAL __attribute__((visibility("hidden")))
#elif defined(__GNUC__) && defined(__APPLE__)
  #define POINT_CLOUD_MSG_WRAPPER_PUBLIC __attribute__((visibility("default")))
  #define POINT_CLOUD_MSG_WRAPPER_LOCAL __attribute__((visibility("hidden")))
#else  // !(defined(__GNUC__) && defined(__APPLE__))
  #error "Unsupported Build Configuration"
#endif  // _MSC_VER

#endif  // POINT_CLOUD_MSG_WRAPPER__VISIBILITY_CONTROL_HPP_
