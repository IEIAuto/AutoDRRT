^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package tvm_vendor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.9.1 (2022-11-10)
------------------
* opencl dependency
* add opencl to compilation
* Contributors: Xinyu Wang

0.9.0 (2022-08-22)
------------------
* feat: update tvm version to v0.9.0 (`#11 <https://github.com/autowarefoundation/tvm_vendor/issues/11>`_)
  * feat: update tvm version to v0.9.0
  * ci: fix ros1 ci
* Contributors: Daisuke Nishimatsu

0.8.2 (2022-06-06)
------------------
* fix: modify patch issue (`#10 <https://github.com/autowarefoundation/tvm_vendor/issues/10>`_)
  * fix: update patch
  * fix: add workaround to avoid multiple patch apply
* Contributors: Daisuke Nishimatsu

0.8.1 (2022-05-25)
------------------
* feat: support jammy (`#9 <https://github.com/autowarefoundation/tvm_vendor/issues/9>`_)
  * fix CMakeLists.txt to build on jammy
  * ci: update ros2 ci
* Contributors: Daisuke Nishimatsu

0.8.0 (2022-02-07)
------------------
* Update TVM version (`#7 <https://github.com/autowarefoundation/tvm_vendor/issues/7>`_)
  This patch is needed to properly run the packages that
  use ML models from the modelzoo.
  Issue-Id: SCM-3720
  Change-Id: I331efb40337ac15ec6b84ed9b9050fa42e49dc0f
* CI: Add Galactic CI
* Contributors: Joshua Whitley, Luca Foschiani

0.7.3 (2021-06-08)
------------------
* Merge pull request `#6 <https://github.com/autowarefoundation/tvm_vendor/issues/6>`_ from kurcha01-arm/main
  Fix 'No source or binary directory provided'
* Merge pull request `#5 <https://github.com/autowarefoundation/tvm_vendor/issues/5>`_ from kurcha01-arm/main
  Fix header inclusion, export without namespacing
* Merge pull request `#4 <https://github.com/autowarefoundation/tvm_vendor/issues/4>`_ from kurcha01-arm/main
  Fix dlpack and tvm-runtime header inclusion
* Merge pull request `#3 <https://github.com/autowarefoundation/tvm_vendor/issues/3>`_ from ambroise-arm/vulkan-backend
  Add Vulkan backend
* Contributors: Ambroise Vincent, Joshua Whitley, Kurtis Charnock

0.7.2 (2020-12-09)
------------------
* Removing LLVM dependency.
* Contributors: Joshua Whitley

0.7.1 (2020-12-01)
------------------
* Updating LLVM dependency.
* Moving catkin_package() back to after add_library().
* Contributors: Joshua Whitley

0.7.0 (2020-11-16)
------------------
* Initial release.
* Contributors: Joshua Whitley
