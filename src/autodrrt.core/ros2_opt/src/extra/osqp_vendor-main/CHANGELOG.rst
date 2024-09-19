^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package osqp_vendor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2022-10-23)
------------------
* feat: enable to build with both ros1 and ros2 (`#16 <https://github.com/tier4/osqp_vendor/issues/16>`_)
  * feat: enable to build with both ros1 and ros2
  * fix github action
* Contributors: Daisuke Nishimatsu

0.0.4 (2021-12-07)
------------------
* Suppress update of pinned git repository (`#8 <https://github.com/tier4/osqp_vendor/issues/8>`_)
  The source at that ref will never change, so there is no need to try to
  update it. This will avoid unnecessary invalidation of the build and
  install targets for the external project.
* Always preserve source permissions in vendor packages (`#7 <https://github.com/tier4/osqp_vendor/issues/7>`_)
  In vendor packages where we're installing an executable, we use
  USE_SOURCE_PERMISSIONS to make sure that the executable permissions on
  the binaries are maintained when the external project's staging
  directory is recursively installed to the final installation directory.
  In most of our vendor packages, we aren't using that flag where we don't
  expect an executable binary to be installed. However, for reasons I
  won't go into here, some systems use executable permissions on shared
  object libraries as well. The linker seems to handle this on our behalf,
  but we're losing the permissions during the recursive copy operation if
  we don't use this flag.
* Add buildtool_depend on git (`#6 <https://github.com/tier4/osqp_vendor/issues/6>`_)
* Update workflows (`#9 <https://github.com/tier4/osqp_vendor/issues/9>`_)
  * Update workflows
  * Remove artifact
  * Skip tests
* Contributors: Daisuke Nishimatsu, Scott K Logan

0.0.3 (2021-04-13)
------------------
* Release 0.0.3
* Merge pull request `#5 <https://github.com/tier4/osqp_vendor/issues/5>`_ from tier4/update-0.6.2
  Bump version to 0.6.2
* Bump version to 0.6.2
* Contributors: Esteve Fernandez

0.0.2 (2020-11-04)
------------------
* Release 0.0.2
* Added missing CONTRIBUTING.md
* Contributors: Esteve Fernandez

0.0.1 (2020-10-19)
------------------
* Update README.md
* Removed unused BUILD_SHARED_LIBS=ON, not needed since osqp builds both shared and static libraries
* Merge pull request `#1 <https://github.com/tier4/osqp_vendor/issues/1>`_ from tier4/vendor
  Added OSQP vendor package
* Added OSQP vendor package
* Initial commit
* Contributors: Esteve Fernandez
