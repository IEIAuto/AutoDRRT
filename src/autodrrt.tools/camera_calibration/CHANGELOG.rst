^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package camera_calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.0.6 (2024-08-20)
------------------
* [backport] use correct synchronous service call (`#1006 <https://github.com/ros-perception/image_pipeline/issues/1006>`_)
  Backport from https://github.com/ros-perception/image_pipeline/pull/792
  to fix https://github.com/ros-perception/image_pipeline/pull/838
  Co-authored-by: Christian Rauch <Rauch.Christian@gmx.de>
* Contributors: Balint Rozgonyi

3.0.5 (2024-07-24)
------------------
* Change camera info message to lower case (backport `#1005 <https://github.com/ros-perception/image_pipeline/issues/1005>`_) (`#1008 <https://github.com/ros-perception/image_pipeline/issues/1008>`_)
  Change camera info message to lower case since message type had been
  change in rolling and humble.
  [](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg)<hr>This
  is an automatic backport of pull request `#1005 <https://github.com/ros-perception/image_pipeline/issues/1005>`_ done by
  [Mergify](https://mergify.com).
  ---------
  Co-authored-by: SFhmichael <146928033+SFhmichael@users.noreply.github.com>
  Co-authored-by: Alejandro Hernández Cordero <ahcorde@gmail.com>
* [backport humble] Fix spelling error for cv2.aruco.DICT from 6x6_50 to 7x7_1000 (`#961 <https://github.com/ros-perception/image_pipeline/issues/961>`_) (`#1004 <https://github.com/ros-perception/image_pipeline/issues/1004>`_)
  Backport `#961 <https://github.com/ros-perception/image_pipeline/issues/961>`_ fix to humble
  Co-authored-by: Vishal Balaji <vishalvichu45@gmail.com>
  Co-authored-by: Vishal Balaji <vishal.balaji@schanzer-racing.de>
* Contributors: Balint Rozgonyi, mergify[bot]

3.0.4 (2024-03-01)
------------------

3.0.3 (2022-01-24)
------------------
* [backport humble] ROS 2: Added more aruco dicts, fixed aruco linerror bug (`#873 <https://github.com/ros-perception/image_pipeline/issues/873>`_) (`#889 <https://github.com/ros-perception/image_pipeline/issues/889>`_)
  backport `#873 <https://github.com/ros-perception/image_pipeline/issues/873>`_
* [backport humble] ROS 2: Fixing thrown Exception in camerachecker.py (`#871 <https://github.com/ros-perception/image_pipeline/issues/871>`_) (`#887 <https://github.com/ros-perception/image_pipeline/issues/887>`_)
  backport `#871 <https://github.com/ros-perception/image_pipeline/issues/871>`_
* Contributors: Alejandro Hernández Cordero

3.0.2 (2022-01-17)
------------------
* Fix QoS incompatibility camera_calibration ROS2
* Contributors: Michal Wojcik

3.0.0 (2022-04-29)
------------------
* Some small fixes noticed while reviewing.
* fix premature camera model change in camera_calibration
* Fix shebang lines for noetic python3
* Update fisheye distortion model definition
* Fix calibration yaml formatting (`#580 <https://github.com/ros-perception/image_pipeline/issues/580>`_) (`#585 <https://github.com/ros-perception/image_pipeline/issues/585>`_)
* updated linear_error function to handle partial board views (`#561 <https://github.com/ros-perception/image_pipeline/issues/561>`_)
* Fix missing detected checkerboard points (`#558 <https://github.com/ros-perception/image_pipeline/issues/558>`_)
* ChArUco board, Noetic (`#549 <https://github.com/ros-perception/image_pipeline/issues/549>`_)
* fix `#503 <https://github.com/ros-perception/image_pipeline/issues/503>`_: (`#545 <https://github.com/ros-perception/image_pipeline/issues/545>`_)
* Minimal Noetic (`#530 <https://github.com/ros-perception/image_pipeline/issues/530>`_)
* Apply `#509 <https://github.com/ros-perception/image_pipeline/issues/509>`_ and `#526 <https://github.com/ros-perception/image_pipeline/issues/526>`_ to Noetic Branch (`#528 <https://github.com/ros-perception/image_pipeline/issues/528>`_)
* Add Fisheye calibration tool (`#440 <https://github.com/ros-perception/image_pipeline/issues/440>`_)
* camera_calibration: Improve YAML formatting, make config dumping methods static (`#438 <https://github.com/ros-perception/image_pipeline/issues/438>`_)
* camera_calibration: Fix all-zero distortion coeffs returned for a rational_polynomial model (`#433 <https://github.com/ros-perception/image_pipeline/issues/433>`_)
* Make sure 'calibrate' button works even if not receiving images anymore
* Add a comment
* Replace deque with a modified Queue, add --queue-size param
* Remove print statement
* Cosmetic changes
* Add max-chessboard-speed option to allow more accurate calibration of rolling shutter cameras.
* revert back
* added missing imports
* update pytest.ini
* fixes to pass tests
* rebase change
* implemented fisheye mono and stereo calibration based on the melodic branch
* trimmed whitespace at line endings
* Update camera_calibration setup.cfg to use underscores (`#688 <https://github.com/ros-perception/image_pipeline/issues/688>`_)
* Add maintainer (`#667 <https://github.com/ros-perception/image_pipeline/issues/667>`_)
* Fixed crash when rosargs are given (`#597 <https://github.com/ros-perception/image_pipeline/issues/597>`_)
* Contributors: Chris Lalancette, David Torres Ocaña, DavidTorresOcana, Gabor Soros, Jacob Perron, John Stechschulte, Joshua Whitley, Martin Valgur, Matthijs den Toom, Michael Carroll, Patrick Musau, Photon, Spiros Evangelatos, Victor Dubois, jaiveersinghNV, soeroesg

2.2.1 (2020-08-27)
------------------
* remove email blasts from steve macenski (`#596 <https://github.com/ros-perception/image_pipeline/issues/596>`_)
* Add pytest.ini to fix warning (`#584 <https://github.com/ros-perception/image_pipeline/issues/584>`_)
  Fixes the following warning:
  Warning: The 'junit_family' default value will change to 'xunit2' in pytest 6.0.
  Add 'junit_family=xunit1' to your pytest.ini file to keep the current format in future versions of pytest and silence this warning.
* [Foxy] Use ament_auto Macros (`#573 <https://github.com/ros-perception/image_pipeline/issues/573>`_)
* Contributors: Jacob Perron, Joshua Whitley, Steve Macenski

2.2.0 (2020-07-27)
------------------
* Removed basestring (no longer exists in new python 3 version). (`#554 <https://github.com/ros-perception/image_pipeline/issues/554>`_)
  Fixes `#551 <https://github.com/ros-perception/image_pipeline/issues/551>`_
* Initial ROS2 commit.
* Contributors: Michael Carroll, PfeifferMicha

1.12.23 (2018-05-10)
--------------------
* camera_checker: Ensure cols + rows are in correct order (`#319 <https://github.com/ros-perception/image_pipeline/issues/319>`_)
  Without this commit, specifying a smaller column than row size lead to
  huge reported errors:
  ```
  $ rosrun camera_calibration cameracheck.py --size 6x7 --square 0.0495
  Linearity RMS Error: 13.545 Pixels      Reprojection RMS Error: 22.766 Pixels
  $ rosrun camera_calibration cameracheck.py --size 7x6 --square 0.0495
  Linearity RMS Error: 0.092 Pixels      Reprojection RMS Error: 0.083 Pixels
  ```
  This commit switches columns and rows around if necessary.
* Contributors: Martin Günther

1.12.22 (2017-12-08)
--------------------
* Changed flags CV_LOAD_IMAGE_COLOR by IMREAD_COLOR to adapt to Opencv3. (`#252 <https://github.com/ros-perception/image_pipeline/issues/252>`_)
* Fixed stereo calibration problem with chessboard with the same number of rows and cols by rotating the corners to same direction.
* Contributors: jbosch

1.12.21 (2017-11-05)
--------------------
* re-add the calibration nodes but now using the Python modules.
  Fixes `#298 <https://github.com/ros-perception/image_pipeline/issues/298>`_
* Move nodes to Python module.
* Contributors: Vincent Rabaud

1.12.20 (2017-04-30)
--------------------
* properly save bytes buffer as such
  This is useful for Python 3 and fixes `#256 <https://github.com/ros-perception/image_pipeline/issues/256>`_.
* Get tests slightly looser.
  OpenCV 3.2 gives slightly different results apparently.
* Use floor division where necessary. (`#247 <https://github.com/ros-perception/image_pipeline/issues/247>`_)
* Fix and Improve Camera Calibration Checker Node (`#254 <https://github.com/ros-perception/image_pipeline/issues/254>`_)
  * Fix according to calibrator.py API
  * Add approximate to cameracheck
* Force first corner off chessboard to be uppler left.
  Fixes `#140 <https://github.com/ros-perception/image_pipeline/issues/140>`_
* fix doc jobs
  This is a proper fix for `#233 <https://github.com/ros-perception/image_pipeline/issues/233>`_
* During stereo calibration check that the number of corners detected in the left and right images are the same. This fixes `ros-perception/image_pipeline#225 <https://github.com/ros-perception/image_pipeline/issues/225>`_
* Contributors: Leonard Gerard, Martin Peris, Vincent Rabaud, hgaiser

1.12.19 (2016-07-24)
--------------------
* Fix array check in camerachecky.py
  This closes `#205 <https://github.com/ros-perception/image_pipeline/issues/205>`_
* Contributors: Vincent Rabaud

1.12.18 (2016-07-12)
--------------------

1.12.17 (2016-07-11)
--------------------
* fix typo np -> numpy
* fix failing tests
* Contributors: Shingo Kitagawa, Vincent Rabaud

1.12.16 (2016-03-19)
--------------------
* clean OpenCV dependency in package.xml
* Contributors: Vincent Rabaud

1.12.15 (2016-01-17)
--------------------
* better 16 handling in mkgray
  This re-uses `#150 <https://github.com/ros-perception/image_pipeline/issues/150>`_ and therefore closes `#150 <https://github.com/ros-perception/image_pipeline/issues/150>`_
* fix OpenCV2 compatibility
* fix tests with OpenCV3
* [Calibrator]: add yaml file with calibration data in output
* Contributors: Vincent Rabaud, sambrose

1.12.14 (2015-07-22)
--------------------
* remove camera_hammer and install Python nodes properly
  camera_hammer was just a test for camera info, nothing to do with
  calibration. Plus the test was basic.
* Correct three errors that prevented the node to work properly.
* Contributors: Filippo Basso, Vincent Rabaud

1.12.13 (2015-04-06)
--------------------
* replace Queue by deque of fixed size for simplicity
  That is a potential fix for `#112 <https://github.com/ros-perception/image_pipeline/issues/112>`_
* Contributors: Vincent Rabaud

1.12.12 (2014-12-31)
--------------------
* try to improve `#112 <https://github.com/ros-perception/image_pipeline/issues/112>`_
* Contributors: Vincent Rabaud

1.12.11 (2014-10-26)
--------------------

1.12.10 (2014-09-28)
--------------------
* Update calibrator.py
  bugfix: stereo calibrator crashed after the signature of the method for the computation of the epipolar error changed but the function call was not updated
* Contributors: Volker Grabe

1.12.9 (2014-09-21)
-------------------
* fix bad Python
* only analyze the latest image
  fixes `#97 <https://github.com/ros-perception/image_pipeline/issues/97>`_
* flips width and height during resize to give correct aspect ratio
* Contributors: Russell Toris, Vincent Rabaud

1.12.8 (2014-08-19)
-------------------
* install scripts in the local bin (they are now rosrun-able again)
  fixes `#93 <https://github.com/ros-perception/image_pipeline/issues/93>`_
* fix default Constructor for OpenCV flags
  this does not change anything in practice as the flag is set by the node.
  It just fixes the test.
* Contributors: Vincent Rabaud

1.12.6 (2014-07-27)
-------------------
* make sure the GUI is started in its processing thread and fix a typo
  This fully fixes `#85 <https://github.com/ros-perception/image_pipeline/issues/85>`_
* fix bad call to save an image
* have display be in its own thread
  that could be a fix for `#85 <https://github.com/ros-perception/image_pipeline/issues/85>`_
* fix bad usage of Numpy
  fixes `#89 <https://github.com/ros-perception/image_pipeline/issues/89>`_
* fix asymmetric circle calibration
  fixes `#35 <https://github.com/ros-perception/image_pipeline/issues/35>`_
* add more tests
* improve unittests to include all patterns
* install Python scripts properly
  and fixes `#86 <https://github.com/ros-perception/image_pipeline/issues/86>`_
* fix typo that leads to segfault
  fixes `#84 <https://github.com/ros-perception/image_pipeline/issues/84>`_
* also print self.report() on calibrate ... allows to use the params without having to commit them (e.g. for extrensic calibration between to cameras not used as stereo pair)
* fixes `#76 <https://github.com/ros-perception/image_pipeline/issues/76>`_
  Move Python approximate time synchronizer to ros_comm
* remove all trace of cv in Python (use cv2)
* remove deprecated file (as mentioned in its help)
* fixes `#25 <https://github.com/ros-perception/image_pipeline/issues/25>`_
  This is just removing deprecated options that were around since diamondback
* fixes `#74 <https://github.com/ros-perception/image_pipeline/issues/74>`_
  calibrator.py is now using the cv2 only API when using cv_bridge.
  The API got changed too but it seems to only be used internally.
* Contributors: Vincent Rabaud, ahb

1.12.5 (2014-05-11)
-------------------
* Fix `#68 <https://github.com/ros-perception/image_pipeline/issues/68>`_: StringIO issues in calibrator.py
* fix architecture independent
* Contributors: Miquel Massot, Vincent Rabaud

1.12.4 (2014-04-28)
-------------------

1.12.3 (2014-04-12)
-------------------
* camera_calibration: Fix Python import order
* Contributors: Scott K Logan

1.12.2 (2014-04-08)
-------------------
* Fixes a typo on stereo camera info service calls
  Script works after correcting the call names.
* Contributors: JoonasMelin

1.11.4 (2013-11-23 13:10:55 +0100)
----------------------------------
- add visualization during calibration and several calibration flags (#48)
