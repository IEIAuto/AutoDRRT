#!/bin/bash
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    libdir="x86_64"
else
    # 否则执行脚本b
    libdir="aarch64"
fi
cp "libs/$libdir/crop_box_filter/libcrop_box_filter_opt.a" "src/universe/autoware.universe/sensing/pointcloud_preprocessor/src/crop_box_filter/lib/libcrop_box_filter_opt.a"

cp "libs/$libdir/ground_segmentation/libground_segmentation_opt.a" "src/universe/autoware.universe/perception/ground_segmentation/lib/libground_segmentation_opt.a"

cp "libs/$libdir/multi_object_tracker/libmulti_object_tracker_opt.a" "src/universe/autoware.universe/perception/multi_object_tracker/src/lib/libmulti_object_tracker_opt.a"

cp "libs/$libdir/ndt_scan_matcher/libndt_scan_matcher_opt.a" "src/universe/autoware.universe/localization/ndt_scan_matcher/src/lib/libndt_scan_matcher_opt.a"

cp "libs/$libdir/occupancy_grid_map_outlier_filter/liboccupancy_grid_map_outlier_filter_opt.a" "src/universe/autoware.universe/perception/occupancy_grid_map_outlier_filter/src/lib/liboccupancy_grid_map_outlier_filter_opt.a"

cp "libs/$libdir/probabilistic_occupancy_grid_map/libpointcloud_based_occupancy_grid_map_opt.a" "src/universe/autoware.universe/perception/probabilistic_occupancy_grid_map/lib/libpointcloud_based_occupancy_grid_map_opt.a"

cp "libs/$libdir/ring_outlier_filter/libring_outlier_filter_opt.a" "src/universe/autoware.universe/sensing/pointcloud_preprocessor/src/outlier_filter/lib/libring_outlier_filter_opt.a"

echo "Done"