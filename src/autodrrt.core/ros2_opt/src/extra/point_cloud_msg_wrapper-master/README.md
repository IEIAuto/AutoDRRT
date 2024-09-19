# Point Cloud Message Wrapper

This is a modern take on the already existing `sensor_msgs::PointCloud2Modifier` and
`sensor_msgs::PointCloud2Iterator`. These existing classes, while flexible do not offer much safety
in usage. If a wrong type is provided into the iterator it will happily reinterpret the data as it
was told returning garbage data as a result. This is exactly what this class aims to solve. With the
current implementation, this class allows for the following usage:

```c++
sensor_msgs::PointCloud2 cloud;
PointCloud2Modifier<Point2D> modifier{cloud, "frame_id"};
for (const auto& point : {{42, 42}, {23, 42}}) {
  modifier.push_back(point);
}
```

For more details please see [design doc](point_cloud_msg_wrapper/design/point_cloud_msg_wrapper-design.md).


## Acknowledgments

Many thanks go to @xmfcx who has pointed out an idea to use an existing iterator instead of writing
our own as well as for his reviews of the initial version of this code. This has significantly
increased the readability and made the code less error prone. Thanks!
