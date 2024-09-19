Point Cloud Message Wrapper (#point-cloud-msg-wrapper)
===========

# Purpose / Use cases

`PointCloudMsgWrapper` is a class that aims to be used instead of `PointCloud2Iterator` and
`PointCloud2Modifier`. The core idea is that it wraps a point cloud message and allows treating it
essentially as a vector of points. To create a wrapper the user needs to specify the type of points
they want to use and a message that the wrapper wraps. If the fields stored in the point cloud
message do not match the fields or methods of the point struct a runtime error is thrown. An array
of PointField objects can be generated from a given point type automatically through the mechanism
of field generators defined in file `field_generators.hpp` and passed as a tuple into the
`PointCloudMsgWrapper`. If a point has `float x`, `float y`, and `uint8_t intensity` members (or
functions that return non-const references to such fields, e.g. `float& x()`, `float& y()` and
`uint8_t& intensity()`), these generators must be generated (these particular ones _are_ generated
by default, see `default_field_generators.hpp`):

```c++
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(x);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(y);
LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER(intensity);
```

These create classes `field_x_generator`, `field_y_generator`, `field_intensity_generator` that are
then passed as a tuple as a template parameter into the `PointCloudMsgWrapper`.

Other field generators can be generated if needed by the user, but we strive to provide a sane
default value. It is expected that the user provides all needed generators that can be created with
the macros above into the wrapper class upon construction. If the user fails to provide the
appropriate generators an error will be thrown when wrapping an existing point cloud message as an
expected field from a point cloud message would not be matched by the generated ones. See the
default ones in the variable `detail::DefaultFieldGenerators` in `point_cloud_msg_wrapper.hpp` file.

For convenience, there are two typedefs: `PointCloud2View` and `PointCloud2Modifier`.

- `PointCloudView<PointT>{msg}` wraps a constant point cloud and allows read-only access.
  Modification through this view is impossible.
- `PointCloud2Modifier<PointT>{msg}` wraps a mutable message and allows read-write access to the
  underlying data.
- `PointCloud2Modifier<PointT>{msg, new_frame_id}` initializes an empty mutable message. This
  constructor is to be used solely to initialize a new message and will throw if the point cloud
  message is already initialized (i.e., has non-zero number of fields).

## Assumptions / Known limits

### Points used with this class _must_ have an equality operator

If a point has no equality operator, the user must define a free standing operator for it. Otherwise
a `static_assert` will be hit.

### For non-common fields, the generators for these must be created by the user

This package provides a set of generators for common field names, see `default_field_generators.hpp`
file. If other fields are required a custom generator must be created to work with these. We provide
a macro for these purposes: `LIDAR_UTILS__DEFINE_FIELD_GENERATOR_FOR_MEMBER`, also see
`ProcessPointWithCustomField` unit test for an example.

### Field offsets must match _exactly_

To offer safety, fields of the point struct/class must match those stored in the cloud message
_exactly_. That is, if an offset or a type is not strictly matching between the point and the field
stored in the message the wrapper cannot be created for such a combination and an error will be
thrown. If such a situation is encountered, a custom data structure can be created in order to
accomodate for these changes, see `CustomAlignedPoint` struct in the unit tests of this package for
an example.

### Working with points that have virtual methods

We allow point type to have any additional functions as long as it has either:
- public members with names matching the wanted fields, e.g. `float x;`
- public methods that return a reference to the needed field, e.g. `float& x() {return m_x;}`.

That means that some methods can be virtual. In general it is not an issue to work with such types
with `PointCloudMsgWrapper` but one must be careful. It is _forbidden_ to call virtual methods when
reading/writing to a wrapper. The reason is that we memory-map the points into the message,
_including_ the vtable pointer. It is not guaranteed that the same vtable is present when reading
from such a point, so calling any of these methods is undefined behavior. As long as these are not
called, the system behavior is defined.

## Inputs / Outputs / API

See the API documentation.

# Security considerations

TBD by a security specialist.
