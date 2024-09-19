#include "multi_lidar_calibration/multi_lidar_calibration.hpp"

MULTI_LIDAR_CALIBRATION::MULTI_LIDAR_CALIBRATION(const std::string & node_name, const rclcpp::NodeOptions & node_options)
: rclcpp::Node(node_name, node_options)
{
    current_guess_ = Eigen::Matrix4f::Identity();
    RCLCPP_INFO(this->get_logger(), "[%s] The node start ... ",__APP_NAME__);
    InitializeROSIo();

    cloud_parent_subscriber_->subscribe(this, points_parent_topic_str, rclcpp::SensorDataQoS{}.keep_last(1).get_rmw_qos_profile());
    RCLCPP_INFO(this->get_logger(), "[%s] Subscribing to... %s",__APP_NAME__, points_parent_topic_str.c_str());

    cloud_child_subscriber_->subscribe(this, points_child_topic_str, rclcpp::SensorDataQoS{}.keep_last(1).get_rmw_qos_profile());
    RCLCPP_INFO(this->get_logger(), "[%s] Subscribing to... %s",__APP_NAME__, points_child_topic_str.c_str());

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    calibrated_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(calibrated_points_topic_str,1);

    using std::placeholders::_1;
    using std::placeholders::_2;
    sync_ptr_ = std::make_shared<Sync>(SyncPolicyT(100),*cloud_parent_subscriber_,*cloud_child_subscriber_);

    sync_ptr_->registerCallback(std::bind(&MULTI_LIDAR_CALIBRATION::syncCallback,this, _1,_2));

}   

void MULTI_LIDAR_CALIBRATION::InitializeROSIo()
{
    //x, y, z, yaw, pitch, roll
    std::string init_file_path;
    init_file_path = declare_parameter("init_params_file_path", "home/lyl/");
    RCLCPP_INFO(this->get_logger(), "[%s] The init_file_path is: %s",__APP_NAME__,init_file_path.c_str());

    std::ifstream ifs(init_file_path); //使用std::ifstream(输入文件流)打开init_file_path路径下的文件，并将其对象命名为ifs
    ifs>>child_topic_num_; //从ifs中读取一个整数，并赋值给child_topic_num_

    for (int j=0; j<child_topic_num_; ++j)
    {
        std::string child_name;
        ifs>>child_name;
        std::vector<double> tmp_transfer;
        for(int k=0; k<6; ++k){
            double tmp_xyzypr;
            ifs>>tmp_xyzypr;
            tmp_transfer.push_back(tmp_xyzypr);
        }
        transfer_map_.insert(std::pair<std::string, std::vector<double>>(child_name, tmp_transfer));
    }

    points_parent_topic_str = declare_parameter("points_parent_src","points_raw");
    RCLCPP_INFO(this->get_logger(), "[%s] points_parent_src: %s",__APP_NAME__, points_parent_topic_str.c_str());
    
    points_child_topic_str = declare_parameter("points_child_src","points_raw");
    RCLCPP_INFO(this->get_logger(), "[%s] points_child_src: %s",__APP_NAME__, points_child_topic_str.c_str());

    voxel_size_ = declare_parameter("voxel_size", 0.1);
    RCLCPP_INFO(this->get_logger(), "[%s] voxel_size: %.2f",__APP_NAME__, voxel_size_);

    ndt_epsilon_ = declare_parameter("ndt_epsilon",0.01);
    RCLCPP_INFO(this->get_logger(), "[%s] ndt_epsilon: %.2f",__APP_NAME__, ndt_epsilon_);

    ndt_step_size_ = declare_parameter("ndt_step_size",0.1);
    RCLCPP_INFO(this->get_logger(), "[%s] ndt_step_size: %.2f",__APP_NAME__, ndt_step_size_);

    ndt_resolution_ = declare_parameter("ndt_resolution", 1.0);
    RCLCPP_INFO(this->get_logger(), "[%s] ndt_resolution: %.2f",__APP_NAME__, ndt_resolution_);

    ndt_iterations_ = declare_parameter("ndt_iterations", 400);
    RCLCPP_INFO(this->get_logger(), "[%s] ndt_iterations: %d",__APP_NAME__, ndt_iterations_);
}

void MULTI_LIDAR_CALIBRATION::syncCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &in_parent_cloud_msg,
                                           const sensor_msgs::msg::PointCloud2::ConstSharedPtr &in_child_cloud_msg)
{
    pcl::PointCloud<PointT>::Ptr parent_cloud (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr child_cloud (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr child_filtered_cloud (new pcl::PointCloud<PointT>);

    pcl::fromROSMsg(*in_parent_cloud_msg, *parent_cloud);
	pcl::fromROSMsg(*in_child_cloud_msg, *child_cloud);

	parent_frame_ = in_parent_cloud_msg->header.frame_id;
	child_frame_ = in_child_cloud_msg->header.frame_id;

	DownsampleCloud(child_cloud, child_filtered_cloud, voxel_size_);
	in_parent_cloud_ = parent_cloud;
	in_child_cloud_ = child_cloud;
    in_child_filtered_cloud_ = child_filtered_cloud;
}

void MULTI_LIDAR_CALIBRATION::DownsampleCloud(pcl::PointCloud<PointT>::ConstPtr in_cloud_ptr,
                                              pcl::PointCloud<PointT>::Ptr out_cloud_ptr,
                                              double in_leaf_size)
{
    pcl::VoxelGrid<PointT> voxelized;
	voxelized.setInputCloud(in_cloud_ptr);
	voxelized.setLeafSize((float)in_leaf_size, (float)in_leaf_size, (float)in_leaf_size);
	voxelized.filter(*out_cloud_ptr);
}

void MULTI_LIDAR_CALIBRATION::Run()
{
    rclcpp::Rate loop_rate(10);

    while(rclcpp::ok()){

        PerformNdtOptimize();

        loop_rate.sleep();
    }

}

void MULTI_LIDAR_CALIBRATION::PerformNdtOptimize()
{
    if (in_parent_cloud_== nullptr || in_child_cloud_== nullptr){
        return;
    }

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    ndt.setTransformationEpsilon(ndt_epsilon_);
    ndt.setStepSize(ndt_step_size_);
    ndt.setResolution(ndt_resolution_);

    ndt.setMaximumIterations(ndt_iterations_);

    ndt.setInputSource(in_child_filtered_cloud_);
    ndt.setInputTarget(in_parent_cloud_);

    pcl::PointCloud<PointT>::Ptr output_cloud(new pcl::PointCloud<PointT>);

    if(current_guess_ == Eigen::Matrix4f::Identity())
    {
        Eigen::Translation3f init_translation(transfer_map_[points_child_topic_str][0],
                transfer_map_[points_child_topic_str][1], transfer_map_[points_child_topic_str][2]);
        Eigen::AngleAxisf init_rotation_x(transfer_map_[points_child_topic_str][5], Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf init_rotation_y(transfer_map_[points_child_topic_str][4], Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf init_rotation_z(transfer_map_[points_child_topic_str][3], Eigen::Vector3f::UnitZ());

        Eigen::Matrix4f init_guess_ = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();

        current_guess_ = init_guess_;
    }

    ndt.align(*output_cloud, current_guess_);

    std::cout << "Normal Distributions Transform converged:" << ndt.hasConverged ()
              << " score: " << ndt.getFitnessScore () << " prob:" << ndt.getTransformationProbability() << std::endl;
    std::cout << "transformation from " << child_frame_ << " to " << parent_frame_ << std::endl;

    pcl::transformPointCloud (*in_child_cloud_, *output_cloud, ndt.getFinalTransformation());

    current_guess_ = ndt.getFinalTransformation();

    Eigen::Matrix3f rotation_matrix = current_guess_.block(0,0,3,3);
    Eigen::Vector3f translation_vector = current_guess_.block(0,3,3,1);


    std::cout << "This transformation can be replicated using:" << std::endl;
    std::cout << "rosrun tf static_transform_publisher " << translation_vector.transpose()
              << " " << rotation_matrix.eulerAngles(2,1,0).transpose() << " /" << parent_frame_
              << " /" << child_frame_ << " 10" << std::endl;

    std::cout << "Corresponding transformation matrix:" << std::endl
              << std::endl << current_guess_ << std::endl << std::endl;

    PublishCloud(output_cloud);

    tf2::Transform t_transform;
    MatrixToTranform(current_guess_, t_transform);
    
    geometry_msgs::msg::TransformStamped transform_stamped;
    auto now = this->now();
    transform_stamped.header.stamp = now;
    transform_stamped.header.frame_id = parent_frame_;
    transform_stamped.child_frame_id = child_frame_;
    tf2::convert(t_transform, transform_stamped.transform);   //数据格式的转换
    // transform_stamped.transform = t_transform;

    // 发布变换
    tf_broadcaster_->sendTransform(transform_stamped);

    //or The following code format also works
    /**/
    // geometry_msgs::msg::TransformStamped transform_stamped_1;
    // transform_stamped_1.header.frame_id = parent_frame_;
    // transform_stamped_1.child_frame_id = child_frame_;
    // transform_stamped_1.header.stamp = this->now();

    // tf2::Vector3 origin;
    // origin.setValue(static_cast<double>(current_guess_(0,3)),static_cast<double>(current_guess_(1,3)),static_cast<double>(current_guess_(2,3)));
    // transform_stamped_1.transform.translation.x = origin.x();
    // transform_stamped_1.transform.translation.y = origin.y();
    // transform_stamped_1.transform.translation.z = origin.z();

}

void MULTI_LIDAR_CALIBRATION::PublishCloud(pcl::PointCloud<PointT>::ConstPtr in_cloud_to_publish_ptr)
{
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header.frame_id = parent_frame_;
    calibrated_cloud_publisher_->publish(cloud_msg);
}

void MULTI_LIDAR_CALIBRATION::MatrixToTranform(Eigen::Matrix4f &matrix, tf2::Transform &trans)
{
    tf2::Vector3 origin;
    origin.setValue(static_cast<double>(matrix(0,3)),static_cast<double>(matrix(1,3)),static_cast<double>(matrix(2,3)));

    tf2::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(matrix(0,0)), static_cast<double>(matrix(0,1)), static_cast<double>(matrix(0,2)),
    static_cast<double>(matrix(1,0)), static_cast<double>(matrix(1,1)), static_cast<double>(matrix(1,2)),
    static_cast<double>(matrix(2,0)), static_cast<double>(matrix(2,1)), static_cast<double>(matrix(2,2)));

    tf2::Quaternion tfqt;
    tf3d.getRotation(tfqt);

    trans.setOrigin(origin);
    trans.setRotation(tfqt);
    // double x = tfqt.x();
}