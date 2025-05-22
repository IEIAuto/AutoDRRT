#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>


using namespace cv;
using namespace Eigen;

// 定义颜色常量 (BGR格式)
const cv::Scalar OBJECT_PALETTE(0, 255, 0); 

// 3D旋转矩阵生成（绕指定轴旋转）
Matrix3d rotation_3d_in_axis(float angle, int axis=2) {
    Matrix3d R = Matrix3d::Identity();
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    
    switch(axis) {
        case 0: // X轴
            R << 1,     0,      0,
                 0, cos_a, -sin_a,
                 0, sin_a,  cos_a;
            break;
        case 1: // Y轴
            R << cos_a, 0, sin_a,
                     0, 1,     0,
                -sin_a, 0, cos_a;
            break;
        case 2: // Z轴
            R << cos_a, -sin_a, 0,
                 sin_a,  cos_a, 0,
                     0,      0, 1;
            break;
        default:
            throw std::invalid_argument("Invalid rotation axis");
    }
    return R;
}

// 生成3D框的8个角点
std::vector<Vector3d> tocorners(const VectorXd& bbox) {
    // 解析参数 [x,y,z,w,h,l,yaw]
    Vector3d center = bbox.segment<3>(0);
    Vector3d dims = bbox.segment<3>(3);
    float yaw = bbox[6];

    std::vector<Vector3d> corners_local = {
        {-dims[0]/2,  -dims[1]/2,  -dims[2]/2}, 
        {-dims[0]/2,  -dims[1]/2,   dims[2]/2},  
        {-dims[0]/2,   dims[1]/2,   dims[2]/2},   
        {-dims[0]/2,   dims[1]/2,  -dims[2]/2},   
        { dims[0]/2,  -dims[1]/2,  -dims[2]/2},
        { dims[0]/2,  -dims[1]/2,   dims[2]/2},  
        { dims[0]/2,   dims[1]/2,   dims[2]/2},   
        { dims[0]/2,   dims[1]/2,  -dims[2]/2}    
    };
    // 应用旋转
    Matrix3d R = rotation_3d_in_axis(yaw, 2);
    std::vector<Vector3d> corners_rotated;
    for(auto& p : corners_local) {
        corners_rotated.push_back(R * p + center);
    }
    return corners_rotated;
}

// 主可视化函数
cv::Mat visualize_camera(
    const cv::Mat& image,
    const std::vector<VectorXd>& bboxes,
    const Matrix4d& transform,
    int thickness = 3) {
    cv::Mat canvas;
    cvtColor(image, canvas, COLOR_RGB2BGR); // 转换颜色空间

    for(const auto& bbox : bboxes) {
        // 生成3D角点
        auto corners = tocorners(bbox);
         
        // 转换为齐次坐标
        MatrixXd corners_h(4, 8); 
        for (int i = 0; i < 8; ++i) {
            corners_h.col(i) << corners[i][0], corners[i][1], corners[i][2], 1;
        }

        // 应用变换矩阵
        MatrixXd transformed = transform * corners_h;
        std::vector<Point2d> proj_points;

        // 投影到图像平面
        for (int i = 0; i < 8; ++i) {
            double z = transformed(2, i);
            if (z <= 1e-5) continue; // 忽略不可见点
            proj_points.emplace_back(transformed(0, i)/std::abs(z), transformed(1, i)/std::abs(z));
        }

        // 绘制3D边线
        const std::vector<std::pair<int,int>> edges = {
            {0,1}, {0,3}, {0,4}, {1,2}, 
            {1,5}, {3,2}, {3,7}, {4,5}, 
            {4,7}, {2,6}, {5,6}, {6,7}
        };

        for (const auto& edge : edges) {
            if (edge.first >= proj_points.size() || 
                edge.second >= proj_points.size()) continue;

            cv::line(canvas, 
                Point(proj_points[edge.first].x, proj_points[edge.first].y),
                Point(proj_points[edge.second].x, proj_points[edge.second].y),
                OBJECT_PALETTE, thickness, LINE_AA);
        }
    }

    cvtColor(canvas, canvas, COLOR_BGR2RGB);
    return canvas;
}
