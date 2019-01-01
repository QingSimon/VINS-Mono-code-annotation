#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        
        // 默认初始化为非关键帧
        ImageFrame(const map<int, vector<pair<int, Vector3d>>>& _points, double _t):points{_points},t{_t},is_key_frame{false}
        {
        };
        map<int, vector<pair<int, Vector3d> > > points;
        double t; // 图像帧对应的时间戳
        Matrix3d R;
        Vector3d T;
        IntegrationBase *pre_integration; // 图像帧对应的一段预积分
        bool is_key_frame; // true 是关键帧
};

// 调用这个函数就可以完成camera与IMU对齐的所有操作
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);