#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

/**
 * FeatureTracker类定义了相机的行为
 * 
 * 
**/
class FeatureTracker
{
  public:
    FeatureTracker();//构造函数实际上是空的，什么都没做

    void readImage(const cv::Mat &_img);//FeatureTracker类中最主要的处理函数

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    vector<cv::Point2f> undistortedPoints();

    cv::Mat mask;
    cv::Mat fisheye_mask;

    // prev_img是上一次发布的帧的图像数据
    // cur_img是光流跟踪的前一帧的图像数据
    // forw_img是光流跟踪的后一帧的图像数据
    cv::Mat prev_img, cur_img, forw_img;

    vector<cv::Point2f> n_pts;//每一帧中新提取的特征点

    // prev_img是上一次发布的帧的特征点数据
    // cur_img是光流跟踪的前一帧的特征点数据
    // forw_img是光流跟踪的后一帧的特征点数据
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    
    vector<int> ids;//能够被跟踪到的特征点的id

    vector<int> track_cnt; //当前帧forw_img中每个特征点被追踪的时间次数

    camodocal::CameraPtr m_camera;

    static int n_id;
};
