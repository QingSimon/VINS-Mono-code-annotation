#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;//相机成像的像素行数
extern int COL;//相机成像的像素列数
extern int FOCAL_LENGTH;//相机焦距
const int NUM_OF_CAM = 1;//相机个数（好像可以支持双目相机）


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;//应该是每一帧中特征点的最大数量
extern int MIN_DIST;//两个特征点之间的最小像素距离
extern int WINDOW_SIZE;
extern int FREQ;//视觉前端的数据发布频率
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;//双目相机控制开关
extern int EQUALIZE;//控制FeatureTracker::readImage()中，是否对读入的图片做直方图均衡化。   
extern int FISHEYE;
extern bool PUB_THIS_FRAME;//控制是否发布这一帧的数据

void readParameters(ros::NodeHandle &n);//供ROS节点调用，读取参数
