#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

struct RetriveData
{
    /* data */
    int old_index;
    int cur_index;
    double header;
    Vector3d P_old;
    Matrix3d R_old;
    vector<cv::Point2f> measurements;
    vector<int> features_ids; 
    bool relocalized;
    bool relative_pose;
    Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    double loop_pose[7];
};

class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header);

    // internal
    void clearState();
    bool initialStructure(); // 视觉结构初始化
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)]; // 滑动窗口中各关键帧在前一图像帧对应IMU坐标系中的位置
    Vector3d Vs[(WINDOW_SIZE + 1)]; // 滑动窗口中各关键帧在前一图像帧对应IMU坐标系中的速度
    Matrix3d Rs[(WINDOW_SIZE + 1)]; // 滑动窗口中各关键帧在前一图像帧对应IMU坐标系中的旋转
    Vector3d Bas[(WINDOW_SIZE + 1)]; // 滑动窗口中各关键帧对应时刻加速度计的偏置
    Vector3d Bgs[(WINDOW_SIZE + 1)]; // 滑动窗口中各关键帧对应时刻陀螺仪的偏置

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    // 用于滑动窗口内关键帧之间IMU数据的预积分
    // 数组大小为(WINDOW_SIZE + 1)，其中每一个元素都是IntegrationBase类型的指针
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]; 

    Vector3d acc_0, gyr_0; // 预积分中初始时刻（即前一图像关键帧时刻）的IMU数据

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count; // 目前不太确定这个变量的含义，似乎是滑动窗口中关键帧的计数
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager; // 管理特征点的对象
    MotionEstimator m_estimator; // 5点法恢复图像之间的相对运动
    InitialEXRotation initial_ex_rotation;

    bool first_imu; // ture：预积分中初始时刻（即前一图像关键帧时刻）的IMU数据已确定 false： 未指定
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];

    RetriveData retrive_pose_data, front_pose;
    vector<RetriveData> retrive_data_vector;
    int loop_window_index;
    bool relocalize;
    Vector3d relocalize_t;
    Matrix3d relocalize_r;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame; // 存储所有的图像特征点数据，map中的索引值是图像时间戳
    IntegrationBase *tmp_pre_integration;

};
