#include "feature_tracker.h"

int FeatureTracker::n_id = 0;//FeatureTracker类的static成员变量n_id初始化


 //判断跟踪的特征点是否还在图像边界内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//去除无法追踪的特征
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];//j++表示先取出j的值，再加1
    v.resize(j);
}

//去除无法追踪到的特征点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//构造函数实际上是空的，什么都没做
//我就很纳闷，为什么不在构造函数内初始化成员数据
FeatureTracker::FeatureTracker()
{
}


//对于光流跟踪到的特征点，按照被追踪到的次数排序，然后加上mask去掉部分点，使特征点分布均匀
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time

    //（cnt, pts, id）序列
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    //对光流跟踪到的特征点forw_pts，按照被跟踪到的次数从大到小排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    //清空容器forw_pts，ids，track_cnt
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    //遍历cnt_pts_id
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置，id，被追踪次数分别存入forw_pts, ids, track_cnt
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);

            //在mask中将当前特征点周围半径为MIN_DIST的区域设置为0，后面不再选取该区域内的点
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}


//添将新检测到的特征点n_pts添加到forw_pts中
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);//将新提取的特征点push到forw_pts中
        ids.push_back(-1);//新提取的特征点id初始化为-1
        track_cnt.push_back(1);//新提取的特征点被跟踪的次数初始化为1
    }
}

/**
 * 对新来的图像使用光流法进行特征点跟踪
 * 
 * optional: 使用createCLAHE对图像进行自适应直方图均衡化
 * calcOpticalFlowPyrLK() LK金字塔光流法
 * 设置mask，在setMask()函数中
 * goodFeaturesToTrack()添加一些特征点，确保每帧都有足够的特征点
 * addPoints()添加新的追踪点
*/
//在FeatureTracker实例的实际使用过程中，由于会不断读入新的图像数据，所以FeatureTracker::readImage()会被实例频繁调用
void FeatureTracker::readImage(const cv::Mat &_img)
{
    cv::Mat img;//用来存放直方图均衡化后的图像
    TicToc t_r;//开始计时


    //根据控制参数EQUALIZE判断是否进行直方图均衡化处理
    if (EQUALIZE)
    {
        //调用cv::createCLAHE对图像进行自适应直方图均衡
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;



    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧forw_img
        //同时，还将读入的图像赋给prev_img、cur_img，这是为了避免后面使用到这些数据时，它们是空的
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入
        //所以只需要更新当前帧forw_img的数据
        forw_img = img;
    }

    //此时的forw_pts实际上还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    //调用cv::calcOpticalFlowPyrLK()对前一帧的特征点cur_pts进行金字塔光流跟踪，得到forw_pts
    if (cur_pts.size() > 0)//如果前一帧中特征点数量为0，说明当前读入的是第一帧图像
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        //calcOpticalFlowPyrLK() LK金字塔光流法 
        //ref: https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
        //status标记了从前一帧cur_img到forw_img特征点的跟踪状态
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        //清理无法追踪到的特征点
        for (int i = 0; i < int(forw_pts.size()); i++)
            //有些特征点虽然能够跟踪到，但是位于图像边界外了，把这些特征点状态也标记为无效
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;

        //对于无法跟踪的特征点，不仅要从当前帧数据forw_pts中剔除，而且还要从prev_pts和cur_pts中剔除
        //仔细思考一下，会发现prev_pts和cur_pts中的特征点是一一对应的
        //此外，在记录特征点id的ids，和记录特征点被跟踪次数的track_cnt中，也要把这些跟踪失败的特征点对应位置的记录删除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    if (PUB_THIS_FRAME)
    {
        //通过F矩阵剔除outliers
        rejectWithF();

        //光流追踪成功,特征点被成功跟踪的次数就加1，数值代表被追踪的次数，数值越大，说明被追踪的就越久
        for (auto &n : track_cnt)
            n++;

        ROS_DEBUG("set mask begins");
        TicToc t_m;
        
        //为下面的goodFeaturesToTrack保证相邻的特征点之间要相隔30个像素,设置mask image
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;

        //计算是否需要提取新的特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            // 调用cv::goodFeaturesToTrack()在mask中不为0的区域检测新的特征点
            // ref:https://docs.opencv.org/3.1.0/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
            // forw_img:输入图像
            // n_pts:存放检测到的角点的vector
            // MAX_CNT - forw_pts.size():返回的角点的数量的最大值
            // 0.1:角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
            // MIN_DIST:返回角点之间欧式距离的最小值
            // mask:和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.1, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;

        //添将新检测到的特征点n_pts添加到forw_pts中
        addPoints();
        
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());

        //由于当前帧数据需要发布，所以当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
        prev_img = forw_img;
        prev_pts = forw_pts;
    }

    //把当前帧的数据forw_img、forw_pts赋给上一帧cur_img、cur_pts
    cur_img = forw_img;
    cur_pts = forw_pts;
}

//调用cv::findFundamentalMat对prev_pts和forw_pts计算F矩阵，通过F矩阵去除outliers
void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_prev_pts(prev_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < prev_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_prev_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = prev_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}


//更新特征点id
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            //n_id先赋给ids[i]，然后再加1
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

//读取相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

/**
 * @breif Visualize undistortedPoints Points
*/
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

/**
 * @breif undistortedPoints Points
*/
vector<cv::Point2f> FeatureTracker::undistortedPoints()
{
    vector<cv::Point2f> un_pts;
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }

    return un_pts;
}
