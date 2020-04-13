#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;



// 移除扩展名，用来讲模板组织成类目
string categorizer::remove_extention(string full_name)
{
    //find_last_of找出字符最后一次出现的地方
    int last_index = full_name.find_last_of(".");
    string name = full_name.substr(0, last_index);
    return name;
}

// 构造函数
categorizer::categorizer(int _clusters)
{
    cout << "开始初始化..." << endl;
    clusters = _clusters;
    //初始化指针
    int minHessian = 400;
    categories_size = 2;
    bowtrainer = new BOWKMeansTrainer(clusters);
    descriptorMacher = BFMatcher::create();

    //读取训练集
    make_train_set();
}

//构造训练集合
void categorizer::make_train_set()
{
    cout << "读取训练集..." << endl;
    string categor;
    //递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
 
    cout << "发现 " << categories_size << "种类别物体..." << endl;
}
Mat categorizer::getMyMat(Mat mymat) {


    vector<IPoint> ips1 = surf.GetAllFeatures(mymat);
    std::cout << "ips1" << endl;
    std::cout << ips1.size() << endl;
    Mat test;//行 列
    for (int t = 0; t < ips1.size(); t++) {
        Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
        test.push_back(tempmat);
        //std::cout << "进入循环" << endl;
        //float* value = ips1[t].descriptor;//读出第i行第j列像素值
        //std::cout << " this->IPoints[t].descriptor" << endl;
        //test.at<float>(0, t) = *value; //将第i行第j列像素值设置为128
    }
    std::cout << "循环外" << endl;
    cout << test.size() << endl;//[1000 x 1]
    return test;
}
Mat Mycluster(const Mat& _descriptors) {
    Mat labels, vocabulary;
    int K{ 4 }, attemps{ 100 };
    //int flags = ANN::KMEANS_RANDOM_CENTERS;
    std::vector<int> best_labels;
    double compactness_measure{ 0. };
    const int myK{ 4 }, myattemps{ 100 }, max_iter_count{ 100 };
    const double epsilon{ 0.001 };
    //ANN::kmeans<float>(_descriptors, myK, best_labels, vocabulary, compactness_measure, max_iter_count, epsilon, myattemps, flags);
    return vocabulary;
}
// 训练图片feature聚类，得出词典
void categorizer::bulid_vacab()
{
    //如果之前已经生成好，就不需要重新聚类生成词典
   
}

//构造bag of words
void categorizer::compute_bow_image()
{
    cout << "构造bag of words..." << endl;
    
    //如果词典存在则直接读取


    //如果bow.txt已经存在说明之前已经训练过了，下面就不用重新构造BOW

    // //如BOW已经存在，则不需要构造
  
}

//训练分类器

void categorizer::trainSvm()
{
    cout << "trainSvm" << endl;
  
    //如果训练结果已经存在则不需要重新训练
   
}


//对测试图片进行分类

void categorizer::category_By_svm(Mat input_pic)
{
    cout << "物体分类开始..." << endl;
    ////输入的灰度图
    //Mat gray_pic;
    ////Mat threshold_image;
    //string prediction_category;
    //float curConfidence;

    //    //读取图片
    //    cvtColor(input_pic, gray_pic, CV_BGR2GRAY);

    //    // 提取BOW描述子
    //    vector<KeyPoint>kp;


    //    Mat newImage;
    //    gray_pic.convertTo(newImage, CV_32F);

    //    Mat test;
    //    featureDecter->detect(gray_pic, kp);
    //    cout << gray_pic.size() << endl;
    //    bowDescriptorExtractor->compute(gray_pic, kp, test);



    //    int sign = 0;
    //    float best_score = -2.0f;
    //    for (int i = 0; i < categories_size; i++)
    //    {
    //        string cate_na = category_name[i];
    //        string f_path = string(DATA_FOLDER) + cate_na + string("SVM.xml");
    //        FileStorage svm_fs(f_path, FileStorage::READ);
    //        //读取SVM.xml
    //        if (svm_fs.isOpened())
    //        {
    //            svm_fs.release();
    //            Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
    //            if (sign == 0)
    //            {
    //                cout << "进入if循环" << endl;
    //                float score_Value = st_svm->predict(test, noArray(), true);
    //                float class_Value = st_svm->predict(test, noArray(), false);
    //                sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
    //            }
    //            curConfidence = sign * st_svm->predict(test, noArray(), true);
    //        }
    //        else
    //        {
    //            if (sign == 0)
    //            {
    //                float scoreValue = stor_svms[i]->predict(test, noArray(), true);
    //                float classValue = stor_svms[i]->predict(test, noArray(), false);
    //                sign = (scoreValue < 0.0f) == (classValue < 0.0f) ? 1 : -1;
    //            }
    //            curConfidence = sign * stor_svms[i]->predict(test, noArray(), true);
    //        }
    //        if (curConfidence > best_score)
    //        {
    //            best_score = curConfidence;
    //            prediction_category = cate_na;
    //        }
    //    }

    //    //获取该目录下的文件名
    //    for (; begin_iterater != end_iterator; ++begin_iterater)
    //    {

    //        if (begin_iterater->path().filename().string() == prediction_category)
    //        {
    //            string filename = string(RESULT_FOLDER) + prediction_category + string("/") + train_pic_name;
    //            imwrite(filename, input_pic);
    //        }
    //    }
    //    cout << "这张图属于:" << prediction_category << endl;
    //
}


