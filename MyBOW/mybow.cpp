#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace cv;
using namespace std;
#define TRAIN_FOLDER "C:/Users/huangzb/source/repos/MyBOW/MyBOW/data/train_images/"
#define DATA_FOLDER "C:/Users/huangzb/source/repos/MyBOW/MyBOW/data/"



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
    cout << "执行categorizer构造函数..." << endl;
    clusters = _clusters;
    //初始化指针
    int minHessian = 400;
    categories_size = 2;
    //读取训练集
    make_train_set();
}

//构造训练集合
void categorizer::make_train_set()
{
    cout << "读取训练集..." << endl;
    string categor;
    //递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
 //递归迭代rescursive 直接定义两个迭代器：i为迭代起点（有参数），end_iter迭代终点
    for (boost::filesystem::recursive_directory_iterator i(TRAIN_FOLDER), end_iter; i != end_iter; i++)
    {
        // level == 0即为目录，因为TRAIN__FOLDER中设置如此
        if (i.level() == 0)
        {
            // 将类目名称设置为目录的名称
            if ((i->path()).filename().string() != ".DS_Store") {
                categor = (i->path()).filename().string();
                category_name.push_back(categor);

            }
        }
        else {
            // 读取文件夹下的文件。level 1表示这是一副训练图，通过multimap容器来建立由类目名称到训练图的一对多的映射
            string filename = string(TRAIN_FOLDER) + categor + string("/") + (i->path()).filename().string();

            if ((i->path()).filename().string() != ".DS_Store") {
                Mat temp = imread(filename, 0);
                pair<string, Mat> p(categor, temp);
                //得到训练集
                train_set.insert(p);
            }
            cout << "train_set.size()" << endl;
            cout << train_set.size() << endl;//1,2,3
        }

    }
    cout << "发现 " << categories_size << "种类别物体..." << endl;
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
    FileStorage vacab_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);
    //如果之前已经生成好，就不需要重新聚类生成词典
    if (vacab_fs.isOpened())
    {
        cout << "图片已经聚类，词典已经存在.." << endl;
        vacab_fs.release();
    }
    else
    {
        //存放kmeans的输入矩阵，64*提取到的特征点
        vector<IPoint>my_vocab_descriptors;
        // 对于每一幅模板，提取SURF算子，存入到my_vocab_descriptors中
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            Mat templ = (*i).second;
            templ.convertTo(templ, CV_32F);
            vector<IPoint> ips1 = surf.GetAllFeatures(templ);
            //将每一张图的特征点放在总的里面
            my_vocab_descriptors.insert(my_vocab_descriptors.end(), ips1.begin(), ips1.end());

        }
        cout << my_vocab_descriptors.size() << endl;
        cout << "训练图片开始聚类..." << endl;
        // 对ORB描述子进行聚类
        cout << "vocab_descriptors" << endl;
        cout << my_vocab_descriptors.size() << endl;//[64 x 21460]列 行87051   //17551

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //将vector<IPinot>类型和vector<vector<float>>进行转换
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }

        //使用mykmeans进行聚类
        vector<int> best_labels;
        vector<vector<float>> centers;
        double compactness_measure{ 0. };
        const int attemps{ 100 }, max_iter_count{ 100 };
        const double epsilon{ 0.001 };
        const int flags = ANN::KMEANS_RANDOM_CENTERS;
        cout << my_data[0].size() << endl;//5571264
        ANN::kmeans<float>(my_data, clusters, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);
        
        
        //vocab = bowtrainer->cluster(vocab_descriptors);
        cout << "聚类完毕，得出词典..." << endl;
        cout << "vocab" << endl;
        cout << centers.size() << endl;//5571264
        //以文件格式保存词典
        FileStorage file_stor(DATA_FOLDER "vocab.xml", FileStorage::WRITE);
        file_stor << "vocabulary" << centers;
        file_stor.release();
    }
   
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


