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
            cout << train_set.size() << endl;//for{1,2,3}
        }

    }
    categories_size = category_name.size();
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
        //vector<IPoint>my_vocab_descriptors;
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
        cout << my_vocab_descriptors.size() << endl;//455
        cout << "训练图片开始聚类..." << endl;
        // 对ORB描述子进行聚类
        cout << "vocab_descriptors" << endl;
        cout << my_vocab_descriptors.size() << endl;//455

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //将vector<IPinot>类型和vector<vector<float>>进行转换
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }
        //将所有的特征点放在属性中
        main_data = my_data;
        //使用mykmeans进行聚类
        vector<int> best_labels;
        vector<vector<float>> centers;
        double compactness_measure{ 0. };
        const int attemps{ 100 }, max_iter_count{ 100 };
        const double epsilon{ 0.001 };
        const int flags = ANN::KMEANS_RANDOM_CENTERS;
        cout << my_data[0].size() << endl;//64
        ANN::kmeans<float>(my_data, clusters, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);

        main_centers = centers;
        cout << centers.size() << endl;//20 
        cout << centers[0].size() << endl;//64
        cout << best_labels.size() << endl;//455
        //vocab = bowtrainer->cluster(vocab_descriptors);
        cout << "聚类完毕，得出词典..." << endl;
        for (int i = 0; i < best_labels.size();i++) {
            cout << best_labels[i];
            cout << "    ";
        }//13    14    10    4    17    6    17    5    11    0    11    0 。。。
        cout << "vocab" << endl;
        cout << centers.size() << endl;//20
        cout << centers[0].size() << endl;//64
        //以文件格式保存词典
        FileStorage file_stor(DATA_FOLDER "vocab.xml", FileStorage::WRITE);
        file_stor << "vocabulary" << centers;
        file_stor.release();
    }
   
}


float categorizer::mydistance(vector<float>& fv, vector<float>& col)
{
    float distance = 0;
        for (size_t i = 0; i < fv.size(); ++i)
            distance += (col[i] - fv[i]) * (col[i] - fv[i]); 
        distance = sqrt(distance);
    
    return distance;
}

int categorizer::mymatch(vector<vector<float>>& input)
{
    //float dist;
    //double minDist = DBL_MAX;
    //int minIndex = 0;
    //for (size_t i = 0; i < input.size(); ++i)
    //{
    //    dist = mydistance(main_data[i],col);// main_data[i].distance(f);
    //    if (dist < minDist)
    //    {
    //        minDist = dist;
    //        minIndex = i;
    //    }
    //}
    return 0;
}
bool categorizer::getBoF(vector<vector<float>>& input,FeatureHistogram& hist, bool normalized)
{
    bool built = true;

        int idx;
        hist.resize(clusters);
        hist.zero();
        for (size_t i = 0; i < input.size(); ++i)
        {
            float dist;
            double minDist = DBL_MAX;
            int minIndex = 0;
            for (size_t j = 0; j < clusters; ++j)
            {
                dist = mydistance(input[i], main_centers[j]);// main_data[i].distance(f);
                if (dist < minDist)
                {
                    minDist = dist;
                    minIndex = j;
                }
            }
            idx = minIndex;
            hist.addAt(idx);
        }
        if (normalized)
            hist.normalize();
        return true;
    

}

//构造bag of words一幅图像就可以使用一个K维的向量表示
void categorizer::compute_bow_image()
{
    cout << "构造bag of words..." << endl;
    
    //如果词典存在则直接读取

    FeatureHistogram hist;
    SVMClassifier svm;
    multimap<string, Mat> ::iterator i = train_set.begin();
    int j = 0;
    for (; i != train_set.end(); i++)
    {
        
        Mat templ = (*i).second;
        templ.convertTo(templ, CV_32F);
        vector<IPoint> bof_feature = surf.GetAllFeatures(templ);
        vector<vector<float>> bof_descriptor(bof_feature.size());
        //将vector<IPinot>类型和vector<vector<float>>进行转换
        for (int i = 0; i < bof_feature.size(); i++)
        {
            float* my_bof_descriptor = bof_feature[i].descriptor;
            vector<float> my_temp(my_bof_descriptor, my_bof_descriptor + 64);
            bof_descriptor[i] = my_temp;
            // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }

            getBoF(bof_descriptor, hist, true);
            int n = atoi((*i).first.c_str());
            cout << "* i first==" << endl;
            cout << (*i).first.c_str() << endl;
            cout << n << endl;
            hist.setLabel(n);
            //main_labels.push_back(n); j++;
            svm.add(hist);
    }

    cout << "train   bag of words..." << endl;
    cout << svm.length<< endl;//20
    cout << svm.size << endl;//3
    cout << svm.trainData.size() << endl;//3
    svm_problem prob;
    prob.l = svm.size;        // 训练样本数
    prob.y = new double[categories_size];
    prob.x = new svm_node * [svm.size];
    main_labels = svm.svm_labels;
 /*   prob.y[0] = 0;
    prob.y[1] = 1;
    prob.y[2] = 2;*/
    svm_node* node = new svm_node[svm.size * (1 + svm.length)];
    for (int k=0; k < main_labels.size();k++) {
        prob.y[k] = main_labels[k];
        cout << k << "k=" << endl;
    }
    //memcpy(prob.y, &main_labels[0], main_labels.size() * sizeof(main_labels[0]));
    for (int i = 0; i < categories_size; i++) {
        cout << prob.y[i];
        cout << "    ";
    }
    // 按照格式打包
    for (int i = 0; i < svm.size; i++)
    {
        for (int j = 0; j < svm.length; j++)
        {   // 看不懂指针就得复习C语言了，类比成二维数组的操作
            node[(svm.length + 1) * i + j].index = j + 1;
            node[(svm.length + 1) * i + j].value = svm.svm_trainData[i][j];
        }
        node[(svm.length + 1) * i + svm.length].index = -1;
        prob.x[i] = &node[(svm.length + 1) * i];
    }






    std::cout << main_data.size() << "over\n";//455over
    std::cout << main_data[5].size() << "over\n";//64over
    svm_model* svmModel;
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0.5;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 40;
    param.C = 500;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    // param.probability = 0;
    param.nr_weight = 0;
    param.weight = NULL;
    param.weight_label = NULL;

    cout << "  ================================== svm_train "<<endl;
    svmModel = svm_train(&prob, &param);
    svm_save_model("model.txt", svmModel);
    cout << "  ================================== svm_save_model "<<endl;

    FeatureHistogram predict_hist;
    double result;


    Mat templ = cv::imread("C:\\Users\\huangzb\\source\\repos\\MyBOW\\MyBOW\\data\\test_image\\100.png");
    templ.convertTo(templ, CV_32F);
    vector<IPoint> bof_feature = surf.GetAllFeatures(templ);
    vector<vector<float>> bof_descriptor(bof_feature.size());
    //将vector<IPinot>类型和vector<vector<float>>进行转换
    for (int i = 0; i < bof_feature.size(); i++)
    {
        float* my_bof_descriptor = bof_feature[i].descriptor;
        vector<float> my_temp(my_bof_descriptor, my_bof_descriptor + 64);
        bof_descriptor[i] = my_temp;
        // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
    }

    getBoF(bof_descriptor, predict_hist, true);
    svm_node* input = new svm_node[2];
    cout << "                                                 ";
    cout<< predict_hist.size <<endl;//2
        for (int l = 0; l < predict_hist.data.size();l++) {
            input[l].index = l+1;
            input[l].value = predict_hist.data[l];
        }
    input[predict_hist.data.size()].index = -1;
    cout << "  ================================== svm_predict " << endl;
    int predictValue=svm_predict(svmModel, input);
    cout << svmModel->label[1] << endl;//2
    cout << svmModel->label << endl;//0000025757E108A0
    cout << "  ================================== "; 
    cout << predictValue << endl;// 2



    // //如BOW已经存在，则不需要构造
     // labels已经得到了每个样本（特征点）所属的簇，需要进行统计得到每一张图像的BoF
    int index = 0;
    for (int i = 0; i < allsamples_bow.size();i++) {//descriptor_list是总的图的特征点集合
        // For all keypoints of each image 
        auto cluster = new int[clusters];
        for (int i = 0; i < clusters; i++) {  
            cluster[main_labels[index]] ++;       //labels是kmeans参数的best——label	vector<int>
            index++;
        }
        vector<float>temp_bof(clusters);
        main_bof.push_back(temp_bof);
        delete cluster;
    }
  
}

//训练分类器

void categorizer::trainSvm()
{


    cout << "trainSvm" << endl;
    int flag = 0;
    for (int k = 0; k < categories_size; k++)
    {
        string svm_file_path = string(DATA_FOLDER) + category_name[k] + string("SVM.xml");
        cout << svm_file_path << endl;
        FileStorage svm_fil(svm_file_path, FileStorage::READ);
        //判断训练结果是否存在
        if (svm_fil.isOpened())
        {
            svm_fil.release();
            continue;
        }
        else
        {
            flag = -1;
            break;
        }
    }

    //如果训练结果已经存在则不需要重新训练
    //svm_problem problem;
    //problem.l = clusters;//有多少数据
    //problem.x = new svm_node * [clusters];//特征矩阵
    //problem.y = new double[clusters];//对应的标签
    //for (int i = 0; i < clusters; ++i) {
    //    problem.x[i] = new svm_node[64 + 1];
    //    for (int j = 0; j < 64; ++j) {
    //        problem.x[i][j].index = j + 1;
    //        problem.x[i][j].value = main_data[i][j];
    //    }
    //    problem.x[i][64].index = -1;
    //    problem.y[i] = main_labels[i];

    //}

    //svm_problem prob;
    //prob.l = clusters;        // 训练样本数
    //prob.y = new double[clusters];
    //prob.x = new svm_node * [clusters];
    //svm_node* node = new svm_node[clusters * (1 + 64)];
    //memcpy(prob.y, &main_labels[0], main_labels.size() * sizeof(main_labels[0]));
    //for (int i = 0; i < clusters; i++) {
    //    cout << prob.y[i];
    //    cout << "    ";
    //}
    //// 按照格式打包
    //for (int i = 0; i < clusters; i++)
    //{
    //    for (int j = 0; j < 64; j++)
    //    {   // 看不懂指针就得复习C语言了，类比成二维数组的操作
    //        node[(64 + 1) * i + j].index = j + 1;
    //        node[(64 + 1) * i + j].value = main_data[i][j];
    //    }
    //    node[(64 + 1) * i + 64].index = -1;
    //    prob.x[i] = &node[(64 + 1) * i];
    //}






    //std::cout << main_data.size() << "over\n";//338
    //std::cout << main_data[5].size() << "over\n";//64
    //svm_model* svmModel;
    //svm_parameter param;
    //param.svm_type = C_SVC;
    //param.kernel_type = RBF;
    //param.degree = 3;
    //param.gamma = 0.5;
    //param.coef0 = 0;
    //param.nu = 0.5;
    //param.cache_size = 40;
    //param.C = 500;
    //param.eps = 1e-3;
    //param.p = 0.1;
    //param.shrinking = 1;
    //// param.probability = 0;
    //param.nr_weight = 0;
    //param.weight = NULL;
    //param.weight_label = NULL;

    //svmModel = svm_train(&prob, &param);

    //svm_save_model("model.txt", svmModel);
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


