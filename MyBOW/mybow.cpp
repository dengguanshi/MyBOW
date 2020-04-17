#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace cv;
using namespace std;
#define TRAIN_FOLDER "C:/Users/huangzb/source/repos/MyBOW/MyBOW/data/train_images/"
#define DATA_FOLDER "C:/Users/huangzb/source/repos/MyBOW/MyBOW/data/"



// �Ƴ���չ����������ģ����֯����Ŀ
string categorizer::remove_extention(string full_name)
{
    //find_last_of�ҳ��ַ����һ�γ��ֵĵط�
    int last_index = full_name.find_last_of(".");
    string name = full_name.substr(0, last_index);
    return name;
}

// ���캯��
categorizer::categorizer(int _clusters)
{
    cout << "ִ��categorizer���캯��..." << endl;
    clusters = _clusters;
    //��ʼ��ָ��
    int minHessian = 400;
    categories_size = 2;
    //��ȡѵ����
    make_train_set();
}

//����ѵ������
void categorizer::make_train_set()
{
    cout << "��ȡѵ����..." << endl;
    string categor;
    //�ݹ����rescursive ֱ�Ӷ���������������iΪ������㣨�в�������end_iter�����յ�
 //�ݹ����rescursive ֱ�Ӷ���������������iΪ������㣨�в�������end_iter�����յ�
    for (boost::filesystem::recursive_directory_iterator i(TRAIN_FOLDER), end_iter; i != end_iter; i++)
    {
        // level == 0��ΪĿ¼����ΪTRAIN__FOLDER���������
        if (i.level() == 0)
        {
            // ����Ŀ��������ΪĿ¼������
            if ((i->path()).filename().string() != ".DS_Store") {
                categor = (i->path()).filename().string();
                category_name.push_back(categor);

            }
        }
        else {
            // ��ȡ�ļ����µ��ļ���level 1��ʾ����һ��ѵ��ͼ��ͨ��multimap��������������Ŀ���Ƶ�ѵ��ͼ��һ�Զ��ӳ��
            string filename = string(TRAIN_FOLDER) + categor + string("/") + (i->path()).filename().string();

            if ((i->path()).filename().string() != ".DS_Store") {
                Mat temp = imread(filename, 0);
                pair<string, Mat> p(categor, temp);
                //�õ�ѵ����
                train_set.insert(p);
            }
            cout << "train_set.size()" << endl;
            cout << train_set.size() << endl;//1,2,3
        }

    }
    cout << "���� " << categories_size << "���������..." << endl;
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
// ѵ��ͼƬfeature���࣬�ó��ʵ�
void categorizer::bulid_vacab()
{
    FileStorage vacab_fs(DATA_FOLDER "vocab.xml", FileStorage::READ);
    //���֮ǰ�Ѿ����ɺã��Ͳ���Ҫ���¾������ɴʵ�
    if (vacab_fs.isOpened())
    {
        cout << "ͼƬ�Ѿ����࣬�ʵ��Ѿ�����.." << endl;
        vacab_fs.release();
    }
    else
    {
        //���kmeans���������64*��ȡ����������
        vector<IPoint>my_vocab_descriptors;
        // ����ÿһ��ģ�壬��ȡSURF���ӣ����뵽my_vocab_descriptors��
        multimap<string, Mat> ::iterator i = train_set.begin();
        for (; i != train_set.end(); i++)
        {
            Mat templ = (*i).second;
            templ.convertTo(templ, CV_32F);
            vector<IPoint> ips1 = surf.GetAllFeatures(templ);
            //��ÿһ��ͼ������������ܵ�����
            my_vocab_descriptors.insert(my_vocab_descriptors.end(), ips1.begin(), ips1.end());

        }
        cout << my_vocab_descriptors.size() << endl;
        cout << "ѵ��ͼƬ��ʼ����..." << endl;
        // ��ORB�����ӽ��о���
        cout << "vocab_descriptors" << endl;
        cout << my_vocab_descriptors.size() << endl;//[64 x 21460]�� ��87051   //17551

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }

        //ʹ��mykmeans���о���
        vector<int> best_labels;
        vector<vector<float>> centers;
        double compactness_measure{ 0. };
        const int attemps{ 100 }, max_iter_count{ 100 };
        const double epsilon{ 0.001 };
        const int flags = ANN::KMEANS_RANDOM_CENTERS;
        cout << my_data[0].size() << endl;//5571264
        ANN::kmeans<float>(my_data, clusters, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);
        
        
        //vocab = bowtrainer->cluster(vocab_descriptors);
        cout << "������ϣ��ó��ʵ�..." << endl;
        cout << "vocab" << endl;
        cout << centers.size() << endl;//5571264
        //���ļ���ʽ����ʵ�
        FileStorage file_stor(DATA_FOLDER "vocab.xml", FileStorage::WRITE);
        file_stor << "vocabulary" << centers;
        file_stor.release();
    }
   
}

//����bag of words
void categorizer::compute_bow_image()
{
    cout << "����bag of words..." << endl;
    
    //����ʵ������ֱ�Ӷ�ȡ


    //���bow.txt�Ѿ�����˵��֮ǰ�Ѿ�ѵ�����ˣ�����Ͳ������¹���BOW

    // //��BOW�Ѿ����ڣ�����Ҫ����
  
}

//ѵ��������

void categorizer::trainSvm()
{
    cout << "trainSvm" << endl;
  
    //���ѵ������Ѿ���������Ҫ����ѵ��
   
}


//�Բ���ͼƬ���з���

void categorizer::category_By_svm(Mat input_pic)
{
    cout << "������࿪ʼ..." << endl;
    ////����ĻҶ�ͼ
    //Mat gray_pic;
    ////Mat threshold_image;
    //string prediction_category;
    //float curConfidence;

    //    //��ȡͼƬ
    //    cvtColor(input_pic, gray_pic, CV_BGR2GRAY);

    //    // ��ȡBOW������
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
    //        //��ȡSVM.xml
    //        if (svm_fs.isOpened())
    //        {
    //            svm_fs.release();
    //            Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
    //            if (sign == 0)
    //            {
    //                cout << "����ifѭ��" << endl;
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

    //    //��ȡ��Ŀ¼�µ��ļ���
    //    for (; begin_iterater != end_iterator; ++begin_iterater)
    //    {

    //        if (begin_iterater->path().filename().string() == prediction_category)
    //        {
    //            string filename = string(RESULT_FOLDER) + prediction_category + string("/") + train_pic_name;
    //            imwrite(filename, input_pic);
    //        }
    //    }
    //    cout << "����ͼ����:" << prediction_category << endl;
    //
}


