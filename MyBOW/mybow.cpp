#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>

using namespace cv;
using namespace std;



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
    cout << "��ʼ��ʼ��..." << endl;
    clusters = _clusters;
    //��ʼ��ָ��
    int minHessian = 400;
    categories_size = 2;
    bowtrainer = new BOWKMeansTrainer(clusters);
    descriptorMacher = BFMatcher::create();

    //��ȡѵ����
    make_train_set();
}

//����ѵ������
void categorizer::make_train_set()
{
    cout << "��ȡѵ����..." << endl;
    string categor;
    //�ݹ����rescursive ֱ�Ӷ���������������iΪ������㣨�в�������end_iter�����յ�
 
    cout << "���� " << categories_size << "���������..." << endl;
}
Mat categorizer::getMyMat(Mat mymat) {


    vector<IPoint> ips1 = surf.GetAllFeatures(mymat);
    std::cout << "ips1" << endl;
    std::cout << ips1.size() << endl;
    Mat test;//�� ��
    for (int t = 0; t < ips1.size(); t++) {
        Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
        test.push_back(tempmat);
        //std::cout << "����ѭ��" << endl;
        //float* value = ips1[t].descriptor;//������i�е�j������ֵ
        //std::cout << " this->IPoints[t].descriptor" << endl;
        //test.at<float>(0, t) = *value; //����i�е�j������ֵ����Ϊ128
    }
    std::cout << "ѭ����" << endl;
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
// ѵ��ͼƬfeature���࣬�ó��ʵ�
void categorizer::bulid_vacab()
{
    //���֮ǰ�Ѿ����ɺã��Ͳ���Ҫ���¾������ɴʵ�
   
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


