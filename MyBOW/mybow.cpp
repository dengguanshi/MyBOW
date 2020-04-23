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
            cout << train_set.size() << endl;//for{1,2,3}
        }

    }
    categories_size = category_name.size();
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
        //vector<IPoint>my_vocab_descriptors;
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
        cout << my_vocab_descriptors.size() << endl;//455
        cout << "ѵ��ͼƬ��ʼ����..." << endl;
        // ��ORB�����ӽ��о���
        cout << "vocab_descriptors" << endl;
        cout << my_vocab_descriptors.size() << endl;//455

        vector<vector<float>> my_data(my_vocab_descriptors.size());
        //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
        for (size_t i = 0; i < my_vocab_descriptors.size(); i++)
        {
            float *my_descriptor =my_vocab_descriptors[i].descriptor;
            vector<float> my_temp(my_descriptor, my_descriptor+64);
            my_data[i] = my_temp;
           // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
        }
        //�����е����������������
        main_data = my_data;
        //ʹ��mykmeans���о���
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
        cout << "������ϣ��ó��ʵ�..." << endl;
        for (int i = 0; i < best_labels.size();i++) {
            cout << best_labels[i];
            cout << "    ";
        }//13    14    10    4    17    6    17    5    11    0    11    0 ������
        cout << "vocab" << endl;
        cout << centers.size() << endl;//20
        cout << centers[0].size() << endl;//64
        //���ļ���ʽ����ʵ�
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

//����bag of wordsһ��ͼ��Ϳ���ʹ��һ��Kά��������ʾ
void categorizer::compute_bow_image()
{
    cout << "����bag of words..." << endl;
    
    //����ʵ������ֱ�Ӷ�ȡ

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
        //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
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
    prob.l = svm.size;        // ѵ��������
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
    // ���ո�ʽ���
    for (int i = 0; i < svm.size; i++)
    {
        for (int j = 0; j < svm.length; j++)
        {   // ������ָ��͵ø�ϰC�����ˣ���ȳɶ�ά����Ĳ���
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
    //��vector<IPinot>���ͺ�vector<vector<float>>����ת��
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



    // //��BOW�Ѿ����ڣ�����Ҫ����
     // labels�Ѿ��õ���ÿ�������������㣩�����Ĵأ���Ҫ����ͳ�Ƶõ�ÿһ��ͼ���BoF
    int index = 0;
    for (int i = 0; i < allsamples_bow.size();i++) {//descriptor_list���ܵ�ͼ�������㼯��
        // For all keypoints of each image 
        auto cluster = new int[clusters];
        for (int i = 0; i < clusters; i++) {  
            cluster[main_labels[index]] ++;       //labels��kmeans������best����label	vector<int>
            index++;
        }
        vector<float>temp_bof(clusters);
        main_bof.push_back(temp_bof);
        delete cluster;
    }
  
}

//ѵ��������

void categorizer::trainSvm()
{


    cout << "trainSvm" << endl;
    int flag = 0;
    for (int k = 0; k < categories_size; k++)
    {
        string svm_file_path = string(DATA_FOLDER) + category_name[k] + string("SVM.xml");
        cout << svm_file_path << endl;
        FileStorage svm_fil(svm_file_path, FileStorage::READ);
        //�ж�ѵ������Ƿ����
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

    //���ѵ������Ѿ���������Ҫ����ѵ��
    //svm_problem problem;
    //problem.l = clusters;//�ж�������
    //problem.x = new svm_node * [clusters];//��������
    //problem.y = new double[clusters];//��Ӧ�ı�ǩ
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
    //prob.l = clusters;        // ѵ��������
    //prob.y = new double[clusters];
    //prob.x = new svm_node * [clusters];
    //svm_node* node = new svm_node[clusters * (1 + 64)];
    //memcpy(prob.y, &main_labels[0], main_labels.size() * sizeof(main_labels[0]));
    //for (int i = 0; i < clusters; i++) {
    //    cout << prob.y[i];
    //    cout << "    ";
    //}
    //// ���ո�ʽ���
    //for (int i = 0; i < clusters; i++)
    //{
    //    for (int j = 0; j < 64; j++)
    //    {   // ������ָ��͵ø�ϰC�����ˣ���ȳɶ�ά����Ĳ���
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


