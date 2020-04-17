#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;


double train_data(Surf mysurf)
{
    int clusters = 100;
    categorizer c(clusters);
    //��������
    c.bulid_vacab();
    //����BOW
    c.compute_bow_image();
    //ѵ��������
    c.trainSvm();
	return 0.0;
}

Mat my_bow(Mat inputmat, Surf mysurf)
{
    categorizer mycate(100);
	mysurf.inputmat = inputmat; 
	vector<IPoint>testmat= mysurf.GetAllFeatures(mysurf.inputmat);
	cout << testmat.size()<< endl;
	//����ʶ��
	String str= mycate.mysvm(testmat);
	return Mat();
}
String  categorizer::mysvm(vector<IPoint>& testmat) {
    int sign = 0;
    float best_score = -2.0f;
    float curConfidence;
    //Mat threshold_image;
    string prediction_category;
    category_name.push_back("qq");
    category_name.push_back("ww");

    Mat test(1, 1000, CV_32F);//�� ��
  
    Mat src_img(1, testmat.size(), CV_32F);
    for (int t = 0; t < testmat.size(); t++) {
        //Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
        //test.push_back(tempmat);
        float *value = testmat[t].descriptor;//������i�е�j������ֵ
        src_img.at<float>(0, t) = *value; //����i�е�j������ֵ����Ϊ128
     }
    resize(src_img, test, test.size(), 0, 0, INTER_LINEAR);


    //featureDecter->detect(gray_pic, kp);
    //bowDescriptorExtractor->compute(gray_pic, kp, test);


    for (int i = 0; i < categories_size; i++)
    {
        string cate_na = category_name[i];
        string f_path = string("C:\\Users\\huangzb\\source\\repos\\ConsoleApplication10\\ConsoleApplication10\\data\\")+cate_na + string("SVM.xml");
        FileStorage svm_fs(f_path, FileStorage::READ);
        //��ȡSVM.xml
        if (svm_fs.isOpened())
        {
            svm_fs.release();
            Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
            if (sign == 0)
            {
                cout << "����ifѭ��" << endl;
                cout << test << endl;
                float score_Value = st_svm->predict(test, noArray(), true);
                float class_Value = st_svm->predict(test, noArray(), false);
                sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
            }
            curConfidence = sign * st_svm->predict(test, noArray(), true);
        }
        else
        {
            cout << "�Ҳ���xml�ļ�" << endl;
        }
        if (curConfidence > best_score)
        {
            best_score = curConfidence;
            prediction_category = cate_na;
        }
   
    }
    cout << "����ͼ����:" << prediction_category << endl;

	return "5";
}



