#include "func.h"
#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;




BaseClassifier::BaseClassifier()
{
    size = 0;
    length = 0;
}

BaseClassifier::~BaseClassifier() {}

BaseClassifier::BaseClassifier(const BaseClassifier& cpy)
{
    trainData = cpy.trainData;
    size = cpy.size;
    length = cpy.length;
}

BaseClassifier& BaseClassifier::operator=(const BaseClassifier& rhs)
{
    if (this == &rhs)
        return *this;
    trainData = rhs.trainData;
    size = rhs.size;
    length = rhs.length;
}
void BaseClassifier::add(const FeatureHistogram& trainFeature)
{
    assert(trainFeature.size);
    if (size == 0)
    {
        length = trainFeature.size;
        trainData.push_back(trainFeature);
        svm_trainData.push_back(trainFeature.data);
        svm_labels.push_back(trainFeature.label);

    }
    else
    {
        assert(length == trainFeature.size);
        trainData.push_back(trainFeature);
        svm_trainData.push_back(trainFeature.data);
        svm_labels.push_back(trainFeature.label);
    }
    size++;
}


double train_data(Surf mysurf, categorizer c)
{

    //特征聚类
    c.bulid_vacab();
    //构造BOW
    c.compute_bow_image();
    //训练分类器 
    c.trainSvm();


	return 0.0;
}

Mat my_bow(Mat inputmat, Surf mysurf, categorizer c)
{
	mysurf.inputmat = inputmat; 
	vector<IPoint>testmat= mysurf.GetAllFeatures(mysurf.inputmat);
	cout << testmat.size()<< endl;
	//进行识别
	String str= c.mysvm(testmat);
	return Mat();
}
String  categorizer::mysvm(vector<IPoint>& testmat) {

    svm_model* svmModel = svm_load_model("model.txt");
    vector<vector<float>> my_mat_data(testmat.size());
    for (size_t i = 0; i < testmat.size(); i++)
    {
        float* my_testmat = my_vocab_descriptors[i].descriptor;
        vector<float> my_temp(my_testmat, my_testmat + 64);
        my_mat_data[i] = my_temp;
        // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
    }

    svm_problem problem;
    problem.l = clusters;//有多少数据
    problem.x = new svm_node * [clusters];//特征矩阵
    problem.y = new double[clusters];//对应的标签
    for (int i = 0; i < clusters; ++i) {
        problem.x[i] = new svm_node[64 + 1];
        for (int j = 0; j < 64; ++j) {
            problem.x[i][j].index = j + 1;
            problem.x[i][j].value = main_data[i][j];
        }
        problem.x[i][64].value = -1;
        problem.y[i] = main_labels[i];
    }

   // int predictValue = svm_predict(svmModel, input);




















    //int sign = 0;
    //float best_score = -2.0f;
    //float curConfidence;
    ////Mat threshold_image;
    //string prediction_category;
    //category_name.push_back("qq");
    //category_name.push_back("ww");

    //Mat test(1, 1000, CV_32F);//行 列
  
    //Mat src_img(1, testmat.size(), CV_32F);
    //for (int t = 0; t < testmat.size(); t++) {
    //    //Mat tempmat(1, 64, CV_32F, ips1[t].descriptor);
    //    //test.push_back(tempmat);
    //    float *value = testmat[t].descriptor;//读出第i行第j列像素值
    //    src_img.at<float>(0, t) = *value; //将第i行第j列像素值设置为128
    // }
    //resize(src_img, test, test.size(), 0, 0, INTER_LINEAR);


    ////featureDecter->detect(gray_pic, kp);
    ////bowDescriptorExtractor->compute(gray_pic, kp, test);


    //for (int i = 0; i < categories_size; i++)
    //{
    //    string cate_na = category_name[i];
    //    string f_path = string("C:\\Users\\huangzb\\source\\repos\\ConsoleApplication10\\ConsoleApplication10\\data\\")+cate_na + string("SVM.xml");
    //    FileStorage svm_fs(f_path, FileStorage::READ);
    //    //读取SVM.xml
    //    if (svm_fs.isOpened())
    //    {
    //        svm_fs.release();
    //        Ptr<SVM> st_svm = Algorithm::load<SVM>(f_path.c_str());
    //        if (sign == 0)
    //        {
    //            cout << "进入if循环" << endl;
    //            cout << test << endl;
    //            float score_Value = st_svm->predict(test, noArray(), true);
    //            float class_Value = st_svm->predict(test, noArray(), false);
    //            sign = (score_Value < 0.0f) == (class_Value < 0.0f) ? 1 : -1;
    //        }
    //        curConfidence = sign * st_svm->predict(test, noArray(), true);
    //    }
    //    else
    //    {
    //        cout << "找不到xml文件" << endl;
    //    }
    //    if (curConfidence > best_score)
    //    {
    //        best_score = curConfidence;
    //        prediction_category = cate_na;
    //    }
   
    //}
    //cout << "这张图属于:" << prediction_category << endl;

	return "5";
}


FeatureVector::FeatureVector()
{
    size = 0;
}

FeatureVector::~FeatureVector()
{
    size = 0;
}

FeatureVector::FeatureVector(const FeatureVector& cpy)
{
    data = cpy.data;
    size = cpy.size;
}

FeatureVector& FeatureVector::operator=(const FeatureVector& rhs)
{
    if (this == &rhs)
        return *this;
    data = rhs.data;
    size = rhs.size;
    return *this;
}

FeatureVector& FeatureVector::operator=(const std::vector<float> rhs)
{
    data = rhs;
    size = rhs.size();
    return *this;
}

void FeatureVector::normalize()
{
    double mag = 0;
    for (size_t i = 0; i < size; ++i)
        mag += data[i] * data[i];
    mag = sqrt(mag);
    for (size_t i = 0; i < size; ++i)
        data[i] /= mag;
}
FeatureHistogram::FeatureHistogram()
{
    size = 0;
    label = DEFAULT_LABEL_VALUE;
}

FeatureHistogram::~FeatureHistogram()
{

}

FeatureHistogram::FeatureHistogram(const FeatureHistogram& cpy)
{
    data = cpy.data;
    size = cpy.size;
    label = cpy.label;
}


SVMParameters::SVMParameters(const SVMParameters& cpy)
{
    type = cpy.type;
    kernel = cpy.kernel;
    degree = cpy.degree;
    gamma = cpy.gamma;
    coef0 = cpy.coef0;
    C = cpy.C;
    cache = cpy.cache;
    eps = cpy.eps;
    nu = cpy.nu;
    p = cpy.p;
    termType = cpy.termType;
    iterations = cpy.iterations;
    shrinking = cpy.shrinking;
    probability = cpy.probability;
    weight = cpy.weight;
    kFold = cpy.kFold;
}

SVMParameters::SVMParameters(/*int _type,
    int _kernel,*/
    double _degree,
    double _gamma,
    double _coef0,
    double _C,
    double _cache,
    double _eps,
    double _nu,
    double _p,
    int _termType,
    int _iterations,
    int _shrinking,
    int _probability,
    int _weight,
    int _kFold)
{
    /*type = _type;
    kernel = _kernel;*/
    degree = _degree;
    gamma = _gamma;
    coef0 = _coef0;
    C = _C;
    cache = _cache;
    eps = _eps;
    nu = _nu;
    p = _p;
    termType = _termType;
    iterations = _iterations;
    shrinking = _shrinking;
    probability = _probability;
    weight = _weight;
    kFold = _kFold;
}

SVMParameters& SVMParameters::operator=(const SVMParameters& rhs)
{
    if (this == &rhs)
        return *this;

    type = rhs.type;
    kernel = rhs.kernel;
    degree = rhs.degree;
    gamma = rhs.gamma;
    coef0 = rhs.coef0;
    C = rhs.C;
    cache = rhs.cache;
    eps = rhs.eps;
    nu = rhs.nu;
    p = rhs.p;
    termType = rhs.termType;
    iterations = rhs.iterations;
    shrinking = rhs.shrinking;
    probability = rhs.probability;
    weight = rhs.weight;
    kFold = rhs.kFold;

    return *this;
}

void SVMParameters::setDefault()
{
    /*type = CvSVM::NU_SVC;
    kernel = CvSVM::RBF;
    degree = 3;*/
    gamma = 1;
    coef0 = 0.5;
    C = 1;
    cache = 256;
    eps = 0.0001;
    nu = 0.5;
    p = 0.2;
    termType = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    iterations = 1000;
    shrinking = 0;
    probability = 0;
    weight = 0;
    kFold = 10;
}

void SVMParameters::set(/*int _type,
    int _kernel,*/
    double _degree,
    double _gamma,
    double _coef0,
    double _C,
    double _cache,
    double _eps,
    double _nu,
    double _p,
    int _termType,
    int _iterations,
    int _shrinking,
    int _probability,
    int _weight,
    int _kFold)
{
   /* type = _type;
    kernel = _kernel;*/
    degree = _degree;
    gamma = _gamma;
    coef0 = _coef0;
    C = _C;
    cache = _cache;
    eps = _eps;
    nu = _nu;
    p = _p;
    termType = _termType;
    iterations = _iterations;
    shrinking = _shrinking;
    probability = _probability;
    weight = _weight;
    kFold = _kFold;
}


SVMClassifier::SVMClassifier()
{
    SVMParameters defaultParams;
    setParameters(defaultParams, true);
}

SVMClassifier::SVMClassifier(const SVMClassifier& cpy)
{
    /*params = cpy.params;
    model = cpy.model;*/
    autoTrain = cpy.autoTrain;
    kFold = cpy.kFold;
}

SVMClassifier::SVMClassifier(const SVMParameters& _params, bool _autoTrain)
{
    setParameters(_params, _autoTrain);
}

void SVMClassifier::setParameters(const SVMParameters& _params, bool _autoTrain)
{
    svm_parameter params;
    params.svm_type = _params.type;
    params.kernel_type = _params.kernel;
    params.degree = _params.degree;
    params.gamma = _params.gamma;
    params.coef0 = _params.coef0;
    params.C = _params.C;
    params.nu = _params.nu;
    params.p = _params.p;
    //params.class_weights = NULL;
    //params.term_crit = cvTermCriteria(_params.termType,
    //    _params.iterations,
    //    _params.eps);

    autoTrain = _autoTrain;
    kFold = _params.kFold;
}

