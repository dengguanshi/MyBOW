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

    c.category_By_svm(Mat());

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
    cout << "mysvm()" << endl;
    FeatureHistogram predict_hist;
    svm_model* svmModel = svm_load_model("model.txt");
    vector<vector<float>> bof_descriptor(testmat.size());
    //将vector<IPinot>类型和vector<vector<float>>进行转换
    for (int i = 0; i < testmat.size(); i++)
    {
        float* my_bof_descriptor = testmat[i].descriptor;
        vector<float> my_temp(my_bof_descriptor, my_bof_descriptor + 64);
        bof_descriptor[i] = my_temp;
        cout << i << endl;
        // my_data.insert(my_data.end(), my_temp.begin(), my_temp.end());
    }
    getBoF(bof_descriptor, predict_hist, true);
    svm_node* input = new svm_node[2];
    cout << "                                                 ";
    cout << predict_hist.size << endl;//2
    for (int l = 0; l < predict_hist.data.size(); l++) {
        input[l].index = l + 1;
        input[l].value = predict_hist.data[l];
    }
    input[predict_hist.data.size()].index = -1;
    cout << "  ================================== svm_predict " << endl;
    int predictValue = svm_predict(svmModel, input);
    cout << "这幅图的类别id是： ";
    cout << predictValue << endl;// 2

	return predictValue+"";
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

