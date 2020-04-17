#pragma once
#ifndef MACHINE_LEARNING_SVM_H
#define MACHINE_LEARNING_SVM_H
#include <vector>
#include <utility>
#include <iostream>

using namespace std;

class SVM {
private:
    std::vector<std::vector<double>> inData;//从文件都的数据
    std::vector<std::vector<double>> trainData;//分割后的训练数据，里面包含真值
    std::vector<std::vector<double>> testData;
    unsigned long indim = 0;
    std::vector<std::vector<double>> trainDataF;//真正的训练数据，特征
    std::vector<std::vector<double>> testDataF;
    std::vector<double> trainDataGT;//真值
    std::vector<double> testDataGT;
    std::vector<double> w;
    std::vector<double> alpha;//拉格朗日乘数
    double b;
    std::vector<double> E;
    double tol = 0.001;
    double eps = 0.0005;
    double C = 1.0;
public:
    void setTrainD(vector<std::vector<double>>& trainF, vector<double>& trainGT) { trainDataF = trainF; trainDataGT = trainGT; }
    void setTestD(vector<std::vector<double>>& testF, vector<double>& testGT) { testDataGT = testGT; testDataGT = testGT; }
    template <class T1, class T2>
    friend double operator * (const vector<T1>& v1, const vector<T2>& v2);
    template <class T1, class T2>

    friend auto operator * (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1 + v2[0])>;
    template <class T1, class T2>
    friend auto operator + (const vector<T1>& v1, const vector<T2>& v2)->vector<decltype(v1[0] + v2[0])>;

    virtual void getData(const std::string& filename);
    virtual void run();
    void createTrainTest();
    void SMO();
    int SMOTakeStep(int& i1, int& i2);
    int SMOExamineExample(int i2);
    double kernel(std::vector<double>&, std::vector<double>&);
    double computeE(int& i);
    std::pair<double, double> SMOComputeOB(int& i1, int& i2, double& L, double& H);
    void initialize();
    void train();
    double predict(const std::vector<double>& inputData);
};

template <class T1, class T2>
double operator * (const vector<T1>& v1, const vector<T2>& v2) {
    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()) {
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    decltype(v1[0] * v2[0]) re = 0;
    for (int i = 0; i < v1.size(); ++i) {
        re += v1[i] * v2[i];
    }
    return re;
}

template <class T1, class T2>
auto operator * (const T1& arg1, const vector<T2>& v2)->vector<decltype(arg1* v2[0])> {


    if (v2.empty()) {
        cout << "vector must not empty." << endl;
        throw v2.empty();
    }
    vector<decltype(arg1* v2[0])> re(v2.size());
    for (int i = 0; i < v2.size(); ++i) {
        re[i] = arg1 * v2[i];
    }
    return re;
}
template <class T1, class T2>
auto operator + (const vector<T1>& v1, const vector<T2>& v2) ->vector<decltype(v1[0] + v2[0])> {

    if (v1.size() != v2.size()) {
        cout << "two vector must have same size." << endl;
        throw v1.size() != v2.size();
    }
    if (v1.empty()) {
        cout << "vector must not empty." << endl;
        throw v1.empty();
    }
    vector<decltype(v1[0] + v2[0])> re(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        re[i] = v1[i] + v2[i];
    }
    return re;
}
#endif //MACHINE_LEARNING_SVM_H
