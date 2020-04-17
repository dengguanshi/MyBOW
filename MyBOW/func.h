#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "mykmeans.h"

using namespace cv;
using namespace std;




//��������
class IPoint :public Point2f
{
public:
	//float x;
	//float y;
	float dx;
	float dy;
	float scale;
	float orientation;
	float laplacian;
	float descriptor[64];
	float operator-(const IPoint& rhs);
};

//ͼ����
class IntegralImg
{
public:
	int Width;		//ͼƬ�Ŀ�
	int Height;		//ͼƬ�ĸ�
	Mat Original;	//ԭʼͼƬ
	Mat Integral;	//����ͼ��
	IntegralImg(Mat img);
	float AreaSum(int x, int y, int dx, int dy);
};

//��Ӧ����
class ResponseLayer
{
public:
	//����ͼ��Ŀ��
	int Width;
	//����ͼ��ĸ߶�
	int Height;
	//ģ�����õĲ���
	int Step;
	//ģ��ĳ��ȵ�1/3
	int Lobe;
	//Lobe*2-1
	int Lobe2;
	//ģ��ĳ���һ�룬�߿�
	int Border;
	//ģ�峤��
	int Size;
	//ģ��Ԫ�ظ���
	int Count;
	//����������
	int Octave;
	//����������
	int Interval;
	//��˹������ͼƬ
	Mat* Data;
	//Laplacian����
	Mat* LapData;

	ResponseLayer(IntegralImg* img, int octave, int interval);
	void BuildLayerData(IntegralImg* img);
	float GetResponse(int x, int y, int step);
	float GetLaplacian(int x, int y, int step);
};

//����Hessian������
class FastHessian
{
public:

	IntegralImg Img;
	//ͼ��ѵ�����
	int Octaves;
	//Ϊͼ�����ÿ���е��м��������ֵ��2����ÿ��ͼ�����������Ĳ���
	int Intervals;
	//Hessian��������ʽ��Ӧֵ����ֵ
	float Threshold;

	map<int, ResponseLayer*> Pyramid;
	//������ʸ������
	vector<IPoint> IPoints;
	//���캯��
	FastHessian(IntegralImg iImg, int octaves, int intervals, float threshold);
	void GeneratePyramid();
	void GetIPoints();
	void ShowIPoint();
	bool IsExtremum(int r, int c,
		int step, ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateExtremum(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	void InterpolateStep(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b,
		double* xi, double* xr, double* xc);
	Mat Deriv3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
	Mat Hessian3D(int r, int c, int step,
		ResponseLayer* t, ResponseLayer* m, ResponseLayer* b);
};


//surf������
class SurfDescriptor
{
public:
	IntegralImg& Img;
	std::vector<IPoint>& IPoints;

	void GetOrientation();
	void GetDescriptor();

	float gaussian(int x, int y, float sig);
	float gaussian(float x, float y, float sig);
	float haarX(int row, int column, int s);
	float haarY(int row, int column, int s);
	float getAngle(float X, float Y);
	float RotateX(float x, float y, float si, float co);
	float RotateY(float x, float y, float si, float co);
	int fRound(float flt);
	void DrawOrientation();

	SurfDescriptor(IntegralImg& img, std::vector<IPoint>& iPoints);
};


//surfʹ����
class Surf
{
public:
	Mat inputmat;

	vector<IPoint> GetAllFeatures(Mat img);
};

//���ٹ���bow�㷨��
class categorizer
{
private:
	// //����Ŀ���Ƶ����ݵ�mapӳ��
	map<string, Mat> result_objects;
	//�������ѵ��ͼƬ��BOW
	map<string, Mat> allsamples_bow;
	//����Ŀ���Ƶ�ѵ��ͼ����ӳ�䣬�ؼ��ֿ����ظ�����
	multimap<string, Mat> train_set;
	// ѵ���õ���SVM
	//Ptr<SVM>* stor_svms;
	//��Ŀ���ƣ�Ҳ����TRAIN_FOLDER���õ�Ŀ¼��
	vector<string> category_name;
	//��Ŀ��Ŀ
	int categories_size;
	//��SURF���������Ӿ��ʿ�ľ�����Ŀ
	int clusters;
	//���ѵ��ͼƬ�ʵ�
	Mat vocab;
	Surf surf;


	//����ѵ������
	void make_train_set();
	// �Ƴ���չ����������ģ����֯����Ŀ
	string remove_extention(string);

	Mat getMyMat(Mat);

public:
	//���캯��
	categorizer(int);
	// ����ó��ʵ�
	void bulid_vacab();
	//����BOW
	void compute_bow_image();
	//ѵ��������
	void trainSvm();
	//������ͼƬ����
	void category_By_svm(Mat input_pic);
	Mat Mycluster(const Mat& _descriptors);
	String mysvm(vector<IPoint>& testmat);
};
//��������
double train_data(Surf mysurf);
Mat my_bow(Mat inputmat, Surf mysurf);
Mat ReadFloatImg(const char* szFilename);

Mat ReadFloatImg(const char* szFilename);