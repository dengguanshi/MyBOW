#include <iostream>
#include <opencv2/opencv.hpp>
#include "func.h"
#include "main.h"
using namespace cv;

int main(void)
{
    Surf mysurf;
    //��������ͼƬ
    Mat inputmat= ReadFloatImg("C:\\Users\\huangzb\\source\\repos\\TrainData\\TrainData\\testpic\\hz1.jpg");
    //ͼ��ѵ������
    //train_data(mysurf);
    //ͼ��ʶ����
    Mat output=my_bow(inputmat,mysurf);
    imshow("output", output);
    system("pause");
    return 0;
}