#include <iostream>
#include <opencv2/opencv.hpp>
#include "func.h"
#include "main.h"
using namespace cv;

int main(void)
{
    Surf mysurf;
    //²âÊÔÊäÈëÍ¼Æ¬
    Mat inputmat= ReadFloatImg("C:\\Users\\huangzb\\source\\repos\\TrainData\\TrainData\\testpic\\hz1.jpg");
    //Í¼ÏñÑµÁ·º¯Êı
    //train_data(mysurf);
    //Í¼ÏñÊ¶±ğº¯Êı
    Mat output=my_bow(inputmat,mysurf);
    imshow("output", output);
    system("pause");
    return 0;
}