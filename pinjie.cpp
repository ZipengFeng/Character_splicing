// pinjie.cpp : 定义控制台应用程序的入口点。
//
#include "stdafx.h"

#include <iomanip>
#include <algorithm>
#include <direct.h>
#include <io.h>
#include <vector>
#include <iostream>

#include "CvvImage.h"
#include "CameraDS.h"

#include <opencv2\opencv.hpp>
#include <opencv2\stitching\stitcher.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include <opencv2/core/core.hpp>

#define MAX_HEIGHT 200 //合成图片的最大高度

using namespace std;
using namespace cv;

Mat rotateImage1(Mat img, int degree);

int GetSim(const Mat* src1, const Mat* src2);
bool isEmptyPic(Mat* img, unsigned int maxNum = 30);

void selectAndStitchImg2(Mat* leftImg, Mat* rightImg, Mat &preFrameImg, Mat &imageResult, int maxDistance = 0);
long long qiuhe(vector<long long> numVec);

vector<string> getFiles(string cate_dir);

unsigned int leftSwitchCol(Mat* leftImg, Mat* rightImg);

vector<int> nVec;
unsigned int minHessian = 1000;
unsigned int match_window_width = 57;

int _tmain(int argc, _TCHAR* argv[])
{
	vector<Mat> matVec;
	Mat input_image;

	string target_dir = "E:\\wrong11\\";
	vector<string> file_names = getFiles(target_dir + "*");

	DWORD start_time_io = GetTickCount();
#pragma omp parallel for
	for (int i = 0; i<file_names.size(); i++) {
		string file_name = target_dir + file_names[i];
		input_image = imread(file_name);
		// Check for invalid input
		if (!input_image.data) {
			cout << "Could not open or find the image." << endl;
			return -1;
		}
		//亮度调节
		Mat illu_image = Mat::zeros(input_image.size(), input_image.type());
		input_image.convertTo(illu_image, -1, 2.5, 0);

		matVec.push_back(illu_image);
	}
	cout << "IO : " << GetTickCount() - start_time_io << " ms." << endl;

	DWORD start_time_proc = GetTickCount();

	unsigned int nowFrame = 2;
	Mat Frame_previous = matVec[nowFrame];
	Mat Frame_result;

	for (unsigned int i = nowFrame; i<matVec.size(); i++) {
		selectAndStitchImg2(&matVec[i - 1], &matVec[i], Frame_previous, Frame_result);
		Frame_previous = Frame_result;
	}
	cout << "Image Process : " << GetTickCount() - start_time_proc << " ms." << endl;
	imshow("Frame_result", Frame_result);
	waitKey(0);
	// imwrite("F:\\result_images\\result.jpg", Frame_result);

	////亮度调节
	//Mat illu_image = Mat::zeros(Frame_result.size(), Frame_result.type());
	//Frame_result.convertTo(illu_image, -1, 2.5, 0);

	////腐蚀膨胀去孤立点
	//Mat element_dil = getStructuringElement(2, Size(3, 3));
	//Mat element_ero = getStructuringElement(1, Size(2, 2));
	//Mat dilation_dst;
	//erode(illu_image, dilation_dst, element_dil);
	//// erode(dilation_dst, dilation_dst, element_dil);
	//dilate(dilation_dst, dilation_dst, element_ero);
	//// dilate(dilation_dst, dilation_dst, element);

	//// Binary
	//Mat gray_image;
	//Mat processed_image;
	//int blockSize = 205;//205  
	//int constValue = 75;//115 
	//cvtColor(dilation_dst, gray_image, CV_BGR2GRAY);
	//adaptiveThreshold(gray_image, processed_image, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, blockSize, constValue);
	//imshow("Ero & Dil Iamge", dilation_dst);
	//imshow("Binary Image", processed_image);
	//waitKey(0);
	return 0;
}



//左图匹配点x坐标的增量，即之前的拼接图片的宽度和
int preFrameWid = 0;
//左图匹配点y坐标的增量
int preFrameHig = 0;

//计算图片相似度并进行拼凑，拼凑时使用左图覆盖右图
//若两张图片相似度很高，则取上一帧图作为结果
//若两张图片相似度不是很高，则将左图拼接至右图，作为结果
//拼接的过程为：首先获取这一帧图片（即右图）与上一帧图片（不是左图，而是上一次拼接时的右图）准确的匹配点，将此匹配点映射到左图，然后进行拼接
void selectAndStitchImg2(Mat* leftImg, Mat* rightImg, Mat &preFrameImg, Mat &imageResult, int maxDistance)
{
	if (leftImg->data == NULL || rightImg->data == NULL)
		return;

	//灰度图转换
	// Mat leftGray, rightGray;
	// cvtColor(*leftImg, leftGray, CV_RGB2GRAY);
	// cvtColor(*rightImg, rightGray, CV_RGB2GRAY);

	int shift_width = 0;
	//提取特征点      
	///*
	SurfDescriptorExtractor surfDetector(minHessian);// 海塞矩阵阈值
	Mat imageDesc1, imageDesc2;
	Mat mask1, mask2;
	mask1 = Mat::zeros(leftImg->rows, leftImg->cols, CV_8UC1);
	mask1(Rect(leftImg->cols - match_window_width, 0, match_window_width, leftImg->rows)).setTo(255);
	mask2 = Mat::zeros(rightImg->rows, rightImg->cols, CV_8UC1);
	mask2(Rect(0, 0, match_window_width, rightImg->rows)).setTo(255);
	vector<KeyPoint> keyPoint2, keyPoint1;
	//特征点描述，为下边的特征点匹配做准备      
#pragma omp parallel sections 
	{
#pragma omp section 
		{
			surfDetector.detect(*leftImg, keyPoint1, mask1);
			surfDetector.compute(*leftImg, keyPoint1, imageDesc1);
		}
#pragma omp section 
		{
		surfDetector.detect(*rightImg, keyPoint2, mask2);
		surfDetector.compute(*rightImg, keyPoint2, imageDesc2);
	}
	}
	//获得匹配特征点，并提取最优配对       
	FlannBasedMatcher matcher;
	vector<DMatch> matchPoints;

	if (imageDesc2.empty()) {
		// cvError(0, "MatchFinder", "2nd descriptor empty", __FILE__, __LINE__);
		cout << "Feature descriptor empty." << endl;
		imageResult = preFrameImg;
		minHessian = 1800;
		return;
	}
	matcher.match(imageDesc1, imageDesc2, matchPoints);//*/

	if (matchPoints.size() == 0) {
		cout << "Match Failed." << endl;
		imageResult = preFrameImg;

		//int transFormCols = preFrameImg.cols + rightImg->cols;
		//int transFormRows = max(preFrameImg.rows, rightImg->rows);
		//imageResult = Mat(transFormRows, transFormCols, CV_8UC3, Scalar(255, 255, 255));
		//preFrameImg.copyTo(Mat(imageResult, Rect(0, 0, preFrameImg.cols, preFrameImg.rows)));
		//rightImg->copyTo(Mat(imageResult, Rect(preFrameImg.cols, int(transFormRows- rightImg->rows)/2, rightImg->cols, rightImg->rows)));
		////imshow("res", imageResult);
		////waitKey(0);//
		return;
	}

//	//使用单应矩阵过滤匹配点
//	if (matchPoints.size() >= 8) {
//		vector<Point2f> srcPoints(matchPoints.size());
//		vector<Point2f> dstPoints(matchPoints.size());
//#pragma omp parallel for
//		for (size_t i = 0; i < matchPoints.size(); i++) {
//			srcPoints[i] = keyPoint2[matchPoints[i].trainIdx].pt;
//			dstPoints[i] = keyPoint1[matchPoints[i].queryIdx].pt;
//		}
//		vector<uchar> inliersMask(srcPoints.size());
//		findHomography(srcPoints, dstPoints, CV_FM_RANSAC, 3, inliersMask);
//		vector<DMatch> inliers;
//#pragma omp parallel for
//		for (size_t i = 0; i < inliersMask.size(); i++){
//			if (inliersMask[i])
//				inliers.push_back(matchPoints[i]);
//		}
//		matchPoints.swap(inliers);
//	}
	// cout << "match point num " << matchPoints.size() << endl;
	sort(matchPoints.begin(), matchPoints.end()); //特征点排序，opencv按照匹配点准确度排序 
	//获取最强匹配点
	Point2i originalLinkPoint, basedImagePoint;
	originalLinkPoint = keyPoint1[matchPoints[0].queryIdx].pt;
	basedImagePoint = keyPoint2[matchPoints[0].trainIdx].pt;

	int distanceY = std::abs(originalLinkPoint.y - basedImagePoint.y);
	int distanceX = std::abs(originalLinkPoint.x - basedImagePoint.x);
	// cout<<"distance-x:"<<distanceX<<" "<<"distance-y:"<<distanceY<<endl;

	//因为是向右扫描，即图片上的字体是向左运动的，所以这一帧图片关键点的x坐标一定小于上一帧关键点的x坐标
	//则不满足此条件的一定为错误的匹配点
	//若最强匹配点不符合要求，再寻找第二个匹配点，直至符合要求
	size_t nowPoint = 0;

	while (((originalLinkPoint.x<basedImagePoint.x) || distanceY>6) && nowPoint<matchPoints.size() - 1)
	{
		nowPoint++;
		originalLinkPoint = keyPoint1[matchPoints[nowPoint].queryIdx].pt;
		basedImagePoint = keyPoint2[matchPoints[nowPoint].trainIdx].pt;
		distanceY = std::abs(originalLinkPoint.y - basedImagePoint.y);
		distanceX = std::abs(originalLinkPoint.x - basedImagePoint.x);
	}
	//画匹配图
	matchPoints.erase(matchPoints.begin() + 1, matchPoints.end());
	Mat imageMatches;
	drawMatches(*leftImg, keyPoint1, *rightImg, keyPoint2, matchPoints, imageMatches, Scalar(255, 0, 0));
	//imshow("匹配图",imageMatches);
	//waitKey();

	//如果最后依然没找到正确匹配点，则该图不拼接
	if (((originalLinkPoint.x<basedImagePoint.x) || distanceY>6))
	{
		imageResult = preFrameImg;
		return;
	}

	//若两张图像非常相似，就不需要拼接了
	if (distanceY <= maxDistance && distanceX <= maxDistance) {
		imageResult = preFrameImg;
		return;
	}
	else {
		//拼接左图和右图
		//用左图覆盖右图（即以右图作为主图）
		//X1为左图匹配点横坐标映射到上一帧图片的横坐标=当前横坐标+上一帧图片拼接的宽度=X1+当前横坐标+上一帧左图匹配点横坐标-上一帧右图匹配点横坐标
		int X1 = originalLinkPoint.x + preFrameWid;
		int Y1 = originalLinkPoint.y + preFrameHig;

		int transFormCols = X1 + rightImg->cols - basedImagePoint.x + shift_width;
		int transFormRows;
		if (basedImagePoint.y >= Y1)
			transFormRows = preFrameImg.rows + basedImagePoint.y - Y1;
		else if (preFrameImg.rows - Y1 >= rightImg->rows - basedImagePoint.y)
			transFormRows = preFrameImg.rows;
		else
			transFormRows = rightImg->rows - basedImagePoint.y + Y1;
		imageResult = Mat(transFormRows, transFormCols, CV_8UC3, Scalar(255, 255, 255));


		Mat ROIMat2,ROIMat3;
		if (basedImagePoint.y >= Y1) {
			ROIMat2 = preFrameImg(Rect(Point(0, 0), Point(X1, preFrameImg.rows)));
			ROIMat2.copyTo(Mat(imageResult, Rect(0, basedImagePoint.y - Y1, ROIMat2.cols, ROIMat2.rows)));
		}
		else {
			ROIMat2 = preFrameImg(Rect(Point(0, 0), Point(X1, preFrameImg.rows)));
			ROIMat2.copyTo(Mat(imageResult, Rect(0, 0, ROIMat2.cols, ROIMat2.rows)));
		}

		if (basedImagePoint.y >= Y1)
		{
			ROIMat3 = (*rightImg)(Rect(Point(basedImagePoint.x, 0), Point(rightImg->cols, rightImg->rows)));
			ROIMat3.copyTo(Mat(imageResult, Rect(X1 + shift_width, 0, ROIMat3.cols, ROIMat3.rows)));
		}
		else
		{
			ROIMat3 = (*rightImg)(Rect(Point(basedImagePoint.x, 0), Point(rightImg->cols, rightImg->rows)));
			ROIMat3.copyTo(Mat(imageResult, Rect(X1 + shift_width, Y1 - basedImagePoint.y, ROIMat3.cols, ROIMat3.rows)));

		}

		preFrameWid += (originalLinkPoint.x - basedImagePoint.x);
		if (basedImagePoint.y >= Y1)
			preFrameHig = 0;
		else
			preFrameHig = Y1 - basedImagePoint.y;
	}
	//imshow("拼接前", preFrameImg);
	//waitKey(0);
	//imshow("拼接后", imageResult);
	//waitKey(0);
}

//根据图片的特征点数目判断图片是否为白板（基本上只有背景，几乎没有什么内容）
bool isEmptyPic(Mat* img, unsigned int maxNum)
{
	//采用ORB进行特征点提取，此算法速度为sift的100倍，surf的10倍
	SurfDescriptorExtractor surfDetector(400);// 海塞矩阵阈值
	// ORB orb;
	vector<KeyPoint> keyPoints;
	Mat imageDescs;
	surfDetector(*img, Mat(), keyPoints, imageDescs);

	if (keyPoints.size()<maxNum)
		return true;
	else
		return false;
}

vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名
	_finddata_t file;
	long lf;
	//输入文件夹路径
	if ((lf = _findfirst(cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
	}
	else {
		while (_findnext(lf, &file) == 0) {
			//输出文件名
			//cout<<file.name<<endl;
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			files.push_back(file.name);
		}
	}
	_findclose(lf);
	//排序，按从小到大排序
	sort(files.begin(), files.end());
	return files;
}




