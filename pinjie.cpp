// pinjie.cpp : �������̨Ӧ�ó������ڵ㡣
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

#define MAX_HEIGHT 200 //�ϳ�ͼƬ�����߶�

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
		//���ȵ���
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

	////���ȵ���
	//Mat illu_image = Mat::zeros(Frame_result.size(), Frame_result.type());
	//Frame_result.convertTo(illu_image, -1, 2.5, 0);

	////��ʴ����ȥ������
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



//��ͼƥ���x�������������֮ǰ��ƴ��ͼƬ�Ŀ�Ⱥ�
int preFrameWid = 0;
//��ͼƥ���y���������
int preFrameHig = 0;

//����ͼƬ���ƶȲ�����ƴ�գ�ƴ��ʱʹ����ͼ������ͼ
//������ͼƬ���ƶȺܸߣ���ȡ��һ֡ͼ��Ϊ���
//������ͼƬ���ƶȲ��Ǻܸߣ�����ͼƴ������ͼ����Ϊ���
//ƴ�ӵĹ���Ϊ�����Ȼ�ȡ��һ֡ͼƬ������ͼ������һ֡ͼƬ��������ͼ��������һ��ƴ��ʱ����ͼ��׼ȷ��ƥ��㣬����ƥ���ӳ�䵽��ͼ��Ȼ�����ƴ��
void selectAndStitchImg2(Mat* leftImg, Mat* rightImg, Mat &preFrameImg, Mat &imageResult, int maxDistance)
{
	if (leftImg->data == NULL || rightImg->data == NULL)
		return;

	//�Ҷ�ͼת��
	// Mat leftGray, rightGray;
	// cvtColor(*leftImg, leftGray, CV_RGB2GRAY);
	// cvtColor(*rightImg, rightGray, CV_RGB2GRAY);

	int shift_width = 0;
	//��ȡ������      
	///*
	SurfDescriptorExtractor surfDetector(minHessian);// ����������ֵ
	Mat imageDesc1, imageDesc2;
	Mat mask1, mask2;
	mask1 = Mat::zeros(leftImg->rows, leftImg->cols, CV_8UC1);
	mask1(Rect(leftImg->cols - match_window_width, 0, match_window_width, leftImg->rows)).setTo(255);
	mask2 = Mat::zeros(rightImg->rows, rightImg->cols, CV_8UC1);
	mask2(Rect(0, 0, match_window_width, rightImg->rows)).setTo(255);
	vector<KeyPoint> keyPoint2, keyPoint1;
	//������������Ϊ�±ߵ�������ƥ����׼��      
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
	//���ƥ�������㣬����ȡ�������       
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

//	//ʹ�õ�Ӧ�������ƥ���
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
	sort(matchPoints.begin(), matchPoints.end()); //����������opencv����ƥ���׼ȷ������ 
	//��ȡ��ǿƥ���
	Point2i originalLinkPoint, basedImagePoint;
	originalLinkPoint = keyPoint1[matchPoints[0].queryIdx].pt;
	basedImagePoint = keyPoint2[matchPoints[0].trainIdx].pt;

	int distanceY = std::abs(originalLinkPoint.y - basedImagePoint.y);
	int distanceX = std::abs(originalLinkPoint.x - basedImagePoint.x);
	// cout<<"distance-x:"<<distanceX<<" "<<"distance-y:"<<distanceY<<endl;

	//��Ϊ������ɨ�裬��ͼƬ�ϵ������������˶��ģ�������һ֡ͼƬ�ؼ����x����һ��С����һ֡�ؼ����x����
	//�������������һ��Ϊ�����ƥ���
	//����ǿƥ��㲻����Ҫ����Ѱ�ҵڶ���ƥ��㣬ֱ������Ҫ��
	size_t nowPoint = 0;

	while (((originalLinkPoint.x<basedImagePoint.x) || distanceY>6) && nowPoint<matchPoints.size() - 1)
	{
		nowPoint++;
		originalLinkPoint = keyPoint1[matchPoints[nowPoint].queryIdx].pt;
		basedImagePoint = keyPoint2[matchPoints[nowPoint].trainIdx].pt;
		distanceY = std::abs(originalLinkPoint.y - basedImagePoint.y);
		distanceX = std::abs(originalLinkPoint.x - basedImagePoint.x);
	}
	//��ƥ��ͼ
	matchPoints.erase(matchPoints.begin() + 1, matchPoints.end());
	Mat imageMatches;
	drawMatches(*leftImg, keyPoint1, *rightImg, keyPoint2, matchPoints, imageMatches, Scalar(255, 0, 0));
	//imshow("ƥ��ͼ",imageMatches);
	//waitKey();

	//��������Ȼû�ҵ���ȷƥ��㣬���ͼ��ƴ��
	if (((originalLinkPoint.x<basedImagePoint.x) || distanceY>6))
	{
		imageResult = preFrameImg;
		return;
	}

	//������ͼ��ǳ����ƣ��Ͳ���Ҫƴ����
	if (distanceY <= maxDistance && distanceX <= maxDistance) {
		imageResult = preFrameImg;
		return;
	}
	else {
		//ƴ����ͼ����ͼ
		//����ͼ������ͼ��������ͼ��Ϊ��ͼ��
		//X1Ϊ��ͼƥ��������ӳ�䵽��һ֡ͼƬ�ĺ�����=��ǰ������+��һ֡ͼƬƴ�ӵĿ��=X1+��ǰ������+��һ֡��ͼƥ��������-��һ֡��ͼƥ��������
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
	//imshow("ƴ��ǰ", preFrameImg);
	//waitKey(0);
	//imshow("ƴ�Ӻ�", imageResult);
	//waitKey(0);
}

//����ͼƬ����������Ŀ�ж�ͼƬ�Ƿ�Ϊ�װ壨������ֻ�б���������û��ʲô���ݣ�
bool isEmptyPic(Mat* img, unsigned int maxNum)
{
	//����ORB������������ȡ�����㷨�ٶ�Ϊsift��100����surf��10��
	SurfDescriptorExtractor surfDetector(400);// ����������ֵ
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
	vector<string> files;//����ļ���
	_finddata_t file;
	long lf;
	//�����ļ���·��
	if ((lf = _findfirst(cate_dir.c_str(), &file)) == -1) {
		cout << cate_dir << " not found!!!" << endl;
	}
	else {
		while (_findnext(lf, &file) == 0) {
			//����ļ���
			//cout<<file.name<<endl;
			if (strcmp(file.name, ".") == 0 || strcmp(file.name, "..") == 0)
				continue;
			files.push_back(file.name);
		}
	}
	_findclose(lf);
	//���򣬰���С��������
	sort(files.begin(), files.end());
	return files;
}




