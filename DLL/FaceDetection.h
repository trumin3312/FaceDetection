#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

#ifdef DLL_EXPORTS
#define FACEDETECTION_DECLSPEC __declspec(dllexport)
#else
#define FACEDETECTION_DECLSPEC __declspec(dllimport)
#endif

Net				network;
vector<Mat>		outputs;

extern "C" FACEDETECTION_DECLSPEC	void	loadNetwork();
extern "C" FACEDETECTION_DECLSPEC	void	unloadNetwork();
extern "C" FACEDETECTION_DECLSPEC	void	invokeNetwork(Mat& frame);
extern "C" FACEDETECTION_DECLSPEC	void	predictBbox(Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, vector<int>& indices);
extern "C" FACEDETECTION_DECLSPEC	void	drawBbox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);