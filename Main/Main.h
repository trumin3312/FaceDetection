#pragma once
#include <opencv2/highgui.hpp>

using namespace cv;

Rect	rectDetectedBox;	// 검출된 Box 좌표
int		nDetectedClassId;	// 검출된 Class 번호
int		nDetectedCount;		// 총 검출 갯수