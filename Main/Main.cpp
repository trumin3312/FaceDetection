#include "Main.h"
#include "../DLL/FaceDetection.h"
#pragma comment(lib, "../include/lib/FaceDetection.lib")

void modifyFrame(Mat& frame)
{
    int rows = frame.rows;
    int cols = frame.cols * frame.channels();

    for (int j = 0; j < rows; j++)
    {
        short* pData = frame.ptr<short>(j);

        for (int i = 0; i < cols; i++)
        {
            // bit shift
            pData[i] >>= 2;
            // type cast
            pData[i] = (uchar)pData[i];
        }
    }

    frame.convertTo(frame, CV_8U);
}

int main()
{
    // Open a camera stream
    VideoCapture cap;
    cap.open(0, CAP_DSHOW);
    cap.set(CAP_PROP_CONVERT_RGB, 0);
    if (!cap.isOpened()) 
        return -1;

    loadNetwork();   

    // Process frames  
    while (waitKey(1) < 0)
    {
        // Debug
        /*short  data16[80 * 82 * 1];
        uchar   data8[80 * 82 * 1];
        memset(data16, NULL, sizeof(ushort) * (80 * 82 * 1));
        memset(data8, NULL, sizeof(uchar) * (80 * 82 * 1));*/
        
        // get frame from the camera
        Mat frame;
        cap >> frame;
        //memcpy(data16, frame.data, sizeof(short) * (80 * 82 * 1));

        if (frame.empty())
            break;
       
        modifyFrame(frame);
        //memcpy(data8, frame.data, sizeof(uchar) * (80 * 82 * 1));

        invokeNetwork(frame);

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        vector<int> indices;
        predictBbox(frame, classIds, confidences, boxes, indices);

        nDetectedCount = indices.size();
        for (size_t i = 0; i < nDetectedCount; ++i)
        {
            int idx = indices[i];
            rectDetectedBox = boxes[idx];
            nDetectedClassId = classIds[idx];
            drawBbox(nDetectedClassId, confidences[idx], rectDetectedBox.x, rectDetectedBox.y, rectDetectedBox.x + rectDetectedBox.width, rectDetectedBox.y + rectDetectedBox.height, frame);
        }

        // Create a window
        static const string kWinName = "FaceDetection";
        namedWindow(kWinName, WINDOW_NORMAL);
        imshow(kWinName, frame);
    }

    cap.release();
    unloadNetwork();
    
    return 0;
}