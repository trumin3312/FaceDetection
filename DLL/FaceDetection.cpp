#include "pch.h"
#include "FaceDetection.h"

#include <fstream>
#include <sstream>
#include <iostream>

float   confThreshold   = 0.5;
float   nmsThreshold    = 0.4;
int     inpWidth        = 416;
int     inpHeight       = 416;

void loadNetwork()
{
    String modelConfiguration = "yolov3-tiny-gray.cfg";
    String modelWeights = "yolov3-tiny-gray.weights";

    network = readNetFromDarknet(modelConfiguration, modelWeights);
    network.setPreferableBackend(DNN_BACKEND_OPENCV);
    network.setPreferableTarget(DNN_TARGET_CPU);
}

void unloadNetwork()
{
    network.~Net();
}

void invokeNetwork(Mat& frame)
{
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

    network.setInput(blob);

    vector<String> outLayerNames;
    vector<int> outLayers = network.getUnconnectedOutLayers();
    vector<String> layersNames = network.getLayerNames();

    outLayerNames.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
        outLayerNames[i] = layersNames[outLayers[i] - 1];

    network.forward(outputs, outLayerNames);
}

void predictBbox(Mat& frame, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes, vector<int>& indices)
{
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        float* data = (float*)outputs[i].data;
        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols)
        {
            Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
            Point classIdPoint;
            double confidence;

            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
}

void drawBbox(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 1);

    int baseLine;
    string label = format("%.2f", conf);
    Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 0.80, 1, &baseLine);
    
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top - (labelSize.height)*0.2), FONT_HERSHEY_PLAIN, 0.80, Scalar(0, 0, 255), 1);
}