/*
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
*/
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
using namespace std;

#include "ObjectDetector.h"

const size_t network_width = 416;
const size_t network_height = 416;

const char* about = "This sample uses You only look once (YOLO)-Detector "
"(https://arxiv.org/abs/1612.08242) "
"to detect objects on camera/video/image.\n"
"Models can be downloaded here: "
"https://pjreddie.com/darknet/yolo/\n"
"Default network is 416x416.\n"
"Class names can be downloaded here: "
"https://github.com/pjreddie/darknet/tree/master/data\n";

const char* params
= "{ help           | false | print usage         }"
"{ cfg            |       | model configuration }"
"{ model          |       | model weights       }"
"{ camera_device  | 0     | camera device number}"
"{ video          |       | video or image for detection}"
"{ min_confidence | 0.24  | min confidence      }"
"{ class_names    |       | class names         }";

bool CVObjectDetector::Initialize(bool in_is_tiny)
{
	String modelConfiguration; //parser.get<String>("cfg");
	String modelBinary; //parser.get<String>("model");

	if (in_is_tiny)
	{
		modelConfiguration = "./tiny-yolo.cfg";
		modelBinary = "./tiny-yolo.weights";
	}
	else
	{
		modelConfiguration = "./yolo.cfg";
		modelBinary = "./yolo.weights";
	}

	//! [Initialize network]
	net = readNetFromDarknet(modelConfiguration, modelBinary);
	//! [Initialize network]

	// store class names
	ifstream classNamesFile("./coco.names");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (classNamesFile >> className)
			class_names.push_back(className);
	}

	confidence_threshold = 0.3f;

	return true;
}

void CVObjectDetector::Run(Mat in_image, std::vector<Rect>& out_rects, std::vector<string>& out_classes, std::vector<float>& out_confidences)
{
	out_rects.clear();
	out_classes.clear();

	if (in_image.channels() == 4)
		cvtColor(in_image, in_image, COLOR_BGRA2BGR);

	resize(in_image, resized_image, Size(network_width, network_height));
	//! [Resizing without keeping aspect ratio]

	//! [Prepare blob]
	input_blob = blobFromImage(resized_image, 1 / 255.F); //Convert Mat to batch of images
													   //! [Prepare blob]

													   //! [Set input blob]
	net.setInput(input_blob, "data");                   //set the network input
													   //! [Set input blob]

													   //! [Make forward pass]
	detection_mat = net.forward("detection_out");   //compute output
													   //! [Make forward pass]

	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;

	for (int i = 0; i < detection_mat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detection_mat.cols - probability_index;
		float *prob_array_ptr = &detection_mat.at<float>(i, probability_index);

		size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detection_mat.at<float>(i, (int)objectClass + probability_index);

		if (confidence > confidence_threshold)
		{
			float x = detection_mat.at<float>(i, 0);
			float y = detection_mat.at<float>(i, 1);
			float width = detection_mat.at<float>(i, 2);
			float height = detection_mat.at<float>(i, 3);
			int xLeftBottom = static_cast<int>((x - width / 2) * in_image.cols);
			int yLeftBottom = static_cast<int>((y - height / 2) * in_image.rows);
			int xRightTop = static_cast<int>((x + width / 2) * in_image.cols);
			int yRightTop = static_cast<int>((y + height / 2) * in_image.rows);

			Rect out_rect(xLeftBottom, yLeftBottom,
						 xRightTop - xLeftBottom,
						 yRightTop - yLeftBottom);

			if (objectClass < class_names.size())
			{
				out_classes.push_back(class_names[objectClass]);
				out_rects.push_back(out_rect);
				out_confidences.push_back(confidence);
			}
			else
			{
				out_classes.push_back(String("OBJ No:" + objectClass));
				out_rects.push_back(out_rect);
				out_confidences.push_back(confidence);
			}
		}
	}
}

// Test code

CVObjectDetector object_detector;

int main()
{
	object_detector.Initialize(true);

	VideoCapture cap(0);
	Mat frame;

	std::vector<Rect> rects;
	std::vector<string> class_names;
	std::vector<float> confidences;

	ostringstream ss;

	for (;;)
	{
		cap >> frame; // get a new frame from camera/video or read image

		object_detector.Run(frame, rects, class_names, confidences);

		for (int i = 0; i < rects.size(); i++)
		{
			rectangle(frame, rects[i], Scalar(0, 255, 0));

			ss.str("");
			ss << confidences[i];
			String conf(ss.str());
			String label = String(class_names[i]) + ": " + conf;
			int baseLine = 0;
			Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			rectangle(frame, Rect(Point(rects[i].x, rects[i].y - labelSize.height),
				Size(labelSize.width, labelSize.height + baseLine)),
				Scalar(255, 255, 255), CV_FILLED);
			putText(frame, label, Point(rects[i].x, rects[i].y),
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

		}


		imshow("detections", frame);
		if (waitKey(1) >= 0) break;
	}
}

