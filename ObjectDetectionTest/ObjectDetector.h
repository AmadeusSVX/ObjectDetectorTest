#ifndef __OBJECT_DETECTOR__
#define __OBJECT_DETECTOR__

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

class CVObjectDetector
{
private:

	float confidence_threshold;

	const size_t network_width = 416;
	const size_t network_height = 416;

	dnn::Net net;
	Mat resized_image;
	Mat input_blob;
	Mat detection_mat;

	std::vector<string> class_names;

public:
	bool Initialize(bool in_is_tiny);
	void Run(Mat in_image, std::vector<Rect>& out_rects, std::vector<string>& out_classes, std::vector<float>& out_confidences);
	void Finalize(){};	// Nothing now
};

#endif //__OBJECT_DETECTOR__