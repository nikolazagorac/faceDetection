// A program to detect faces via webcam or video file in real-time

// Include OpenCV libraries

#include<iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;


// Function for Face Detection 
void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade, double scale);
std::string cascadeName, nestedCascadeName;

int main(int argc, const char** argv)
{
	// VideoCapture class for video
	VideoCapture capture;
	Mat frame, image;

	// OpenCV XML classifiers with facial features 
	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	// Load classifiers from "opencv/data/haarcascades" directory 
	nestedCascade.load("C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Change path before execution 
	cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml");

	
	capture.open(0); // 0 for webcam, or video path for video
	if (capture.isOpened())
	{
		// Capture frames from video and detect faces 
		std::cout << "Searching for Face\nPress E to Exit" << std::endl;
		while (1)
		{
			capture >> frame;
			if (frame.empty())
				break;
			Mat frame1 = frame.clone();
			detectAndDraw(frame1, cascade, nestedCascade, scale);
			char c = (char)waitKey(5); 
			if (c == 'e' || c == 'E') 	// Press E to exit 
				break;
		}
	}
	else
		std::cout << "No input found\n";
	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale)
{
	std::vector<Rect> faces, faces2;
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);  
	double fx = 1 / scale;

	
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	
	cascade.detectMultiScale(smallImg, faces, 1.1,2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center;
		Scalar color = Scalar(255, 255, 255); 
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
				cvPoint(cvRound((r.x + r.width - 1)*scale),
					cvRound((r.y + r.height - 1)*scale)), color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);

		 
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 2,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
	}

	// Show Face Detected Image
	imshow("Face Detection", img);
}
