#include "stdafx.h"
#include <opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;



/****************the original single gaussion model*************************/

int single_gaussion()
{
	//create new windows
	cvNamedWindow("origin", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("backgournd", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("diff", CV_WINDOW_AUTOSIZE);

	double alpha = 0.1;    //background alpha value
	double std_init = 20;    //init std
	double var_init = std_init * std_init;    //init var
	//double lamda = 2.5 * 1.2;    //background updating ratio 
	double lamda = 2.5 * 2;
	char path[50];
	char path_f[50];
	char path_b[50];
	int nCount = 0;
	CvRect rect = { 0, 28, 288, 260 };

	//vedio file
	CvCapture *capture = NULL;

	//read video file
	capture = cvCaptureFromFile("D:\\Github\\moving_target_detection_and_tracking\\video\\traffic.avi");

	IplImage *frame = NULL;      //orginal image
	IplImage *frame_u = NULL;    //mean image
	IplImage *frame_var = NULL;  //var image
	IplImage *frame_std = NULL;  //std image
	IplImage *diff = NULL;  //diff image

	CvScalar pixel = { 0 };        //pixel origin value
	CvScalar pixel_u = { 0 };      //pixel mean
	CvScalar pixel_var = { 0 };    //pixel var
	CvScalar pixel_std = { 0 };    //pixel std

	//init frame_u, frame_var, frame_std
	frame = cvQueryFrame(capture);
	frame_u = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
	frame_var = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
	frame_std = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);
	diff = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 3);

	for (int y = 0; y < frame->height; ++y)
	{
		for (int x = 0; x < frame->width; ++x)
		{
			pixel = cvGet2D(frame, y, x);

			pixel_u.val[0] = pixel.val[0];//mean image initialization
			pixel_u.val[1] = pixel.val[1];
			pixel_u.val[2] = pixel.val[2];

			pixel_std.val[0] = std_init;//std initialization
			pixel_std.val[1] = std_init;
			pixel_std.val[2] = std_init;

			pixel_var.val[0] = var_init;//var initialization
			pixel_var.val[1] = var_init;
			pixel_var.val[2] = var_init;

			cvSet2D(frame_u, y, x, pixel_u);//put the init value into the mean image 
			cvSet2D(frame_var, y, x, pixel_var);//put the init value into the var image 
			cvSet2D(frame_std, y, x, pixel_std);//put the init value into the std image 
		}
	}

	while (cvWaitKey(10) != 27)        //push ESC to quit, frame ratio ms
	{
		frame = cvQueryFrame(capture);//read from the second frame
		int i = 0;

		//video end, quit
		if (!frame)
		{
			break;
		}
		nCount++;

		//updating single Gaussion background 
		for (int y = 0; y < frame->height; ++y)
		{
			for (int x = 0; x < frame->width; ++x)
			{
				pixel = cvGet2D(frame, y, x);//take out current frame pixel value 
				pixel_u = cvGet2D(frame_u, y, x);//take out previous mean value 取出上次期望的值
				pixel_std = cvGet2D(frame_std, y, x);//take out previous std value
				pixel_var = cvGet2D(frame_var, y, x);// 

				//when |I-u| < lamda*std, take it as background, updating 
				/*if (fabs(pixel.val[0] - pixel_u.val[0]) < lamda * pixel_std.val[0] &&
				fabs(pixel.val[1] - pixel_u.val[1]) < lamda * pixel_std.val[1] &&
				fabs(pixel.val[2] - pixel_u.val[2]) < lamda * pixel_std.val[2])*/
				{
					//update mean value, u = (1-alpha)*u + alpha*I
					pixel_u.val[0] = (1 - alpha) * pixel_u.val[0] + alpha * pixel.val[0];
					pixel_u.val[1] = (1 - alpha) * pixel_u.val[1] + alpha * pixel.val[1];
					pixel_u.val[2] = (1 - alpha) * pixel_u.val[2] + alpha * pixel.val[2];

					//update var = (1-alpha)*var + alpha*(I-u)^2
					pixel_var.val[0] = (1 - alpha) * pixel_var.val[0] +
						alpha *(pixel.val[0] - pixel_u.val[0]) * (pixel.val[0] - pixel_u.val[0]);
					pixel_var.val[1] = (1 - alpha) * pixel_var.val[1] +
						alpha *(pixel.val[1] - pixel_u.val[1]) * (pixel.val[1] - pixel_u.val[1]);
					pixel_var.val[2] = (1 - alpha) * pixel_var.val[2] +
						alpha *(pixel.val[2] - pixel_u.val[2]) * (pixel.val[2] - pixel_u.val[2]);
					//update std
					pixel_std.val[0] = sqrt(pixel_var.val[0]);
					pixel_std.val[1] = sqrt(pixel_var.val[1]);
					pixel_std.val[2] = sqrt(pixel_var.val[2]);

					//write into matrix
					cvSet2D(frame_u, y, x, pixel_u);// write into mean image using the updated mean value
					cvSet2D(frame_var, y, x, pixel_var);//write into var image using the updated var value
					cvSet2D(frame_std, y, x, pixel_std);//write into std image using the updated std value
				}
			}
		}
		cvAbsDiff(frame, frame_u, diff);//origin frame subtract the background

		cvSetImageROI(frame, rect);
		cvSetImageROI(frame_u, rect);
		cvSetImageROI(diff, rect);
		sprintf_s(path, "c:\\testv\\%d.jpg", nCount);
		sprintf_s(path_f, "c:\\testf\\%d.jpg", nCount);
		sprintf_s(path_b, "c:\\testb\\%d.jpg", nCount);

		cvSaveImage(path, frame);
		cvSaveImage(path_f, diff);
		cvSaveImage(path_b, frame_u);
		//denoise
		/*cvDilate(diff, diff, 0, 2);
		cvErode(diff, diff, 0, 3);
		cvDilate(diff, diff, 0, 1);*/
		//show results
		cvShowImage("origin", frame);
		cvShowImage("processing", frame_u);
		cvShowImage("diff", diff);
		cvResetImageROI(frame);
		cvResetImageROI(frame_u);
		cvResetImageROI(diff);
	}

	//release memory
	cvReleaseCapture(&capture);
	cvReleaseImage(&frame);
	cvReleaseImage(&frame_u);
	cvReleaseImage(&frame_var);
	cvReleaseImage(&frame_std);
	cvDestroyWindow("origin");
	cvDestroyWindow("processing");
	return 0;
}


int main()
{
	dangaosisample();
	return 0;
}

