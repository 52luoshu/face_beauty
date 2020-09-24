/************************************************************************************
* File: face_beauty.h
* Brief: �������ա�������������
* 3rdPart: OpenCV 3.4.2, dlib 19.20.0
* Date: 2020.09.22
************************************************************************************
* Format Support:
*			input				output
*	(1)	ARGB				YUV_I420
*	(2)	YUV_I420			YUV_I420
*	(3)	YUV_YV12			YUV_I420
*	(4)	YVYU				YUV_I420
*	(5)	YUY2				YUV_I420
*	(6)	UYVY				YUV_I420
************************************************************************************/
#ifdef _DEBUG
#pragma comment(lib, "dlib19.20.0_debug_32bit_msvc1916.lib")
#pragma comment(lib, "opencv_world342d.lib")
#else
#pragma comment(lib, "dlib19.20.0_release_32bit_msvc1916.lib")
#pragma comment(lib, "opencv_world342.lib")
#endif


#ifndef FACE_BEAUTY_H
#define FACE_BEAUTY_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

typedef enum PixelFmt {
	//actually ARGB32
	PixelFmt_RGB,	  //0xAARRGGBB

	//planar 4:2:0
	PixelFmt_I420,  //Y1Y2Y3Y4U1V1
	PixelFmt_YV12,  //Y1Y2Y3Y4V1U1

	//packed 4:2:2
	PixelFmt_YVYU,  //Y1V1Y2U1 Y3V3Y4U3
	PixelFmt_YUY2,  //Y1U1Y2V1 Y3U3Y4V3
	PixelFmt_UYVY,  //U1Y1V1Y2 U3Y3V3Y4
	PixelFmt_HDYC,

	PixelFmt_UNKNOWN

}PixelFmt;


class FaceBeauty {
private:
	int mDistanceFace;
	int mDistanceEye;
	bool mInited = false;
	int mWidth;
	int mHeight;
	int mEnlargeEyeDegree = 0;
	int mShrinkFaceDegree = 0;
	dlib::frontal_face_detector mDlibSvmFaceDetector;
	dlib::shape_predictor mDlibSpFaceLandmark;
	
	std::vector<dlib::rectangle> mDlibRectsFaces;
	std::vector<dlib::full_object_detection> mDlibDetsShapes;


public:
	FaceBeauty();
	~FaceBeauty();

	/**
	* @@brief: ��ʼ�������ؼ�����ģ��
	* @return: true-��ʼ���ɹ���false-��ʼ��ʧ��
	* @param width: ����ͼ����
	* @param height: ����ͼ��߶�
	* @param enlargeFace: ���ۿ��Ʋ�����0-10)��Ĭ��Ϊ5
	* @param shrinkFace: �������Ʋ�����0-10����Ĭ��Ϊ5
	* @param predictorData: �ؼ�����ģ���ļ�·��
	**/
	bool Init(int width, int height, int enlargeEye = 5, int shrinkFace = 5, std::string predictorData = "./data/shape_predictor_68_face_landmarks.dat");


	/**
	* @@brief: ʵ�ִ��ۡ���������Ч��
	* @return: true-ִ�гɹ���false-ִ��ʧ��
	* @param input: ��ǰ֡ͼ��
	* @param output: �������Ч����ͼ��
	* @param fmt: ��ͼ�����ݸ�ʽ
	**/
	bool Process(uint8_t *input, uint8_t *output, PixelFmt fmt);

private:

	/**
	* @@brief: ��ǰ֡�ؼ�����
	* @return: �����ؼ���
	* @param img: �����ͼ��
	**/
	std::vector<dlib::full_object_detection> faceLandmarkDetect(cv::Mat& img);



	/**
	* @@brief: �Ŵ��۾�
	* @return: �۾��Ŵ���ͼ��
	* @param img: ������ͼ��
	* @param landmarks: ��⵽�������ؼ���
	**/
	cv::Mat bigEye(cv::Mat& img, std::vector < dlib::full_object_detection> landmarks);



	/**
	* @@brief: ����
	* @return: �������ͼ��
	* @param img: ������ͼ��
	* @param landmarks: ��⵽�������ؼ���
	**/
	cv::Mat liftFace(cv::Mat& img, std::vector<dlib::full_object_detection> landmarks);



	/**
	* @@brief: ��⻭���е���������������λÿ��������68���ؼ���
	* @return: ��������������ؼ����ͼ��֡
	* @param frame: ������ͼ��֡
	**/
	cv::Mat drawLandmarks(cv::Mat& frame);


	/**
	* @@brief: У���ļ��Ƿ����
	* @return: true-�ļ����ڣ�false-�ļ������ڣ�
	* @param file: ��У���ļ�·��
	**/
	bool fileCheck(std::string& file);


	/**
	* @@brief: ����ֲ�˫���Բ�ֵ���
	* @return: �ֲ�˫���Բ�ֵ��ֵ
	* @param img: ����ֵͼ��
	* @param ux: �任��ӳ��λ��ƫ��ԭ������x����
	* @param uy: �任��ӳ��λ��ƫ��ԭ������y����
	**/
	cv::Vec3b bilinearInsert(cv::Mat& img, float ux, float uy);



	/**
	* @@brief: �ֲ�ƽ���㷨�������۾��Ŵ����
	* @return: �۾��Ŵ���ͼ��
	* @param img: �������ͼ��
	* @param startX: ԭʼ��x����
	* @param startY: ԭʼ��y����
	* @param endX: ê��x����
	* @param endY: ê��y����
	* @param radius: �Ŵ�뾶
	**/
	cv::Mat bigEyeAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius);




	/**
	* @@brief: �ֲ�ƽ���㷨��������������
	* @return: �������ͼ��
	* @param img: �������ͼ��
	* @param startX: ԭʼ��x����
	* @param startY: ԭʼ��y����
	* @param endX: ê��x����
	* @param endY: ê��y����
	* @param radius: �����뾶
	**/
	cv::Mat liftFaceAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius);



	/**
	* @@brief: ���ʱ�����ĥƤ�����㷨
	* @return: ĥƤ���ͼ��
	* @param img: ԭʼ����ͼ��
	**/
	cv::Mat skinDenise(cv::Mat& img);



	/**
	* @@brief: ����Ӧͼ��ֲ��Աȶ���ǿ
	* @return: �Աȶ���ǿ���ͼ��
	* @param img: ԭʼ����ͼ��
	* @param winSize: �ֲ���ֵ���ڴ�С
	* @param maxCg: ��ǿ���ȵ�����
	**/
	cv::Mat contrastEnhance(cv::Mat& img, int winSize, int maxCg);


};

#endif // !FACE_BEAUTY_H
