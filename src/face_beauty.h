/************************************************************************************
* File: face_beauty.h
* Brief: 人脸美颜——瘦脸、大眼
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
	* @@brief: 初始化人脸关键点检测模型
	* @return: true-初始化成功；false-初始化失败
	* @param width: 输入图像宽度
	* @param height: 输入图像高度
	* @param enlargeFace: 大眼控制参数（0-10)，默认为5
	* @param shrinkFace: 瘦脸控制参数（0-10），默认为5
	* @param predictorData: 关键点检测模型文件路径
	**/
	bool Init(int width, int height, int enlargeEye = 5, int shrinkFace = 5, std::string predictorData = "./data/shape_predictor_68_face_landmarks.dat");


	/**
	* @@brief: 实现大眼、瘦脸美颜效果
	* @return: true-执行成功；false-执行失败
	* @param input: 当前帧图像
	* @param output: 完成美颜效果的图像
	* @param fmt: 输图像数据格式
	**/
	bool Process(uint8_t *input, uint8_t *output, PixelFmt fmt);

private:

	/**
	* @@brief: 当前帧关键点检测
	* @return: 人脸关键点
	* @param img: 待检测图像
	**/
	std::vector<dlib::full_object_detection> faceLandmarkDetect(cv::Mat& img);



	/**
	* @@brief: 放大眼睛
	* @return: 眼睛放大后的图像
	* @param img: 待处理图像
	* @param landmarks: 检测到的人脸关键点
	**/
	cv::Mat bigEye(cv::Mat& img, std::vector < dlib::full_object_detection> landmarks);



	/**
	* @@brief: 瘦脸
	* @return: 瘦脸后的图像
	* @param img: 待处理图像
	* @param landmarks: 检测到的人脸关键点
	**/
	cv::Mat liftFace(cv::Mat& img, std::vector<dlib::full_object_detection> landmarks);



	/**
	* @@brief: 检测画面中的所有人脸，并定位每张人脸的68个关键点
	* @return: 标记人脸与人脸关键点的图像帧
	* @param frame: 待检测的图像帧
	**/
	cv::Mat drawLandmarks(cv::Mat& frame);


	/**
	* @@brief: 校验文件是否存在
	* @return: true-文件存在；false-文件不存在；
	* @param file: 待校验文件路径
	**/
	bool fileCheck(std::string& file);


	/**
	* @@brief: 计算局部双线性插值结果
	* @return: 局部双线性插值的值
	* @param img: 待插值图像
	* @param ux: 变换点映射位置偏离原有坐标x的量
	* @param uy: 变换点映射位置偏离原有坐标y的量
	**/
	cv::Vec3b bilinearInsert(cv::Mat& img, float ux, float uy);



	/**
	* @@brief: 局部平移算法，进行眼睛放大操作
	* @return: 眼睛放大后的图像
	* @param img: 待处理的图像
	* @param startX: 原始点x坐标
	* @param startY: 原始点y坐标
	* @param endX: 锚点x坐标
	* @param endY: 锚点y坐标
	* @param radius: 放大半径
	**/
	cv::Mat bigEyeAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius);




	/**
	* @@brief: 局部平移算法，进行瘦脸操作
	* @return: 瘦脸后的图像
	* @param img: 待处理的图像
	* @param startX: 原始点x坐标
	* @param startY: 原始点y坐标
	* @param endX: 锚点x坐标
	* @param endY: 锚点y坐标
	* @param radius: 收缩半径
	**/
	cv::Mat liftFaceAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius);



	/**
	* @@brief: 肤质保留的磨皮美白算法
	* @return: 磨皮后的图像
	* @param img: 原始输入图像
	**/
	cv::Mat skinDenise(cv::Mat& img);



	/**
	* @@brief: 自适应图像局部对比度增强
	* @return: 对比度增强后的图像
	* @param img: 原始输入图像
	* @param winSize: 局部均值窗口大小
	* @param maxCg: 增强幅度的上限
	**/
	cv::Mat contrastEnhance(cv::Mat& img, int winSize, int maxCg);


};

#endif // !FACE_BEAUTY_H
