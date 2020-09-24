#include "face_beauty.h"
//#define SHARPEN

FaceBeauty::FaceBeauty(){}



FaceBeauty::~FaceBeauty(){}



/**
* @@brief: ��ʼ�������ؼ�����ģ��
* @return: true-��ʼ���ɹ���false-��ʼ��ʧ��
* @param width: ����ͼ����
* @param height: ����ͼ��߶�
* @param enlargeFace: ���ۿ��Ʋ�����0-10����Ĭ��Ϊ5
* @param shrinkFace: �������Ʋ�����0-10����Ĭ��Ϊ5
* @param predictorData: �ؼ�����ģ���ļ�·��
**/
bool  FaceBeauty::Init(int width, int height, int enlargeEye, int shrinkFace, std::string predictorData) {
	cv::setUseOptimized(true);
	if ((fileCheck(predictorData) == false) || (width <= 0) || (height <= 0) || 
		(enlargeEye < 0) || (enlargeEye > 10) || (shrinkFace < 0) || (shrinkFace > 10)) {
		mInited = false;
	}
	else{
		mDlibSvmFaceDetector = dlib::get_frontal_face_detector();
		dlib::deserialize(predictorData) >> mDlibSpFaceLandmark;
		mWidth = width;
		mHeight = height;
		mEnlargeEyeDegree = enlargeEye;
		mShrinkFaceDegree = shrinkFace;
		mInited = true;
	}
	return mInited;
}




/**
* @@brief: ʵ�ִ��ۡ���������Ч��
* @return: true-ִ�гɹ���false-ִ��ʧ��
* @param input: ��ǰ֡ͼ��
* @param output: �������Ч����ͼ��
* @param fmt: ��ͼ�����ݸ�ʽ
**/
bool FaceBeauty::Process(uint8_t *input, uint8_t *output, PixelFmt fmt) {
	if ((mInited == false) || (fmt == PixelFmt_UNKNOWN) || (input == nullptr) || (output == nullptr)) {
		return false;
	}
	
	cv::UMat UframeBGR;
	if (fmt == PixelFmt_I420) {
		cv::UMat UframeI420 = cv::Mat(mHeight * 3 / 2, mWidth, CV_8UC1, input).getUMat(cv::ACCESS_READ);
		cv::cvtColor(UframeI420, UframeBGR, cv::COLOR_YUV2BGR_I420);
	}
	if (fmt == PixelFmt_RGB) {
		cv::UMat UframeARGB = cv::Mat(mHeight, mWidth, CV_8UC4, input).getUMat(cv::ACCESS_READ);
		std::vector<cv::UMat> UfrmChARGB;
		cv::split(UframeARGB, UfrmChARGB);
		std::vector<cv::UMat> UfrmChBGR;
		UfrmChBGR = { UfrmChARGB[3],UfrmChARGB[2],UfrmChARGB[1] };
		cv::merge(UfrmChBGR, UframeBGR);
	}
	if (fmt == PixelFmt_UYVY) {
		cv::UMat UframeUYVY = cv::Mat(mHeight, mWidth, CV_8UC2, input).getUMat(cv::ACCESS_READ);
		cv::cvtColor(UframeUYVY, UframeBGR, cv::COLOR_YUV2BGR_UYVY);
	}
	if (fmt == PixelFmt_YUY2) {
		cv::UMat UframeYUY2 = cv::Mat(mHeight, mWidth, CV_8UC2, input).getUMat(cv::ACCESS_READ);
		cv::cvtColor(UframeYUY2, UframeBGR, cv::COLOR_YUV2BGR_YUY2);
	}
	if (fmt == PixelFmt_YVYU) {
		cv::UMat UframeYVYU = cv::Mat(mHeight, mWidth, CV_8UC2, input).getUMat(cv::ACCESS_READ);
		cv::cvtColor(UframeYVYU, UframeBGR, cv::COLOR_YUV2BGR_YVYU);
	}
	if (fmt == PixelFmt_HDYC) {
		//TODO
		return false;
	}
	cv::Mat frameBGR = UframeBGR.getMat(cv::ACCESS_READ);
	std::vector<dlib::full_object_detection> landmarks;
	landmarks = faceLandmarkDetect(frameBGR);
	cv::Mat frameEnlargeEye = bigEye(frameBGR, landmarks);
	cv::Mat frameShrinkFace = liftFace(frameEnlargeEye, landmarks);
	cv::Mat frameSkinDenise = skinDenise(frameShrinkFace);
	//cv::Mat frameEnhanceContrast = contrastEnhance(frameSkinDenise, 11, 255);
	cv::UMat UretI420;
	cv::cvtColor(frameSkinDenise.getUMat(cv::ACCESS_READ), UretI420, cv::COLOR_BGR2YUV_I420);
	cv::Mat retI420 = UretI420.getMat(cv::ACCESS_READ);
	memcpy_s(output, mWidth * mHeight * 3 / 2, retI420.data, mWidth * mHeight * 3 / 2);
	return true;
}




/**
* @@brief: У���ļ��Ƿ����
* @return: true-�ļ����ڣ�false-�ļ������ڣ�
* @param file: ��У���ļ�·��
**/
bool FaceBeauty::fileCheck(std::string& file) {
	struct stat buffer;
	return (stat(file.c_str(), &buffer) == 0);
}



/**
* @@brief: ��⻭���е���������������λÿ��������68���ؼ���
* @return: ��������������ؼ����ͼ��֡
* @param frame: ������ͼ��֡
**/
cv::Mat FaceBeauty::drawLandmarks(cv::Mat& frame) {
	cv::Mat cvImgFrameGray;
	cv::cvtColor(frame, cvImgFrameGray, cv::COLOR_BGR2GRAY);
	dlib::cv_image<unsigned char> dlibImgFrameGray(cvImgFrameGray);

	mDlibRectsFaces.clear();
	mDlibDetsShapes.clear();

	mDlibRectsFaces = mDlibSvmFaceDetector(dlibImgFrameGray);
	for (unsigned int idxFace = 0; idxFace < mDlibRectsFaces.size(); idxFace++) {
		mDlibDetsShapes.push_back(mDlibSpFaceLandmark(dlibImgFrameGray, mDlibRectsFaces[idxFace]));
		
		cv::rectangle(frame, cvRect(
			mDlibRectsFaces[idxFace].left(),
			mDlibRectsFaces[idxFace].top(),
			mDlibRectsFaces[idxFace].width(),
			mDlibRectsFaces[idxFace].height()),
			cv::Scalar(0, 255, 0), 2);

		for (unsigned int idxLandmark = 0; idxLandmark < mDlibSpFaceLandmark.num_parts(); idxLandmark++) {
			cv::circle(frame, cvPoint(
				mDlibDetsShapes[idxFace].part(idxLandmark).x(),
				mDlibDetsShapes[idxFace].part(idxLandmark).y()),
				3, cv::Scalar(0, 0, 255), -1);
		}
	}
	return frame;
}



/**
* @@brief: ��ǰ֡�ؼ�����
* @return: �����ؼ���
* @param img: �����ͼ��
**/
std::vector<dlib::full_object_detection> FaceBeauty::faceLandmarkDetect(cv::Mat& img) {
	cv::UMat UcvImgGray;
	cv::cvtColor(img.getUMat(cv::ACCESS_READ), UcvImgGray, cv::COLOR_BGR2GRAY);
	dlib::cv_image<unsigned char> dlibImgGray(UcvImgGray.getMat(cv::ACCESS_READ));
	
	mDlibRectsFaces.clear();
	mDlibDetsShapes.clear();

	mDlibRectsFaces = mDlibSvmFaceDetector(dlibImgGray);
	for (int idxFaces = 0; idxFaces < mDlibRectsFaces.size(); idxFaces++) {
		mDlibDetsShapes.push_back(mDlibSpFaceLandmark(dlibImgGray, mDlibRectsFaces[idxFaces]));
	}
	return mDlibDetsShapes;
}



/**
* @@brief: ����ֲ�˫���Բ�ֵ���
* @return: �ֲ�˫���Բ�ֵ��ֵ
* @param img: ����ֵͼ��
* @param ux: �任��ӳ��λ��ƫ��ԭ������x����
* @param uy: �任��ӳ��λ��ƫ��ԭ������y����
**/
cv::Vec3b FaceBeauty::bilinearInsert(cv::Mat& img, float ux, float uy) {
	img.convertTo(img, CV_32F);
	int x1 = (int)ux;
	int y1 = (int)uy;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	x1 = (x1 < img.cols ? x1 : img.cols - 1);
	x2 = (x2 < img.cols ? x2 : img.cols - 1);
	y1 = (y1 < img.rows ? y1 : img.rows - 1);
	y2 = (y2 < img.rows ? y2 : img.rows - 1);

	cv::Vec3f part1 = img.at<cv::Vec3f>(y1, x1)*((float)x2 - ux)*((float)y2 - uy);
	cv::Vec3f part2 = img.at<cv::Vec3f>(y1, x2)*(ux - (float)x1)*((float)y2 - uy);
	cv::Vec3f part3 = img.at<cv::Vec3f>(y2, x1)*((float)x2 - ux)*(uy - (float)y1);
	cv::Vec3f part4 = img.at<cv::Vec3f>(y2, x2)*(ux - (float)x1)*(uy - (float)y1);

	cv::Vec3f value = part1 + part2 + part3 + part4;
	cv::Vec3b insetValue = cv::Vec3b((int)value[0], (int)value[1], (int)value[2]);

	return insetValue;
}



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
cv::Mat FaceBeauty::bigEyeAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius) {
	float maxDistance = (float)(radius*radius);
	cv::Mat copyImg = img.clone();
	float distance = 0.0;
	float rNorm = 0.0;
	float ratio = 0.0;
	float ux = 0.0;
	float uy = 0.0;
	cv::Vec3b value = cv::Vec3b(0, 0, 0);
	for (int col = 0; col < img.cols; col++) {
		for (int row = 0; row < img.rows; row++) {
			if ((std::abs(col - startX) > radius) && (std::abs(row - startY) > radius)) {
				continue;
			}
			distance = (col - startX)*(col - startX) + (row - startY)*(row - startY);
			if (distance < maxDistance) {
				rNorm = std::sqrt(distance) / radius;
				if (mDistanceEye <= 6000) {
					ratio = 1 - ((rNorm - 1)*(rNorm - 1)*mDistanceEye*0.6 / 6000)*(mEnlargeEyeDegree / 5.0);
				}
				else {
					ratio = 1 - ((rNorm - 1)*(rNorm - 1)*0.4)*(mEnlargeEyeDegree / 5.0);
				}
				ux = startX + ratio * (col - startX);
				uy = startY + ratio * (row - startY);
				value = bilinearInsert(img, ux, uy);
				copyImg.at<cv::Vec3b>(row, col) = value;
			}
		}
	}
	return copyImg;
}



/**
* @@brief: �Ŵ��۾�
* @return: �۾��Ŵ���ͼ��
* @param img: ������ͼ��
* @param landmarks: ��⵽�������ؼ���
**/
cv::Mat FaceBeauty::bigEye(cv::Mat& img, std::vector < dlib::full_object_detection> landmarks) {
	if (landmarks.size() == 0) {
		return img;
	}
	else {
		for (int faceIndex = 0; faceIndex < landmarks.size(); faceIndex++) {
			dlib::point landmarkLeftUp = landmarks[faceIndex].part(38);
			int leftUpX = landmarkLeftUp.x();
			int leftUpY = landmarkLeftUp.y();
			dlib::point landmarkLeftDown = landmarks[faceIndex].part(27);
			int leftDownX = landmarkLeftDown.x();
			int leftDownY = landmarkLeftDown.y();
			dlib::point landmarkRightUp = landmarks[faceIndex].part(43);
			int rightUpX = landmarkRightUp.x();
			int rightUpY = landmarkRightUp.y();
			dlib::point landmarkRightDown = landmarks[faceIndex].part(27);
			int rightDownX = landmarkRightDown.x();
			int rightDownY = landmarkRightDown.y();

			dlib::point endPoint = landmarks[faceIndex].part(30);
			int endPointX = endPoint.x();
			int endPointY = endPoint.y();

			int distanceLeft = (endPointX - leftUpX)*(endPointX - leftUpX) + (endPointY - leftUpY)*(endPointY - leftUpY);
			int distanceRight = (endPointX - rightUpX)*(endPointX - rightUpX) + (endPointY - rightUpY)*(endPointY - rightUpY);
			mDistanceEye = distanceLeft < distanceRight ? distanceLeft : distanceRight;

			float radiusLeft = std::sqrt((leftUpX - leftDownX)*(leftUpX - leftDownX) + (leftUpY - leftDownY)*(leftUpY - leftDownY));
			float radiusRight = std::sqrt((rightUpX - rightDownX)*(rightUpX - rightDownX) + (rightUpY - rightDownY)*(rightUpY - rightDownY));

			cv::Mat bigEyeLeft = bigEyeAlgo(img, leftUpX, leftUpY, endPointX, endPointY, radiusLeft);
			img = bigEyeAlgo(bigEyeLeft, rightUpX, rightUpY, endPointX, endPointY, radiusRight);
		}
	}
	return img;
}



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
cv::Mat FaceBeauty::liftFaceAlgo(cv::Mat& img, int startX, int startY, int endX, int endY, int radius) {
	float maxDistance = (float)(radius*radius);
	cv::Mat copyImg = img.clone();
	float distance = 0.0;
	float rNorm = 0.0;
	float ratio = 0.0;
	float ux = 0.0;
	float uy = 0.0;
	float ddmc = (endX - startX)*(endX - startX) + (endY - startY)*(endY - startY);
	cv::Vec3b value = cv::Vec3b(0, 0, 0);
	for (int col = 0; col < img.cols; col++) {
		for (int row = 0; row < img.rows; row++) {
			if ((std::abs(col - startX) > radius) && (std::abs(row - startY) > radius)) {
				continue;
			}
			distance = (col - startX)*(col - startX) + (row - startY)*(row - startY);
			if (distance < maxDistance) {
				ratio = (maxDistance - distance) / (maxDistance - distance + ddmc);
				ratio = ratio * ratio * mShrinkFaceDegree / 5.0;
				if (mDistanceFace <= 15000) {
					ratio = ratio * mDistanceFace / 15000;
				}
				ux = col - ratio * (endX - startX);
				uy = row - ratio * (endY - startY);
				value = bilinearInsert(img, ux, uy);
				copyImg.at<cv::Vec3b>(row, col) = value;
			}
		}
	}
	return copyImg;
}




/**
* @@brief: ����
* @return: �������ͼ��
* @param img: ������ͼ��
* @param landmarks: ��⵽�������ؼ���
**/
cv::Mat FaceBeauty::liftFace(cv::Mat& img, std::vector<dlib::full_object_detection> landmarks) {
	if (landmarks.size() == 0) {
		return img;
	}
	else {
		for (int faceIndex = 0; faceIndex < landmarks.size(); faceIndex++) {
			dlib::point landmarkLeftUp = landmarks[faceIndex].part(3);
			int leftUpX = landmarkLeftUp.x();
			int leftUpY = landmarkLeftUp.y();
			dlib::point landmarkLeftDown = landmarks[faceIndex].part(6);
			int leftDownX = landmarkLeftDown.x();
			int leftDownY = landmarkLeftDown.y();
			dlib::point landmarkRightUp = landmarks[faceIndex].part(12);
			int rightUpX = landmarkRightUp.x();
			int rightUpY = landmarkRightUp.y();
			dlib::point landmarkRightDown = landmarks[faceIndex].part(15);
			int rightDownX = landmarkRightDown.x();
			int rightDownY = landmarkRightDown.y();

			dlib::point endPoint = landmarks[faceIndex].part(30);
			int endPointX = endPoint.x();
			int endPointY = endPoint.y();

			int distanceLeft = (endPointX - leftUpX)*(endPointX - leftUpX) + (endPointY - leftUpY)*(endPointY - leftUpY);
			int distanceRight = (endPointX - rightUpX)*(endPointX - rightUpX) + (endPointY - rightUpY)*(endPointY - rightUpY);
			mDistanceFace = distanceLeft < distanceRight ? distanceLeft : distanceRight;


			float radiusLeft = std::sqrt((leftUpX - leftDownX)*(leftUpX - leftDownX) + (leftUpY - leftDownY)*(leftUpY - leftDownY));
			float radiusRight = std::sqrt((rightUpX - rightDownX)*(rightUpX - rightDownX) + (rightUpY - rightDownY)*(rightUpY - rightDownY));

			cv::Mat liftFaceLeft = liftFaceAlgo(img, leftUpX, leftUpY, endPointX, endPointY, radiusLeft);
			img = liftFaceAlgo(liftFaceLeft, rightUpX, rightUpY, endPointX, endPointY, radiusRight);
		}
	}
	return img;
}



/**
* @@brief: ���ʱ�����ĥƤ�����㷨
* @return: ĥƤ���ͼ��
* @param img: ԭʼ����ͼ��
**/
cv::Mat FaceBeauty::skinDenise(cv::Mat& img) {
	cv::UMat Uimg = img.getUMat(cv::ACCESS_READ);

	int deniseDegree = 3;
	int textureDegree = 2;
	int dx = deniseDegree * 5;
	int fc = deniseDegree * 12.5;
	double transparency = 0.1;
	cv::UMat temp1;
	cv::UMat temp2;
	cv::UMat temp3;
	cv::UMat temp4;

	cv::bilateralFilter(Uimg, temp1, dx, fc, fc);

	cv::UMat temp22;
	
	cv::subtract(temp1, Uimg, temp22);
	cv::add(temp22, cv::Scalar(128, 128, 128, 128), temp2);
	cv::GaussianBlur(temp2, temp3, cv::Size(2 * textureDegree - 1, 2 * textureDegree - 1), 0, 0);

	cv::UMat temp44;
	temp3.convertTo(temp44, temp3.type(), 2, -255);
	cv::add(Uimg, temp44, temp4);
	cv::UMat dst;
	cv::addWeighted(Uimg, transparency, temp4, 1 - transparency, 0.0, dst);
#ifdef SHARPEN
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	cv::UMat Ukernel = kernel.getUMat(cv::ACCESS_READ);
	cv::filter2D(dst, dst, dst.depth(), Ukernel);
#endif // SHARPEN
	cv::Mat ret;
	cv::add(dst.getMat(cv::ACCESS_READ), cv::Scalar(10, 10, 10), ret);

	return ret;
}



/**
* @@brief: ����Ӧͼ��ֲ��Աȶ���ǿ
* @return: �Աȶ���ǿ���ͼ��
* @param img: ԭʼ����ͼ��
* @param winSize: �ֲ���ֵ���ڴ�С
* @param maxCg: ��ǿ���ȵ�����
**/
cv::Mat FaceBeauty::contrastEnhance(cv::Mat& img, int winSize, int maxCg) {
	cv::Mat ycc;
	cv::cvtColor(img, ycc, cv::COLOR_BGR2YCrCb);

	std::vector<cv::Mat> channels(3);
	cv::split(ycc, channels);

	cv::Mat localMeansMatrix(img.rows, img.cols, CV_32FC1);
	cv::Mat localVarianceMatrix(img.rows, img.cols, CV_32FC1);

	cv::Mat temp = channels[0].clone();

	cv::Scalar mean;
	cv::Scalar dev;
	cv::meanStdDev(temp, mean, dev);

	float meansGlobal = mean.val[0];
	cv::Mat enhanceMatrix(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (localVarianceMatrix.at<float>(i, j) >= 0.01) {
				float cg = 0.2*meansGlobal / localVarianceMatrix.at<float>(i, j);
				float cgs = cg > maxCg ? maxCg : cg;
				cgs = cgs < 1 ? 1 : cgs;

				int e = localMeansMatrix.at<float>(i, j) + cgs * (temp.at<uchar>(i, j) - localMeansMatrix.at<float>(i, j));
				if (e > 255) {
					e = 255;
				}
				else if (e < 0) {
					e = 0;
				}
				enhanceMatrix.at<uchar>(i, j) = e;
			}
			else {
				enhanceMatrix.at<uchar>(i, j) = temp.at<uchar>(i, j);
			}
		}
	}
	channels[0] = enhanceMatrix;
	cv::merge(channels, ycc);
	cv::Mat dst;
	cv::cvtColor(ycc, dst, cv::COLOR_YCrCb2BGR);
	return dst;
}


