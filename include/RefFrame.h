#ifndef EDGE_DEVICE_SLAM_REFFRAME_H
#define EDGE_DEVICE_SLAM_REFFRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "Utils.h"
#include <mutex>

namespace EdgeDeviceSLAM {
    class Map;
	class Camera;
	class CameraPose;
	class ORBDetector;
	class MapPoint;
	class RefFrame {
	public:
		RefFrame();
		RefFrame(Camera* pCam, float* data, Map* pMap);
		RefFrame(Camera* pCam, cv::Mat desc, float* data, Map* pMap);
		virtual ~RefFrame();
    public:
		Map* mpMap;
		Camera* mpCamera;
		static ORBDetector* Detector;
	public:
		//int TrackedMapPoints(const int &minObs);
		void EraseMapPointMatch(const size_t &idx);
        void UpdateMapPoints();
        bool is_in_frustum(MapPoint* pMP, float viewingCosLimit);
	public:
		static bool weightComp(int a, int b) {
			return a>b;
		}

		static bool lId(RefFrame* pKF1, RefFrame* pKF2) {
			return pKF1->mnId<pKF2->mnId;
		}
	public:
		int N;
		int mnId;
		//cv::Mat imgGray;
		static int nId;
		cv::Mat K, D;
		float fx, fy, cx, cy, invfx, invfy;
		bool mbDistorted;
		int FRAME_GRID_COLS;
		int FRAME_GRID_ROWS;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;

		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		float mnMinX;
		float mnMaxX;
		float mnMinY;
		float mnMaxY;

	public:
		
		std::vector<cv::KeyPoint> mvKeys,mvKeysUn;
		std::vector<bool> mvbOutliers;
		std::vector<MapPoint*> mvpMapPoints;
		cv::Mat mDescriptors;
	private:
        void UndistortKeyPoints();
	private:
		std::mutex mMutexFeatures;
	public:
		CameraPose* mpCamPose;
		void SetPose(cv::Mat Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();

	};
}
#endif