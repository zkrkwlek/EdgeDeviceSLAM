#ifndef EDGE_DEVICE_SLAM_MAPPOINT_H
#define EDGE_DEVICE_SLAM_MAPPOINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeDeviceSLAM {
	class RefFrame;
	class Frame;
	class ORBDetector;
	class Map;

	class MapPoint {
	public:
		MapPoint();
		MapPoint(int id, float _x, float _y, float _z, Map* pMap);
		virtual ~MapPoint();

	public:
		void SetWorldPos(float x, float y, float z);
		cv::Mat GetWorldPos();
		void SetDescriptor(const cv::Mat& _desc);
		cv::Mat GetDescriptor();
        void UpdateNormalAndDepth();
        void SetMapPointInfo(float _min, float _max, const cv::Mat& _norm);
	public:
		RefFrame* mpRefKF;
		static ORBDetector* Detector;
		Map* mpMap;
    public:
	    int mnID;
		float mTrackProjX;
        float mTrackProjY;
        bool mbTrackInView;
        int mnTrackScaleLevel;
        float mTrackViewCos;

		cv::Mat GetNormal();
		std::map<RefFrame*, size_t> GetObservations();
		int Observations();
		void AddObservation(RefFrame* pKF, size_t idx);
		void EraseObservation(RefFrame* pKF);
		void ComputeDistinctiveDescriptors();

		float GetMinDistanceInvariance();
		float GetMaxDistanceInvariance();
		int PredictScale(const float &currentDist, Frame* pF);
		int PredictScale(const float &currentDist, RefFrame* pKF);

		bool IsInKeyFrame(RefFrame *pKF);
        void SetBadFlag();
        bool isBad();

	private:
		cv::Mat mWorldPos, mNormalVector;
		std::map<RefFrame*, size_t> mObservations;
		cv::Mat mDescriptor;

		int nObs;
        bool mbBad;
		float mfMinDistance;
		float mfMaxDistance;

		std::mutex mMutexPos;
		std::mutex mMutexFeatures;



	};
}
#endif