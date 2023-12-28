#ifndef EDGE_DEVICE_SLAM_MAP_H
#define EDGE_DEVICE_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include "ConcurrentMap.h"
#include "ConcurrentVector.h"
#include "ConcurrentVariable.h"

namespace EdgeDeviceSLAM {
	class Frame;
	class RefFrame;
	class MapPoint;

	class Map {
	public:
		Map();
		virtual ~Map();
    public:
		ConcurrentVector<MapPoint*> LocalMapPoints;
        ConcurrentMap<int, MapPoint*> MapPoints;
		ConcurrentVariable<RefFrame*> ReferenceFrame;
	private:
	};
}

#endif