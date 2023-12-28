#include "./EdgeDeviceSLAM/include/Map.h"
//#include "MapPoint.h"
//#include "RefFrame.h"

namespace EdgeDeviceSLAM {
	Map::Map(){

	}
	Map::~Map() {
		ReferenceFrame.Release();
        auto mapMPs = MapPoints.Get();
        for(auto iter = mapMPs.begin(), iend = mapMPs.end(); iter != iend; iter++)
            delete iter->second;
        MapPoints.Release();
		LocalMapPoints.Release();
	}
}