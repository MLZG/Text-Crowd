#pragma once
#include "RVO.h"

class Car
{
	friend class RVOSimulator;
public:
	static float phiMax, vDrivingMax, steeringMax, aDrivingMax,
		length, radius,
		deltaPhi, deltav1, dtc, errorDesired,
		decelerateTime;
	static float lengthHalf;
	static int vNum, phiNum;
	static float ka, kv, kp;
	static std::vector<std::vector<std::vector<std::vector<float>>>> errorLookupTable;
public:
	static void initCars(float length, float radius, float  vDrivingMax, float steeringMax, float aDrivingMax, float phiMax, float dtc, float errorDesired, float ka, float kv, float kp);

	static void computeLookupTable(float timeConsumed, float errorClamped);

	static void saveToFile();

	static bool readFromFile();

	inline static void getVelocityIndex(float vx, float vy, int& ix, int& iy) {
		ix = (int)((vx + vDrivingMax) / deltav1);
		iy = (int)((vy + vDrivingMax) / deltav1);
	}

	inline static void getCarIndex(const Car& car, int& iv0, int& iphi) {
		iv0 = (int)((car.v1 + vDrivingMax) / deltav1);
		iphi = (int)((-abs(car.phi) + phiMax) / deltaPhi);
	}

	inline static RVO::Vector2 preferredVelocity(const RVO::Vector2& dir) {
		float t = RVO::abs(dir) / Car::vDrivingMax;

		float v = Car::aDrivingMax * t;/* / ( abs(v1) / aDrivingMax)*/;
		return fmin(vDrivingMax, v) * RVO::normalize(dir);
	}

	inline static bool inAgentORCA(const RVO::Vector2& vd, const std::vector<RVO::Line>& orcaLines_) {
		for (size_t i = 0; i < orcaLines_.size(); ++i) {
			if (det(orcaLines_[i].direction, orcaLines_[i].point - vd) > 0.01f) {
				return false;
			}
		}
		return true;
	}

public:
	Car(float x, float y, float theta, float phi);

	void control();

	void track();

	void update();

	void initTracking(const RVO::Vector2& vd);

	void initDecelerating();

	inline float getError() const {
		return sqrt(pow(zDesired1 - center1, 2) + pow(zDesired2 - center2, 2));
	}

	inline RVO::Vector2 getVelocity() {
		return RVO::Vector2(v1 * cos(theta), v1 * sin(theta));
	}

	inline RVO::Vector2 getCenter() const {
		return RVO::Vector2(center1, center2);
	}

public:
	float backX, backY, theta, phi;
	float center1, center2;
	float zDesired1, zDesired2;
	float vDesired1, vDesired2;
private:
	float thetaDesired;
	float v1, v2;
	float xi1;
	float xi2;
	bool decelerating;
	//float s1;
};



