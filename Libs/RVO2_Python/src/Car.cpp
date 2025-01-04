#include "Car.h"
#include <queue>
#include <set>
#include <fstream>
#include <iostream>

float Car::phiMax, Car::vDrivingMax, Car::steeringMax, Car::aDrivingMax, Car::length, Car::radius, Car::deltaPhi, Car::deltav1, Car::dtc, Car::errorDesired;
float Car::decelerateTime;
float Car::ka, Car::kv, Car::kp;
float Car::lengthHalf;
int Car::vNum, Car::phiNum;
std::vector<std::vector<std::vector<std::vector<float>>>> Car::errorLookupTable;

void Car::initCars(float length, float radius, float vDrivingMax, float steeringMax, float aDrivingMax, float phiMax, float dtc, float errorDesired, float ka, float kv, float kp) {
	Car::length = length;
	Car::radius = radius;

	Car::phiMax = phiMax;
	Car::vDrivingMax = vDrivingMax;
	Car::steeringMax = steeringMax;
	Car::aDrivingMax = aDrivingMax;

	Car::dtc = dtc;
	Car::errorDesired = errorDesired;

	//feedback gains solved from λ3+kaλ2+kvλ+kp
	Car::ka = ka;
	Car::kv = kv;
	Car::kp = kp;

	lengthHalf = length / 2;
	decelerateTime = vDrivingMax / aDrivingMax;
}

void Car::computeLookupTable(float timeConsumed, float errorClamped) {
	errorLookupTable.resize(vNum);
	for (auto& phivec : errorLookupTable)
	{
		phivec.resize(phiNum);
		for (auto& vxvec : phivec)
		{
			vxvec.resize(vNum);
			for (auto& vyvec : vxvec)
			{
				vyvec.resize(vNum);
			}
		}
	}

#pragma omp parallel for
	for (int iv0 = 0; iv0 < vNum; iv0++)
	{
		float v0 = -vDrivingMax + iv0 * deltav1;
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			float phi = -phiMax + iphi * deltaPhi;
			//for each desired velocity
			for (int ivx = 0; ivx < vNum; ivx++)
			{
				float vx = -vDrivingMax + ivx * deltav1;
				for (int ivy = 0; ivy < vNum; ivy++)
				{
					Car car(0, 0, 0, phi);
					car.v1 = v0;
					float vy = -vDrivingMax + ivy * deltav1;
					car.initTracking(RVO::Vector2(vx, vy));
					car.track();
					float t = 0;
					float maxError = 0;
					float error = 0;
					while (true)
					{
						car.update();
						t += Car::dtc;
						error = car.getError();
						if (error > maxError)
							maxError = error;
						else if (t > timeConsumed)
							break;
						if (maxError > errorClamped)
							break;
						car.track();
					}
					errorLookupTable[iv0][iphi][ivx][ivy] = maxError;
				}
			}
		}
	}
	//omp for
}

void Car::saveToFile() {
	std::ifstream infile("lookup");
	if (infile.good())
	{
		return;
	}
	std::ofstream file("lookup", std::ofstream::out);
	for (int iv = 0; iv < vNum; iv++)
	{
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			for (int ix = 0; ix < vNum; ix++)
			{
				for (int iy = 0; iy < vNum; iy++)
				{
					file << errorLookupTable[iv][iphi][ix][iy] << " ";
				}
			}
		}
	}
}

bool Car::readFromFile() {
	std::ifstream file("lookup", std::ifstream::in);
	if (!file.is_open())
	{
		return false;
	}
	errorLookupTable.resize(vNum);
	for (auto& phivec : errorLookupTable)
	{
		phivec.resize(phiNum);
		for (auto& vxvec : phivec)
		{
			vxvec.resize(vNum);
			for (auto& vyvec : vxvec)
			{
				vyvec.resize(vNum);
			}
		}
	}
	for (int iv = 0; iv < vNum; iv++)
	{
		for (int iphi = 0; iphi < phiNum; iphi++)
		{
			for (int ix = 0; ix < vNum; ix++)
			{
				for (int iy = 0; iy < vNum; iy++)
				{
					file >> errorLookupTable[iv][iphi][ix][iy];
				}
			}
		}
	}
	return true;
}

Car::Car(float centerx, float centery, float theta, float phi) :
	v1(1 / INTMAX_MAX),
	v2(0.f),
	center1(centerx), center2(centery), theta(theta), phi(phi),
	thetaDesired(0), vDesired1(0), vDesired2(0), xi1(0), xi2(0), zDesired1(0), zDesired2(0)
{
	backX = center1 - lengthHalf * cos(theta);
	backY = center2 - lengthHalf * sin(theta);
	initTracking(RVO::Vector2(0, 0));
}

void Car::control()
{
	if (decelerating)
	{
		float a = fmin(abs(v1 / dtc), aDrivingMax);
		v1 -= copysign(a, v1) * dtc;
	}
	else
	{
		track();
	}
}

void Car::track()
{
	const float cosTheta = cos(theta);
	const float sinTheta = sin(theta);
	const float tanPhi = tan(phi);
	//gai
	const float cosPhiSq = cos(pow(phi, 2));
	const float xi1Sq = pow(xi1, 2);//xi1 * xi1;
	const float xi1Cub = xi1Sq * xi1;

	const float zd1 = xi1 * cosTheta;
	const float zd2 = xi1 * sinTheta;
	const float zdd1 = -(xi1Sq)*tanPhi * sinTheta / length + xi2 * cosTheta;
	const float	zdd2 = xi1Sq * tanPhi * cosTheta / length + xi2 * sinTheta;

	//应适当增大/减小vdesired
	//cosθdcosθ(t) +sinθdsinθ(t)<1
	//float factor = cos(thetaDesired) * cosTheta + sin(thetaDesired) * sinTheta;
	//factor = 1 - abs(factor);

	float r1 = -ka * zdd1 + kv * (vDesired1 - zd1) + kp * (zDesired1 - center1);
	float r2 = -ka * zdd2 + kv * (vDesired2 - zd2) + kp * (zDesired2 - center2);
	//float r1 = -ka * zdd1 + kv * (vDesired1 - zd1) + kp * (zDesired1 - backX);
	//float r2 = -ka * zdd2 + kv * (vDesired2 - zd2) + kp * (zDesired2 - backY);

	//更新xi
	float xi2d = xi1Cub * tan(pow(phi, 2)) / (pow(length, 2)) + r1 * cosTheta + r2 * sinTheta;
	xi2 += xi2d * dtc;
	xi2 = fmax(fmin(xi2, aDrivingMax), -aDrivingMax);
	xi1 += xi2 * dtc;
	xi1 = fmax(fmin(xi1, vDrivingMax), -vDrivingMax);

	//更新速度
	v1 = xi1;
	v2 = -3 * xi2 * cosPhiSq * tanPhi / xi1 - length * r1 * cosPhiSq * sinTheta / xi1Sq + length * r2 * cosPhiSq * cosTheta / xi1Sq;
	v2 = fmax(fmin(v2, steeringMax), -steeringMax);
}

void Car::update() {
	//更新位置
	backX += v1 * dtc * cos(theta);
	backY += v1 * dtc * sin(theta);
	theta += v1 * tan(phi) * dtc / length;
	center1 = backX + lengthHalf * cos(theta);
	center2 = backY + lengthHalf * sin(theta);
	phi += v2 * dtc;
	phi = fmax(fmin(phi, phiMax), -phiMax);

	zDesired1 += vDesired1 * dtc;
	zDesired2 += vDesired2 * dtc;
}

void Car::initTracking(const RVO::Vector2& vd)
{
	vDesired1 = vd.x();
	vDesired2 = vd.y();
	thetaDesired = atan2(vDesired2, vDesired1);
	xi1 = v1;//xi1=v0=v1(0)
	xi2 = 0.f;
	//float thetaICR = copysign((PI / 2.f - abs(atan2(2.f, abs(tan(phi))))), phi) + theta;
	//float s1 = copysign(1, cos(thetaDesired) * cos(thetaICR) + sin(thetaDesired) * sin(thetaICR));
	//跟踪中心位置
	zDesired1 = center1 /*- (s1 * lengthHalf) * cos(thetaDesired)*/;
	zDesired2 = center2 /*- (s1 * lengthHalf) * sin(thetaDesired)*/;

	decelerating = false;
}

void Car::initDecelerating() {
	vDesired1 = 0.f;
	vDesired2 = 0.f;
	thetaDesired = 0.f;
	decelerating = true;
}

