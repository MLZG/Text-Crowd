#include "AgentCar.h"
#include "Obstacle.h"
#include <fstream>
#include <queue>
#include <set>

void RVO::AgentCar::track() {
	car.control();
	car.update();
	Agent::position_ = car.getCenter();
}

void RVO::AgentCar::computeNeighbors() {
	Agent::computeNeighbors();
	float minDistSq = INFINITY;
	float minObsDistSq = INFINITY;
	for (size_t i = 0; i < Agent::agentNeighbors_.size(); i++)
	{
		float distSq = Agent::agentNeighbors_[i].first;
		if (distSq < minDistSq)
		{
			minDistSq = distSq;
		}
	}
	for (size_t i = 0; i < Agent::obstacleNeighbors_.size(); i++)
	{
		float distSq = Agent::obstacleNeighbors_[i].first;
		if (distSq < minObsDistSq)
		{
			minObsDistSq = distSq;
		}
	}
	Agent::radius_ = Car::radius + fmin(fmin(Car::errorDesired, sqrt(minDistSq) / 2 - Car::radius), sqrt(minObsDistSq) - Car::radius);
}

void RVO::AgentCar::computeNewVelocity() {
	RVO::Vector2 vd;
	bool find = false;
	Agent::timeHorizon_ = Agent::sim_->defaultAgent_->timeHorizon_;
	Agent::timeHorizonObst_ = Agent::sim_->defaultAgent_->timeHorizonObst_;
	do
	{
		Agent::computeNewVelocity();
		find = searchDesiredVelocity(vd);
		if (Agent::orcaLines_.size() == 0)
			break;
		Agent::timeHorizon_ /= 2;
		//Agent::timeHorizonObst_ /= 2;

	} while (!find && Agent::timeHorizon_ > Car::decelerateTime);

	if (find)
	{
		car.initTracking(vd);
		Agent::lambda = 0.5f;
	}
	else
	{
		car.initDecelerating();
		Agent::lambda = 0.f;
	}
}
bool RVO::AgentCar::searchDesiredVelocity(RVO::Vector2& vd) {
	const auto& vi = Agent::newVelocity_;
	const auto& orca = Agent::orcaLines_;
	const float errorMax = Agent::radius_ - Car::radius;
	
	float cosTheta = cos(car.theta);
	float sinTheta = sin(car.theta);
	float vxWorld = vi.x();
	float vyWorld = vi.y();
	float vxLocal = vxWorld * cosTheta + vyWorld * sinTheta;
	float vyLocal = -vxWorld * sinTheta + vyWorld * cosTheta;
	int iv0, iphi, ix, iy;
	float error;

	Car::getCarIndex(car, iv0, iphi);
	Car::getVelocityIndex(vxLocal, vyLocal, ix, iy);
	error = Car::errorLookupTable[iv0][iphi][ix][car.phi > 0 ? (Car::vNum - iy - 1) : iy];
	if (error < errorMax)
	{
		vd = vi;
		return true;
	}

	std::set<std::pair<int, int>> ivset;
	std::queue<std::pair<int, int>> ivqueue;
	auto op = [&ivset, &ivqueue](int ix, int iy) {
		std::pair<int, int> up(ix, iy + 1), down(ix, iy - 1), left(ix - 1, iy), right(ix + 1, iy);
		if (ivset.insert(up).second)
		{
			ivqueue.push(up);
		}
		if (ivset.insert(down).second)
		{
			ivqueue.push(down);
		}
		if (ivset.insert(left).second)
		{
			ivqueue.push(left);
		}
		if (ivset.insert(right).second)
		{
			ivqueue.push(right);
		}
	};
	auto inAgentORCA = [](const RVO::Vector2& vd, const std::vector<RVO::Line>& orcaLines_) {
		for (size_t i = 0; i < orcaLines_.size(); ++i) {
			if (det(orcaLines_[i].direction, orcaLines_[i].point - vd) > Car::deltav1) {
				return false;
			}
		}
		return true;
	};
	ivset.emplace(ix, iy);
	op(ix, iy);
	while (!ivqueue.empty())
	{
		auto iv = ivqueue.front();
		ivqueue.pop();
		ix = iv.first;
		iy = iv.second;
		if (ix >= 0 && ix < Car::vNum && iy >= 0 && iy < Car::vNum)
		{
			error = Car::errorLookupTable[iv0][iphi][ix][car.phi > 0 ? (Car::vNum - iy - 1) : iy];
			vxLocal = ix * Car::deltav1 - Car::vDrivingMax;
			vyLocal = iy * Car::deltav1 - Car::vDrivingMax;
			vxWorld = vxLocal * cosTheta - vyLocal * sinTheta;
			vyWorld = vxLocal * sinTheta + vyLocal * cosTheta;
			RVO::Vector2 vt(vxWorld, vyWorld);
			if (inAgentORCA(vt, orca))
			{
				if (error < errorMax)
				{
					vd = vt;
					if (vt * vi > 0)
						return true;
				}
				op(ix, iy);
			}
		}
	}
	return false;
}

void RVO::AgentCar::update() {
	Agent::velocity_ = car.getVelocity();
	Agent::position_ = car.getCenter();
}
