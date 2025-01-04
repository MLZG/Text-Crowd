/*
 * RVOSimulator.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

#include "RVOSimulator.h"

#include "Agent.h"
#include "KdTree.h"
#include "Obstacle.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "AgentCar.h"

namespace RVO {
	RVOSimulator::RVOSimulator() : defaultAgent_(NULL), globalTime_(0.0f), kdTree_(NULL), timeStep_(0.0f), carTime_(0.f)
	{
		kdTree_ = new KdTree(this);
	}

	RVOSimulator::RVOSimulator(float timeStep, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity) : defaultAgent_(NULL), globalTime_(0.0f), kdTree_(NULL), timeStep_(timeStep), carTime_(0.f)
	{
		kdTree_ = new KdTree(this);
		defaultAgent_ = new Agent(this);

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->timeHorizonObst_ = timeHorizonObst;
		defaultAgent_->velocity_ = velocity;
	}

	RVOSimulator::~RVOSimulator()
	{
		if (defaultAgent_ != NULL) {
			delete defaultAgent_;
		}

		for (size_t i = 0; i < agents_.size(); ++i) {
			delete agents_[i];
		}

		for (size_t i = 0; i < obstacles_.size(); ++i) {
			delete obstacles_[i];
		}

		delete kdTree_;
	}

	size_t RVOSimulator::addAgent(const Vector2& position)
	{
		if (defaultAgent_ == NULL) {
			return RVO_ERROR;
		}

		Agent* agent = new Agent(this);

		agent->position_ = position;
		agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
		agent->maxSpeed_ = defaultAgent_->maxSpeed_;
		agent->neighborDist_ = defaultAgent_->neighborDist_;
		agent->radius_ = defaultAgent_->radius_;
		agent->timeHorizon_ = defaultAgent_->timeHorizon_;
		agent->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;
		agent->velocity_ = defaultAgent_->velocity_;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	size_t RVOSimulator::addAgentCar(const Vector2& position)
	{
		if (defaultAgent_ == NULL) {
			return RVO_ERROR;
		}

		Agent* car = new AgentCar(this, position, 0, 0);
		car->velocity_ = defaultAgent_->velocity_;
		car->maxSpeed_ = Car::vDrivingMax;
		car->radius_ = Car::radius;
		car->position_ = position;
		car->timeHorizon_ = defaultAgent_->timeHorizon_;
		car->maxNeighbors_ = defaultAgent_->maxNeighbors_;
		car->neighborDist_ = defaultAgent_->neighborDist_;
		car->timeHorizon_ = defaultAgent_->timeHorizon_;
		car->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;
		car->lambda = defaultAgent_->lambda;
		car->id_ = agents_.size();

		agents_.push_back(car);
		size_t index = agents_.size() - 1;
		carIndices_.push_back(index);

		return index;
	}

	void RVOSimulator::setCarLookupTable()
	{
		if(!Car::readFromFile()) {
			Car::computeLookupTable(getTimeStep(),5);
			Car::saveToFile();
		}
	}

	size_t RVOSimulator::addAgent(const Vector2& position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity)
	{
		Agent* agent = new Agent(this);

		agent->position_ = position;
		agent->maxNeighbors_ = maxNeighbors;
		agent->maxSpeed_ = maxSpeed;
		agent->neighborDist_ = neighborDist;
		agent->radius_ = radius;
		agent->timeHorizon_ = timeHorizon;
		agent->timeHorizonObst_ = timeHorizonObst;
		agent->velocity_ = velocity;

		agent->id_ = agents_.size();

		agents_.push_back(agent);

		return agents_.size() - 1;
	}

	size_t RVOSimulator::addAgentCar(const Vector2& position, float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity, float lambda)
	{
		Agent* car = new AgentCar(this, position, 0, 0);
		car->position_ = position;
		car->maxNeighbors_ = maxNeighbors;
		car->maxSpeed_ = maxSpeed;
		car->neighborDist_ = neighborDist;
		car->radius_ = radius;
		car->timeHorizon_ = timeHorizon;
		car->timeHorizonObst_ = timeHorizonObst;
		car->velocity_ = velocity;
		car->id_ = agents_.size();
		car->lambda = defaultAgent_->lambda;

		agents_.push_back(car);
		size_t index = agents_.size() - 1;
		carIndices_.push_back(index);

		return index;
	}

	size_t RVOSimulator::addObstacle(const std::vector<Vector2>& vertices)
	{
		if (vertices.size() < 2) {
			return RVO_ERROR;
		}

		const size_t obstacleNo = obstacles_.size();

		for (size_t i = 0; i < vertices.size(); ++i) {
			Obstacle* obstacle = new Obstacle();
			obstacle->point_ = vertices[i];

			if (i != 0) {
				obstacle->prevObstacle_ = obstacles_.back();
				obstacle->prevObstacle_->nextObstacle_ = obstacle;
			}

			if (i == vertices.size() - 1) {
				obstacle->nextObstacle_ = obstacles_[obstacleNo];
				obstacle->nextObstacle_->prevObstacle_ = obstacle;
			}

			obstacle->unitDir_ = normalize(vertices[(i == vertices.size() - 1 ? 0 : i + 1)] - vertices[i]);

			if (vertices.size() == 2) {
				obstacle->isConvex_ = true;
			}
			else {
				obstacle->isConvex_ = (leftOf(vertices[(i == 0 ? vertices.size() - 1 : i - 1)], vertices[i], vertices[(i == vertices.size() - 1 ? 0 : i + 1)]) >= 0.0f);
			}

			obstacle->id_ = obstacles_.size();

			obstacles_.push_back(obstacle);
		}

		return obstacleNo;
	}

	void RVOSimulator::clearObstacle() {
		for (size_t i = 0; i < obstacles_.size(); ++i) {
			delete obstacles_[i];
		}
		obstacles_.clear();
	}

	void RVOSimulator::doStep()
	{
		kdTree_->buildAgentTree();

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->computeNeighbors();
			agents_[i]->computeNewVelocity();
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->update();
		}

		globalTime_ += timeStep_;
	}

	void RVOSimulator::computeAgents() {
		kdTree_->buildAgentTree();

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->computeNeighbors();
			agents_[i]->computeNewVelocity();
		}
	}

	void RVOSimulator::updateAgents() {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
			agents_[i]->update();
		}

		globalTime_ += timeStep_;
	}

	void RVOSimulator::doStepCar() {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < static_cast<int>(carIndices_.size()); ++i) {
			static_cast<AgentCar*>(agents_[carIndices_[i]])->track();
		}
		carTime_ += Car::dtc;
	}

	size_t RVOSimulator::getAgentAgentNeighbor(size_t agentNo, size_t neighborNo) const
	{
		return agents_[agentNo]->agentNeighbors_[neighborNo].second->id_;
	}

	size_t RVOSimulator::getAgentMaxNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->maxNeighbors_;
	}

	AgentCar* RVOSimulator::getAgentCar(size_t agentNo) const {
		return (AgentCar*)agents_[carIndices_[agentNo]];
	}

	float RVOSimulator::getAgentCarTheta(size_t agentNo) const
	{
		return ((AgentCar*)agents_[carIndices_[agentNo]])->car.theta;
	}

	float RVOSimulator::getAgentCarPhi(size_t agentNo) const
	{
		return ((AgentCar*)agents_[carIndices_[agentNo]])->car.phi;
	}

	float RVOSimulator::getAgentMaxSpeed(size_t agentNo) const
	{
		return agents_[agentNo]->maxSpeed_;
	}

	float RVOSimulator::getAgentNeighborDist(size_t agentNo) const
	{
		return agents_[agentNo]->neighborDist_;
	}

	size_t RVOSimulator::getAgentNumAgentNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->agentNeighbors_.size();
	}

	size_t RVOSimulator::getAgentNumObstacleNeighbors(size_t agentNo) const
	{
		return agents_[agentNo]->obstacleNeighbors_.size();
	}

	size_t RVOSimulator::getAgentNumORCALines(size_t agentNo) const
	{
		return agents_[agentNo]->orcaLines_.size();
	}

	size_t RVOSimulator::getAgentObstacleNeighbor(size_t agentNo, size_t neighborNo) const
	{
		return agents_[agentNo]->obstacleNeighbors_[neighborNo].second->id_;
	}

	const Line& RVOSimulator::getAgentORCALine(size_t agentNo, size_t lineNo) const
	{
		return agents_[agentNo]->orcaLines_[lineNo];
	}


	const std::vector<Line>& RVOSimulator::getAgentORCA(size_t agentNo) const {
		return agents_[agentNo]->orcaLines_;
	}

	const Vector2& RVOSimulator::getAgentPosition(size_t agentNo) const
	{
		return agents_[agentNo]->position_;
	}

	const Vector2& RVOSimulator::getAgentPrefVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->prefVelocity_;
	}

	float RVOSimulator::getAgentRadius(size_t agentNo) const
	{
		return agents_[agentNo]->radius_;
	}

	float RVOSimulator::getAgentTimeHorizon(size_t agentNo) const
	{
		return agents_[agentNo]->timeHorizon_;
	}

	float RVOSimulator::getAgentTimeHorizonObst(size_t agentNo) const
	{
		return agents_[agentNo]->timeHorizonObst_;
	}

	const Vector2& RVOSimulator::getAgentVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->velocity_;
	}

	const Vector2& RVOSimulator::getAgentNewVelocity(size_t agentNo) const
	{
		return agents_[agentNo]->newVelocity_;
	}

	float RVOSimulator::getGlobalTime() const
	{
		return globalTime_;
	}

	size_t RVOSimulator::getNumAgents() const
	{
		return agents_.size();
	}

	size_t RVOSimulator::getNumAgentCars() const
	{
		return carIndices_.size();
	}

	size_t RVOSimulator::getNumObstacleVertices() const
	{
		return obstacles_.size();
	}

	const Vector2& RVOSimulator::getObstacleVertex(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->point_;
	}

	size_t RVOSimulator::getNextObstacleVertexNo(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->nextObstacle_->id_;
	}

	size_t RVOSimulator::getPrevObstacleVertexNo(size_t vertexNo) const
	{
		return obstacles_[vertexNo]->prevObstacle_->id_;
	}

	float RVOSimulator::getTimeStep() const
	{
		return timeStep_;
	}

	void RVOSimulator::processObstacles()
	{
		kdTree_->buildObstacleTree();
	}

	bool RVOSimulator::queryVisibility(const Vector2& point1, const Vector2& point2, float radius) const
	{
		return kdTree_->queryVisibility(point1, point2, radius);
	}

	void RVOSimulator::setAgentDefaults(float neighborDist, size_t maxNeighbors, float timeHorizon, float timeHorizonObst, float radius, float maxSpeed, const Vector2& velocity)
	{
		if (defaultAgent_ == NULL) {
			defaultAgent_ = new Agent(this);
		}

		defaultAgent_->maxNeighbors_ = maxNeighbors;
		defaultAgent_->maxSpeed_ = maxSpeed;
		defaultAgent_->neighborDist_ = neighborDist;
		defaultAgent_->radius_ = radius;
		defaultAgent_->timeHorizon_ = timeHorizon;
		defaultAgent_->timeHorizonObst_ = timeHorizonObst;
		defaultAgent_->velocity_ = velocity;
	}

	void RVOSimulator::setAgentLambda(size_t agentNo, float lambda) {
		agents_[agentNo]->lambda = lambda;
	}

	void RVOSimulator::setAgentMaxNeighbors(size_t agentNo, size_t maxNeighbors)
	{
		agents_[agentNo]->maxNeighbors_ = maxNeighbors;
	}

	void RVOSimulator::setAgentMaxSpeed(size_t agentNo, float maxSpeed)
	{
		agents_[agentNo]->maxSpeed_ = maxSpeed;
	}

	void RVOSimulator::setAgentNeighborDist(size_t agentNo, float neighborDist)
	{
		agents_[agentNo]->neighborDist_ = neighborDist;
	}

	void RVOSimulator::setAgentPosition(size_t agentNo, const Vector2& position)
	{
		agents_[agentNo]->position_ = position;
	}

	void RVOSimulator::setAgentPrefVelocity(size_t agentNo, const Vector2& prefVelocity)
	{
		agents_[agentNo]->prefVelocity_ = prefVelocity;
	}

	void RVOSimulator::setAgentRadius(size_t agentNo, float radius)
	{
		agents_[agentNo]->radius_ = radius;
	}

	void RVOSimulator::setAgentTimeHorizon(size_t agentNo, float timeHorizon)
	{
		agents_[agentNo]->timeHorizon_ = timeHorizon;
	}

	void RVOSimulator::setAgentTimeHorizonObst(size_t agentNo, float timeHorizonObst)
	{
		agents_[agentNo]->timeHorizonObst_ = timeHorizonObst;
	}

	void RVOSimulator::setAgentVelocity(size_t agentNo, const Vector2& velocity)
	{
		agents_[agentNo]->velocity_ = velocity;
	}

	void RVOSimulator::setCarProperties(float length, float radius, float vDrivingMax, float vSteeringMax, float aDrivingMax, float phiMax, float dtc, float errorPreferred, float ka, float kv, float kp, float deltaV, float deltaPhi)
	{
		Car::initCars(length, radius, vDrivingMax, vSteeringMax, aDrivingMax, phiMax, dtc, errorPreferred, ka, kv, kp);
		Car::deltaPhi = deltaPhi;
		Car::deltav1 = deltaV;
		Car::vNum = 2 * (size_t)(Car::vDrivingMax / deltaV) + 1;
		Car::phiNum = (size_t)(Car::phiMax / deltaPhi) + 1;//due to symmetry
		//if (!Car::readFromFile())
		//	Car::computeLookupTable(timeStep_, 5);
	}

	void RVOSimulator::setAgentCar(size_t agentCarNo, const Vector2& position)
	{
		AgentCar* car = static_cast<AgentCar*>(agents_[carIndices_[agentCarNo]]);
		car->velocity_ = defaultAgent_->velocity_;
		car->maxSpeed_ = Car::vDrivingMax;
		car->radius_ = Car::radius;
		car->position_ = position;
		car->timeHorizon_ = defaultAgent_->timeHorizon_;
		car->maxNeighbors_ = defaultAgent_->maxNeighbors_;
		car->neighborDist_ = defaultAgent_->neighborDist_;
		car->timeHorizon_ = defaultAgent_->timeHorizon_;
		car->timeHorizonObst_ = defaultAgent_->timeHorizonObst_;
		car->lambda = defaultAgent_->lambda;
		
	 	car->car=Car(position.x(),position.y(),0,0);
	}

	void RVOSimulator::setTimeStep(float timeStep)
	{
		timeStep_ = timeStep;
	}

	bool RVOSimulator::shouldUpdate() {
		if (carTime_ > timeStep_ - Car::dtc / 2.f)
		{
			carTime_ = 0.f;
			return true;
		}
		else
			return false;
	}
}
