#pragma once
#include "Agent.h"
#include "Car.h"
#include <iostream>
namespace RVO {
	class AgentCar : public RVO::Agent
	{
		friend class RVOSimulator;
		friend class CarRenderer;
	public:

		void track();

		void computeNeighbors() override;
		void computeNewVelocity() override;
		bool searchDesiredVelocity(RVO::Vector2& vd);

		void update() override;
		std::vector<Line> getORCA() const{
			return Agent::orcaLines_;
		}

		AgentCar(RVO::RVOSimulator* sim, Vector2 position, float theta, float phi) :RVO::Agent(sim), car(position.x(), position.y(), theta, phi)
		{
		}
	private:
		Car car;
	};

}
