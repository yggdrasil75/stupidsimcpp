#ifndef temp_hpp
#define temp_hpp

#include "../vectorlogic/vec2.hpp"
#include "../timing_decorator.hpp"
#include <vector>
#include <unordered_map>

class Temp {
private:

protected:
    static Vec2 findClosestPoint(const Vec2& position, std::unordered_map<Vec2, Temp> others) {
        if (others.empty()) {
            return position;
        }
        
        auto closest = others.begin();
        float minDistance = position.distance(closest->first);
        
        for (auto it = std::next(others.begin()); it != others.end(); ++it) {
            float distance = position.distance(it->first);
            if (distance < minDistance) {
                minDistance = distance;
                closest = it;
            }
        }
        
        return closest->first;
    }
    

public:
    float temp;
    float conductivity = 0.5;
    float specific_heat = 900.0;
    float diffusivity = 2000.0;
    
    Temp() : temp(0.0) {};
    Temp(float temp) : temp(temp) {};
    
    Temp(const Vec2& testPos, const std::unordered_map<Vec2, Temp>& others) {
        TIME_FUNCTION;
        float power = 2.0;
        float num = 0.0;
        float den = 0.0;
        
        for (const auto& [point, tempObj] : others) {
            float dist = testPos.distance(point);
            float weight = 1.0 / std::pow(dist, power);
            num += weight * tempObj.temp;
            den += weight;
        }

        if (den < 1e-10 && den > -1e-10) {
            den = 1e-10;
        }
        this->temp = num / den;
    }
    
    static float calTempIDW(const Vec2& testPos, const std::unordered_map<Vec2, Temp>& others) {
        TIME_FUNCTION;
        float power = 2.0;
        float num = 0.0;
        float den = 0.0;
        for (const auto& [point, temp] : others) {
            float dist = testPos.distance(point);

            float weight = 1.0 / std::pow(dist, power);
            num += weight * temp.temp;
            den += weight;
        }

        if (den < 1e-10 && den > -1e-10) {
            den = 1e-10;
        }
        return num / den;
    }
    
    void calLapl(const Vec2& testPos, const std::unordered_map<Vec2, Temp>& others, float deltaTime) {
        //TIME_FUNCTION;
        float dt = deltaTime;
        float sumWeights = 0.0f;
        float sumTempWeights = 0.0f;
        float searchRadius = 25.0f;

        for (const auto& [point, tempObj] : others) {
            float dist = testPos.distance(point);
            
            if (dist < 0.001f || dist > searchRadius) continue;

            float weight = 1.0f / (dist * dist);

            sumTempWeights += weight * tempObj.temp;
            sumWeights += weight;
        }

        if (sumWeights < 1e-10f) return;

        float equilibriumTemp = sumTempWeights / sumWeights;

        float rate = this->diffusivity * 0.01f;
        
        float lerpFactor = 1.0f - std::exp(-rate * dt);

        this->temp += (equilibriumTemp - this->temp) * lerpFactor;
    }
    
};

#endif