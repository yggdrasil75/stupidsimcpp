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
    
    Temp(float temp) : temp(temp) {

    };
    
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
    
    void calGrad(const Vec2& testPos, const std::unordered_map<Vec2, Temp>& others) {
        int meh;
        //std::cout << meh++ <<  std::endl;
        std::vector<std::pair<Vec2, float>> nearbyPoints;
        for (const auto& [point, ctemp] : others) {
            if (point.distance(testPos) <= 25) {
                nearbyPoints.emplace_back(point, ctemp.temp);
            }
        }
        //std::cout << meh++ <<  std::endl;
        float sumX = 0,sumY = 0,sumT = 0,sumX2 = 0,sumY2 = 0,sumXY = 0,sumXT = 0, sumYT = 0;
        int n = nearbyPoints.size();
        //std::cout << meh++ <<  std::endl;
        for (const auto& [point, ctemp] : nearbyPoints) {
            float x = point.x - testPos.x;
            float y = point.y - testPos.y;

            sumX += x;
            sumY += y;
            sumT += ctemp;
            sumX2 += x * x;
            sumY2 += y * y;
            sumXY += x * y;
            sumXT += x * ctemp;
            sumYT += y * ctemp;
        }

        //std::cout << meh++ <<  std::endl;
        float det = sumX2 * sumY2 - sumXY * sumXY;
    
        Vec2 calpoint;
        if (std::abs(det) < 1e-10) {
            calpoint = Vec2(0, 0); // Singular matrix, cannot solve
        }
        else {
            float a = (sumXT * sumY2 - sumYT * sumXY) / det;
            float b = (sumX2 * sumYT - sumXY * sumXT) / det;
            calpoint =  Vec2(a, b); // ∇T = (∂T/∂x, ∂T/∂y)
        }
        //std::cout << meh++ <<  std::endl;
        Vec2 closest = findClosestPoint(testPos, others);
        Vec2 displacement = testPos - closest;

        float estimatedTemp = temp + (calpoint.x * displacement.x + calpoint.y * displacement.y);
        temp = estimatedTemp;
        
    }
    
};

#endif