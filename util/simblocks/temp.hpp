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
        double minDistance = position.distance(closest->first);
        
        for (auto it = std::next(others.begin()); it != others.end(); ++it) {
            double distance = position.distance(it->first);
            if (distance < minDistance) {
                minDistance = distance;
                closest = it;
            }
        }
        
        return closest->first;
    }
    

public:
    double temp;
    Temp(float temp) : temp(temp) {
        //std::cout << "setting temp to: " << temp << std::endl;
    };
    
    Temp(const Vec2& testPos, const std::unordered_map<Vec2, Temp>& others) {
        TIME_FUNCTION;
        double power = 2.0;
        double num = 0.0;
        double den = 0.0;
        
        for (const auto& [point, tempObj] : others) {
            double dist = testPos.distance(point);
            double weight = 1.0 / std::pow(dist, power);
            num += weight * tempObj.temp;
            den += weight;
        }

        if (den < 1e-10 && den > -1e-10) {
            den = 1e-10;
        }
        this->temp = num / den;
    }
    
    static double calTempIDW(const Vec2& testPos, std::unordered_map<Vec2, Temp> others) {
        TIME_FUNCTION;
        double power = 2.0;
        double num = 0.0;
        double den = 0.0;
        for (const auto& [point, temp] : others) {
            double dist = testPos.distance(point);

            double weight = 1.0 / std::pow(dist, power);
            num += weight * temp.temp;
            den += weight;
        }

        if (den < 1e-10 && den > -1e-10) {
            den = 1e-10;
        }
        return num / den;
    }
    
    static float calGrad(const Vec2& testPos, std::unordered_map<Vec2, Temp> others) {
        std::vector<std::pair<Vec2, double>> nearbyPoints;
        for (const auto& [point, temp] : others) {
            if (point.distance(testPos) <= 25) {
                nearbyPoints.emplace_back(point, temp.temp);
            }
        }
        double sumX, sumY, sumT, sumX2, sumY2, sumXY, sumXT, sumYT = 0;
        int n = nearbyPoints.size();
        for (const auto& [point, temp] : nearbyPoints) {
            double x = point.x - testPos.x;
            double y = point.y - testPos.y;

            sumX += x;
            sumY += y;
            sumT += temp;
            sumX2 += x * x;
            sumY2 += y * y;
            sumXY += x * y;
            sumXT += x * temp;
            sumYT += y * temp;
        }

        double det = sumX2 * sumY2 - sumXY * sumXY;

        if (std::abs(det) < 1e-10) {
            Vec2 calpoint = Vec2(0, 0); // Singular matrix, cannot solve
        }
        
        double a = (sumXT * sumY2 - sumYT * sumXY) / det;
        double b = (sumX2 * sumYT - sumXY * sumXT) / det;
        
        Vec2 calpoint =  Vec2(a, b); // ∇T = (∂T/∂x, ∂T/∂y)

        Vec2 closest = findClosestPoint(testPos, others);
        Vec2 displacement = testPos - closest;

        float refTemp = others.at(closest).temp;
        float estimatedTemp = refTemp + (calpoint.x * displacement.x + calpoint.y * displacement.y);
        return estimatedTemp;
    }
};

#endif