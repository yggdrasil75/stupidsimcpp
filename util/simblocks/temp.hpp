#ifndef temp_hpp
#define temp_hpp

#include "../vectorlogic/vec2.hpp"
#include "../timing_decorator.hpp"
#include <vector>
#include <unordered_map>

class Temp {
private:

protected:

public:
    double temp;
    
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
};

#endif