#ifndef WATER_HPP
#define WATER_HPP

#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include <cmath>

// Water constants (SI units: Kelvin, Pascals, Meters)
struct WaterConstants {
    // Thermodynamic properties at STP (Standard Temperature and Pressure)
    static constexpr float STANDARD_TEMPERATURE = 293.15f;
    static constexpr float STANDARD_PRESSURE = 101325.0f;
    static constexpr float FREEZING_POINT = 273.15f;
    static constexpr float BOILING_POINT = 373.15f;
    
    // Reference densities (kg/m³)
    static constexpr float DENSITY_STP = 998.0f;
    static constexpr float DENSITY_0C = 999.8f;
    static constexpr float DENSITY_4C = 1000.0f;
    
    // Viscosity reference values (Pa·s)
    static constexpr float VISCOSITY_0C = 0.001792f;
    static constexpr float VISCOSITY_20C = 0.001002f;
    static constexpr float VISCOSITY_100C = 0.000282f;
    
    // Thermal properties
    static constexpr float SPECIFIC_HEAT_CAPACITY = 4182.0f;
    static constexpr float THERMAL_CONDUCTIVITY = 0.598f;
    static constexpr float LATENT_HEAT_VAPORIZATION = 2257000.0f;
    static constexpr float LATENT_HEAT_FUSION = 334000.0f;
    
    // Other physical constants
    static constexpr float SURFACE_TENSION = 0.0728f;
    static constexpr float SPEED_OF_SOUND = 1482.0f;
    static constexpr float BULK_MODULUS = 2.15e9f;
};

class WaterThermodynamics {
public:
    // Calculate density based on temperature (empirical relationship for 0-100°C)
    static float calculateDensity(float temperature_K) {
        // Empirical formula for pure water density vs temperature
        float T = temperature_K - 273.15f; // Convert to Celsius for empirical formulas
        
        if (T <= 0.0f) return WaterConstants::DENSITY_0C;
        if (T >= 100.0f) return 958.4f; // Density at 100°C
        
        // Polynomial approximation for 0-100°C range
        return 1000.0f * (1.0f - (T + 288.9414f) * (T - 3.9863f) * (T - 3.9863f) / 
                         (508929.2f * (T + 68.12963f)));
    }
    
    // Calculate dynamic viscosity based on temperature (using Vogel-Fulcher-Tammann equation)
    static float calculateViscosity(float temperature_K) {
        float T = temperature_K;
        // Vogel-Fulcher-Tammann equation parameters for water
        constexpr float A = -3.7188f;
        constexpr float B = 578.919f;
        constexpr float C = -137.546f;
        
        return 0.001f * std::exp(A + B / (T - C)); // Returns in Pa·s
    }
    
    // Calculate viscosity using simpler Arrhenius-type equation (good for 0-100°C)
    static float calculateViscositySimple(float temperature_K) {
        float T = temperature_K - 273.15f; // Celsius
        
        if (T <= 0.0f) return WaterConstants::VISCOSITY_0C;
        if (T >= 100.0f) return WaterConstants::VISCOSITY_100C;
        
        // Simple exponential decay model for 0-100°C range
        return 0.001792f * std::exp(-0.024f * T);
    }
    
    // Calculate thermal conductivity (W/(m·K))
    static float calculateThermalConductivity(float temperature_K) {
        float T = temperature_K - 273.15f; // Celsius
        // Linear approximation for 0-100°C
        return 0.561f + 0.002f * T - 0.00001f * T * T;
    }
    
    // Calculate surface tension (N/m)
    static float calculateSurfaceTension(float temperature_K) {
        float T = temperature_K - 273.15f; // Celsius
        // Linear decrease with temperature
        return 0.07564f - 0.000141f * T - 0.00000025f * T * T;
    }
    
    // Calculate speed of sound in water (m/s)
    static float calculateSpeedOfSound(float temperature_K, float pressure_Pa = WaterConstants::STANDARD_PRESSURE) {
        float T = temperature_K - 273.15f; // Celsius
        // Empirical formula for pure water
        return 1402.5f + 5.0f * T - 0.055f * T * T + 0.0003f * T * T * T;
    }
    
    // Calculate bulk modulus (compressibility) in Pa
    static float calculateBulkModulus(float temperature_K, float pressure_Pa = WaterConstants::STANDARD_PRESSURE) {
        float T = temperature_K - 273.15f; // Celsius
        // Approximation - decreases slightly with temperature
        return WaterConstants::BULK_MODULUS * (1.0f - 0.0001f * T);
    }
    
    // Check if water should change phase
    static bool isFrozen(float temperature_K, float pressure_Pa = WaterConstants::STANDARD_PRESSURE) {
        return temperature_K <= WaterConstants::FREEZING_POINT;
    }
    
    static bool isBoiling(float temperature_K, float pressure_Pa = WaterConstants::STANDARD_PRESSURE) {
        // Simple boiling point calculation (neglecting pressure effects for simplicity)
        return temperature_K >= WaterConstants::BOILING_POINT;
    }
};

struct WaterParticle {
    Vec3 velocity;
    Vec3 acceleration;
    Vec3 force;
    
    float temperature;
    float pressure;
    float density;
    float mass;
    float viscosity;
    
    float volume;
    float energy;
    
    WaterParticle(float percent = 1.0f, float temp_K = WaterConstants::STANDARD_TEMPERATURE) 
        : velocity(0, 0, 0), acceleration(0, 0, 0), force(0, 0, 0),
          temperature(temp_K), pressure(WaterConstants::STANDARD_PRESSURE),
          volume(1.0f * percent) {
        
        updateThermodynamicProperties();
        
        // Mass is density × volume
        mass = density * volume;
        energy = mass * WaterConstants::SPECIFIC_HEAT_CAPACITY * temperature;
    }
    
    // Update all temperature-dependent properties
    void updateThermodynamicProperties() {
        density = WaterThermodynamics::calculateDensity(temperature);
        viscosity = WaterThermodynamics::calculateViscosity(temperature);
        
        // If we have a fixed mass, adjust volume for density changes
        if (mass > 0.0f) {
            volume = mass / density;
        }
    }
    
    // Add thermal energy and update temperature
    void addThermalEnergy(float energy_joules) {
        energy += energy_joules;
        temperature = energy / (mass * WaterConstants::SPECIFIC_HEAT_CAPACITY);
        updateThermodynamicProperties();
    }
    
    // Set temperature directly
    void setTemperature(float temp_K) {
        temperature = temp_K;
        energy = mass * WaterConstants::SPECIFIC_HEAT_CAPACITY * temperature;
        updateThermodynamicProperties();
    }
    
    // Check phase state
    bool isFrozen() const { return WaterThermodynamics::isFrozen(temperature, pressure); }
    bool isBoiling() const { return WaterThermodynamics::isBoiling(temperature, pressure); }
    bool isLiquid() const { return !isFrozen() && !isBoiling(); }
};


#endif