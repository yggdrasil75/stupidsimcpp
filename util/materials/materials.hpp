#ifndef G3_MATERIALS_HPP
#define G3_MATERIALS_HPP

#include "../../eigen/Eigen/Dense"
#include "colors.hpp"

struct Material {
    float ior;
    float dispersion;
    float chromaticity;
    float bandwidth;
    float transmission;
    float roughness;
    float emittance;
    bool light;
    float density;
    float speedOfSound;
    float audioAbsorption;
    Eigen::Vector3f rgb; //for the fast version.

    // Constructor with sensible defaults
    Material(float ior = 1.5f, float dispersion = 0.0f, float chromaticity = 550.0f, 
             float bandwidth = 100.0f, float transmission = 0.0f, float roughness = 0.8f, 
             float emittance = 0.0f, float density = 1000.0f, float speedOfSound = 343.0f,
             float audioAbsorption = 0.1f, bool light = false)
        : ior(ior), dispersion(dispersion), chromaticity(chromaticity), bandwidth(bandwidth),
         transmission(transmission), roughness(roughness), emittance(emittance), 
         density(density), speedOfSound(speedOfSound), audioAbsorption(audioAbsorption), light(light) {}

          
    static Material fromRGB(float r, float g, float b, float ior = 1.5f, float transmission = 0.0f, 
                           float roughness = 0.8f, float emittance = 0.0f, bool isLight = false) {
        float hue = ColorUtils::rgbToHue(r, g, b);
        float wavelength = ColorUtils::hueToWavelength(hue);

        float maxVal = std::max({r, g, b});
        float minVal = std::min({r, g, b});
        float delta = maxVal - minVal;
        
        float saturation = (maxVal > 0.0f) ? (delta / maxVal) : 0.0f;
        
        float bandwidth = 10.0f + (1.0f - saturation) * 290.0f;

        Material outerial(ior, 0.02f, wavelength, bandwidth, transmission, roughness, emittance, 2.0f, 3000.0f, 0.1f, isLight);
        outerial.rgb = Eigen::Vector3f(r, g, b);
        return outerial;
    }
};

namespace Materials {
    // Metals
    static const Material Gold(0.47f, 0.25f, 580.0f, 150.0f, 0.95f, 0.2f, 0.0f, 19300.0f, 3240.0f, 0.01f, false);
    static const Material Silver(0.18f, 0.35f, 450.0f, 200.0f, 0.98f, 0.15f, 0.0f, 10500.0f, 3650.0f, 0.008f, false);
    static const Material Copper(0.64f, 0.22f, 620.0f, 100.0f, 0.90f, 0.3f, 0.0f, 8960.0f, 4760.0f, 0.012f, false);
    static const Material Iron(2.91f, 0.15f, 500.0f, 180.0f, 0.65f, 0.4f, 0.0f, 7870.0f, 5130.0f, 0.015f, false);
    static const Material Aluminum(1.44f, 0.20f, 500.0f, 200.0f, 0.92f, 0.25f, 0.0f, 2700.0f, 6320.0f, 0.009f, false);
    static const Material Platinum(2.06f, 0.18f, 500.0f, 180.0f, 0.90f, 0.2f, 0.0f, 21450.0f, 3960.0f, 0.01f, false);
    static const Material Bronze(1.15f, 0.12f, 590.0f, 120.0f, 0.85f, 0.35f, 0.0f, 8700.0f, 3580.0f, 0.02f, false);
    static const Material Brass(0.85f, 0.15f, 570.0f, 130.0f, 0.88f, 0.3f, 0.0f, 8520.0f, 4700.0f, 0.018f, false);

    // Glass
    static const Material Glass(1.52f, 0.008f, 550.0f, 80.0f, 0.92f, 0.05f, 0.0f, 2500.0f, 5300.0f, 0.05f, false);
    static const Material FlintGlass(1.62f, 0.028f, 550.0f, 100.0f, 0.90f, 0.05f, 0.0f, 3600.0f, 5100.0f, 0.06f, false);
    static const Material Borosilicate(1.47f, 0.005f, 550.0f, 70.0f, 0.95f, 0.04f, 0.0f, 2230.0f, 5600.0f, 0.04f, false);
    static const Material FusedSilica(1.46f, 0.004f, 550.0f, 60.0f, 0.98f, 0.02f, 0.0f, 2200.0f, 5968.0f, 0.03f, false);
    static const Material LeadCrystal(1.57f, 0.025f, 550.0f, 90.0f, 0.93f, 0.03f, 0.0f, 3800.0f, 4200.0f, 0.07f, false);
    static const Material Obsidian(1.48f, 0.010f, 500.0f, 150.0f, 0.30f, 0.4f, 0.0f, 2600.0f, 4800.0f, 0.15f, false);
    static const Material StainedGlassRed(1.52f, 0.008f, 650.0f, 50.0f, 0.60f, 0.08f, 0.0f, 2550.0f, 5200.0f, 0.08f, false);
    static const Material StainedGlassBlue(1.52f, 0.008f, 470.0f, 50.0f, 0.65f, 0.08f, 0.0f, 2550.0f, 5200.0f, 0.08f, false);
    static const Material StainedGlassGreen(1.52f, 0.008f, 530.0f, 50.0f, 0.60f, 0.08f, 0.0f, 2550.0f, 5200.0f, 0.08f, false);
    static const Material StainedGlassYellow(1.52f, 0.008f, 580.0f, 50.0f, 0.70f, 0.08f, 0.0f, 2550.0f, 5200.0f, 0.08f, false);
    static const Material FrostedGlass(1.52f, 0.008f, 550.0f, 80.0f, 0.85f, 0.7f, 0.0f, 2500.0f, 5300.0f, 0.25f, false);
    static const Material TemperedGlass(1.52f, 0.008f, 550.0f, 80.0f, 0.91f, 0.06f, 0.0f, 2500.0f, 5400.0f, 0.06f, false);

    // Water
    static const Material FreshWater(1.333f, 0.018f, 475.0f, 200.0f, 0.99f, 0.1f, 0.0f, 1000.0f, 1480.0f, 0.002f, false);
    static const Material SaltWater(1.339f, 0.020f, 475.0f, 250.0f, 0.85f, 0.15f, 0.0f, 1025.0f, 1530.0f, 0.003f, false);
    static const Material BrackishWater(1.336f, 0.019f, 480.0f, 220.0f, 0.70f, 0.2f, 0.0f, 1010.0f, 1500.0f, 0.005f, false);
    static const Material DistilledWater(1.333f, 0.017f, 475.0f, 180.0f, 0.995f, 0.05f, 0.0f, 1000.0f, 1480.0f, 0.001f, false);
    static const Material MurkyWater(1.340f, 0.022f, 550.0f, 300.0f, 0.30f, 0.4f, 0.0f, 1030.0f, 1450.0f, 0.05f, false);

    // Gems
    static const Material Diamond(2.42f, 0.044f, 550.0f, 50.0f, 0.99f, 0.01f, 0.0f, 3520.0f, 12000.0f, 0.005f, false);
    static const Material Emerald(1.58f, 0.014f, 530.0f, 40.0f, 0.85f, 0.1f, 0.0f, 2750.0f, 8500.0f, 0.02f, false);
    static const Material Ruby(1.77f, 0.018f, 690.0f, 30.0f, 0.80f, 0.15f, 0.0f, 4000.0f, 8800.0f, 0.015f, false);
    static const Material Sapphire(1.77f, 0.018f, 460.0f, 40.0f, 0.85f, 0.12f, 0.0f, 3980.0f, 8700.0f, 0.015f, false);
    static const Material Amethyst(1.54f, 0.013f, 420.0f, 60.0f, 0.75f, 0.2f, 0.0f, 2650.0f, 7200.0f, 0.025f, false);
    static const Material Topaz(1.63f, 0.014f, 590.0f, 50.0f, 0.88f, 0.15f, 0.0f, 3560.0f, 8000.0f, 0.018f, false);
    static const Material Opal(1.45f, 0.010f, 550.0f, 200.0f, 0.70f, 0.3f, 0.0f, 2100.0f, 4500.0f, 0.03f, false);

    // Natural Materials
    static const Material PlantLeaf(1.45f, 0.05f, 550.0f, 100.0f, 0.15f, 0.7f, 0.0f, 800.0f, 350.0f, 0.4f, false);
    static const Material Wood(1.50f, 0.10f, 650.0f, 150.0f, 0.10f, 0.9f, 0.0f, 700.0f, 3300.0f, 0.35f, false);
    static const Material Soil(1.55f, 0.08f, 600.0f, 200.0f, 0.05f, 0.95f, 0.0f, 1600.0f, 300.0f, 0.6f, false);
    static const Material Snow(1.31f, 0.015f, 450.0f, 100.0f, 0.98f, 0.6f, 0.0f, 250.0f, 320.0f, 0.5f, false);
    static const Material Ice(1.31f, 0.016f, 475.0f, 120.0f, 0.90f, 0.3f, 0.0f, 917.0f, 3980.0f, 0.08f, false);
    static const Material Sand(1.54f, 0.06f, 580.0f, 180.0f, 0.20f, 0.85f, 0.0f, 1600.0f, 200.0f, 0.7f, false);
    static const Material Rock(1.55f, 0.07f, 550.0f, 200.0f, 0.08f, 0.8f, 0.0f, 2750.0f, 5500.0f, 0.2f, false);
    static const Material Marble(1.50f, 0.005f, 550.0f, 150.0f, 0.40f, 0.4f, 0.0f, 2710.0f, 6100.0f, 0.15f, false);

    // Stars and Lights
    static const Material Sun(1.0f, 0.0f, 580.0f, 300.0f, 1.0f, 0.0f, 1.0f, 1408.0f, 0.0f, 0.0f, true);
    static const Material RedStar(1.0f, 0.0f, 700.0f, 200.0f, 1.0f, 0.0f, 0.8f, 0.1f, 0.0f, 0.0f, true);
    static const Material YellowStar(1.0f, 0.0f, 580.0f, 250.0f, 1.0f, 0.0f, 0.9f, 0.2f, 0.0f, 0.0f, true);
    static const Material BlueStar(1.0f, 0.0f, 450.0f, 150.0f, 1.0f, 0.0f, 1.2f, 0.3f, 0.0f, 0.0f, true);
    static const Material BrownStar(1.0f, 0.0f, 1000.0f, 400.0f, 1.0f, 0.0f, 0.3f, 0.05f, 0.0f, 0.0f, true);
    static const Material WhiteStar(1.0f, 0.0f, 550.0f, 200.0f, 1.0f, 0.0f, 1.1f, 0.4f, 0.0f, 0.0f, true);
    static const Material Moon(1.0f, 0.0f, 580.0f, 150.0f, 1.0f, 0.0f, 0.015f, 3340.0f, 0.0f, 0.0f, true);
    static const Material LEDWhite(1.0f, 0.0f, 550.0f, 100.0f, 1.0f, 0.0f, 0.7f, 2000.0f, 0.0f, 0.0f, true);
    static const Material LEDWarm(1.0f, 0.0f, 600.0f, 80.0f, 1.0f, 0.0f, 0.6f, 2000.0f, 0.0f, 0.0f, true);
    static const Material NeonRed(1.0f, 0.0f, 640.0f, 10.0f, 1.0f, 0.0f, 0.5f, 0.9f, 0.0f, 0.0f, true);
    static const Material NeonBlue(1.0f, 0.0f, 480.0f, 15.0f, 1.0f, 0.0f, 0.5f, 0.9f, 0.0f, 0.0f, true);

    // Flames
    static const Material CandleFlame(1.0002f, 0.002f, 650.0f, 100.0f, 0.7f, 0.9f, 0.8f, 0.2f, 343.0f, 0.8f, true);
    static const Material WoodFire(1.0003f, 0.003f, 620.0f, 150.0f, 0.6f, 0.95f, 1.2f, 0.3f, 350.0f, 0.7f, true);
    static const Material GasFlame(1.0002f, 0.001f, 580.0f, 80.0f, 0.8f, 0.8f, 1.0f, 0.25f, 345.0f, 0.75f, true);
    static const Material PropaneFlame(1.0002f, 0.0015f, 590.0f, 120.0f, 0.75f, 0.85f, 1.1f, 0.28f, 348.0f, 0.72f, true);
    static const Material AlcoholFlame(1.0002f, 0.002f, 610.0f, 90.0f, 0.85f, 0.7f, 0.9f, 0.22f, 340.0f, 0.78f, true);
    static const Material Plasma(1.0001f, 0.005f, 450.0f, 200.0f, 0.9f, 0.5f, 2.0f, 0.001f, 1000.0f, 0.3f, true);
    
    // Organic materials
    static const Material Skin(1.38f, 0.03f, 580.0f, 120.0f, 0.25f, 0.6f, 0.0f, 1100.0f, 1540.0f, 0.25f, false);
    static const Material Hair(1.55f, 0.04f, 600.0f, 100.0f, 0.10f, 0.8f, 0.0f, 1300.0f, 100.0f, 0.6f, false);
    static const Material Cotton(1.52f, 0.06f, 580.0f, 150.0f, 0.30f, 0.9f, 0.0f, 1520.0f, 200.0f, 0.55f, false);
    static const Material Silk(1.54f, 0.05f, 570.0f, 130.0f, 0.40f, 0.7f, 0.0f, 1320.0f, 150.0f, 0.45f, false);
    static const Material Leather(1.50f, 0.08f, 630.0f, 140.0f, 0.15f, 0.85f, 0.0f, 860.0f, 400.0f, 0.5f, false);
    
    // Plastics and Polymers
    static const Material Acrylic(1.49f, 0.008f, 550.0f, 90.0f, 0.92f, 0.1f, 0.0f, 1180.0f, 2670.0f, 0.12f, false);
    static const Material Polycarbonate(1.58f, 0.010f, 550.0f, 95.0f, 0.88f, 0.15f, 0.0f, 1200.0f, 2270.0f, 0.15f, false);
    static const Material PVC(1.54f, 0.012f, 550.0f, 120.0f, 0.70f, 0.3f, 0.0f, 1380.0f, 2380.0f, 0.18f, false);
    static const Material Rubber(1.52f, 0.020f, 600.0f, 200.0f, 0.10f, 0.9f, 0.0f, 1100.0f, 1500.0f, 0.4f, false);
    
    // Atmosphere Layers
    static const Material Troposphere(1.00029f, 0.00015f, 475.0f, 350.0f, 0.95f, 0.3f, 0.0f, 1.225f, 343.0f, 0.01f, false);
    static const Material Tropopause(1.00027f, 0.00012f, 470.0f, 320.0f, 0.97f, 0.2f, 0.0f, 0.66f, 295.0f, 0.008f, false);
    static const Material Stratosphere(1.00025f, 0.00010f, 450.0f, 300.0f, 0.92f, 0.1f, 0.0f, 0.12f, 300.0f, 0.006f, false);
    static const Material OzoneLayer(1.00026f, 0.00011f, 300.0f, 200.0f, 0.85f, 0.05f, 0.0f, 0.08f, 310.0f, 0.01f, false);
    static const Material Stratopause(1.00022f, 0.00009f, 460.0f, 280.0f, 0.96f, 0.08f, 0.0f, 0.04f, 315.0f, 0.005f, false);
    static const Material Mesosphere(1.00018f, 0.00008f, 490.0f, 250.0f, 0.88f, 0.15f, 0.0f, 0.008f, 280.0f, 0.004f, false);
    static const Material Mesopause(1.00015f, 0.00007f, 500.0f, 230.0f, 0.94f, 0.12f, 0.0f, 0.005f, 275.0f, 0.003f, false);
    static const Material Thermosphere(1.00010f, 0.00006f, 520.0f, 200.0f, 0.82f, 0.05f, 0.02f, 0.0002f, 500.0f, 0.001f, false);
    static const Material Ionosphere(1.00008f, 0.00020f, 430.0f, 400.0f, 0.75f, 0.08f, 0.05f, 0.0001f, 1000.0f, 0.002f, false);
    static const Material Exosphere(1.00001f, 0.00005f, 550.0f, 150.0f, 0.90f, 0.02f, 0.0f, 0.000001f, 0.0f, 0.0f, false);
    static const Material KarmanLine(1.000005f, 0.00004f, 560.0f, 120.0f, 0.98f, 0.01f, 0.0f, 0.0000005f, 0.0f, 0.0f, false);
    
    // Atmospheric Conditions
    static const Material ClearSky(1.00028f, 0.00013f, 475.0f, 320.0f, 0.97f, 0.1f, 0.0f, 1.2f, 343.0f, 0.02f, false);
    static const Material HazySky(1.00030f, 0.00018f, 550.0f, 400.0f, 0.85f, 0.5f, 0.0f, 1.3f, 343.0f, 0.1f, false);
    static const Material FoggyAir(1.00032f, 0.00025f, 580.0f, 500.0f, 0.60f, 0.9f, 0.0f, 1.4f, 343.0f, 0.3f, false);
    static const Material RainyAir(1.00031f, 0.00022f, 500.0f, 450.0f, 0.70f, 0.7f, 0.0f, 1.35f, 343.0f, 0.15f, false);
    static const Material MartianAtmosphere(1.00015f, 0.00030f, 650.0f, 200.0f, 0.80f, 0.6f, 0.0f, 0.02f, 240.0f, 0.08f, false);
    static const Material VenusianAtmosphere(1.00050f, 0.00040f, 580.0f, 600.0f, 0.30f, 0.8f, 0.1f, 67.0f, 400.0f, 0.4f, false);
    static const Material TitanAtmosphere(1.00040f, 0.00035f, 620.0f, 400.0f, 0.40f, 0.7f, 0.0f, 5.3f, 200.0f, 0.25f, false);
    static const Material AuroraBorealis(1.00012f, 0.00050f, 557.7f, 50.0f, 0.90f, 0.3f, 0.3f, 0.001f, 0.0f, 0.0f, true);
    static const Material AuroraRed(1.00012f, 0.00050f, 630.0f, 40.0f, 0.90f, 0.3f, 0.2f, 0.001f, 0.0f, 0.0f, true);
    static const Material AuroraBlue(1.00012f, 0.00050f, 428.0f, 60.0f, 0.90f, 0.3f, 0.25f, 0.001f, 0.0f, 0.0f, true);
    static const Material SeaLevelAir(1.00029f, 0.00015f, 550.0f, 300.0f, 0.95f, 0.1f, 0.0f, 1.225f, 343.0f, 0.01f, false);
    static const Material MountainAir(1.00025f, 0.00012f, 500.0f, 280.0f, 0.97f, 0.05f, 0.0f, 0.96f, 330.0f, 0.008f, false);
    static const Material HighAltitudeAir(1.00020f, 0.00010f, 480.0f, 250.0f, 0.98f, 0.03f, 0.0f, 0.74f, 320.0f, 0.006f, false);
}

#endif