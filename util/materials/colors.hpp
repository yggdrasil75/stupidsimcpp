#ifndef COLORS
#define COLORS

#include <cmath>
#include <algorithm>
#include <cstdint>

namespace ColorUtils {
    inline float cieX(float wavelength) {
        float t1 = (wavelength - 442.0f) * ((wavelength < 442.0f) ? 0.0624f : 0.0374f);
        float t2 = (wavelength - 599.8f) * ((wavelength < 599.8f) ? 0.0264f : 0.0323f);
        float t3 = (wavelength - 501.1f) * ((wavelength < 501.1f) ? 0.0490f : 0.0382f);
        
        return 0.362f * expf(-0.5f * t1 * t1) + 1.056f * expf(-0.5f * t2 * t2) - 0.065f * expf(-0.5f * t3 * t3);
    }
    
    inline float cieY(float wavelength) {
        float t1 = (wavelength - 568.8f) * ((wavelength < 568.8f) ? 0.0213f : 0.0247f);
        float t2 = (wavelength - 530.9f) * ((wavelength < 530.9f) ? 0.0613f : 0.0322f);
        
        return 0.821f * expf(-0.5f * t1 * t1) + 0.286f * expf(-0.5f * t2 * t2);
    }
    
    inline float cieZ(float wavelength) {
        float t1 = (wavelength - 437.0f) * ((wavelength < 437.0f) ? 0.0845f : 0.0278f);
        float t2 = (wavelength - 459.0f) * ((wavelength < 459.0f) ? 0.0385f : 0.0725f);
        
        return 1.217f * expf(-0.5f * t1 * t1) + 0.681f * expf(-0.5f * t2 * t2);
    }

    inline float gammaCorrect(float c) {
        return (c <= 0.0031308f) ? (12.92f * c) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
    }

    inline void wavelengthToRGB(float wavelength, float* rgbOut) {
        float x = cieX(wavelength);
        float y = cieY(wavelength);
        float z = cieZ(wavelength);

        float r = 3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
        float g = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
        float b = 0.0556434f * x - 0.2040259f * y + 1.0572252f * z;

        rgbOut[0] = gammaCorrect(std::max(0.0f, std::min(1.0f, r)));
        rgbOut[1] = gammaCorrect(std::max(0.0f, std::min(1.0f, g)));
        rgbOut[2] = gammaCorrect(std::max(0.0f, std::min(1.0f, b)));
    }

    inline float rgbToHue(float r, float g, float b) {
        float maxVal = std::max({r, g, b});
        float minVal = std::min({r, g, b});
        float delta = maxVal - minVal;

        if (delta < 0.00001f) return 0.0f;

        float hue = 0.0f;
        if (maxVal == r) {
            hue = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        } else if (maxVal == g) {
            hue = (b - r) / delta + 2.0f;
        } else {
            hue = (r - g) / delta + 4.0f;
        }
        return hue * 60.0f;
    }

    inline float hueToWavelength(float hue) {
        float wl = 0.0f;
        if (hue >= 260.0f) {
            wl = 380.0f + (hue - 260.0f) * 0.5f; 
        } else if (hue < 120.0f) {
                wl = 650.0f - (hue / 120.0f) * (650.0f - 510.0f);
        } else if (hue < 240.0f) {
            wl = 510.0f - ((hue - 120.0f) / 120.0f) * (510.0f - 440.0f);
        } else {
            wl = 440.0f - ((hue - 240.0f) / 120.0f) * (440.0f - 380.0f);
        }
        return wl;
    }
}


#endif