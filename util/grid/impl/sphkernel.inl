struct SPHKernels {
    float h, h2, h3, h4, h6, h9;
    float poly6_k, spiky_k, visc_k, visc_l_k, gauss_k, wendland_k, spline_k;

    SPHKernels(float smoothingRadius = 0.2f) {
        update(smoothingRadius);
    }

    void update(float smoothingRadius) {
        h = std::max(smoothingRadius, 0.0001f);
        h2 = h * h;
        h3 = h2 * h;
        h4 = h2 * h2;
        h6 = h3 * h3;
        h9 = h6 * h3;

        
        poly6_k = 315.0f / (64.0f * M_PI * h9);
        spiky_k = 15.0f / (M_PI * h6);
        visc_k = 15.0f / (2.0f * M_PI * h3);
        visc_l_k = 45.0f / (M_PI * h6);
        gauss_k = 1.0f / std::pow(M_PI * h2, 1.5f);
        wendland_k = 21.0f / (16.0f * M_PI * h3);
        spline_k = 8.0f / (M_PI * h3);
    }

    inline float Poly6(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return poly6_k * hr2 * hr2 * hr2;
    }

    inline float Poly6Grad(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return -6.0f * poly6_k * r * hr2 * hr2;
    }

    inline float Poly6Laplacian(float r) const {
        if (r >= h) return 0.0f;
        float hr2 = h2 - r*r;
        return -6.0f * poly6_k * hr2 * (3.0f * h2 - 7.0f * r*r);
    }

    inline float Spiky(float r) const {
        if (r >= h) return 0.0f;
        float hr = h - r;
        return spiky_k * hr * hr * hr;
    }

    inline float SpikyGrad(float r) const {
        if (r >= h) return 0.0f;
        float hr = h - r;
        return -3.0f * spiky_k * hr * hr;
    }

    inline float SpikyLaplacian(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        float hr = h - r;
        return -6.0f * spiky_k * hr * (h - 2.0f * r) / r;
    }

    inline float Visc(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        return visc_k * (-(r*r*r)/(2.0f*h3) + (r*r)/h2 + h/(2.0f*r) - 1.0f);
    }

    inline float ViscGrad(float r) const {
        if (r >= h || r < 0.0001f) return 0.0f;
        return visc_k * (-1.5f*(r*r)/h3 + 2.0f*r/h2 - h/(2.0f*r*r));
    }

    inline float ViscLaplacian(float r) const {
        if (r >= h) return 0.0f;
        return visc_l_k * (h - r);
    }

    inline float Gauss(float r) const {
        if (r >= h) return 0.0f;
        return gauss_k * std::exp(-(r*r)/h2);
    }

    inline float GaussGrad(float r) const {
        if (r >= h) return 0.0f;
        return Gauss(r) * (-2.0f * r / h2);
    }
    
    inline float GaussLaplacian(float r) const {
        if (r >= h) return 0.0f;
        return Gauss(r) * (4.0f*r*r - 6.0f*h2) / h4;
    }

    inline float Wendland(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return wendland_k * (oq*oq*oq*oq) * (4.0f*q + 1.0f);
    }

    inline float WendlandGrad(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return -20.0f * wendland_k / h * q * (oq*oq*oq);
    }

    inline float WendlandLaplacian(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        float oq = 1.0f - q;
        return -60.0f * wendland_k / h2 * (oq*oq) * (1.0f - 2.0f*q);
    }

    inline float CubicSpline(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        if (q < 0.5f) return spline_k * (1.0f - 6.0f*q*q + 6.0f*q*q*q);
        float oq = 1.0f - q;
        return spline_k * 2.0f * (oq*oq*oq);
    }

    inline float CubicSplineGrad(float r) const {
        if (r >= h) return 0.0f;
        float q = r / h;
        if (q < 0.5f) return spline_k * (6.0f/h) * q * (3.0f*q - 2.0f);
        float oq = 1.0f - q;
        return -6.0f * spline_k / h * (oq*oq);
    }
    
    inline float CubicSplineLaplacian(float r) const {
        if (r >= h) return 0.0f;
        float q = std::max(r / h, 0.0001f);
        if (q < 0.5f) return spline_k * (36.0f/h2) * (2.0f*q - 1.0f);
        return -12.0f * spline_k / h2 * (1.0f - q) * (1.0f - 2.0f*q) / q;
    }
};
