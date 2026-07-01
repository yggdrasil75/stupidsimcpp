#pragma once
namespace Grid {

struct CelestialBody {
    Vec3 direction;
    float angularRadius;
    float cosAngularRadius;
    uint8_t r, g, b, emittance;
    bool baked;
    struct PixelBackup {
        size_t x, y;
        std::vector<uint8_t> data;
    };
    std::vector<PixelBackup> backup;

    CelestialBody() : angularRadius(0), cosAngularRadius(1), r(255), g(255), b(255), emittance(255), baked(false) {}
};

struct Skybox {
    frame skybox;
    std::map<int, CelestialBody> bodies;
    Eigen::Quaternionf skyRotation;

    Skybox(size_t w = 1024, size_t h = 1024) : skybox(w, h, frame::colormap::RGBA), skyRotation(Eigen::Quaternionf::Identity()) { }

    void dirToUV(const Vec3& dir, float& u, float& v) const {
        Vec3 d = dir.normalized();
        u = 0.5f + (std::atan2(d.z(), d.x()) / (2.0f * M_PI));
        v = 0.5f - (std::asin(d.y()) / M_PI);
    }

    const Vec3 uvToDir(float u, float v) const {
        float theta = (u - 0.5f) * 2.0f * M_PI;
        float phi = (0.5f - v) * M_PI;
        float y = std::sin(phi);
        float cosphi = std::cos(phi);
        float x = std::cos(theta) * cosphi;
        float z = std::sin(theta) * cosphi;
        return Vec3(x, y, z);
    }
    
    std::vector<uint8_t> sample(const Vec3& dir) {
        Vec3 rotatedDir = skyRotation * dir;
        for (auto it = bodies.rbegin(); it != bodies.rend(); ++it) {
            if (!it->second.baked) {
                if (rotatedDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                    return {it->second.r, it->second.g, it->second.b, it->second.emittance};
                }
            }
        }

        float u, v;
        dirToUV(rotatedDir, u, v);

        u = std::clamp(u, 0.0f, 0.9999f);
        v = std::clamp(v, 0.0f, 0.9999f);
        size_t x = static_cast<size_t>(u * skybox.getWidth());
        size_t y = static_cast<size_t>(v * skybox.getHeight());

        return skybox.getPixel(x, y);
    }

    Vec3 sampleVector(const Vec3& dir) {
        std::vector<uint8_t> px = sample(dir);
        if (px.size() >= 3) {
            float r = px[0] / 255.0f;
            float g = px[1] / 255.0f;
            float b = px[2] / 255.0f;
            float e = px.size() >= 4 ? (px[3] / 255.0f) : 1.0f;
            return Vec3(r, g, b) * e;
        }
        return Vec3::Zero();
    }

    void setBackground(float r, float g, float b, float e) {
        size_t w = skybox.getWidth();
        size_t h = skybox.getHeight();
        std::vector<float> data(w * h * 4);

        for (size_t i = 0; i < data.size(); i += 4) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            data[i + 3] = e;
        }
        skybox.setData(data, frame::colormap::RGBA);
    }
    
    void addBody(int id, const Vec3& dir, float angularRadius, uint8_t r, uint8_t g, uint8_t b, uint8_t emittance) {
        removeBody(id);
        CelestialBody body;
        body.direction = dir.normalized();
        body.angularRadius = angularRadius;
        body.cosAngularRadius = std::cos(angularRadius);
        body.r = r;
        body.g = g;
        body.b = b;
        body.emittance = emittance;
        body.baked = false;
        bodies[id] = std::move(body);
    }

    void removeBody(int id) {
        auto it = bodies.find(id);
        if (it != bodies.end()) {
            if (it->second.baked) {
                resetBody(id);
            }
            bodies.erase(it);
        }
    }

    void moveBody(int id, const Vec3& newDir) {
        auto it = bodies.find(id);
        if (it != bodies.end()) {
            bool wasBaked = it->second.baked;
            if (wasBaked) resetBody(id);
            
            it->second.direction = newDir.normalized();
            
            if (wasBaked) bakeBody(id);
        }
    }

    void bakeBody(int id) {
        auto it = bodies.find(id);
        if (it == bodies.end() || it->second.baked) return;

        size_t w = skybox.getWidth();
        size_t h = skybox.getHeight();
        std::vector<uint8_t> newColor = {it->second.r, it->second.g, it->second.b, it->second.emittance};
        
        it->second.backup.clear();

        for (size_t y = 0; y < h; ++y) {
            float v = (static_cast<float>(y) + 0.5f) / h; 
            for (size_t x = 0; x < w; ++x) {
                float u = (static_cast<float>(x) + 0.5f) / w;
                Vec3 pixelDir = uvToDir(u, v);
                
                if (pixelDir.dot(it->second.direction) >= it->second.cosAngularRadius) {
                    typename CelestialBody::PixelBackup backup;
                    backup.x = x;
                    backup.y = y;
                    backup.data = skybox.getPixel(x, y);
                    it->second.backup.push_back(std::move(backup));
                    
                    skybox.setPixel(x, y, newColor);
                }
            }
        }
        it->second.baked = true;
    }

    void resetBody(int id) {
        auto it = bodies.find(id);
        if (it == bodies.end() || !it->second.baked) return;

        for (const auto& backup : it->second.backup) {
            skybox.setPixel(backup.x, backup.y, backup.data);
        }
        
        it->second.backup.clear();
        it->second.backup.shrink_to_fit();
        it->second.baked = false;
    }
};

}