#include "../util/grid/grid3.hpp"
#include "../util/output/bmpwriter.hpp"
#include "../util/noise/pnoise2.hpp"
#include "../util/timing_decorator.cpp"
#include <cmath>
#include <random>
#include <filesystem>

// Separate component storage
class ComponentManager {
private:
    std::unordered_map<size_t, Vec4ui8> colors_;
    std::unordered_map<size_t, float> sizes_;
    
public:
    void set_color(size_t id, const Vec4ui8& color) { colors_[id] = color; }
    Vec4ui8 get_color(size_t id) const {
        auto it = colors_.find(id);
        return it != colors_.end() ? it->second : Vec4ui8(1, 1, 1, 1);
    }
};

// Generate noise-based positions
std::vector<Vec3f> generate_noise_positions(int width, int height, float scale = 0.01f, float threshold = 0.3f) {
    std::vector<Vec3f> positions;
    PNoise2 noise_generator(std::random_device{}());
    
    // Generate positions based on 2D noise
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Convert to normalized coordinates
            float nx = static_cast<float>(x) / width;
            float ny = static_cast<float>(y) / height;
            
            // Generate noise at multiple octaves for more interesting patterns
            float noise_value = 0.0f;
            float amplitude = 1.0f;
            float frequency = 1.0f;
            float max_value = 0.0f;
            
            for (int i = 0; i < 4; i++) {
                Vec2 sample_point(nx * frequency * scale, ny * frequency * scale);
                noise_value += amplitude * noise_generator.permute(sample_point);
                max_value += amplitude;
                amplitude *= 0.5f;
                frequency *= 2.0f;
            }
            
            noise_value = (noise_value / max_value + 1.0f) * 0.5f; // Normalize to [0, 1]
            
            // Threshold to create sparse distribution
            if (noise_value > threshold) {
                // Map to world coordinates [-512, 512]
                float world_x = (nx * 2.0f - 1.0f) * 512.0f;
                float world_y = (ny * 2.0f - 1.0f) * 512.0f;
                float world_z = noise_value * 100.0f; // Use noise value for height
                
                positions.emplace_back(world_x, world_y, world_z);
            }
        }
    }
    
    return positions;
}

// Generate 3D noise-based positions (more varied in 3D space)
std::vector<Vec3f> generate_3d_noise_positions(int grid_size, float scale = 0.02f, float threshold = 0.4f) {
    std::vector<Vec3f> positions;
    PNoise2 noise_generator(std::random_device{}());
    int sizehalf = grid_size / 2;
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            for (int z = 0; z < grid_size; z++) {
                // Convert to normalized coordinates
                float nx = static_cast<float>(x) / sizehalf;
                float ny = static_cast<float>(y) / sizehalf;
                float nz = static_cast<float>(z) / sizehalf;
                
                // Generate 3D noise
                Vec3f sample_point(nx, ny, nz);
                float noise_value = noise_generator.permute(sample_point);
                
                // Threshold to create sparse distribution
                if (noise_value > threshold) {
                    // Map to world coordinates [-512, 512]
                    float world_x = (x - sizehalf);
                    float world_y = (y - sizehalf);
                    float world_z = (z - sizehalf);
                    
                    positions.emplace_back(world_x, world_y, world_z);
                }
            }
        }
    }
    
    return positions;
}

// Generate gradient color based on position
Vec4ui8 get_gradient_color(const Vec3f& pos, const Vec3f& min_pos, const Vec3f& max_pos) {
    // Normalize position to [0, 1]
    Vec3f normalized(
        (pos.x - min_pos.x) / (max_pos.x - min_pos.x),
        (pos.y - min_pos.y) / (max_pos.y - min_pos.y),
        (pos.z - min_pos.z) / (max_pos.z - min_pos.z)
    );
    
    // Create RGB gradient based on normalized coordinates
    uint8_t r = static_cast<uint8_t>(normalized.x * 255);
    uint8_t g = static_cast<uint8_t>(normalized.y * 255);
    uint8_t b = static_cast<uint8_t>(normalized.z * 255);
    
    return Vec4ui8(r, g, b, 255);
}

// Generate noise-based color
Vec4ui8 get_noise_color(const Vec3f& pos, PNoise2& noise_generator) {
    // Use noise to generate color
    Vec3f sample_point(pos.x * 0.005f, pos.y * 0.005f, pos.z * 0.005f);
    float noise_r = noise_generator.permute(sample_point);
    noise_r = (noise_r + 1.0f) * 0.5f; // Normalize to [0, 1]
    
    sample_point = Vec3f(pos.x * 0.007f, pos.y * 0.007f, pos.z * 0.007f);
    float noise_g = noise_generator.permute(sample_point);
    noise_g = (noise_g + 1.0f) * 0.5f;
    
    sample_point = Vec3f(pos.x * 0.009f, pos.y * 0.009f, pos.z * 0.009f);
    float noise_b = noise_generator.permute(sample_point);
    noise_b = (noise_b + 1.0f) * 0.5f;
    
    uint8_t r = static_cast<uint8_t>(noise_r * 255);
    uint8_t g = static_cast<uint8_t>(noise_g * 255);
    uint8_t b = static_cast<uint8_t>(noise_b * 255);
    
    return Vec4ui8(r, g, b, 255);
}

// Function to save frame with given filename
bool save_frame(const frame& f, const std::string& filename) {
    return BMPWriter::saveBMP(filename, f);
}

// Create directory if it doesn't exist
void ensure_directory(const std::string& path) {
    std::filesystem::create_directories(path);
}

int main() {
    // Create output directory
    ensure_directory("output");
    
    // Create pure spatial grid
    Grid3 grid(10.0f, 15.0f); // Larger cell size for sparse distribution
    
    // Separate component storage
    ComponentManager components;
    
    // Generate noise-based positions
    std::cout << "Generating noise-based positions..." << std::endl;
    
    // Option 1: 2D noise (faster)
    // auto positions = generate_noise_positions(200, 200, 0.02f, 0.4f);
    
    // Option 2: 3D noise (more interesting, but slower)
    auto positions = generate_3d_noise_positions(1024, 0.05f, 0.5f);
    
    std::cout << "Generated " << positions.size() << " positions" << std::endl;
    
    // Add positions to grid
    std::cout << "Adding positions to grid..." << std::endl;
    std::vector<size_t> ids = grid.bulk_add_positions(positions);
    
    // Generate bounds for color gradient
    auto bounds = grid.get_bounds();
    Vec3f min_pos = bounds.first;
    Vec3f max_pos = bounds.second;
    
    // Create noise generator for colors
    PNoise2 color_noise_generator(42); // Fixed seed for consistent colors
    
    // Assign colors to components
    std::cout << "Assigning colors..." << std::endl;
    for (size_t i = 0; i < ids.size(); i++) {
        Vec3f pos = grid.get_position(ids[i]);
        
        // Choose color method:
        // Vec4ui8 color = get_gradient_color(pos, min_pos, max_pos);
        Vec4ui8 color = get_noise_color(pos, color_noise_generator);
        
        components.set_color(ids[i], color);
    }
    
    // Query neighbors for first few points (optional)
    if (!ids.empty()) {
        auto neighbors = grid.get_neighbors(ids[0]);
        std::cout << "First point has " << neighbors.size() << " neighbors" << std::endl;
    }
    
    // Region to render (full noise map area)
    Vec3f render_min(-512, -512, -512);
    Vec3f render_max(512, 512, 512);
    Vec2 resolution(1024, 1024); // Higher resolution for better detail
    
    // Define multiple view angles
    struct ViewAngle {
        std::string name;
        Vec3f eye_position;
        Vec3f look_at;
    };
    
    std::vector<ViewAngle> view_angles = {
        // Orthographic views
        {"front", Vec3f(0, 0, -800), Vec3f(0, 0, 0)},
        {"back", Vec3f(0, 0, 800), Vec3f(0, 0, 0)},
        {"left", Vec3f(-800, 0, 0), Vec3f(0, 0, 0)},
        {"right", Vec3f(800, 0, 0), Vec3f(0, 0, 0)},
        {"top", Vec3f(0, 800, 0), Vec3f(0, 0, 0)},
        {"bottom", Vec3f(0, -800, 0), Vec3f(0, 0, 0)},
        
        // 45° angle views
        {"front_right_top", Vec3f(800, 800, -800), Vec3f(0, 0, 0)},
        {"front_left_top", Vec3f(-800, 800, -800), Vec3f(0, 0, 0)},
        {"front_right_bottom", Vec3f(800, -800, -800), Vec3f(0, 0, 0)},
        {"front_left_bottom", Vec3f(-800, -800, -800), Vec3f(0, 0, 0)},
        
        {"back_right_top", Vec3f(800, 800, 800), Vec3f(0, 0, 0)},
        {"back_left_top", Vec3f(-800, 800, 800), Vec3f(0, 0, 0)},
        {"back_right_bottom", Vec3f(800, -800, 800), Vec3f(0, 0, 0)},
        {"back_left_bottom", Vec3f(-800, -800, 800), Vec3f(0, 0, 0)},
        
        // Additional interesting angles
        {"diagonal_xy", Vec3f(800, 800, 0), Vec3f(0, 0, 0)},
        {"diagonal_xz", Vec3f(800, 0, 800), Vec3f(0, 0, 0)},
        {"diagonal_yz", Vec3f(0, 800, 800), Vec3f(0, 0, 0)},
        
        // Isometric-like views
        {"isometric_1", Vec3f(600, 600, 600), Vec3f(0, 0, 0)},
        {"isometric_2", Vec3f(-600, 600, 600), Vec3f(0, 0, 0)},
        {"isometric_3", Vec3f(600, -600, 600), Vec3f(0, 0, 0)},
        {"isometric_4", Vec3f(-600, -600, 600), Vec3f(0, 0, 0)}
    };
    
    // Render and save from each angle
    std::cout << "Rendering from multiple angles..." << std::endl;
    int saved_count = 0;
    
    for (const auto& view : view_angles) {
        // Create view ray
        Vec3f direction = (view.look_at - view.eye_position).normalized();
        Ray3<float> view_ray(view.eye_position, direction);
        
        // Render the frame
        auto frame = Grid3View::render_region(
            grid,
            render_min,
            render_max,
            resolution,
            view_ray,
            [&](size_t id) { return components.get_color(id); },
            frame::colormap::RGBA,
            Vec4ui8(0, 0, 0, 255) // Black background
        );
        
        // Save the rendered frame
        std::string filename = "output/view_" + view.name + ".bmp";
        bool saved = save_frame(frame, filename);
        
        if (saved) {
            saved_count++;
            std::cout << "Saved: " << filename << std::endl;
        } else {
            std::cout << "Failed to save: " << filename << std::endl;
        }
    }
    
    std::cout << "\nSuccessfully saved " << saved_count << " out of " << view_angles.size() 
              << " views to output/" << std::endl;
    
    // Optional: Save a summary image with grid statistics
    std::cout << "\nGrid statistics:" << std::endl;
    std::cout << "Total positions: " << grid.size() << std::endl;
    std::cout << "Bounds: [" << min_pos << "] to [" << max_pos << "]" << std::endl;
    
    return 0;
}