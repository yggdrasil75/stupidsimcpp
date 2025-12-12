#ifndef GRID2_HPP
#define GRID2_HPP

#include <unordered_map>
#include "../vectorlogic/vec2.hpp"
#include "../vectorlogic/vec3.hpp"
#include "../vectorlogic/vec4.hpp"
#include "../timing_decorator.hpp"
#include "../output/frame.hpp"
#include "../noise/pnoise2.hpp"
#include <vector>
#include <unordered_set>
#include <execution>
#include <algorithm>
#include <random>

constexpr float EPSILON = 0.0000000000000000000000001;
constexpr float ELEMENTARY_CHARGE = 1.602176634e-19f;  // Coulomb
constexpr float COULOMB_CONSTANT = 8.9875517923e9f;    // N·m²/C²
constexpr float ELECTRON_MASS = 9.1093837015e-31f;     // kg (relative units)
constexpr float PROTON_MASS = 1.67262192369e-27f;      // kg (relative units)
constexpr float NEUTRON_MASS = 1.67492749804e-27f;     // kg (relative units)

/// @brief Represents different types of atoms/elements
enum class ElementType {
    HYDROGEN,    // 1 proton, 1 electron
    HELIUM,      // 2 protons, 2 neutrons, 2 electrons
    LITHIUM,     // 3 protons, 4 neutrons, 3 electrons
    CARBON,      // 6 protons, 6 neutrons, 6 electrons
    OXYGEN,      // 8 protons, 8 neutrons, 8 electrons
    IRON,        // 26 protons, 30 neutrons, 26 electrons
    URANIUM,     // 92 protons, 146 neutrons, 92 electrons
    CUSTOM       // User-defined composition
};

/// @brief Represents a single atom with subatomic particle composition
class AtomicPixel {
protected:
    size_t id;
    Vec4f color;
    Vec2 pos;
    Vec2 velocity;
    Vec2 acceleration;
    
    // Atomic composition
    int protons;
    int neutrons;
    int electrons;
    
    // Physical properties
    float mass;           // Total mass in relative units
    float charge;         // Net charge in elementary charge units
    float radius;         // Atomic radius (for collision/repulsion)
    ElementType element;
    
    // State properties
    bool ionized;
    float excitation;
    float temperature;
    
public:
    // Constructors
    AtomicPixel(size_t id, Vec4f color, Vec2 pos) 
        : id(id), color(color), pos(pos), velocity(Vec2(0, 0)), acceleration(Vec2(0, 0)),
          protons(1), neutrons(0), electrons(1), 
          mass(PROTON_MASS + ELECTRON_MASS),
          charge(0.0f),
          radius(1.0f),
          element(ElementType::HYDROGEN),
          ionized(false),
          excitation(0.0f),
          temperature(300.0f) {
        updateProperties();
    }
    
    AtomicPixel(size_t id, Vec4f color, Vec2 pos, ElementType element)
        : id(id), color(color), pos(pos), velocity(Vec2(0, 0)), acceleration(Vec2(0, 0)),
          element(element),
          ionized(false),
          excitation(0.0f),
          temperature(300.0f) {
        setElement(element);
    }
    
    AtomicPixel(size_t id, Vec4f color, Vec2 pos, int p, int n, int e)
        : id(id), color(color), pos(pos), velocity(Vec2(0, 0)), acceleration(Vec2(0, 0)),
          protons(p), neutrons(n), electrons(e),
          element(ElementType::CUSTOM),
          ionized(false),
          excitation(0.0f),
          temperature(300.0f) {
        updateProperties();
    }

    // Getters
    Vec4f getColor() const { return color; }
    Vec2 getPosition() const { return pos; }
    Vec2 getVelocity() const { return velocity; }
    Vec2 getAcceleration() const { return acceleration; }
    
    int getProtons() const { return protons; }
    int getNeutrons() const { return neutrons; }
    int getElectrons() const { return electrons; }
    int getAtomicNumber() const { return protons; }
    int getMassNumber() const { return protons + neutrons; }
    
    float getMass() const { return mass; }
    float getCharge() const { return charge; }
    float getRadius() const { return radius; }
    ElementType getElement() const { return element; }
    
    bool isIonized() const { return ionized; }
    float getExcitation() const { return excitation; }
    float getTemperature() const { return temperature; }
    
    // Setters
    void setColor(Vec4f newColor) { color = newColor; }
    void setPosition(Vec2 newPos) { pos = newPos; }
    void setVelocity(Vec2 newVel) { velocity = newVel; }
    void setAcceleration(Vec2 newAcc) { acceleration = newAcc; }
    
    void setProtons(int p) { 
        protons = p; 
        updateProperties(); 
    }
    
    void setNeutrons(int n) { 
        neutrons = n; 
        updateProperties(); 
    }
    
    void setElectrons(int e) { 
        electrons = e; 
        updateProperties(); 
    }
    
    void setElement(ElementType elem) {
        element = elem;
        switch (element) {
            case ElementType::HYDROGEN:
                protons = 1; neutrons = 0; electrons = 1;
                break;
            case ElementType::HELIUM:
                protons = 2; neutrons = 2; electrons = 2;
                break;
            case ElementType::LITHIUM:
                protons = 3; neutrons = 4; electrons = 3;
                break;
            case ElementType::CARBON:
                protons = 6; neutrons = 6; electrons = 6;
                break;
            case ElementType::OXYGEN:
                protons = 8; neutrons = 8; electrons = 8;
                break;
            case ElementType::IRON:
                protons = 26; neutrons = 30; electrons = 26;
                break;
            case ElementType::URANIUM:
                protons = 92; neutrons = 146; electrons = 92;
                break;
            case ElementType::CUSTOM:
                // Keep existing values
                break;
        }
        updateProperties();
    }
    
    void setIonized(bool ion) { ionized = ion; }
    void setExcitation(float exc) { excitation = exc; }
    void setTemperature(float temp) { temperature = temp; }
    
    // Movement
    void move(Vec2 newPos) { pos = newPos; }
    
    void applyForce(Vec2 force, float deltaTime) {
        acceleration = force / mass;
        velocity += acceleration * deltaTime;
        pos += velocity * deltaTime;
    }
    
    // Atomic operations
    void addProton() { 
        protons++; 
        updateProperties(); 
    }
    
    void removeProton() { 
        if (protons > 0) protons--; 
        updateProperties(); 
    }
    
    void addNeutron() { 
        neutrons++; 
        updateProperties(); 
    }
    
    void removeNeutron() { 
        if (neutrons > 0) neutrons--; 
        updateProperties(); 
    }
    
    void addElectron() { 
        electrons++; 
        updateProperties(); 
    }
    
    void removeElectron() { 
        if (electrons > 0) electrons--; 
        updateProperties(); 
    }
    
    void ionize() {
        if (electrons > 0) {
            electrons--;
            ionized = true;
            updateProperties();
        }
    }
    
    void recombine() {
        electrons = protons;  // Neutral state
        ionized = false;
        updateProperties();
    }
    
    // Update derived properties
    void updateProperties() {
        // Calculate net charge (in elementary charge units)
        charge = static_cast<float>(protons - electrons);
        
        // Calculate mass (simplified relative units)
        mass = protons * PROTON_MASS + neutrons * NEUTRON_MASS + electrons * ELECTRON_MASS;
        
        // Calculate approximate atomic radius (empirical)
        // Rough scaling: radius ~ (mass number)^(1/3) * Bohr radius scale
        radius = 0.1f * std::pow(static_cast<float>(protons + neutrons), 1.0f/3.0f);
        
        // Update ionization state
        ionized = (electrons != protons);
        
        // Update color based on element/charge (simplified visualization)
        updateColorFromProperties();
    }
    
    // Update visualization color based on atomic properties
    void updateColorFromProperties() {
        // Base color from element type
        switch (element) {
            case ElementType::HYDROGEN:
                color = Vec4f(1.0f, 0.5f, 0.5f, 1.0f);  // Pinkish
                break;
            case ElementType::HELIUM:
                color = Vec4f(0.8f, 0.9f, 1.0f, 1.0f);  // Light blue
                break;
            case ElementType::LITHIUM:
                color = Vec4f(0.8f, 0.5f, 0.2f, 1.0f);  // Bronze
                break;
            case ElementType::CARBON:
                color = Vec4f(0.2f, 0.2f, 0.2f, 1.0f);  // Dark gray
                break;
            case ElementType::OXYGEN:
                color = Vec4f(0.9f, 0.1f, 0.1f, 1.0f);  // Red
                break;
            case ElementType::IRON:
                color = Vec4f(0.7f, 0.4f, 0.1f, 1.0f);  // Rust orange
                break;
            case ElementType::URANIUM:
                color = Vec4f(0.0f, 0.8f, 0.0f, 1.0f);  // Green
                break;
            case ElementType::CUSTOM:
                // Keep existing or calculate from composition
                break;
        }
        
        // Modify based on charge
        if (charge > 0) {
            // Positive charge - shift toward red
            color.r = std::min(1.0f, color.r + charge * 0.3f);
            color.g = std::max(0.0f, color.g - charge * 0.2f);
            color.b = std::max(0.0f, color.b - charge * 0.2f);
        } else if (charge < 0) {
            // Negative charge - shift toward blue
            float absCharge = std::abs(charge);
            color.r = std::max(0.0f, color.r - absCharge * 0.2f);
            color.g = std::max(0.0f, color.g - absCharge * 0.2f);
            color.b = std::min(1.0f, color.b + absCharge * 0.3f);
        }
        
        // Modify based on temperature (simplified blackbody)
        float tempFactor = temperature / 1000.0f;
        color.r = std::min(1.0f, color.r * (1.0f + tempFactor * 0.5f));
        color.g = std::min(1.0f, color.g * (1.0f + tempFactor * 0.3f));
        color.b = std::max(0.0f, color.b * (1.0f - tempFactor * 0.2f));
    }
    
    // Calculate Coulomb force from another atom
    Vec2 calculateCoulombForce(const AtomicPixel& other, float distance) const {
        if (distance < EPSILON) return Vec2(0, 0);
        
        // Coulomb's law: F = k * q1 * q2 / r²
        float forceMagnitude = COULOMB_CONSTANT * charge * other.charge * 
                              (ELEMENTARY_CHARGE * ELEMENTARY_CHARGE) / (distance * distance);
        
        // Direction vector (from this to other)
        Vec2 direction = other.pos - pos;
        direction.normalized();
        
        return direction * forceMagnitude;
    }
    
    // Calculate gravitational force (mass-based, simplified)
    Vec2 calculateGravitationalForce(const AtomicPixel& other, float distance, float G = 6.67430e-11f) const {
        if (distance < EPSILON) return Vec2(0, 0);
        
        // Newton's law of universal gravitation: F = G * m1 * m2 / r²
        float forceMagnitude = G * mass * other.mass / (distance * distance);
        
        // Direction vector (from this to other)
        Vec2 direction = other.pos - pos;
        direction.normalized();
        
        return direction * forceMagnitude;
    }
    
    // Calculate Lennard-Jones potential (for short-range repulsion/attraction)
    Vec2 calculateLennardJonesForce(const AtomicPixel& other, float distance) const {
        if (distance < EPSILON) return Vec2(0, 0);
        
        // Simplified Lennard-Jones parameters
        float sigma = (radius + other.radius) * 0.5f;
        float epsilon = 1.0e-3f;  // Interaction strength
        
        float r = distance;
        float r6 = std::pow(sigma / r, 6);
        float r12 = r6 * r6;
        
        // LJ force: F = 24 * epsilon * (2*(sigma/r)^12 - (sigma/r)^6) / r
        float forceMagnitude = 24.0f * epsilon * (2.0f * r12 - r6) / r;
        
        Vec2 direction = other.pos - pos;
        direction.normalized();
        
        return direction * forceMagnitude;
    }
};

/// @brief A bidirectional lookup helper to map internal IDs to 2D positions and vice-versa.
/// @details Maintains two hashmaps to allow O(1) lookup in either direction.
class reverselookupassistant {
private:
    std::unordered_map<size_t, Vec2> Positions;
    /// "Positions" reversed - stores the reverse mapping from Vec2 to ID.
    std::unordered_map<Vec2, size_t, Vec2::Hash> ƨnoiƚiƨoꟼ;
    size_t next_id;
public:
    /// @brief Get the Position associated with a specific ID.
    /// @throws std::out_of_range if the ID does not exist.
    Vec2 at(size_t id) const {
        auto it = Positions.at(id);
        return it;
    }

    /// @brief Get the ID associated with a specific Position.
    /// @throws std::out_of_range if the Position does not exist.
    size_t at(const Vec2& pos) const {
        size_t id = ƨnoiƚiƨoꟼ.at(pos);
        return id;
    }

    /// @brief Finds a position by ID (Wrapper for at).
    Vec2 find(size_t id) {
        return Positions.at(id);
    }

    /// @brief Registers a new position and assigns it a unique ID.
    /// @return The newly generated ID.
    size_t set(const Vec2& pos) {
        size_t id = next_id++;
        Positions[id] = pos;
        ƨnoiƚiƨoꟼ[pos] = id;
        return id;
    }

    /// @brief Removes an entry by ID.
    size_t remove(size_t id) {
        Vec2& pos = Positions[id];
        Positions.erase(id);
        ƨnoiƚiƨoꟼ.erase(pos);
        return id;
    }

    /// @brief Removes an entry by Position.
    size_t remove(const Vec2& pos) {
        size_t id = ƨnoiƚiƨoꟼ[pos];
        Positions.erase(id);
        ƨnoiƚiƨoꟼ.erase(pos);
        return id;
    }

    void reserve(size_t size) {
        Positions.reserve(size);
        ƨnoiƚiƨoꟼ.reserve(size);
    }

    size_t size() const {
        return Positions.size();
    }

    size_t getNext_id() {
        return next_id + 1;
    }
    
    size_t bucket_count() {
        return Positions.bucket_count();
    }

    bool empty() const {
        return Positions.empty();
    }
    
    void clear() {
        Positions.clear();
        Positions.rehash(0);
        ƨnoiƚiƨoꟼ.clear();
        ƨnoiƚiƨoꟼ.rehash(0);
        next_id = 0;
    }
    
    using iterator = typename std::unordered_map<size_t, Vec2>::iterator;
    using const_iterator = typename std::unordered_map<size_t, Vec2>::const_iterator;

    iterator begin() { 
        return Positions.begin(); 
    }
    iterator end() { 
        return Positions.end(); 
    }
    const_iterator begin() const { 
        return Positions.begin(); 
    }
    const_iterator end() const { 
        return Positions.end(); 
    }
    const_iterator cbegin() const { 
        return Positions.cbegin(); 
    }
    const_iterator cend() const { 
        return Positions.cend(); 
    }

    bool contains(size_t id) const {
        return (Positions.find(id) != Positions.end());
    }

    bool contains(const Vec2& pos) const {
        return (ƨnoiƚiƨoꟼ.find(pos) != ƨnoiƚiƨoꟼ.end());
    }
    
};

/// @brief Accelerates spatial queries by bucketizing positions into a grid.
class SpatialGrid {
private:
    float cellSize;
public:
    std::unordered_map<Vec2, std::unordered_set<size_t>, Vec2::Hash> grid;
    
    /// @brief Initializes the spatial grid.
    /// @param cellSize The dimension of the spatial buckets. Larger cells mean more items per bucket but fewer buckets.
    SpatialGrid(float cellSize = 2.0f) : cellSize(cellSize) {}
    
    /// @brief Converts world coordinates to spatial grid coordinates.
    Vec2 worldToGrid(const Vec2& worldPos) const {
        return (worldPos / cellSize).floor();
    }
    
    /// @brief Adds an object ID to the spatial index at the given position.
    void insert(size_t id, const Vec2& pos) {
        Vec2 gridPos = worldToGrid(pos);
        grid[gridPos].insert(id);
    }
    
    /// @brief Removes an object ID from the spatial index.
    void remove(size_t id, const Vec2& pos) {
        Vec2 gridPos = worldToGrid(pos);
        auto cellIt = grid.find(gridPos);
        if (cellIt != grid.end()) {
            cellIt->second.erase(id);
            if (cellIt->second.empty()) {
                grid.erase(cellIt);
            }
        }
    }
    
    /// @brief Moves an object within the spatial index (removes from old cell, adds to new if changed).
    void update(size_t id, const Vec2& oldPos, const Vec2& newPos) {
        Vec2 oldGridPos = worldToGrid(oldPos);
        Vec2 newGridPos = worldToGrid(newPos);
        
        if (oldGridPos != newGridPos) {
            remove(id, oldPos);
            insert(id, newPos);
        }
    }
    
    /// @brief Returns all IDs located in the specific grid cell containing 'center'.
    std::unordered_set<size_t> find(const Vec2& center) const {
        //Vec2 g2pos = worldToGrid(center);
        auto cellIt = grid.find(worldToGrid(center));
        if (cellIt != grid.end()) {
            return cellIt->second;
        }
        return std::unordered_set<size_t>();
    }

    /// @brief Finds all object IDs within a square area around the center.
    /// @param center The world position center.
    /// @param radius The search radius (defines the bounds of grid cells to check).
    /// @return A vector of candidate IDs (Note: this returns objects in valid grid cells, further distance checks may be required).
    std::vector<size_t> queryRange(const Vec2& center, float radius) const {
        std::vector<size_t> results;
        float radiusSq = radius * radius;
        
        // Calculate grid bounds for the query
        Vec2 minGrid = worldToGrid(center - Vec2(radius, radius));
        Vec2 maxGrid = worldToGrid(center + Vec2(radius, radius));
        
        size_t estimatedSize = (maxGrid.x - minGrid.x + 1) * (maxGrid.y - minGrid.y + 1) * 10;
        results.reserve(estimatedSize);

        // Check all relevant grid cells
        for (int x = minGrid.x; x <= maxGrid.x; ++x) {
            for (int y = minGrid.y; y <= maxGrid.y; ++y) {
                auto cellIt = grid.find(Vec2(x, y));
                if (cellIt != grid.end()) {
                    results.insert(results.end(), cellIt->second.begin(), cellIt->second.end());
                }
            }
        }
        
        return results;
    }
    
    void clear() {
        grid.clear();
        grid.rehash(0);
    }
};

/// @brief The main simulation grid class managing atomic interactions
class Grid2 { 
protected:
    //all positions
    reverselookupassistant Positions;
    std::unordered_map<size_t, AtomicPixel> Atoms;
    
    std::vector<size_t> unassignedIDs;
    
    float neighborRadius = 10.0f;  // Increased for atomic interactions
    
    //TODO: spatial map
    SpatialGrid spatialGrid;
    float spatialCellSize = neighborRadius * 1.5f;

    // Default background color for empty spaces
    Vec4f defaultBackgroundColor = Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    PNoise2 noisegen;

    bool regenpreventer = false;
    
    // Physics simulation parameters
    float timeStep = 0.016f;  // ~60 FPS
    float coulombStrength = 1.0f;  // Scaling factor for Coulomb force
    float gravityStrength = 1.0e-6f;  // Scaling factor for gravitational force
    float dampingFactor = 0.99f;  // Velocity damping
    float boundaryRepulsion = 100.0f;  // Force at boundaries
    
    // Simulation bounds
    Vec2 simulationBoundsMin = Vec2(-100.0f, -100.0f);
    Vec2 simulationBoundsMax = Vec2(100.0f, 100.0f);
    
    // Random number generator for atomic variations
    std::mt19937 rng;
    std::uniform_real_distribution<float> randomDist;
    
public:
    Grid2() : rng(std::random_device{}()), randomDist(0.0f, 1.0f) {
        optimizeSpatialGrid();
    }
    
    /// @brief Populates the grid with random atoms of various elements.
    /// @param minx Start X index.
    /// @param miny Start Y index.
    /// @param maxx End X index.
    /// @param maxy End Y index.
    /// @param density Probability of placing an atom at each position.
    /// @return Reference to self for chaining.
    Grid2 generateRandomAtoms(size_t minx, size_t miny, size_t maxx, size_t maxy, float density = 0.3f) {
        TIME_FUNCTION;
        std::cout << "Generating random atoms in region: (" << minx << ", " << miny 
                << ") to (" << maxx << ", " << maxy << ") with density: " << density << std::endl;
        
        std::vector<Vec2> poses;
        std::vector<Vec4f> colors;
        std::vector<ElementType> elements;
        
        // Common elements with probabilities
        std::vector<std::pair<ElementType, float>> elementProbs = {
            {ElementType::HYDROGEN, 0.4f},
            {ElementType::HELIUM, 0.2f},
            {ElementType::CARBON, 0.15f},
            {ElementType::OXYGEN, 0.15f},
            {ElementType::IRON, 0.1f}
        };
        
        #pragma omp parallel for
        for (int x = minx; x < maxx; x++) {
            #pragma omp parallel for
            for (int y = miny; y < maxy; y++) {
                if (randomDist(rng) < density) {
                    // Choose random element
                    float rand = randomDist(rng);
                    float accum = 0.0f;
                    ElementType chosenElement = ElementType::HYDROGEN;
                    
                    for (const auto& [elem, prob] : elementProbs) {
                        accum += prob;
                        if (rand <= accum) {
                            chosenElement = elem;
                            break;
                        }
                    }
                    
                    #pragma omp critical
                    poses.push_back(Vec2(x, y));
                    #pragma omp critical
                    elements.push_back(chosenElement);
                    
                    // Create placeholder color (will be set by AtomicPixel constructor)
                    #pragma omp critical
                    colors.push_back(Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
                }
            }
        }
        
        bulkAddAtoms(poses, colors, elements);
        return *this;
    }
    
    /// @brief Adds a new atom to the grid.
    /// @param pos The 2D world position.
    /// @param color The color vector.
    /// @param element The type of atom/element.
    /// @return The unique ID assigned to the new atom.
    size_t addAtom(const Vec2& pos, const Vec4f& color, ElementType element = ElementType::HYDROGEN) {
        size_t id = Positions.set(pos);
        Atoms.emplace(id, AtomicPixel(id, color, pos, element));
        spatialGrid.insert(id, pos);
        return id;
    }
    
    /// @brief Adds a new custom atom with specific proton/neutron/electron counts.
    size_t addCustomAtom(const Vec2& pos, const Vec4f& color, int protons, int neutrons, int electrons) {
        size_t id = Positions.set(pos);
        Atoms.emplace(id, AtomicPixel(id, color, pos, protons, neutrons, electrons));
        spatialGrid.insert(id, pos);
        return id;
    }

    /// @brief Batch insertion of atoms for efficiency.
    std::vector<size_t> bulkAddAtoms(const std::vector<Vec2> poses, 
                                     const std::vector<Vec4f> colors,
                                     const std::vector<ElementType> elements) {
        TIME_FUNCTION;
        
        if (poses.size() != colors.size() || poses.size() != elements.size()) {
            throw std::invalid_argument("Vector sizes must match");
        }
        
        // Reserve space in maps to avoid rehashing
        if (Positions.bucket_count() < Positions.size() + poses.size()) {
            Positions.reserve(Positions.size() + poses.size());
            Atoms.reserve(Atoms.size() + poses.size());
        }
        
        // Batch insertion
        std::vector<size_t> newids;
        newids.reserve(poses.size());
        
        for (size_t i = 0; i < poses.size(); ++i) {
            size_t id = Positions.set(poses[i]);
            Atoms.emplace(id, AtomicPixel(id, colors[i], poses[i], elements[i]));
            spatialGrid.insert(id, poses[i]);
            newids.push_back(id);
        }
        
        shrinkIfNeeded();
        
        return newids;
    }
    
    /// @brief Updates the physics simulation for all atoms.
    /// @param deltaTime Time step for integration (if 0, uses internal timeStep).
    void updatePhysics(float deltaTime = 0.0f) {
        TIME_FUNCTION;
        
        float dt = (deltaTime == 0.0f) ? timeStep : deltaTime;
        
        // Calculate forces between all atoms
        std::unordered_map<size_t, Vec2> forces;
        
        // For each atom, calculate forces from neighbors
        #pragma omp parallel for
        for (const auto& [id1, atom1] : Atoms) {
            Vec2 totalForce(0, 0);
            Vec2 pos1 = atom1.getPosition();
            
            // Get neighbors within interaction radius
            auto neighbors = getNeighbors(id1);
            
            #pragma omp parallel for
            for (size_t id2 : neighbors) {
                auto& atom2 = Atoms.at(id2);
                Vec2 pos2 = atom2.getPosition();
                float distance = pos1.distance(pos2);
                
                if (distance > EPSILON) {
                    // Coulomb force (charge-based "gravity")
                    Vec2 coulombForce = atom1.calculateCoulombForce(atom2, distance);
                    totalForce += coulombForce * coulombStrength;
                    
                    // Gravitational force (mass-based)
                    Vec2 gravityForce = atom1.calculateGravitationalForce(atom2, distance);
                    totalForce += gravityForce * gravityStrength;
                    
                    // Short-range repulsion (Lennard-Jones)
                    if (distance < (atom1.getRadius() + atom2.getRadius()) * 2.0f) {
                        Vec2 ljForce = atom1.calculateLennardJonesForce(atom2, distance);
                        totalForce += ljForce;
                    }
                }
            }
            
            // Boundary forces (keep atoms within simulation bounds)
            if (pos1.x < simulationBoundsMin.x) {
                totalForce.x += boundaryRepulsion;
            } else if (pos1.x > simulationBoundsMax.x) {
                totalForce.x -= boundaryRepulsion;
            }
            
            if (pos1.y < simulationBoundsMin.y) {
                totalForce.y += boundaryRepulsion;
            } else if (pos1.y > simulationBoundsMax.y) {
                totalForce.y -= boundaryRepulsion;
            }
            
            // Damping
            Vec2 velocity = atom1.getVelocity();
            totalForce -= velocity * (1.0f - dampingFactor) * atom1.getMass() / dt;
            
            // Apply force
            #pragma omp critical
            Atoms.at(id1).applyForce(totalForce, dt);
            
            // Update spatial grid if position changed significantly
            Vec2 newPos = Atoms.at(id1).getPosition();
            if (pos1.distanceSquared(newPos) > 0.01f) {
                spatialGrid.update(id1, pos1, newPos);
                Positions.at(id1) = newPos;
            }
        }
    }
    
    /// @brief Simulates atomic interactions and reactions.
    void simulateAtomicInteractions() {
        TIME_FUNCTION;
        
        // Check for collisions and possible reactions
        #pragma omp parallel for
        for (const auto& [id1, atom1] : Atoms) {
            Vec2 pos1 = atom1.getPosition();
            float radius1 = atom1.getRadius();
            
            auto neighbors = getNeighbors(id1);
            
            #pragma omp parallel for
            for (size_t id2 : neighbors) {
                if (id1 >= id2) continue;  // Avoid duplicate checks
                
                auto& atom2 = Atoms.at(id2);
                Vec2 pos2 = atom2.getPosition();
                float distance = pos1.distance(pos2);
                float combinedRadius = radius1 + atom2.getRadius();
                
                // If atoms are very close, they might interact
                if (distance < combinedRadius * 0.5f) {
                    // Simple fusion reaction (for demonstration)
                    if (randomDist(rng) < 0.01f) {  // 1% chance per collision
                        simulateFusion(id1, id2);
                    }
                    
                    // Electron transfer (ionization/recombination)
                    if (randomDist(rng) < 0.05f) {
                        simulateElectronTransfer(id1, id2);
                    }
                }
            }
        }
    }
    
    /// @brief Simulates a simple fusion reaction between two atoms.
    void simulateFusion(size_t id1, size_t id2) {
        auto& atom1 = Atoms.at(id1);
        auto& atom2 = Atoms.at(id2);
        
        // Simple hydrogen fusion: H + H → He (simplified)
        if (atom1.getElement() == ElementType::HYDROGEN && 
            atom2.getElement() == ElementType::HYDROGEN) {
            
            // Create a helium atom at midpoint
            Vec2 midPoint = (atom1.getPosition() + atom2.getPosition()) * 0.5f;
            Vec4f heColor = Vec4f(0.8f, 0.9f, 1.0f, 1.0f);
            
            // Remove original atoms
            removeID(id1);
            removeID(id2);
            
            // Add helium atom
            addAtom(midPoint, heColor, ElementType::HELIUM);
            
            // Add energy release (velocity boost to nearby atoms)
            auto neighbors = getPositionVecRegion(midPoint, 5.0f);
            #pragma omp parallel for
            for (size_t neighborId : neighbors) {
                if (Atoms.find(neighborId) != Atoms.end()) {
                    Vec2 dir = Atoms.at(neighborId).getPosition() - midPoint;
                    if (dir.length() > EPSILON) {
                        dir.normalized();
                        Atoms.at(neighborId).setVelocity(
                            Atoms.at(neighborId).getVelocity() + dir * 10.0f
                        );
                    }
                }
            }
        }
    }
    
    /// @brief Simulates electron transfer between atoms.
    void simulateElectronTransfer(size_t id1, size_t id2) {
        auto& atom1 = Atoms.at(id1);
        auto& atom2 = Atoms.at(id2);
        
        // Atom with higher electron affinity gains an electron
        float affinity1 = atom1.getProtons() / static_cast<float>(atom1.getRadius());
        float affinity2 = atom2.getProtons() / static_cast<float>(atom2.getRadius());
        
        if (affinity1 > affinity2 && atom2.getElectrons() > 0) {
            // atom1 gains electron from atom2
            atom1.addElectron();
            atom2.removeElectron();
        } else if (affinity2 > affinity1 && atom1.getElectrons() > 0) {
            // atom2 gains electron from atom1
            atom2.addElectron();
            atom1.removeElectron();
        }
    }
    
    /// @brief Sets the default background color.
    void setDefault(const Vec4f& color) {
        defaultBackgroundColor = color;
    }
    
    /// @brief Sets the default background color components.
    void setDefault(float r, float g, float b, float a = 0.0f) {
        defaultBackgroundColor = Vec4f(r, g, b, a);
    }
        
    /// @brief Moves an atom to a new position and updates spatial indexing.
    void setPosition(size_t id, const Vec2& newPosition) {
        Vec2 oldPosition = Positions.at(id);
        Atoms.at(id).setPosition(newPosition);
        spatialGrid.update(id, oldPosition, newPosition);
        Positions.at(id) = newPosition;
    }
        
    // Set color by id
    void setColor(size_t id, const Vec4f color) {
        Atoms.at(id).setColor(color);
    }
    
    /// @brief Sets the radius used for neighbor queries.
    /// @details Triggers an optimization of the spatial grid cell size.
    void setNeighborRadius(float radius) {
        neighborRadius = radius;
        optimizeSpatialGrid();
    }
    
    /// @brief Sets physics simulation parameters.
    void setPhysicsParameters(float newTimeStep = 0.016f, 
                             float newCoulombStrength = 1.0f,
                             float newGravityStrength = 1.0e-6f,
                             float newDamping = 0.99f) {
        timeStep = newTimeStep;
        coulombStrength = newCoulombStrength;
        gravityStrength = newGravityStrength;
        dampingFactor = newDamping;
    }
    
    /// @brief Sets simulation boundaries.
    void setSimulationBounds(const Vec2& min, const Vec2& max) {
        simulationBoundsMin = min;
        simulationBoundsMax = max;
    }

    // Get current default background color
    Vec4f getDefaultBackgroundColor() const {
        return defaultBackgroundColor;
    }

    // Get position from id
    Vec2 getPositionID(size_t id) const {
        Vec2 it = Positions.at(id);
        return it;
    }

    /// @brief Finds the ID of an atom at a given position.
    /// @param pos The position to query.
    /// @param radius If 0.0, performs an exact match. If > 0.0, returns the first atom found within the radius.
    /// @return The ID of the found atom.
    /// @throws std::out_of_range If no atom is found.
    size_t getPositionVec(const Vec2& pos, float radius = 0.0f) const {
        TIME_FUNCTION;
        if (radius == 0.0f) {
            // Exact match - use spatial grid to find the cell
            Vec2 gridPos = spatialGrid.worldToGrid(pos);
            auto cellIt = spatialGrid.grid.find(gridPos);
            if (cellIt != spatialGrid.grid.end()) {
                for (size_t id : cellIt->second) {
                    if (Positions.at(id) == pos) {
                        return id;
                    }
                }
            }
            throw std::out_of_range("Position not found");
        } else {
            auto results = getPositionVecRegion(pos, radius);
            if (!results.empty()) {
                return results[0]; // Return first found
            }
            throw std::out_of_range("No positions found in radius");
        }
    }

    /// @brief Finds an atom ID or creates a new one at the given position.
    /// @param pos Target position.
    /// @param radius Search radius for existing atoms.
    /// @param create If true, creates a new atom if none is found.
    /// @return The ID of the existing or newly created atom.
    size_t getOrCreatePositionVec(const Vec2& pos, float radius = 0.0f, bool create = true) {
        //TIME_FUNCTION; //called too many times and average time is less than 0.0000001 so ignore it.
        if (radius == 0.0f) {
            Vec2 gridPos = spatialGrid.worldToGrid(pos);
            auto cellIt = spatialGrid.grid.find(gridPos);
            if (cellIt != spatialGrid.grid.end()) {
                for (size_t id : cellIt->second) {
                    if (Positions.at(id) == pos) {
                        return id;
                    }
                }
            }
            if (create) {
                return addAtom(pos, defaultBackgroundColor, ElementType::HYDROGEN);
            }
            throw std::out_of_range("Position not found");
        } else {
            auto results = getPositionVecRegion(pos, radius);
            if (!results.empty()) {
                return results[0];
            }
            if (create) {
                return addAtom(pos, defaultBackgroundColor, ElementType::HYDROGEN);
            }
            throw std::out_of_range("No positions found in radius");
        }
    }

    /// @brief Returns a list of all atom IDs within a specified radius of a position.
    std::vector<size_t> getPositionVecRegion(const Vec2& pos, float radius = 1.0f) const {
        //TIME_FUNCTION;
        float searchRadius = (radius == 0.0f) ? std::numeric_limits<float>::epsilon() : radius;
        
        // Get candidates from spatial grid
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, searchRadius);
        
        // Fine-filter by exact distance
        std::vector<size_t> results;
        float radiusSq = searchRadius * searchRadius;
        
        for (size_t id : candidates) {
            if (Positions.at(id).distanceSquared(pos) <= radiusSq) {
                results.push_back(id);
            }
        }
        
        return results;
    }
    
    Vec4f getColor(size_t id) {
        return Atoms.at(id).getColor();
    }
    
    /// @brief Gets the atomic properties of an atom.
    AtomicPixel& getAtom(size_t id) {
        return Atoms.at(id);
    }
    
    /// @brief Gets all atoms of a specific element type.
    std::vector<size_t> getAtomsByElement(ElementType element) const {
        std::vector<size_t> result;
        #pragma omp parallel for
        for (const auto& [id, atom] : Atoms) {
            if (atom.getElement() == element) {
                #pragma omp critical
                result.push_back(id);
            }
        }
        return result;
    }
    
    /// @brief Gets atoms with a specific charge.
    std::vector<size_t> getAtomsByCharge(float minCharge, float maxCharge) const {
        std::vector<size_t> result;
        #pragma omp parallel for
        for (const auto& [id, atom] : Atoms) {
            float charge = atom.getCharge();
            if (charge >= minCharge && charge <= maxCharge) {
                #pragma omp critical
                result.push_back(id);
            }
        }
        return result;
    }

    /// @brief Calculates the axis-aligned bounding box of all atoms in the grid.
    void getBoundingBox(Vec2& minCorner, Vec2& maxCorner) const {
        TIME_FUNCTION;
        if (Positions.empty()) {
            minCorner = Vec2(0, 0);
            maxCorner = Vec2(0, 0);
            return;
        }
        
        // Initialize with first position
        auto it = Positions.begin();
        minCorner = it->second;
        maxCorner = it->second;
        
        // Find min and max coordinates
        //#pragma omp parallel for
        for (const auto& [id, pos] : Positions) {
            minCorner.x = std::min(minCorner.x, pos.x);
            minCorner.y = std::min(minCorner.y, pos.y);
            maxCorner.x = std::max(maxCorner.x, pos.x);
            maxCorner.y = std::max(maxCorner.y, pos.y);
        }
        
    }

    /// @brief Renders a specific region of the grid into a Frame object.
    /// @param minCorner Top-left coordinate of the region.
    /// @param maxCorner Bottom-right coordinate of the region.
    /// @param res The output resolution (width, height) in pixels.
    /// @param outChannels Color format (RGB, RGBA, BGR).
    /// @return A Frame object containing the rendered image.
    frame getGridRegionAsFrame(const Vec2& minCorner, const Vec2& maxCorner,
                       Vec2& res, frame::colormap outChannels = frame::colormap::RGB)  const {
        TIME_FUNCTION;
        size_t width = static_cast<int>(maxCorner.x - minCorner.x);
        size_t height = static_cast<int>(maxCorner.y - minCorner.y);
        size_t outputWidth = static_cast<int>(res.x);
        size_t outputHeight = static_cast<int>(res.y);
        float widthScale = outputWidth / width;
        float heightScale = outputHeight / height;
        
        frame outframe = frame();
        outframe.colorFormat = outChannels;

        if (width <= 0 || height <= 0) {
            width = height = 0;
            return outframe;
        }
        // if (regenpreventer) return outframe;
        // else regenpreventer = true;

        std::cout << "Rendering region: " << minCorner << " to " << maxCorner 
                << " at resolution: " << res << std::endl;
        std::cout << "Scale factors: " << widthScale << " x " << heightScale << std::endl;
        
        std::unordered_map<Vec2,Vec4f> colorBuffer;
        colorBuffer.reserve(outputHeight*outputWidth);
        std::unordered_map<Vec2,Vec4f> colorTempBuffer;
        colorTempBuffer.reserve(outputHeight * outputWidth);
        std::unordered_map<Vec2,int> countBuffer;
        countBuffer.reserve(outputHeight * outputWidth);
        std::cout << "built buffers" << std::endl;

        for (const auto& [id, pos] : Positions) {
            if (pos.x >= minCorner.x && pos.x <= maxCorner.x && 
                pos.y >= minCorner.y && pos.y <= maxCorner.y) {
                float relx = pos.x - minCorner.x;
                float rely = pos.y - minCorner.y;
                int pixx = static_cast<int>(relx * widthScale);
                int pixy = static_cast<int>(rely * heightScale);
                Vec2 pix = Vec2(pixx,pixy);
                
                colorTempBuffer[pix] += Atoms.at(id).getColor();
                countBuffer[pix]++;
            }
        }
        std::cout << std::endl << "built initial buffer" << std::endl;

        for (size_t y = 0; y < outputHeight; ++y) {
            for (size_t x = 0; x < outputWidth; ++x) {
                if (countBuffer[Vec2(x,y)] > 0) colorBuffer[Vec2(x,y)] = colorTempBuffer[Vec2(x,y)] / static_cast<float>(countBuffer[Vec2(x,y)]) * 255;
                else colorBuffer[Vec2(x,y)] = defaultBackgroundColor;
            }
        }
        std::cout << "blended second buffer" << std::endl;

        switch (outChannels) {
            case frame::colormap::RGBA: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*4, 0);
                std::cout << "outputting RGBA: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 4;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+0] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+2] = getColor.b;
                    colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGBA);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                //regenpreventer = false;
                return result;
                break;
            }
            case frame::colormap::BGR: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*3, 0);
                std::cout << "outputting BGR: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 3;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+2] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+0] = getColor.b;
                    //colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::BGR);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                //regenpreventer = false;
                return result;
                break;
            }
            case frame::colormap::RGB: 
            default: {
                std::vector<uint8_t> colorBuffer2(outputWidth*outputHeight*3, 0);
                std::cout << "outputting RGB: " << std::endl;
                for (const auto& [v2,getColor] : colorBuffer) {
                    size_t index = (v2.y * outputWidth + v2.x) * 3;
                    // std::cout << "index: " << index << std::endl;
                    colorBuffer2[index+0] = getColor.r;
                    colorBuffer2[index+1] = getColor.g;
                    colorBuffer2[index+2] = getColor.b;
                    //colorBuffer2[index+3] = getColor.a;
                }
                frame result = frame(res.x,res.y, frame::colormap::RGB);
                result.setData(colorBuffer2);
                std::cout << "returning result" << std::endl;
                //regenpreventer = false;
                return result;
                break;
            }
        }
    }

    /// @brief Renders the entire grid into a Frame. Auto-calculates bounds.
    frame getGridAsFrame(frame::colormap outchannel = frame::colormap::RGB) const {
        Vec2 min;
        Vec2 max;
        getBoundingBox(min,max);
        Vec2 res = (max + 1) - min;
        std::cout << "getting grid as frame with the following: " << min << max << res << std::endl;
        return getGridRegionAsFrame(min, max, res, outchannel);
    }

    /// @brief Removes an atom from the grid entirely.
    size_t removeID(size_t id) {
        Vec2 oldPosition = Positions.at(id);
        Positions.remove(id);
        Atoms.erase(id);
        unassignedIDs.push_back(id);
        spatialGrid.remove(id, oldPosition);
        return id;
    }
    
    /// @brief Updates multiple positions simultaneously.
    void bulkUpdatePositions(const std::unordered_map<size_t, Vec2>& newPositions) {
        TIME_FUNCTION;
        for (const auto& [id, newPos] : newPositions) {
            Vec2 oldPosition = Positions.at(id);
            Positions.at(id) = newPos;
            Atoms.at(id).setPosition(newPos);
            spatialGrid.update(id, oldPosition, newPos);
        }
    }
    
    void shrinkIfNeeded() {
        //TODO: garbage collector
    }
    
    //clear
    void clear() {
        Positions.clear();
        Atoms.clear();
        spatialGrid.clear();
        Atoms.rehash(0);
        defaultBackgroundColor = Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
    }

    /// @brief Rebuilds the spatial hashing grid based on the current neighbor radius.
    void optimizeSpatialGrid() {
        //std::cout << "optimizeSpatialGrid()" << std::endl;
        spatialCellSize = neighborRadius * 1.5f;
        spatialGrid = SpatialGrid(spatialCellSize);
        
        // Rebuild spatial grid
        spatialGrid.clear();
        for (const auto& [id, pos] : Positions) {
            spatialGrid.insert(id, pos);
        }
    }

    /// @brief Gets IDs of atoms within `neighborRadius` of the given ID.
    std::vector<size_t> getNeighbors(size_t id) const {
        Vec2 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, neighborRadius);
        
        std::vector<size_t> neighbors;
        float radiusSq = neighborRadius * neighborRadius;
        
        for (size_t candidateId : candidates) {
            if (candidateId != id && pos.distanceSquared(Positions.at(candidateId)) <= radiusSq) {
                neighbors.push_back(candidateId);
            }
        }
        
        return neighbors;
    }
    
    /// @brief Gets IDs of atoms within a custom distance of the given ID.
    std::vector<size_t> getNeighborsRange(size_t id, float dist) const {
        Vec2 pos = Positions.at(id);
        std::vector<size_t> candidates = spatialGrid.queryRange(pos, dist);
        
        std::vector<size_t> neighbors;
        float radiusSq = dist * dist;
        
        for (size_t candidateId : candidates) {
            if (candidateId != id && 
                pos.distanceSquared(Positions.at(candidateId)) <= radiusSq) {
                neighbors.push_back(candidateId);
            }
        }
        
        return neighbors;
    }

    /// @brief Fills empty spots in the bounding box with default background atoms.
    Grid2 backfillGrid() {
        Vec2 Min;
        Vec2 Max;
        getBoundingBox(Min, Max);
        std::vector<Vec2> newPos;
        std::vector<Vec4f> newColors;
        for (size_t x = Min.x; x < Max.x; x++) {
            for (size_t y = Min.y; y < Max.y; y++) {
                Vec2 pos = Vec2(x,y);
                if (Positions.contains(pos)) continue;
                newPos.push_back(pos);
                newColors.push_back(defaultBackgroundColor);
            }
        }
        // Create hydrogen atoms for backfill
        std::vector<ElementType> elements(newPos.size(), ElementType::HYDROGEN);
        bulkAddAtoms(newPos, newColors, elements);
        return *this;
    }
    
    /// @brief Gets statistical information about the atomic system.
    void getStatistics(size_t& totalAtoms, 
                      size_t& totalProtons,
                      size_t& totalNeutrons,
                      size_t& totalElectrons,
                      float& totalCharge,
                      float& totalMass) const {
        totalAtoms = Atoms.size();
        totalProtons = 0;
        totalNeutrons = 0;
        totalElectrons = 0;
        totalCharge = 0.0f;
        totalMass = 0.0f;
        
        for (const auto& [id, atom] : Atoms) {
            totalProtons += atom.getProtons();
            totalNeutrons += atom.getNeutrons();
            totalElectrons += atom.getElectrons();
            totalCharge += atom.getCharge();
            totalMass += atom.getMass();
        }
    }
};

#endif