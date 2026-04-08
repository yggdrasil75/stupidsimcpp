#ifndef PLANT_HPP
#define PLANT_HPP

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include "../grid/grid3eigen.hpp"

using v3 = Eigen::Vector3f;

enum class ParticleType : uint8_t { AIR, SUN, DIRT, PLANT, WATER, SNOW };
enum class PlantPart : uint8_t { SEED, ROOT, STEM, LEAF, FLOWER, FRUIT, PITCHER };

inline float mixFloat(float a, float b, float weight) {
    return a * (1.0f - weight) + b * weight;
}

inline void mutateFloat(float& val, float min, float max, float rate, float variance, std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    if (chance(rng) < rate) {
        std::normal_distribution<float> dist(0.0f, variance);
        val = std::clamp(val + dist(rng), min, max);
    }
}

inline void mutateBool(bool& val, float rate, std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    if (chance(rng) < rate) {
        val = !val;
    }
}

inline v3 mixColor(v3 c1, v3 c2, float weight) {
    return (c1 * (1.0f - weight) + c2 * weight).cwiseMin(1.0f).cwiseMax(0.0f);
}

inline void mutateColor(v3& color, float rate, float variance, std::mt19937& rng) {
    mutateFloat(color.x(), 0.0f, 1.0f, rate, variance, rng);
    mutateFloat(color.y(), 0.0f, 1.0f, rate, variance, rng);
    mutateFloat(color.z(), 0.0f, 1.0f, rate, variance, rng);
}

struct StemDNA {
    float woodiness = 0.8f;
    float hollowness = 0.0f;
    float flexibility = 0.1f;
    
    int nodesPerWhorl = 1;
    float divergenceAngle = 137.5f;
    
    float maxHeight = 20.0f;
    float maxGirth = 3.0f;           
    int maxBranchDepth = 3;          
    float internodeLength = 1.0f;    
    float apicalDominance = 0.9f;
    float branchAngle = 1.0f;
    float branchAngleVariance = 0.2f;
    
    float gravitropism = -1.0f;
    float phototropism = 0.5f;
    float thigmotropism = 0.0f;
    ///TODO: missing tropisms: Aerotropism, Chemotropism, Electrotropism, Exotropism, HelioTropism, Hydrotropism, Inotropism, Magnetotropism, Skototropism, selenotropism, thermotropism
    
    v3 barkColor = v3(0.35f, 0.2f, 0.1f);

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(woodiness, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(hollowness, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(flexibility, 0.0f, 1.0f, rate, 0.1f, rng);
        
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);
        if (chance(rng) < rate * 0.1f) {
            std::uniform_int_distribution<int> nodeDist(-1, 1);
            nodesPerWhorl = std::clamp(nodesPerWhorl + nodeDist(rng), 1, 5);
        }
        mutateFloat(divergenceAngle, 10.0f, 180.0f, rate, 5.0f, rng);
        
        mutateFloat(maxHeight, 0.5f, 150.0f, rate, 2.0f, rng);
        mutateFloat(maxGirth, 0.5f, 15.0f, rate, 0.5f, rng);
        if (chance(rng) < rate * 0.1f) maxBranchDepth = std::clamp(maxBranchDepth + (chance(rng)>0.5f?1:-1), 0, 5);
        
        mutateFloat(internodeLength, 0.1f, 10.0f, rate, 0.2f, rng);
        mutateFloat(apicalDominance, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(branchAngle, 0.1f, 3.14f, rate, 0.1f, rng);
        mutateFloat(branchAngleVariance, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(gravitropism, -1.0f, 1.0f, rate, 0.2f, rng);
        mutateFloat(phototropism, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(thigmotropism, 0.0f, 1.0f, rate, 0.2f, rng);
        mutateColor(barkColor, rate, 0.05f, rng);
    }

    static StemDNA crossover(const StemDNA& a, const StemDNA& b, float w) {
        StemDNA c;
        c.woodiness = mixFloat(a.woodiness, b.woodiness, w);
        c.hollowness = mixFloat(a.hollowness, b.hollowness, w);
        c.flexibility = mixFloat(a.flexibility, b.flexibility, w);
        c.nodesPerWhorl = (w < 0.5f) ? a.nodesPerWhorl : b.nodesPerWhorl;
        c.divergenceAngle = mixFloat(a.divergenceAngle, b.divergenceAngle, w);
        c.maxHeight = mixFloat(a.maxHeight, b.maxHeight, w);
        c.maxGirth = mixFloat(a.maxGirth, b.maxGirth, w);
        c.maxBranchDepth = (w < 0.5f) ? a.maxBranchDepth : b.maxBranchDepth;
        c.internodeLength = mixFloat(a.internodeLength, b.internodeLength, w);
        c.apicalDominance = mixFloat(a.apicalDominance, b.apicalDominance, w);
        c.branchAngle = mixFloat(a.branchAngle, b.branchAngle, w);
        c.branchAngleVariance = mixFloat(a.branchAngleVariance, b.branchAngleVariance, w);
        c.gravitropism = mixFloat(a.gravitropism, b.gravitropism, w);
        c.phototropism = mixFloat(a.phototropism, b.phototropism, w);
        c.thigmotropism = mixFloat(a.thigmotropism, b.thigmotropism, w);
        c.barkColor = mixColor(a.barkColor, b.barkColor, w);
        return c;
    }
};

struct LeafDNA {
    float lengthMultiplier = 1.0f;
    float widthMultiplier = 1.0f;
    float thickness = 0.1f;
    float lobingDepth = 0.0f;
    
    float leafDensity = 0.8f;        
    bool isDeciduous = true;
    float leafDropTemp = 5.0f;
    v3 color = v3(0.1f, 0.6f, 0.1f);
    v3 autumnColor = v3(0.8f, 0.3f, 0.1f);
    bool evergreen = false;

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(lengthMultiplier, 0.1f, 5.0f, rate, 0.2f, rng);
        mutateFloat(widthMultiplier, 0.1f, 5.0f, rate, 0.2f, rng);
        mutateFloat(thickness, 0.01f, 1.0f, rate, 0.05f, rng);
        mutateFloat(lobingDepth, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(leafDensity, 0.1f, 1.0f, rate, 0.1f, rng);
        mutateFloat(leafDropTemp, -10.0f, 20.0f, rate, 2.0f, rng);
        mutateBool(isDeciduous, rate * 0.05f, rng);
        mutateBool(evergreen, rate * 0.05f, rng);
        mutateColor(color, rate, 0.05f, rng);
        mutateColor(autumnColor, rate, 0.05f, rng);
    }

    static LeafDNA crossover(const LeafDNA& a, const LeafDNA& b, float w) {
        LeafDNA c;
        c.lengthMultiplier = mixFloat(a.lengthMultiplier, b.lengthMultiplier, w);
        c.widthMultiplier = mixFloat(a.widthMultiplier, b.widthMultiplier, w);
        c.thickness = mixFloat(a.thickness, b.thickness, w);
        c.lobingDepth = mixFloat(a.lobingDepth, b.lobingDepth, w);
        c.leafDensity = mixFloat(a.leafDensity, b.leafDensity, w);
        c.leafDropTemp = mixFloat(a.leafDropTemp, b.leafDropTemp, w);
        c.isDeciduous = (w < 0.5f) ? a.isDeciduous : b.isDeciduous;
        c.evergreen = (w < 0.5f) ? a.evergreen : b.evergreen;
        c.color = mixColor(a.color, b.color, w);
        c.autumnColor = mixColor(a.autumnColor, b.autumnColor, w);
        return c;
    }
};

struct RootDNA {
    float verticalDrive = 0.8f;
    float horizontalDrive = 0.2f;
    float adventitiousRate = 0.0f;
    
    float maxDepth = 10.0f;
    float spreadRadius = 5.0f;
    float waterSeekStrength = 0.8f;  

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(verticalDrive, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(horizontalDrive, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(adventitiousRate, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(maxDepth, 1.0f, 50.0f, rate, 2.0f, rng);
        mutateFloat(spreadRadius, 1.0f, 30.0f, rate, 1.0f, rng);
        mutateFloat(waterSeekStrength, 0.0f, 1.0f, rate, 0.1f, rng);
    }

    static RootDNA crossover(const RootDNA& a, const RootDNA& b, float w) {
        RootDNA c;
        c.verticalDrive = mixFloat(a.verticalDrive, b.verticalDrive, w);
        c.horizontalDrive = mixFloat(a.horizontalDrive, b.horizontalDrive, w);
        c.adventitiousRate = mixFloat(a.adventitiousRate, b.adventitiousRate, w);
        c.maxDepth = mixFloat(a.maxDepth, b.maxDepth, w);
        c.spreadRadius = mixFloat(a.spreadRadius, b.spreadRadius, w);
        c.waterSeekStrength = mixFloat(a.waterSeekStrength, b.waterSeekStrength, w);
        return c;
    }
};

struct SpecialDNA {
    float flowerProbability = 0.0f;
    float flowerBloomTemp = 15.0f;
    float flowerBloomSeason = 0.25f;
    v3 flowerColor = v3(1.0f, 0.8f, 0.9f);

    float carnivorousTrapProbability = 0.0f;
    float passiveDigestionRate = 0.0f;
    float pitcherFormationDrive = 0.0f;

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(flowerProbability, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(flowerBloomTemp, 0.0f, 40.0f, rate, 2.0f, rng);
        mutateColor(flowerColor, rate, 0.1f, rng);
        mutateFloat(carnivorousTrapProbability, 0.0f, 1.0f, rate, 0.02f, rng);
        mutateFloat(passiveDigestionRate, 0.0f, 5.0f, rate, 0.1f, rng);
        mutateFloat(pitcherFormationDrive, 0.0f, 1.0f, rate, 0.05f, rng);
    }

    static SpecialDNA crossover(const SpecialDNA& a, const SpecialDNA& b, float w) {
        SpecialDNA c;
        c.flowerProbability = mixFloat(a.flowerProbability, b.flowerProbability, w);
        c.flowerBloomTemp = mixFloat(a.flowerBloomTemp, b.flowerBloomTemp, w);
        c.flowerColor = mixColor(a.flowerColor, b.flowerColor, w);
        c.carnivorousTrapProbability = mixFloat(a.carnivorousTrapProbability, b.carnivorousTrapProbability, w);
        c.passiveDigestionRate = mixFloat(a.passiveDigestionRate, b.passiveDigestionRate, w);
        c.pitcherFormationDrive = mixFloat(a.pitcherFormationDrive, b.pitcherFormationDrive, w);
        return c;
    }
};

struct PlantDNA {
    std::string speciesName = "Emergent Plant";
    int generation = 0; 
    
    std::vector<float> genomeMarker = std::vector<float>(10, 0.5f);
    bool isSterile = false;

    StemDNA stem;
    LeafDNA leaf;
    RootDNA root;
    SpecialDNA special;

    // Metabolism
    float optimalTemp = 22.0f;
    float tempTolerance = 15.0f;
    float photosynthesisEfficiency = 2.5f;
    float waterAbsorptionRate = 5.0f;
    float maintenanceCostMultiplier = 1.0f; 
    
    // Reproduction
    float seedDispersalRadius = 10.0f;
    float growthCostEnergy = 5.0f;
    float growthCostWater = 5.0f;

    float calculateGeneticDistance(const PlantDNA& other) const {
        float sumSq = 0.0f;
        for (size_t i = 0; i < genomeMarker.size(); ++i) {
            float diff = genomeMarker[i] - other.genomeMarker[i];
            sumSq += diff * diff;
        }
        return std::sqrt(sumSq);
    }

    void mutate(float rate, std::mt19937& rng) {
        generation++;
        
        for (float& marker : genomeMarker) {
            mutateFloat(marker, 0.0f, 1.0f, rate, 0.1f, rng);
        }

        stem.mutate(rate, rng);
        leaf.mutate(rate, rng);
        root.mutate(rate, rng);
        special.mutate(rate, rng);

        mutateFloat(optimalTemp, -10.0f, 45.0f, rate, 1.0f, rng);
        mutateFloat(tempTolerance, 5.0f, 30.0f, rate, 1.0f, rng);
        mutateFloat(photosynthesisEfficiency, 0.5f, 5.0f, rate, 0.1f, rng);
        mutateFloat(waterAbsorptionRate, 1.0f, 20.0f, rate, 0.5f, rng);
        mutateFloat(seedDispersalRadius, 1.0f, 100.0f, rate, 2.0f, rng);
    }

    static std::shared_ptr<PlantDNA> crossbreed(const PlantDNA& mother, const PlantDNA& father, float compatibilityThreshold, std::mt19937& rng) {
        auto child = std::make_shared<PlantDNA>();
        
        float distance = mother.calculateGeneticDistance(father);
        if (distance > compatibilityThreshold) {
            // Too genetically drifted to produce offspring
            return nullptr; 
        }

        std::uniform_real_distribution<float> weightDist(0.0f, 1.0f);
        float w = weightDist(rng);

        child->generation = std::max(mother.generation, father.generation) + 1;
        for (size_t i = 0; i < child->genomeMarker.size(); ++i) {
            child->genomeMarker[i] = mixFloat(mother.genomeMarker[i], father.genomeMarker[i], w);
        }

        if (distance > compatibilityThreshold * 0.75f) {
            std::uniform_real_distribution<float> chance(0.0f, 1.0f);
            if (chance(rng) < 0.5f) child->isSterile = true;
        }

        child->stem = StemDNA::crossover(mother.stem, father.stem, w);
        child->leaf = LeafDNA::crossover(mother.leaf, father.leaf, w);
        child->root = RootDNA::crossover(mother.root, father.root, w);
        child->special = SpecialDNA::crossover(mother.special, father.special, w);

        child->optimalTemp = mixFloat(mother.optimalTemp, father.optimalTemp, w);
        child->tempTolerance = mixFloat(mother.tempTolerance, father.tempTolerance, w);
        child->photosynthesisEfficiency = mixFloat(mother.photosynthesisEfficiency, father.photosynthesisEfficiency, w);
        child->waterAbsorptionRate = mixFloat(mother.waterAbsorptionRate, father.waterAbsorptionRate, w);
        child->seedDispersalRadius = mixFloat(mother.seedDispersalRadius, father.seedDispersalRadius, w);

        child->mutate(0.02f, rng); 
        return child;
    }
    
    void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&generation), sizeof(generation));
        out.write(reinterpret_cast<const char*>(&isSterile), sizeof(isSterile));
        size_t mSize = genomeMarker.size();
        out.write(reinterpret_cast<const char*>(&mSize), sizeof(mSize));
        out.write(reinterpret_cast<const char*>(genomeMarker.data()), mSize * sizeof(float));
        
        out.write(reinterpret_cast<const char*>(&stem), sizeof(StemDNA));
        out.write(reinterpret_cast<const char*>(&leaf), sizeof(LeafDNA));
        out.write(reinterpret_cast<const char*>(&root), sizeof(RootDNA));
        out.write(reinterpret_cast<const char*>(&special), sizeof(SpecialDNA));

        out.write(reinterpret_cast<const char*>(&optimalTemp), sizeof(optimalTemp));
        out.write(reinterpret_cast<const char*>(&tempTolerance), sizeof(tempTolerance));
        out.write(reinterpret_cast<const char*>(&photosynthesisEfficiency), sizeof(photosynthesisEfficiency));
        out.write(reinterpret_cast<const char*>(&waterAbsorptionRate), sizeof(waterAbsorptionRate));
        out.write(reinterpret_cast<const char*>(&maintenanceCostMultiplier), sizeof(maintenanceCostMultiplier));
        out.write(reinterpret_cast<const char*>(&seedDispersalRadius), sizeof(seedDispersalRadius));
        out.write(reinterpret_cast<const char*>(&growthCostEnergy), sizeof(growthCostEnergy));
        out.write(reinterpret_cast<const char*>(&growthCostWater), sizeof(growthCostWater));
    }

    void deserialize(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&generation), sizeof(generation));
        in.read(reinterpret_cast<char*>(&isSterile), sizeof(isSterile));
        size_t mSize;
        in.read(reinterpret_cast<char*>(&mSize), sizeof(mSize));
        genomeMarker.resize(mSize);
        in.read(reinterpret_cast<char*>(genomeMarker.data()), mSize * sizeof(float));
        
        in.read(reinterpret_cast<char*>(&stem), sizeof(StemDNA));
        in.read(reinterpret_cast<char*>(&leaf), sizeof(LeafDNA));
        in.read(reinterpret_cast<char*>(&root), sizeof(RootDNA));
        in.read(reinterpret_cast<char*>(&special), sizeof(SpecialDNA));

        in.read(reinterpret_cast<char*>(&optimalTemp), sizeof(optimalTemp));
        in.read(reinterpret_cast<char*>(&tempTolerance), sizeof(tempTolerance));
        in.read(reinterpret_cast<char*>(&photosynthesisEfficiency), sizeof(photosynthesisEfficiency));
        in.read(reinterpret_cast<char*>(&waterAbsorptionRate), sizeof(waterAbsorptionRate));
        in.read(reinterpret_cast<char*>(&maintenanceCostMultiplier), sizeof(maintenanceCostMultiplier));
        in.read(reinterpret_cast<char*>(&seedDispersalRadius), sizeof(seedDispersalRadius));
        in.read(reinterpret_cast<char*>(&growthCostEnergy), sizeof(growthCostEnergy));
        in.read(reinterpret_cast<char*>(&growthCostWater), sizeof(growthCostWater));
    }
};

struct PlantsimParticle {
    ParticleType pt;
    float energy;
    v3 velocity = v3(0, 0, 0);
    float timeOutOfBounds = 0.0f;
    bool anchored = false;
    float strength = 15.0f;

    PlantsimParticle(ParticleType t = ParticleType::AIR, float e = 0.0f) : pt(t), energy(e) {
    }
    virtual ~PlantsimParticle() = default;
    
    virtual void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&pt), sizeof(pt));
        out.write(reinterpret_cast<const char*>(&energy), sizeof(energy));
        out.write(reinterpret_cast<const char*>(&velocity.x()), sizeof(float) * 3);
        out.write(reinterpret_cast<const char*>(&timeOutOfBounds), sizeof(timeOutOfBounds));
        out.write(reinterpret_cast<const char*>(&anchored), sizeof(anchored));
        out.write(reinterpret_cast<const char*>(&strength), sizeof(strength));
        serializeDerived(out);
    }

    virtual void serializeDerived(std::ofstream& out) const = 0;
    virtual void deserializeDerived(std::ifstream& in) = 0;

    void deserializeBase(std::ifstream& in) {
        in.read(reinterpret_cast<char*>(&energy), sizeof(energy));
        in.read(reinterpret_cast<char*>(&velocity.x()), sizeof(float) * 3);
        in.read(reinterpret_cast<char*>(&timeOutOfBounds), sizeof(timeOutOfBounds));
        in.read(reinterpret_cast<char*>(&anchored), sizeof(anchored));
        in.read(reinterpret_cast<char*>(&strength), sizeof(strength));
    }

    static std::shared_ptr<PlantsimParticle> deserialize(std::ifstream& in);
};

struct AirParticle : public PlantsimParticle {
    float co2;
    float temperature;
    float oxygen;

    AirParticle(float c = 400.0f, float t = 20.0f) : PlantsimParticle(ParticleType::AIR, 0.0f), co2(c), temperature(t), oxygen(21.0f) {
        }
        
    void serializeDerived(std::ofstream& out) const override {
        out.write(reinterpret_cast<const char*>(&co2), sizeof(co2));
        out.write(reinterpret_cast<const char*>(&temperature), sizeof(temperature));
        out.write(reinterpret_cast<const char*>(&oxygen), sizeof(oxygen));
    }
    void deserializeDerived(std::ifstream& in) override {
        in.read(reinterpret_cast<char*>(&co2), sizeof(co2));
        in.read(reinterpret_cast<char*>(&temperature), sizeof(temperature));
        in.read(reinterpret_cast<char*>(&oxygen), sizeof(oxygen));
    }
};

struct DirtParticle : public PlantsimParticle {
    float nitrogen;
    float phosphorus;
    float potassium;
    float carbon;
    float magnesium;
    
    float hydration;
    float temperature;

    DirtParticle(float n = 100.0f, float p = 100.0f, float k = 100.0f, float c = 100.0f, float mg = 100.0f) 
        : PlantsimParticle(ParticleType::DIRT, 0.0f), nitrogen(n), phosphorus(p), potassium(k), 
          carbon(c), magnesium(mg), hydration(0.0f), temperature(20.0f) {
          }

    void serializeDerived(std::ofstream& out) const override {
        out.write(reinterpret_cast<const char*>(&nitrogen), sizeof(nitrogen));
        out.write(reinterpret_cast<const char*>(&phosphorus), sizeof(phosphorus));
        out.write(reinterpret_cast<const char*>(&potassium), sizeof(potassium));
        out.write(reinterpret_cast<const char*>(&carbon), sizeof(carbon));
        out.write(reinterpret_cast<const char*>(&magnesium), sizeof(magnesium));
        out.write(reinterpret_cast<const char*>(&hydration), sizeof(hydration));
        out.write(reinterpret_cast<const char*>(&temperature), sizeof(temperature));
    }
    void deserializeDerived(std::ifstream& in) override {
        in.read(reinterpret_cast<char*>(&nitrogen), sizeof(nitrogen));
        in.read(reinterpret_cast<char*>(&phosphorus), sizeof(phosphorus));
        in.read(reinterpret_cast<char*>(&potassium), sizeof(potassium));
        in.read(reinterpret_cast<char*>(&carbon), sizeof(carbon));
        in.read(reinterpret_cast<char*>(&magnesium), sizeof(magnesium));
        in.read(reinterpret_cast<char*>(&hydration), sizeof(hydration));
        in.read(reinterpret_cast<char*>(&temperature), sizeof(temperature));
    }
};

struct PlantParticle : public PlantsimParticle {
    PlantPart part;
    std::shared_ptr<PlantDNA> dna;
    v3 seedPos;
    v3 growthDir;
    int branchDepth;
    float internodeProgress = 0.0f;
    float currentAngle = 0.0f;
    int structuralPhase = 0;
    bool isMature = false;
    v3 baseColor;

    float age = 0.0f;
    float water = 10.0f;
    
    float nitrogen = 0.0f, phosphorus = 0.0f, potassium = 0.0f, carbon = 0.0f, magnesium = 0.0f;
    float totalBiomass = 1.0f; 

    PlantParticle(PlantPart p = PlantPart::SEED, std::shared_ptr<PlantDNA> d = nullptr, v3 sp = v3(0,0,0), v3 gd = v3(0,1,0), int bd = 0) 
        : PlantsimParticle(ParticleType::PLANT, 10.0f), part(p), dna(d), seedPos(sp), growthDir(gd), branchDepth(bd) {
        }

    void serializeDerived(std::ofstream& out) const override {
        out.write(reinterpret_cast<const char*>(&part), sizeof(part));
        out.write(reinterpret_cast<const char*>(&seedPos.x()), sizeof(float) * 3);
        out.write(reinterpret_cast<const char*>(&growthDir.x()), sizeof(float) * 3);
        out.write(reinterpret_cast<const char*>(&branchDepth), sizeof(branchDepth));
        out.write(reinterpret_cast<const char*>(&internodeProgress), sizeof(internodeProgress));
        out.write(reinterpret_cast<const char*>(&currentAngle), sizeof(currentAngle));
        out.write(reinterpret_cast<const char*>(&structuralPhase), sizeof(structuralPhase));
        out.write(reinterpret_cast<const char*>(&isMature), sizeof(isMature));
        out.write(reinterpret_cast<const char*>(&baseColor.x()), sizeof(float) * 3);
        out.write(reinterpret_cast<const char*>(&age), sizeof(age));
        out.write(reinterpret_cast<const char*>(&water), sizeof(water));
        out.write(reinterpret_cast<const char*>(&nitrogen), sizeof(nitrogen));
        out.write(reinterpret_cast<const char*>(&phosphorus), sizeof(phosphorus));
        out.write(reinterpret_cast<const char*>(&potassium), sizeof(potassium));
        out.write(reinterpret_cast<const char*>(&carbon), sizeof(carbon));
        out.write(reinterpret_cast<const char*>(&magnesium), sizeof(magnesium));
        out.write(reinterpret_cast<const char*>(&totalBiomass), sizeof(totalBiomass));
        
        bool hasDna = (dna != nullptr);
        out.write(reinterpret_cast<const char*>(&hasDna), sizeof(hasDna));
        if (hasDna) dna->serialize(out);
    }

    void deserializeDerived(std::ifstream& in) override {
        in.read(reinterpret_cast<char*>(&part), sizeof(part));
        in.read(reinterpret_cast<char*>(&seedPos.x()), sizeof(float) * 3);
        in.read(reinterpret_cast<char*>(&growthDir.x()), sizeof(float) * 3);
        in.read(reinterpret_cast<char*>(&branchDepth), sizeof(branchDepth));
        in.read(reinterpret_cast<char*>(&internodeProgress), sizeof(internodeProgress));
        in.read(reinterpret_cast<char*>(&currentAngle), sizeof(currentAngle));
        in.read(reinterpret_cast<char*>(&structuralPhase), sizeof(structuralPhase));
        in.read(reinterpret_cast<char*>(&isMature), sizeof(isMature));
        in.read(reinterpret_cast<char*>(&baseColor.x()), sizeof(float) * 3);
        in.read(reinterpret_cast<char*>(&age), sizeof(age));
        in.read(reinterpret_cast<char*>(&water), sizeof(water));
        in.read(reinterpret_cast<char*>(&nitrogen), sizeof(nitrogen));
        in.read(reinterpret_cast<char*>(&phosphorus), sizeof(phosphorus));
        in.read(reinterpret_cast<char*>(&potassium), sizeof(potassium));
        in.read(reinterpret_cast<char*>(&carbon), sizeof(carbon));
        in.read(reinterpret_cast<char*>(&magnesium), sizeof(magnesium));
        in.read(reinterpret_cast<char*>(&totalBiomass), sizeof(totalBiomass));
        
        bool hasDna;
        in.read(reinterpret_cast<char*>(&hasDna), sizeof(hasDna));
        if (hasDna) {
            dna = std::make_shared<PlantDNA>();
            dna->deserialize(in);
        }
    }
};

struct WaterParticle : public PlantsimParticle {
    int splashCount = 0;
    float size = 0.2f;

    WaterParticle() : PlantsimParticle(ParticleType::WATER, 5.0f) {
    }
    
    void serializeDerived(std::ofstream& out) const override {
        out.write(reinterpret_cast<const char*>(&splashCount), sizeof(splashCount));
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    }
    void deserializeDerived(std::ifstream& in) override {
        in.read(reinterpret_cast<char*>(&splashCount), sizeof(splashCount));
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
    }
};

struct SnowParticle : public PlantsimParticle {
    float meltProgress = 0.0f;
    SnowParticle() : PlantsimParticle(ParticleType::SNOW, 2.0f) {
    }

    void serializeDerived(std::ofstream& out) const override {
        out.write(reinterpret_cast<const char*>(&meltProgress), sizeof(meltProgress));
    }
    void deserializeDerived(std::ifstream& in) override {
        in.read(reinterpret_cast<char*>(&meltProgress), sizeof(meltProgress));
    }
};

inline std::shared_ptr<PlantsimParticle> PlantsimParticle::deserialize(std::ifstream& in) {
    ParticleType type;
    in.read(reinterpret_cast<char*>(&type), sizeof(type));

    std::shared_ptr<PlantsimParticle> particle;
    switch (type) {
        case ParticleType::AIR:
            particle = std::make_shared<AirParticle>();
            break;
        case ParticleType::DIRT:
            particle = std::make_shared<DirtParticle>();
            break;
        case ParticleType::PLANT:
            particle = std::make_shared<PlantParticle>();
            break;
        case ParticleType::WATER:
            particle = std::make_shared<WaterParticle>();
            break;
        case ParticleType::SNOW:
            particle = std::make_shared<SnowParticle>();
            break;
        default:
            return nullptr;
            break;
    }

    particle->pt = type;
    particle->deserializeBase(in);
    particle->deserializeDerived(in);
    return particle;
}

struct PlantConfig {
    float voxelSize = 0.5f;
    float groundSize = 20.0f;
    
    float dayDuration = 60.0f;
    float timeOfDay = 0.3f;
    float season = 0.0f;
    int currentDay = 0;
    int daysPerYear = 24;
    float latitude = 45.0f;
    float axialTilt = 23.5f;
    float sunDistance = 60.0f;
    float sunSize = 25.0f;
    v3 sunColor = v3(1.0f, 0.95f, 0.8f);
    float sunIntensity = 50.0f;
    float precipRate = 200.0f;
    
    float growthSpeedMultiplier = 1.0f; 

    float baseMutationRate = 0.05f;  
    float somaticMutationRate = 0.005f;
    float cosmicRayIntensity = 0.0f; 
    float geneticCompatibilityThreshold = 1.2f; 
};

class PlantSim {
public:
    enum class WeatherState { CLEAR, RAIN, SNOW };

    Octree<std::shared_ptr<PlantsimParticle>, uint8_t, "output/plants"> grid;
    PlantConfig config;
    std::mt19937 rng;

    std::vector<v3> activeRoots;
    std::vector<v3> activeLeaves;
    std::vector<v3> activeMeristems; 
    std::vector<v3> activeFlowers;
    std::vector<v3> seeds;
    std::vector<v3> activeRainDrops;

    WeatherState currentWeather = WeatherState::CLEAR;
    float weatherTimer = 0.0f;
    float atmosphericMoisture = 0.0f;
    float currentTemperature = 20.0f;
    float extendedHeatTimer = 0.0f;
    float rainAccumulator = 0.0f;

    int leafCount = 0;
    int rootCount = 0;
    float totalPlantEnergy = 0.0f;
    float totalPlantWater = 0.0f;

    PlantSim(PlantConfig cfg = PlantConfig()) : config(cfg) {
        std::random_device rd;
        rng = std::mt19937(rd());
    }

    void initWorld(bool spawnDefaultSeed = true) {
        grid.clear();
        leafCount = 0;
        rootCount = 0;
        totalPlantEnergy = 10000.0f;
        totalPlantWater = 10000.0f;
        
        activeRoots.clear();
        activeLeaves.clear();
        activeMeristems.clear();
        activeFlowers.clear();
        seeds.clear();
        activeRainDrops.clear();
        
        currentWeather = WeatherState::CLEAR;
        weatherTimer = 0.0f;
        atmosphericMoisture = 0.0f;
        currentTemperature = 20.0f;
        extendedHeatTimer = 0.0f;
        rainAccumulator = 0.0f;

        float minGround = -config.groundSize;
        float maxGround = config.groundSize;
        float vSize = config.voxelSize;

        for (float x = minGround; x <= maxGround; x += vSize) {
            for (float z = minGround; z <= maxGround; z += vSize) {
                float y = -vSize;
                auto dirt = std::make_shared<DirtParticle>();
                dirt->hydration = 200.0f;
                grid.queuedset(dirt, v3(x, y, z), true, v3(0.4f, 0.25f, 0.1f), vSize, true, 0);
            }
        }
        
        auto air = std::make_shared<AirParticle>();
        grid.queuedset(air, v3(0.0f, vSize / 2.0f, 0.0f), false, v3(0.0f, 0.0f, 0.0f), vSize, true, 2);

        if (spawnDefaultSeed) {
            auto seedDNA = std::make_shared<PlantDNA>();
            v3 startPos = v3(0.0f, 0.0f, 0.0f);
            auto seed = std::make_shared<PlantParticle>(PlantPart::SEED, seedDNA, startPos, v3(0.0f, 1.0f, 0.0f), 0);
            grid.queuedset(seed, startPos, true, v3(0.2f, 0.8f, 0.2f), vSize, true, 1);
            activeMeristems.push_back(startPos);
        }
    }

    void update(float dt) {
        simulateEnvironment(dt);
        processRain(dt);
        processMetabolism(dt);
        processGrowth(dt);
        processPollination();
        processDeath();          
    }

private:
    void simulateEnvironment(float dt) {
        config.timeOfDay += dt / config.dayDuration;
        if (config.timeOfDay >= 1.0f) {
            config.timeOfDay -= 1.0f;
            config.currentDay += 1;
            if (config.currentDay >= config.daysPerYear) {
                config.currentDay = 0;
            }
            config.season = static_cast<float>(config.currentDay) / static_cast<float>(config.daysPerYear);
        }

        atmosphericMoisture += dt * 0.5f;
        if (atmosphericMoisture > 100.0f) {
            currentWeather = WeatherState::RAIN;
            weatherTimer = 15.0f;
            atmosphericMoisture = 0.0f;
        }

        if (weatherTimer > 0.0f) {
            weatherTimer -= dt;
            if (weatherTimer <= 0.0f) {
                currentWeather = WeatherState::CLEAR;
            }
        }

        if (currentWeather == WeatherState::RAIN) {
            rainAccumulator += config.precipRate * dt;
            std::uniform_real_distribution<float> dist(-config.groundSize, config.groundSize);
            
            while (rainAccumulator >= 1.0f) {
                rainAccumulator -= 1.0f;
                float rx = dist(rng);
                float rz = dist(rng);
                v3 spawnPos(rx, 50.0f, rz);
                
                if (!grid.find(spawnPos, config.voxelSize)) {
                    auto drop = std::make_shared<WaterParticle>();
                    drop->velocity = v3(0, -10.0f, 0); 
                    drop->size = config.voxelSize * 0.2f;
                    
                    if (grid.set(drop, spawnPos, true, v3(0.4f, 0.6f, 0.9f), drop->size, true, 3, 0.0f, 0.1f, 0.0f, 0.8f, 1.33f)) {
                        activeRainDrops.push_back(spawnPos);
                    }
                }
            }
        }

        float hourAngle = (config.timeOfDay - 0.5f) * 2.0f * M_PI;
        float dec = -config.axialTilt * M_PI / 180.0f * std::cos(2.0f * M_PI * config.season);
        float lat = config.latitude * M_PI / 180.0f;

        float sinAlt = std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(hourAngle);
        float alt = std::asin(std::clamp(sinAlt, -1.0f, 1.0f));
        float cosAz = (std::sin(dec) - std::sin(alt) * std::sin(lat)) / (std::cos(alt) * std::cos(lat));
        float az = std::acos(std::clamp(cosAz, -1.0f, 1.0f));
        
        if (hourAngle > 0.0f) {
            az = 2.0f * M_PI - az;
        }

        v3 sunDir(
            std::cos(alt) * std::sin(az),
            std::sin(alt),
            std::cos(alt) * std::cos(az)
        );

        if (sunDir.y() > 0.0f) {
            float intensity = config.sunIntensity * sunDir.y();
            std::uniform_real_distribution<float> distOffset(-0.5f, 0.5f);
            
            for (const v3& leafPos : activeLeaves) {
                auto leafNode = grid.find(leafPos);
                if (!leafNode) continue;
                
                auto p = std::static_pointer_cast<PlantParticle>(leafNode->data);
                if (!p) continue;
                
                int raysToCast = std::max(1, static_cast<int>(p->dna->leaf.widthMultiplier * p->dna->leaf.lengthMultiplier * 2.0f));
                float hitsReachingSun = 0.0f;
                
                for (int r = 0; r < raysToCast; ++r) {
                    v3 offset(0, 0, 0);
                    if (r > 0) {
                        offset = v3(
                            distOffset(rng) * p->dna->leaf.widthMultiplier * config.voxelSize,
                            0.0f,
                            distOffset(rng) * p->dna->leaf.lengthMultiplier * config.voxelSize
                        );
                    }
                    
                    v3 rayOrigin = leafPos + offset;
                    
                    using GridType = std::decay_t<decltype(grid)>;
                    GridType::RayHit hitInfo;
                    
                    bool occluded = grid.raycast(rayOrigin, sunDir, config.sunDistance, hitInfo, leafNode);
                    
                    if (!occluded) {
                        hitsReachingSun += 1.0f;
                    }
                }
                
                if (hitsReachingSun > 0.0f) {
                    float receivedEnergy = (intensity * dt * hitsReachingSun) / static_cast<float>(raysToCast);
                    p->energy += receivedEnergy;
                    totalPlantEnergy += receivedEnergy;
                }
            }
        }
    }

    void processRain(float dt) {
        std::vector<v3> nextDrops;
        for (const v3& pos : activeRainDrops) {
            auto node = grid.find(pos);
            if (!node || node->data->pt != ParticleType::WATER) continue;
            
            auto drop = std::static_pointer_cast<WaterParticle>(node->data);
            drop->velocity.y() -= 9.81f * dt;
            v3 nextPos = pos + drop->velocity * dt;
            
            auto hit = grid.find(nextPos, config.voxelSize * 0.6f);
            if (hit && hit.get() != node.get() && hit->data->pt != ParticleType::WATER) {
                if (hit->data->pt == ParticleType::DIRT) {
                    auto dirt = std::static_pointer_cast<DirtParticle>(hit->data);
                    dirt->hydration += 10.0f;
                    grid.remove(pos);
                } else if (hit->data->pt == ParticleType::PLANT) {
                    grid.remove(pos);
                    totalPlantWater += 2.0f; 
                    if (drop->splashCount < 1) { 
                        std::uniform_real_distribution<float> angleDist(0, 2*M_PI);
                        for(int i=0; i<2; ++i) {
                            float a = angleDist(rng);
                            v3 splashVel(cos(a)*1.5f, 2.0f, sin(a)*1.5f);
                            v3 splashPos = pos + v3(0, config.voxelSize * 1.2f, 0); 
                            if (!grid.find(splashPos, drop->size)) {
                                auto sdrop = std::make_shared<WaterParticle>();
                                sdrop->splashCount = drop->splashCount + 1;
                                sdrop->size = drop->size * 0.7f;
                                sdrop->velocity = splashVel;
                                if(grid.set(sdrop, splashPos, true, node->color, sdrop->size, true, 3, 0.0f, 0.1f, 0.0f, 0.8f, 1.33f)) {
                                    nextDrops.push_back(splashPos);
                                }
                            }
                        }
                    }
                } else {
                    grid.remove(pos);
                }
            } else if (nextPos.y() < -config.groundSize) {
                grid.remove(pos);
            } else {
                if (grid.move(pos, nextPos)) {
                    nextDrops.push_back(nextPos);
                } else {
                    grid.remove(pos);
                }
            }
        }
        activeRainDrops = std::move(nextDrops);
    }

    void processMetabolism(float dt) {
        float maintenanceEnergy = 0.0f;
        float maintenanceWater = 0.0f;

        for (const v3& rootPos : activeRoots) {
            auto rootData = grid.find(rootPos);
            if (rootData) {
                auto dirts = grid.findInRadius(rootPos, config.voxelSize * 1.5f, 0);
                for (auto& dirtData : dirts) {
                    auto d = std::static_pointer_cast<DirtParticle>(dirtData->data);
                    if (d && d->hydration > 0.0f) {
                        float draw = std::min(d->hydration, dt * 5.0f);
                        d->hydration -= draw;
                        totalPlantWater += draw;
                    }
                }
            }
        }

        maintenanceEnergy += static_cast<float>(leafCount + rootCount) * dt * 0.1f;
        maintenanceWater += static_cast<float>(leafCount) * dt * 0.2f;

        totalPlantEnergy -= maintenanceEnergy;
        totalPlantWater -= maintenanceWater;

        if (totalPlantEnergy < 0.0f) {
            totalPlantEnergy = 0.0f;
        }
        if (totalPlantWater < 0.0f) {
            totalPlantWater = 0.0f;
        }
    }

    void processGrowth(float dt) {
        std::vector<v3> newMeristems;
        std::vector<v3> toRemove;

        std::uniform_real_distribution<float> chance(0.0f, 1.0f);

        float progressRate = (dt * config.growthSpeedMultiplier) / (config.dayDuration * 2.0f);

        for (const v3& pos : activeMeristems) {
            auto node = grid.find(pos);
            if (!node) {
                toRemove.push_back(pos);
                continue;
            }
            
            auto p = std::static_pointer_cast<PlantParticle>(node->data);
            if (!p || !p->dna) {
                toRemove.push_back(pos);
                continue;
            }

            if (pos.y() - p->seedPos.y() >= p->dna->stem.maxHeight) {
                toRemove.push_back(pos);
                continue;
            }

            float energyCost = p->dna->growthCostEnergy;
            float waterCost = p->dna->growthCostWater;

            p->internodeProgress += progressRate * p->dna->photosynthesisEfficiency;

            if (p->internodeProgress >= 1.0f) {
                p->internodeProgress = 1.0f; 
                
                if (totalPlantEnergy >= energyCost && totalPlantWater >= waterCost) {
                    p->internodeProgress = 0.0f;
                    totalPlantEnergy -= energyCost;
                    totalPlantWater -= waterCost;

                    float nodeSize = config.voxelSize;
                    if (p->dna->stem.maxGirth < 1.0f) {
                        nodeSize = config.voxelSize * std::max(0.01f, p->dna->stem.maxGirth);
                    }

                    v3 nextPos = pos + p->growthDir * nodeSize;
                    auto newStem = std::make_shared<PlantParticle>(PlantPart::STEM, p->dna, p->seedPos, p->growthDir, p->branchDepth);
                    
                    bool success = grid.set(newStem, nextPos, true, p->dna->stem.barkColor, nodeSize, true, 1);
                    
                    if (success) {
                        toRemove.push_back(pos);
                        newMeristems.push_back(nextPos);

                        if (chance(rng) < (1.0f - p->dna->stem.apicalDominance)) {
                            if (p->branchDepth < p->dna->stem.maxBranchDepth) {
                                v3 branchDir = v3(chance(rng) - 0.5f, chance(rng), chance(rng) - 0.5f).normalized();
                                v3 branchPos = pos + branchDir * nodeSize;
                                auto newBranch = std::make_shared<PlantParticle>(PlantPart::STEM, p->dna, p->seedPos, branchDir, p->branchDepth + 1);
                                if (grid.set(newBranch, branchPos, true, p->dna->stem.barkColor, nodeSize, true, 1)) {
                                    newMeristems.push_back(branchPos);
                                }
                            }
                        }

                        if (chance(rng) < p->dna->leaf.leafDensity) {
                            v3 leafDir = v3(chance(rng) - 0.5f, chance(rng) * 0.5f, chance(rng) - 0.5f).normalized();
                            v3 leafPos = pos + leafDir * nodeSize;
                            auto newLeaf = std::make_shared<PlantParticle>(PlantPart::LEAF, p->dna, p->seedPos, leafDir, p->branchDepth + 1);
                            if (grid.set(newLeaf, leafPos, true, p->dna->leaf.color, nodeSize * 1.5f, true, 1)) {
                                activeLeaves.push_back(leafPos);
                                leafCount += 1;
                            }
                        }

                        if (pos.y() <= config.voxelSize * 2.0f) {
                            v3 rootDir = v3(chance(rng) - 0.5f, -1.0f, chance(rng) - 0.5f).normalized();
                            v3 rootPos = pos + rootDir * nodeSize;
                            auto newRoot = std::make_shared<PlantParticle>(PlantPart::ROOT, p->dna, p->seedPos, rootDir, p->branchDepth + 1);
                            if (grid.set(newRoot, rootPos, true, v3(0.5f, 0.4f, 0.3f), nodeSize, true, 1)) {
                                activeRoots.push_back(rootPos);
                                rootCount += 1;
                            }
                        }
                    } else {
                        toRemove.push_back(pos);
                    }
                }
            }
        }

        for (const v3& pos : toRemove) {
            auto it = std::find(activeMeristems.begin(), activeMeristems.end(), pos);
            if (it != activeMeristems.end()) {
                activeMeristems.erase(it);
            }
        }

        for (const v3& pos : newMeristems) {
            activeMeristems.push_back(pos);
        }
    }

    void processPollination() {
        // In bloom season, find activeFlowers close to each other.
        // Attempt crossbreeding:
        // auto childDNA = PlantDNA::crossbreed(*flowerA.dna, *flowerB.dna, config.geneticCompatibilityThreshold, rng);
        // if (childDNA != nullptr) {
        //     Spawn fruit/seed voxel with childDNA.
        // } else {
        //     Maybe fallback to asexual cloning (self-pollination) if special.flowerProbability is low enough.
        // }
    }

    void processDeath() {
        // Remove dead/starved plants, convert to DirtParticle
    }
};

#endif