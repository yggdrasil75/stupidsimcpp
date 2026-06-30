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
#include "../noise/pnoise2.hpp"

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
    float barkThickness = 0.4f;
    
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
    float skototropism = 0.0f;
    float aerotropism = 0.0f;
    float hydrotropism = 0.0f;
    float thermotropism = 0.0f;
    float heliotropism = 0.0f;
    ///TODO: missing tropisms: Chemotropism, Electrotropism, Exotropism, Inotropism, Magnetotropism, selenotropism
    
    v3 barkColor = v3(0.35f, 0.2f, 0.1f);

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(woodiness, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(hollowness, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(flexibility, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(barkThickness, 0.0f, 1.0f, rate, 0.08f, rng);
        
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
        mutateFloat(skototropism, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(thigmotropism, 0.0f, 1.0f, rate, 0.2f, rng);
        mutateFloat(aerotropism, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(hydrotropism, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(thermotropism, 0.0f, 1.0f, rate, 0.1f, rng);
        mutateFloat(heliotropism, 0.0f, 1.0f, rate, 0.1f, rng);
        
        mutateColor(barkColor, rate, 0.05f, rng);
    }

    static StemDNA crossover(const StemDNA& a, const StemDNA& b, float w) {
        StemDNA c;
        c.woodiness = mixFloat(a.woodiness, b.woodiness, w);
        c.hollowness = mixFloat(a.hollowness, b.hollowness, w);
        c.flexibility = mixFloat(a.flexibility, b.flexibility, w);
        c.barkThickness = mixFloat(a.barkThickness, b.barkThickness, w);
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
        c.skototropism = mixFloat(a.skototropism, b.skototropism, w);
        c.thigmotropism = mixFloat(a.thigmotropism, b.thigmotropism, w);
        c.aerotropism = mixFloat(a.aerotropism, b.aerotropism, w);
        c.hydrotropism = mixFloat(a.hydrotropism, b.hydrotropism, w);
        c.thermotropism = mixFloat(a.thermotropism, b.thermotropism, w);
        c.heliotropism = mixFloat(a.heliotropism, b.heliotropism, w);
        
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
    
    float fungalSaprotrophy = 0.0f;

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(flowerProbability, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(flowerBloomTemp, 0.0f, 40.0f, rate, 2.0f, rng);
        mutateColor(flowerColor, rate, 0.1f, rng);
        mutateFloat(carnivorousTrapProbability, 0.0f, 1.0f, rate, 0.02f, rng);
        mutateFloat(passiveDigestionRate, 0.0f, 5.0f, rate, 0.1f, rng);
        mutateFloat(pitcherFormationDrive, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(fungalSaprotrophy, 0.0f, 1.0f, rate, 0.05f, rng);
    }

    static SpecialDNA crossover(const SpecialDNA& a, const SpecialDNA& b, float w) {
        SpecialDNA c;
        c.flowerProbability = mixFloat(a.flowerProbability, b.flowerProbability, w);
        c.flowerBloomTemp = mixFloat(a.flowerBloomTemp, b.flowerBloomTemp, w);
        c.flowerColor = mixColor(a.flowerColor, b.flowerColor, w);
        c.carnivorousTrapProbability = mixFloat(a.carnivorousTrapProbability, b.carnivorousTrapProbability, w);
        c.passiveDigestionRate = mixFloat(a.passiveDigestionRate, b.passiveDigestionRate, w);
        c.pitcherFormationDrive = mixFloat(a.pitcherFormationDrive, b.pitcherFormationDrive, w);
        c.fungalSaprotrophy = mixFloat(a.fungalSaprotrophy, b.fungalSaprotrophy, w);
        return c;
    }
};
enum class GrowthForm : uint8_t { HERB, SHRUB, TREE, VINE, AQUATIC, FUNGUS };
struct GrowthFormDNA {
    float woodiness = 0.2f;
    float caulescence = 0.3f;
    float secondaryGrowth = 0.0f;
    float floweringInvestment = 0.1f;
    float fruitInvestment = 0.0f;
    float leafLobing = 0.0f;
    float succulence = 0.0f;

    void mutate(float rate, std::mt19937& rng) {
        mutateFloat(woodiness, 0.0f, 1.0f, rate, 0.08f, rng);
        mutateFloat(caulescence, 0.0f, 1.0f, rate, 0.08f, rng);
        mutateFloat(secondaryGrowth, 0.0f, 1.0f, rate, 0.06f, rng);
        mutateFloat(floweringInvestment, 0.0f, 1.0f, rate, 0.06f, rng);
        mutateFloat(fruitInvestment, 0.0f, 1.0f, rate, 0.05f, rng);
        mutateFloat(leafLobing, 0.0f, 1.0f, rate, 0.08f, rng);
        mutateFloat(succulence, 0.0f, 1.0f, rate, 0.06f, rng);
    }

    static GrowthFormDNA crossover(const GrowthFormDNA& a, const GrowthFormDNA& b, float w) {
        GrowthFormDNA c;
        c.woodiness = mixFloat(a.woodiness, b.woodiness, w);
        c.caulescence = mixFloat(a.caulescence, b.caulescence, w);
        c.secondaryGrowth = mixFloat(a.secondaryGrowth, b.secondaryGrowth, w);
        c.floweringInvestment = mixFloat(a.floweringInvestment, b.floweringInvestment, w);
        c.fruitInvestment = mixFloat(a.fruitInvestment, b.fruitInvestment, w);
        c.leafLobing = mixFloat(a.leafLobing, b.leafLobing, w);
        c.succulence = mixFloat(a.succulence, b.succulence, w);
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
    GrowthFormDNA form;
    GrowthForm classifyForm() const {
        if (photosynthesisEfficiency < 0.05f || special.fungalSaprotrophy > 0.5f) return GrowthForm::FUNGUS;
        if (stem.aerotropism > 0.6f) return GrowthForm::AQUATIC;
        if (stem.thigmotropism > 0.6f && stem.maxGirth < 1.0f) return GrowthForm::VINE;
        if (form.woodiness > 0.6f && stem.maxHeight > 8.0f && form.caulescence > 0.4f) return GrowthForm::TREE;
        if (form.woodiness > 0.35f && stem.maxBranchDepth >= 2) return GrowthForm::SHRUB;
        return GrowthForm::HERB;
    }

    static const char* formName(GrowthForm f) {
        switch (f) {
            case GrowthForm::HERB:    return "Herb";
            case GrowthForm::SHRUB:   return "Shrub";
            case GrowthForm::TREE:    return "Tree";
            case GrowthForm::VINE:    return "Vine";
            case GrowthForm::AQUATIC: return "Aquatic";
            case GrowthForm::FUNGUS:  return "Fungus";
        }
        return "Unknown";
    }

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
        form.mutate(rate, rng);

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
        child->form = GrowthFormDNA::crossover(mother.form, father.form, w);

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
        out.write(reinterpret_cast<const char*>(&form), sizeof(GrowthFormDNA));

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
        in.read(reinterpret_cast<char*>(&form), sizeof(GrowthFormDNA));

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
    int plantId = 0;
    
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
        out.write(reinterpret_cast<const char*>(&plantId), sizeof(plantId));
        
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
        in.read(reinterpret_cast<char*>(&plantId), sizeof(plantId));
        
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
    float plantVoxelSize = 0.1f;
    float stemFacetSize = 0.03f;
    float waterVoxelSize = 0.08f;
    float groundSize = 20.0f;
    float waterLevel = 0.0f;
    
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
    float moonSize = 10.0f;
    v3 moonColor = v3(0.7f, 0.75f, 0.9f);
    float precipRate = 200.0f;
    
    float growthSpeedMultiplier = 1.0f; 

    float baseMutationRate = 0.05f;  
    float somaticMutationRate = 0.005f;
    float cosmicRayIntensity = 0.0f; 
    float geneticCompatibilityThreshold = 1.2f; 

    int   terrainSeed       = 42;
    float terrainScale      = 0.015f;
    float terrainAmplitude  = 2.0f;
    int   terrainOctaves    = 5;
    float terrainPersistence= 0.5f;
    float terrainLacunarity = 2.0f;
    float ridgeStrength     = 0.25f;
    float ridgeOffset       = 0.1f;
    float ridgeOffsetSafe() const { return ridgeOffset <= 0.0f ? 1.0f : ridgeOffset; }
    float rainfallScale     = 0.01f;
};

struct PlantState {
    float energy = 100.0f;
    float water = 100.0f;
    int leafCount = 0;
    int rootCount = 0;
};

class PlantSim {
public:
    enum class WeatherState { CLEAR, RAIN, SNOW };

    Grid::Octree<std::shared_ptr<PlantsimParticle>, uint8_t> grid = Grid::Octree<std::shared_ptr<PlantsimParticle>, uint8_t>(Vector3f::Constant(-1024), Vector3f::Constant(1024), "output/plant", 16, 16);
    PlantConfig config;
    std::mt19937 rng;
    std::uniform_real_distribution<float> chanceDist{0.0f, 1.0f};

    int nextPlantId = 1;
    std::unordered_map<int, std::shared_ptr<PlantState>> plantStates;

    std::vector<v3> activeRoots;
    std::vector<v3> activeLeaves;
    std::vector<v3> activeMeristems; 
    std::vector<v3> activeFlowers;
    std::vector<v3> seeds;
    std::vector<v3> activeRainDrops;
    std::vector<v3> pooledWater;

    WeatherState currentWeather = WeatherState::CLEAR;
    float weatherTimer = 0.0f;
    float atmosphericMoisture = 0.0f;
    float currentTemperature = 20.0f;
    float extendedHeatTimer = 0.0f;
    float rainAccumulator = 0.0f;

    PNoise2 terrainNoise;
    PNoise2 rainNoise;
    static constexpr int WATER_OBJ = 3;

    PlantSim(PlantConfig cfg = PlantConfig()) : config(cfg), terrainNoise((unsigned int)cfg.terrainSeed),
          rainNoise((unsigned int)(cfg.terrainSeed ^ 0x9E3779B9u)) {
        std::random_device rd;
        rng = std::mt19937(rd());
        grid.setPhysicsSmoothingRadius(config.waterVoxelSize * 2.5f);
        grid.setPhysicsRestDensity(1000.0f);
        grid.setPhysicsUseGravityPoint(false);
        grid.setPhysicsGravity(Eigen::Vector3f(0.0f, -9.81f, 0.0f));
    }
    
    bool spawnWaterVoxel(const v3& pos, const v3& vel = v3(0,0,0)) {
        auto w = std::make_shared<WaterParticle>();
        w->size = config.waterVoxelSize;
        w->velocity = vel;
        float mass = config.waterVoxelSize * config.waterVoxelSize * config.waterVoxelSize * 1000.0f;
        bool ok = grid.set(w, pos, true, v3(0.1f, 0.4f, 0.8f), config.waterVoxelSize,
                           true, WATER_OBJ, 0.0f, 0.0f, 0.0f, 0.6f, 1.33f,
                           v3(0.2f, 0.1f, 0.0f), Grid::BodyType::FLUID, mass);
        return ok;
    }

    float getTerrainHeight(float x, float z) const {
        Eigen::Vector2f p(x * config.terrainScale, z * config.terrainScale);
        PNoise2& gen = const_cast<PNoise2&>(terrainNoise);

        float base = gen.fractalNoise(p, config.terrainOctaves, config.terrainPersistence, config.terrainLacunarity);
        float ridges = gen.ridgedNoise(p, config.terrainOctaves, config.ridgeOffsetSafe());
        ridges = ridges * 2.0f - 1.0f;

        float h = mixFloat(base, ridges, config.ridgeStrength) * config.terrainAmplitude;
        int surfaceYIndex = static_cast<int>(std::round(h / config.voxelSize));
        return surfaceYIndex * config.voxelSize;
    }

    float getRainfallDensity(float x, float z) const {
        Eigen::Vector2f p(x * config.rainfallScale, z * config.rainfallScale);
        PNoise2& gen = const_cast<PNoise2&>(rainNoise);
        return gen.normalizedNoise(p);
    }
    
    v3 getCelestialDir(float timeOffset) const {
        float t = std::fmod(config.timeOfDay + timeOffset, 1.0f);
        if (t < 0.0f) t += 1.0f;
        float hAngle = (t - 0.5f) * 2.0f * M_PI;
        float dec = -config.axialTilt * M_PI / 180.0f * std::cos(2.0f * M_PI * config.season);
        float lat = config.latitude * M_PI / 180.0f;

        float sinAlt = std::sin(lat) * std::sin(dec) + std::cos(lat) * std::cos(dec) * std::cos(hAngle);
        float alt = std::asin(std::clamp(sinAlt, -1.0f, 1.0f));
        
        float denom = std::cos(alt) * std::cos(lat);
        if (std::abs(denom) < 1e-5f) denom = 1e-5f;
        
        float cosAz = (std::sin(dec) - std::sin(alt) * std::sin(lat)) / denom;
        float az = std::acos(std::clamp(cosAz, -1.0f, 1.0f));
        
        if (hAngle > 0.0f) {
            az = 2.0f * M_PI - az;
        }

        return v3(std::cos(alt) * std::sin(az), std::sin(alt), std::cos(alt) * std::cos(az));
    }
    
    void updateSkyBodies() {
        v3 sunDir = getCelestialDir(0.0f);
        v3 moonDir = getCelestialDir(0.5f);
        
        grid.addSkyBody(1, sunDir, config.sunSize * M_PI / 180.0f, 
                        static_cast<uint8_t>(std::clamp(config.sunColor.x() * 255.0f, 0.0f, 255.0f)), 
                        static_cast<uint8_t>(std::clamp(config.sunColor.y() * 255.0f, 0.0f, 255.0f)), 
                        static_cast<uint8_t>(std::clamp(config.sunColor.z() * 255.0f, 0.0f, 255.0f)), 255);
                        
        grid.addSkyBody(2, moonDir, config.moonSize * M_PI / 180.0f, 
                        static_cast<uint8_t>(std::clamp(config.moonColor.x() * 255.0f, 0.0f, 255.0f)), 
                        static_cast<uint8_t>(std::clamp(config.moonColor.y() * 255.0f, 0.0f, 255.0f)), 
                        static_cast<uint8_t>(std::clamp(config.moonColor.z() * 255.0f, 0.0f, 255.0f)), 100);
    }

    void initWorld(bool spawnDefaultSeed = true) {
        grid.clear();
        plantStates.clear();
        nextPlantId = 1;
        
        activeRoots.clear();
        activeLeaves.clear();
        activeMeristems.clear();
        activeFlowers.clear();
        seeds.clear();
        activeRainDrops.clear();
        pooledWater.clear();
        
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
                float surfaceY = getTerrainHeight(x, z);
                int depth = 10 + static_cast<int>(getRainfallDensity(x, z) * 8.0f);

                for (int d = 0; d < depth; ++d) {
                    float y = surfaceY - vSize * d - vSize;
                    
                    auto dirt = std::make_shared<DirtParticle>();
                    dirt->hydration = 100.0f + d * 10.0f; 
                    grid.queuedset(dirt, v3(x, y, z), true, v3(0.4f - d*0.015f, 0.25f - d*0.01f, 0.1f), vSize, true, 0);
                }

                float wvs = config.waterVoxelSize;
                int sub = std::max(1, (int)std::round(vSize / wvs));
                for (float y = surfaceY; y < config.waterLevel; y += vSize) {
                    for (int ix = 0; ix < sub; ++ix) {
                        for (int iz = 0; iz < sub; ++iz) {
                            for (int iy = 0; iy < sub; ++iy) {
                                v3 wpos(x - vSize*0.5f + (ix+0.5f)*wvs,
                                        y + vSize - vSize*0.5f + (iy+0.5f)*wvs,
                                        z - vSize*0.5f + (iz+0.5f)*wvs);
                                spawnWaterVoxel(wpos);
                            }
                        }
                    }
                }
            }
        }
        
        auto air = std::make_shared<AirParticle>();
        grid.queuedset(air, v3(0.0f, std::max(0.0f, config.waterLevel) + vSize / 2.0f, 0.0f), false, v3(0.0f, 0.0f, 0.0f), vSize, true, 2);

        if (spawnDefaultSeed) {
            auto seedDNA = std::make_shared<PlantDNA>();
            v3 startPos = v3(0.0f, getTerrainHeight(0.0f, 0.0f), 0.0f);
            if (startPos.y() < config.waterLevel) startPos.y() = config.waterLevel;
            
            int pId = nextPlantId++;
            auto state = std::make_shared<PlantState>();
            state->energy = 50.0f;
            state->water = 50.0f;
            plantStates[pId] = state;

            auto seed = std::make_shared<PlantParticle>(PlantPart::SEED, seedDNA, startPos, v3(0.0f, 1.0f, 0.0f), 0);
            seed->plantId = pId;

            grid.queuedset(seed, startPos, true, v3(0.2f, 0.8f, 0.2f), config.plantVoxelSize, true, 1);
            activeMeristems.push_back(startPos);
        }
        
        updateSkyBodies();
        grid.save("output/plants.yggs");
    }

    float stemRadius(const PlantDNA& dna) const {
        return config.plantVoxelSize * std::clamp(0.3f + dna.stem.maxGirth * 0.8f, 0.18f, 16.0f);
    }

    static void perpBasis(const v3& axis, v3& u, v3& w) {
        v3 a = axis.normalized();
        v3 ref = (std::abs(a.y()) < 0.9f) ? v3(0,1,0) : v3(1,0,0);
        u = a.cross(ref).normalized();
        w = a.cross(u).normalized();
    }

    void buildDiscRing(const v3& center, const v3& u, const v3& w,
                       float radius, const PlantDNA& dna, int plantId, bool isRoot) {
        const float facet = config.stemFacetSize;
        int barkRings = isRoot ? 0 : (int)std::ceil(dna.stem.barkThickness * 3.0f);
        barkRings = std::clamp(barkRings, 0, 4);
        int totalRings = std::max(1, barkRings + 1); // + one wood ring

        v3 woodCol = isRoot ? v3(0.45f, 0.34f, 0.26f) : v3(0.55f, 0.42f, 0.30f);
        v3 paper(0.85f, 0.82f, 0.75f);
        v3 barkCol = mixColor(paper, dna.stem.barkColor,
                              std::clamp(dna.stem.barkThickness * 1.5f, 0.0f, 1.0f));

        for (int ring = 0; ring < totalRings; ++ring) {
            float r = radius - ring * facet;
            if (r < facet * 0.5f) break;
            int N = std::clamp((int)std::round(2.0f * (float)M_PI * r / facet), 6, 1024);
            bool isBark = (ring < barkRings);
            // Outer bark rings darken slightly to read as furrows.
            v3 col = isBark
                ? mixColor(barkCol, barkCol * 0.7f, (float)ring / std::max(1, barkRings))
                : woodCol;
            for (int s = 0; s < N; ++s) {
                float ang = (float)s / N * 2.0f * (float)M_PI;
                v3 vp = center + (u * std::cos(ang) + w * std::sin(ang)) * r;
                if (grid.find(vp, facet * 0.5f)) continue;
                auto bv = std::make_shared<PlantParticle>(
                    isRoot ? PlantPart::ROOT : PlantPart::STEM, nullptr, center, w, 0);
                bv->plantId = plantId;
                bv->isMature = true; // wall voxels are inert; never growth tips
                grid.set(bv, vp, true, col, facet, true, 1);
            }
        }
    }

    void buildCylinderSegment(const v3& from, const v3& to, float radius,
                              const PlantDNA& dna, int plantId, bool isRoot) {
        v3 axis = to - from;
        float len = axis.norm();
        if (len < 1e-5f) { 
            v3 u, w; perpBasis(v3(0,1,0), u, w);
            buildDiscRing(from, u, w, radius, dna, plantId, isRoot);
            return;
        }
        axis /= len;
        v3 u, w; perpBasis(axis, u, w);
        int rings = std::max(1, (int)std::round(len / config.stemFacetSize));
        for (int i = 0; i <= rings; ++i) {
            float t = (float)i / rings;
            buildDiscRing(from + (to - from) * t, u, w, radius, dna, plantId, isRoot);
        }
    }
    
    void buildBlob(const v3& center, float radius, const v3& color,
                   int plantId, PlantPart part) {
        const float facet = config.stemFacetSize;
        int n = std::max(1, (int)std::round(radius / facet));
        float r2 = radius * radius;
        for (int ix = -n; ix <= n; ++ix)
        for (int iy = -n; iy <= n; ++iy)
        for (int iz = -n; iz <= n; ++iz) {
            v3 off(ix * facet, iy * facet, iz * facet);
            if (off.squaredNorm() > r2) continue;        // carve a sphere
            v3 vp = center + off;
            if (grid.find(vp, facet * 0.5f)) continue;
            auto bv = std::make_shared<PlantParticle>(part, nullptr, center, v3(0,1,0), 0);
            bv->plantId = plantId;
            bv->isMature = true;
            grid.set(bv, vp, true, color, facet, true, 1);
        }
    }

    void buildLeafBlade(const v3& base, const v3& dir, const PlantDNA& dna,
                        int plantId, float lobing) {
        const float facet = config.stemFacetSize;
        float halfW = facet * std::clamp(2.0f * dna.leaf.widthMultiplier, 1.0f, 12.0f);
        float halfL = facet * std::clamp(3.0f * dna.leaf.lengthMultiplier, 1.5f, 18.0f);

        v3 along = dir.normalized();          // leaf points away from the stem
        v3 u, w; perpBasis(along, u, w);      // blade spans (along, u); w = normal
        int nL = std::max(1, (int)std::round(halfL / facet));
        int nW = std::max(1, (int)std::round(halfW / facet));
        int layers = (dna.leaf.thickness > 0.5f) ? 2 : 1;

        v3 col = mixColor(dna.leaf.color, dna.leaf.color * 0.7f, lobing);

        for (int il = 0; il <= nL; ++il) {
            float ly = (float)il / nL;
            float widthProfile = std::sqrt(std::max(0.0f, 1.0f - (ly - 0.4f) * (ly - 0.4f) / 0.36f));
            for (int iw = -nW; iw <= nW; ++iw) {
                float wx = (float)iw / std::max(1, nW);
                if (std::abs(wx) > widthProfile) continue;
                if (lobing > 0.05f) {
                    float lobe = 0.5f + 0.5f * std::cos(ly * (float)M_PI * (3.0f + lobing * 6.0f));
                    if (std::abs(wx) > widthProfile * (1.0f - lobing * lobe)) continue;
                }
                for (int k = 0; k < layers; ++k) {
                    v3 vp = base + along * (ly * halfL) + u * (wx * halfW) + w * (k * facet);
                    if (grid.find(vp, facet * 0.5f)) continue;
                    auto lv = std::make_shared<PlantParticle>(PlantPart::LEAF, nullptr, base, along, 0);
                    lv->plantId = plantId;
                    lv->isMature = true;
                    grid.set(lv, vp, true, col, facet, true, 1);
                }
            }
        }
    }
    
    size_t getActiveWaterVoxelCount() {
        using GridT = std::decay_t<decltype(grid)>;
        std::vector<std::shared_ptr<GridT::NodeData>> nodes;
        grid.collectNodesByObjectId(WATER_OBJ, nodes);
        return nodes.size();
    }

    void update(float dt) {
        simulateEnvironment(dt);
        int substeps = 4;
        float subDt = dt / substeps;
        for (int i = 0; i < substeps; ++i) {
            grid.stepPhysics(subDt);
        }
        processSoilAbsorption(dt);
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
            weatherTimer = 600.0f;
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
                if (chanceDist(rng) > getRainfallDensity(rx, rz)) continue;
                v3 spawnPos(rx, 50.0f, rz);
                
                if (!grid.find(spawnPos, config.waterVoxelSize)) {
                    spawnWaterVoxel(spawnPos, v3(0, -10.0f, 0));
                }
            }
        }

        v3 sunDir = getCelestialDir(0.0f);
        v3 moonDir = getCelestialDir(0.5f);
        v3 daySky(0.53f, 0.81f, 0.92f);
        v3 sunsetSky(0.9f, 0.45f, 0.25f);
        v3 nightSky(0.02f, 0.03f, 0.08f);

        v3 currentSky;
        if (sunDir.y() > 0.1f) {
            currentSky = daySky;
        } else if (sunDir.y() > 0.0f) {
            float t = sunDir.y() / 0.1f;
            currentSky = sunsetSky + (daySky - sunsetSky) * t;
        } else if (sunDir.y() > -0.1f) {
            float t = (sunDir.y() + 0.1f) / 0.1f;
            currentSky = nightSky + (sunsetSky - nightSky) * t;
        } else {
            currentSky = nightSky;
        }

        grid.setBackgroundColor(currentSky);
        grid.setSkylight(currentSky * 0.4f);
        grid.moveSkyBody(1, sunDir);
        grid.moveSkyBody(2, moonDir);
        v3 activeLightDir = sunDir;
        float activeIntensity = 0.0f;

        if (sunDir.y() > 0.0f) {
            activeLightDir = sunDir;
            activeIntensity = config.sunIntensity * sunDir.y();
        } else if (moonDir.y() > 0.0f) {
            activeLightDir = moonDir;
            activeIntensity = config.sunIntensity * 0.02f * moonDir.y();
        }

        if (activeIntensity > 0.0f) {
            std::uniform_real_distribution<float> distOffset(-0.5f, 0.5f);
            
            // #pragma omp parallel for
            for (const v3& leafPos : activeLeaves) {
                auto leafNode = grid.find(leafPos);
                if (!leafNode) continue;
                
                auto p = std::static_pointer_cast<PlantParticle>(leafNode->data);
                if (!p) continue;
                
                int raysToCast = std::max(1, static_cast<int>(p->dna->leaf.widthMultiplier * p->dna->leaf.lengthMultiplier * 2.0f));
                float hitsReachingLight = 0.0f;
                
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
                    
                    bool occluded = grid.raycast(rayOrigin, activeLightDir, config.sunDistance, hitInfo, leafNode);
                    
                    if (!occluded || (hitInfo.node && hitInfo.node->data->pt == ParticleType::WATER)) {
                        hitsReachingLight += 1.0f;
                    }
                }
                
                if (hitsReachingLight > 0.0f) {
                    float receivedEnergy = (activeIntensity * dt * hitsReachingLight) / static_cast<float>(raysToCast);
                    p->energy += receivedEnergy * 100;
                    
                    auto it = plantStates.find(p->plantId);
                    if (it != plantStates.end()) {
                        it->second->energy += receivedEnergy * 100;
                    }
                }
            }
        }
    }

    void processSoilAbsorption(float dt) {
        using GridT = std::decay_t<decltype(grid)>;
        std::vector<std::shared_ptr<GridT::NodeData>> waterNodes;
        grid.collectNodesByObjectId(WATER_OBJ, waterNodes);
        if (waterNodes.empty()) return;

        for (auto& node : waterNodes) {
            if (!node || node->data->pt != ParticleType::WATER) continue;
            auto w = std::static_pointer_cast<WaterParticle>(node->data);
            v3 pos = node->position;

            if (w->velocity.norm() > 0.5f) continue;

            auto neigh = grid.findInRadius(pos, config.voxelSize * 0.9f, 0);
            for (auto& n : neigh) {
                if (n->data->pt == ParticleType::DIRT) {
                    auto d = std::static_pointer_cast<DirtParticle>(n->data);
                    if (d->hydration < 200.0f) {
                        if (chanceDist(rng) < std::clamp(dt * 2.0f, 0.0f, 1.0f)) {
                            d->hydration += 30.0f;
                            grid.remove(pos);
                        }
                    }
                    break;
                } else if (n->data->pt == ParticleType::PLANT) {
                    auto p = std::static_pointer_cast<PlantParticle>(n->data);
                    auto it = plantStates.find(p->plantId);
                    if (it != plantStates.end()) it->second->water += dt * 1.0f;
                }
            }
        }
    }

    void processMetabolism(float dt) {
        for (const v3& rootPos : activeRoots) {
            auto rootData = grid.find(rootPos);
            if (rootData) {
                auto p = std::static_pointer_cast<PlantParticle>(rootData->data);
                if (!p) continue;
                auto it = plantStates.find(p->plantId);
                if (it == plantStates.end()) continue;
                auto& state = it->second;

                auto dirts = grid.findInRadius(rootPos, config.voxelSize * 1.5f, 0);
                for (auto& dirtData : dirts) {
                    if (dirtData->data->pt == ParticleType::DIRT) {
                        auto d = std::static_pointer_cast<DirtParticle>(dirtData->data);
                        if (d && d->hydration > 0.0f) {
                            float draw = std::min(d->hydration, dt * p->dna->waterAbsorptionRate);
                            d->hydration -= draw;
                            state->water += draw;
                            
                            if (p->dna->special.fungalSaprotrophy > 0.0f) {
                                state->energy += draw * p->dna->special.fungalSaprotrophy * 5.0f;
                            }
                        }
                    } else if (dirtData->data->pt == ParticleType::WATER) {
                        state->water += dt * p->dna->waterAbsorptionRate * 2.0f;
                    }
                }
            }
        }

        for (auto& pair : plantStates) {
            auto& state = pair.second;
            float maintenanceEnergy = static_cast<float>(state->leafCount + state->rootCount) * dt * 0.1f;
            float maintenanceWater = static_cast<float>(state->leafCount) * dt * 0.2f;

            state->energy -= maintenanceEnergy;
            state->water -= maintenanceWater;

            if (state->energy < 0.0f) state->energy = 0.0f;
            if (state->water < 0.0f) state->water = 0.0f;
        }
    }

    void processGrowth(float dt) {
        std::vector<v3> newMeristems;
        std::vector<v3> toRemove;

        std::uniform_real_distribution<float> chance(0.0f, 1.0f);

        float progressRate = (dt * config.growthSpeedMultiplier) / (config.dayDuration * 2.0f);
        v3 sunDir = getCelestialDir(0.0f);

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

            auto it = plantStates.find(p->plantId);
            if (it == plantStates.end()) {
                toRemove.push_back(pos);
                continue;
            }
            auto& state = it->second;

            if (p->part != PlantPart::ROOT) {
            if (pos.y() - p->seedPos.y() >= p->dna->stem.maxHeight) {
                toRemove.push_back(pos);
                continue;
                }
            } else {
                if (p->seedPos.y() - pos.y() >= p->dna->root.maxDepth) {
                    toRemove.push_back(pos);
                    continue;
                }
            }

            float energyCost = p->dna->growthCostEnergy;
            float waterCost = p->dna->growthCostWater;

            p->internodeProgress += progressRate * p->dna->photosynthesisEfficiency;

            if (p->internodeProgress >= 1.0f) {
                p->internodeProgress = 1.0f; 
                
                if (state->energy >= energyCost && state->water >= waterCost) {
                    p->internodeProgress = 0.0f;
                    state->energy -= energyCost;
                    state->water -= waterCost;
                    v3 envForce(0, 0, 0);

                    if (p->part == PlantPart::STEM || p->part == PlantPart::LEAF) {
                        envForce += v3(0, -p->dna->stem.gravitropism, 0);
                        envForce += sunDir * p->dna->stem.phototropism;
                        envForce -= sunDir * p->dna->stem.skototropism;
                        envForce += v3(0, p->dna->stem.aerotropism, 0);
                        if (p->dna->stem.heliotropism > 0.0f && sunDir.y() > 0.0f) {
                            v3 toSun = sunDir; toSun.y() = std::abs(toSun.y());
                            envForce += toSun.normalized() * p->dna->stem.heliotropism;
                        }
                        
                        if (p->dna->stem.hydrotropism > 0.0f || p->dna->stem.thermotropism > 0.0f) {
                            auto cells = grid.findInRadius(pos, config.voxelSize * 2.5f, 0);
                            v3 wetForce(0,0,0), warmForce(0,0,0);
                            for (auto& n : cells) {
                                if (n->data->pt == ParticleType::WATER) {
                                    wetForce += (n->position - pos).normalized();
                                } else if (n->data->pt == ParticleType::DIRT) {
                                    auto d = std::static_pointer_cast<DirtParticle>(n->data);
                                    if (d->hydration > 80.0f)
                                        wetForce += (n->position - pos).normalized() * (d->hydration / 200.0f);
                                    if (d->temperature > currentTemperature)
                                        warmForce += (n->position - pos).normalized();
                                }
                            }
                            if (wetForce.norm() > 0) envForce += wetForce.normalized() * p->dna->stem.hydrotropism;
                            if (warmForce.norm() > 0) envForce += warmForce.normalized() * p->dna->stem.thermotropism;
                        }

                        if (p->dna->stem.thigmotropism > 0.0f) {
                            auto neighbors = grid.findInRadius(pos, config.voxelSize * 2.0f, 0);
                            v3 touchForce(0,0,0);
                            for(auto& n : neighbors) {
                                if (n->data->pt == ParticleType::DIRT || 
                                   (n->data->pt == ParticleType::PLANT && std::static_pointer_cast<PlantParticle>(n->data)->plantId != p->plantId)) {
                                    v3 diff = pos - n->position; 
                                    touchForce += -diff.normalized() * 0.5f; 
                                    touchForce.y() += 1.0f;
                                }
                            }
                            if (touchForce.norm() > 0) {
                                envForce += touchForce.normalized() * p->dna->stem.thigmotropism * 2.0f;
                            }
                        }
                    } else if (p->part == PlantPart::ROOT) {
                        envForce += v3(0, -p->dna->root.verticalDrive, 0);
                        v3 horiz(chance(rng)-0.5f, 0, chance(rng)-0.5f);
                        envForce += horiz * p->dna->root.horizontalDrive;
                        
                        if (p->dna->root.waterSeekStrength > 0.0f) {
                            auto neighbors = grid.findInRadius(pos, config.voxelSize * 2.5f, 0);
                            v3 waterForce(0,0,0);
                            for(auto& n : neighbors) {
                                if (n->data->pt == ParticleType::DIRT) {
                                    auto d = std::static_pointer_cast<DirtParticle>(n->data);
                                    if (d->hydration > 50.0f) {
                                        waterForce += (n->position - pos).normalized() * (d->hydration / 100.0f);
                                    }
                                } else if (n->data->pt == ParticleType::WATER) {
                                    waterForce += (n->position - pos).normalized();
                                }
                            }
                            if (waterForce.norm() > 0) {
                                envForce += waterForce.normalized() * p->dna->root.waterSeekStrength;
                            }
                        }
                    }

                    if (envForce.norm() > 0.001f) {
                        p->growthDir = (p->growthDir + envForce * 0.35f).normalized();
                    }

                    float radius = stemRadius(*p->dna);
                    float nodeSize = config.plantVoxelSize;

                    PlantPart nextPart = p->part == PlantPart::SEED ? PlantPart::STEM : p->part;
                    v3 nextPos = pos + p->growthDir * nodeSize;
                    auto newSegment = std::make_shared<PlantParticle>(nextPart, p->dna, p->seedPos, p->growthDir, p->branchDepth);
                    newSegment->plantId = p->plantId;
                    
                    bool success = grid.set(newSegment, nextPos, true,
                                            nextPart == PlantPart::ROOT ? v3(0.45f, 0.34f, 0.26f) : p->dna->stem.barkColor,
                                            config.stemFacetSize, true, 1);
                    
                    if (success) {
                        bool isRoot = (nextPart == PlantPart::ROOT);
                        float segRadius = isRoot ? radius * 0.45f : radius;
                        buildCylinderSegment(pos, nextPos, segRadius, *p->dna, p->plantId, isRoot);
                        
                        toRemove.push_back(pos);
                        newMeristems.push_back(nextPos);

                        if (nextPart == PlantPart::ROOT) {
                            activeRoots.push_back(nextPos);
                            state->rootCount += 1;
                        }

                        if (p->part != PlantPart::ROOT) {
                        if (chance(rng) < (1.0f - p->dna->stem.apicalDominance)) {
                            if (p->branchDepth < p->dna->stem.maxBranchDepth) {
                                    v3 branchDir = v3(chance(rng) - 0.5f, chance(rng) + 0.2f, chance(rng) - 0.5f).normalized();
                                v3 branchPos = pos + branchDir * nodeSize;
                                auto newBranch = std::make_shared<PlantParticle>(PlantPart::STEM, p->dna, p->seedPos, branchDir, p->branchDepth + 1);
                                    newBranch->plantId = p->plantId;
                                if (grid.set(newBranch, branchPos, true, p->dna->stem.barkColor, config.stemFacetSize, true, 1)) {
                                    newMeristems.push_back(branchPos);
                                }
                            }
                        }

                        if (p->dna->leaf.leafDensity > 0.0f && chance(rng) < p->dna->leaf.leafDensity) {
                            v3 leafDir = v3(chance(rng) - 0.5f, chance(rng) * 0.5f, chance(rng) - 0.5f).normalized();
                            v3 leafPos = pos + leafDir * radius;
                            auto newLeaf = std::make_shared<PlantParticle>(PlantPart::LEAF, p->dna, p->seedPos, leafDir, p->branchDepth + 1);
                            newLeaf->plantId = p->plantId;
                            float lob = std::max(p->dna->leaf.lobingDepth, p->dna->form.leafLobing);
                            float leafDim = 0.5f * (p->dna->leaf.widthMultiplier + p->dna->leaf.lengthMultiplier);
                            float leafScale = config.plantVoxelSize * std::clamp(leafDim, 0.3f, 4.0f) * (1.0f + lob * 0.6f) *
                                                std::clamp(0.4f + p->dna->leaf.thickness * 2.0f, 0.4f, 2.0f);
                            v3 leafCol = mixColor(p->dna->leaf.color, p->dna->leaf.color * 0.7f, lob);
                            if (grid.set(newLeaf, leafPos, true, leafCol, config.stemFacetSize, true, 1)) {
                                buildLeafBlade(leafPos, leafDir, *p->dna, p->plantId, lob);
                                activeLeaves.push_back(leafPos);
                                    state->leafCount += 1;
                            }
                        }

                        bool inBloom = std::abs(config.season - p->dna->special.flowerBloomSeason) < 0.2f;
                        float flowerChance = std::max(p->dna->special.flowerProbability,
                                                        p->dna->form.floweringInvestment);
                        if (inBloom && flowerChance > 0.0f && chance(rng) < flowerChance * 0.3f) {
                            v3 fDir = v3(chance(rng) - 0.5f, chance(rng) * 0.5f + 0.3f, chance(rng) - 0.5f).normalized();
                            v3 fPos = pos + fDir * radius;
                            auto flower = std::make_shared<PlantParticle>(PlantPart::FLOWER, p->dna, p->seedPos, fDir, p->branchDepth + 1);
                            flower->plantId = p->plantId;
                            if (grid.set(flower, fPos, true, p->dna->special.flowerColor, config.stemFacetSize, true, 1)) {
                                buildBlob(fPos, config.stemFacetSize * 2.5f, p->dna->special.flowerColor, p->plantId, PlantPart::FLOWER);
                                activeFlowers.push_back(fPos);
                            }
                        }
                        if (p->dna->form.fruitInvestment > 0.0f &&
                            state->energy > energyCost * 4.0f &&
                            chance(rng) < p->dna->form.fruitInvestment * 0.1f) {
                            v3 frDir = v3(chance(rng) - 0.5f, -chance(rng) * 0.5f, chance(rng) - 0.5f).normalized();
                            v3 frPos = pos + frDir * radius;
                            auto fruit = std::make_shared<PlantParticle>(PlantPart::FRUIT, p->dna, p->seedPos, frDir, p->branchDepth + 1);
                            fruit->plantId = p->plantId;
                            v3 fruitColor = v3(0.8f, 0.2f, 0.15f);
                            if (grid.set(fruit, frPos, true, fruitColor, config.stemFacetSize, true, 1)) {
                                buildBlob(frPos, config.stemFacetSize * 3.0f, fruitColor, p->plantId, PlantPart::FRUIT);
                                state->energy -= energyCost * 2.0f;
                            }
                        }

                            if (pos.y() <= p->seedPos.y() + config.voxelSize * 2.0f) {
                                v3 rootDir = v3(chance(rng) - 0.5f, -1.0f, chance(rng) - 0.5f).normalized();
                                v3 rootPos = pos + rootDir * config.plantVoxelSize;
                                auto newRoot = std::make_shared<PlantParticle>(PlantPart::ROOT, p->dna, p->seedPos, rootDir, p->branchDepth + 1);
                                newRoot->plantId = p->plantId;
                                if (grid.set(newRoot, rootPos, true, v3(0.45f, 0.34f, 0.26f), config.stemFacetSize, true, 1)) {
                                    activeRoots.push_back(rootPos);
                                    newMeristems.push_back(rootPos);
                                    state->rootCount += 1;
                                }
                            }
                        } else {
                            if (chance(rng) < 0.15f && p->branchDepth < p->dna->stem.maxBranchDepth + 2) {
                                v3 branchDir = v3(chance(rng) - 0.5f, -chance(rng) - 0.2f, chance(rng) - 0.5f).normalized();
                                v3 branchPos = pos + branchDir * config.plantVoxelSize;
                                auto newBranch = std::make_shared<PlantParticle>(PlantPart::ROOT, p->dna, p->seedPos, branchDir, p->branchDepth + 1);
                                newBranch->plantId = p->plantId;
                                if (grid.set(newBranch, branchPos, true, v3(0.45f, 0.34f, 0.26f), config.stemFacetSize, true, 1)) {
                                    newMeristems.push_back(branchPos);
                                    activeRoots.push_back(branchPos);
                                    state->rootCount += 1;
                                }
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
        if (activeFlowers.size() < 1) return;

        std::vector<v3> survivingFlowers;
        std::vector<v3> newSeedSites;

        struct FlowerRef { v3 pos; std::shared_ptr<PlantParticle> p; };
        std::vector<FlowerRef> flowers;
        flowers.reserve(activeFlowers.size());
        for (const v3& fp : activeFlowers) {
            auto node = grid.find(fp);
            if (node && node->data->pt == ParticleType::PLANT) {
                auto p = std::static_pointer_cast<PlantParticle>(node->data);
                if (p && p->part == PlantPart::FLOWER && p->dna)
                    flowers.push_back({fp, p});
            }
        }

        std::vector<bool> used(flowers.size(), false);

        for (size_t i = 0; i < flowers.size(); ++i) {
            if (used[i]) continue;
            auto& A = flowers[i];
            survivingFlowers.push_back(A.pos);

            int mate = -1;
            float bestDist = 1e9f;
            for (size_t j = 0; j < flowers.size(); ++j) {
                if (j == i || used[j]) continue;
                if (flowers[j].p->plantId == A.p->plantId) continue;
                float d = (flowers[j].pos - A.pos).norm();
                if (d < config.voxelSize * 12.0f && d < bestDist) {
                    bestDist = d; mate = (int)j;
                }
            }

            std::shared_ptr<PlantDNA> childDNA;
            if (mate >= 0 && !A.p->dna->isSterile && !flowers[mate].p->dna->isSterile) {
                childDNA = PlantDNA::crossbreed(*A.p->dna, *flowers[mate].p->dna,
                                                config.geneticCompatibilityThreshold, rng);
                if (childDNA) used[mate] = true;
            }

            if (!childDNA && chanceDist(rng) < 0.25f) {
                childDNA = std::make_shared<PlantDNA>(*A.p->dna);
                childDNA->mutate(config.baseMutationRate, rng);
            }
            if (!childDNA) continue;

            if (childDNA->calculateGeneticDistance(*A.p->dna) > config.geneticCompatibilityThreshold * 0.5f) {
                childDNA->speciesName = A.p->dna->speciesName + " sp." + std::to_string(childDNA->generation);
            }

            float r = childDNA->seedDispersalRadius;
            std::uniform_real_distribution<float> off(-r, r);
            float sx = A.pos.x() + off(rng);
            float sz = A.pos.z() + off(rng);
            if (std::abs(sx) > config.groundSize || std::abs(sz) > config.groundSize) continue;
            float sy = getTerrainHeight(sx, sz);
            if (sy < config.waterLevel) sy = config.waterLevel;
            v3 seedPos(sx, sy, sz);

            if (!grid.find(seedPos, config.voxelSize * 0.5f)) {
                int pId = nextPlantId++;
                auto st = std::make_shared<PlantState>();
                st->energy = 40.0f; st->water = 40.0f;
                plantStates[pId] = st;
                auto seed = std::make_shared<PlantParticle>(PlantPart::SEED, childDNA, seedPos, v3(0,1,0), 0);
                seed->plantId = pId;
                if (grid.set(seed, seedPos, true, v3(0.2f, 0.8f, 0.2f), config.plantVoxelSize, true, 1)) {
                    activeMeristems.push_back(seedPos);
                    seeds.push_back(seedPos);
                    newSeedSites.push_back(seedPos);
                } else {
                    plantStates.erase(pId);
                }
            }

            grid.remove(A.pos);
            survivingFlowers.pop_back();
        }

        activeFlowers = std::move(survivingFlowers);
    }

    void processDeath() {
        std::vector<int> deadIds;
        for (auto& kv : plantStates) {
            if (kv.second->energy <= 0.0f && kv.second->water <= 0.0f) {
                deadIds.push_back(kv.first);
            }
        }
        if (deadIds.empty()) return;
        std::unordered_map<int, bool> dead;
        for (int id : deadIds) dead[id] = true;

        auto decompose = [&](const v3& pos) {
            auto node = grid.find(pos);
            if (!node || node->data->pt != ParticleType::PLANT) return;
            auto p = std::static_pointer_cast<PlantParticle>(node->data);
            if (!p || !dead.count(p->plantId)) return;
            grid.remove(pos);
            auto dirt = std::make_shared<DirtParticle>();
            dirt->hydration = 60.0f;
            dirt->nitrogen += 40.0f;
            grid.set(dirt, pos, true, v3(0.3f, 0.2f, 0.12f), config.voxelSize, true, 0);
        };

        auto prune = [&](std::vector<v3>& list) {
            std::vector<v3> keep;
            keep.reserve(list.size());
            for (const v3& pos : list) {
                auto node = grid.find(pos);
                if (node && node->data->pt == ParticleType::PLANT) {
                    auto p = std::static_pointer_cast<PlantParticle>(node->data);
                    if (p && dead.count(p->plantId)) { decompose(pos); continue; }
                }
                keep.push_back(pos);
            }
            list = std::move(keep);
        };

        prune(activeMeristems);
        prune(activeLeaves);
        prune(activeRoots);
        prune(activeFlowers);
        prune(seeds);

        for (int id : deadIds) plantStates.erase(id);
    }
};

#endif