#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <unordered_map>

#include "../eigen/Eigen/Dense"
#include "../util/grid/grid3eigen.hpp"
#include "../util/grid/grid3physics.cpp"
#include "../util/character.hpp"

using Vec3 = Eigen::Vector3f;
using Octree = Grid::Octree<int>;
using NodePtr = std::shared_ptr<Grid::NodeData_<int>>;

int main(int argc, char** argv) {
    float height = 2.0f, detail = 0.035f, jointStrength = 4000.0f;
    int steps = 180;
    float jointStiffness = 600.0f;
    if (argc > 1) jointStrength = std::stof(argv[1]);
    if (argc > 2) steps = std::stoi(argv[2]);
    if (argc > 3) detail = std::stof(argv[3]);
    if (argc > 4) jointStiffness = std::stof(argv[4]);

    Vec3 minB(-3, -3, -3), maxB(3, 3, 5);
    Octree oct(minB, maxB, "output/dropstore", 4);
    oct.setPhysicsGravity(Vec3(0.0f, 0.0f, -9.81f));
    oct.setPhysicsVelocityDamping(0.5f);

    Character::CharacterRig rig =
        Character::buildRiggedCharacter(oct, height, detail, Vec3(0,0,0));
    size_t jb = Character::bindJoints(oct, rig, detail, jointStrength, jointStiffness);

    auto all = oct.findInRadius(Vec3(0,0,height*0.5f), height*1.5f, -1);
    std::unordered_map<int, std::vector<NodePtr>> byObj;
    for (auto& n : all) byObj[n->objectId].push_back(n);

    auto centroid = [&](int objId)->Vec3{
        auto it = byObj.find(objId);
        if (it==byObj.end() || it->second.empty()) return Vec3::Zero();
        Vec3 c = Vec3::Zero();
        for (auto& n : it->second) c += n->position;
        return c / (float)it->second.size();
    };
    auto maxSpeed = [&]()->float{
        float mx=0;
        for (auto& kv:byObj)
            for (auto& n:kv.second)
                mx = std::max(mx, n->physics.velocity.norm());
        return mx;
    };

    for (auto& b : rig.bones) {
        std::string nm = b.name;
        if (nm=="pelvis.L"||nm=="pelvis.R"||nm=="lumbar") {
            for (auto& n : byObj[b.objectId]) {
                n->physics.velocity.setZero();
                n->setStatic(true);
            }
        }
    }

    std::unordered_map<std::string,Vec3> childStart;
    for (auto& j : rig.joints) {
        if (j.parentObjId < 0) continue;
        childStart[j.name] = centroid(j.childObjId);
    }

    std::cout << "jointStrength=" << jointStrength << " steps=" << steps
              << " detail=" << detail << " jointBonds=" << jb << "\n";

    float explodeAt=-1;
    for (int s=0;s<steps;s++){
        oct.multiStepPhysics(1.0f/60.0f, 4);
        if (maxSpeed()>80.0f && explodeAt<0) explodeAt=s/60.0f;
    }

    double bondStretchSum=0;
    float bondStretchMax=0;
    long jointBondCount=0;
    for (auto& kv : byObj) for (auto& n : kv.second) {
        oct.forEachBond(n, [&](uint32_t, Grid::Bond_<int>& bd, uint32_t otherId) {
            if (bd.stiffnessOverride <= 0.0f) return;
            auto o = oct.pointById(otherId);
            if (!o) return;
            float len = (o->position - n->position).norm();
            float rel = bd.restLength>1e-5f ? std::abs(len-bd.restLength)/bd.restLength : 0.0f;
            bondStretchSum += rel;
            bondStretchMax = std::max(bondStretchMax, rel);
            ++jointBondCount;
        });
    }
    float avgBondStretch = jointBondCount ? (float)(bondStretchSum/jointBondCount) : 0.0f;

    float maxTravel=0, sumTravel=0;
    int nJ=0;
    std::cout << "\nJoint            childMove\n";
    for (auto& j : rig.joints) {
        if (j.parentObjId < 0) continue;
        float travel=(centroid(j.childObjId)-childStart[j.name]).norm();
        maxTravel=std::max(maxTravel,travel);
        sumTravel+=travel;
        ++nJ;
        std::cout<<std::left<<std::setw(16)<<j.name
                 <<std::fixed<<std::setprecision(3)<<travel<<"\n";
    }

    std::cout << "\n=== VERDICT ===\n" << std::setprecision(1)
              << "joint-bond stretch: avg " << avgBondStretch*100 << "%  max "
              << bondStretchMax*100 << "%  (" << jointBondCount << " joint bonds intact)\n"
              << std::setprecision(3)
              << "limb travel:        max " << maxTravel << " m  avg "
              << (nJ?sumTravel/nJ:0) << " m\n"
              << "final max speed:    " << maxSpeed() << " m/s\n"
              << "exploded:           " << (explodeAt>=0?"YES":"no") << "\n";
    bool held  = jointBondCount>0 && bondStretchMax < 1.0f && avgBondStretch < 0.3f;
    bool moved = maxTravel > 0.02f;
    bool stable= explodeAt<0 && maxSpeed()<15.0f;
    std::cout << "result: " << (held?"HELD ":"TORE ")
              << (moved?"MOVED ":"FROZEN ") << (stable?"STABLE":"UNSTABLE");
    if (held&&moved&&stable) std::cout << "  <-- healthy articulated ragdoll";
    std::cout << "\n";
    return 0;
}