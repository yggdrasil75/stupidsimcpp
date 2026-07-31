#include <iostream>
#include <string>
#include <unordered_map>

#include "../eigen/Eigen/Dense"
#include "../util/grid/grid3eigen.hpp"
#include "../util/character.hpp"

using Vec3 = Eigen::Vector3f;

int main(int argc, char** argv) {
    std::string outfile = "./output/character.obj";
    float height = 2.0f;
    float detail = 0.01f;

    if (argc > 1) outfile = argv[1];
    if (argc > 2) height  = std::stof(argv[2]);
    if (argc > 3) detail  = std::stof(argv[3]);

    std::cout << "Building rigged character: height=" << height
              << " detail=" << detail << " -> " << outfile << std::endl;

    float m = height * 0.6f + 1.0f;
    Vec3 minBound(-m, -m, -1.0f);
    Vec3 maxBound( m,  m, height + 1.0f);

    Grid::Octree<int> octree(minBound, maxBound, "output/characterstore", 32);
    octree.setBackgroundColor(Vec3(0.02f, 0.02f, 0.03f));
    octree.setSkylight(Vec3(0.05f, 0.05f, 0.06f));

    std::unordered_map<int, std::string> boneNames;
    std::unordered_map<std::string, int> muscleVoxCount;
    size_t voxTotal = 0;
    std::array<size_t,4> layerCounts{0,0,0,0};

    Character::RigCallbacks cb;
    cb.onBone   = [&](const Character::RigBone& b){ boneNames[b.objectId] = b.name; };
    cb.onMuscle = [&](const Character::RigMuscle& mu){ muscleVoxCount[mu.name] = (int)mu.voxels.size(); };
    cb.onVoxel  = [&](int, int layer, const Vec3&){ ++voxTotal; if(layer>=0&&layer<4) ++layerCounts[layer]; };

    Character::CharacterRig rig =
        Character::buildRiggedCharacter(octree, height, detail, Vec3(0,0,0), cb);

    std::cout << "Emitted " << voxTotal << " voxels "
              << "(skeleton=" << layerCounts[1]
              << " muscle=" << layerCounts[2]
              << " flesh="  << layerCounts[3] << ")\n";
    std::cout << "Rig: " << rig.bones.size() << " bones, "
              << rig.joints.size() << " joints, "
              << rig.muscles.size() << " muscles.\n";

    size_t jbonds = Character::bindJoints(octree, rig, detail);
    std::cout << "Joint constraint bonds created: " << jbonds << "\n";

    std::cout << "\nSample addressable muscles:\n";
    for (const char* nm : {"biceps.L","quad.R","deltoid.L","abdomen","calf.L"}) {
        auto it = muscleVoxCount.find(nm);
        const Character::RigMuscle* rm = rig.findMuscle(nm);
        std::cout << "  " << nm << ": "
                  << (it==muscleVoxCount.end()?0:it->second) << " voxels";
        if (rm) std::cout << ", actuates '" << Character::jointName(rm->actuates)
                          << "' between obj " << rm->originObjId
                          << " and obj " << rm->insertObjId;
        std::cout << "\n";
    }

    std::cout << "\nSample joints (parent obj -> child obj @ pivot):\n";
    for (const auto& j : rig.joints) {
        std::cout << "  " << j.name << ": obj " << j.parentObjId
                  << " -> obj " << j.childObjId
                  << "  swing[" << j.limits.swingMin << "," << j.limits.swingMax << "]\n";
    }

    for (const auto& b : rig.bones)
        octree.saveObject(b.objectId, outfile + ".bone." + b.name);
    octree.saveObject(Character::OID_MUSCLE, outfile + ".muscle");
    octree.saveObject(Character::OID_FLESH,  outfile + ".flesh");
    octree.save(outfile);

    std::cout << "Saved:\n"
              << "  full scene : " << outfile << "  (editor: Load full scene)\n"
              << "  skeleton   : " << outfile << ".skeleton  (editor: Load object)\n"
              << "  muscle     : " << outfile << ".muscle\n"
              << "  flesh      : " << outfile << ".flesh" << std::endl;
    return 0;
}