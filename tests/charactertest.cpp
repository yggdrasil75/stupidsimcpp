#include <iostream>
#include <string>

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

    std::cout << "Building character: height=" << height
              << " detail=" << detail
              << " -> " << outfile << std::endl;

    float m = height * 0.6f + 1.0f;
    Vec3 minBound(-m, -m, -1.0f);
    Vec3 maxBound( m,  m, height + 1.0f);

    Grid::Octree<int> octree(minBound, maxBound, "output/characterstore", 4);
    octree.setBackgroundColor(Vec3(0.02f, 0.02f, 0.03f));
    octree.setSkylight(Vec3(0.05f, 0.05f, 0.06f));

    size_t n = Character::buildCharacter(octree, height, detail, Vec3(0.0f, 0.0f, 0.0f));
    std::cout << "Inserted " << n << " voxels across skeleton/muscle/flesh layers." << std::endl;

    octree.saveObject(Character::OID_SKELETON, outfile + ".skeleton");
    octree.saveObject(Character::OID_MUSCLE,   outfile + ".muscle");
    octree.saveObject(Character::OID_FLESH,    outfile + ".flesh");

    octree.save(outfile);

    std::cout << "Saved:\n"
              << "  full scene : " << outfile << "  (editor: Load full scene)\n"
              << "  skeleton   : " << outfile << ".skeleton  (editor: Load object)\n"
              << "  muscle     : " << outfile << ".muscle\n"
              << "  flesh      : " << outfile << ".flesh" << std::endl;
    return 0;
}