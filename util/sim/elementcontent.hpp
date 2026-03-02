#ifndef ELEMENTS
#define ELEMENTS

#include <array>

struct BaseElementProps {
    float density;           // kg/m^3
    float meltingPoint;      // Kelvin
    float boilingPoint;      // Kelvin
    float specificHeat;      // J/(kg*K)
    float electronegativity; // Pauling scale
};

static const std::array<BaseElementProps, 118> ELEMENT_DB = {{
    // 1: Hydrogen
    {0.08988f, 14.01f, 20.28f, 14304.0f, 2.20f},
    // 2: Helium
    {0.1785f, 0.95f, 4.22f, 5193.0f, 0.0f},  // No electronegativity, using 0
    // 3: Lithium
    {534.0f, 453.69f, 1560.0f, 3582.0f, 0.98f},
    // 4: Beryllium
    {1850.0f, 1560.0f, 2742.0f, 1825.0f, 1.57f},
    // 5: Boron
    {2340.0f, 2349.0f, 4200.0f, 1026.0f, 2.04f},
    // 6: Carbon
    {2267.0f, 4000.0f, 4300.0f, 709.0f, 2.55f},
    // 7: Nitrogen
    {1.2506f, 63.15f, 77.36f, 1040.0f, 3.04f},
    // 8: Oxygen
    {1.429f, 54.36f, 90.2f, 918.0f, 3.44f},
    // 9: Fluorine
    {1.696f, 53.53f, 85.03f, 824.0f, 3.98f},
    // 10: Neon
    {0.9002f, 24.56f, 27.07f, 1030.0f, 0.0f},  // No electronegativity
    // 11: Sodium
    {968.0f, 370.87f, 1156.0f, 1228.0f, 0.93f},
    // 12: Magnesium
    {1738.0f, 923.0f, 1363.0f, 1023.0f, 1.31f},
    // 13: Aluminium
    {2700.0f, 933.47f, 2792.0f, 897.0f, 1.61f},
    // 14: Silicon
    {2329.0f, 1687.0f, 3538.0f, 705.0f, 1.9f},
    // 15: Phosphorus
    {1823.0f, 317.3f, 550.0f, 769.0f, 2.19f},
    // 16: Sulfur
    {2070.0f, 388.36f, 717.87f, 710.0f, 2.58f},
    // 17: Chlorine
    {3.2f, 171.6f, 239.11f, 479.0f, 3.16f},
    // 18: Argon
    {1.784f, 83.8f, 87.3f, 520.0f, 0.0f},  // No electronegativity
    // 19: Potassium
    {890.0f, 336.53f, 1032.0f, 757.0f, 0.82f},
    // 20: Calcium
    {1550.0f, 1115.0f, 1757.0f, 647.0f, 1.0f},
    // 21: Scandium
    {2985.0f, 1814.0f, 3109.0f, 568.0f, 1.36f},
    // 22: Titanium
    {4506.0f, 1941.0f, 3560.0f, 523.0f, 1.54f},
    // 23: Vanadium
    {6110.0f, 2183.0f, 3680.0f, 489.0f, 1.63f},
    // 24: Chromium
    {7150.0f, 2180.0f, 2944.0f, 449.0f, 1.66f},
    // 25: Manganese
    {7210.0f, 1519.0f, 2334.0f, 479.0f, 1.55f},
    // 26: Iron
    {7874.0f, 1811.0f, 3134.0f, 449.0f, 1.83f},
    // 27: Cobalt
    {8900.0f, 1768.0f, 3200.0f, 421.0f, 1.88f},
    // 28: Nickel
    {8908.0f, 1728.0f, 3186.0f, 444.0f, 1.91f},
    // 29: Copper
    {8960.0f, 1357.77f, 2835.0f, 385.0f, 1.9f},
    // 30: Zinc
    {7140.0f, 692.88f, 1180.0f, 388.0f, 1.65f},
    // 31: Gallium
    {5910.0f, 302.9146f, 2673.0f, 371.0f, 1.81f},
    // 32: Germanium
    {5323.0f, 1211.4f, 3106.0f, 320.0f, 2.01f},
    // 33: Arsenic
    {5727.0f, 1090.0f, 887.0f, 329.0f, 2.18f},  // Sublimes at 887K
    // 34: Selenium
    {4810.0f, 453.0f, 958.0f, 321.0f, 2.55f},
    // 35: Bromine
    {3102.8f, 265.8f, 332.0f, 474.0f, 2.96f},
    // 36: Krypton
    {3.749f, 115.79f, 119.93f, 248.0f, 3.0f},
    // 37: Rubidium
    {1532.0f, 312.46f, 961.0f, 363.0f, 0.82f},
    // 38: Strontium
    {2640.0f, 1050.0f, 1655.0f, 301.0f, 0.95f},
    // 39: Yttrium
    {4472.0f, 1799.0f, 3609.0f, 298.0f, 1.22f},
    // 40: Zirconium
    {6520.0f, 2128.0f, 4682.0f, 278.0f, 1.33f},
    // 41: Niobium
    {8570.0f, 2750.0f, 5017.0f, 265.0f, 1.6f},
    // 42: Molybdenum
    {10280.0f, 2896.0f, 4912.0f, 251.0f, 2.16f},
    // 43: Technetium
    {11000.0f, 2430.0f, 4538.0f, 0.0f, 1.9f},  // No specific heat data
    // 44: Ruthenium
    {12450.0f, 2607.0f, 4423.0f, 238.0f, 2.2f},
    // 45: Rhodium
    {12410.0f, 2237.0f, 3968.0f, 243.0f, 2.28f},
    // 46: Palladium
    {12023.0f, 1828.05f, 3236.0f, 244.0f, 2.2f},
    // 47: Silver
    {10490.0f, 1234.93f, 2435.0f, 235.0f, 1.93f},
    // 48: Cadmium
    {8650.0f, 594.22f, 1040.0f, 232.0f, 1.69f},
    // 49: Indium
    {7310.0f, 429.75f, 2345.0f, 233.0f, 1.78f},
    // 50: Tin
    {7265.0f, 505.08f, 2875.0f, 228.0f, 1.96f},
    // 51: Antimony
    {6697.0f, 903.78f, 1860.0f, 207.0f, 2.05f},
    // 52: Tellurium
    {6240.0f, 722.66f, 1261.0f, 202.0f, 2.1f},
    // 53: Iodine
    {4933.0f, 386.85f, 457.4f, 214.0f, 2.66f},
    // 54: Xenon
    {5.894f, 161.4f, 165.03f, 158.0f, 2.6f},
    // 55: Caesium
    {1930.0f, 301.59f, 944.0f, 242.0f, 0.79f},
    // 56: Barium
    {3510.0f, 1000.0f, 2170.0f, 204.0f, 0.89f},
    // 57: Lanthanum
    {6162.0f, 1193.0f, 3737.0f, 195.0f, 1.1f},
    // 58: Cerium
    {6770.0f, 1068.0f, 3716.0f, 192.0f, 1.12f},
    // 59: Praseodymium
    {6770.0f, 1208.0f, 3793.0f, 193.0f, 1.13f},
    // 60: Neodymium
    {7010.0f, 1297.0f, 3347.0f, 190.0f, 1.14f},
    // 61: Promethium
    {7260.0f, 1315.0f, 3273.0f, 0.0f, 1.13f},  // No specific heat data
    // 62: Samarium
    {7520.0f, 1345.0f, 2067.0f, 197.0f, 1.17f},
    // 63: Europium
    {5244.0f, 1099.0f, 1802.0f, 182.0f, 1.2f},
    // 64: Gadolinium
    {7900.0f, 1585.0f, 3546.0f, 236.0f, 1.2f},
    // 65: Terbium
    {8230.0f, 1629.0f, 3503.0f, 182.0f, 1.2f},
    // 66: Dysprosium
    {8540.0f, 1680.0f, 2840.0f, 170.0f, 1.22f},
    // 67: Holmium
    {8790.0f, 1734.0f, 2993.0f, 165.0f, 1.23f},
    // 68: Erbium
    {9066.0f, 1802.0f, 3141.0f, 168.0f, 1.24f},
    // 69: Thulium
    {9320.0f, 1818.0f, 2223.0f, 160.0f, 1.25f},
    // 70: Ytterbium
    {6900.0f, 1097.0f, 1469.0f, 155.0f, 1.1f},
    // 71: Lutetium
    {9841.0f, 1925.0f, 3675.0f, 154.0f, 1.27f},
    // 72: Hafnium
    {13310.0f, 2506.0f, 4876.0f, 144.0f, 1.3f},
    // 73: Tantalum
    {16690.0f, 3290.0f, 5731.0f, 140.0f, 1.5f},
    // 74: Tungsten
    {19250.0f, 3695.0f, 6203.0f, 132.0f, 2.36f},
    // 75: Rhenium
    {21020.0f, 3459.0f, 5869.0f, 137.0f, 1.9f},
    // 76: Osmium
    {22590.0f, 3306.0f, 5285.0f, 130.0f, 2.2f},
    // 77: Iridium
    {22560.0f, 2719.0f, 4701.0f, 131.0f, 2.2f},
    // 78: Platinum
    {21450.0f, 2041.4f, 4098.0f, 133.0f, 2.28f},
    // 79: Gold
    {19300.0f, 1337.33f, 3129.0f, 129.0f, 2.54f},
    // 80: Mercury
    {13534.0f, 234.43f, 629.88f, 140.0f, 2.0f},
    // 81: Thallium
    {11850.0f, 577.0f, 1746.0f, 129.0f, 1.62f},
    // 82: Lead
    {11340.0f, 600.61f, 2022.0f, 129.0f, 2.33f},  // Using 4+ value
    // 83: Bismuth
    {9780.0f, 544.7f, 1837.0f, 122.0f, 2.02f},
    // 84: Polonium
    {9196.0f, 527.0f, 1235.0f, 0.0f, 2.0f},  // No specific heat data
    // 85: Astatine
    {8930.0f, 575.0f, 610.0f, 0.0f, 2.2f},  // Approx density, no specific heat
    // 86: Radon
    {9.73f, 202.0f, 211.3f, 94.0f, 2.2f},
    // 87: Francium
    {2480.0f, 281.0f, 890.0f, 0.0f, 0.79f},  // Approx values
    // 88: Radium
    {5500.0f, 973.0f, 2010.0f, 94.0f, 0.9f},
    // 89: Actinium
    {10000.0f, 1323.0f, 3471.0f, 120.0f, 1.1f},
    // 90: Thorium
    {11700.0f, 2115.0f, 5061.0f, 113.0f, 1.3f},
    // 91: Protactinium
    {15370.0f, 1841.0f, 4300.0f, 0.0f, 1.5f},  // No specific heat data
    // 92: Uranium
    {19100.0f, 1405.3f, 4404.0f, 116.0f, 1.38f},
    // 93: Neptunium
    {20450.0f, 917.0f, 4273.0f, 0.0f, 1.36f},  // No specific heat data
    // 94: Plutonium
    {19850.0f, 912.5f, 3501.0f, 0.0f, 1.28f},  // No specific heat data
    // 95: Americium
    {12000.0f, 1449.0f, 2880.0f, 0.0f, 1.13f},  // No specific heat data
    // 96: Curium
    {13510.0f, 1613.0f, 3383.0f, 0.0f, 1.28f},  // No specific heat data
    // 97: Berkelium
    {14780.0f, 1259.0f, 2900.0f, 0.0f, 1.3f},  // No specific heat data
    // 98: Californium
    {15100.0f, 1173.0f, 1743.0f, 0.0f, 1.3f},  // No specific heat data
    // 99: Einsteinium
    {8840.0f, 1133.0f, 1269.0f, 0.0f, 1.3f},  // No specific heat data
    // 100: Fermium
    {9700.0f, 1125.0f, 1800.0f, 0.0f, 1.3f},  // Estimated values
    // 101: Mendelevium
    {10300.0f, 1100.0f, 0.0f, 0.0f, 1.3f},  // Estimated
    // 102: Nobelium
    {9900.0f, 1100.0f, 0.0f, 0.0f, 1.3f},  // Estimated
    // 103: Lawrencium
    {14400.0f, 1900.0f, 0.0f, 0.0f, 1.3f},  // Estimated
    // 104: Rutherfordium
    {17000.0f, 2400.0f, 5800.0f, 0.0f, 0.0f},  // Estimated
    // 105: Dubnium
    {21600.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 106: Seaborgium
    {23500.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 107: Bohrium
    {26500.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 108: Hassium
    {28000.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 109: Meitnerium
    {27500.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 110: Darmstadtium
    {26500.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 111: Roentgenium
    {23000.0f, 0.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 112: Copernicium
    {14000.0f, 283.0f, 340.0f, 0.0f, 0.0f},  // Estimated
    // 113: Nihonium
    {16000.0f, 700.0f, 1400.0f, 0.0f, 0.0f},  // Estimated
    // 114: Flerovium
    {11400.0f, 284.0f, 0.0f, 0.0f, 0.0f},  // Estimated
    // 115: Moscovium
    {13500.0f, 700.0f, 1400.0f, 0.0f, 0.0f},  // Estimated
    // 116: Livermorium
    {12900.0f, 700.0f, 1100.0f, 0.0f, 0.0f},  // Estimated
    // 117: Tennessine
    {7200.0f, 700.0f, 883.0f, 0.0f, 0.0f},  // Estimated
    // 118: Oganesson
    {7000.0f, 325.0f, 450.0f, 0.0f, 0.0f}  // Estimated
}};

struct PointProperties {
    float weight = 0.0f;       // Total mass
    float density = 0.0f;      // Mass / Volume
    float meltingPoint = 0.0f; 
    float boilingPoint = 0.0f;
    float specificHeat = 0.0f;
    float electronegativity = 0.0f;
};

struct elementContent {
    float hydrogen      = 0.0f;
    float helium        = 0.0f;
    float lithium       = 0.0f;
    float beryllium     = 0.0f;
    float boron         = 0.0f;
    float carbon        = 0.0f;
    float nitrogen      = 0.0f;
    float oxygen        = 0.0f;
    float fluorine      = 0.0f;
    float neon          = 0.0f;
    float sodium        = 0.0f;
    float magnesium     = 0.0f;
    float aluminum      = 0.0f;
    float silicon       = 0.0f;
    float phosporus     = 0.0f;
    float sulfur        = 0.0f;
    float chlorine      = 0.0f;
    float argon         = 0.0f;
    float potassium     = 0.0f;
    float calcium       = 0.0f;
    float scandium      = 0.0f;
    float titanium      = 0.0f;
    float vanadium      = 0.0f;
    float chromium      = 0.0f;
    float manganese     = 0.0f;
    float iron          = 0.0f;
    float cobalt        = 0.0f;
    float nickel        = 0.0f;
    float copper        = 0.0f;
    float zinc          = 0.0f;
    float gallium       = 0.0f;
    float germanium     = 0.0f;
    float arsenic       = 0.0f;
    float selenium      = 0.0f;
    float bromine       = 0.0f;
    float krypton       = 0.0f;
    float rubidium      = 0.0f;
    float strontium     = 0.0f;
    float yttrium       = 0.0f;
    float zirconium     = 0.0f;
    float niobium       = 0.0f;
    float molybdenum    = 0.0f;
    float technetium    = 0.0f;
    float ruthenium     = 0.0f;
    float rhodium       = 0.0f;
    float palladium     = 0.0f;
    float silver        = 0.0f;
    float cadmium       = 0.0f;
    float indium        = 0.0f;
    float tin           = 0.0f;
    float antimony      = 0.0f;
    float tellurium     = 0.0f;
    float iodine        = 0.0f;
    float xenon         = 0.0f;
    float caesium       = 0.0f;
    float barium        = 0.0f;
    float lanthanum     = 0.0f;
    float cerium        = 0.0f;
    float praseodymium  = 0.0f;
    float neodymium     = 0.0f;
    float promethium    = 0.0f;
    float samarium      = 0.0f;
    float europium      = 0.0f;
    float gadolinium    = 0.0f;
    float terbium       = 0.0f;
    float dysprosium    = 0.0f;
    float holmium       = 0.0f;
    float erbium        = 0.0f;
    float thulium       = 0.0f;
    float ytterbium     = 0.0f;
    float lutetium      = 0.0f;
    float hafnium       = 0.0f;
    float tantalum      = 0.0f;
    float tungsten      = 0.0f;
    float rhenium       = 0.0f;
    float osmium        = 0.0f;
    float iridium       = 0.0f;
    float platinum      = 0.0f;
    float gold          = 0.0f;
    float mercury       = 0.0f;
    float thallium      = 0.0f;
    float lead          = 0.0f;
    float bismuth       = 0.0f;
    float polonium      = 0.0f;
    float astatine      = 0.0f;
    float radon         = 0.0f;
    float francium      = 0.0f;
    float radium        = 0.0f;
    float actinium      = 0.0f;
    float thorium       = 0.0f;
    float protactinium  = 0.0f;
    float uranium       = 0.0f;
    float neptunium     = 0.0f;
    float plutonium     = 0.0f;
    float americium     = 0.0f;
    float curium        = 0.0f;
    float berkelium     = 0.0f;
    float californium   = 0.0f;
    float einsteinium   = 0.0f;
    float fermium       = 0.0f;
    float mendelevium   = 0.0f;
    float nobelium      = 0.0f;
    float lawrencium    = 0.0f;
    float rutherfordium = 0.0f;
    float dubnium       = 0.0f;
    float seaborgium    = 0.0f;
    float bohrium       = 0.0f;
    float hassium       = 0.0f;
    float meitnerium    = 0.0f;
    float darmstadtium  = 0.0f;
    float roentgenium   = 0.0f;
    float cpernicium    = 0.0f;
    float nihnium       = 0.0f;
    float flerovium     = 0.0f;
    float moscovium     = 0.0f;
    float livermorium   = 0.0f;
    float tennessine    = 0.0f;
    float oganesson     = 0.0f;
};

#endif