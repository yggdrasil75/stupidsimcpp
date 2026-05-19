this project:
Create a planet using 3d coordinates in space (voxels) which will simulate a world at a very large scale.
This first simulation will utilize real world geological scale physics to create a planet, including the following:
minor noise filter to create a base map
tectonics
erosion
major weather cycles
climate
some tidal forces from nearby planets/moons/stars
Impact cratering
surface flows
oceans/lakes
subsurface flows
all of this will then be presented to the user as a visible planet. each item will have various settings to play with, such as how many plates there should be.

second simulation:
after a "reasonable" planet has been created, capabilities of exploring the world becomes a thing. 
it then should be possible to "focus" on a region of the surface, which will use something similar to eulerian diffusion to create a new region for later retrieval
the region will be 10^3 miles (grid aligned) which may cause some minor issues with mountains being of a tiny chunk or valleys being of a tiny chunk. (will need to figure out how best to handle such edge cases)
each of these minor regions will then be able to be fully explored, having minor versions of the above applied: localized weather, erosion, surface flows, etc to make it best match.
plant life and animal life will be simulated.

test chamber:
plants and animals will be simulated partially in "test chambers" (partial copies of the main world wtih variations such as ground noise and rotations of objects)
this is done in order to prevent having to fully calculate all physics every single frame
plants will grow in a "test chamber" the test chamber will be a semi-random variations on the factors may be found in the main world.
this will be done a hundred times or so, randomizing various factors each time.
then in the main world the plant will group sections of itself, find the more similar sections from the various sims and reuse that result.
animals will simulate various things such as how they fight, how they dig, how they eat, how they fly, run, walk, etc. hundreds of times each.
reuse those sims in the main world, same as the other.
when a "novel" instance appears (ie: the first time a tree grows on a cliff face - determined if nothing simulated so far fits), then it will be added to the test chamber to simulate varitions 

plants:
plants in the world will be simulated primarily just using the test chamber, with minimal in-world sims. 
plants will track their sunlight by having the test chamber version check that the sun can be seen from various leaves, randomly select some small number every frame of the test chamber.
plants will track nutrients in the soil, simulating various growth patterns for roots based on those nutrients, as well as using those nutrients to try and influence how the plant may grow (lots of iron, make it heavier, more squat type stuff).
plants will track strength of limbs and how those limbs will interact with physics (wind, gravity, air pressure, a bird landing on the branch, a lion jumping up to grab the bird on the branch, etc)
plants will then mostly just copy their pre-simulated physics in the main world.

animals:
animals will use ai models to run.
not llms, but various smaller sizes of models.
will try weird approaches to it just to see what I can do.
start with a basic unet, try setting up moes, something weird with multilayered something or other.
have it set up to be able to change its own size automatically for various things.
species will reuse a model, same as reusing the test chamber simulations of physics.
randomly assign "hero" creatures at birth. these creatures will use the model, but will constantly train it up to a certain age, then they will use their own version of the model from that point onwards.
the hero creatures can spawn variants of the creatures, or be merged into the general model of its own species.
the animals section really will be hard to track, because I want to be able to simulate using 100m up to 10b models a massive number of small creatures of varying sizes, but that will make this impossible to run. not using learning models will make it impossible to reach the scale I want though. generic a* pathfinding is pointless for this sim since the goal is to make a realistic world.
they can see the world by having a small render of their vision, will probably have something in the eyes "dna" determine resolution
will have various other inputs as well such as touch and sound.
smell is minor, taste is also minor. but taste can be simulated just by way of nutrient values.

the user interaction (what makes it a game):
the primary goal is a worldbuilding tool. be able to figure out where game might be found, what kind of game, what fauna there is, what flora, etc.
since I do want to be able to present this as a fun way to create a world, lets explore how we can do that:
creating the main world has a lot of sliders, checkboxes, variations on methodology, etc. 
users can design and import creatures, assign various classes of voxels to them (ie: assign some voxels as bones, some as muscle, tissue, fat, skin, etc)
they can then be sent to the test chamber to try out everything and dropped into the world.
same with plants.
minor goal with plants:
the test chamber for plants will have a "bonzai mode" where the player can just run it like maintaining a bonzai tree, trimming the right branch, watering at the right time, adding the right soil mixture, etc.
animal sim might have a "tomogatchi mode" where there is no other animals, just that 1. there will likely be other plants just to give it a world to interact with.
this tomogatchi mode will likely force that 1 creature to be a model.

the engine:
I want to avoid reusing a common engine, both for being able to optimize directly and for being able to customize it more.
the current goals of the engine:
be able to track voxels.
render efficiently
fluid physics
gas physics by way of eulerian fields
full pbr for the user to view, partial pbr for some of the creatures, others will have the "fast" (non-pbr) version.
lod


goal 1:
the bonzai tree.

goal 2:
physics simulation for animals (randomly generated animals)

goal 3:
tomogatchi with basic environment

goal 4:
plant/animal creation.

goal 5:
generally realized sub environments (the 10^3 miles)
this will be randomized+the factors that will be used later for the planet

goal 6:
planet creation and generalized planet simulation

goal 7:
integration







Thanks to Amanatides, Woo, Occam, Incf, and so on.

How to compile:
if on windows, install wsl.
after installing wsl, open a terminal and run wsl (debian based preferred) then do below

if on debian based linux (or wsl with a debian based distro):
git clone this repo
git submodule init
git submodule update --recursive
apt install build-essential libtbb-dev libglfw3-dev
make -j 8
./bin/g2gradc

note: you can set the number (-j 8) to your thread count (4 core/8 threads is why I use 8. on another system I use 32 even though its only compiling 9 files)

ffmpeg -framerate 30 -i debug_material_%d.bmp -c:v libx264 -pix_fmt yuv420p materialtest.mp4
ffmpeg -framerate 30 -i debug_fluid_%d.bmp -c:v libx264 -pix_fmt yuv420p fluidtest.mp4