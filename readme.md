Thanks to Amanatides, Woo, Occam, Incf, and so on.

How to compile:
if on windows, install wsl.
after installing wsl, open a terminal and run wsl (debian based preferred)

if on debian based linux (or wsl with a debian based distro):
apt install build-essential libtbb-dev libglfw3-dev
make -j 8
./bin/g2gradc

note: you can set the number (-j 8) to your thread count (4 core/8 threads is why I use 8. on another system I use 32 even though its only compiling 9 files)

ffmpeg -framerate 30 -i debug_material_%d.bmp -c:v libx264 -pix_fmt yuv420p materialtest.mp4
ffmpeg -framerate 30 -i debug_fluid_%d.bmp -c:v libx264 -pix_fmt yuv420p fluidtest.mp4