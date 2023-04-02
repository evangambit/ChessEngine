
g++ src/main.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o main

g++ src/selfplay.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o selfplay
