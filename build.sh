
g++ src/main.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o main

g++ src/uci.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o uci

# g++ src/selfplay.cpp src/game/*.cpp -std=c++20 -O3 -DNDEBUG -o selfplay
