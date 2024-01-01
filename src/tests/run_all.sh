
for test in ./src/tests/*.cpp
do
  echo $test
  g++ -std=c++20 \
  -I/opt/homebrew/Cellar/googletest/1.14.0/include \
  -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest \
  -o test src/game/*.cpp $test
  ./test
done

for test in ./src/tests/integration/*.cpp
do
  echo $test
  sh src/tests/integration/build.sh $test
  ./test
done
