




## Profiling

```
brew install gperftools
go install github.com/google/pprof@latest

sh build.sh src/uci.cpp -o uci -DNNUE_EVAL=1 -DPRINT_PV_CHANGES=0 -DDEBUG_TT=0 -L$(brew --prefix gperftools)/lib -lprofiler

CPUPROFILE=/tmp/prof.out ./uci "move e2e4 c7c5 g1f3 d7d6" "go depth 8" "lazyquit"

~/go/bin/pprof -png ./uci /tmp/prof.out
```