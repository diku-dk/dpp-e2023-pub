CXX = nvcc -O3 -arch=sm_70
#-arch=sm_80

#-arch=sm_60

HELPERS  = ../helper.h
SRC_CGBN = cgbn-perf.cu
EXE_CGBN = big-num-cgbn

default: run-1024

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<


run-2048:
	$(CXX) -ICGBN -DNUM_BITS=65536 -DTHD_PER_INST=32 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 65536

run-1024:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 131072

run-512:
	$(CXX) -ICGBN -DNUM_BITS=16384 -DTHD_PER_INST=32 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 262144
	
run-256:
	$(CXX) -ICGBN -DNUM_BITS=8192 -DTHD_PER_INST=32 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 524288
	
run-128:
	$(CXX) -ICGBN -DNUM_BITS=4096 -DTHD_PER_INST=16 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 1048576

run-64:
	$(CXX) -ICGBN -DNUM_BITS=2048 -DTHD_PER_INST=8 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 2097152
	
run-32:
	$(CXX) -ICGBN -DNUM_BITS=1024 -DTHD_PER_INST=4 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 4194304

run-16:
	$(CXX) -ICGBN -DNUM_BITS=512  -DTHD_PER_INST=2 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN) 8388608


clean:
	rm -f $(EXE_CGBN) $(EXE_OUR)

