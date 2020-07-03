LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python test_proposed.py --arch squeezenet  --seed 0
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python test_proposed.py --arch mobilenet  --seed 0
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python test_fair.py --seed 0
LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 python test_unfair.py --seed 0