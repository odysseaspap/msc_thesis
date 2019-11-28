#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudalib=/usr/local/cuda-10.0/lib64/
python=/opt/anaconda3/envs/thesis/bin/python
TF_INC=$($python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$($python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$nvcc tf_auctionmatch_g.cu  -c -o tf_auctionmatch_g.cu.o -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
	 -x cu -Xcompiler -fPIC -O2

g++ tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -std=c++11 -shared -fPIC -I $TF_INC \
	-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2 -D_GLIBCXX_USE_CXX11_ABI=0
