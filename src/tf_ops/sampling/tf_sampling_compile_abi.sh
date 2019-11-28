#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudainc=/usr/local/cuda-10.0/include/
cudalib=/usr/local/cuda-10.0/lib64/
python=/opt/anaconda3/envs/thesis/bin/python
TF_INC=$($python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$($python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

#TF_CFLAGS=( $($python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $($python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
#echo $TF_CFLAGS
#echo $TF_LFLAGS

$nvcc tf_sampling_g.cu -c -o tf_sampling_g.cu.o -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1\
 -x cu -Xcompiler -fPIC -O2

g++ tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -std=c++11 -shared -fPIC -I $TF_INC \
-I $TF_INC/external/nsync/public -I $cudainc -L $TF_LIB -ltensorflow_framework -lcudart -L $cudalib -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#$nvcc -std=c++11 tf_sampling_g.cu -c -o tf_sampling_g.cu.o   \
#       ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

#g++ -std=c++11 -shared tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so  \
#	 ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}  -I $cudainc -L $cudalib -O2 -D_GLIBCXX_USE_CXX11_ABI=0
