# arts


# Install

 - https://www.tensorflow.org/install/lang_c

```
$ curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.12.0.tar.gz
$ sudo tar -C /usr/local -xzf libtensorflow.tar.gz
```


```
$ gcc example/hello_tf.c -ltensorflow -o hello_tf
$ ./hello_tf
Hello from TensorFlow C library version 1.12.0
```

# Reference

 1. [Deep Learning Book](https://www.deeplearningbook.org)
