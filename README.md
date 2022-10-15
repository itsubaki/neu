# neu

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/neu)](https://pkg.go.dev/github.com/itsubaki/neu)
[![Go Report Card](https://goreportcard.com/badge/github.com/itsubaki/neu?style=flat-square)](https://goreportcard.com/report/github.com/itsubaki/neu)
[![tests](https://github.com/itsubaki/neu/workflows/tests/badge.svg?branch=main)](https://github.com/itsubaki/neu/actions)
[![codecov](https://codecov.io/gh/itsubaki/neu/branch/main/graph/badge.svg?token=KMJ2GUC1FJ)](https://codecov.io/gh/itsubaki/neu)


# MNIST

```shell
go run cmd/mnist/main.go --dir ./testdata
```

```shell
   0: loss=[[2.2003]], train_acc=0.2000, test_acc=0.2000
predict: [9 2 2 1 0 2 6 1 1 5 7 9 1 7 9 9 3 1 1 6]
label  : [9 9 0 4 7 6 4 9 8 2 3 6 9 5 7 4 6 5 8 7]

 100: loss=[[0.3619]], train_acc=0.9300, test_acc=0.8300
predict: [2 1 2 1 2 0 2 9 3 1 3 7 7 4 1 7 9 8 1 8]
label  : [2 1 2 1 2 0 2 9 5 1 3 7 7 4 1 7 9 8 1 8]

 200: loss=[[0.2274]], train_acc=0.9500, test_acc=0.9000
predict: [8 6 3 6 4 8 1 4 8 0 7 6 1 2 9 6 8 5 2 0]
label  : [8 6 3 0 4 8 1 4 8 0 7 6 1 3 4 6 8 5 2 0]

...
```

# Reference

 1. [Deep Learning from Scratch](https://www.oreilly.com/library/view/deep-learning-from/9781492041405/)
 2. [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
