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
   0: loss=[[2.2485]], train_acc=0.1800, test_acc=0.1300
predict: [9 4 1 4 8 3 2 4 2 4 9 4 4 3 4 2 6 8 4 8]
label  : [0 6 8 7 8 2 0 1 3 1 8 7 7 7 5 6 1 8 8 8]

 100: loss=[[0.4916]], train_acc=0.8800, test_acc=0.8900
predict: [4 3 3 6 4 8 3 1 6 7 2 9 8 1 8 7 3 0 5 6]
label  : [4 3 3 6 4 8 3 1 6 7 2 9 8 1 8 3 8 0 5 6]

 200: loss=[[0.4164]], train_acc=0.8900, test_acc=0.8600
predict: [4 4 3 1 0 0 3 3 7 6 1 9 8 2 9 5 2 6 6 6]
label  : [9 4 3 1 0 0 3 3 7 6 1 9 8 2 9 9 2 6 6 6]

 300: loss=[[0.3602]], train_acc=0.8800, test_acc=0.8800
predict: [7 9 8 5 2 6 3 6 2 6 9 4 9 8 2 1 5 7 0 9]
label  : [7 9 8 6 2 6 3 6 2 6 9 4 9 8 2 1 5 7 0 9]

 400: loss=[[0.2036]], train_acc=0.9400, test_acc=0.9500
predict: [4 1 2 6 7 1 8 7 5 5 7 6 7 5 8 4 7 4 3 2]
label  : [4 1 2 6 7 1 8 7 5 5 7 6 7 5 8 4 7 4 3 2]

...
```

# Reference

 1. [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
