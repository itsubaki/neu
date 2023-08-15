# neu

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/neu)](https://pkg.go.dev/github.com/itsubaki/neu)
[![Go Report Card](https://goreportcard.com/badge/github.com/itsubaki/neu?style=flat-square)](https://goreportcard.com/report/github.com/itsubaki/neu)
[![tests](https://github.com/itsubaki/neu/workflows/tests/badge.svg?branch=main)](https://github.com/itsubaki/neu/actions)
[![codecov](https://codecov.io/gh/itsubaki/neu/branch/main/graph/badge.svg?token=KMJ2GUC1FJ)](https://codecov.io/gh/itsubaki/neu)

Deep Learning framework for Go from scratch

## MNIST

```shell
make mnistdl
go run cmd/mnist/main.go --dir ./testdata
```

```shell
*model.MLP
 0: *layer.Affine: W(784, 50), B(1, 50): 39250
 1: *layer.BatchNorm: G(1, 50), B(1, 50): 100
 2: *layer.ReLU
 3: *layer.Affine: W(50, 50), B(1, 50): 2550
 4: *layer.BatchNorm: G(1, 50), B(1, 50): 100
 5: *layer.ReLU
 6: *layer.Affine: W(50, 50), B(1, 50): 2550
 7: *layer.BatchNorm: G(1, 50), B(1, 50): 100
 8: *layer.ReLU
 9: *layer.Affine: W(50, 10), B(1, 10): 510
10: *layer.SoftmaxWithLoss

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

## Seq2Seq

```shell
make additiondl
go run cmd/seq2seq/main.go --dir ./testdata
```

```shell
*model.PeekySeq2Seq
 0: *model.Encoder
 1: *layer.TimeEmbedding: W(13, 64): 832
 2: *layer.TimeLSTM: Wx(64, 512), Wh(128, 512), B(1, 512): 98816
 3: *model.PeekyDecoder
 4: *layer.TimeEmbedding: W(13, 64): 832
 5: *layer.TimeLSTM: Wx(192, 512), Wh(128, 512), B(1, 512): 164352
 6: *layer.TimeAffine: W(256, 13), B(1, 13): 3341
 7: *layer.TimeSoftmaxWithLoss

...
[9 0 + 4 0 8  ] [_ 4 9 8  ]; [2 9 8  ] ([6 9 10 5])
[6 5 5 + 9    ] [_ 6 6 4  ]; [4 6 6  ] ([11 1 1 5])
[9 3 + 4 0 3  ] [_ 4 9 6  ]; [4 9 6  ] ([11 9 1 5])
[5 8 + 8 5    ] [_ 1 4 3  ]; [4 4 1  ] ([11 11 0 5])
[2 0 + 7 4 5  ] [_ 7 6 5  ]; [4 9 8  ] ([11 9 10 5])
60,  0: loss=0.6508, train_acc=0.2000

[9 0 + 4 0 8  ] [_ 4 9 8  ]; [4 9 8  ] ([11 9 10 5])
[6 5 5 + 9    ] [_ 6 6 4  ]; [6 6 4  ] ([1 1 11 5])
[9 3 + 4 0 3  ] [_ 4 9 6  ]; [4 9 6  ] ([11 9 1 5])
[5 8 + 8 5    ] [_ 1 4 3  ]; [1 4 3  ] ([0 11 8 5])
[2 0 + 7 4 5  ] [_ 7 6 5  ]; [7 6 5  ] ([3 1 4 5])
80,  0: loss=0.1715, train_acc=1.0000
...
```

# Links

- [oreilly-japan/deep-learning-from-scratch](https://github.com/oreilly-japan/deep-learning-from-scratch)
- [oreilly-japan/deep-learning-from-scratch-2](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
