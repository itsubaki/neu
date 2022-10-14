package neu_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/neu"
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/optimizer"
)

func Example_mnist() {
	// data
	train, test := mnist.Must(mnist.Load("./testdata"))

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// hyper-parameter
	batchSize := 10
	iter := 1000

	// init
	rand.Seed(1) // for test
	n := neu.New(&neu.Config{
		InputSize:     784,
		HiddenSize:    50,
		OutputSize:    10,
		BatchSize:     batchSize,
		WeightInitStd: 0.01,
		Optimizer:     &optimizer.SGD{LearningRate: 0.1},
	})

	// learning
	for i := 0; i < iter; i++ {
		mask := neu.Random(train.N, batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		grads := n.Gradient(xbatch, tbatch)
		n.Optimize(grads)

		if i%200 == 0 {
			loss := n.Loss(xbatch, tbatch)
			acc := n.Accuracy(xbatch, tbatch)
			mask := neu.Random(test.N, batchSize)
			tacc := n.Accuracy(matrix.Batch(xt, mask), matrix.Batch(tt, mask))

			fmt.Printf("loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", loss, acc, tacc)
		}
	}

	// Output:
	// loss=[[2.2971]], train_acc=0.3000, test_acc=0.2000
	// loss=[[0.3096]], train_acc=1.0000, test_acc=0.5000
	// loss=[[0.1905]], train_acc=1.0000, test_acc=1.0000
	// loss=[[0.0858]], train_acc=1.0000, test_acc=1.0000
	// loss=[[0.0340]], train_acc=1.0000, test_acc=0.9000

}

func Example_neu() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	batchSize, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	n := neu.New(&neu.Config{
		InputSize:     inSize,
		HiddenSize:    hiddenSize,
		OutputSize:    outSize,
		BatchSize:     batchSize,
		WeightInitStd: 0.01,
		Optimizer:     &optimizer.SGD{LearningRate: 0.1},
	})

	// learning
	for i := 0; i < 10000; i++ {
		y := n.Predict(x)
		loss := n.Loss(x, t)
		grads := n.Gradient(x, t)
		n.Optimize(grads)

		if i%2000 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f\n", layer.Softmax(y), loss)
		}
	}

	// Output:
	// predict=[[0.5000 0.5000] [0.5000 0.5000] [0.5000 0.5000]], loss=[[0.6931]]
	// predict=[[0.9876 0.0124] [0.0238 0.9762] [0.0038 0.9962]], loss=[[0.0135]]
	// predict=[[0.9967 0.0033] [0.0076 0.9924] [0.0010 0.9990]], loss=[[0.0040]]
	// predict=[[0.9982 0.0018] [0.0044 0.9956] [0.0005 0.9995]], loss=[[0.0022]]
	// predict=[[0.9988 0.0012] [0.0031 0.9969] [0.0004 0.9996]], loss=[[0.0015]]
}

func Example_accuracy() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	batchSize, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	n := neu.New(&neu.Config{
		InputSize:     inSize,
		HiddenSize:    hiddenSize,
		OutputSize:    outSize,
		BatchSize:     batchSize,
		WeightInitStd: 0.01,
		Optimizer:     &optimizer.SGD{LearningRate: 0.1},
	})

	// learning
	for i := 0; i < 1000; i++ {
		y := n.Predict(x)
		loss := n.Loss(x, t)
		grads := n.Gradient(x, t)
		n.Optimize(grads)

		if i%250 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f, acc=%.4f\n", layer.Softmax(y), loss, n.Accuracy(x, t))
		}
	}

	// Output:
	// predict=[[0.5000 0.5000] [0.5000 0.5000] [0.5000 0.5000]], loss=[[0.6931]], acc=0.6667
	// predict=[[0.3373 0.6627] [0.3373 0.6627] [0.3203 0.6797]], loss=[[0.6281]], acc=0.6667
	// predict=[[0.4506 0.5494] [0.4505 0.5495] [0.0703 0.9297]], loss=[[0.4896]], acc=0.6667
	// predict=[[0.5120 0.4880] [0.4824 0.5176] [0.0191 0.9809]], loss=[[0.4491]], acc=1.0000
}

func Example_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	batchSize, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	n := neu.New(&neu.Config{
		InputSize:     inSize,
		HiddenSize:    hiddenSize,
		OutputSize:    outSize,
		BatchSize:     batchSize,
		WeightInitStd: 0.01,
		Optimizer:     &optimizer.SGD{LearningRate: 0.1},
	})

	// gradient
	ngrads := n.NumericalGradient(x, t)
	grads := n.Gradient(x, t)

	// check
	for _, k := range []string{"W1", "B1", "W2", "B2"} {
		diff := matrix.FuncWith(ngrads[k], grads[k], func(a, b float64) float64 { return math.Abs(a - b) })
		fmt.Printf("%v: %v\n", k, diff.Avg())
	}

	// Output:
	// W1: 1.6667580752359698e-10
	// B1: 0.0011093844469993565
	// W2: 2.8191449832846066e-10
	// B2: 0.11111413456359369

}

func Example_optimize() {
	// initial weight
	params := make(map[string]matrix.Matrix)
	params["W1"] = matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	params["B1"] = matrix.New([]float64{0.1, 0.2, 0.3})
	params["W2"] = matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	params["B2"] = matrix.New([]float64{0.1, 0.2})

	// training
	x := matrix.New([]float64{0.5, 0.5})
	t := matrix.New([]float64{1, 0})

	// learning
	for i := 0; i < 40; i++ {
		// layer
		layers := []neu.Layer{
			&layer.Affine{W: params["W1"], B: params["B1"]},
			&layer.ReLU{},
			&layer.Affine{W: params["W2"], B: params["B2"]},
		}
		last := &layer.SoftmaxWithLoss{}

		// forward
		for _, l := range layers {
			x = l.Forward(x, nil)
		}
		loss := last.Forward(x, t)

		if i%10 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f\n", layer.Softmax(x), loss)
		}

		// backward
		dout, _ := last.Backward(matrix.New([]float64{1}))
		for i := len(layers) - 1; i > -1; i-- {
			dout, _ = layers[i].Backward(dout)
		}

		// gradient
		grads := make(map[string]matrix.Matrix)
		grads["W1"] = layers[0].(*layer.Affine).DW
		grads["B1"] = layers[0].(*layer.Affine).DB
		grads["W2"] = layers[2].(*layer.Affine).DW
		grads["B2"] = layers[2].(*layer.Affine).DB

		// optimize
		opt := &optimizer.SGD{LearningRate: 0.1}
		params = opt.Update(params, grads)
	}

	// Output:
	// predict=[[0.3555 0.6445]], loss=[[1.0343]]
	// predict=[[0.8930 0.1070]], loss=[[0.1132]]
	// predict=[[0.9878 0.0122]], loss=[[0.0122]]
	// predict=[[0.9982 0.0018]], loss=[[0.0018]]
}

func Example_layer() {
	// weight
	W1 := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B1 := matrix.New([]float64{0.1, 0.2, 0.3})
	W2 := matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	B2 := matrix.New([]float64{0.1, 0.2})
	W3 := matrix.New([]float64{0.1, 0.3}, []float64{0.2, 0.4})
	B3 := matrix.New([]float64{0.1, 0.2})

	// layer
	layers := []neu.Layer{
		&layer.Affine{W: W1, B: B1},
		&layer.Sigmoid{},
		&layer.Affine{W: W2, B: B2},
		&layer.Sigmoid{},
		&layer.Affine{W: W3, B: B3},
	}

	// forward
	x := matrix.New([]float64{1.0, 0.5})
	for _, l := range layers {
		x = l.Forward(x, nil)
	}

	// backward
	dout := matrix.New([]float64{1})
	for i := len(layers) - 1; i > -1; i-- {
		dout, _ = layers[i].Backward(dout)
	}

	fmt.Println(x)
	fmt.Println(dout)

	// Output:
	// [[0.3168270764110298 0.6962790898619668]]
	// [[0.004530872169552449 0.005957136450888734]]

}

func Example_simpleNet() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC7%E5%88%B7%E3%81%BE%E3%81%A7

	// weight
	W := matrix.New(
		[]float64{0.47355232, 0.99773930, 0.84668094},
		[]float64{0.85557411, 0.03563661, 0.69422093},
	)

	// data
	x := matrix.New([]float64{0.6, 0.9})
	t := []float64{0, 0, 1}

	// predict
	p := matrix.Dot(x, W)
	y := activation.Softmax(p[0])
	e := loss.CrossEntropyError(y, t)

	fmt.Println(p)
	fmt.Println(e)

	// gradient
	fW := func(w ...float64) float64 {
		p := matrix.Dot(x, W)
		y := activation.Softmax(p[0])
		e := loss.CrossEntropyError(y, t)
		return e
	}

	grad := func(f func(x ...float64) float64, x matrix.Matrix) matrix.Matrix {
		out := make(matrix.Matrix, 0)
		for _, r := range x {
			out = append(out, numerical.Gradient(f, r))
		}

		return out
	}

	dW := grad(fW, W)
	for _, r := range dW {
		fmt.Println(r)
	}

	// Output:
	// [[1.054148091 0.630716529 1.132807401]]
	// 0.9280682857864075
	// [0.2192475712392561 0.14356242984070455 -0.3628100010055757]
	// [0.3288713569016277 0.21534364482433954 -0.5442150014750569]

}

func Example_neuralNetwork() {
	// weight
	W1 := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B1 := matrix.New([]float64{0.1, 0.2, 0.3})
	W2 := matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	B2 := matrix.New([]float64{0.1, 0.2})
	W3 := matrix.New([]float64{0.1, 0.3}, []float64{0.2, 0.4})
	B3 := matrix.New([]float64{0.1, 0.2})

	// data
	x := matrix.New([]float64{1.0, 0.5})

	// forward
	A1 := matrix.Dot(x, W1).Add(B1)
	Z1 := matrix.Func(A1, activation.Sigmoid)
	A2 := matrix.Dot(Z1, W2).Add(B2)
	Z2 := matrix.Func(A2, activation.Sigmoid)
	A3 := matrix.Dot(Z2, W3).Add(B3)
	y := matrix.Func(A3, func(v float64) float64 { return v }) // identity

	// result
	fmt.Println(A1)
	fmt.Println(Z1)

	fmt.Println(A2)
	fmt.Println(Z2)

	fmt.Println(A3)
	fmt.Println(y)

	// Output:
	// [[0.30000000000000004 0.7 1.1]]
	// [[0.574442516811659 0.6681877721681662 0.7502601055951177]]
	// [[0.5161598377933344 1.2140269561658172]]
	// [[0.6262493703990729 0.7710106968556123]]
	// [[0.3168270764110298 0.6962790898619668]]
	// [[0.3168270764110298 0.6962790898619668]]

}

func Example_perceptron() {
	f := func(x, w []float64, b float64) int {
		var sum float64
		for i := range x {
			sum = sum + x[i]*w[i]
		}

		v := sum + b
		if v <= 0 {
			return 0
		}

		return 1
	}

	AND := func(x []float64) int { return f(x, []float64{0.5, 0.5}, -0.7) }
	NAND := func(x []float64) int { return f(x, []float64{-0.5, -0.5}, 0.7) }
	OR := func(x []float64) int { return f(x, []float64{0.5, 0.5}, -0.2) }
	XOR := func(x []float64) int { return AND([]float64{float64(NAND(x)), float64(OR(x))}) }

	fmt.Println("AND")
	fmt.Println(AND([]float64{0, 0}))
	fmt.Println(AND([]float64{1, 0}))
	fmt.Println(AND([]float64{0, 1}))
	fmt.Println(AND([]float64{1, 1}))
	fmt.Println()

	fmt.Println("NAND")
	fmt.Println(NAND([]float64{0, 0}))
	fmt.Println(NAND([]float64{1, 0}))
	fmt.Println(NAND([]float64{0, 1}))
	fmt.Println(NAND([]float64{1, 1}))
	fmt.Println()

	fmt.Println("OR")
	fmt.Println(OR([]float64{0, 0}))
	fmt.Println(OR([]float64{1, 0}))
	fmt.Println(OR([]float64{0, 1}))
	fmt.Println(OR([]float64{1, 1}))
	fmt.Println()

	fmt.Println("XOR")
	fmt.Println(XOR([]float64{0, 0}))
	fmt.Println(XOR([]float64{1, 0}))
	fmt.Println(XOR([]float64{0, 1}))
	fmt.Println(XOR([]float64{1, 1}))
	fmt.Println()

	// Output:
	// AND
	// 0
	// 0
	// 0
	// 1
	//
	// NAND
	// 1
	// 1
	// 1
	// 0
	//
	// OR
	// 0
	// 1
	// 1
	// 1
	//
	// XOR
	// 0
	// 1
	// 1
	// 0
	//

}
