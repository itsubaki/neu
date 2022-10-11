package neu_test

import (
	"fmt"

	"github.com/itsubaki/neu"
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/loss"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

func ExampleNeu() {

}

func Example_sgd() {
	// initial weight
	params := make(map[string]matrix.Matrix)
	params["W1"] = matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	params["B1"] = matrix.New([]float64{0.1, 0.2, 0.3})
	params["W2"] = matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	params["B2"] = matrix.New([]float64{0.1, 0.2})

	// training
	T := matrix.New([]float64{1, 0})
	X := matrix.New([]float64{0.5, 0.5})
	learningRate := 0.1
	loop := 40

	// learning
	for i := 0; i < loop; i++ {
		// layer
		layers := []neu.Layer{
			&layer.Affine{W: params["W1"], B: params["B1"]},
			&layer.ReLU{},
			&layer.Affine{W: params["W2"], B: params["B2"]},
		}
		last := &layer.SoftmaxWithLoss{}

		// forward
		for _, l := range layers {
			X = l.Forward(X, nil)
		}
		loss := last.Forward(X, T)

		if i%10 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f\n", layer.Softmax(X), loss)
		}

		// backward
		dout, _ := last.Backward(matrix.New([]float64{1}))
		for i := len(layers) - 1; i > -1; i-- {
			dout, _ = layers[i].Backward(dout)
		}

		grads := make(map[string]matrix.Matrix)
		grads["W1"] = layers[0].(*layer.Affine).DW
		grads["B1"] = layers[0].(*layer.Affine).DB
		grads["W2"] = layers[2].(*layer.Affine).DW
		grads["B2"] = layers[2].(*layer.Affine).DB

		opt := &optimizer.SGD{LearningRate: learningRate}
		params = opt.Update(params, grads)
	}

	// Output:
	// predict=[[0.3555 0.6445]], loss=[[1.0343]]
	// predict=[[0.8930 0.1070]], loss=[[0.1132]]
	// predict=[[0.9878 0.0122]], loss=[[0.0122]]
	// predict=[[0.9982 0.0018]], loss=[[0.0018]]
}

func Example_layer() {
	// initial weight
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
	X := matrix.New([]float64{1.0, 0.5})
	for _, l := range layers {
		X = l.Forward(X, nil)
	}

	// backward
	dout := matrix.New([]float64{1})
	for i := len(layers) - 1; i > -1; i-- {
		dout, _ = layers[i].Backward(dout)
	}

	fmt.Println(X)
	fmt.Println(dout)

	// Output:
	// [[0.3168270764110298 0.6962790898619668]]
	// [[0.004530872169552449 0.005957136450888734]]

}

func Example_simpleNet() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC7%E5%88%B7%E3%81%BE%E3%81%A7

	// rand.NormFloat64()
	W := matrix.New(
		[]float64{0.47355232, 0.99773930, 0.84668094},
		[]float64{0.85557411, 0.03563661, 0.69422093},
	)
	x := matrix.New([]float64{0.6, 0.9})

	// predict
	p := matrix.Dot(x, W)
	fmt.Println(p)

	y := activation.Softmax(p[0])
	e := loss.CrossEntropyError(y, []float64{0, 0, 1})

	fmt.Println(e)

	// Output:
	// [[1.054148091 0.630716529 1.132807401]]
	// 0.9280682857864075

}

func Example_neuralNetwork() {
	// network
	W1 := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B1 := matrix.New([]float64{0.1, 0.2, 0.3})
	W2 := matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	B2 := matrix.New([]float64{0.1, 0.2})
	W3 := matrix.New([]float64{0.1, 0.3}, []float64{0.2, 0.4})
	B3 := matrix.New([]float64{0.1, 0.2})

	// forward
	X := matrix.New([]float64{1.0, 0.5})
	A1 := matrix.Dot(X, W1).Add(B1)
	Z1 := matrix.Func(A1, activation.Sigmoid)
	A2 := matrix.Dot(Z1, W2).Add(B2)
	Z2 := matrix.Func(A2, activation.Sigmoid)
	A3 := matrix.Dot(Z2, W3).Add(B3)
	Y := matrix.Func(A3, func(v float64) float64 { return v }) // identity

	// result
	fmt.Println(A1)
	fmt.Println(Z1)

	fmt.Println(A2)
	fmt.Println(Z2)

	fmt.Println(A3)
	fmt.Println(Y)

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
