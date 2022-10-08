package neu_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

func Example_simpleNet() {
	W := matrix.New(
		[]float64{0.47355232, 0.99773930, 0.84668094},
		[]float64{0.85557411, 0.03563661, 0.69422093},
	)
	x := matrix.New([]float64{0.6, 0.9})

	// predict
	p := matrix.Dot(x, W)
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata
	fmt.Println(p) 

	// Output:
	// [[1.054148091 0.630716529 1.132807401]]

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
	Z1 := activation.Sigmoid(A1[0])
	A2 := matrix.Dot(matrix.New(Z1), W2).Add(B2)
	Z2 := activation.Sigmoid(A2[0])
	A3 := matrix.Dot(matrix.New(Z2), W3).Add(B3)
	Y := activation.Identity(A3[0])

	// result
	fmt.Println(A1[0])
	fmt.Println(Z1)

	fmt.Println(A2[0])
	fmt.Println(Z2)

	fmt.Println(A3[0])
	fmt.Println(Y)

	// Output:
	// [0.30000000000000004 0.7 1.1]
	// [0.574442516811659 0.6681877721681662 0.7502601055951177]
	// [0.5161598377933344 1.2140269561658172]
	// [0.6262493703990729 0.7710106968556123]
	// [0.3168270764110298 0.6962790898619668]
	// [0.3168270764110298 0.6962790898619668]

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
