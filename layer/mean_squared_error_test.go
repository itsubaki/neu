package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMeanSquaredError() {
	l := &layer.MeanSquaredError{}
	fmt.Println(l)

	// forward
	target := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)
	q := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	loss := l.Forward(target, q)
	fmt.Println(loss)

	// backward
	dout := matrix.New([]float64{1})
	gx0, gx1 := l.Backward(dout)
	for _, r := range gx0 {
		fmt.Printf("%.4f\n", r)
	}

	for _, r := range gx1 {
		fmt.Printf("%.4f\n", r)
	}

	// Output:
	// *layer.MeanSquaredError
	// [[0.0695]]
	// [0.0100 0.0050 -0.0400 0.0000 0.0050 0.0100 0.0000 0.0100 0.0000 0.0000]
	// [0.0100 0.0050 -0.0900 0.0000 0.0050 0.0100 0.0000 0.0600 0.0000 0.0000]
	// [-0.0100 -0.0050 0.0400 -0.0000 -0.0050 -0.0100 -0.0000 -0.0100 -0.0000 -0.0000]
	// [-0.0100 -0.0050 0.0900 -0.0000 -0.0050 -0.0100 -0.0000 -0.0600 -0.0000 -0.0000]
}

func ExampleMeanSquaredError_Backward() {
	f := func(target, q matrix.Matrix) (matrix.Matrix, matrix.Matrix, matrix.Matrix) {
		l := &layer.MeanSquaredError{}
		loss := l.Forward(target, q)
		gx0, gx1 := l.Backward(matrix.New([]float64{1}))
		return loss, gx0, gx1
	}

	target, q := matrix.New([]float64{1}), matrix.New([]float64{2})
	fmt.Println(f(target, q))
	fmt.Println(f(target.Broadcast(1, 4), q.Broadcast(1, 4)))

	// Output:
	// [[1]] [[-2]] [[2]]
	// [[1]] [[-0.5 -0.5 -0.5 -0.5]] [[0.5 0.5 0.5 0.5]]
}

func ExampleMeanSquaredError_Params() {
	l := &layer.MeanSquaredError{}

	l.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// []
	// []
}
