package matrix_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMatrix_Dot() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Dot(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [19 22]
	// [43 50]

}

func ExampleMatrix_Dimension() {
	fmt.Println(matrix.New().Dimension())
	fmt.Println(matrix.New([]float64{1, 2, 3}).Dimension())

	// Output:
	// 0 0
	// 1 3

}

func ExampleMatrix_Add() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Add(B) {
		fmt.Println(r)
	}

	// Output:
	// [6 8]
	// [10 12]

}

func ExampleMatrix_Sub() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Sub(B) {
		fmt.Println(r)
	}

	// Output:
	// [-4 -4]
	// [-4 -4]

}

func ExampleMatrix_Mul() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Mul(B) {
		fmt.Println(r)
	}

	// Output:
	// [5 12]
	// [21 32]

}

func ExampleMatrix_Func() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.Func(func(v float64) float64 { return v * 3.0 }) {
		fmt.Println(r)
	}

	// Output:
	// [3 6]
	// [9 12]

}

func ExampleMatrix_T() {
	A := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	for _, r := range A.T() {
		fmt.Println(r)
	}

	// Output:
	// [1 4]
	// [2 5]
	// [3 6]
}

func ExampleZero() {
	for _, r := range matrix.Zero(2, 3) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]

}

func ExampleRand() {
	rand.Seed(1)
	for _, r := range matrix.Rand(2, 3) {
		fmt.Println(r)
	}

	// Output:
	// [-1.233758177597947 -0.12634751070237293 -0.5209945711531503]
	// [2.28571911769958 0.3228052526115799 0.5900672875996937]

}

func ExampleMask() {
	x := matrix.New([]float64{0, 1}, []float64{2, 3})
	for _, r := range matrix.Mask(x, [][]bool{{true, false}, {true, false}}) {
		fmt.Println(r)
	}

	// Output:
	// [0 1]
	// [0 3]
}

func ExampleSumAxis1() {
	x := matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	fmt.Println(matrix.SumAxis1(x))

	// Output:
	// [5 7 9]
}
