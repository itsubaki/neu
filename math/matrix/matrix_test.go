package matrix_test

import (
	"fmt"

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

func ExampleMatrix_Shape() {
	fmt.Println(matrix.New().Shape())
	fmt.Println(matrix.New([]float64{1, 2, 3}).Shape())

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

func ExampleMatrix_T() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.T() {
		fmt.Println(r)
	}

	// Output:
	// [1 3]
	// [2 4]

}

func ExampleSumAxis1() {
	x := matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	fmt.Println(matrix.SumAxis1(x))

	// Output:
	// [5 7 9]
}

func ExampleCrossEntropyError() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC3%E5%88%B7%E3%81%BE%E3%81%A7

	t := matrix.New(
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		[]float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
	)
	y := matrix.New(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)

	fmt.Println(matrix.CrossEntropyError(y, t))

	// Output:
	// [0.510825457099338 2.302584092994546]

}
