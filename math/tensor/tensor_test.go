package tensor_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
)

func ExampleZero() {
	fmt.Println(tensor.Zero(2, 2, 2))

	// Output:
	// [[[0 0] [0 0]] [[0 0] [0 0]]]
}

func ExampleZeroLike() {
	x := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	fmt.Println(tensor.ZeroLike(x))

	// Output:
	// [[[0 0 0] [0 0 0]]]
}

func ExampleOneHot() {
	ts := []matrix.Matrix{
		{
			{0, 1, 2},
			{3, 4, 5},
		},
		{
			{0, 1, 2},
			{3, 4, 5},
		},
	}

	onehot := tensor.OneHot(ts, 6)
	for _, m := range onehot {
		for _, r := range m {
			fmt.Println(r)
		}
		fmt.Println()
	}

	// Output:
	// [1 0 0 0 0 0]
	// [0 1 0 0 0 0]
	// [0 0 1 0 0 0]
	// [0 0 0 1 0 0]
	// [0 0 0 0 1 0]
	// [0 0 0 0 0 1]
	//
	// [1 0 0 0 0 0]
	// [0 1 0 0 0 0]
	// [0 0 1 0 0 0]
	// [0 0 0 1 0 0]
	// [0 0 0 0 1 0]
	// [0 0 0 0 0 1]
}

func ExampleAdd() {
	x := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}
	y := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	fmt.Println(tensor.Add(x, y))

	// Output:
	// [[[2 4 6] [8 10 12]]]
}

func ExampleMul() {
	x := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}
	y := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	fmt.Println(tensor.Mul(x, y))

	// Output:
	// [[[1 4 9] [16 25 36]]]
}

func ExampleSum() {
	x := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	fmt.Println(tensor.Sum(x))

	// Output:
	// [[2 4 6] [8 10 12]]
}

func ExampleConcat() {
	hs := []matrix.Matrix{
		{
			{1, 2},
			{3, 4},
		},
		{
			{5, 6},
			{7, 8},
		},
	}

	out := []matrix.Matrix{
		{
			{10, 20},
			{30, 40},
		},
		{
			{50, 60},
			{70, 80},
		},
	}

	for _, t := range tensor.Concat(hs, out) {
		fmt.Println(t)
	}

	// Output:
	// [[1 2 10 20] [3 4 30 40]]
	// [[5 6 50 60] [7 8 70 80]]
}

func ExampleSplit() {
	dout := []matrix.Matrix{
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		},
		{
			{9, 10, 11, 12},
			{13, 14, 15, 16},
		},
		{
			{17, 18, 19, 20},
			{21, 22, 23, 24},
		},
	}

	dhs, dout := tensor.Split(dout, 2)
	for _, r := range dhs {
		fmt.Println(r)
	}
	for _, r := range dout {
		fmt.Println(r)
	}

	// Output:
	// [[1 2] [5 6]]
	// [[9 10] [13 14]]
	// [[17 18] [21 22]]
	// [[3 4] [7 8]]
	// [[11 12] [15 16]]
	// [[19 20] [23 24]]

}

func ExampleRepeat() {
	a := matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	for _, r := range tensor.Repeat(a, 3) {
		fmt.Println(r)
	}

	// Output:
	// [[1 2 3] [4 5 6]]
	// [[1 2 3] [4 5 6]]
	// [[1 2 3] [4 5 6]]
}

func ExampleFlatten() {
	xs := []matrix.Matrix{
		{
			{0, 1, 2},
			{0, 1, 2},
		},
		{
			{3, 4, 5},
			{3, 4, 5},
		},
	}

	fmt.Println(tensor.Flatten(xs))

	// Output:
	// [0 1 2 0 1 2 3 4 5 3 4 5]
}

func ExampleArgmax() {
	xs := []matrix.Matrix{{{0, 1, 2}, {0, 1, 2}}, {{3, 10, 5}, {3, 4, 5}}}
	fmt.Println(tensor.Argmax(xs))

	// Output:
	// 7
}
