package vector_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/vector"
)

func ExampleZero() {
	fmt.Println(vector.Zero(3))

	// Output:
	// [0 0 0]
}

func ExampleRand() {
	fmt.Println(vector.Rand(3, rand.Const(1)))
	fmt.Println(len(vector.Rand(10)))

	// Output:
	// [0.23842319087387442 0.50092138792625 0.04999911180706662]
	// 10
}

func ExampleRandn() {
	fmt.Println(vector.Randn(3, rand.Const(1)))
	fmt.Println(len(vector.Randn(10)))

	// Output:
	// [-0.8024826241110656 0.424707052949676 -0.4985070978632815]
	// 10
}

func ExampleArgmax() {
	v := []float64{-1, -2, -3}
	fmt.Println(vector.Argmax(v))

	// Output:
	// 0
}

func ExampleAdd() {
	v := []float64{1, 2, 3}
	w := []float64{4, 5, 6}

	fmt.Println(vector.Add(v, w))

	// Output:
	// [5 7 9]
}

func ExampleMul() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Mul(v, 2))

	// Output:
	// [2 4 6]
}

func ExampleDiv() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Div(v, 2))

	// Output:
	// [0.5 1 1.5]
}

func ExampleAbs() {
	v := []float64{1, -2, -3}
	fmt.Println(vector.Abs(v))

	// Output:
	// [1 2 3]
}

func ExampleSum() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Sum(v))

	// Output:
	// 6
}

func ExampleMean() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Mean(v))

	// Output:
	// 2
}

func ExampleMax() {
	v := []int{1, 2, 3}
	fmt.Println(vector.Max(v))

	// Output:
	// 3
}

func ExampleInt() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Int(v))

	// Output:
	// [1 2 3]
}

func ExamplePow2() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Pow2(v))

	// Output:
	// [1 4 9]
}

func ExampleCos() {
	x := []float64{1, 2, 3}
	y := []float64{1, 2, 4}
	fmt.Println(vector.Cos(x, y))

	// Output:
	// 0.9914601333935124
}

func ExampleContains() {
	fmt.Println(vector.Contains(3, []int{1, 2, 3}))
	fmt.Println(vector.Contains(0, []int{1, 2, 3}))

	// Output:
	// true
	// false
}

func ExampleT() {
	fmt.Println(vector.T([]int{1, 2, 3}))

	// Output:
	// [[1] [2] [3]]
}

func ExampleChoice() {
	s := rand.Const(1)
	p := []float64{0.1, 0.2, 0.3, 0.4}

	for i := 0; i < 10; i++ {
		fmt.Print(vector.Choice(p, s))
	}

	// Output:
	// 1202321031
}

func ExampleChoice_rand() {
	p := []float64{0.1, 0.2, 0.3, 0.4}
	if vector.Choice(p) < 0 {
		fmt.Println("invalid")
	}

	// Output:
}

func ExampleShuffle() {
	x := [][]float64{{0, 1}, {0, 2}, {0, 3}, {0, 4}}
	t := [][]float64{{1, 0}, {2, 0}, {3, 0}, {4, 0}}

	s := rand.Const(1234)
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(x, t)

	fmt.Println(vector.Shuffle([][]float64{{0}}, [][]float64{{1}}))
	fmt.Println(vector.Shuffle([][]int{{1, 2, 3}, {4, 5, 6}}, [][]int{{7, 8, 9}, {10, 11, 12}}, rand.Const(2)))

	// Output:
	// [[0 4] [0 3] [0 2] [0 1]] [[4 0] [3 0] [2 0] [1 0]]
	// [[0 3] [0 4] [0 2] [0 1]] [[3 0] [4 0] [2 0] [1 0]]
	// [[0 3] [0 2] [0 1] [0 4]] [[3 0] [2 0] [1 0] [4 0]]
	// [[0 1] [0 2] [0 3] [0 4]] [[1 0] [2 0] [3 0] [4 0]]
	// [[0]] [[1]]
	// [[4 5 6] [1 2 3]] [[10 11 12] [7 8 9]]

}

func ExampleReverse() {
	fmt.Println(vector.Reverse([]int{1, 2, 3, 4, 5, 6, 7, 8, 9}))

	// Output:
	// [9 8 7 6 5 4 3 2 1]
}

func ExampleMatchCount() {
	fmt.Println(vector.MatchCount(
		[]int{1, 2, 3, 4, 5},
		[]int{0, 2, 4, 3, 5},
	))

	// Output:
	// 2
}

func ExampleEquals() {
	fmt.Println(vector.Equals(
		[]int{1, 2, 3, 4, 5},
		[]int{1, 2, 3, 4, 5},
	))

	fmt.Println(vector.Equals(
		[]int{1, 2, 3, 4, 5},
		[]int{0, 2, 4, 3, 5},
	))

	fmt.Println(vector.Equals(
		[]int{1, 2, 3},
		[]int{0, 2, 4, 3, 5},
	))

	// Output:
	// true
	// false
	// false
}
