package vector_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/vector"
)

func ExampleAdd() {
	v := []float64{1, 2, 3}
	w := []float64{4, 5, 6}

	fmt.Println(vector.Add(v, w))

	// Output:
	// [5 7 9]
}

func ExampleInt() {
	v := []float64{1, 2, 3}
	fmt.Println(vector.Int(v))

	// Output:
	// [1 2 3]
}

func ExampleMax() {
	v := []int{1, 2, 3}
	fmt.Println(vector.Max(v))

	// Output:
	// 3
}

func ExampleAbs() {
	v := []float64{1, -2, -3}
	fmt.Println(vector.Abs(v))

	// Output:
	// [1 2 3]
}

func ExampleContains() {
	fmt.Println(vector.Contains(3, []int{1, 2, 3}))
	fmt.Println(vector.Contains(0, []int{1, 2, 3}))

	// Output:
	// true
	// false
}

func ExampleTranspose() {
	fmt.Println(vector.Transpose([]int{1, 2, 3}))
	fmt.Println(vector.T([]int{1, 2, 3}))

	// Output:
	// [[1] [2] [3]]
	// [[1] [2] [3]]
}

func ExampleChoice() {
	s := rand.NewSource(1)
	p := []float64{0.1, 0.2, 0.3, 0.4}

	for i := 0; i < 10; i++ {
		fmt.Print(vector.Choice(p, s))
	}

	// Output:
	// 3332230102
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

	s := rand.NewSource(1234)
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(vector.Shuffle(x, t, s))
	fmt.Println(x, t)

	fmt.Println(vector.Shuffle([][]float64{{0}}, [][]float64{{1}}))
	fmt.Println(vector.Shuffle([][]int{{1, 2, 3}, {4, 5, 6}}, [][]int{{7, 8, 9}, {10, 11, 12}}, rand.NewSource(2)))

	// Output:
	// [[0 4] [0 2] [0 3] [0 1]] [[4 0] [2 0] [3 0] [1 0]]
	// [[0 2] [0 3] [0 1] [0 4]] [[2 0] [3 0] [1 0] [4 0]]
	// [[0 2] [0 4] [0 3] [0 1]] [[2 0] [4 0] [3 0] [1 0]]
	// [[0 1] [0 2] [0 3] [0 4]] [[1 0] [2 0] [3 0] [4 0]]
	// [[0]] [[1]]
	// [[4 5 6] [1 2 3]] [[10 11 12] [7 8 9]]

}

func ExampleReverse() {
	fmt.Println(vector.Reverse([]int{1, 2, 3, 4, 5, 6, 7, 8, 9}))

	// Output:
	// [9 8 7 6 5 4 3 2 1]
}
