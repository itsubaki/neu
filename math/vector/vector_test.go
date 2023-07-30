package vector_test

import (
	"fmt"

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
