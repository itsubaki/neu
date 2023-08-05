package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

func ExampleSplit() {
	x := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{1.1, 1.2, 1.3},
		[]float64{2.1, 2.2, 2.3},
	)

	xs, ts := model.Split(x)
	for _, v := range xs {
		fmt.Println(v)
	}

	for _, v := range ts {
		fmt.Println(v)
	}

	// Output:
	// [0.3]
	// [1.3]
	// [2.3]
	// [0.2 0.3]
	// [1.2 1.3]
	// [2.2 2.3]
}
