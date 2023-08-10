package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeSum() {
	hs := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
		},
	}

	sum := layer.TimeSum(hs)
	for _, r := range sum {
		fmt.Println(r)
	}

	// Output:
	// [2 4 6]
	// [8 10 12]
}
