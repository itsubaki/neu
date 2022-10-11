package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMask() {
	x := matrix.New([]float64{0, 1}, []float64{2, 3})
	mask := layer.Mask(x, func(x float64) bool { return x > 2 })

	for _, r := range x.Mul(mask) {
		fmt.Println(r)
	}

	// Output:
	// [0 1]
	// [2 0]
}
