package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleMask() {
	x := matrix.New([]float64{0, 1}, []float64{2, 3})
	mask := [][]bool{{true, false}, {true, false}}

	for _, r := range layer.Mask(x, mask) {
		fmt.Println(r)
	}

	// Output:
	// [0 1]
	// [0 3]
}
