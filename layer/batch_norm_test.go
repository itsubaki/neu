package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleBatchNorm() {
	x := matrix.New([]float64{1.0, 1.0}, []float64{2.0, 2.0})

	n := &layer.BatchNorm{
		Gamma:    matrix.One(1, len(x[0])),
		Beta:     matrix.Zero(1, len(x[0])),
		Momentum: 0.9,
	}

	fmt.Println(n.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(n.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(n.Backward(x))
	fmt.Println(n.DGamma)
	fmt.Println(n.DBeta)

	// Output:
	// [[-0.9999998000000601 -0.9999998000000601] [0.9999998000000601 0.9999998000000601]]
	// [[5.3758612705744575 5.3758612705744575] [11.700403941838523 11.700403941838523]]
	// [[-3.9999975998128434e-07 -3.9999975998128434e-07] [3.9999975953719513e-07 3.9999975953719513e-07]] []
	// [[0.9999998000000601 0.9999998000000601]]
	// [[3 3]]
}
