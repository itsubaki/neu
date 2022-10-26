package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleBatchNorm() {
	x := matrix.New([]float64{1.0, 2.0}, []float64{3.0, 4.0})

	n := &layer.BatchNorm{
		Gamma:    matrix.One(1, len(x[0])),
		Beta:     matrix.Zero(1, len(x[0])),
		Momentum: 0.9,
	}

	fmt.Println(n.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(n.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(n.Backward(x))

	// Output:
	// [[-0.9999999500000037 -0.9999999500000037] [0.9999999500000037 0.9999999500000037]]
	// [[2.529820863224588 5.375869334352251] [8.854373021286058 11.70042149241372]]
	// [[-9.999998495935358e-08 -9.999998518139819e-08] [9.999998495935358e-08 9.999998473730898e-08]] []
}
