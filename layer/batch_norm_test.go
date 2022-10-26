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
		Beta:     matrix.One(1, len(x[0])),
		Momentum: 0.9,
	}

	fmt.Println(n.Forward(x, nil, layer.Opts{Train: true}))
	fmt.Println(n.Forward(x, nil, layer.Opts{Train: false}))
	fmt.Println(n.Backward(matrix.New([]float64{1, 2})))

	// Output:
	// [[4.9999996254435075e-08 4.9999996254435075e-08] [1.9999999500000039 1.9999999500000039]]
	// [[3.529820863224588 6.375869334352251] [9.854373021286058 12.70042149241372]]
	// [[0.2500000124999972 0.5000000249999944]] []
}
