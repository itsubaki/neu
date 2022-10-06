package godeep_test

import (
	"fmt"

	"github.com/itsubaki/godeep"
	"github.com/itsubaki/godeep/activation"
	"github.com/itsubaki/godeep/matrix"
)

func Example() {
	X := matrix.New([]float64{1.0, 0.5})
	W1 := matrix.New(
		[]float64{0.1, 0.3, 0.5},
		[]float64{0.2, 0.4, 0.6},
	)
	B1 := matrix.New([]float64{0.1, 0.2, 0.3})

	A1 := matrix.Dot(X, W1).Add(B1)
	Z1 := activation.Sigmoid(A1[0])

	fmt.Println(A1[0])
	fmt.Println(Z1)

	W2 := matrix.New(
		[]float64{0.1, 0.4},
		[]float64{0.2, 0.5},
		[]float64{0.3, 0.6},
	)
	B2 := matrix.New([]float64{0.1, 0.2})

	A2 := matrix.Dot(matrix.New(Z1), W2).Add(B2)
	Z2 := activation.Sigmoid(A2[0])

	fmt.Println(A2[0])
	fmt.Println(Z2)

	W3 := matrix.New(
		[]float64{0.1, 0.3},
		[]float64{0.2, 0.4},
	)
	B3 := matrix.New([]float64{0.1, 0.2})

	A3 := matrix.Dot(matrix.New(Z2), W3).Add(B3)
	Y := activation.Identity(A3[0])

	fmt.Println(A3[0])
	fmt.Println(Y)

	// Output:
	// [0.30000000000000004 0.7 1.1]
	// [0.574442516811659 0.6681877721681662 0.7502601055951177]
	// [0.5161598377933344 1.2140269561658172]
	// [0.6262493703990729 0.7710106968556123]
	// [0.3168270764110298 0.6962790898619668]
	// [0.3168270764110298 0.6962790898619668]

}

func ExampleSoftmax() {
	y := godeep.Softmax([]float64{0.3, 2.9, 4.0})
	fmt.Println(y)

	var sum float64
	for i := range y {
		sum = sum + y[i]
	}
	fmt.Println(sum)

	// Output:
	// [0.01821127329554753 0.24519181293507386 0.7365969137693786]
	// 1
}
