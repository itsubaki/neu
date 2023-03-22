package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleTanh() {
	fmt.Println(activation.Tanh(-1e+7))
	fmt.Println(activation.Tanh(0.0))
	fmt.Println(activation.Tanh(1e+7))

	// Output:
	// -1
	// 0
	// 1

}

func ExampleTanh_step() {
	x := plot.Range(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		y = append(y, activation.Tanh(xi))
	}

	if err := plot.Save(x, y, "ExampleTanh.png"); err != nil {
		panic(err)
	}

	// Output:

}
