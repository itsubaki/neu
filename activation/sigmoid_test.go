package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleSigmoid() {
	fmt.Println(activation.Sigmoid(-1e+7))
	fmt.Println(activation.Sigmoid(0.0))
	fmt.Println(activation.Sigmoid(1e+7))

	// Output:
	// 0
	// 0.5
	// 1

}

func ExampleSigmoid_plot() {
	x := plot.XRange(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		y = append(y, activation.Sigmoid(xi))
	}

	if err := plot.Save(x, y, "ExampleSigmoid.png"); err != nil {
		panic(err)
	}

	// Output:

}
