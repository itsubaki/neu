package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleReLU() {
	fmt.Println(activation.ReLU(-1.0))
	fmt.Println(activation.ReLU(-0.1))
	fmt.Println(activation.ReLU(0.0))
	fmt.Println(activation.ReLU(0.1))
	fmt.Println(activation.ReLU(1.0))

	// Output:
	// 0
	// 0
	// 0
	// 0.1
	// 1

}

func ExampleReLU_plot() {
	x := plot.Range(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		y = append(y, activation.ReLU(xi))
	}

	if err := plot.Save(x, y, "ExampleReLU.png"); err != nil {
		panic(err)
	}

	// Output:

}
