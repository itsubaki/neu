package activation_test

import (
	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleReLU() {
	x := plot.XRange(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		y = append(y, activation.ReLU(xi))
	}

	if err := plot.Save(x, y, "ExampleReLU.png"); err != nil {
		panic(err)
	}

	// Output:

}
