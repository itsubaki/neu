package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleStep() {
	fmt.Println(activation.Step(-1.0))
	fmt.Println(activation.Step(-0.1))
	fmt.Println(activation.Step(0.0))
	fmt.Println(activation.Step(0.1))
	fmt.Println(activation.Step(1.0))

	// Output:
	// 0
	// 0
	// 0
	// 1
	// 1

}

func ExampleStep_plot() {
	x := plot.Range(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		y = append(y, activation.Step(xi))
	}

	if err := plot.Save(x, y, "ExampleStep.png"); err != nil {
		panic(err)
	}

	// Output:

}
