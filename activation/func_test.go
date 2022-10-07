package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleStep() {
	xrange := plot.XRange(-6, 6, 0.1)
	yrange := make([]float64, 0)

	for _, x := range xrange {
		y := activation.Step([]float64{x})
		yrange = append(yrange, y[0])
	}

	if err := plot.Save(xrange, yrange, "ExampleStep.png"); err != nil {
		panic(err)
	}

	// Output:
}

func ExampleSigmoid() {
	xrange := plot.XRange(-6, 6, 0.1)
	yrange := make([]float64, 0)

	for _, x := range xrange {
		y := activation.Sigmoid([]float64{x})
		yrange = append(yrange, y[0])
	}

	if err := plot.Save(xrange, yrange, "ExampleSigmoid.png"); err != nil {
		panic(err)
	}

	// Output:
}

func ExampleReLU() {
	xrange := plot.XRange(-6, 6, 0.1)
	yrange := make([]float64, 0)

	for _, x := range xrange {
		y := activation.ReLU([]float64{x})
		yrange = append(yrange, y[0])
	}

	if err := plot.Save(xrange, yrange, "ExampleReLU.png"); err != nil {
		panic(err)
	}

	// Output:
}

func ExampleSoftmax() {
	y := activation.Softmax([]float64{0.3, 2.9, 4.0})
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
