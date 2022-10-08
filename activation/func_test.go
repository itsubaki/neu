package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/plot"
)

func ExampleStep() {
	x := plot.XRange(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		v := activation.Step([]float64{xi})
		y = append(y, v[0])
	}

	if err := plot.Save(x, y, "ExampleStep.png"); err != nil {
		panic(err)
	}

	// Output:
}

func ExampleSigmoid() {
	x := plot.XRange(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		v := activation.Sigmoid([]float64{xi})
		y = append(y, v[0])
	}

	if err := plot.Save(x, y, "ExampleSigmoid.png"); err != nil {
		panic(err)
	}

	// Output:
}

func ExampleReLU() {
	x := plot.XRange(-6, 6, 0.1)
	y := make([]float64, 0)

	for _, xi := range x {
		v := activation.ReLU([]float64{xi})
		y = append(y, v[0])
	}

	if err := plot.Save(x, y, "ExampleReLU.png"); err != nil {
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

func ExampleIdentity() {
	fmt.Println(activation.Identity([]float64{1, 2, 3, 4, 5}))

	// Output:
	// [1 2 3 4 5]
}
