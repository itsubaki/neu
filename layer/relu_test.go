package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleMask() {
	x := []float64{1.0, -0.5, -2.0, 3.0}
	fmt.Println(layer.Mask(x))

	// Output:
	// [false true true false]

}

func ExampleReLU() {
	x := []float64{1.0, -0.5, -2.0, 3.0}
	relu := layer.ReLU{}

	fmt.Println(relu.Forward(x, []float64{}))
	fmt.Println(relu.Backward(x))

	// Output:
	// [1 0 0 3]
	// [1 0 0 3] []
}
