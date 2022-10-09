package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
)

func ExampleSigmoid() {
	x := []float64{0.0}
	sigmoid := layer.Sigmoid{}

	fmt.Println(sigmoid.Forward(x, []float64{}))
	fmt.Println(sigmoid.Backward(x))

	// Output:
	// [0.5]
	// [0] []
}
