package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
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
