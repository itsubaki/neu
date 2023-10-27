package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
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
