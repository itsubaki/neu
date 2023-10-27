package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
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
