package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
)

func ExampleSigmoid() {
	fmt.Println(activation.Sigmoid(-1e+7))
	fmt.Println(activation.Sigmoid(0.0))
	fmt.Println(activation.Sigmoid(1e+7))

	// Output:
	// 0
	// 0.5
	// 1

}
