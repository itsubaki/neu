package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
)

func ExampleIdentity() {
	fmt.Println(activation.Identity(100))

	// Output:
	// 100
}
