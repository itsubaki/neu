package weight_test

import (
	"fmt"

	"github.com/itsubaki/neu/weight"
)

func ExampleHe() {
	fmt.Println(weight.He(1))
	fmt.Println(weight.He(2))
	fmt.Println(weight.He(4))

	// Output:
	// 1.4142135623730951
	// 1
	// 0.7071067811865476

}
