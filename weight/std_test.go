package weight_test

import (
	"fmt"

	"github.com/itsubaki/neu/weight"
)

func ExampleStd() {
	fmt.Println(weight.Std(0.1)(0))
	fmt.Println(weight.Std(0.01)(0))
	fmt.Println(weight.Std(0.001)(0))

	// Output:
	// 0.1
	// 0.01
	// 0.001

}
