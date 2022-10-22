package weight_test

import (
	"fmt"

	"github.com/itsubaki/neu/weight"
)

func ExampleXavier() {
	fmt.Println(weight.Xavier(1))
	fmt.Println(weight.Xavier(2))
	fmt.Println(weight.Xavier(4))

	// Output:
	// 1
	// 0.7071067811865476
	// 0.5

}
