package activation_test

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
)

func ExampleSoftmax() {
	y := activation.Softmax([]float64{0.3, 2.9, 4.0})
	fmt.Println(y)

	var sum float64
	for i := range y {
		sum = sum + y[i]
	}
	fmt.Println(sum)

	// Output:
	// [0.01821127329554753 0.24519181293507386 0.7365969137693786]
	// 1

}
