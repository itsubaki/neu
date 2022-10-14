package mnist_test

import (
	"fmt"

	"github.com/itsubaki/neu/mnist"
)

func ExampleLoad() {
	train, test := mnist.Must(mnist.Load("../testdata"))

	fmt.Println(train.N)
	fmt.Println(test.N)

	fmt.Println(len(train.Image[0]))
	fmt.Println(len(mnist.OneHot(train.Label)[0]))

	// Output:
	// 60000
	// 10000
	// 784
	// 10

}

func ExampleNormalize() {
	img := []mnist.Image{{byte(10)}, {byte(20)}, {byte(255)}}
	for _, r := range mnist.Normalize(img) {
		fmt.Printf("%.4f\n", r[0])
	}

	// Output:
	// 0.0392
	// 0.0784
	// 1.0000
}

func ExampleOneHot() {
	label := []mnist.Label{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, r := range mnist.OneHot(label) {
		fmt.Printf("%.1f\n", r)
	}

	// Output:
	// [1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
	// [0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
	// [0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
	// [0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0]
	// [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]
	// [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0]
	// [0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0]
	// [0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0]
	// [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0]
	// [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0]
}
