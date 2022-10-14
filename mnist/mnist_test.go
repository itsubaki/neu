package mnist_test

import (
	"fmt"
	"os"

	"github.com/itsubaki/neu/mnist"
)

func ExampleLoad() {
	train, test, err := mnist.Load("../testdata")
	if err != nil {
		fmt.Printf("load mnist: %v", err)
		os.Exit(1)
	}

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

func ExampleOneHot() {
	label := []mnist.Label{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, r := range mnist.OneHot(label) {
		fmt.Println(r)
	}

	// Output:
	// [1 0 0 0 0 0 0 0 0 0]
	// [0 1 0 0 0 0 0 0 0 0]
	// [0 0 1 0 0 0 0 0 0 0]
	// [0 0 0 1 0 0 0 0 0 0]
	// [0 0 0 0 1 0 0 0 0 0]
	// [0 0 0 0 0 1 0 0 0 0]
	// [0 0 0 0 0 0 1 0 0 0]
	// [0 0 0 0 0 0 0 1 0 0]
	// [0 0 0 0 0 0 0 0 1 0]
	// [0 0 0 0 0 0 0 0 0 1]
}

func ExampleImage2f64() {
	img := []mnist.Image{{byte(10)}, {byte(20)}}
	for _, r := range mnist.Image2f64(img) {
		fmt.Printf("%.1f\n", r[0])
	}

	// Output:
	// 10.0
	// 20.0
}

func ExampleOneHotLabel2f64() {
	label := []mnist.Label{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, r := range mnist.OneHotLabel2f64(mnist.OneHot(label)) {
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
