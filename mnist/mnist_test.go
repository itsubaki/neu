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
	oh := mnist.OneHot(label)
	for _, r := range oh {
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
