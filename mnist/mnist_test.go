package mnist_test

import (
	"fmt"
	"testing"

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

func ExampleLoad_notfound() {
	_, _, err := mnist.Load("invalid_dir")
	fmt.Println(err)

	// Output:
	// load training data: load=invalid_dir/train-images-idx3-ubyte.gz: file=invalid_dir/train-images-idx3-ubyte.gz open: open invalid_dir/train-images-idx3-ubyte.gz: no such file or directory
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

func TestMust(t *testing.T) {
	defer func() {
		if rec := recover(); rec != nil {
			err, ok := rec.(error)
			if !ok {
				t.Fail()
			}

			if err.Error() != "something went wrong" {
				t.Fail()
			}
		}
	}()

	mnist.Must(nil, nil, fmt.Errorf("something went wrong"))
	t.Fail()
}
