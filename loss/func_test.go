package loss_test

import (
	"fmt"

	"github.com/itsubaki/neu/loss"
)

func ExampleMeanSquaredError() {
	t := []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	y1 := []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	y2 := []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0}

	fmt.Println(loss.MeanSquaredError(y1, t))
	fmt.Println(loss.MeanSquaredError(y2, t))

	// Output:
	// 0.09750000000000003
	// 0.7224999999999999
}

func ExampleCrossEntropyError() {
	t := []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	y1 := []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	y2 := []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0}

	fmt.Println(loss.CrossEntropyError(y1, t))
	fmt.Println(loss.CrossEntropyError(y2, t))

	// Output:
	// 0.510825457099338
	// 2.302584092994546
}
