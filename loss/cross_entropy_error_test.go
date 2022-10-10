package loss_test

import (
	"fmt"

	"github.com/itsubaki/neu/loss"
)

func ExampleCrossEntropyError() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC3%E5%88%B7%E3%81%BE%E3%81%A7

	t := []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	y1 := []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	y2 := []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0}

	fmt.Println(loss.CrossEntropyError(y1, t))
	fmt.Println(loss.CrossEntropyError(y2, t))

	// Output:
	// 0.510825457099338
	// 2.302584092994546

}
