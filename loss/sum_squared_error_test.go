package loss_test

import (
	"fmt"

	"github.com/itsubaki/neu/loss"
)

func ExampleSumSquaredError() {
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC3%E5%88%B7%E3%81%BE%E3%81%A7
	// https://github.com/oreilly-japan/deep-learning-from-scratch/wiki/errata#%E7%AC%AC13%E5%88%B7%E3%81%BE%E3%81%A7

	t := []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	y1 := []float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0}
	y2 := []float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0}

	fmt.Println(loss.SumSquaredError(y1, t))
	fmt.Println(loss.SumSquaredError(y2, t))

	// Output:
	// 0.09750000000000003
	// 0.5974999999999999

}
