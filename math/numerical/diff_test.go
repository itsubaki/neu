package numerical_test

import (
	"math"

	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/plot"
)

func ExampleDiff() {
	f := func(x float64) float64 {
		return math.Pow(x, 3) - 2*math.Pow(x, 2) + 1
	}

	x := plot.XRange(0, 2, 1e-3)
	y := numerical.Diff(f, x, 1e-3)

	if err := plot.Save(x, y, "ExampleDiff.png"); err != nil {
		panic(err)
	}

	// Output:

}
