package diff_test

import (
	"fmt"
	"math"

	"github.com/itsubaki/neu/math/diff"
	"github.com/itsubaki/neu/plot"
)

func ExampleDiff() {
	f := func(x float64) float64 {
		return math.Pow(x, 3) - 2*math.Pow(x, 2) + 1
	}

	xrange := plot.XRange(0, 2, 1e-3)
	yrange := diff.Diff(f, xrange, 1e-3)

	if err := plot.Save(xrange, yrange, "ExampleDiff.png"); err != nil {
		panic(err)
	}

	// Output:

}

func ExampleGrad() {
	f := func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2)
	}

	fmt.Println(diff.Grad(f, []float64{3, 4}))
	fmt.Println(diff.Grad(f, []float64{0, 2}))
	fmt.Println(diff.Grad(f, []float64{3, 0}))

	// Output:
	// [6.00000000000378 7.999999999999119]
	// [0 4.000000000004]
	// [6.000000000012662 0]
}

func ExampleGradDescent() {
	f := func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2)
	}

	fmt.Println(diff.GradDescent(f, []float64{-3, 4}, 0.1, 100))
	fmt.Println(diff.GradDescent(f, []float64{-3, 4}, 10, 100))
	fmt.Println(diff.GradDescent(f, []float64{-3, 4}, 1e-10, 100))

	// Output:
	// [-6.111107928998789e-10 8.148143905314271e-10]
	// [-2.5898374737328363e+13 -1.2952486168965398e+12]
	// [-2.999999939999995 3.9999999199999934]
}
