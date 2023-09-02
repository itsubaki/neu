package numerical_test

import (
	"fmt"
	"math"

	"github.com/itsubaki/neu/math/numerical"
)

func ExampleGradient() {
	// p104
	f := func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2)
	}

	fmt.Println(numerical.Gradient(f, []float64{3, 4}))
	fmt.Println(numerical.Gradient(f, []float64{0, 2}))
	fmt.Println(numerical.Gradient(f, []float64{3, 0}))

	// Output:
	// [6.00000000000378 7.999999999999119]
	// [0 4.000000000004]
	// [6.000000000012662 0]

}

func ExampleGradientDescent() {
	// p108
	f := func(x ...float64) float64 {
		return math.Pow(x[0], 2) + math.Pow(x[1], 2)
	}

	fmt.Println(numerical.GradientDescent(f, []float64{-3, 4}, 0.1, 100))
	fmt.Println(numerical.GradientDescent(f, []float64{-3, 4}, 10, 100))
	fmt.Println(numerical.GradientDescent(f, []float64{-3, 4}, 1e-10, 100))

	// Output:
	// [-6.111107928998789e-10 8.148143905314271e-10]
	// [-2.5898374737328363e+13 -1.2952486168965398e+12]
	// [-2.999999939999995 3.9999999199999934]

}
