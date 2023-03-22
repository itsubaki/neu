package loss

import "math"

// SumSquaredError returns the sum squared error.
func SumSquaredError(y, t []float64) float64 {
	var sum float64
	for i := range y {
		sum = sum + math.Pow((y[i]-t[i]), 2)
	}

	return 0.5 * sum
}
