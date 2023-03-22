package loss

import "math"

// CrossEntropyError returns the cross entropy error.
func CrossEntropyError(y, t []float64) float64 {
	var sum float64
	for i := range y {
		sum = sum + (t[i] * math.Log(y[i]+1e-7))
	}

	return -1.0 * sum
}
