package loss

import "math"

func CrossEntropyError(y, t []float64) float64 {
	delta := 1e-7

	var sum float64
	for i := range y {
		sum = sum + (t[i] * math.Log(y[i]+delta))
	}

	return -1.0 * sum
}
