package activation

import "math"

// Softmax returns the softmax of the input vector.
func Softmax(a []float64) []float64 {
	var max float64
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}

	expa := make([]float64, len(a))
	for i := range a {
		expa[i] = math.Exp(a[i] - max)
	}

	var sum float64
	for i := range expa {
		sum = sum + expa[i]
	}

	out := make([]float64, len(expa))
	for i := range expa {
		out[i] = expa[i] / sum
	}

	return out
}
