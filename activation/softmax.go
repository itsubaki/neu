package activation

import "math"

func Softmax(a []float64) []float64 {
	var max float64
	for i := range a {
		if a[i] > max {
			max = a[i]
		}
	}

	expa := make([]float64, 0)
	for i := range a {
		expa = append(expa, math.Exp(a[i]-max))
	}

	var sum float64
	for i := range expa {
		sum = sum + expa[i]
	}

	out := make([]float64, 0)
	for i := range expa {
		out = append(out, expa[i]/sum)
	}

	return out
}
