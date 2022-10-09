package activation

import "math"

func Step(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

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
