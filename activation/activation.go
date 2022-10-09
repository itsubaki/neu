package activation

import "math"

func Step(x []float64) []float64 {
	out := make([]float64, 0, len(x))
	for i := range x {
		if x[i] > 0 {
			out = append(out, 1)
			continue
		}

		out = append(out, 0)
	}

	return out
}

func Sigmoid(x []float64) []float64 {
	out := make([]float64, 0, len(x))
	for i := range x {
		v := 1 / (1 + math.Exp(-x[i]))
		out = append(out, v)
	}

	return out
}

func ReLU(x []float64) []float64 {
	out := make([]float64, 0, len(x))
	for i := range x {
		v := math.Max(0, x[i])
		out = append(out, v)
	}

	return out
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

func Identity(x []float64) []float64 {
	return x
}
