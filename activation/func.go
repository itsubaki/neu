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

func Identity(x []float64) []float64 {
	return x
}
