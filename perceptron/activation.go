package perceptron

import "math"

func Step(x float64) float64 {
	if x > 0 {
		return 1
	}

	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}
