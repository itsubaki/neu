package activation

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
