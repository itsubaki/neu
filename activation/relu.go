package activation

import "math"

func ReLU(x float64) float64 {
	return math.Max(0, x)
}
