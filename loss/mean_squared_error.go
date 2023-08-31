package loss

import "math"

func MeanSquaredError(y, t []float64) float64 {
	var sum float64
	for i := range y {
		sum = sum + math.Pow(y[i]-t[i], 2)
	}

	return sum / float64(len(y))
}
