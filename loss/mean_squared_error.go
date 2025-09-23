package loss

func MeanSquaredError(y, t []float64) float64 {
	var sum float64
	for i := range y {
		sum = sum + (y[i]-t[i])*(y[i]-t[i])
	}

	return sum / float64(len(y))
}
