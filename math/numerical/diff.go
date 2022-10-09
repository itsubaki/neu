package numerical

func Diff(f func(x float64) float64, x []float64, h float64) []float64 {
	out := make([]float64, 0)
	for _, xi := range x {
		out = append(out, (f(xi+h)-f(xi-h))/(2*h))
	}

	return out
}
