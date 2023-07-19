package numerical

// Diff returns the numerical differentiation of f at x.
func Diff(f func(x float64) float64, x []float64, h float64) []float64 {
	out := make([]float64, len(x))
	for i, xi := range x {
		out[i] = (f(xi+h) - f(xi-h)) / (2 * h)
	}

	return out
}
