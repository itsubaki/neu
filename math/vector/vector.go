package vector

func Add(v, w []float64) []float64 {
	out := make([]float64, 0)
	for i := range v {
		out = append(out, v[i]+w[i])
	}

	return out
}
