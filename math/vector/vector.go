package vector

func Add(v, w []float64) []float64 {
	out := make([]float64, 0)
	for i := range v {
		out = append(out, v[i]+w[i])
	}

	return out
}

func Int(v []float64) []int {
	out := make([]int, 0)
	for _, i := range v {
		out = append(out, int(i))
	}

	return out
}
