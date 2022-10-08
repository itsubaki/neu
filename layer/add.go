package layer

type Add struct{}

func (l *Add) Forward(x, y []float64) []float64 {
	out := make([]float64, 0)
	for i := range x {
		out = append(out, x[i]+y[i])
	}

	return out
}

func (l *Add) Backwward(dout []float64) ([]float64, []float64) {
	// dx := dout * 1
	// dy := dout * 1
	return dout, dout
}
