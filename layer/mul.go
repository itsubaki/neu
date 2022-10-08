package layer

type Mul struct {
	x []float64
	y []float64
}

func (l *Mul) Forward(x, y []float64) []float64 {
	l.x, l.y = x, y

	out := make([]float64, 0)
	for i := range x {
		out = append(out, x[i]*y[i])
	}

	return out
}

func (l *Mul) Backward(dout []float64) ([]float64, []float64) {
	dx, dy := make([]float64, 0), make([]float64, 0)

	for i := range dout {
		dx = append(dx, dout[i]*l.y[i])
		dy = append(dy, dout[i]*l.x[i])
	}

	return dx, dy
}
