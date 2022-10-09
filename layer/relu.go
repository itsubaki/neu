package layer

type ReLU struct {
	mask []bool
}

func (l *ReLU) Forward(x, _ []float64) []float64 {
	l.mask = mask(x)

	out := make([]float64, 0)
	for i := range x {
		if l.mask[i] {
			out = append(out, 0)
			continue
		}

		out = append(out, x[i])
	}

	return out
}

func (l *ReLU) Backward(dout []float64) ([]float64, []float64) {
	dx := make([]float64, 0)
	for i := range dout {
		if l.mask[i] {
			dx = append(dx, 0)
			continue
		}

		dx = append(dx, dout[i])
	}

	return dx, []float64{}
}

func mask(x []float64) []bool {
	out := make([]bool, 0)
	for i := range x {
		out = append(out, x[i] <= 0)
	}

	return out
}
