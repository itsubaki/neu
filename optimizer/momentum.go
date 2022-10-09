package optimizer

type Momentum struct {
	LearningRate float64
	Momentum     float64
}

func (m *Momentum) Update(params, grads *map[string][]float64) {
	// TODO
}
