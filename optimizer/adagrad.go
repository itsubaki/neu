package optimizer

type AdaGrad struct {
	LearningRate float64
}

func (g *AdaGrad) Update(params, grads *map[string][]float64) {
	// TODO
}
