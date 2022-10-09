package optimizer

type Adam struct {
	LearningRate float64
}

func (a *Adam) Update(params, grads *map[string][]float64) {
	// TODO
}
