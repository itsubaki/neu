package optimizer

type SGD struct {
	LearningRate float64
}

func (d *SGD) Update(params, grads *map[string][]float64) {
	// TODO
}
