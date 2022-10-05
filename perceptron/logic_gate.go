package perceptron

func LogicGate(x, w []float64, b float64) int {
	var sum float64
	for i := range x {
		sum = sum + x[i]*w[i]
	}

	v := sum + b
	if v <= 0 {
		return 0
	}

	return 1
}

func AND(x []float64) int {
	return LogicGate(x, []float64{0.5, 0.5}, -0.7)
}

func NAND(x []float64) int {
	return LogicGate(x, []float64{-0.5, -0.5}, 0.7)
}

func OR(x []float64) int {
	return LogicGate(x, []float64{0.5, 0.5}, -0.2)
}
