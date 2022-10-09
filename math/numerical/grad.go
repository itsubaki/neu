package numerical

func Grad(f func(x ...float64) float64, x []float64) []float64 {
	h := 1e-4
	grad := make([]float64, 0)

	for i := range x {
		xi := x[i]

		// fxh1
		x[i] = xi + h
		fxh1 := f(x...)

		// fxh2
		x[i] = xi - h
		fxh2 := f(x...)

		// grad
		grad = append(grad, (fxh1-fxh2)/(2*h))
		x[i] = xi
	}

	return grad
}

func GradDescent(f func(x ...float64) float64, x []float64, learningRate float64, step int) []float64 {
	for i := 0; i < step; i++ {
		grad := Grad(f, x)

		for i := range grad {
			x[i] = x[i] - learningRate*grad[i]
		}
	}

	return x
}