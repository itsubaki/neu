package trainer

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

type Model interface {
	Predict(x matrix.Matrix) matrix.Matrix
	Loss(x, t matrix.Matrix) matrix.Matrix
	Gradient(x, t matrix.Matrix) map[string]matrix.Matrix
	Optimize(grads map[string]matrix.Matrix)
}

func Train(m Model, x, t, xt, tt matrix.Matrix, iter, batchSize int, verbose bool) (float64, float64) {
	for i := 0; i < iter+1; i++ {
		// batch
		mask := Random(len(x), batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		// update
		grads := m.Gradient(xbatch, tbatch)
		m.Optimize(grads)

		if verbose && i%(iter/batchSize) == 0 {
			// train data
			loss := m.Loss(xbatch, tbatch)
			acc := model.Accuracy(m.Predict(xbatch), tbatch)

			// test data
			mask := Random(len(xt), batchSize)
			xtbatch := matrix.Batch(xt, mask)
			ttbatch := matrix.Batch(tt, mask)
			yt := m.Predict(xtbatch)
			tacc := model.Accuracy(yt, ttbatch)

			// print
			fmt.Printf("%4d: loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", i, loss, acc, tacc)
			fmt.Printf("predict: %v\n", yt.Argmax()[:20])
			fmt.Printf("label  : %v\n", ttbatch.Argmax()[:20])
			fmt.Println()
		}
	}

	return model.Accuracy(m.Predict(x), t), model.Accuracy(m.Predict(xt), tt)
}

func Random(trainSize, batchSize int) []int {
	tmp := make(map[int]bool)

	for c := 0; c < batchSize; {
		n := rand.Intn(trainSize)
		if _, ok := tmp[n]; !ok {
			tmp[n] = true
			c++
		}
	}

	out := make([]int, 0, len(tmp))
	for k := range tmp {
		out = append(out, k)
	}

	return out
}
