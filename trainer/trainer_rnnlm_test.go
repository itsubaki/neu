package trainer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
)

var _ trainer.RNNLM = (*TestRNNLM)(nil)

type TestRNNLM struct{}

func (m *TestRNNLM) Predict(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	return []matrix.Matrix{matrix.New()}
}

func (m *TestRNNLM) Forward(xs, ts []matrix.Matrix) matrix.Matrix {
	return matrix.New([]float64{1})
}

func (m *TestRNNLM) Backward() []matrix.Matrix { return []matrix.Matrix{matrix.New()} }
func (m *TestRNNLM) Layers() []model.TimeLayer { return make([]model.TimeLayer, 0) }
func (m *TestRNNLM) Params() [][]matrix.Matrix { return [][]matrix.Matrix{} }
func (m *TestRNNLM) Grads() [][]matrix.Matrix  { return [][]matrix.Matrix{} }

func (m *TestRNNLM) SetParams(p [][]matrix.Matrix) {
	for i, l := range m.Layers() {
		l.SetParams(p[i]...)
	}
}

func ExampleRNNLMTrainer() {
	tr := trainer.NewRNNLM(&TestRNNLM{}, &optimizer.SGD{
		LearningRate: 0.1,
	})

	tr.Fit(&trainer.RNNLMInput{
		Train:      []int{0, 1, 2, 3, 4, 5},
		TrainLabel: []int{1, 2, 3, 4, 5, 6},
		Epochs:     3,
		BatchSize:  1,
		TimeSize:   2,
		Verbose: func(epoch, j int, perplexity float64, m trainer.RNNLM) {
			fmt.Printf("%d, %d: %T\n", epoch, j, m)
		},
	})

	// Output:
	// 0, 0: *trainer_test.TestRNNLM
	// 0, 1: *trainer_test.TestRNNLM
	// 0, 2: *trainer_test.TestRNNLM
	// 1, 0: *trainer_test.TestRNNLM
	// 1, 1: *trainer_test.TestRNNLM
	// 1, 2: *trainer_test.TestRNNLM
	// 2, 0: *trainer_test.TestRNNLM
	// 2, 1: *trainer_test.TestRNNLM
	// 2, 2: *trainer_test.TestRNNLM

}

func ExamplePerplexity() {
	fmt.Println(trainer.Perplexity(1.0, 2))
	fmt.Println(trainer.Perplexity(1.0, 1))

	// Output:
	// 1.6487212707001282
	// 2.718281828459045
}
