package trainer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func ExampleSeq2SeqTrainer() {
	tr := trainer.NewSeq2Seq(
		model.NewSeq2Seq(&model.RNNLMConfig{
			VocabSize:   13,
			WordVecSize: 16,
			HiddenSize:  128,
			WeightInit:  weight.Xavier,
		}), &optimizer.Adam{
			Alpha: 0.001,
			Beta1: 0.9,
			Beta2: 0.999,
		})

	tr.Fit(&trainer.Seq2SeqInput{
		Train:      [][]int{{0, 1, 2, 3, 4, 5}},
		TrainLabel: [][]int{{1, 2, 3, 4, 5, 6}},
		Epochs:     3,
		BatchSize:  1,
		Verbose: func(epoch, j int, loss float64, m trainer.Seq2Seq) {
			fmt.Printf("%d: %T\n", epoch, m)
		},
	})

	// Output:
	// 0: *model.Seq2Seq
	// 1: *model.Seq2Seq
	// 2: *model.Seq2Seq
}

func ExampleSeq2SeqTrainer_rand() {
	tr := trainer.NewSeq2Seq(
		model.NewSeq2Seq(&model.RNNLMConfig{
			VocabSize:   13,
			WordVecSize: 16,
			HiddenSize:  128,
			WeightInit:  weight.Xavier,
		}), &optimizer.Adam{
			Alpha: 0.001,
			Beta1: 0.9,
			Beta2: 0.999,
		})

	s := rand.Const(1)
	tr.Fit(&trainer.Seq2SeqInput{
		Train:      [][]int{{0, 1, 2, 3, 4, 5}},
		TrainLabel: [][]int{{1, 2, 3, 4, 5, 6}},
		Epochs:     3,
		BatchSize:  1,
		Verbose: func(epoch, j int, loss float64, m trainer.Seq2Seq) {
			fmt.Printf("%d: %T\n", epoch, m)
		},
	}, s)

	// Output:
	// 0: *model.Seq2Seq
	// 1: *model.Seq2Seq
	// 2: *model.Seq2Seq
}

func ExampleTime() {
	xs := matrix.New(
		// (N, T) (2, 3)
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	// (T, N, 1) (3, 2, 1)
	txs := trainer.Time(xs)
	for _, tx := range txs {
		fmt.Println(tx)
	}
	fmt.Println()

	// Output:
	// [[1] [4]]
	// [[2] [5]]
	// [[3] [6]]
}
