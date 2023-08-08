package trainer_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func ExampleSeq2SeqTrainer() {
	tr := trainer.NewSeq2Seq(
		model.NewSeq2Seq(&model.Seq2SeqConfig{
			VocabSize:   13,
			WordVecSize: 16,
			HiddenSize:  128,
			WeightInit:  weight.Xavier,
		}), &optimizer.Adam{
			LearningRate: 0.001,
			Beta1:        0.9,
			Beta2:        0.999,
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
		model.NewSeq2Seq(&model.Seq2SeqConfig{
			VocabSize:   13,
			WordVecSize: 16,
			HiddenSize:  128,
			WeightInit:  weight.Xavier,
		}), &optimizer.Adam{
			LearningRate: 0.001,
			Beta1:        0.9,
			Beta2:        0.999,
		})

	s := rand.NewSource(1)
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

func ExampleFloat64() {
	fmt.Printf("%.2f", trainer.Float64([][]int{
		{1, 2, 3},
		{4, 5, 6},
	}))

	// Output:
	// [[1.00 2.00 3.00] [4.00 5.00 6.00]]
}

func ExampleTime() {
	xs := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

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

func ExampleSeqAccuracy() {
	fmt.Println(trainer.SeqAccuracy([]int{1, 2, 3}, []int{1}))
	fmt.Println(trainer.SeqAccuracy([]int{1, 2, 3}, []int{4, 5, 6}))
	fmt.Println(trainer.SeqAccuracy([]int{1, 2, 3}, []int{1, 2, 3}))

	// Output:
	// 0
	// 0
	// 1
}
