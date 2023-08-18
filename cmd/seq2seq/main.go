package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/neu/dataset/sequence"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir string
	var epochs, dataSize, wordvecSize, hiddenSize, batchSize int
	var learningRate, beta1, beta2, max float64
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&dataSize, "data-size", -1, "")
	flag.IntVar(&wordvecSize, "wordvec-size", 64, "")
	flag.IntVar(&hiddenSize, "hidden-size", 128, "")
	flag.IntVar(&batchSize, "batch-size", 2, "")
	flag.Float64Var(&learningRate, "learning-rate", 0.001, "")
	flag.Float64Var(&beta1, "beta1", 0.9, "")
	flag.Float64Var(&beta2, "beta2", 0.999, "")
	flag.Float64Var(&max, "grads-cliping-max", 5.0, "")
	flag.Parse()

	// data
	x, t, v := sequence.Must(sequence.Load(dir, sequence.AdditionTxt))
	xt, tt := x.Train, t.Train
	xv, tv := x.Test, t.Test
	if dataSize > 0 {
		xt, tt = x.Train[:dataSize], t.Train[:dataSize]
		xv, tv = x.Test[:dataSize], t.Test[:dataSize]
	}

	// model
	m := model.NewPeekySeq2Seq(&model.RNNLMConfig{
		VocabSize:   len(v.RuneToID), // 13
		WordVecSize: wordvecSize,
		HiddenSize:  hiddenSize,
		WeightInit:  weight.Xavier,
	})

	// summary
	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.NewSeq2Seq(m, &optimizer.Adam{
		LearningRate: learningRate,
		Beta1:        beta1,
		Beta2:        beta2,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(max),
		},
	})

	now := time.Now()
	tr.Fit(&trainer.Seq2SeqInput{
		Train:      xt,
		TrainLabel: tt,
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, loss float64, m trainer.Seq2Seq) {
			if epoch%20 != 0 || j != 0 {
				return
			}

			tacc := generate(xt, tt, m, v, 5)
			vacc := generate(xv, tv, m, v, 5)
			fmt.Printf("%2d, %2d: loss=%.04f, train_acc=%.4f, test_acc=%.4f\n", epoch, j, loss, tacc, vacc)
			fmt.Println()
		},
	})

	fmt.Printf("elapsed=%v\n", time.Since(now))
}

func generate(x, t [][]int, m trainer.Seq2Seq, v *sequence.Vocab, top int) float64 {
	xs, ts := vector.Shuffle(x, t)
	xm := matrix.From(xs)

	var acc int
	for k := 0; k < top; k++ {
		q, correct := xm[k], ts[k]                        // (1, 7), (5)
		tq := vector.Reverse(trainer.Time(matrix.New(q))) // (7, 1, 1)

		guess := m.Generate(tq, correct[0], len(correct[1:]))
		if vector.Equals(correct[1:], guess) {
			acc++
		}

		fmt.Printf("%v %v; %v\n", v.ToString(xs[k]), v.ToString(correct), v.ToString(guess))
	}

	return float64(acc) / float64(top)
}
