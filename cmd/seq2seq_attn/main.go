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
	var epochs, dataSize, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&dataSize, "data-size", 10, "")
	flag.IntVar(&batchSize, "batch-size", 2, "")
	flag.Parse()

	// data
	x, t, v := sequence.Must(sequence.Load(dir, sequence.DateTxt))

	// model
	m := model.NewAttentionSeq2Seq(&model.RNNLMConfig{
		VocabSize:   len(v.RuneToID),
		WordVecSize: 64,
		HiddenSize:  128,
		WeightInit:  weight.Xavier,
	})

	// layer
	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// training
	tr := trainer.NewSeq2Seq(m, &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(5.0),
		},
	})

	now := time.Now()
	xt, tt := x.Train[:dataSize], t.Train[:dataSize]
	xv, tv := x.Test[:dataSize], t.Test[:dataSize]
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

	var acc int
	for k := 0; k < top; k++ {
		q, correct := matrix.From(xs)[k], ts[k]
		tq := vector.Reverse(trainer.Time(matrix.New(q)))
		guess := m.Generate(tq, correct[0], len(correct[1:]))

		acc += trainer.SeqAccuracy(correct[1:], guess)
		fmt.Printf("%v %v; %v\n", v.ToString(xs[k]), v.ToString(correct), v.ToString(guess))
	}

	return float64(acc) / float64(top)
}
