package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/dataset/sequence"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir string
	var epochs, batchSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 30, "")
	flag.IntVar(&batchSize, "batch-size", 128, "")
	flag.Parse()

	// data
	x, t, v := sequence.Must(sequence.Load(dir, sequence.Addition))

	// model
	m := model.NewSeq2Seq(&model.Seq2SeqConfig{
		VocabSize:   len(v.RuneToID), // 13
		WordVecSize: 16,
		HiddenSize:  128,
		WeightInit:  weight.Xavier,
	})

	// training
	tr := trainer.NewSeq2Seq(m, &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(5.0),
		},
	})

	tr.Fit(&trainer.Seq2SeqInput{
		Train:      x.Train,
		TrainLabel: t.Train,
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, loss float64, m *model.Seq2Seq) {
			if j%(len(x.Train)/batchSize/10) != 0 {
				return
			}

			var acc int
			for k := 0; k < len(x.Test); k++ {
				q, ans := trainer.Float64(x.Test)[k], t.Test[k] // (1, 7), (5)
				tq := trainer.Time(matrix.New(q))               // (7, 1, 1)
				guess := m.Generate(tq, ans[0], len(ans[1:]))

				acc += trainer.SeqAccuracy(ans, guess)
				fmt.Printf("%v %v, %v(%v)\n", v.ToString(x.Test[k]), v.ToString(ans), v.ToString(guess), guess)
			}

			fmt.Printf("%2d, %2d: loss=%.4f, acc=%.4f\n", epoch, j, loss, float64(acc)/10.0)
		},
	})
}
