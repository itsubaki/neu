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

	tr.Fit(&trainer.Seq2SeqInput{
		Train:      x.Train,
		TrainLabel: t.Train,
		Epochs:     epochs,
		BatchSize:  batchSize,
		Verbose: func(epoch, j int, loss float64, m *model.Seq2Seq) {
			fmt.Printf("%2d, %2d: loss=%.04f\n", epoch, j, loss)
		},
	})

	generate(x.Test, t.Test, m, v, 100)
}

func generate(xs, ts [][]int, m *model.Seq2Seq, v *sequence.Vocab, top int) {
	for k := 0; k < top; k++ {
		q, ans := trainer.Float64(xs)[k], ts[k] // (1, 7), (5)
		tq := trainer.Time(matrix.New(q))       // (7, 1, 1)
		guess := m.Generate(tq, ans[0], len(ans[1:]))

		fmt.Printf("%v %v, %v (%v)\n", v.ToString(xs[k]), v.ToString(ans), v.ToString(guess), guess)
	}
}
