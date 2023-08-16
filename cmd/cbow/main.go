package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
)

func main() {
	// flags
	var epochs int
	flag.IntVar(&epochs, "epochs", 1000, "")

	text := "You say goodbye and I say hello ."
	corpus, id2w, w2id := ptb.PreProcess(text)
	fmt.Println(corpus)
	fmt.Println(id2w)
	fmt.Println(w2id)
	fmt.Println()

	c, t := ptb.CreateContextsTarget(corpus, 1)
	for i := range c {
		fmt.Printf("%v: %v\n", c[i], t[i])
	}
	fmt.Println()

	m := model.NewCBOWNegs(model.CBOWNegsConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  7,
			HiddenSize: 5,
		},
		Corpus:     []int{0, 1, 2, 3, 4, 1, 5, 6},
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}
	fmt.Println()

	// training
	tr := trainer.New(m, &optimizer.Adam{
		LearningRate: 0.001,
		Beta1:        0.9,
		Beta2:        0.999,
	})

	now := time.Now()
	tr.Fit(&trainer.Input{
		Train:      matrix.From(c),
		TrainLabel: matrix.From([][]int{t}).T(),
		Epochs:     epochs,
		BatchSize:  1,
		Verbose: func(epoch, j int, loss float64, m trainer.Model) {
			fmt.Println(loss)
		},
	})
	fmt.Printf("elapsed=%v\n", time.Since(now))

	Win := m.Embedding[0].Params()[0]
	for id, word := range id2w {
		fmt.Printf("%v: %.4f\n", word, Win[id])
	}
	fmt.Println()
}
