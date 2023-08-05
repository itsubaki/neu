package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flags
	var dir string
	var epochs, corpusSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 100, "")
	flag.IntVar(&corpusSize, "corpus-size", 1000, "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))
	corpus := train.Corpus[:corpusSize]

	// model
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   vector.Max(corpus) + 1,
			WordVecSize: 100,
			HiddenSize:  100,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	})

	// layer
	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// training
	tr := trainer.NewRNNLM(m, &optimizer.SGD{
		LearningRate: 20,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(0.25),
		},
	})

	tr.Fit(&trainer.RNNLMInput{
		Train:      corpus[:len(corpus)-1],
		TrainLabel: corpus[1:],
		Epochs:     epochs,
		BatchSize:  20,
		TimeSize:   35,
		Verbose: func(epoch int, perplexity float64, m trainer.RNNLM) {
			fmt.Printf("%2d: ppl=%.04f\n", epoch, perplexity)
		},
	})
}
