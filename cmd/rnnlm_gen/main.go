package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/optimizer/hook"
	"github.com/itsubaki/neu/trainer"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir, start string
	var epochs, length int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&epochs, "epochs", 10, "")
	flag.StringVar(&start, "start", "you", "")
	flag.IntVar(&length, "length", 100, "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))

	// model
	m := model.NewRNNLMGen(&model.RNNLMConfig{
		VocabSize:   vector.Max(train.Corpus) + 1,
		WordVecSize: 100,
		HiddenSize:  100,
		WeightInit:  weight.Xavier,
	})

	// layer
	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// train
	tr := trainer.NewRNNLM(m, &optimizer.SGD{
		LearningRate: 20,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(0.25),
		},
	})

	tr.Fit(&trainer.RNNLMInput{
		Train:      train.Corpus[:len(train.Corpus)-1],
		TrainLabel: train.Corpus[1:],
		Epochs:     epochs,
		BatchSize:  20,
		TimeSize:   35,
		Verbose: func(epoch, j int, perplexity float64, m trainer.RNNLM) {
			fmt.Printf("%2d, %2d: ppl=%.04f\n", epoch, j, perplexity)
		},
	})

	// generate
	startID := train.WordToID[start]
	skipIDs := []int{train.WordToID["N"], train.WordToID["<unk>"], train.WordToID["$"]}

	wordIDs := m.Generate(startID, skipIDs, length)
	words := make([]string, length)
	for i, id := range wordIDs {
		words[i] = train.IDToWord[id]
	}

	txt := strings.ReplaceAll(strings.Join(words, " "), "<eos>", "\n")
	fmt.Println(txt)
}
