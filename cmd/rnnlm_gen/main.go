package main

import (
	"flag"
	"fmt"
	"math"
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
	var length int
	var epochs, wordvecSize, hiddenSize, batchSize, timeSize int
	flag.StringVar(&dir, "dir", "./testdata", "")
	flag.IntVar(&length, "length", 100, "")
	flag.IntVar(&epochs, "epochs", 10, "")
	flag.IntVar(&wordvecSize, "wordvec-size", 650, "")
	flag.IntVar(&hiddenSize, "hidden-size", 650, "")
	flag.IntVar(&batchSize, "batch-size", 20, "")
	flag.IntVar(&timeSize, "time-size", 35, "")
	flag.StringVar(&start, "start", "you", "")
	flag.Parse()

	// data
	train := ptb.Must(ptb.Load(dir, ptb.TrainTxt))

	// model
	m := model.NewRNNLMGen(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   vector.Max(train.Corpus) + 1,
			WordVecSize: wordvecSize,
			HiddenSize:  hiddenSize,
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

	// params
	filename := fmt.Sprintf("%s/rnnlm_gen.gob", dir)
	if params, ok := model.Load(filename); ok {
		m.SetParams(params)
	}

	// train
	tr := trainer.NewRNNLM(m, &optimizer.SGD{
		LearningRate: 20,
		Hooks: []optimizer.Hook{
			hook.GradsClipping(0.25),
		},
	})

	min := math.MaxFloat64
	tr.Fit(&trainer.RNNLMInput{
		Train:      train.Corpus[:len(train.Corpus)-1],
		TrainLabel: train.Corpus[1:],
		Epochs:     epochs,
		BatchSize:  batchSize,
		TimeSize:   timeSize,
		Verbose: func(epoch, j int, perplexity float64, m trainer.RNNLM) {
			if perplexity < min {
				min = perplexity
				if err := model.Save(m.Params(), filename); err != nil {
					fmt.Println(err)
				}
			}

			fmt.Printf("%2d, %2d: ppl=%.04f\n", epoch, j, perplexity)
		},
	})

	// generate
	startID := train.WordToID[start]
	skipIDs := []int{
		train.WordToID["N"],
		train.WordToID["<unk>"],
		train.WordToID["$"],
	}

	wordIDs := m.Generate(startID, skipIDs, length)
	words := make([]string, length)
	for i, id := range wordIDs {
		words[i] = train.IDToWord[id]
	}

	txt := strings.ReplaceAll(strings.Join(words, " "), "\n", "")
	fmt.Println(start, txt)
}
