package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/itsubaki/neu/dataset/ptb"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func main() {
	// flag
	var dir, start string
	var length int
	flag.StringVar(&dir, "dir", "./testdata", "")
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
