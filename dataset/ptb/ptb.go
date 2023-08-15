package ptb

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

const (
	TrainTxt = "ptb.train.txt"
	TestTxt  = "ptb.test.txt"
	ValidTxt = "ptb.valid.txt"
)

type Dataset struct {
	Corpus   []int
	IDToWord map[int]string
	WordToID map[string]int
}

func Load(dir, fileName string) (*Dataset, error) {
	path := filepath.Clean(path.Join(dir, fileName))
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("open file=%v: %v", path, err)
	}

	corpus, id2w, w2id := PreProcess(string(bytes))
	return &Dataset{
		IDToWord: id2w,
		WordToID: w2id,
		Corpus:   corpus,
	}, nil
}

func PreProcess(text string) ([]int, map[int]string, map[string]int) {
	rep := strings.TrimSpace(strings.ReplaceAll(text, "\n", "<eos>"))
	words := strings.Split(rep, " ")

	id2w := make(map[int]string)
	w2id := make(map[string]int)

	for _, w := range words {
		if _, ok := w2id[w]; ok {
			continue
		}

		id := len(w2id)
		w2id[w] = id
		id2w[id] = w
	}

	corpus := make([]int, 0)
	for _, w := range words {
		corpus = append(corpus, w2id[w])
	}

	return corpus, id2w, w2id
}

func CreateContextsTarget(corpus []int, windowSize int) ([][]int, []int) {
	contexts := make([][]int, 0)
	target := corpus[windowSize : len(corpus)-windowSize]

	for i := windowSize; i < len(corpus)-windowSize; i++ {
		cs := make([]int, 0)
		for t := -windowSize; t < windowSize+1; t++ {
			if t == 0 {
				continue
			}

			cs = append(cs, corpus[i+t])
		}

		contexts = append(contexts, cs)
	}

	return contexts, target
}

func Must(dataset *Dataset, err error) *Dataset {
	if err != nil {
		panic(err)
	}

	return dataset
}
