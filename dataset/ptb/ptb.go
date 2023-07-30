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
	WordToID map[string]int
	IDToWord map[int]string
	Corpus   []int
}

func Load(dir, fileName string) (*Dataset, error) {
	path := filepath.Clean(path.Join(dir, fileName))
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("open file=%v: %v", path, err)
	}

	corpus, w2id, id2w := preprocess(string(bytes))
	return &Dataset{
		WordToID: w2id,
		IDToWord: id2w,
		Corpus:   corpus,
	}, nil
}

func preprocess(text string) ([]int, map[string]int, map[int]string) {
	rep := strings.TrimSpace(strings.ReplaceAll(text, "\n", "<eos>"))
	words := strings.Split(rep, " ")

	w2id := make(map[string]int)
	id2w := make(map[int]string)

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

	return corpus, w2id, id2w
}

func Must(dataset *Dataset, err error) *Dataset {
	if err != nil {
		panic(err)
	}

	return dataset
}
