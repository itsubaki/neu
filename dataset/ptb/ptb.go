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

func Load(dir string) (*Dataset, *Dataset, *Dataset, error) {
	train, err := load(dir, TrainTxt)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load training data: %v", err)
	}

	test, err := load(dir, TestTxt)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load test data: %v", err)
	}

	valid, err := load(dir, ValidTxt)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load valid data: %v", err)
	}

	return train, test, valid, nil
}

func load(dir, fileName string) (*Dataset, error) {
	path := filepath.Clean(path.Join(dir, fileName))
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("open file=%v: %v", path, err)
	}

	word := word(bytes)
	w2id := make(map[string]int)
	id2w := make(map[int]string)

	for _, w := range word {
		if _, ok := w2id[w]; ok {
			continue
		}

		id := len(w2id)
		w2id[w] = id
		id2w[id] = w
	}

	corpus := make([]int, 0)
	for _, w := range word {
		corpus = append(corpus, w2id[w])
	}

	return &Dataset{
		WordToID: w2id,
		IDToWord: id2w,
		Corpus:   corpus,
	}, nil
}

func word(bytes []byte) []string {
	rep := strings.TrimSpace(strings.ReplaceAll(string(bytes), "\n", "<eos>"))
	spl := strings.Split(rep, " ")
	return spl
}

func Must(train, test, valid *Dataset, err error) (*Dataset, *Dataset, *Dataset) {
	if err != nil {
		panic(err)
	}

	return train, test, valid
}
