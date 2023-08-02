package sequence

import (
	"fmt"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

const Addition = "addition.txt"

type Dataset struct {
	Train [][]int
	Test  [][]int
}

type Vocab struct {
	WordToID map[string]int
	IDToWord map[int]string
}

func (v Vocab) ToWord(x []int) []string {
	words := make([]string, len(x))
	for i, id := range x {
		words[i] = v.IDToWord[id]
	}

	return words
}

func Load(dir, fileName string, s ...rand.Source) (*Dataset, *Dataset, *Vocab, error) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// read
	path := filepath.Clean(path.Join(dir, fileName))
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("open file=%v: %v", path, err)
	}

	// preprocess
	q, ans := make([]string, 0), make([]string, 0)
	for _, line := range strings.Split(string(bytes), "\n") {
		spl := strings.Split(line, "_")
		if len(spl) != 2 {
			continue
		}

		q, ans = append(q, spl[0]), append(ans, fmt.Sprintf("_%s", spl[1]))
	}

	// vocab
	w2id := make(map[string]int)
	id2w := make(map[int]string)
	for i := range q {
		update(q[i], w2id, id2w)
		update(ans[i], w2id, id2w)
	}

	// data
	x, t := data(q, w2id), data(ans, w2id)
	xs, ts := shuffle(x, t, s[0])

	// 10% for validation set
	idx := len(xs) - len(xs)/10
	return &Dataset{
			Train: xs[:idx],
			Test:  xs[idx:],
		}, &Dataset{
			Train: ts[:idx],
			Test:  ts[idx:],
		}, &Vocab{
			WordToID: w2id,
			IDToWord: id2w,
		}, nil
}

func data(txt []string, w2id map[string]int) [][]int {
	x := make([][]int, len(txt))
	for i, s := range txt {
		x[i] = make([]int, len(s))
		for j, w := range s {
			x[i][j] = w2id[string(w)]
		}
	}

	return x
}

func update(v string, w2id map[string]int, id2w map[int]string) {
	words := make([]string, 0)
	for _, w := range v {
		words = append(words, string(w))
	}

	for _, w := range words {
		if _, ok := w2id[w]; ok {
			continue
		}

		id := len(w2id)
		w2id[w] = id
		id2w[id] = w
	}
}

func shuffle(x, t [][]int, s ...rand.Source) ([][]int, [][]int) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	xs, ts := make([][]int, len(x)), make([][]int, len(x))
	for i := range x {
		xs[i], ts[i] = x[i], t[i]
	}

	for i := 0; i < len(x); i++ {
		j := rng.Intn(i + 1)
		xs[i], xs[j] = xs[j], xs[i]
		ts[i], ts[j] = ts[j], ts[i]
	}

	return xs, ts
}

func Must(train, test *Dataset, vocab *Vocab, err error) (*Dataset, *Dataset, *Vocab) {
	if err != nil {
		panic(err)
	}

	return train, test, vocab
}
