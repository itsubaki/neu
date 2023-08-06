package sequence

import (
	"fmt"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/itsubaki/neu/math/vector"
)

const Addition = "addition.txt"

type Vocab struct {
	RuneToID map[rune]int
	IDToRune map[int]rune
}

func (v Vocab) ID(words []string) [][]int {
	x := make([][]int, len(words))
	for i, s := range words {
		x[i] = make([]int, len(s))
		for j, w := range s {
			x[i][j] = v.RuneToID[w]
		}
	}

	return x
}

func (v Vocab) Rune(x []int) []rune {
	words := make([]rune, len(x))
	for i, id := range x {
		words[i] = v.IDToRune[id]
	}

	return words
}

func (v Vocab) ToString(x []int) []string {
	words := make([]string, len(x))
	for i, w := range v.Rune(x) {
		words[i] = string(w)
	}

	return words
}

type Dataset struct {
	Train [][]int
	Test  [][]int
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
		idx := strings.Index(line, "_")
		if idx == -1 {
			continue
		}

		q, ans = append(q, line[:idx]), append(ans, line[idx:])
	}

	// vocab
	v := vocab(q, ans)

	// data
	x, t := v.ID(q), v.ID(ans)
	xs, ts := vector.Shuffle(x, t, s[0])

	// 10% for validation set
	idx := len(xs) - len(xs)/10
	xd := &Dataset{Train: xs[:idx], Test: xs[idx:]}
	td := &Dataset{Train: ts[:idx], Test: ts[idx:]}
	return xd, td, v, nil
}

func vocab(q, ans []string) *Vocab {
	r2id := make(map[rune]int)
	id2r := make(map[int]rune)

	words := make([]rune, 0)
	for _, w := range append(q, ans...) {
		for _, v := range w {
			words = append(words, v)
		}
	}

	for _, r := range words {
		if _, ok := r2id[r]; ok {
			continue
		}

		id := len(r2id)
		r2id[r] = id
		id2r[id] = r
	}

	return &Vocab{
		RuneToID: r2id,
		IDToRune: id2r,
	}
}

func Must(train, test *Dataset, vocab *Vocab, err error) (*Dataset, *Dataset, *Vocab) {
	if err != nil {
		panic(err)
	}

	return train, test, vocab
}
