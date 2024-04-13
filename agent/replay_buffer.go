package agent

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/rand"
)

type Buffer struct {
	State     []float64
	Action    int
	Reward    float64
	NextState []float64
	Done      bool
}

type ReplayBuffer struct {
	Buffer    *Deque[Buffer]
	BatchSize int
	Source    randv2.Source
}

func NewReplayBuffer(bufferSize, batchSize int, s ...randv2.Source) *ReplayBuffer {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	return &ReplayBuffer{
		Buffer:    NewDeque[Buffer](bufferSize),
		BatchSize: batchSize,
		Source:    s[0],
	}
}

func (b *ReplayBuffer) Add(state []float64, action int, reward float64, next []float64, done bool) {
	b.Buffer.Add(Buffer{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: next,
		Done:      done,
	})
}

func (b *ReplayBuffer) Len() int {
	return b.Buffer.Len()
}

func (b *ReplayBuffer) Batch() ([][]float64, []int, []float64, [][]float64, []bool) {
	g := randv2.New(b.Source)

	counter := make(map[int]bool)
	for c := 0; c < b.BatchSize; {
		n := g.IntN(b.Len())
		if _, ok := counter[n]; !ok {
			counter[n] = true
			c++
		}
	}

	batch := make([]Buffer, 0, b.BatchSize)
	for k := range counter {
		batch = append(batch, b.Buffer.Get(k))
	}

	state := make([][]float64, b.BatchSize)
	action := make([]int, b.BatchSize)
	reward := make([]float64, b.BatchSize)
	next := make([][]float64, b.BatchSize)
	done := make([]bool, b.BatchSize)

	for i := range batch {
		state[i] = batch[i].State
		action[i] = batch[i].Action
		reward[i] = batch[i].Reward
		next[i] = batch[i].NextState
		done[i] = batch[i].Done
	}

	return state, action, reward, next, done
}
