package agent

import (
	"math/rand"
	"time"
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
	Source    rand.Source
}

func NewReplayBuffer(bufferSize, batchSize int, s ...rand.Source) *ReplayBuffer {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	return &ReplayBuffer{
		Buffer:    NewDeque[Buffer](bufferSize),
		BatchSize: batchSize,
		Source:    s[0],
	}
}

func (b *ReplayBuffer) Append(state []float64, action int, reward float64, next []float64, done bool) {
	b.Buffer.Append(Buffer{
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

func (b *ReplayBuffer) Batch() []Buffer {
	rng := rand.New(b.Source)

	counter := make(map[int]bool)
	for c := 0; c < b.BatchSize; {
		n := rng.Intn(b.Len())
		if _, ok := counter[n]; !ok {
			counter[n] = true
			c++
		}
	}

	out := make([]Buffer, 0, b.BatchSize)
	for k := range counter {
		out = append(out, b.Buffer.Get(k))
	}

	return out
}
