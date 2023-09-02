package agent

import (
	"math/rand"
	"time"
)

type ReplayBuffer[T any] struct {
	Buffer    *Deque[T]
	BatchSize int
	Source    rand.Source
}

func NewReplyBuffer[T any](bufferSize, batchSize int, s ...rand.Source) *ReplayBuffer[T] {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	return &ReplayBuffer[T]{
		Buffer:    NewDeque[T](bufferSize),
		BatchSize: batchSize,
		Source:    s[0],
	}
}

func (b *ReplayBuffer[T]) Append(m T) {
	b.Buffer.Append(m)
}

func (b *ReplayBuffer[T]) Len() int {
	return b.Buffer.Len()
}

func (b *ReplayBuffer[T]) Batch() []T {
	rng := rand.New(b.Source)

	counter := make(map[int]bool)
	for c := 0; c < b.BatchSize; {
		n := rng.Intn(b.Len())
		if _, ok := counter[n]; !ok {
			counter[n] = true
			c++
		}
	}

	out := make([]T, 0, b.BatchSize)
	for k := range counter {
		out = append(out, b.Buffer.Get(k))
	}

	return out
}
