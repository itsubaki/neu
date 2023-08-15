package layer

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
)

type NegativeSamplingLoss struct {
	sampler         *UnigramSampler
	embeddingDot    []*EmbeddingDot
	sigmoidWithLoss []*SigmoidWithLoss
	s               rand.Source
}

func NewNegativeSamplingLoss(W matrix.Matrix, corpus []int, power float64, sampleSize int, s ...rand.Source) *NegativeSamplingLoss {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	embed, loss := make([]*EmbeddingDot, sampleSize+1), make([]*SigmoidWithLoss, sampleSize+1)
	for i := 0; i < sampleSize+1; i++ {
		embed[i], loss[i] = &EmbeddingDot{Embedding: Embedding{W: W}}, &SigmoidWithLoss{}
	}

	return &NegativeSamplingLoss{
		sampler:         NewUnigramSampler(corpus, power, sampleSize),
		embeddingDot:    embed,
		sigmoidWithLoss: loss,
		s:               s[0],
	}
}

func (l *NegativeSamplingLoss) Params() []matrix.Matrix {
	params := make([]matrix.Matrix, 0)
	for _, e := range l.embeddingDot {
		params = append(params, e.Params()...)
	}

	return params
}

func (l *NegativeSamplingLoss) Grads() []matrix.Matrix {
	grads := make([]matrix.Matrix, 0)
	for _, e := range l.embeddingDot {
		grads = append(grads, e.Grads()...)
	}

	return grads
}

func (l *NegativeSamplingLoss) SetParams(p ...matrix.Matrix) {
	for i, pp := range p {
		l.embeddingDot[i].SetParams(pp)
	}
}

func (l *NegativeSamplingLoss) String() string {
	a, b := l.embeddingDot[0].Embedding.W.Dimension()
	s := len(l.embeddingDot)
	return fmt.Sprintf("%T: W(%v, %v)*%v: %v", l, a, b, s, a*b*s)
}

func (l *NegativeSamplingLoss) Forward(h, target matrix.Matrix, _ ...Opts) matrix.Matrix {
	// correct
	correct := matrix.One(1, len(target))                // (1, N)
	score := l.embeddingDot[0].Forward(h, target)        // (1, N)
	loss := l.sigmoidWithLoss[0].Forward(score, correct) // (1, 1)

	// negative
	sampled := l.sampler.NegativeSample(vector.Int(matrix.Flatten(target)), l.s) // (N, S)
	label := matrix.Zero(1, len(target))                                         // (1, N)
	for i := 0; i < l.sampler.sampleSize; i++ {
		negative := matrix.Column(matrix.From(sampled), i)    // (N, 1)
		score := l.embeddingDot[i+1].Forward(h, negative)     // (1, N)
		nloss := l.sigmoidWithLoss[i+1].Forward(score, label) // (1, 1)
		loss = loss.Add(nloss)                                // (1, 1)
	}

	return loss
}

func (l *NegativeSamplingLoss) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dh := matrix.Zero(1, 1)
	for i := 0; i < l.sampler.sampleSize+1; i++ {
		dscore, _ := l.sigmoidWithLoss[i].Backward(dout) //
		dh0, _ := l.embeddingDot[i].Backward(dscore)     //
		dh = dh0.Add(dh)                                 // Broadcast
	}

	return dh, nil
}

type UnigramSampler struct {
	corpus     []int
	power      float64
	sampleSize int
	vocabSize  int
	wordProb   []float64
}

func NewUnigramSampler(corpus []int, power float64, size int) *UnigramSampler {
	counts := make(map[int]int, 0)
	for _, id := range corpus {
		counts[id]++
	}

	plist := make([]float64, len(counts))
	for i := range counts {
		plist[i] = math.Pow(float64(counts[i]), power)
	}

	sum := vector.Sum(plist)
	p := vector.Div(plist, sum)

	return &UnigramSampler{
		corpus:     corpus,
		power:      power,
		sampleSize: size,
		vocabSize:  len(counts),
		wordProb:   p,
	}
}

func (s *UnigramSampler) NegativeSample(target []int, seed ...rand.Source) [][]int {
	if len(seed) == 0 {
		seed = append(seed, rand.NewSource(time.Now().UnixNano()))
	}

	N := len(target)
	out := make([][]int, N)

	for i := 0; i < N; i++ {
		p := make([]float64, len(s.wordProb))
		copy(p, s.wordProb)
		p[target[i]] = 0
		p = vector.Div(p, vector.Sum(p))

		sampled := make([]int, s.sampleSize)
		for j := 0; j < s.sampleSize; j++ {
			sampled[j] = vector.Choice(p, seed[0])
		}

		out[i] = sampled
	}

	return out
}
