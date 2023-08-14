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
	W               matrix.Matrix
	sampler         *UnigramSampler
	embeddingDot    []*EmbeddingDot
	sigmoidWithLoss []*SigmoidWithLoss
}

func NewNegativeSamplingLoss(W matrix.Matrix, corpus []int, power float64, sampleSize int) *NegativeSamplingLoss {
	embed := make([]*EmbeddingDot, 0)
	for i := 0; i < sampleSize+1; i++ {
		embed = append(embed, &EmbeddingDot{W: W})
	}
	loss := make([]*SigmoidWithLoss, 0)
	for i := 0; i < sampleSize+1; i++ {
		loss = append(loss, &SigmoidWithLoss{})
	}

	return &NegativeSamplingLoss{
		W:               W,
		sampler:         NewUnigramSampler(corpus, power, sampleSize),
		embeddingDot:    embed,
		sigmoidWithLoss: loss,
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
	a, b := l.W.Dimension()
	s := len(l.embeddingDot)
	return fmt.Sprintf("%T: W(%v, %v)*%v: %v", l, a, b, s, a*b*s)
}

func (l *NegativeSamplingLoss) Forward(h, target matrix.Matrix, opts ...Opts) matrix.Matrix {
	return nil
}

func (l *NegativeSamplingLoss) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	return nil, nil
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
