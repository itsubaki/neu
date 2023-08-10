package model

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Affine)(nil)
	_ Layer = (*layer.BatchNorm)(nil)
	_ Layer = (*layer.Dot)(nil)
	_ Layer = (*layer.Dropout)(nil)
	_ Layer = (*layer.EmbeddingDot)(nil)
	_ Layer = (*layer.Embedding)(nil)
	_ Layer = (*layer.Mul)(nil)
	_ Layer = (*layer.ReLU)(nil)
	_ Layer = (*layer.RNN)(nil)
	_ Layer = (*layer.Sigmoid)(nil)
	_ Layer = (*layer.SigmoidWithLoss)(nil)
	_ Layer = (*layer.Softmax)(nil)
	_ Layer = (*layer.SoftmaxWithLoss)(nil)
)

var (
	_ TimeLayer = (*layer.TimeAffine)(nil)
	_ TimeLayer = (*layer.TimeDropout)(nil)
	_ TimeLayer = (*layer.TimeEmbedding)(nil)
	_ TimeLayer = (*layer.TimeLSTM)(nil)
	_ TimeLayer = (*layer.TimeRNN)(nil)
	_ TimeLayer = (*layer.TimeSoftmaxWithLoss)(nil)
)

var (
	_ AttentionLayer = (*layer.Attention)(nil)
	_ AttentionLayer = (*layer.AttentionWeight)(nil)
	_ AttentionLayer = (*layer.WeightSum)(nil)
)

var (
	_ WeightInit = weight.Std(0.01)
	_ WeightInit = weight.He
	_ WeightInit = weight.Xavier
	_ WeightInit = weight.Glorot
)

// WeightInit is an interface that represents a weight initializer.
type WeightInit func(prevNodeNum int) float64

// Layer is an interface that represents a layer.
type Layer interface {
	Forward(x, y matrix.Matrix, opts ...layer.Opts) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
}

// TimeLayer is an interface that represents a time layer.
type TimeLayer interface {
	Forward(xs, ys []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix
	Backward(dout []matrix.Matrix) []matrix.Matrix
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
	SetState(h ...matrix.Matrix)
	ResetState()
}

// AttentionLayer is an interface that represents an attention layer.
type AttentionLayer interface {
	Forward(hs []matrix.Matrix, a matrix.Matrix) matrix.Matrix
	Backward(da matrix.Matrix) ([]matrix.Matrix, matrix.Matrix)
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
}

// Save saves the params to a file.
func Save(params [][]matrix.Matrix, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer f.Close()

	if err := gob.NewEncoder(f).Encode(params); err != nil {
		return fmt.Errorf("failed to encode: %v", err)
	}

	return nil
}

// Load loads the params from a file.
func Load(filename string) ([][]matrix.Matrix, bool) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, false
	}
	defer f.Close()

	var params [][]matrix.Matrix
	if err := gob.NewDecoder(f).Decode(&params); err != nil {
		return nil, false
	}

	return params, true
}

func Concat(a, b []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(a))
	for t := 0; t < len(a); t++ {
		out[t] = make(matrix.Matrix, len(a[t]))

		for i := 0; i < len(a[t]); i++ {
			out[t][i] = append(out[t][i], a[t][i]...)
		}

		for i := 0; i < len(b[t]); i++ {
			out[t][i] = append(out[t][i], b[t][i]...)
		}
	}

	return out
}

func Split(dout []matrix.Matrix, H int) ([]matrix.Matrix, []matrix.Matrix) {
	a, b := make([]matrix.Matrix, len(dout)), make([]matrix.Matrix, len(dout))
	for t := range dout {
		a[t], b[t] = matrix.New(), matrix.New()
		for _, r := range dout[t] {
			a[t] = append(a[t], r[H:])
			b[t] = append(b[t], r[:H])
		}
	}

	return a, b
}

func Flatten(m []matrix.Matrix) []float64 {
	flatten := make([]float64, 0)
	for _, s := range m {
		flatten = append(flatten, matrix.Flatten(s)...)
	}

	return flatten
}

func Choice(p []float64, s ...rand.Source) int {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	cumsum := make([]float64, len(p))
	var sum float64
	for i, prob := range p {
		sum += prob
		cumsum[i] = sum
	}

	r := rand.New(s[0]).Float64()
	for i, prop := range cumsum {
		if r <= prop {
			return i
		}
	}

	return -1
}

func Contains[T comparable](v T, s []T) bool {
	for _, ss := range s {
		if v == ss {
			return true
		}
	}

	return false
}

func Argmax(score []matrix.Matrix) int {
	var arg int
	max := math.SmallestNonzeroFloat64
	for i, v := range Flatten(score) {
		if v > max {
			max = v
			arg = i
		}
	}

	return arg
}
