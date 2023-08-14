package model

import (
	"encoding/gob"
	"fmt"
	"os"

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
	String() string
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
	String() string
}

// AttentionLayer is an interface that represents an attention layer.
type AttentionLayer interface {
	Forward(hs []matrix.Matrix, a matrix.Matrix) matrix.Matrix
	Backward(da matrix.Matrix) ([]matrix.Matrix, matrix.Matrix)
	Params() []matrix.Matrix
	Grads() []matrix.Matrix
	SetParams(p ...matrix.Matrix)
	String() string
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
