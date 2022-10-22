package model_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/mnist"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/trainer"
)

func Example_mnist() {
	// data
	train, test := mnist.Must(mnist.Load("../testdata"))

	x := matrix.New(mnist.Normalize(train.Image)...)
	t := matrix.New(mnist.OneHot(train.Label)...)

	xt := matrix.New(mnist.Normalize(test.Image)...)
	tt := matrix.New(mnist.OneHot(test.Label)...)

	// init
	rand.Seed(1) // for test
	n := model.New(&model.Config{
		InputSize:  784,
		HiddenSize: []int{50},
		OutputSize: 10,
		WeightInit: model.Std(0.01),
		Optimizer:  &optimizer.SGD{LearningRate: 0.1},
	})

	// training
	batchSize := 10
	iter := 1000

	for i := 0; i < iter; i++ {
		// batch
		mask := trainer.Random(train.N, batchSize)
		xbatch := matrix.Batch(x, mask)
		tbatch := matrix.Batch(t, mask)

		// update
		grads := n.Gradient(xbatch, tbatch)
		n.Optimize(grads)

		if i%200 == 0 {
			loss := n.Loss(xbatch, tbatch)
			acc := model.Accuracy(n.Predict(xbatch), tbatch)

			mask := trainer.Random(test.N, batchSize)
			xtbatch := matrix.Batch(xt, mask)
			ttbatch := matrix.Batch(tt, mask)
			tacc := model.Accuracy(n.Predict(xtbatch), ttbatch)

			fmt.Printf("loss=%.04f, train_acc=%.04f, test_acc=%.04f\n", loss, acc, tacc)
		}
	}

	// Output:
	// loss=[[2.2971]], train_acc=0.3000, test_acc=0.2000
	// loss=[[0.3096]], train_acc=1.0000, test_acc=0.5000
	// loss=[[0.1905]], train_acc=1.0000, test_acc=1.0000
	// loss=[[0.0858]], train_acc=1.0000, test_acc=1.0000
	// loss=[[0.0340]], train_acc=1.0000, test_acc=0.9000

}

func Example_accuracy() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	_, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	n := model.New(&model.Config{
		InputSize:  inSize,
		HiddenSize: []int{hiddenSize},
		OutputSize: outSize,
		WeightInit: model.Std(0.01),
		Optimizer:  &optimizer.SGD{LearningRate: 0.1},
	})

	// training
	for i := 0; i < 1000; i++ {
		y := n.Predict(x)
		loss := n.Loss(x, t)
		grads := n.Gradient(x, t)
		n.Optimize(grads)

		if i%250 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f, acc=%.4f\n", layer.Softmax(y), loss, model.Accuracy(y, t))
		}
	}

	// Output:
	// predict=[[0.5000 0.5000] [0.5000 0.5000] [0.5000 0.5000]], loss=[[0.6931]], acc=0.3333
	// predict=[[0.3373 0.6627] [0.3373 0.6627] [0.3203 0.6797]], loss=[[0.6281]], acc=0.6667
	// predict=[[0.4506 0.5494] [0.4505 0.5495] [0.0703 0.9297]], loss=[[0.4896]], acc=0.6667
	// predict=[[0.5120 0.4880] [0.4824 0.5176] [0.0191 0.9809]], loss=[[0.4491]], acc=1.0000
}

func Example_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	_, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	n := model.New(&model.Config{
		InputSize:  inSize,
		HiddenSize: []int{hiddenSize},
		OutputSize: outSize,
		WeightInit: model.Std(0.01),
		Optimizer:  &optimizer.SGD{LearningRate: 0.1},
	})

	// gradient
	ngrads := n.NumericalGradient(x, t)
	grads := n.Gradient(x, t)

	// check
	for _, k := range []string{"W1", "W2", "B1", "B2"} {
		diff := matrix.FuncWith(ngrads[k], grads[k], func(a, b float64) float64 { return math.Abs(a - b) })
		fmt.Printf("%v: %v\n", k, diff.Avg())
	}

	// Output:
	// W1: 1.6658328893821156e-10
	// W2: 2.8191449832846066e-10
	// B1: 7.510020527910314e-13
	// B2: 3.3325323139932195e-08

}

func Example_optimize() {
	// initial weight
	params := make(map[string]matrix.Matrix)
	params["W1"] = matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	params["B1"] = matrix.New([]float64{0.1, 0.2, 0.3})
	params["W2"] = matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	params["B2"] = matrix.New([]float64{0.1, 0.2})

	// data
	x := matrix.New([]float64{0.5, 0.5})
	t := matrix.New([]float64{1, 0})

	// training
	for i := 0; i < 40; i++ {
		// layer
		layers := []model.Layer{
			&layer.Affine{W: params["W1"], B: params["B1"]},
			&layer.ReLU{},
			&layer.Affine{W: params["W2"], B: params["B2"]},
		}
		last := &layer.SoftmaxWithLoss{}

		// forward
		for _, l := range layers {
			x = l.Forward(x, nil)
		}
		loss := last.Forward(x, t)

		if i%10 == 0 {
			fmt.Printf("predict=%.04f, loss=%.04f\n", layer.Softmax(x), loss)
		}

		// backward
		dout, _ := last.Backward(matrix.New([]float64{1}))
		for i := len(layers) - 1; i > -1; i-- {
			dout, _ = layers[i].Backward(dout)
		}

		// gradient
		grads := make(map[string]matrix.Matrix)
		grads["W1"] = layers[0].(*layer.Affine).DW
		grads["B1"] = layers[0].(*layer.Affine).DB
		grads["W2"] = layers[2].(*layer.Affine).DW
		grads["B2"] = layers[2].(*layer.Affine).DB

		// optimize
		opt := &optimizer.SGD{LearningRate: 0.1}
		params = opt.Update(params, grads)
	}

	// Output:
	// predict=[[0.3555 0.6445]], loss=[[1.0343]]
	// predict=[[0.8930 0.1070]], loss=[[0.1132]]
	// predict=[[0.9878 0.0122]], loss=[[0.0122]]
	// predict=[[0.9982 0.0018]], loss=[[0.0018]]

}

func Example_layer() {
	// weight
	W1 := matrix.New([]float64{0.1, 0.3, 0.5}, []float64{0.2, 0.4, 0.6})
	B1 := matrix.New([]float64{0.1, 0.2, 0.3})
	W2 := matrix.New([]float64{0.1, 0.4}, []float64{0.2, 0.5}, []float64{0.3, 0.6})
	B2 := matrix.New([]float64{0.1, 0.2})
	W3 := matrix.New([]float64{0.1, 0.3}, []float64{0.2, 0.4})
	B3 := matrix.New([]float64{0.1, 0.2})

	// layer
	layers := []model.Layer{
		&layer.Affine{W: W1, B: B1},
		&layer.Sigmoid{},
		&layer.Affine{W: W2, B: B2},
		&layer.Sigmoid{},
		&layer.Affine{W: W3, B: B3},
	}

	// forward
	x := matrix.New([]float64{1.0, 0.5})
	for _, l := range layers {
		x = l.Forward(x, nil)
	}

	// backward
	dout := matrix.New([]float64{1})
	for i := len(layers) - 1; i > -1; i-- {
		dout, _ = layers[i].Backward(dout)
	}

	// print
	fmt.Println(x)
	fmt.Println(dout)

	// Output:
	// [[0.3168270764110298 0.6962790898619668]]
	// [[0.004530872169552449 0.005957136450888734]]

}

func ExampleXavier() {
	fmt.Println(model.Xavier(1))
	fmt.Println(model.Xavier(2))
	fmt.Println(model.Xavier(4))

	// Output:
	// 1
	// 0.7071067811865476
	// 0.5

}

func ExampleHe() {
	fmt.Println(model.He(1))
	fmt.Println(model.He(2))
	fmt.Println(model.He(4))

	// Output:
	// 1.4142135623730951
	// 1
	// 0.7071067811865476

}

func Example_multiLayer() {
	rand.Seed(1) // for test

	input := 784
	hidden := []int{100, 100, 100}
	out := 10
	weightDecayLambda := 1e-6

	// size
	size := append([]int{input}, hidden...)
	size = append(size, out)

	fmt.Println(size)
	fmt.Println()

	// params
	params := make(map[string]matrix.Matrix)
	for i := 0; i < len(size)-1; i++ {
		W, B := fmt.Sprintf("W%v", i+1), fmt.Sprintf("B%v", i+1)
		params[W] = matrix.Randn(size[i], size[i+1])
		params[B] = matrix.Zero(1, size[i+1])
	}

	for k, v := range params {
		a, b := v.Dimension()
		fmt.Printf("%v: %v, %v\n", k, a, b)
	}
	fmt.Println()

	// weight init
	for i := 0; i < len(size)-1; i++ {
		W := fmt.Sprintf("W%v", i+1)
		params[W] = params[W].Func(func(v float64) float64 {
			return model.He(size[i]) * v
		})

		fmt.Printf("%v: He(%v)\n", W, size[i])
	}
	fmt.Println()

	// layer
	layers := make([]model.Layer, 0)
	for i := 0; i < len(size)-1; i++ {
		W, B := fmt.Sprintf("W%v", i+1), fmt.Sprintf("B%v", i+1)
		layers = append(layers, &layer.Affine{W: params[W], B: params[B]})
		layers = append(layers, &layer.ReLU{})
	}
	layers = layers[:len(layers)-1] // remove last ReLu

	for i, l := range layers {
		fmt.Printf("%v: %T\n", i, l)
	}
	fmt.Println()

	// decay
	var decay float64
	for i := 0; i < len(size)-1; i++ {
		W := fmt.Sprintf("W%v", i+1)
		sump2 := params[W].Func(func(v float64) float64 { return v * v }).Sum()
		decay = decay + 0.5*weightDecayLambda*sump2

		fmt.Printf("%v: decay=%v\n", W, decay)
	}
	fmt.Println()

	// gradient
	var j int
	for i := 0; i < len(layers); i++ {
		if _, ok := layers[i].(*layer.Affine); !ok {
			continue
		}

		W, B := fmt.Sprintf("W%v", j+1), fmt.Sprintf("B%v", j+1)
		fmt.Printf("grads[%v]=(%T)layer[%v].DW + %v * %v\n", W, layers[i], i, weightDecayLambda, W)
		fmt.Printf("grads[%v]=(%T)layer[%v].DB\n", B, layers[i], i)
		j++
	}
	fmt.Println()

	// Unordered output:
	// [784 100 100 100 10]
	//
	// W1: 784, 100
	// B1: 1, 100
	// W2: 100, 100
	// B2: 1, 100
	// W3: 100, 100
	// B3: 1, 100
	// W4: 100, 10
	// B4: 1, 10
	//
	// W1: He(784)
	// W2: He(100)
	// W3: He(100)
	// W4: He(100)
	//
	// 0: *layer.Affine
	// 1: *layer.ReLU
	// 2: *layer.Affine
	// 3: *layer.ReLU
	// 4: *layer.Affine
	// 5: *layer.ReLU
	// 6: *layer.Affine
	//
	// W1: decay=0.00010047182907097258
	// W2: decay=0.00019823447431020557
	// W3: decay=0.0002991742587073469
	// W4: decay=0.00030882534324749964
	//
	// grads[W1]=(*layer.Affine)layer[0].DW + 1e-06 * W1
	// grads[B1]=(*layer.Affine)layer[0].DB
	// grads[W2]=(*layer.Affine)layer[2].DW + 1e-06 * W2
	// grads[B2]=(*layer.Affine)layer[2].DB
	// grads[W3]=(*layer.Affine)layer[4].DW + 1e-06 * W3
	// grads[B3]=(*layer.Affine)layer[4].DB
	// grads[W4]=(*layer.Affine)layer[6].DW + 1e-06 * W4
	// grads[B4]=(*layer.Affine)layer[6].DB

}
