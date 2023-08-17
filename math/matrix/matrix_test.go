package matrix_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
)

func ExampleZero() {
	for _, r := range matrix.Zero(2, 3) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0]
	// [0 0 0]

}

func ExampleOne() {
	for _, r := range matrix.One(2, 3) {
		fmt.Println(r)
	}

	// Output:
	// [1 1 1]
	// [1 1 1]

}

func ExampleRand() {
	fmt.Println(matrix.Rand(2, 3).Dim())

	s := rand.NewSource(1)
	for _, r := range matrix.Rand(2, 3, s) {
		fmt.Println(r)
	}

	// Output:
	// 2 3
	// [0.6046602879796196 0.9405090880450124 0.6645600532184904]
	// [0.4377141871869802 0.4246374970712657 0.6868230728671094]

}
func ExampleRandn() {
	fmt.Println(matrix.Randn(2, 3).Dim())

	s := rand.NewSource(1)
	for _, r := range matrix.Randn(2, 3, s) {
		fmt.Println(r)
	}

	// Output:
	// 2 3
	// [-1.233758177597947 -0.12634751070237293 -0.5209945711531503]
	// [2.28571911769958 0.3228052526115799 0.5900672875996937]

}

func ExampleMask() {
	x := matrix.New(
		[]float64{1, -0.5},
		[]float64{-2, 3},
	)
	mask := matrix.Mask(x, func(x float64) bool { return x > 0 })

	for _, r := range mask {
		fmt.Println(r)
	}

	// Output:
	// [1 0]
	// [0 1]
}

func ExampleBatch() {
	x := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
		[]float64{5, 6},
		[]float64{7, 8},
		[]float64{9, 10},
	)
	for _, r := range matrix.Batch(x, []int{0, 2, 4}) {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [5 6]
	// [9 10]
}

func ExampleColumn() {
	x := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	fmt.Println(matrix.Column(x, 0))
	fmt.Println(matrix.Column(x, 1))
	fmt.Println(matrix.Column(x, 2))

	// Output:
	// [[1] [4]]
	// [[2] [5]]
	// [[3] [6]]
}

func ExampleFrom() {
	fmt.Printf("%.2f", matrix.From([][]int{
		{1, 2, 3},
		{4, 5, 6},
	}))

	// Output:
	// [[1.00 2.00 3.00] [4.00 5.00 6.00]]
}

func ExampleIdentity() {
	fmt.Println(matrix.Identity(3))

	// Output:
	// [[1 0 0] [0 1 0] [0 0 1]]
}

func ExampleOneHot() {
	for _, v := range matrix.OneHot([]int{0, 1, 2, 3, 4, 1, 5, 6}, 7) {
		fmt.Println(v)
	}

	// Output:
	// [1 0 0 0 0 0 0]
	// [0 1 0 0 0 0 0]
	// [0 0 1 0 0 0 0]
	// [0 0 0 1 0 0 0]
	// [0 0 0 0 1 0 0]
	// [0 1 0 0 0 0 0]
	// [0 0 0 0 0 1 0]
	// [0 0 0 0 0 0 1]
}

func ExampleInt() {
	fmt.Println(matrix.Int(matrix.New([]float64{1.1, 2.2, 3.3})))

	// Output:
	// [[1 2 3]]
}

func ExampleDot() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.Dot(A, B) {
		fmt.Println(r)
	}

	// Output:
	// [19 22]
	// [43 50]

}

func ExampleMatrix_Dim() {
	fmt.Println(matrix.New().Dim())
	fmt.Println(matrix.New([]float64{1, 2, 3}).Dim())

	// Output:
	// 0 0
	// 1 3

}

func ExampleMatrix_Add() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Add(B) {
		fmt.Println(r)
	}

	// Output:
	// [6 8]
	// [10 12]

}

func ExampleMatrix_Sub() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Sub(B) {
		fmt.Println(r)
	}

	// Output:
	// [-4 -4]
	// [-4 -4]

}

func ExampleMatrix_Mul() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range A.Mul(B) {
		fmt.Println(r)
	}

	// Output:
	// [5 12]
	// [21 32]

}

func ExampleMatrix_Div() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 2},
		[]float64{1, 8},
	)

	for _, r := range A.Div(B) {
		fmt.Println(r)
	}

	// Output:
	// [0.2 1]
	// [3 0.5]

}

func ExampleMatrix_AddC() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.AddC(2) {
		fmt.Println(r)
	}

	// Output:
	// [3 4]
	// [5 6]

}

func ExampleMatrix_MulC() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.MulC(2) {
		fmt.Println(r)
	}

	// Output:
	// [2 4]
	// [6 8]

}

func ExampleMatrix_Pow2() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.Pow2() {
		fmt.Println(r)
	}

	// Output:
	// [1 4]
	// [9 16]

}

func ExampleMatrix_Sqrt() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.Sqrt(0) {
		fmt.Println(r)
	}

	// Output:
	// [1 1.4142135623730951]
	// [1.7320508075688772 2]

}

func ExampleMatrix_Abs() {
	A := matrix.New(
		[]float64{-1, 2},
		[]float64{3, -4},
	)

	for _, r := range A.Abs() {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]

}

func ExampleMatrix_Avg() {
	A := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	fmt.Println(A.Sum())
	fmt.Println(A.Avg())

	// Output:
	// 21
	// 3.5
}

func ExampleMatrix_Argmax() {
	A := matrix.New(
		[]float64{1, 2, 3},
		[]float64{6, 5, 4},
	)
	fmt.Println(A.Argmax())

	// Output:
	// [2 0]
}

func ExampleMatrix_SumAxis0() {
	x := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	fmt.Println(x.SumAxis0())

	// Output:
	// [5 7 9]
}

func ExampleMatrix_SumAxis1() {
	x := matrix.New(
		[]float64{0, 1, 4},
		[]float64{27, 40, 55},
		[]float64{18, 28, 40},
	)
	fmt.Println(x.SumAxis1())

	// Output:
	// [5 122 86]
}

func ExampleMatrix_MeanAxis0() {
	x := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	fmt.Println(x.MeanAxis0())

	// Output:
	// [2.5 3.5 4.5]
}

func ExampleMatrix_MaxAxis1() {
	x := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	fmt.Println(x.MaxAxis1())

	// Output:
	// [3 6]
}

func ExampleMatrix_T() {
	A := matrix.New(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	for _, r := range A.T() {
		fmt.Println(r)
	}

	// Output:
	// [1 4]
	// [2 5]
	// [3 6]
}

func ExampleMatrix_F() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range A.F(func(v float64) float64 { return v * 3.0 }) {
		fmt.Println(r)
	}

	// Output:
	// [3 6]
	// [9 12]

}

func ExampleMatrix_Broadcast() {
	m := matrix.New([]float64{1})
	for _, r := range m {
		fmt.Println(r)
	}
	fmt.Println()

	for _, r := range m.Broadcast(3, 5) {
		fmt.Println(r)
	}

	// Output:
	// [1]
	//
	// [1 1 1 1 1]
	// [1 1 1 1 1]
	// [1 1 1 1 1]
}

func ExampleMatrix_Broadcast_row() {
	m := matrix.New([]float64{1, 2})
	for _, r := range m {
		fmt.Println(r)
	}
	fmt.Println()

	for _, r := range m.Broadcast(5, -1) {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	//
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
	// [1 2]
}

func ExampleMatrix_Broadcast_column() {
	m := matrix.New(
		[]float64{1},
		[]float64{2},
	)
	for _, r := range m {
		fmt.Println(r)
	}
	fmt.Println()

	for _, r := range m.Broadcast(-1, 5) {
		fmt.Println(r)
	}

	// Output:
	// [1]
	// [2]
	//
	// [1 1 1 1 1]
	// [2 2 2 2 2]
}

func ExampleMatrix_Broadcast_noEffect() {
	m := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	for _, r := range m {
		fmt.Println(r)
	}
	fmt.Println()

	for _, r := range m.Broadcast(2, 2) {
		fmt.Println(r)
	}

	// Output:
	// [1 2]
	// [3 4]
	//
	// [1 2]
	// [3 4]
}

func ExampleF() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	for _, r := range matrix.F(A, func(v float64) float64 { return v * 2 }) {
		fmt.Println(r)
	}

	// Output:
	// [2 4]
	// [6 8]

}

func ExampleF2() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	for _, r := range matrix.F2(A, B, func(a, b float64) float64 { return a * b }) {
		fmt.Println(r)
	}

	// Output:
	// [5 12]
	// [21 32]

}

func ExampleF3() {
	A := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	B := matrix.New(
		[]float64{5, 6},
		[]float64{7, 8},
	)

	C := matrix.New(
		[]float64{9, 9},
		[]float64{9, 9},
	)

	for _, r := range matrix.F3(A, B, C, func(a, b, c float64) float64 { return a*b - c }) {
		fmt.Println(r)
	}

	// Output:
	// [-4 3]
	// [12 23]

}

func ExamplePadding() {
	x := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)
	pad := 1

	p := matrix.Padding(x, pad)
	for _, v := range p {
		fmt.Println(v)
	}
	fmt.Println()

	for _, v := range matrix.Unpadding(p, pad) {
		fmt.Println(v)
	}

	// Output:
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]
	//
	// [1 2]
	// [3 4]

}

func ExampleReshape() {
	x := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fmt.Println(matrix.Reshape(x, 1, 4))
	fmt.Println(matrix.Reshape(x, 4, 1))
	fmt.Println(matrix.Reshape(x, 2, 2))
	fmt.Println()

	fmt.Println(matrix.Reshape(x, 1, -1))
	fmt.Println(matrix.Reshape(x, 4, -1))
	fmt.Println(matrix.Reshape(x, 2, -1))
	fmt.Println()

	fmt.Println(matrix.Reshape(x, -1, 1))
	fmt.Println(matrix.Reshape(x, -1, 4))
	fmt.Println(matrix.Reshape(x, -1, 2))
	fmt.Println()

	// Output:
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1 2 3 4]]
	// [[1] [2] [3] [4]]
	// [[1 2] [3 4]]
	//
	// [[1] [2] [3] [4]]
	// [[1 2 3 4]]
	// [[1 2] [3 4]]
}

func ExampleHStack() {
	a := matrix.New([]float64{1, 2, 3}, []float64{4, 5, 6})
	b := matrix.New([]float64{7, 8, 9}, []float64{10, 11, 12})

	for _, r := range matrix.HStack(a, b) {
		fmt.Println(r)
	}

	// Output:
	// [1 2 3 7 8 9]
	// [4 5 6 10 11 12]

}
