package rand_test

import (
	"fmt"
	randv2 "math/rand/v2"
	"testing"

	"github.com/itsubaki/neu/math/rand"
)

func ExampleConst() {
	s := rand.Const()
	fmt.Printf("%.13f\n", randv2.New(s).Float64())
	fmt.Printf("%.13f\n", randv2.New(s).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(2)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(3)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1, 0)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1, 1)).Float64())
	fmt.Printf("%.13f\n", randv2.New(rand.Const(1, 2)).Float64())

	// Output:
	// 0.9999275824803
	// 0.8856419373529
	// 0.2384231908739
	// 0.2384231908739
	// 0.2384231908739
	// 0.8269781200925
	// 0.8353847703964
	// 0.2384231908739
	// 0.3402859786606
	// 0.6764556596678
}

func TestMustNewSource(t *testing.T) {
	v := randv2.New(rand.MustNewSource()).Float64()
	if v >= 0 && v < 1 {
		return
	}

	t.Fail()
}

func TestConst(t *testing.T) {
	v := randv2.New(rand.Const()).Float64()
	if v >= 0 && v < 1 {
		return
	}

	t.Fail()
}

func TestMustPanic(t *testing.T) {
	defer func() {
		if rec := recover(); rec != nil {
			err, ok := rec.(error)
			if !ok {
				t.Fail()
			}

			if err.Error() != "something went wrong" {
				t.Fail()
			}
		}
	}()

	rand.Must(-1, fmt.Errorf("something went wrong"))
	t.Fail()
}
