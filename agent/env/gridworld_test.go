package env_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent/env"
)

func ExampleGridWorld() {
	e := env.NewGridWorld()

	fmt.Println(e.Height(), e.Width())
	fmt.Println(e.Shape())

	for _, a := range e.Actions() {
		fmt.Println(a, e.ActionMeaning[a])
	}

	for i := 0; i < e.Height()*e.Width(); i++ {
		s := e.State()
		fmt.Println(s, e.Reward(nil, 0, s))
	}

	// Output:
	// 3 4
	// 3 4
	// 0 UP
	// 1 RIGHT
	// 2 DOWN
	// 3 LEFT
	// &{0 0} 0
	// &{0 1} 0
	// &{0 2} 0
	// &{0 3} 1
	// &{1 0} 0
	// &{1 1} 0
	// &{1 2} 0
	// &{1 3} -1
	// &{2 0} 0
	// &{2 1} 0
	// &{2 2} 0
	// &{2 3} 0
}

func ExampleGridWorld_NextState() {
	e := env.NewGridWorld()

	fmt.Println(e.NextState(&env.GridState{Height: 0, Width: 0}, 0))
	fmt.Println(e.NextState(&env.GridState{Height: 0, Width: 0}, 1))
	fmt.Println(e.NextState(&env.GridState{Height: 0, Width: 0}, 2))
	fmt.Println(e.NextState(&env.GridState{Height: 0, Width: 0}, 3))
	fmt.Println()

	fmt.Println(e.NextState(&env.GridState{Height: 1, Width: 0}, 0))
	fmt.Println(e.NextState(&env.GridState{Height: 1, Width: 0}, 1))
	fmt.Println(e.NextState(&env.GridState{Height: 1, Width: 0}, 2))
	fmt.Println(e.NextState(&env.GridState{Height: 1, Width: 0}, 3))
	fmt.Println()

	fmt.Println(e.NextState(&env.GridState{Height: 2, Width: 3}, 0))
	fmt.Println(e.NextState(&env.GridState{Height: 2, Width: 3}, 1))
	fmt.Println(e.NextState(&env.GridState{Height: 2, Width: 3}, 2))
	fmt.Println(e.NextState(&env.GridState{Height: 2, Width: 3}, 3))

	// Output:
	// &{0 0}
	// &{1 0}
	// &{0 0}
	// &{0 1}
	//
	// &{0 0}
	// &{2 0}
	// &{1 0}
	// &{1 0}
	//
	// &{1 3}
	// &{2 3}
	// &{2 2}
	// &{2 3}
}

func ExampleGridState() {
	s0 := &env.GridState{Height: 0, Width: 0}
	s1 := &env.GridState{Height: 0, Width: 1}
	s2 := &env.GridState{Height: 1, Width: 0}
	s3 := &env.GridState{Height: 1, Width: 1}

	fmt.Println(s0.Equals(s0))
	fmt.Println(s0.Equals(s1))
	fmt.Println(s0.Equals(s2))
	fmt.Println(s0.Equals(s3))

	// Output:
	// true
	// false
	// false
	// false
}
