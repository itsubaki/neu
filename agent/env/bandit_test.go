package env_test

import (
	"fmt"

	"github.com/itsubaki/neu/agent/env"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleBandit() {
	bandit := env.NewBandit(10, rand.Const(1))

	for i := range 10 {
		fmt.Print(bandit.Play(i))
	}

	// Output:
	// 1101110011
}

func ExampleNonStatBandit() {
	bandit := env.NewNonStatBandit(10, rand.Const(1))

	for i := range 10 {
		fmt.Print(bandit.Play(i))
	}

	// Output:
	// 1101110010
}
