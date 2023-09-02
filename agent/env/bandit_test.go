package env_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/agent/env"
)

func ExampleBandit() {
	bandit := env.NewBandit(10, rand.NewSource(1))

	for i := 0; i < 10; i++ {
		fmt.Print(bandit.Play(i))
	}

	// Output:
	// 1111110001
}

func ExampleNonStatBandit() {
	bandit := env.NewNonStatBandit(10, rand.NewSource(1))

	for i := 0; i < 10; i++ {
		fmt.Print(bandit.Play(i, rand.NewSource(1)))
	}

	// Output:
	// 1110000010
}
