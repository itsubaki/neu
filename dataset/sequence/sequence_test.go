package sequence_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/dataset/sequence"
)

func ExampleLoad() {
	s := rand.NewSource(1)
	x, t, v := sequence.Must(sequence.Load("../../testdata", sequence.Addition, s))

	fmt.Println(len(x.Train), len(x.Train[0]), len(x.Test), len(x.Test[0]))
	fmt.Println(len(t.Train), len(t.Train[0]), len(t.Test), len(t.Test[0]))
	fmt.Println(len(v.IDToWord), len(v.WordToID))
	fmt.Println(x.Train[0])
	fmt.Println(t.Train[0])
	fmt.Println(v.ToWord(x.Train[0]), v.ToWord(t.Train[0]))
	fmt.Println(v.ToWord(x.Train[9]), v.ToWord(t.Train[9]))

	// Output:
	// 45000 7 5000 7
	// 45000 5 5000 5
	// 13 13
	// [4 10 1 2 11 11 5]
	// [6 1 8 12 5]
	// [5 3 6 + 8 8  ] [_ 6 2 4  ]
	// [7 9 6 + 1 0 1] [_ 8 9 7  ]
}
