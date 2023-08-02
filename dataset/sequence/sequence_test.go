package sequence_test

import (
	"fmt"
	"math/rand"
	"testing"

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
	fmt.Println(sequence.String(v.ToWord(x.Train[0])), sequence.String(v.ToWord(t.Train[0])))
	fmt.Println(sequence.String(v.ToWord(x.Train[9])), sequence.String(v.ToWord(t.Train[9])))

	// Output:
	// 45000 7 5000 7
	// 45000 5 5000 5
	// 13 13
	// [4 8 1 2 10 10 5]
	// [12 1 6 11 5]
	// [5 3 6 + 8 8  ] [_ 6 2 4  ]
	// [7 9 6 + 1 0 1] [_ 8 9 7  ]
}

func ExampleLoad_rand() {
	x, t, v := sequence.Must(sequence.Load("../../testdata", sequence.Addition))

	fmt.Println(len(x.Train), len(x.Train[0]), len(x.Test), len(x.Test[0]))
	fmt.Println(len(t.Train), len(t.Train[0]), len(t.Test), len(t.Test[0]))
	fmt.Println(len(v.IDToWord), len(v.WordToID))

	// Output:
	// 45000 7 5000 7
	// 45000 5 5000 5
	// 13 13
}

func ExampleLoad_notfound() {
	_, _, _, err := sequence.Load("invalid_dir", "invlid_file")
	fmt.Println(err)

	// Output:
	// open file=invalid_dir/invlid_file: open invalid_dir/invlid_file: no such file or directory
}

func ExampleShuffle() {
	xs, ts := sequence.Shuffle([][]int{{1, 2, 3}, {4, 5, 6}}, [][]int{{7, 8, 9}, {10, 11, 12}})
	fmt.Println(len(xs), len(xs[0]))
	fmt.Println(len(ts), len(ts[0]))

	// Output:
	// 2 3
	// 2 3
}

func TestMust(t *testing.T) {
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

	sequence.Must(nil, nil, nil, fmt.Errorf("something went wrong"))
	t.Fail()
}
