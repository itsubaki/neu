package ptb_test

import (
	"fmt"
	"testing"

	"github.com/itsubaki/neu/dataset/ptb"
)

func ExamplePreProcess() {
	text := "You say goodbye and I say hello ."
	corpus, w2id, id2w := ptb.PreProcess(text)

	fmt.Println(corpus)
	fmt.Println()

	for k, v := range w2id {
		fmt.Printf("%v: %v\n", k, v)
	}
	fmt.Println()

	for k, v := range id2w {
		fmt.Printf("%v: %v\n", k, v)
	}

	// Unordered output:
	// [0 1 2 3 4 1 5 6]
	//
	// You: 0
	// say: 1
	// goodbye: 2
	// and: 3
	// I: 4
	// hello: 5
	// .: 6
	//
	// 0: You
	// 1: say
	// 2: goodbye
	// 3: and
	// 4: I
	// 5: hello
	// 6: .
}

func ExampleCreateContextsTarget() {
	corpus := []int{0, 1, 2, 3, 4, 1, 5, 6}
	contexts, targets := ptb.CreateContextsTarget(corpus, 1)

	for i := range contexts {
		fmt.Printf("%v: %v\n", contexts[i], targets[i])
	}

	// Output:
	// [0 2]: 1
	// [1 3]: 2
	// [2 4]: 3
	// [3 1]: 4
	// [4 5]: 1
	// [1 6]: 5
}

func ExampleLoad() {
	train := ptb.Must(ptb.Load("../../testdata", ptb.TrainTxt))

	fmt.Println(len(train.Corpus))
	fmt.Println(train.Corpus[:20])
	fmt.Println(train.IDToWord[0])
	fmt.Println(train.IDToWord[1])
	fmt.Println(train.IDToWord[2])
	fmt.Println(train.WordToID["car"])
	fmt.Println(train.WordToID["happy"])
	fmt.Println(train.WordToID["lexus"])
	fmt.Println()

	test := ptb.Must(ptb.Load("../../testdata", ptb.TestTxt))
	fmt.Println(len(test.Corpus))
	fmt.Println(test.Corpus[:20])
	fmt.Println(test.IDToWord[0])
	fmt.Println(test.IDToWord[1])
	fmt.Println(test.IDToWord[2])
	fmt.Println(test.WordToID["car"])
	fmt.Println(test.WordToID["happy"])
	fmt.Println(test.WordToID["lexus"])
	fmt.Println()

	valid := ptb.Must(ptb.Load("../../testdata", ptb.ValidTxt))
	fmt.Println(len(valid.Corpus))
	fmt.Println(valid.Corpus[:20])
	fmt.Println(valid.IDToWord[0])
	fmt.Println(valid.IDToWord[1])
	fmt.Println(valid.IDToWord[2])
	fmt.Println(valid.WordToID["car"])
	fmt.Println(valid.WordToID["happy"])
	fmt.Println(valid.WordToID["lexus"])

	// Output:
	// 929589
	// [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]
	// aer
	// banknote
	// berlitz
	// 3856
	// 4428
	// 7426
	//
	// 82430
	// [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 3 15 16 17 18]
	// no
	// it
	// was
	// 2611
	// 2981
	// 5098
	//
	// 73760
	// [0 1 2 3 4 5 6 7 8 9 3 10 11 12 13 14 14 15 16 17]
	// consumers
	// may
	// want
	// 1801
	// 2014
	// 1826
}

func ExampleLoad_notfound() {
	_, err := ptb.Load("invalid_dir", "invlid_file")
	fmt.Println(err)

	// Output:
	// open file=invalid_dir/invlid_file: open invalid_dir/invlid_file: no such file or directory
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

	ptb.Must(nil, fmt.Errorf("something went wrong"))
	t.Fail()
}
