package mnist_test

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"os"
	"testing"

	"github.com/itsubaki/neu/dataset/mnist"
)

func ExampleLoad() {
	train, test := mnist.Must(mnist.Load("../../testdata"))

	fmt.Println(train.N)
	fmt.Println(test.N)

	fmt.Println(len(train.Image))
	fmt.Println(len(train.Image[0]))
	fmt.Println(len(mnist.OneHot(train.Label)[0]))

	// Output:
	// 60000
	// 10000
	// 60000
	// 784
	// 10

}

func ExampleLoad_notfound() {
	_, _, err := mnist.Load("invalid_dir")
	fmt.Println(err)

	// Output:
	// load training data: load=invalid_dir/train-images-idx3-ubyte.gz: open file=invalid_dir/train-images-idx3-ubyte.gz: open invalid_dir/train-images-idx3-ubyte.gz: no such file or directory
}

func ExampleLoadFile_invalidSize() {
	image := fmt.Sprintf("../../testdata/%s", mnist.TrainImageGZ)
	label := fmt.Sprintf("../../testdata/%s", mnist.TestLabelGZ)
	_, err := mnist.LoadFile(image, label)
	fmt.Println(err)

	// Output:
	// invalid size. image=60000, labels=10000
}

func ExampleLoadFile_invalidImageHeader() {
	image := fmt.Sprintf("../../testdata/%s", mnist.TrainLabelGZ)
	label := fmt.Sprintf("../../testdata/%s", mnist.TrainLabelGZ)
	_, err := mnist.LoadFile(image, label)
	fmt.Println(err)

	// Output:
	// load=../../testdata/train-labels-idx1-ubyte.gz: invalid header={2049 60000 83887105 151126275}
}

func ExampleLoadFile_invalidLabelHeader() {
	image := fmt.Sprintf("../../testdata/%s", mnist.TrainImageGZ)
	label := fmt.Sprintf("../../testdata/%s", mnist.TrainImageGZ)
	_, err := mnist.LoadFile(image, label)
	fmt.Println(err)

	// Output:
	// load=../../testdata/train-images-idx3-ubyte.gz: invalid header={2051 60000}
}

func ExampleLoadFile() {
	image := fmt.Sprintf("../../testdata/%s", mnist.TrainImageGZ)
	_, err := mnist.LoadFile(image, "invalid_file")
	fmt.Println(err)

	// Output:
	// load=invalid_file: open file=invalid_file: open invalid_file: no such file or directory
}

func ExampleNormalize() {
	img := []mnist.Image{{byte(0)}, {byte(10)}, {byte(20)}, {byte(255)}}
	for _, r := range mnist.Normalize(img) {
		fmt.Printf("%.4f\n", r[0])
	}

	// Output:
	// 0.0000
	// 0.0392
	// 0.0784
	// 1.0000
}

func ExampleOneHot() {
	label := []mnist.Label{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	for _, r := range mnist.OneHot(label) {
		fmt.Printf("%.0f\n", r)
	}

	// Output:
	// [1 0 0 0 0 0 0 0 0 0]
	// [0 1 0 0 0 0 0 0 0 0]
	// [0 0 1 0 0 0 0 0 0 0]
	// [0 0 0 1 0 0 0 0 0 0]
	// [0 0 0 0 1 0 0 0 0 0]
	// [0 0 0 0 0 1 0 0 0 0]
	// [0 0 0 0 0 0 1 0 0 0]
	// [0 0 0 0 0 0 0 1 0 0]
	// [0 0 0 0 0 0 0 0 1 0]
	// [0 0 0 0 0 0 0 0 0 1]
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

	mnist.Must(nil, nil, fmt.Errorf("something went wrong"))
	t.Fail()
}

func TestLoadImageGzip(t *testing.T) {
	invalid := []byte("this is not a gzip file.")

	file := "invalid.gz"
	if err := os.WriteFile(file, invalid, 0644); err != nil {
		t.Fatalf("write invalid file: %v", err)
	}
	defer os.Remove(file)

	if _, err := mnist.LoadImage(file); err != nil {
		return
	}

	t.Fatal("unexpected")
}

func TestLoadImageHeader(t *testing.T) {
	invalid := []byte{0x00, 0x08, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}

	buf := new(bytes.Buffer)
	w := gzip.NewWriter(buf)
	if _, err := w.Write(invalid); err != nil {
		t.Fatalf("write gzip data: %v", err)
	}
	w.Close()

	file := "invalid.gz"
	if err := os.WriteFile(file, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write invalid file: %v", err)
	}
	defer os.Remove(file)

	if _, err := mnist.LoadImage(file); err != nil {
		return
	}

	t.Fatal("unexpected")
}

func TestLoadLabelGzip(t *testing.T) {
	invalid := []byte("this is not a gzip file.")

	file := "invalid.gz"
	if err := os.WriteFile(file, invalid, 0644); err != nil {
		t.Fatalf("write invalid file: %v", err)
	}
	defer os.Remove(file)

	if _, err := mnist.LoadLabel(file); err != nil {
		return
	}

	t.Fatal("unexpected")
}

func TestLoadLabelHeader(t *testing.T) {
	invalid := []byte{0x00, 0x08, 0x01, 0x00, 0x00, 0x00}

	buf := new(bytes.Buffer)
	w := gzip.NewWriter(buf)
	if _, err := w.Write(invalid); err != nil {
		t.Fatalf("write gzip data: %v", err)
	}
	w.Close()

	file := "invalid.gz"
	if err := os.WriteFile(file, buf.Bytes(), 0644); err != nil {
		t.Fatalf("write invalid file: %v", err)
	}
	defer os.Remove(file)

	if _, err := mnist.LoadLabel(file); err != nil {
		return
	}

	t.Fatal("unexpected")
}
