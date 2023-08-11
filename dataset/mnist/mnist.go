package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path"
	"path/filepath"
)

const (
	TrainImageGZ = "train-images-idx3-ubyte.gz"
	TrainLabelGZ = "train-labels-idx1-ubyte.gz"
	TestImageGZ  = "t10k-images-idx3-ubyte.gz"
	TestLabelGZ  = "t10k-labels-idx1-ubyte.gz"
)

const (
	Width  = 28
	Height = 28
	Labels = 10 // 0 ~ 9
)

type (
	Image [Width * Height]byte
	Label uint8
)

// Dataset is a dataset of MNIST.
type Dataset struct {
	N     int
	Image []Image
	Label []Label
}

func image(fileName string) ([]Image, error) {
	f, err := os.Open(filepath.Clean(fileName))
	if err != nil {
		return nil, fmt.Errorf("open file=%v: %v", fileName, err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("new gzip reader: %v", err)
	}

	type header struct {
		Magic  int32
		N      int32
		Height int32
		Width  int32
	}

	var h header
	if err := binary.Read(gzr, binary.BigEndian, &h); err != nil {
		return nil, fmt.Errorf("read binary: %v", err)
	}

	if h.Magic != 0x00000803 || h.Width != Width || h.Height != Height {
		return nil, fmt.Errorf("invalid header=%v", h)
	}

	images := make([]Image, h.N)
	for i := int32(0); i < h.N; i++ {
		if err := binary.Read(gzr, binary.BigEndian, &images[i]); err != nil {
			return nil, fmt.Errorf("read image[%v]: %v", i, err)
		}
	}

	return images, nil
}

func label(fileName string) ([]Label, error) {
	f, err := os.Open(filepath.Clean(fileName))
	if err != nil {
		return nil, fmt.Errorf("open file=%v: %v", fileName, err)
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("new gzip reader: %v", err)
	}

	type header struct {
		Magic int32
		N     int32
	}

	var h header
	if err := binary.Read(gzr, binary.BigEndian, &h); err != nil {
		return nil, fmt.Errorf("read binary: %v", err)
	}

	if h.Magic != 0x00000801 {
		return nil, fmt.Errorf("invalid header=%v", h)
	}

	labels := make([]Label, h.N)
	for i := int32(0); i < h.N; i++ {
		if err := binary.Read(gzr, binary.BigEndian, &labels[i]); err != nil {
			return nil, fmt.Errorf("read label[%v]: %v", i, err)
		}
	}

	return labels, nil
}

func load(imageFileName, labelFileName string) (*Dataset, error) {
	images, err := image(imageFileName)
	if err != nil {
		return nil, fmt.Errorf("load=%v: %v", imageFileName, err)
	}

	labels, err := label(labelFileName)
	if err != nil {
		return nil, fmt.Errorf("load=%v: %v", labelFileName, err)
	}

	if len(images) != len(labels) {
		return nil, fmt.Errorf("invalid size. image=%v, labels=%v", len(images), len(labels))
	}

	return &Dataset{
		N:     len(images),
		Image: images,
		Label: labels,
	}, nil
}

// Must returns training and test dataset or panic.
func Must(train, test *Dataset, err error) (*Dataset, *Dataset) {
	if err != nil {
		panic(err)
	}

	return train, test
}

// Load returns training and test dataset.
func Load(dir string) (*Dataset, *Dataset, error) {
	train, err := load(
		path.Join(dir, TrainImageGZ),
		path.Join(dir, TrainLabelGZ),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("load training data: %v", err)
	}

	test, err := load(
		path.Join(dir, TestImageGZ),
		path.Join(dir, TestLabelGZ),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("load test data: %v", err)
	}

	return train, test, nil
}

// OneHot returns one-hot vector.
func OneHot(label []Label) [][]float64 {
	out := make([][]float64, len(label))
	for i, v := range label {
		out[i] = make([]float64, Labels) // 0 ~ 9
		out[i][v] = 1.0
	}

	return out
}

// Normalize returns normalized image.
func Normalize(img []Image) [][]float64 {
	out := make([][]float64, len(img))
	for i := range img {
		out[i] = make([]float64, len(img[i]))
		for j := range img[i] {
			out[i][j] = float64(img[i][j]) / float64(math.MaxUint8)
		}
	}

	return out
}
