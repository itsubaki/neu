package agent

type Deque[T any] struct {
	data []T
	size int
}

func NewDeque[T any](size int) *Deque[T] {
	return &Deque[T]{
		data: make([]T, 0),
		size: size,
	}
}

func (d *Deque[T]) Len() int {
	return len(d.data)
}

func (d *Deque[T]) Size() int {
	return d.size
}

func (d *Deque[T]) Append(m T) {
	if len(d.data) == d.size {
		d.data = d.data[1:]
	}

	d.data = append(d.data, m)
}

func (d *Deque[T]) Get(i int) T {
	return d.data[i]
}
