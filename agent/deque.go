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

func (q *Deque[T]) Len() int {
	return len(q.data)
}

func (q *Deque[T]) Size() int {
	return q.size
}

func (q *Deque[T]) Add(m T) {
	if len(q.data) == q.size {
		q.data = q.data[1:]
	}

	q.data = append(q.data, m)
}

func (q *Deque[T]) Get(i int) T {
	return q.data[i]
}
