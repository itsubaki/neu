package agent

type Deque struct {
	data []Memory
	size int
}

func NewDeque(size int) *Deque {
	return &Deque{
		data: make([]Memory, 0),
		size: size,
	}
}

func (d *Deque) Append(m Memory) {
	if len(d.data) == d.size {
		d.data = d.data[1:]
	}

	d.data = append(d.data, m)
}

func (d *Deque) Get(i int) Memory {
	return d.data[i]
}
