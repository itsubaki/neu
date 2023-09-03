package env

import "fmt"

type GridWorld struct {
	ActionSpace   []int
	ActionMeaning map[int]string
	RewardMap     [][]float64
	GoalState     *GridState
	WallState     *GridState
	StartState    *GridState
	AgentState    *GridState
	State         []GridState
}

type GridState struct {
	Height int
	Width  int
}

func (s *GridState) Equals(o *GridState) bool {
	return s.Height == o.Height && s.Width == o.Width
}

func (s GridState) String() string {
	return fmt.Sprintf("(%d, %d)", s.Height, s.Width)
}

func NewGridWorld() *GridWorld {
	w := &GridWorld{
		ActionSpace: []int{0, 1, 2, 3},
		ActionMeaning: map[int]string{
			0: "UP",
			1: "DOWN",
			2: "LEFT",
			3: "RIGHT",
		},
		RewardMap: [][]float64{
			{0, 0, 0, 1},
			{0, 0, 0, -1},
			{0, 0, 0, 0},
		},
		GoalState:  &GridState{Height: 0, Width: 3},
		WallState:  &GridState{Height: 1, Width: 1},
		StartState: &GridState{Height: 2, Width: 0},
		AgentState: &GridState{Height: 2, Width: 0},
		State:      make([]GridState, 0),
	}

	for y := 0; y < w.Height(); y++ {
		for x := 0; x < w.Width(); x++ {
			w.State = append(w.State, GridState{Height: y, Width: x})
		}
	}

	return w
}

func (w *GridWorld) Height() int {
	return len(w.RewardMap)
}

func (w *GridWorld) Width() int {
	return len(w.RewardMap[0])
}

func (w *GridWorld) Size() int {
	return w.Height() * w.Width()
}

func (w *GridWorld) Actions() []int {
	return w.ActionSpace
}

func (w *GridWorld) NextState(s *GridState, a int) *GridState {
	moveMap := []GridState{
		{Height: -1, Width: 0},
		{Height: 1, Width: 0},
		{Height: 0, Width: -1},
		{Height: 0, Width: 1},
	}

	move := moveMap[a]
	next := &GridState{
		Height: s.Height + move.Height,
		Width:  s.Width + move.Width,
	}

	// out of range
	if next.Height < 0 || next.Height >= w.Height() {
		next = s
	}
	if next.Width < 0 || next.Width >= w.Width() {
		next = s
	}

	// wall
	if next.Equals(w.WallState) {
		next = s
	}

	return next
}

func (w *GridWorld) Reward(s *GridState, a int, n *GridState) float64 {
	return w.RewardMap[n.Height][n.Width]
}

func (w *GridWorld) Reset() *GridState {
	w.AgentState = w.StartState
	return w.AgentState
}

func (w *GridWorld) Step(a int) (*GridState, float64, bool) {
	s := w.AgentState
	n := w.NextState(s, a)
	r := w.Reward(s, a, n)
	done := n.Equals(w.GoalState)

	w.AgentState = n
	return n, r, done
}

func (w *GridWorld) OneHot(state *GridState) []float64 {
	out := make([]float64, w.Height()*w.Width())
	out[w.Width()*state.Height+state.Width] = 1.0
	return out
}
