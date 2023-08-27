package env

import "github.com/itsubaki/neu/math/matrix"

type GridWorld struct {
	ActionSpace   []Action
	ActionMeaning map[Action]string
	RewardMap     matrix.Matrix
	GoalState     *GridState
	WallState     *GridState
	StartState    *GridState
	AgentState    *GridState
	state         []GridState
	counter       int
}

type Action int

type GridState struct {
	Height int
	Width  int
}

func (s *GridState) Equals(o *GridState) bool {
	return s.Height == o.Height && s.Width == o.Width
}

func NewGridWorld() *GridWorld {
	w := &GridWorld{
		ActionSpace: []Action{0, 1, 2, 3},
		ActionMeaning: map[Action]string{
			0: "UP",
			1: "RIGHT",
			2: "DOWN",
			3: "LEFT",
		},
		RewardMap: matrix.Matrix{
			{0, 0, 0, 1},
			{0, 0, 0, -1},
			{0, 0, 0, 0},
		},
		GoalState:  &GridState{Height: 0, Width: 3},
		WallState:  &GridState{Height: 1, Width: 1},
		StartState: &GridState{Height: 2, Width: 0},
		AgentState: &GridState{Height: 2, Width: 0},
		state:      make([]GridState, 0),
		counter:    -1,
	}

	for y := 0; y < w.Height(); y++ {
		for x := 0; x < w.Width(); x++ {
			w.state = append(w.state, GridState{Height: y, Width: x})
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

func (w *GridWorld) Shape() (int, int) {
	return w.Height(), w.Width()
}

func (w *GridWorld) Actions() []Action {
	return w.ActionSpace
}

func (w *GridWorld) State() *GridState {
	w.counter++
	if w.counter > len(w.state)-1 {
		return nil
	}

	return &w.state[w.counter]
}

func (w *GridWorld) NextState(s *GridState, a Action) *GridState {
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

func (w *GridWorld) Reward(s *GridState, a Action, n *GridState) float64 {
	return w.RewardMap[n.Height][n.Width]
}

func (w *GridWorld) Reset() *GridState {
	w.counter = -1
	w.AgentState = w.StartState
	return w.AgentState
}

func (w *GridWorld) Step(a Action) (*GridState, float64, bool) {
	s := w.AgentState
	n := w.NextState(s, a)
	r := w.Reward(s, a, n)
	done := n.Equals(w.GoalState)

	w.AgentState = n
	return n, r, done
}
