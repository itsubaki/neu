package env

import "github.com/itsubaki/neu/math/matrix"

type GridWorld struct {
	ActionSpace   []int
	ActionMeaning map[int]string
	RewardMap     matrix.Matrix
	GoalState     *GridState
	WallState     *GridState
	StartState    *GridState
	AgentState    *GridState
	state         []GridState
	counter       int
}

type GridState struct {
	Height int
	Width  int
}

func NewGridWorld() *GridWorld {
	w := &GridWorld{
		ActionSpace: []int{0, 1, 2, 3},
		ActionMeaning: map[int]string{
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

func (w *GridWorld) Actions() []int {
	return w.ActionSpace
}

func (w *GridWorld) State() *GridState {
	w.counter++
	if w.counter > len(w.state) {
		return nil
	}

	return &w.state[w.counter]
}

func (w *GridWorld) Reward(s *GridState, a int, n *GridState) float64 {
	return w.RewardMap[n.Height][n.Width]
}
