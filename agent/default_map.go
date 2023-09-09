package agent

import "fmt"

type DefaultMap[T any] map[string]T

func (m DefaultMap[T]) Get(key fmt.Stringer, defaultValue T) T {
	k := key.String()
	if _, ok := m[k]; !ok {
		m[k] = defaultValue
	}

	return m[k]
}
