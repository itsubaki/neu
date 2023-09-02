package agent

import "fmt"

type DefaultMap[T any] map[string]T

func Get[T any](m DefaultMap[T], key fmt.Stringer, defaultValue T) T {
	k := key.String()
	if _, ok := m[k]; !ok {
		m[k] = defaultValue
	}

	return m[k]
}
