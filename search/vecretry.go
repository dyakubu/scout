package search

import (
	"fmt"
	"log"
)

// vecAttempts bounds how many times a vec0 query is tried.
const vecAttempts = 3

// retryVecQuery runs scan, retrying if sqlite-vec traps.
//
// sqlite-vec's knn path intermittently raises a wasm "out of bounds memory
// access", which arrives here as a panic through database/sql and would
// otherwise kill the process mid-search. Observed only on amd64, on
// roughly 1 in 8 queries, and only on the first knn query in a process -
// which is every `scout find`. It's an upstream bug in a prebuilt wasm
// binary, so this is containment, not a fix.
//
// Only panics are retried. A returned error is a real query failure and
// retrying it just fails three times more slowly.
func retryVecQuery[T any](logger *log.Logger, table string, scan func() ([]T, error)) ([]T, error) {
	var lastErr error

	for attempt := 1; attempt <= vecAttempts; attempt++ {
		results, trapped, err := attemptVecQuery(scan)
		if !trapped {
			return results, err
		}

		lastErr = err
		logger.Printf("%s query trapped on attempt %d/%d: %v", table, attempt, vecAttempts, err)
	}

	return nil, fmt.Errorf("%s query failed %d times: %w", table, vecAttempts, lastErr)
}

// attemptVecQuery runs scan once, reporting separately whether it panicked
// so the caller knows what's worth retrying.
func attemptVecQuery[T any](scan func() ([]T, error)) (results []T, trapped bool, err error) {
	defer func() {
		if r := recover(); r != nil {
			results, trapped, err = nil, true, fmt.Errorf("sqlite-vec trapped: %v", r)
		}
	}()

	results, err = scan()
	return results, false, err
}
