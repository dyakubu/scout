package search

import (
	"bytes"
	"errors"
	"log"
	"strings"
	"testing"
)

// The real trap (a wasm "out of bounds memory access" from sqlite-vec)
// can't be reproduced on demand - it's intermittent and amd64-only - so
// these inject a panic to exercise the same path.

func newBufLogger() (*log.Logger, *bytes.Buffer) {
	buf := &bytes.Buffer{}
	return log.New(buf, "", 0), buf
}

func TestRetryVecQuery_NoRetryOnSuccess(t *testing.T) {
	logger, buf := newBufLogger()
	calls := 0

	got, err := retryVecQuery(logger, "vec_chunks", func() ([]int, error) {
		calls++
		return []int{1, 2}, nil
	})

	if err != nil {
		t.Fatalf("retryVecQuery: %v", err)
	}
	if calls != 1 {
		t.Errorf("called %d times, want 1", calls)
	}
	if len(got) != 2 {
		t.Errorf("got %v, want the query's results", got)
	}
	if buf.Len() != 0 {
		t.Errorf("logged %q on a clean run, want nothing", buf.String())
	}
}

func TestRetryVecQuery_RecoversFromTrap(t *testing.T) {
	logger, buf := newBufLogger()
	calls := 0

	got, err := retryVecQuery(logger, "vec_chunks", func() ([]int, error) {
		calls++
		if calls < 3 {
			panic("wasm error: out of bounds memory access")
		}
		return []int{7}, nil
	})

	if err != nil {
		t.Fatalf("retryVecQuery: %v, want the third attempt to succeed", err)
	}
	if calls != 3 {
		t.Errorf("called %d times, want 3", calls)
	}
	if len(got) != 1 || got[0] != 7 {
		t.Errorf("got %v, want the successful attempt's results", got)
	}
	if n := strings.Count(buf.String(), "trapped on attempt"); n != 2 {
		t.Errorf("logged %d traps, want 2 - each one should be visible", n)
	}
}

func TestRetryVecQuery_GivesUpAfterThree(t *testing.T) {
	logger, _ := newBufLogger()
	calls := 0

	_, err := retryVecQuery(logger, "vec_media", func() ([]int, error) {
		calls++
		panic("wasm error: out of bounds memory access")
	})

	if err == nil {
		t.Fatal("retryVecQuery: want an error once every attempt trapped")
	}
	if calls != vecAttempts {
		t.Errorf("called %d times, want %d", calls, vecAttempts)
	}
	// The error has to carry both what failed and what actually went wrong.
	for _, want := range []string{"vec_media", "out of bounds"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q missing %q", err, want)
		}
	}
}

// A query error is a real failure, not a trap - retrying it just fails
// three times more slowly.
func TestRetryVecQuery_DoesNotRetryOrdinaryErrors(t *testing.T) {
	logger, _ := newBufLogger()
	calls := 0
	sentinel := errors.New("no such table")

	_, err := retryVecQuery(logger, "vec_chunks", func() ([]int, error) {
		calls++
		return nil, sentinel
	})

	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want the query's own error returned unwrapped", err)
	}
	if calls != 1 {
		t.Errorf("called %d times, want 1", calls)
	}
}

// A panic mid-scan must not strand the connection it borrowed, or the
// retries would queue behind it until the pool is exhausted.
func TestRetryVecQuery_PanicDoesNotLeakConnections(t *testing.T) {
	db := newTestDB(t)
	db.SetMaxOpenConns(2)

	logger, _ := newBufLogger()
	calls := 0

	got, err := retryVecQuery(logger, "vec_chunks", func() ([]string, error) {
		calls++
		rows, err := db.Query(`SELECT name FROM sqlite_master`)
		if err != nil {
			return nil, err
		}
		defer rows.Close()

		var names []string
		for rows.Next() {
			var name string
			if err := rows.Scan(&name); err != nil {
				return nil, err
			}
			// Trap partway through iterating, the way the real one does.
			if calls < 3 && len(names) == 1 {
				panic("wasm error: out of bounds memory access")
			}
			names = append(names, name)
		}
		return names, rows.Err()
	})

	if err != nil {
		t.Fatalf("retryVecQuery: %v", err)
	}
	if calls != 3 {
		t.Errorf("called %d times, want 3", calls)
	}
	if len(got) == 0 {
		t.Error("no rows returned by the attempt that succeeded")
	}

	// The pool is only two deep; if the trapped attempts had stranded their
	// connections this would block rather than return.
	if err := db.Ping(); err != nil {
		t.Errorf("database unusable after trapped attempts: %v", err)
	}
	if stats := db.Stats(); stats.InUse != 0 {
		t.Errorf("%d connection(s) still checked out after the retries", stats.InUse)
	}
}
