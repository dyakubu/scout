package search

import (
	"fmt"
	"testing"
)

// candidates builds a ranked list from (path, score) pairs, highest first,
// the way the scan hands them over.
func candidates(pairs ...any) []Result {
	var out []Result
	for i := 0; i < len(pairs); i += 2 {
		out = append(out, Result{
			Path:  pairs[i].(string),
			Score: pairs[i+1].(float64),
		})
	}
	return out
}

func paths(results []Result) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i] = r.Path
	}
	return out
}

func TestDiversify_SpreadsAcrossFiles(t *testing.T) {
	// One long document owns the top of the ranking. Without a cap it
	// takes every slot and the query returns nothing else.
	got := diversify(candidates(
		"/a.md", 0.90,
		"/a.md", 0.89,
		"/a.md", 0.88,
		"/a.md", 0.87,
		"/b.md", 0.60,
		"/c.md", 0.55,
	), 4, 2)

	want := []string{"/a.md", "/a.md", "/b.md", "/c.md"}
	if fmt.Sprint(paths(got)) != fmt.Sprint(want) {
		t.Errorf("paths = %v, want %v", paths(got), want)
	}
}

// A query that genuinely matches one document shouldn't be padded with
// unrelated files just to look diverse - the leftover slots come from the
// same file once everyone else has had a turn.
func TestDiversify_BackfillsWhenNothingElseMatches(t *testing.T) {
	got := diversify(candidates(
		"/a.md", 0.90,
		"/a.md", 0.89,
		"/a.md", 0.88,
		"/a.md", 0.87,
		"/a.md", 0.86,
	), 4, 2)

	if len(got) != 4 {
		t.Fatalf("got %d results, want 4 - the cap shouldn't cost results when there's nothing to diversify into", len(got))
	}
	for _, r := range got {
		if r.Path != "/a.md" {
			t.Errorf("unexpected path %q", r.Path)
		}
	}
}

// Diversification decides which results come back, not what order they're
// shown in - the output stays ranked by score.
func TestDiversify_OutputStaysScoreOrdered(t *testing.T) {
	got := diversify(candidates(
		"/a.md", 0.90,
		"/a.md", 0.80,
		"/a.md", 0.70,
		"/b.md", 0.60,
	), 4, 2)

	for i := 1; i < len(got); i++ {
		if got[i].Score > got[i-1].Score {
			t.Errorf("results out of score order at %d: %v", i, got)
		}
	}
}

func TestDiversify_Edges(t *testing.T) {
	all := candidates("/a.md", 0.9, "/a.md", 0.8, "/b.md", 0.7)

	t.Run("no cap returns pure top-k", func(t *testing.T) {
		got := diversify(all, 2, 0)
		if len(got) != 2 || got[0].Score != 0.9 || got[1].Score != 0.8 {
			t.Errorf("got %v, want the two highest scores", paths(got))
		}
	})

	t.Run("fewer candidates than max", func(t *testing.T) {
		if got := diversify(all, 10, 1); len(got) != 3 {
			t.Errorf("got %d results, want all 3", len(got))
		}
	})

	t.Run("max of zero", func(t *testing.T) {
		if got := diversify(all, 0, 2); len(got) != 0 {
			t.Errorf("got %d results, want none", len(got))
		}
	})

	t.Run("no candidates", func(t *testing.T) {
		if got := diversify(nil, 5, 2); len(got) != 0 {
			t.Errorf("got %d results, want none", len(got))
		}
	})
}
