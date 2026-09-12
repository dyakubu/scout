package search

import "sort"

// diversify picks max results from candidates, which arrive ranked by
// score, letting no single file contribute more than perFile of them until
// every other file has had its turn.
//
// Ranking purely by score lets one document own the whole result set: a
// long file has many chunks, several of them will sit near any query it's
// vaguely related to, and if it isn't the document you wanted, the query
// returns nothing useful at all. Capping its share surfaces the runners-up
// from other files instead.
//
// The cap is a preference, not a quota. Once every file has contributed
// its share, whatever is left over fills the remaining slots by score, so
// a query that genuinely matches one document still returns a full set
// rather than being padded with unrelated files to hit a diversity floor.
func diversify(candidates []Result, max, perFile int) []Result {
	if max <= 0 || len(candidates) == 0 {
		return nil
	}
	if perFile <= 0 || len(candidates) <= max {
		return truncate(candidates, max)
	}

	taken := make([]Result, 0, max)
	skipped := make([]Result, 0, len(candidates))
	seen := make(map[string]int, len(candidates))

	for _, c := range candidates {
		if len(taken) == max {
			break
		}
		if seen[c.Path] < perFile {
			seen[c.Path]++
			taken = append(taken, c)
			continue
		}
		skipped = append(skipped, c)
	}

	// Backfill by score from what the cap held back.
	for _, c := range skipped {
		if len(taken) == max {
			break
		}
		taken = append(taken, c)
	}

	// Candidates were ranked before the cap reordered them, so restore
	// score order: diversification decides which results are returned, not
	// the order they're shown in.
	sort.SliceStable(taken, func(i, j int) bool { return taken[i].Score > taken[j].Score })

	return taken
}

func truncate(results []Result, max int) []Result {
	if len(results) > max {
		return results[:max]
	}
	return results
}
