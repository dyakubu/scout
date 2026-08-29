package commands

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
	"github.com/dyakubu/scout/search"
)

func Find(ctx context.Context, args cli.ParsedArgs, deps app.Dependencies) error {

	if len(args.Positional) == 0 {
		return fmt.Errorf("find command requires a query to search for")
	}

	if len(args.Positional) > 1 {
		return fmt.Errorf("find accepts at most one query, got %d: %v", len(args.Positional), args.Positional)
	}

	query := args.Positional[0]

	max := 5

	if value, ok := args.Flags["max"]; ok {
		parsedMax, err := strconv.Atoi(value)

		if err != nil {
			return fmt.Errorf("invalid value %q for --max: expected an integer", value)
		}

		max = parsedMax
	}

	// --restrict alone (no value) means "restrict to the current
	// directory"; --restrict=<path> means "restrict to that directory".
	restrict := args.Flags["restrict"]
	if restrict == "true" {
		restrict = "."
	}

	restrict, err := cli.ExpandHome(restrict)

	if err != nil {
		return err
	}

	results, err := deps.Searcher.Search(query, search.Options{Max: max, Restrict: restrict})

	if err != nil {
		return err
	}

	if len(results) == 0 {
		fmt.Println("no results found")
		return nil
	}

	for _, result := range results {
		fmt.Printf("%s:%d-%d  (score: %.2f)\n", result.Path, result.StartLine, result.EndLine, result.Score)
		fmt.Printf("    %s\n\n", snippet(result.Content))
	}

	return nil
}

// snippet collapses a chunk's content to a single readable preview line.
func snippet(content string) string {
	collapsed := strings.Join(strings.Fields(content), " ")

	const maxLen = 150
	runes := []rune(collapsed)
	if len(runes) > maxLen {
		return string(runes[:maxLen]) + "..."
	}

	return collapsed
}
