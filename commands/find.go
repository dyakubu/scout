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

	// max/mediaMax are left at 0 (rather than a hardcoded default here) so
	// Search falls back to whatever default the Searcher was configured
	// with (see search.NewSearcher) when a flag isn't given.
	var max, mediaMax int

	if value, ok := args.Flags["max"]; ok {
		parsedMax, err := strconv.Atoi(value)

		if err != nil {
			return fmt.Errorf("invalid value %q for --max: expected an integer", value)
		}

		max = parsedMax
	}

	if value, ok := args.Flags["media-max"]; ok {
		parsedMediaMax, err := strconv.Atoi(value)

		if err != nil {
			return fmt.Errorf("invalid value %q for --media-max: expected an integer", value)
		}

		mediaMax = parsedMediaMax
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

	results, err := deps.Searcher.Search(query, search.Options{Max: max, MediaMax: mediaMax, Restrict: restrict})

	if err != nil {
		return err
	}

	if len(results.Files) == 0 && len(results.Media) == 0 {
		fmt.Println("no results found")
		return nil
	}

	for _, result := range results.Files {
		fmt.Printf("%s:%d-%d  (score: %.2f)\n", result.Path, result.StartLine, result.EndLine, result.Score)
		fmt.Printf("    %s\n\n", snippet(result.Content))
	}

	// Only printed when there's actually a media result to show - find
	// works exactly as before when media search isn't configured or
	// found nothing.
	if len(results.Media) > 0 {
		fmt.Println("media:")
		for _, result := range results.Media {
			fmt.Printf("  %s  (score: %.2f)\n", result.Path, result.Score)
		}
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
