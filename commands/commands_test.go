package commands

import (
	"context"
	"strings"
	"testing"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
)

// These tests cover the argument-validation paths that return an error
// before ever touching deps.FileIndexer/deps.Searcher - both are concrete
// structs requiring a real DB/embedder to construct, so a zero-value
// app.Dependencies{} is deliberately used here to prove validation truly
// happens first (a nil-pointer dereference would panic, not return an
// error, if it didn't).

func TestIndex_TooManyPositionalArgs(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"a", "b"}, Flags: map[string]string{}}

	err := Index(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Index with 2 positional args: expected an error, got none")
	}
	if !strings.Contains(err.Error(), "at most one path") {
		t.Errorf("error = %q, want it to mention \"at most one path\"", err.Error())
	}
}

func TestIndex_InvalidRecursiveFlag(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{}, Flags: map[string]string{"recursive": "sometimes"}}

	err := Index(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Index with --recursive=sometimes: expected an error, got none")
	}
}

func TestFind_NoQuery(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{}, Flags: map[string]string{}}

	err := Find(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Find with no query: expected an error, got none")
	}
	if !strings.Contains(err.Error(), "requires a query") {
		t.Errorf("error = %q, want it to mention \"requires a query\"", err.Error())
	}
}

func TestFind_TooManyPositionalArgs(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"one", "two"}, Flags: map[string]string{}}

	err := Find(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Find with 2 positional args: expected an error, got none")
	}
}

func TestFind_InvalidMaxFlag(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"query"}, Flags: map[string]string{"max": "lots"}}

	err := Find(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Find with --max=lots: expected an error, got none")
	}
	if !strings.Contains(err.Error(), "--max") {
		t.Errorf("error = %q, want it to mention --max", err.Error())
	}
}

func TestFind_InvalidMediaMaxFlag(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"query"}, Flags: map[string]string{"media-max": "lots"}}

	err := Find(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Find with --media-max=lots: expected an error, got none")
	}
	if !strings.Contains(err.Error(), "--media-max") {
		t.Errorf("error = %q, want it to mention --media-max", err.Error())
	}
}

func TestSync_TooManyPositionalArgs(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"a", "b"}, Flags: map[string]string{}}

	err := Sync(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Sync with 2 positional args: expected an error, got none")
	}
}

func TestSync_InvalidRecursiveFlag(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{}, Flags: map[string]string{"recursive": "nope"}}

	err := Sync(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("Sync with --recursive=nope: expected an error, got none")
	}
}

func TestConfig_GetWrongArgCount(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"get"}, Flags: map[string]string{}}

	err := Config(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("config get with no key: expected an error, got none")
	}
}

func TestConfig_SetWrongArgCount(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"set", "key"}, Flags: map[string]string{}}

	err := Config(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("config set with no value: expected an error, got none")
	}
}

func TestConfig_UnknownSubcommand(t *testing.T) {
	args := cli.ParsedArgs{Positional: []string{"frobnicate"}, Flags: map[string]string{}}

	err := Config(context.Background(), args, app.Dependencies{})
	if err == nil {
		t.Fatal("config frobnicate: expected an error, got none")
	}
}

func TestConfig_GetSetRoundTrip(t *testing.T) {
	// Unlike find/index, the config command never touches
	// deps.FileIndexer/deps.Searcher - only the config package's own
	// disk-backed state, which os.UserConfigDir() (transitively, via
	// config.Path) resolves under $HOME.
	t.Setenv("HOME", t.TempDir())

	setArgs := cli.ParsedArgs{Positional: []string{"set", "embedder.batch_size", "16"}, Flags: map[string]string{}}
	if err := Config(context.Background(), setArgs, app.Dependencies{}); err != nil {
		t.Fatalf("config set: %v", err)
	}

	getArgs := cli.ParsedArgs{Positional: []string{"get", "embedder.batch_size"}, Flags: map[string]string{}}
	if err := Config(context.Background(), getArgs, app.Dependencies{}); err != nil {
		t.Fatalf("config get: %v", err)
	}
}

func TestSnippet_CollapsesWhitespaceAndTruncates(t *testing.T) {
	got := snippet("line one\n   line two\t\ttabbed")
	want := "line one line two tabbed"
	if got != want {
		t.Errorf("snippet = %q, want %q", got, want)
	}
}

func TestSnippet_TruncatesLongContent(t *testing.T) {
	long := strings.Repeat("word ", 100)
	got := snippet(long)

	runes := []rune(got)
	if len(runes) != 153 { // 150 + "..."
		t.Errorf("len(snippet) = %d, want 153 (150 chars + \"...\")", len(runes))
	}
	if !strings.HasSuffix(got, "...") {
		t.Errorf("snippet = %q, want it to end in \"...\"", got)
	}
}
