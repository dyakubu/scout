package cli

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestParse_PositionalOnly(t *testing.T) {
	got, err := Parse([]string{"find", "hello world"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := []string{"find", "hello world"}
	if !reflect.DeepEqual(got.Positional, want) {
		t.Errorf("Positional = %v, want %v", got.Positional, want)
	}
	if len(got.Flags) != 0 {
		t.Errorf("Flags = %v, want empty", got.Flags)
	}
}

func TestParse_BareFlag(t *testing.T) {
	got, err := Parse([]string{"--recursive"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Flags["recursive"] != "true" {
		t.Errorf(`Flags["recursive"] = %q, want "true"`, got.Flags["recursive"])
	}
}

func TestParse_FlagWithValue(t *testing.T) {
	got, err := Parse([]string{"--max=10"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Flags["max"] != "10" {
		t.Errorf(`Flags["max"] = %q, want "10"`, got.Flags["max"])
	}
}

func TestParse_FlagValueContainingEquals(t *testing.T) {
	// strings.Cut only splits on the first "=", so a value with its own
	// "=" in it must survive intact.
	got, err := Parse([]string{"--restrict=a=b"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Flags["restrict"] != "a=b" {
		t.Errorf(`Flags["restrict"] = %q, want "a=b"`, got.Flags["restrict"])
	}
}

func TestParse_MixedOrder(t *testing.T) {
	got, err := Parse([]string{"--max=5", "query text", "--restrict"})
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !reflect.DeepEqual(got.Positional, []string{"query text"}) {
		t.Errorf("Positional = %v, want [query text]", got.Positional)
	}
	if got.Flags["max"] != "5" || got.Flags["restrict"] != "true" {
		t.Errorf("Flags = %v, want max=5 restrict=true", got.Flags)
	}
}

func TestParse_EmptyFlagName(t *testing.T) {
	if _, err := Parse([]string{"--"}); err == nil {
		t.Error("Parse(\"--\"): expected an error for a missing flag name, got none")
	}
	if _, err := Parse([]string{"--=value"}); err == nil {
		t.Error("Parse(\"--=value\"): expected an error for a missing flag name, got none")
	}
}

func TestParse_NoArgs(t *testing.T) {
	got, err := Parse(nil)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(got.Positional) != 0 || len(got.Flags) != 0 {
		t.Errorf("Parse(nil) = %+v, want empty", got)
	}
}

func TestExpandHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	// os.UserHomeDir() (which ExpandHome uses) checks %USERPROFILE% on
	// Windows, not $HOME - both need to be set for this to be sandboxed
	// consistently across platforms.
	t.Setenv("USERPROFILE", home)

	tests := []struct {
		in   string
		want string
	}{
		{"~", home},
		// filepath.Join, not string concatenation - ExpandHome joins with
		// filepath.Join, which uses "\" on Windows, not "/".
		{"~/Documents", filepath.Join(home, "Documents")},
		{"/absolute/path", "/absolute/path"},
		{"relative/path", "relative/path"},
		{"", ""},
		{"~notquitehome", "~notquitehome"}, // "~foo" isn't "~" or "~/..." - left alone
	}

	for _, tt := range tests {
		got, err := ExpandHome(tt.in)
		if err != nil {
			t.Errorf("ExpandHome(%q): %v", tt.in, err)
			continue
		}
		if got != tt.want {
			t.Errorf("ExpandHome(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
