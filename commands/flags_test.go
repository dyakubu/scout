package commands

import (
	"testing"

	"github.com/dyakubu/scout/cli"
)

func TestBoolFlag_Default(t *testing.T) {
	args := cli.ParsedArgs{Flags: map[string]string{}}

	got, err := boolFlag(args, "recursive", true)
	if err != nil {
		t.Fatalf("boolFlag: %v", err)
	}
	if got != true {
		t.Errorf("boolFlag = %v, want default true", got)
	}
}

func TestBoolFlag_ExplicitValues(t *testing.T) {
	tests := []struct {
		value string
		want  bool
	}{
		{"true", true},
		{"false", false},
	}

	for _, tt := range tests {
		args := cli.ParsedArgs{Flags: map[string]string{"recursive": tt.value}}
		got, err := boolFlag(args, "recursive", false)
		if err != nil {
			t.Errorf("boolFlag(%q): %v", tt.value, err)
			continue
		}
		if got != tt.want {
			t.Errorf("boolFlag(%q) = %v, want %v", tt.value, got, tt.want)
		}
	}
}

func TestBoolFlag_InvalidValue(t *testing.T) {
	args := cli.ParsedArgs{Flags: map[string]string{"recursive": "yes"}}
	if _, err := boolFlag(args, "recursive", true); err == nil {
		t.Error("boolFlag with an invalid value: expected an error, got none")
	}
}
