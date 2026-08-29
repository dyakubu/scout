package commands

import (
	"fmt"

	"github.com/dyakubu/scout/cli"
)

// boolFlag reads a boolean flag from parsed args, returning def if the flag
// was not provided. It errors if the flag was provided with a value other
// than "true" or "false".
func boolFlag(args cli.ParsedArgs, name string, def bool) (bool, error) {
	value, ok := args.Flags[name]

	if !ok {
		return def, nil
	}

	switch value {
	case "true":
		return true, nil
	case "false":
		return false, nil
	default:
		return false, fmt.Errorf("invalid value %q for --%s: expected true or false", value, name)
	}
}
