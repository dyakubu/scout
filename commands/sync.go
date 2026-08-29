package commands

import (
	"fmt"
	"path/filepath"
	"context"

	"github.com/dyakubu/scout/cli"
	"github.com/dyakubu/scout/app"
)

type syncCommandOptions struct {
	isRecursive bool
}

func Sync(ctx context.Context, args cli.ParsedArgs, deps app.Dependencies) error {

	if len(args.Positional) > 1 {
		return fmt.Errorf("sync accepts at most one path, got %d: %v", len(args.Positional), args.Positional)
	}

	dir := "."

	if len(args.Positional) == 1 {
		dir = args.Positional[0]
	}

	recurse, err := boolFlag(args, "recursive", true)

	if err != nil {
		return err
	}

	optionalFlags := syncCommandOptions{recurse}

	absPath, err := filepath.Abs(dir)

	if err != nil {
		return fmt.Errorf("Unable to resolve path %v. Error: %v", dir, err.Error())
	}

	return sync(absPath, optionalFlags)


}

func sync(path string, optionalFlags syncCommandOptions) error {

	fmt.Printf("Syncing %v with the following flags: isRecursive: %v", path, optionalFlags.isRecursive)

	return nil

}
