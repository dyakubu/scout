package commands

import (
	"context"
	"fmt"
	"os"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
	"github.com/dyakubu/scout/config"
)

// Config implements "scout config", "scout config path", "scout config get
// <key>", and "scout config set <key> <value>".
func Config(ctx context.Context, args cli.ParsedArgs, deps app.Dependencies) error {
	if len(args.Positional) == 0 {
		return showConfig()
	}

	switch args.Positional[0] {
	case "path":
		return showConfigPath()

	case "get":
		if len(args.Positional) != 2 {
			return fmt.Errorf("usage: scout config get <key>")
		}
		return getConfigValue(args.Positional[1])

	case "set":
		if len(args.Positional) != 3 {
			return fmt.Errorf("usage: scout config set <key> <value>")
		}
		return setConfigValue(args.Positional[1], args.Positional[2])

	default:
		return fmt.Errorf("unknown config subcommand %q (expected: path, get, set, or no arguments)", args.Positional[0])
	}
}

func showConfigPath() error {
	path, err := config.Path()
	if err != nil {
		return err
	}

	fmt.Println(path)
	return nil
}

func showConfig() error {
	path, err := config.Path()
	if err != nil {
		return err
	}

	// Load first so a missing config file gets created before we try to
	// read it back.
	if _, err := config.Load(); err != nil {
		return err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	fmt.Print(string(data))
	return nil
}

func getConfigValue(key string) error {
	cfg, err := config.LoadForEdit()
	if err != nil {
		return err
	}

	value, err := config.Get(cfg, key)
	if err != nil {
		return err
	}

	fmt.Println(value)
	return nil
}

func setConfigValue(key, value string) error {
	cfg, err := config.LoadForEdit()
	if err != nil {
		return err
	}

	if err := config.Set(cfg, key, value); err != nil {
		return err
	}

	if err := config.Save(cfg); err != nil {
		return err
	}

	fmt.Printf("%s = %s\n", key, value)
	return nil
}
