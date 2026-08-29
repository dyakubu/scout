package commands

import (
	"context"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
)

type Command func(context.Context, cli.ParsedArgs, app.Dependencies) error
