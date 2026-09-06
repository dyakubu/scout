package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
	"github.com/dyakubu/scout/commands"
	"github.com/dyakubu/scout/config"
	scoutdb "github.com/dyakubu/scout/db"
	"github.com/dyakubu/scout/embedder"
	"github.com/dyakubu/scout/indexer"
	"github.com/dyakubu/scout/mediaworker"
	"github.com/dyakubu/scout/search"

	"database/sql"

	_ "github.com/ncruces/go-sqlite3/driver"
)

// version is stamped at build time via -ldflags "-X main.version=...";
// left as "dev" for local builds that don't pass it.
var version = "dev"

const usage = `
		Usage:
		scout <command> [arguments]

		Commands:
		find       Search indexed files
		index      Index files
		sync       Synchronize the index
		config     View or change scout's configuration
		clean      Remove the search index and log, for a fresh start
		version    Print scout's version
		help       Show help

		`

func main() {

	args := os.Args

	if len(args) < 2 {
		fmt.Println(usage)
		return
	}

	// version and help dont need any dependency setup
	switch args[1] {
	case "version":
		fmt.Println("scout " + version)
		return
	case "help":
		fmt.Println(usage)
		return
	}

	// clean only needs config
	if args[1] == "clean" {
		cfg, err := config.Load()
		if err != nil {
			fmt.Println("unable to load config: ", err)
			return
		}

		if err := commands.Clean(cfg); err != nil {
			fmt.Println(err)
			return
		}

		fmt.Print("clean run completed")
		return
	}

	// Commands that require dependency setup
	commandMap := map[string]commands.Command{
		"find":   commands.Find,
		"index":  commands.Index,
		"sync":   commands.Sync,
		"config": commands.Config,
	}

	cmd, ok := commandMap[args[1]]

	if !ok {
		fmt.Printf("Unrecognized command. %v is not a valid command\n", args[1])
		return
	}

	cfg, err := config.Load()

	if err != nil {
		fmt.Println("unable to load config: ", err)
		return
	}

	logFile, err := config.OpenLog()

	if err != nil {
		fmt.Println("unable to open log file: ", err)
		return
	}

	defer logFile.Close()

	logger := log.New(logFile, "", log.LstdFlags)

	db, err := sql.Open("sqlite3", cfg.DB.Path)

	if err != nil {
		fmt.Println("unable to initialize database: ", err)
		return
	}

	defer db.Close()

	if err := db.Ping(); err != nil {
		fmt.Println("database unavailable. shutting down: ", err)
		return
	}

	//db.SetMaxOpenConns(1)
	err = scoutdb.InitDB(db)

	if err != nil {
		fmt.Println("unable to execute db initialization schema: %w ", err)
		return
	}

	embedderLoadStart := time.Now()

	localEmbedder, err := embedder.NewLocalEmbedder(embedder.EmbedderConfig{
		ModelPath:      cfg.Embedder.ModelPath,
		TokenizerPath:  cfg.Embedder.TokenizerPath,
		OrtLibraryPath: cfg.Embedder.OrtLibraryPath,
		BatchSize:      cfg.Embedder.BatchSize,
	})

	if err != nil {
		fmt.Println("unable to load embedding model: ", err)
		return
	}

	defer localEmbedder.Close()

	logger.Printf("embedder loaded in %s", time.Since(embedderLoadStart))

	// The media worker is optional: with no media.model_dir configured, or
	// if the Python worker fails to start (missing python3, etc.), media
	// files are simply skipped rather than aborting the run - text
	// indexing must keep working regardless.
	//
	// "media/worker.py" is a repo-relative path, not resolved against the
	// binary the way the embedder's asset paths are - there's no packaged
	// distribution story for the media worker yet, so this only works run
	// from the repo root during development.
	var mediaEmbedder embedder.MediaEmbedder
	if cfg.Media.ModelDir != "" {
		mediaClient, err := mediaworker.Start("python3", "media/worker.py", cfg.Media.ModelDir, "media-dummy-v1")
		if err != nil {
			logger.Printf("media worker unavailable, continuing without media support: %v", err)
		} else {
			defer mediaClient.Close()
			mediaEmbedder = mediaClient
		}
	}

	searcher, err := search.NewSearcher(db, localEmbedder, logger)

	if err != nil {
		fmt.Println("unable to initialize searcher: ", err)
		return
	}

	// Construct application dependencies
	ctx := context.Background()
	deps := app.Dependencies{
		FileIndexer: &indexer.FileIndexer{
			Db:            db,
			Embedder:      localEmbedder,
			IndexConfig:   cfg.Index,
			MediaConfig:   cfg.Media,
			MediaEmbedder: mediaEmbedder,
			Logger:        logger,
		},
		Searcher: searcher,
		Logger:   logger,
	}
	parsedArgs, err := cli.Parse(args[2:])

	if err != nil {
		fmt.Println(err)
		return
	}

	err = cmd(ctx, parsedArgs, deps)

	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%v run completed", args[1])

}
