package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
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

	// The media worker is optional, and started only for "index" and
	// "find": it loads its model eagerly, so starting it for a command
	// that never uses it would make every scout command pay for that.
	// Every way it can be unavailable - no media.model_dir, nothing at the
	// path it names, a worker that won't start - skips media files rather
	// than aborting the run. Text indexing and search must keep working.
	//
	// The model id labels the stored vectors, identifying the set they can
	// be compared against, so it names the quantization as well as the
	// weights. Vectors from a different one can't be mixed into a single
	// vec_media scan. Unlike LocalEmbedder.ModelID() it's a caller-chosen
	// string rather than a hash of the model files.
	const mediaModelID = "openai/clip-vit-base-patch32-q4f16-onnx"

	var mediaEmbedder embedder.MediaEmbedder
	if (args[1] == "index" || args[1] == "find") && mediaModelAvailable(cfg.Media.ModelDir, logger) {
		mediaCommand := mediaWorkerCommand(logger)
		mediaClient, err := mediaworker.Start(mediaCommand, cfg.Media.ModelDir, mediaModelID)
		if err != nil {
			logger.Printf("media worker unavailable, continuing without media support: %v", err)
		} else {
			defer mediaClient.Close()
			mediaEmbedder = mediaClient
		}
	}

	searcher, err := search.NewSearcher(db, localEmbedder, mediaEmbedder, cfg.Search, logger)

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

// mediaModelAvailable reports whether media.model_dir is configured and
// present on disk. Checking the directory catches an install without the
// media half of the archive before a worker is spawned for it. Which files
// belong inside is the worker's business (media/clip.py's REQUIRED_FILES),
// and it reports a directory missing any of them itself.
func mediaModelAvailable(modelDir string, logger *log.Logger) bool {
	if modelDir == "" {
		return false
	}

	info, err := os.Stat(modelDir)
	if err != nil {
		logger.Printf("media model directory %s unavailable, continuing without media support: %v", modelDir, err)
		return false
	}

	if !info.IsDir() {
		logger.Printf("media model directory %s is not a directory, continuing without media support", modelDir)
		return false
	}

	return true
}

// mediaWorkerCommand returns the command line to run the media worker
// with: the Python interpreter bundled in scout's release archive, or uv
// in a repo checkout, which has no bundled interpreter.
//
// The bundled tree is scout's own. It resolves its stdlib from the
// location of its executable, so it needs no Python on the machine and
// never appears on a PATH. -E and -s keep the user's PYTHONHOME,
// PYTHONPATH and site-packages out of it while leaving the script's
// directory on sys.path, which worker.py needs to import clip.py.
func mediaWorkerCommand(logger *log.Logger) []string {
	dir, err := execDir()
	if err != nil {
		logger.Printf("resolving scout's own directory for the media worker, falling back to uv: %v", err)
		return devMediaCommand
	}

	worker := filepath.Join(dir, "media", "worker.py")

	python := filepath.Join(dir, "media", "python", "bin", "python3")
	if runtime.GOOS == "windows" {
		// python-build-standalone's Windows layout puts the interpreter at
		// the root of the tree rather than in bin/.
		python = filepath.Join(dir, "media", "python", "python.exe")
	}

	if _, err := os.Stat(python); err != nil {
		logger.Printf("no bundled media worker interpreter at %s, falling back to uv: %v", python, err)
		return devMediaCommand
	}

	return []string{python, "-E", "-s", worker}
}

// devMediaCommand runs the worker out of a repo checkout, through the
// media/ project's uv-managed virtualenv. Its paths are relative to the
// working directory, so it only works when run from the repo root.
var devMediaCommand = []string{"uv", "run", "--project", "media", "media/worker.py"}

// execDir returns the directory containing the running scout binary, with
// symlinks resolved. Assets ship beside the binary, and config resolves
// scoutconfig.toml's relative paths against this same directory.
func execDir() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("resolving executable path: %w", err)
	}

	resolved, err := filepath.EvalSymlinks(exe)
	if err != nil {
		return "", fmt.Errorf("resolving executable symlinks: %w", err)
	}

	return filepath.Dir(resolved), nil
}
