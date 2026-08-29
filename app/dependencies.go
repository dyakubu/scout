package app

import (
	"log"

	"github.com/dyakubu/scout/indexer"
	"github.com/dyakubu/scout/search"
)

type Dependencies struct {
	FileIndexer *indexer.FileIndexer
	Searcher    *search.Searcher
	Logger      *log.Logger
}
