# Pipeline
Source code for data pipeline to ingest, transform, train and build data storage for Legal AI-system database.
- `/notebooks`: jupyter notebooks that will be used for data pipeline (kaggle notebook for model training)
- `/scripts`: bash scripts for data pipeline
- `/src`: python scripts for data pipeline

## Flow

### L0: Storage Initialization
Sets up the workspace environment by ensuring the `raw`, `bronze`, and `gold` directory structures exist.

### L1: Ingestion (Bronze Layer)
Extracts legal documents from multiple sources/websites. 
- **Format:** Saved as raw `.html` files to preserve original source fidelity.
- **Storage:** `data/bronze/document_htmls/`

### L2: Structured Processing (Silver Layer)
Transforms raw HTML into a cleaned, hierarchical JSON structure.
- **Recursive Pruning:** Controlled by `document_tree_depth`. 
- **Logic:** If a document's natural depth $N$ exceeds `document_tree_depth`, all sub-trees below that threshold are flattened and merged into their respective parent nodes. This ensures the search tree remains manageable for the C++ traversal engine.
- **Storage:** `data/silver/document_jsons/`

### L3: Hierarchy Clustering (Silver Layer)
Generates vector embeddings for all document nodes and applies clustering algorithms to build the high-level navigation tree. This stage defines the "Neural Path" that the searcher will follow.

### L4: Dataset Construction (Silver Layer)
Prepares the training columns (inputs and labels) based on the document hierarchy.
- **Node Sampling:** Every leaf node is treated as an input.
- **Labeling Logic:** Controlled by `build_depth` ($build\_depth < document\_tree\_depth$). For each input, only take `build_depth` labels (id of its parent at each depth)
- **Storage:** `data/silver/datasets/`

### L5: Model Training (Gold Layer)
Trains a series of decoders (nerual networks model)
- **Tagging:** Each model is uniquely tagged with a `node_id` or `cluster_id`.
- **Purpose:** A model at Node $X$ is responsible for classifying which of its children is most relevant to the user's input query.
- **Output:** Exported as **TorchScript (.jit)** for high-performance C++ inference.
- **Storage:** `data/gold/models/`
