library(Seurat)

data <- readRDS('/home/swier/Downloads/fourCancers.rds')


#duplicate_rows = rownames(data$counts)[duplicated(rownames(data$counts))]
counts <- data$counts[!duplicated(rownames(data$counts)),]

ds <- CreateSeuratObject(counts = counts)

ds[["percent.mt"]] <- PercentageFeatureSet(ds, pattern = "^MT-")

# filter on unique genes, as well as % mitochondrial rna
ds <- subset(ds, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# logNormalize the data
ds <- NormalizeData(ds, normalization.method = "LogNormalize", scale.factor = 10000)


ds_var <- FindVariableFeatures(ds, selection.method = "vst", nfeatures = 2000)