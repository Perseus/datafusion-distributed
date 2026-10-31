use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::stream::BoxStream;
use iceberg::Result;
use iceberg::io::{
    FileMetadata, FileRead, FileWrite, InputFile, LocalFsStorage, OutputFile, Storage,
    StorageConfig, StorageFactory,
};
use serde::{Deserialize, Serialize};

/// Local-file storage that counts how many times a data file is opened for reading, to check
/// which files a scan reads. Manifests, metadata files and writes aren't counted.
///
/// A data file is a `.parquet` file under a table's `data/` directory. Delete files live there
/// too, so tables checked this way shouldn't have any.
///
/// The count isn't serialized: a factory or storage that is deserialized counts on its own.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CountingStorageFactory {
    #[serde(skip)]
    data_file_reads: Arc<AtomicUsize>,
}

impl CountingStorageFactory {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many times a data file has been opened for reading through storages this factory
    /// built.
    pub fn data_file_reads(&self) -> usize {
        self.data_file_reads.load(Ordering::SeqCst)
    }
}

#[typetag::serde]
impl StorageFactory for CountingStorageFactory {
    fn build(&self, _config: &StorageConfig) -> Result<Arc<dyn Storage>> {
        let data_file_reads = Arc::clone(&self.data_file_reads);
        Ok(Arc::new_cyclic(|this| CountingStorage {
            this: this.clone(),
            data_file_reads,
        }))
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct CountingStorage {
    /// This storage, for the files it hands out, so their reads are counted too.
    #[serde(skip)]
    this: Weak<CountingStorage>,
    #[serde(skip)]
    data_file_reads: Arc<AtomicUsize>,
}

impl CountingStorage {
    fn count(&self, path: &str) {
        if path.contains("/data/") && path.ends_with(".parquet") {
            self.data_file_reads.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn arc(&self) -> Arc<dyn Storage> {
        match self.this.upgrade() {
            Some(this) => this,
            None => Arc::new(LocalFsStorage::new()),
        }
    }
}

#[async_trait]
#[typetag::serde]
impl Storage for CountingStorage {
    async fn exists(&self, path: &str) -> Result<bool> {
        LocalFsStorage::new().exists(path).await
    }

    async fn metadata(&self, path: &str) -> Result<FileMetadata> {
        LocalFsStorage::new().metadata(path).await
    }

    async fn read(&self, path: &str) -> Result<Bytes> {
        self.count(path);
        LocalFsStorage::new().read(path).await
    }

    async fn reader(&self, path: &str) -> Result<Box<dyn FileRead>> {
        self.count(path);
        LocalFsStorage::new().reader(path).await
    }

    async fn write(&self, path: &str, bs: Bytes) -> Result<()> {
        LocalFsStorage::new().write(path, bs).await
    }

    async fn writer(&self, path: &str) -> Result<Box<dyn FileWrite>> {
        LocalFsStorage::new().writer(path).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete(path).await
    }

    async fn delete_prefix(&self, path: &str) -> Result<()> {
        LocalFsStorage::new().delete_prefix(path).await
    }

    async fn delete_stream(&self, paths: BoxStream<'static, String>) -> Result<()> {
        LocalFsStorage::new().delete_stream(paths).await
    }

    fn new_input(&self, path: &str) -> Result<InputFile> {
        Ok(InputFile::new(self.arc(), path.to_string()))
    }

    fn new_output(&self, path: &str) -> Result<OutputFile> {
        Ok(OutputFile::new(self.arc(), path.to_string()))
    }
}
