import kagglehub

dataset_path = kagglehub.dataset_download(
    "grouplens/movielens-20m-dataset"
)

print("Dataset downloaded successfully!")
print("Dataset location:")
print(dataset_path)
