# This is an example script showcasing all steps needed to get started quickly
from mmt_retrieval import MultimodalTransformer

# Loading a jointly trained model that can both embed and cross-encode multimodal input
model_path = "https://public.ukp.informatik.tu-darmstadt.de/reimers/mmt-retrieval/models/v1/oscar_join_flickr30k.zip"
model = MultimodalTransformer(model_name_or_path=model_path)

# Image ids are the unique numbers (as string) of each image. If you save the image features separately for each image, this would be the file name
image_ids = ["0", "1", "5"]
# We must load the image features in some way before we can use the model
# Refer to our documentation on more details
feature_folder = "path/to/processed/features"
# Directly load the features from disc. Requires more memory.
# Increase max_workers for more concurrent threads for faster loading with many features
# Remove select to load the entire folder
model.image_dict.load_features_folder(feature_folder, max_workers=1, select=image_ids)
## OR
# Only load the file paths so that features are loaded later just-in-time when there are required.
# Recommended with restricted memory and/ or a lot of images
# Remove select to load the entire folder
model.image_dict.load_file_names(feature_folder, select=image_ids)

sentences = ["The red brown fox jumped over the fence", "A dog being good"]

# Get Embeddings (as a list of numpy arrays)
sentence_embeddings = model.encode(sentences=sentences, convert_to_numpy=True) # convert_to_numpy=True is default
image_embeddings = model.encode(images=image_ids, convert_to_numpy=True)

# Get Pairwise Similarity Matrix (as a tensor)
similarities = model.encode(sentences=sentences, images=image_ids, output_value="logits", convert_to_tensor=True, cross_product_input=True)
similarities = similarities[:,-1].reshape(len(image_ids), len(sentences))



