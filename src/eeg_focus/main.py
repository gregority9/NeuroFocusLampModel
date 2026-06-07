from src.eeg_focus.pipelines.preprocess import PreprocessingPipeline

preprocessing_pipeline = PreprocessingPipeline()

result = preprocessing_pipeline.run()
print(
    {
        "combined_path": result["combined_path"],
        "manifest_path": result["manifest_path"],
        "n_rows": result["n_rows"],
        "n_subjects": result["n_subjects"],
        "data_types": result["data_types"],
        "failures": result["failures"],
    }
)
