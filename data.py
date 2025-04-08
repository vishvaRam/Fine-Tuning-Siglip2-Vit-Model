from bing_image_downloader import downloader

query_string = "a person looking away from camera"

downloader.download(
    query_string, 
    limit=120,  
    output_dir='dataset', 
    adult_filter_off=True, 
    force_replace=False, 
    timeout=5, 
    verbose=True)
