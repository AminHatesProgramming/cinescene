import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


def create_directories():
    """ساخت پوشه‌ها"""
    dirs = ['data/raw', 'data/processed', 'data/index', 'model/base', 'model/finetuned']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ پوشه‌ها ساخته شد\n")


def download_file(url, output_path):
    """دانلود فایل با progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        print(f"✅ دانلود شد: {output_path}\n")
        return True
    except Exception as e:
        print(f"❌ خطا در دانلود {url}: {e}\n")
        return False


def download_wikipedia_plots():
    """دانلود Wikipedia Movie Plots"""
    print("📥 دانلود Wikipedia Movie Plots...")
    url = "https://raw.githubusercontent.com/RaRe-Technologies/movie-plots-by-genre/master/plots.csv"
    output = "data/raw/wikipedia_plots.csv"
    
    if os.path.exists(output):
        print(f"⏭️  فایل موجود است: {output}\n")
        return True
    
    return download_file(url, output)


def download_cmu_corpus():
    """دانلود CMU Movie Summary Corpus"""
    print("📥 دانلود CMU Movie Summary Corpus...")
    url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    output = "data/raw/MovieSummaries.tar.gz"
    
    if os.path.exists(output):
        print(f"⏭️  فایل موجود است: {output}\n")
        return True
    
    success = download_file(url, output)
    
    if success:
        print("📦 استخراج فایل...")
        import tarfile
        with tarfile.open(output, 'r:gz') as tar:
            tar.extractall('data/raw/')
        print("✅ استخراج شد\n")
    
    return success


def download_imdb_datasets():
    """دانلود IMDb datasets (title.basics, title.ratings)"""
    print("📥 دانلود IMDb datasets...")
    
    datasets = {
        'title.basics.tsv.gz': 'https://datasets.imdbws.com/title.basics.tsv.gz',
        'title.ratings.tsv.gz': 'https://datasets.imdbws.com/title.ratings.tsv.gz',
    }
    
    for filename, url in datasets.items():
        output = f"data/raw/{filename}"
        
        if os.path.exists(output):
            print(f"⏭️  فایل موجود است: {output}\n")
            continue
        
        download_file(url, output)
        
        # استخراج gzip
        if output.endswith('.gz'):
            print(f"📦 استخراج {filename}...")
            import gzip
            import shutil
            
            output_unzipped = output[:-3]  # حذف .gz
            with gzip.open(output, 'rb') as f_in:
                with open(output_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"✅ استخراج شد: {output_unzipped}\n")


def main():
    print("🚀 شروع دانلود دیتاست‌ها...\n")
    print("⚠️  توجه: دانلود ممکن است چند دقیقه طول بکشه\n")
    print("=" * 60 + "\n")
    
    # ساخت پوشه‌ها
    create_directories()
    
    # دانلود Wikipedia Plots
    download_wikipedia_plots()
    
    # دانلود CMU Corpus
    download_cmu_corpus()
    
    # دانلود IMDb
    download_imdb_datasets()
    
    print("=" * 60)
    print("\n✅ همه دیتاست‌ها دانلود شدند!")
    print("\n📁 فایل‌های دانلود شده:")
    print("   - data/raw/wikipedia_plots.csv")
    print("   - data/raw/MovieSummaries/ (CMU corpus)")
    print("   - data/raw/title.basics.tsv")
    print("   - data/raw/title.ratings.tsv")
    print("\n🎯 حالا می‌تونی preprocess_tmdb.py رو اجرا کنی")


if __name__ == '__main__':
    main()
