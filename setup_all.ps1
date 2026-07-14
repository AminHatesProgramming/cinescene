# ========================================
# CineScene v2.0 - Complete Setup Script
# ========================================

Write-Host "🎬 Starting CineScene v2.0 Setup..." -ForegroundColor Cyan

# 1️⃣ ساخت ساختار پوشه‌ها
Write-Host "`n📁 Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "model\checkpoints" | Out-Null
New-Item -ItemType Directory -Force -Path "model\index" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green

# 2️⃣ نصب کتابخانه‌ها
Write-Host "`n📦 Installing Python packages..." -ForegroundColor Yellow
pip install --upgrade pip
pip install sentence-transformers==2.3.1 transformers==4.36.0 torch==2.1.2 faiss-cpu==1.7.4 pandas==2.1.4 numpy==1.24.4 tqdm==4.66.1 scikit-learn==1.3.2 requests==2.31.0 streamlit==1.29.0 rank-bm25==0.2.2 python-dotenv==1.0.0 matplotlib==3.8.2 seaborn==0.13.0
Write-Host "✅ Packages installed" -ForegroundColor Green

# 3️⃣ استخراج TMDB از archive.zip
Write-Host "`n📂 Extracting TMDB dataset from archive.zip..." -ForegroundColor Yellow
if (Test-Path "archive.zip") {
    Expand-Archive -Path "archive.zip" -DestinationPath "data\raw\" -Force
    Write-Host "✅ TMDB extracted (5000 movies)" -ForegroundColor Green
} else {
    Write-Host "⚠️  archive.zip not found - skipping TMDB" -ForegroundColor Red
}

# 4️⃣ دانلود Wikipedia Movie Plots
Write-Host "`n🌐 Downloading Wikipedia Movie Plots (77 MB)..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "https://github.com/prust/wikipedia-movie-plots/raw/master/wiki_movie_plots_deduped.csv" -OutFile "data\raw\wiki_plots.csv" -TimeoutSec 300
    Write-Host "✅ Wikipedia plots downloaded (34,886 movies)" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download Wikipedia plots: $_" -ForegroundColor Red
}

# 5️⃣ دانلود CMU Movie Summary Corpus
Write-Host "`n🌐 Downloading CMU Movie Corpus (50 MB)..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz" -OutFile "data\raw\MovieSummaries.tar.gz" -TimeoutSec 300
    Write-Host "✅ CMU dataset downloaded" -ForegroundColor Green
    
    # استخراج
    Write-Host "📂 Extracting CMU dataset..." -ForegroundColor Yellow
    tar -xzf "data\raw\MovieSummaries.tar.gz" -C "data\raw\"
    Write-Host "✅ CMU extracted (42,306 movies)" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download CMU dataset: $_" -ForegroundColor Red
}

# 6️⃣ دانلود IMDb Datasets (اختیاری)
Write-Host "`n🌐 Downloading IMDb datasets (200 MB - Optional)..." -ForegroundColor Yellow
$downloadImdb = Read-Host "Download IMDb data? (y/n)"
if ($downloadImdb -eq "y") {
    try {
        Invoke-WebRequest -Uri "https://datasets.imdbws.com/title.basics.tsv.gz" -OutFile "data\raw\imdb_basics.tsv.gz" -TimeoutSec 600
        Invoke-WebRequest -Uri "https://datasets.imdbws.com/title.ratings.tsv.gz" -OutFile "data\raw\imdb_ratings.tsv.gz" -TimeoutSec 600
        Write-Host "✅ IMDb datasets downloaded" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download IMDb: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⏭️  Skipping IMDb download" -ForegroundColor Yellow
}

# 7️⃣ دانلود مدل BGE-Large (1.3 GB)
Write-Host "`n🤖 Downloading BGE-Large model (1.3 GB)..." -ForegroundColor Yellow
python -c "from sentence_transformers import SentenceTransformer; print('Loading BGE-Large...'); model = SentenceTransformer('BAAI/bge-large-en-v1.5'); print('✅ BGE-Large ready')"

# 8️⃣ دانلود مدل Cross-Encoder (400 MB)
Write-Host "`n🤖 Downloading Cross-Encoder for re-ranking (400 MB)..." -ForegroundColor Yellow
python -c "from sentence_transformers import CrossEncoder; print('Loading Cross-Encoder...'); model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('✅ Cross-Encoder ready')"

# 9️⃣ دانلود مدل چندزبانه (اختیاری - برای فارسی)
Write-Host "`n🌍 Downloading Multilingual model (Optional - 1 GB)..." -ForegroundColor Yellow
$downloadMulti = Read-Host "Download multilingual model for Persian support? (y/n)"
if ($downloadMulti -eq "y") {
    python -c "from sentence_transformers import SentenceTransformer; print('Loading Multilingual...'); model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); print('✅ Multilingual ready')"
} else {
    Write-Host "⏭️  Skipping multilingual model" -ForegroundColor Yellow
}

# 🔟 خلاصه نهایی
Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n📊 Downloaded Resources:" -ForegroundColor Yellow
Write-Host "  • TMDB: 5,000 movies" -ForegroundColor White
Write-Host "  • Wikipedia: 34,886 movies" -ForegroundColor White
Write-Host "  • CMU: 42,306 movies" -ForegroundColor White
Write-Host "  • BGE-Large: 1024-dim embeddings" -ForegroundColor White
Write-Host "  • Cross-Encoder: Re-ranking model" -ForegroundColor White
Write-Host "`n📂 Data location: data\raw\" -ForegroundColor Yellow
Write-Host "🤖 Models cached in: C:\Users\Webhouse\.cache\huggingface\" -ForegroundColor Yellow
Write-Host "`n🚀 Next step: Run 'python scripts/enrich_dataset.py'" -ForegroundColor Cyan
