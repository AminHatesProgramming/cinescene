param(
    [string]$Owner = "AminHatesProgramming",
    [string]$Repo = "cinescene",
    [string]$Branch = "main",
    [string]$PagesBranch = "gh-pages"
)

$ErrorActionPreference = "Stop"

$token = $env:GH_TOKEN
if (-not $token) { $token = $env:GITHUB_TOKEN }
if (-not $token) {
    throw "Set GH_TOKEN or GITHUB_TOKEN to a GitHub Personal Access Token with repo permissions."
}

$headers = @{
    Authorization = "Bearer $token"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

$repoFullName = "$Owner/$Repo"
$repoApi = "https://api.github.com/repos/$repoFullName"
$root = (Resolve-Path ".").Path
$safeRoot = $root.Replace("\", "/")
$publishWorktree = Join-Path $root ".gh-pages-worktree"

function Assert-InsideProject([string]$PathToCheck) {
    $resolved = (Resolve-Path -LiteralPath $PathToCheck).Path
    if (-not $resolved.StartsWith($root)) {
        throw "Refusing to edit outside project: $resolved"
    }
    return $resolved
}

try {
    Invoke-RestMethod -Headers $headers -Uri $repoApi -Method Get | Out-Null
    Write-Host "Repository exists: $repoFullName"
}
catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw }
    Write-Host "Creating repository: $repoFullName"
    $body = @{
        name = $Repo
        description = "CineScene: semantic movie, series, and offline scene retrieval PWA."
        private = $false
        has_issues = $true
        has_projects = $false
        has_wiki = $false
        auto_init = $false
    } | ConvertTo-Json
    Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/user/repos" -Method Post -Body $body -ContentType "application/json" | Out-Null
}

git -c "safe.directory=$safeRoot" remote remove origin 2>$null
git -c "safe.directory=$safeRoot" remote add origin "https://github.com/$repoFullName.git"
git -c "safe.directory=$safeRoot" push -u origin $Branch

if (Test-Path -LiteralPath "data/processed/cinescene_catalog.json") {
    Write-Host "Refreshing docs/data/catalog.sample.json"
    python scripts/build_pwa_catalog.py
}

if (-not (Test-Path -LiteralPath "docs/index.html")) {
    throw "Missing docs/index.html. Build or restore the PWA before publishing."
}

if (Test-Path -LiteralPath $publishWorktree) {
    git -c "safe.directory=$safeRoot" worktree remove $publishWorktree --force 2>$null
    if (Test-Path -LiteralPath $publishWorktree) {
        $target = Assert-InsideProject $publishWorktree
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

git -c "safe.directory=$safeRoot" worktree add -B $PagesBranch $publishWorktree $Branch

$publishRoot = Assert-InsideProject $publishWorktree
Get-ChildItem -LiteralPath $publishRoot -Force |
    Where-Object { $_.Name -notin @(".git", "docs") } |
    Remove-Item -Recurse -Force

$docsRoot = Join-Path $publishRoot "docs"
Get-ChildItem -LiteralPath $docsRoot -Force |
    ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $publishRoot -Force }
Remove-Item -LiteralPath $docsRoot -Recurse -Force

$safePublish = $publishRoot.Replace("\", "/")
git -c "safe.directory=$safePublish" add -A
git -c "safe.directory=$safePublish" commit -m "Publish PWA to GitHub Pages branch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No gh-pages content changes to commit."
}
git -c "safe.directory=$safePublish" push -u origin $PagesBranch --force-with-lease

$pagesBody = @{ source = @{ branch = $PagesBranch; path = "/" } } | ConvertTo-Json -Depth 5
try {
    Invoke-RestMethod -Headers $headers -Uri "$repoApi/pages" -Method Post -Body $pagesBody -ContentType "application/json" | Out-Null
    Write-Host "GitHub Pages enabled from $PagesBranch /"
}
catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 409) { throw }
    Invoke-RestMethod -Headers $headers -Uri "$repoApi/pages" -Method Put -Body $pagesBody -ContentType "application/json" | Out-Null
    Write-Host "GitHub Pages source refreshed: $PagesBranch /"
}

Write-Host ""
Write-Host "Repository: https://github.com/$repoFullName"
Write-Host "Live PWA: https://$($Owner.ToLower()).github.io/$Repo/"
