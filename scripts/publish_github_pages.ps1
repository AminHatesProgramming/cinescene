param(
    [string]$Owner = "aminhatesprogramming",
    [string]$Repo = "cinescene",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$token = $env:GH_TOKEN
if (-not $token) { $token = $env:GITHUB_TOKEN }
if (-not $token) {
    throw "Set GH_TOKEN or GITHUB_TOKEN to a GitHub Personal Access Token with repo/workflow permissions."
}

$headers = @{
    Authorization = "Bearer $token"
    Accept = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

$repoFullName = "$Owner/$Repo"
$repoApi = "https://api.github.com/repos/$repoFullName"

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

$safe = (Resolve-Path ".").Path.Replace("\", "/")
git -c "safe.directory=$safe" remote remove origin 2>$null
git -c "safe.directory=$safe" remote add origin "https://github.com/$repoFullName.git"
git -c "safe.directory=$safe" push -u origin $Branch

Write-Host ""
Write-Host "Pushed to: https://github.com/$repoFullName"
Write-Host "Pages will be deployed by .github/workflows/pages.yml"
Write-Host "Expected URL: https://$Owner.github.io/$Repo"
