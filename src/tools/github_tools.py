"""
GitHub repository fetcher and filter.
Maps live GitHub data into the ProjectEntry Pydantic model.
"""

from __future__ import annotations

from github import Github, GithubException, Auth
from github.Repository import Repository

from src.config import settings
from src.schemas.models import ProjectEntry

def _get_github_client() -> Github:
    """Initializes the GitHub client with token auth."""
    auth = Auth.Token(settings.github_token)
    return Github(auth=auth, per_page=100)

def fetch_relevant_github_projects(
    username: str, 
    manual_skills: list[str], 
    max_projects: int = 10
) -> list[ProjectEntry]:
    """
    Fetches the user's public repositories and returns the most relevant ones.
    Relevance is determined by tech stack overlap with the user's manual skills,
    stars, and recency.
    """
    gh = _get_github_client()
    try:
        user = gh.get_user(username)
        repos = user.get_repos(type="owner", sort="updated", direction="desc")
    except GithubException as e:
        print(f"GitHub API error fetching for user {username}: {e.data}")
        return []

    # Normalize skills for case-insensitive matching
    target_skills = {skill.lower() for skill in manual_skills}
    project_entries = []

    for repo in repos:
        repo: Repository
        # Ignore forks and empty repos
        if repo.fork or repo.size == 0:
            continue

        # Fetch languages (requires an extra API call per repo, heavily rate-limited if unbounded)
        try:
            repo_languages = set(repo.get_languages().keys())
        except GithubException:
            repo_languages = set()
            
        repo_languages_lower = {lang.lower() for lang in repo_languages}
        
        # Calculate overlap
        overlap = target_skills.intersection(repo_languages_lower)
        
        # Only include projects that share at least one technology with the user's profile,
        # OR if it has a high star count indicating quality regardless of tech stack.
        if overlap or repo.stargazers_count > 5:
            # Fallback to repo description if README is missing
            description = repo.description or "No description provided."
            
            entry = ProjectEntry(
                name=repo.name,
                description=description[:300], # Truncate to prevent context bloat
                tech_stack=list(repo_languages) if repo_languages else ["Unknown"],
                deployed_url=repo.homepage if repo.homepage else None,
                github_url=repo.html_url,
                is_pinned=False, # PyGithub doesn't natively expose pinned status without GraphQL; defaulting to False.
                last_commit_date=repo.updated_at.date().isoformat() if repo.updated_at else None,
                stars=repo.stargazers_count
            )
            project_entries.append(entry)

        if len(project_entries) >= max_projects:
            break

    # Sort: Stars first, then most recently updated
    project_entries.sort(key=lambda x: (x.stars, x.last_commit_date or ""), reverse=True)
    return project_entries