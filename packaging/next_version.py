"""Compute the next release version without creating tags or publishing files."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen


STABLE = re.compile(r'v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')


def next_version(tags, bump='patch'):
    if bump not in {'patch', 'minor', 'major'}:
        raise ValueError('Unknown version increment')
    versions = [tuple(map(int, match.groups())) for tag in tags
                if (match := STABLE.fullmatch(tag))]
    if not versions:
        return 'v1.0.0'
    major, minor, patch = max(versions)
    if bump == 'major':
        major, minor, patch = major + 1, 0, 0
    elif bump == 'minor':
        minor, patch = minor + 1, 0
    else:
        patch += 1
    return f'v{major}.{minor}.{patch}'


def remote_tags(repository, owner, repo):
    tags = []
    for endpoint, field in [('tags', 'name'), ('releases', 'tag_name')]:
        result = subprocess.run(
            ['gh', 'api', '--paginate', f'repos/{repository}/{endpoint}?per_page=100',
             '--jq', f'.[].{field}'], check=True, capture_output=True, text=True)
        tags.extend(result.stdout.splitlines())
    # Public installer tags include the product name; source releases remain vX.Y.Z.
    product = repository.split('/')[-1]
    result = subprocess.run(
        ['gh', 'api', '--paginate', 'repos/Normalight/hanzi-soft/releases?per_page=100',
         '--jq', '.[].tag_name'], check=True, capture_output=True, text=True)
    prefix = product + '-'
    tags.extend(tag[len(prefix):] for tag in result.stdout.splitlines() if tag.startswith(prefix))
    return tags


def main():
    parser = argparse.ArgumentParser(description='Calculate the next public release version')
    parser.add_argument('--bump', choices=['patch', 'minor', 'major'], default='patch')
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY'))
    parser.add_argument('--owner', default=os.environ.get('GITCODE_OWNER'))
    parser.add_argument('--repo', default=os.environ.get('GITCODE_REPO'))
    args = parser.parse_args()
    if not args.repository:
        parser.error('Provide the GitHub repository')
    version = next_version(remote_tags(args.repository, args.owner, args.repo), args.bump)
    print(version)
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
            output.write(f'version={version}\n')


if __name__ == '__main__':
    main()
