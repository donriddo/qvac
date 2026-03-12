import { describe, it, expect } from 'vitest';
import { filterPagesByVersion } from '../filter-pages-by-version';

const LATEST = 'v0.7.0';

const pages = [
  { url: '/getting-started' },
  { url: '/about-qvac/overview' },
  { url: '/tutorials/hello-world' },
  { url: '/sdk/api/v0.7.0/loadModel' },
  { url: '/sdk/api/v0.7.0/createSession' },
  { url: '/sdk/api/v0.6.1/loadModel' },
  { url: '/sdk/api/v0.6.1/createSession' },
  { url: '/sdk/api/v0.5.0/loadModel' },
  { url: '/sdk/api/latest/loadModel' },
];

const NON_SDK_URLS = [
  '/getting-started',
  '/about-qvac/overview',
  '/tutorials/hello-world',
];

describe('filterPagesByVersion', () => {
  it('returns latest-version SDK pages + all non-SDK pages when version is null', () => {
    const result = filterPagesByVersion(pages, null, LATEST);
    const urls = result.map((p) => p.url);

    for (const u of NON_SDK_URLS) {
      expect(urls).toContain(u);
    }
    expect(urls).toContain('/sdk/api/v0.7.0/loadModel');
    expect(urls).toContain('/sdk/api/v0.7.0/createSession');
    expect(urls).not.toContain('/sdk/api/v0.6.1/loadModel');
    expect(urls).not.toContain('/sdk/api/v0.5.0/loadModel');
    expect(urls).not.toContain('/sdk/api/latest/loadModel');
  });

  it('returns latest-version SDK pages when version is "latest"', () => {
    const result = filterPagesByVersion(pages, 'latest', LATEST);
    const urls = result.map((p) => p.url);

    expect(urls).toContain('/sdk/api/v0.7.0/loadModel');
    expect(urls).toContain('/sdk/api/v0.7.0/createSession');
    expect(urls).not.toContain('/sdk/api/v0.6.1/loadModel');
    expect(urls).not.toContain('/sdk/api/v0.5.0/loadModel');
  });

  it('filters to a specific version', () => {
    const result = filterPagesByVersion(pages, 'v0.6.1', LATEST);
    const urls = result.map((p) => p.url);

    for (const u of NON_SDK_URLS) {
      expect(urls).toContain(u);
    }
    expect(urls).toContain('/sdk/api/v0.6.1/loadModel');
    expect(urls).toContain('/sdk/api/v0.6.1/createSession');
    expect(urls).not.toContain('/sdk/api/v0.7.0/loadModel');
    expect(urls).not.toContain('/sdk/api/v0.5.0/loadModel');
  });

  it('returns only non-SDK pages for an unknown version', () => {
    const result = filterPagesByVersion(pages, 'v99.0.0', LATEST);
    const urls = result.map((p) => p.url);

    expect(urls).toEqual(NON_SDK_URLS);
  });

  it('always includes non-SDK pages regardless of version', () => {
    for (const version of [null, 'latest', 'v0.7.0', 'v0.6.1', 'v0.5.0', 'v99.0.0']) {
      const result = filterPagesByVersion(pages, version, LATEST);
      const urls = result.map((p) => p.url);

      for (const u of NON_SDK_URLS) {
        expect(urls).toContain(u);
      }
    }
  });

  it('handles empty pages array', () => {
    const result = filterPagesByVersion([], 'v0.7.0', LATEST);
    expect(result).toEqual([]);
  });

  it('preserves original page objects', () => {
    const pagesWithData = [
      { url: '/getting-started', title: 'Getting Started' },
      { url: '/sdk/api/v0.7.0/loadModel', title: 'loadModel' },
    ];
    const result = filterPagesByVersion(pagesWithData, 'v0.7.0', LATEST);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe(pagesWithData[0]);
    expect(result[1]).toBe(pagesWithData[1]);
  });
});
