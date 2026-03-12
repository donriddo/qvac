const SDK_API_PREFIX = '/sdk/api/';

/**
 * Filter pages to only include SDK API pages for a specific version,
 * while always including non-SDK pages (they are version-agnostic).
 */
export function filterPagesByVersion<T extends { url: string }>(
  pages: T[],
  version: string | null,
  latestVersion: string,
): T[] {
  const resolved = !version || version === 'latest' ? latestVersion : version;

  return pages.filter((page) => {
    if (!page.url.startsWith(SDK_API_PREFIX)) return true;
    return page.url.startsWith(`${SDK_API_PREFIX}${resolved}/`);
  });
}
