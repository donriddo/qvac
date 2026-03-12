import { source } from '@/lib/source';
import { LATEST_VERSION } from '@/lib/versions';
import { filterPagesByVersion } from '@/lib/filter-pages-by-version';

export const revalidate = false;

export function GET() {
  const pages = filterPagesByVersion(source.getPages(), null, LATEST_VERSION);
  const index = [
    '# QVAC Documentation (llms.txt)',
    '',
    'Agent index for the QVAC documentation website.',
    '',
    `- Full documentation dump: /llms-full.txt?version=${LATEST_VERSION}`,
    '',
    'Guidance:',
    `- Use /llms-full.txt?version=${LATEST_VERSION} as the primary context for answering questions.`,
    '- When answering, reference the most relevant doc page URL(s).',
    `- Latest SDK version: ${LATEST_VERSION}`,
    `- Total pages: ${pages.length}`,
  ].join('\n');

  return new Response(index);
}
