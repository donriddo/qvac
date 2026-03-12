import { type NextRequest, NextResponse } from 'next/server';
import { source } from '@/lib/source';
import { getLLMText } from '@/lib/get-llm-text';

export const revalidate = false;

export async function GET(request: NextRequest) {
  const slug = request.nextUrl.searchParams.get('slug');
  if (!slug) {
    return NextResponse.json({ error: 'Missing ?slug= parameter' }, { status: 400 });
  }

  const slugParts = slug.split('/').filter(Boolean);
  const page = source.getPage(slugParts);
  if (!page) {
    return NextResponse.json({ error: 'Page not found' }, { status: 404 });
  }

  const text = await getLLMText(page);
  return new Response(text);
}
