import { Readability } from '@mozilla/readability';

const NER_URL = 'http://localhost:8080/ner';

async function main() {
  await enableReader();

  const text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
  const pages = document.body.getElementsByClassName('page');
  for (const page of pages) {
    for (const tag of text_tags) {
      const text_element = page.getElementsByTagName(tag);

      for (const element of text_element) {
        await runNER(element);
      }
    }
  }
}

async function enableReader() {
  const documentClone = document.cloneNode(true);
  const reader = new Readability(documentClone as Document);
  const article = reader.parse();

  const doc = new DOMParser().parseFromString(article.content, 'text/html').body;
  document.body.replaceWith(doc);
}

async function runNER(e: Element) {
  const text = e.textContent;
  if (text === '') return;

  const res = await fetch(NER_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: text,
  });

  console.log(await res.text());
}

main();
