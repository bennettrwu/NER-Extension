import { Readability } from '@mozilla/readability';

const NER_URL = 'http://localhost:8080/ner';

async function main() {
  await enableReader();

  const pages = document.body.getElementsByClassName('page');
  for (const page of pages) {
    const paragraphs = page.getElementsByTagName('p');

    for (const paragraph of paragraphs) {
      await runNER(paragraph);
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
