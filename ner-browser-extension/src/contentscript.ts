import { Readability } from '@mozilla/readability';

const NER_URL = 'http://localhost:8080/ner';

function main() {
  runNER();
}

async function runNER() {
  const documentClone = document.cloneNode(true);
  const reader = new Readability(documentClone as Document);
  const article = reader.parse();
  console.log(document.documentElement.innerHTML);

  const res = await fetch(NER_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(article.textContent),
  });

  console.log(await res.text());

  // const doc = new DOMParser().parseFromString(article.content, 'text/html').body;

  // document.body.replaceWith(doc);
}

main();
