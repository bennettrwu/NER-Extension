import { Readability } from '@mozilla/readability';

function main() {
  runNER();
}

function runNER() {
  const documentClone = document.cloneNode(true);
  const reader = new Readability(documentClone as Document);
  const article = reader.parse();
  console.log(document.documentElement.innerHTML);

  const doc = new DOMParser().parseFromString(article.content, 'text/html').body;

  document.body.replaceWith(doc);
}

main();
