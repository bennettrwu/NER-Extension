import { Readability } from '@mozilla/readability';

const NER_URL = 'http://localhost:8080/ner';
const NER_TYPES_ESCAPES = {
  Location: '{$}',
  Person: '{$$}',
  Organization: '{$$$}',
  Miscellaneous: '{$$$$}',
};
async function main() {
  promptNER();
}

async function startNER() {
  showLoadingReader();
  await enableReader();

  const text_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'];
  const page = document.getElementById('ner-container');
  for (const tag of text_tags) {
    const text_element = page.getElementsByTagName(tag);
    for (const element of text_element) {
      await runNER(element);
    }
  }

  closePrompt();
}

function promptNER() {
  const prompt = document.createElement('div');
  prompt.id = 'ner-prompt';
  prompt.innerHTML = `
  <h3>Named Entity Recognition</h3>
  <button class="ner-button" style="background-color: #96C9DC" id="ner-run-button">Run</button>
  <button class="ner-button" style="background-color: #F06C9B" id="ner-close-button">Close</button>
  `;

  document.body.appendChild(prompt);

  document.getElementById('ner-run-button').onclick = startNER;
  document.getElementById('ner-close-button').onclick = closePrompt;
}

function closePrompt() {
  document.getElementById('ner-prompt').remove();
}

function showLoadingReader() {
  const prompt = document.getElementById('ner-prompt');

  prompt.innerHTML = '<p>Creating Reader View...</p><p class="ner-loading"></p>';
}

async function enableReader() {
  const documentClone = document.cloneNode(true);
  const reader = new Readability(documentClone as Document);
  const article = reader.parse();

  const doc = new DOMParser().parseFromString(
    `
    <div id="ner-container">
      <div id="ner-content">
        <h1>${article.title}</h1>
        <p style="font-style: italic">${article.byline}</p>
        <hr></hr>
        ${article.content}
      </div>
      <div id="ner-prompt">
        <p>Running Named Entity Recognition...</p>
        <p class="ner-loading"></p>
      </div>
    </div>
    `,
    'text/html',
  ).body;
  document.body.replaceWith(doc);
}

async function runNER(e: Element) {
  try {
    const text = e.textContent;
    if (text === '') return;

    console.log(text);
    const res = await fetch(NER_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: text,
    });

    const to_replace = (await res.json()) as Array<{
      str: string;
      n: number;
      type: 'Location' | 'Person' | 'Organization' | 'Miscellaneous';
    }>;
    console.log(to_replace);

    let updated_element = e.innerHTML;

    // Escape links inside paragraph
    const escaped_links: Array<string> = [];
    updated_element = updated_element.replace(/<a(?![^>]*\/>)[^>]*>/, (match) => {
      escaped_links.push(match);
      return '{$$$$$$}';
    });

    // Label entities
    for (const { str, n, type } of to_replace) {
      updated_element = replaceNthInstanceOf(updated_element, str, `${NER_TYPES_ESCAPES[type]}${str}{$$$$$}`, n);
    }

    // Replace escapes with correct <span> tags
    updated_element = updated_element.replaceAll('{$$$$$}', '</span>');
    for (const [type, escape] of Object.entries(NER_TYPES_ESCAPES)) {
      updated_element = updated_element.replaceAll(escape, `<span class="named-entity ${type}">`);
    }

    // Restore links
    updated_element = updated_element.replaceAll('{$$$$$$}', () => {
      return escaped_links.shift();
    });

    e.innerHTML = updated_element;
  } catch (error) {
    console.log(error);
  }
}

function replaceNthInstanceOf(text: string, to_replace: string, replacement: string, n: number) {
  let i = 0;
  return text.replace(to_replace, (match) => {
    return ++i === n ? replacement : match;
  });
}

main();
