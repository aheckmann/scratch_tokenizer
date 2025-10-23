import { test, GPT4_REGEX } from './regex_special_tokenizer.ts'

const js =  `function add(a, b) {
  console.log('Adding numbers', a, b);
  return a + b;
}`

const py = `def add(a, b):
    print("Adding numbers", a, b)
    return a + b`

const jsTraining = new TextDecoder('utf-8').decode(await Deno.readFile('./regex_special_tokenizer.ts'));
test({
  vocabSize: 500,
  // regex: /\s*[\p{L}\p{N}]*.+/gv,
  // regex: /[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/gv,
  regex: GPT4_REGEX,
  trainingText: jsTraining,
  testText: py //js
});
// test(py);
