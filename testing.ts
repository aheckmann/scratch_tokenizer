import { test, GPT4_REGEX } from './regex_special_tokenizer.ts'

const str = new TextDecoder('utf-8').decode(await Deno.readFile('./samples/multilingual.txt'));
// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./samples/blog.txt'));
// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./samples/taylorswift.txt'));

// const str =  `function add(a, b) {
//   console.log('Adding numbers', a, b);
//   return a + b;
// }`

// const str = `def add(a, b):
//     print("Adding numbers", a, b)
//     return a + b

// <|endoftext|>`

const trainingText = new TextDecoder('utf-8').decode(await Deno.readFile('./regex_special_tokenizer.ts'));

test({
  vocabSize: 500,
  regex: GPT4_REGEX,
  printVocab: false,
  printMerges: false,
  printTokens: true,
  trainingText,
  testText: str,
});
