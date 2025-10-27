// Byte-Pair Encoding (BPE) tokenizer

type PackedPair = number
type StatsValue = { packed: PackedPair; count: number; arr: [number, number] }
type Stats = Map<PackedPair, StatsValue>
type Tokens = number[]

export const DEBUG = Deno.env.has("DEBUG");

const debug = (...args: unknown[]) => {
  if (DEBUG) {
    console.log('[debug]', ...args);
  }
}

// packs two 8-bit numbers into one 16-bit number
const pack = (a: number, b: number): PackedPair => (a << 16) | b

// unpacks one 16-bit number into two 8-bit numbers
const unpack = (n: PackedPair) => [(n >> 16) & 0xffff, n & 0xffff]

const combine = (a: Uint8Array, b: Uint8Array) => {
  const combined = new Uint8Array(a.length + b.length);
  combined.set(a, 0);
  combined.set(b, a.length);
  return combined;
}

export class RegexTokenizer {
  #vocab_size: number = 0;
  #regex: RegExp = /\S+/g;
  #vocab: Map<number, Uint8Array<ArrayBuffer>> = new Map();

  // merge order from least to greatest vocab index. see training process
  #merges: [number, [number, number]][] = [];

  constructor(vocab_size: number, regex: RegExp = /\S+/g) {
    this.#vocab_size = vocab_size;
    this.#regex = regex;

    if (this.#vocab_size < 257) throw new Error('vocab_size must be greater than 256');
    if (!(regex instanceof RegExp)) throw new Error('regex must be a RegExp');
    if (!regex.flags.includes('g')) throw new Error('regex must have global flag (g)');
  }

  train(text: string) {
    const encoder = new TextEncoder()

    this.#merges = [];
    this.#vocab.clear();

    for (let i = 0; i < 256; i++) {
      this.#vocab.set(i, new Uint8Array([i]));
    }

    const num_merges = this.#vocab_size - 256

    const stringChunks = [...text.matchAll(this.#regex)].map(m => m[0]); // string[]
    const tokenChunks = stringChunks.map(chunk => Array.from(encoder.encode(chunk)));

    debug('Start Training');

    for (let i = 0; i < num_merges; i++) {
      const stats: Stats = new Map();
      for (const tokenChunk of tokenChunks) {
        RegexTokenizer.get_stats(tokenChunk, stats);
      }

      const top = RegexTokenizer.get_top_pairs(stats, 1)[0]
      if (top.count < 2) break;

      const idx = 256 + i
      this.#merges.push([idx, top.arr])

      // expand our vocab
      const first = this.#vocab.get(top.arr[0])
      const second= this.#vocab.get(top.arr[1])
      this.#vocab.set(idx, combine(first!, second!));

      for (let j = 0; j < tokenChunks.length; j++) {
        tokenChunks[j] = RegexTokenizer.replace(tokenChunks[j], top.arr, idx)
      }
    }

    debug('End Training');
  }

  encode(text: string) {
    const encoder = new TextEncoder()
    const stringChunks = [...text.matchAll(this.#regex)].map(m => m[0]); // string[]
    const tokenChunks = stringChunks.map(chunk => Array.from(encoder.encode(chunk)));

    for (let i = 0; i < tokenChunks.length; i++) {
      let tokens = tokenChunks[i];

      // there are no merges with less than two tokens
      if (tokens.length < 2) continue;

      // check if any token pairs are in merges, if so, take the one with lowest idx. replace it. repeat.
      while (true) {
        const pairs = RegexTokenizer.get_unique_pairs(tokens)
        const matchingMerges = pairs.map(pair => {
          for (const [idx, mergePair] of this.#merges) {
            if (pair[0] === mergePair[0] && pair[1] === mergePair[1]) {
              return { pair, idx }
            }
          }
        });

        const lowest = matchingMerges.sort((a, b) => (a?.idx ?? Infinity) - (b?.idx ?? Infinity)).filter(Boolean)[0]
        // debug({ lowest })

        if (!lowest) break;
        tokens = RegexTokenizer.replace(tokens, lowest.pair, lowest.idx)
      }

      tokenChunks[i] = tokens;
    }

    return tokenChunks;
  }

  decode(tokenChunks: number[][]) {
    debug('decoding %s token chunks', tokenChunks.length);

    const chunks: number[][] = [];

    for (const chunk of tokenChunks) {
      const ret: number[] = [];
      for (const idx of chunk) {
        const replacement = this.#vocab.get(idx)
        if (replacement) {
          ret.push(...Array.from(replacement));
        } else {
          throw new Error(`Token ${idx} not found in vocabulary during decoding.`);
        }
      }
      chunks.push(ret);
    }

    const flattened = chunks.flat();
    return new TextDecoder().decode(new Uint8Array(flattened));
  }

  printMerges() {
    const merges = this.#merges.slice();
    debug('Merges:');
    const decoder = new TextDecoder('utf-8', { fatal: false });
    for (let i = 0; i < merges.length; i++) {
      const [idx, pair] = merges[i];
      debug(`  ${idx} -> '${decoder.decode(new Uint8Array(pair))}'`);
    }
  }

  printVocab() {
    debug('Vocabulary:');
    const vocab = Array.from(this.#vocab.entries()).sort((a, b) => a[0] - b[0]);
    const decoder = new TextDecoder('utf-8', { fatal: false });
    for (const [idx, token] of vocab) {
      if (idx < 256) continue; // skip base vocab
      debug(`  ${idx}: '${decoder.decode(token)}'`);
    }
  }

  [Symbol.for('Deno.customInspect')]() {
    return `RegexTokenizer { merges: ${this.#merges.length} }`;
  }

  static replace(tokens: Tokens, pair: [number, number], replacement: number) {
    const newTokens = []

    for (let i = 0; i < tokens.length; ) {
      if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
        newTokens.push(replacement)
        i += 2
      } else {
        newTokens.push(tokens[i])
        i += 1
      }
    }

    return newTokens
  }

  static get_stats(ids: Tokens, counts: Stats = new Map()) {
    for (let i = 0; i < ids.length - 2; i++) {
      const packed = pack(ids[i], ids[i + 1])
      const val = { packed, count: (counts.get(packed)?.count ?? 0) + 1, arr: [ids[i], ids[i + 1]] } satisfies StatsValue
      counts.set(packed, val)
    }
    return counts
  }

  static get_unique_pairs(ids: Tokens) {
    const stats = RegexTokenizer.get_stats(ids)
    return Array.from(stats.values()).map(stat => stat.arr)
  }

  static get_top_pairs(counts: Stats, max = 10) {
    const sorted = Array.from(counts.values()).sort((a, b) => b.count - a.count).slice(0, max)
    return sorted
  }
}

// const gpt4_pattern = /'s|n't|'re|'ve|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w]+|\s+/g;
// GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
// const r =                  /'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/v;
const GPT4_REGEX = /'([sSdDmMtT]|ll|LL|lL|Ll|ve|VE|vE|Ve|re|RE|rE|Re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/gv;

// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./blog.txt'));
const str = new TextDecoder('utf-8').decode(await Deno.readFile('./taylorswift.txt'));
const tokenizer = new RegexTokenizer(500, GPT4_REGEX);
tokenizer.train(str);
// debug({ tokenizer });
// tokenizer.printMerges();
tokenizer.printVocab();

const test = (str: string) => {
  const encoded = tokenizer.encode(str);
  const decoded = tokenizer.decode(encoded);
  if (decoded !== str) {
    for (let i = 0; i < Math.min(decoded.length, str.length); i++) {
      if (decoded[i] !== str[i]) {
        const prefixA = str.substring(Math.max(0, i - 10), i + 10);
        const prefixB = decoded.substring(Math.max(0, i - 10), i + 10);
        console.error(`Mismatch at index ${i}`);
        console.error(`Original: '${prefixA}'`);
        console.error(`Decoded : '${prefixB}'`);
        break;
      }
    }
    throw new Error(`Decoded does not match original string!`);
  }
  console.log(`Test passed. String length: '${str.length}'`);
}

// test('Hello, world! This is a test of the RegexTokenizer. Let\'s see how it handles contractions like don\'t and I\'ll.');
// test(str);
test('I can\'t believe it\'s not butter! :snowman: ⛄️ ! 3hoo 3,333');
