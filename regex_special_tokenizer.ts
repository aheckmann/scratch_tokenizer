// Byte-Pair Encoding (BPE) tokenizer implementation in TypeScript

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

const print = (() => {
  const te = new TextEncoder();
  const RESET = te.encode('\x1b[0m');
  const COLORS = [
    te.encode('\x1b[41m'), // red background
    te.encode('\x1b[42m'), // green background
    te.encode('\x1b[43m'), // yellow background
    te.encode('\x1b[44m'), // blue background
    te.encode('\x1b[45m'), // magenta background
    te.encode('\x1b[46m'), // cyan background
    te.encode('\x1b[101m'), // bright red background
    te.encode('\x1b[102m'), // bright green background
    te.encode('\x1b[103m'), // bright yellow background
    te.encode('\x1b[104m'), // bright blue background
  ];

  let colorIndex = 0;

  return (args: Uint8Array) => {
    const color = COLORS[colorIndex++ % COLORS.length];
    Deno.stdout.writeSync(color);
    if (args.at(-1) === 10) {
      Deno.stdout.writeSync(args.slice(0, -1));
      Deno.stdout.writeSync(RESET);
      Deno.stdout.writeSync(new Uint8Array([10]));
    } else {
      Deno.stdout.writeSync(args);
      Deno.stdout.writeSync(RESET);
    }
  }
})();

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

const DEFAULT_ALLOWED_SPECIAL: Map<string, number> = new Map([
  ['<|endoftext|>', 100257],
  ['<|fim_prefix|>', 100258],
  ['<|fim_middle|>', 100259],
  ['<|fim_suffix|>', 100260],
  ['<|endofprompt|>', 100276],
]);

export class RegexTokenizer {
  #vocab_size: number = 0;
  #regex: RegExp = /\S+/g;
  #vocab: Map<number, Uint8Array<ArrayBuffer>> = new Map();
  #special_tokens: Map<string, number> = new Map();
  #inverse_special_tokens: Map<number, Uint8Array<ArrayBuffer>> = new Map();

  // merge order from least to greatest vocab index. see training process
  #merges: [number, [number, number]][] = [];

  constructor(
    vocab_size: number,
    regex: RegExp = /\S+/g,
    special_tokens: Map<string, number> = DEFAULT_ALLOWED_SPECIAL,
  ) {
    this.#vocab_size = vocab_size;
    this.#regex = regex;
    this.register_special_tokens(special_tokens);

    if (this.#vocab_size < 257) throw new Error('vocab_size must be greater than 256');
    if (!(regex instanceof RegExp)) throw new Error('regex must be a RegExp');
    if (!regex.flags.includes('g')) throw new Error('regex must have global flag (g)');
  }

  register_special_tokens(tokens: Map<string, number>) {
    this.#special_tokens = tokens;
    const encoder = new TextEncoder();
    for (const [token, idx] of tokens.entries()) {
      this.#inverse_special_tokens.set(idx, encoder.encode(token));
    }
  }

  train(text: string) {
    debug('Starting training...');
    console.time('Training');

    const encoder = new TextEncoder()

    this.#merges = [];
    this.#vocab.clear();

    for (let i = 0; i < 256; i++) {
      this.#vocab.set(i, new Uint8Array([i]));
    }

    const num_merges = this.#vocab_size - 256

    const stringChunks = [...text.matchAll(this.#regex)].map(m => m[0]); // string[]
    const tokenChunks = stringChunks.map(chunk => Array.from(encoder.encode(chunk)));

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

    console.timeEnd('Training');
    debug('Finished training.');
  }

  encode(text: string, allowed_special: 'none_raise' | 'none' | 'all' | Set<string> = 'all') {
    console.time('encode');
    let special = new Map<string, number>();

    if (allowed_special === 'all') {
      special = this.#special_tokens;
    } else if (allowed_special === 'none') {
      special = new Map();
    } else if (allowed_special === 'none_raise') {
      special = new Map();
      for (const token of this.#special_tokens.keys()) {
        if (text.includes(token)) {
          throw new Error(`Special token '${token}' found in text, but allowed_special is 'none_raise'`);
        }
      }
    } else if (allowed_special instanceof Set) {
      special = new Map<string, number>();
      for (const name of allowed_special.values()) {
        if (!this.#special_tokens.has(name)) {
          throw new Error(`Special token '${name}' not found in tokenizer's special tokens`);
        }
        special.set(name, this.#special_tokens.get(name)!);
      }
    } else {
      throw new Error(`Invalid allowed_special value: ${allowed_special}`);
    }

    if (special.size === 0) {
      return this.encode_ordinary(text);
    }

    // split text by special tokens, encode ordinary parts, and insert special tokens as is
    const pattern = Array.from(special.keys()).map(token => RegExp.escape(token)).join('|');
    const regex = new RegExp(`(${pattern})`, 'g');
    const parts = text.split(regex);
    const encodedChunks: number[][] = [];

    for (const part of parts) {
      if (special.has(part)) {
        encodedChunks.push([special.get(part)!]);
      } else if (part.length > 0) {
        const encodedPart = this.encode_ordinary(part);
        encodedChunks.push(...encodedPart);
      }
    }

    console.timeEnd('encode');
    return encodedChunks;
  }

  encode_ordinary(text: string) {
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
        const merges: { pair: [number, number], idx: number }[] = []
        for (const pair of pairs) {
          for (const [idx, mergePair] of this.#merges) {
            if (pair[0] === mergePair[0] && pair[1] === mergePair[1]) {
              merges.push({ pair, idx })
            }
          }
        }
        const lowest = merges.sort((a, b) => a.idx - b.idx)[0]
        if (!lowest) break;
        tokens = RegexTokenizer.replace(tokens, lowest.pair, lowest.idx)
      }

      tokenChunks[i] = tokens;
    }

    return tokenChunks;
  }

  decode(tokenChunks: number[][]) {
    debug('decoding %s token chunks', tokenChunks.length);
    console.time('decode');

    const chunks: number[][] = [];

    for (const chunk of tokenChunks) {
      const ret: number[] = [];
      for (const idx of chunk) {
        if (this.#vocab.has(idx)) {
          ret.push(...Array.from(this.#vocab.get(idx)!));
        } else if (this.#inverse_special_tokens.has(idx)) {
          const inverseToken = this.#inverse_special_tokens.get(idx)!;
          ret.push(...Array.from(inverseToken));
        } else {
          throw new Error(`Token ${idx} not found in vocabulary during decoding.`);
        }
      }
      chunks.push(ret);
    }

    const flattened = chunks.flat();

    console.timeEnd('decode');
    return new TextDecoder().decode(new Uint8Array(flattened));
  }

  printCLIvisualization(tokenChunks: number[][]) {
    console.time('visualize');

    for (const chunk of tokenChunks) {
      const ret: number[] = [];
      for (const idx of chunk) {
        if (this.#vocab.has(idx)) {
          ret.push(...Array.from(this.#vocab.get(idx)!));
        } else if (this.#inverse_special_tokens.has(idx)) {
          ret.push(...Array.from(this.#inverse_special_tokens.get(idx)!));
        } else {
          throw new Error(`Token ${idx} not found in vocabulary during decoding.`);
        }
      }
      print(new Uint8Array(ret));
    }

    console.log()
    console.timeEnd('visualize');
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

export const GPT4_REGEX = /'([sSdDmMtT]|ll|LL|lL|Ll|ve|VE|vE|Ve|re|RE|rE|Re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+/gv;

// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./blog.txt'));
// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./taylorswift.txt'));
// const tokenizer = new RegexTokenizer(4096, GPT4_REGEX);
// tokenizer.train(str);
// debug({ tokenizer });
// tokenizer.printMerges();
// tokenizer.printVocab();

export const test = ({
  vocabSize,
  regex,
  trainingText,
  testText
}: {
  vocabSize: number,
  regex: RegExp,
  trainingText: string,
  testText: string
}) => {
  const tokenizer = new RegexTokenizer(vocabSize, regex);
  tokenizer.train(trainingText);

  const encoded = tokenizer.encode(testText, 'all');
  tokenizer.printCLIvisualization(encoded);

  const decoded = tokenizer.decode(encoded);
  if (decoded !== testText) {
    for (let i = 0; i < Math.min(decoded.length, testText.length); i++) {
      if (decoded[i] !== testText[i]) {
        const prefixA = testText.substring(Math.max(0, i - 10), i + 10);
        const prefixB = decoded.substring(Math.max(0, i - 10), i + 10);
        console.error(`Mismatch at index ${i}`);
        console.error(`Original: '${prefixA}'`);
        console.error(`Decoded : '${prefixB}'`);
        break;
      }
    }
    throw new Error(`Decoded does not match original string!`);
  }
  console.log(`Test passed. String length: '${testText.length}'`);
}
