// Byte-Pair Encoding (BPE) tokenizer implementation in TypeScript

type PackedPair = number
type StatsValue = { packed: PackedPair; count: number; arr: [number, number] }
type Stats = Map<PackedPair, StatsValue>
type Tokens = number[]

export const DEBUG = false;

const debug = (...args: unknown[]) => {
  if (DEBUG) {
    console.log(...args);
  }
}

// packs two 8-bit numbers into one 16-bit number
const pack = (a: number, b: number): PackedPair => (a << 16) | b

// unpacks one 16-bit number into two 8-bit numbers
const unpack = (n: PackedPair) => [(n >> 16) & 0xffff, n & 0xffff]

export class BasicTokenizer {
  #vocab_size: number = 0;

  // merge order from least to greatest vocab index. see training process
  #merges: [number, [number, number]][] = [];

  constructor(vocab_size: number) {
    this.#vocab_size = vocab_size;
    if (this.#vocab_size < 257) throw new Error('vocab_size must be greater than 256');
  }

  train(text: string) {
    const encoder = new TextEncoder()
    const tokens = Array.from(encoder.encode(text))
    const num_merges = this.#vocab_size - 256

    this.#merges = [];

    let currentTokens = tokens;

    debug('Start Training');
    for (let i = 0; i < num_merges; i++) {
      const stats = BasicTokenizer.get_stats(currentTokens)
      const top = BasicTokenizer.get_top_pairs(stats, 1)[0]

      if (top.count < 2) break;

      const idx = 256 + i
      currentTokens = BasicTokenizer.replace(currentTokens, top.arr, idx)
      this.#merges.push([idx, top.arr])
      // debug(`merge ${i + 1}/${num_merges}: ${top.arr} -> ${idx} (${top.count})`);

      if (DEBUG) {
        const decoder = new TextDecoder('utf-8', { fatal: false });
        const step = i < 9 ? ` ${i + 1}` : i + 1;
        console.log(`  Step ${step}: ${currentTokens.length} tokens, vocab size: ${num_merges}, replaced pair ${top.arr} with ${idx}, '${decoder.decode(new Uint8Array([top.arr[0], top.arr[1]]))}'`);
      }
    }
    debug('End Training');
  }

  encode(text: string) {
    const encoder = new TextEncoder()
    let tokens = Array.from(encoder.encode(text))

    // there are no merges with less than two tokens
    if (tokens.length < 2) return tokens;

    // check if any token pairs are in merges, if so, take the one with lowest idx. replace it. repeat.
    while (true) {
      const pairs = BasicTokenizer.get_unique_pairs(tokens)
      const lowest = pairs.map(pair => {
        for (const [idx, mergePair] of this.#merges) {
          if (pair[0] === mergePair[0] && pair[1] === mergePair[1]) {
            return { pair, idx }
          }
        }
      });
      debug({ lowest: lowest.filter(Boolean) });

      const l = lowest.sort((a, b) => (a?.idx ?? Infinity) - (b?.idx ?? Infinity)).filter(Boolean)[0]
      debug({ l })

      if (!l) break;
      tokens = BasicTokenizer.replace(tokens, l.pair, l.idx)
    }

    return tokens;
  }

  decode(tokens: number[]) {
    debug('decoding', tokens);

    const toks = tokens.slice()

    // do not use reverse() as it mutates the array and causes bugs
    for (const merge of this.#merges.toReversed()) {
      debug({ merge });

      for (let i = toks.length - 1; i >= 0; i--) {
        if (merge[0] === toks[i]) {
          toks.splice(i, 1, ...merge[1])
        }
      }
    }

    return new TextDecoder().decode(new Uint8Array(toks));
  }

  printMerges() {
    const merges = this.#merges.slice();
    const decoder = new TextDecoder('utf-8', { fatal: false });
    for (let i = 0; i < merges.length; i++) {
      const [idx, pair] = merges[i];
      console.log(`${idx} -> '${decoder.decode(new Uint8Array(pair))}'`);
    }
  }

  [Symbol.for('Deno.customInspect')]() {
    return `Codec { merges: ${this.#merges.length} }`;
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
    const stats = BasicTokenizer.get_stats(ids)
    return Array.from(stats.values()).map(stat => stat.arr)
  }

  static get_top_pairs(counts: Stats, max = 10) {
    const sorted = Array.from(counts.values()).sort((a, b) => b.count - a.count).slice(0, max)
    return sorted
  }
}

// const str = new TextDecoder('utf-8').decode(await Deno.readFile('./blog.txt'));
const str = new TextDecoder('utf-8').decode(await Deno.readFile('./taylorswift.txt'));
const tokenizer = new BasicTokenizer(500);
tokenizer.train(str);
debug({ tokenizer });
tokenizer.printMerges();

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

test('');
test(str);
