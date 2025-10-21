// Byte-Pair Encoding (BPE) tokenizer implementation in TypeScript

type PackedPair = number
type StatsValue = { packed: PackedPair; count: number; arr: [number, number] }
type Stats = Map<PackedPair, StatsValue>
type Tokens = number[]

const DEBUG = false;

const debug = (...args: unknown[]) => {
  if (DEBUG) {
    console.log(...args);
  }
}

// const input = `Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.`
// console.log(input)
// console.log(`Length: ${input.length}`)

// const encoder = new TextEncoder()
// even though encode() returns an Uint8Array, we convert to regular array b/c
// we will store numbers larger than 255 after compressing pairs
// const tokens = Array.from(encoder.encode(input))

// console.log("UTF-8 encoded bytes: ", tokens.slice(0, 50))
// console.log("Length of encoded: ", tokens.length)

// packs two 8-bit numbers into one 16-bit number
const pack = (a: number, b: number): PackedPair => (a << 16) | b

// unpacks one 16-bit number into two 8-bit numbers
const unpack = (n: PackedPair) => [(n >> 16) & 0xffff, n & 0xffff]

const get_stats = (ids: Tokens, counts: Stats = new Map()) => {
  // loop through ids, two at a time
  for (let i = 0; i < ids.length - 2; i++) {
    const packed = pack(ids[i], ids[i + 1])
    const val = { packed, count: (counts.get(packed)?.count ?? 0) + 1, arr: [ids[i], ids[i + 1]] } satisfies StatsValue
    counts.set(packed, val)
  }
  return counts
}

const get_unique_pairs = (ids: Tokens) => {
  const stats = get_stats(ids)
  return Array.from(stats.values()).map(stat => stat.arr)
}

// const stats = get_stats(tokens)
// console.log(`Number of tokens: ${stats.size}`)

const get_top_pairs = (counts: Stats, max = 10) => {
  const sorted = Array.from(counts.values()).sort((a, b) => b.count - a.count).slice(0, max)
  return sorted
}

// const top_pairs = get_top_pairs(stats, 10)
// console.log('Top 10 most frequent pairs', top_pairs)


/**
 * Replaces all occurrences of a given pair in the token list with a replacement token and returns a new array
 */
// const replace = (tokens: Tokens, pair: [number, number], replacement: number) => {
//   const newTokens = []
//   for (let i = 0; i < tokens.length; ) {
//     if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
//       newTokens.push(replacement)
//       i += 2
//     } else {
//       newTokens.push(tokens[i])
//       i += 1
//     }
//   }

//   return newTokens
// }

// const recursive_bpe = (tokens: Tokens, iterations: number) => {
//   let vocab_size = 257 // started with 256 byte values + 1 for new tokens

//   console.log('BPE training progress:');
//   console.log(`  Step 0: ${tokens.length} tokens, vocab size: ${vocab_size}`);

//   for (let i = 0; i < iterations; i++) {
//     const stats = get_stats(tokens)
//     const top = get_top_pairs(stats, 1)[0]
//     tokens = replace(tokens, top.arr, vocab_size)
//     vocab_size++

//     const decoder = new TextDecoder('utf-8', { fatal: false });
//     console.log(`  Step ${i + 1}: ${tokens.length} tokens, vocab size: ${vocab_size}, replaced pair ${top.arr} with ${vocab_size - 1}, '${decoder.decode(new Uint8Array([top.arr[0], top.arr[1]]))}'`);
//   }

//   return tokens
// }

// const bpe_tokens = recursive_bpe(tokens, 5)

/**
 * const trainer = new Trainer();
 * const codec = trainer.train(input, 20);
 * const encoded = codec.encode(input);
 * const decoded = codec.decode(encoded);
 */

class Trainer {
  #max_vocab_size: number;
  #merges = new Map<number, [number, number]>();

  constructor(max_vocab_size: number) {
    this.#max_vocab_size = max_vocab_size;
    console.assert(this.#max_vocab_size > 256, 'max_vocab_size must be greater than 256');
  }

  train(input: string) {
    const encoder = new TextEncoder()
    const tokens = Array.from(encoder.encode(input))
    const num_merges = this.#max_vocab_size - 256

    let currentTokens = tokens;

    console.log('Training:');
    for (let i = 0; i < num_merges; i++) {
      const stats = get_stats(currentTokens)
      const top = get_top_pairs(stats, 1)[0]

      if (top.count < 2) break;

      const idx = 256 + i
      currentTokens = Trainer.replace(currentTokens, top.arr, idx)
      this.#merges.set(idx, top.arr)
      // console.log(`merge ${i + 1}/${num_merges}: ${top.arr} -> ${idx} (${top.count})`);

      const decoder = new TextDecoder('utf-8', { fatal: false });
      const step = i < 9 ? ` ${i + 1}` : i + 1;
      console.log(`  Step ${step}: ${currentTokens.length} tokens, vocab size: ${num_merges}, replaced pair ${top.arr} with ${idx}, '${decoder.decode(new Uint8Array([top.arr[0], top.arr[1]]))}'`);
    }
    console.log('End Training');

    return new Codec(this.#merges);
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
}

class Codec {
  #merges: [number, [number, number]][];

  constructor(merges: Map<number, [number , number]>) {
    console.assert(merges?.size > 0, 'merges must not be empty');
    // ordered by least to greatest vocab index
    this.#merges = Array.from(merges.entries()).sort((a, b) => a[0] - b[0])
  }

  encode(input: string) {
    const encoder = new TextEncoder()
    let tokens = Array.from(encoder.encode(input))

    // there are no merges with less than two tokens
    if (tokens.length < 2) return tokens;

    // check if any token pairs are in merges, if so, take the one with lowest idx. replace it. repeat.
    while (true) {
      const pairs = get_unique_pairs(tokens)
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
      tokens = Trainer.replace(tokens, l.pair, l.idx)
    }

    return tokens;
  }

  decode(tokens: number[]) {
    debug('decoding', tokens);

    const toks = tokens.slice()

    for (const merge of this.#merges.slice().reverse()) {
      debug({ merge });

      for (let i = toks.length - 1; i >= 0; i--) {
        if (merge[0] === toks[i]) {
          toks.splice(i, 1, ...merge[1])
        }
      }
    }

    return new TextDecoder().decode(new Uint8Array(toks));
  }

  [Symbol.for('Deno.customInspect')]() {
    return `Codec { merges: ${this.#merges.length} }`;
  }
}

const str = new TextDecoder('utf-8').decode(await Deno.readFile('./blog.txt'));
const trainer = new Trainer(277);
const codec = trainer.train(str);
debug({ codec });

const test = (str: string) => {
  const encoded = codec.encode(str);
  const decoded = codec.decode(encoded);
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
