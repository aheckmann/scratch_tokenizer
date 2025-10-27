# Compare tiktoken vs SentencePiece on Chinese text
chinese_text = "你好世界"  # "Hello World" in Chinese

print(f"Text: {chinese_text}")
print(f"UTF-8 bytes: {chinese_text.encode('utf-8')}")
print(f"Unicode code points: {[ord(c) for c in chinese_text]}")

# tiktoken approach: work on bytes
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tiktoken_tokens = enc.encode(chinese_text)
print(f"tiktoken tokens: {tiktoken_tokens} (count: {len(tiktoken_tokens)})")

# SentencePiece approach: work on code points (if we had it installed)
# !pip install sentencepiece  # Uncomment to install

# For comparison, let's see the difference in approach:
print("tiktoken approach:")
print("1. Characters → UTF-8 bytes → BPE merges bytes")
for char in chinese_text:
    utf8_bytes = char.encode('utf-8')
    print(f"  '{char}' → {utf8_bytes} → separate tokens for each byte")

print("\nSentencePiece approach:")
print("2. Characters → Unicode code points → BPE merges code points")
for char in chinese_text:
    code_point = ord(char)
    print(f"  '{char}' → U+{code_point:04X} → can merge whole characters")


print(f"\nMoving on to SentencePiece training...\n")

# SentencePiece tokenization
# Create toy training data
# with open("toy.txt", "w", encoding="utf-8") as f:
#     f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

import sentencepiece as spm
import os

# Train a SentencePiece BPE model
# These settings match those used for training Llama 2

options = dict(
    # Input spec
    # input="taylorswift.txt",
    input="toy.txt",
    input_format="text",
    # Output spec
    model_prefix="tok400", # output filename prefix
    # Algorithm spec - BPE algorithm
    model_type="bpe",
    vocab_size=400,
    # Normalization (turn off to keep raw data)
    normalization_rule_name="identity", # turn off normalization
    remove_extra_whitespaces=False,
    input_sentence_size=200000000, # max number of training sentences
    max_sentence_length=4192, # max number of bytes per sentence
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    # Rare word treatment
    character_coverage=0.99995,
    byte_fallback=True,
    # Merge rules
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    # Special tokens
    unk_id=0, # the UNK token MUST exist
    bos_id=1, # the others are optional, set to -1 to turn off
    eos_id=2,
    pad_id=-1,
    # Systems
    num_threads=os.cpu_count(), # use ~all system resources
)

spm.SentencePieceTrainer.train(**options);

# Load and inspect the trained model
sp = spm.SentencePieceProcessor()
sp.load('tok400.model')

# Show the vocabulary - first few entries
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
print("First 20 tokens:")
for token, idx in vocab[:20]:
    print(f"  {idx}: '{token}'")

print(f"\nTotal vocabulary size: {len(vocab)}")

print("sentencepiece vocab structure:")
print("=" * 40)

print("1. Special tokens")
for i in range(3):
    print(f" {i}: '{sp.id_to_piece(i)}'")

print("\n2. Byte tokens (next 256 entries:")
print(" 3 to 258: <0x00> through <0xFF>")
for i in [3,4,5,256, 257,258]: #show first and last few
    print(f" {i}: '{sp.id_to_piece(i)}'")

print("\n3. Merge tokens (BPE learned merges)")
print("  259-399: Learned BPE merges")
for i in range(259, min(269, sp.get_piece_size())): # show first 10 merges
    print(f" {i}: '{sp.id_to_piece(i)}'")

print("\n4. individual code point tokens:")
print("  These are raw Unicode characters from training data")
# find where individual tokens starrt (after merges)
for i in range(350, min(400, sp.get_piece_size())):
    piece = sp.id_to_piece(i)
    if len(piece) == 1 and not piece.startswith('<'): # single char, not a byte token
        print(f" {i}: '{piece}'")
        if i > 370:
            break

print("=" * 40)

# Test the SentencePiece tokenizer
# test_text = "hello 안녕하세요"
test_text = "안"
ids = sp.encode(test_text)
pieces = [sp.id_to_piece(idx) for idx in ids]

print(f"Text: '{test_text}'")
print(f"Token IDs: {ids}")
print(f"Token pieces: {pieces}")
print(f"Decoded: '{sp.decode(ids)}'")

# Notice how Korean characters become byte tokens due to byte_fallback=True


# Test text with 1000 characters total
test_text_chinese_korean = """在古老的东方,有一座被群山环绕的神秘城市。这座城市的历史可以追溯到数千年前,那时候智慧的先人们就在这里建立了繁荣的文明。城墙高耸入云,宫殿金碧辉煌,街道上商贾云集,人来人往络绎不绝。每当夕阳西下,整座城市都会被染上一层金色的光芒,美丽得令人心醉。城中的图书馆收藏着无数珍贵的古籍,记载着这片土地上发生过的传奇故事。学者们日夜研读这些典籍,希望能够从中找到通往更高智慧的道路。

随着时间的流逝,这座城市经历了无数次的兴衰变迁。战争、瘟疫、自然灾害都曾威胁过它的存在,但每一次它都顽强地挺过来了,并且变得更加坚韧。现代化的浪潮席卷而来,高楼大厦拔地而起,古老的建筑与现代的摩天大楼交相辉映,形成了独特的城市风貌。人们在这里工作、生活、追求着自己的梦想。科技的发展让生活变得更加便利,但人们依然没有忘记那些传统的价值观和文化遗产,他们将古老的智慧与现代的创新完美地结合在一起。

한반도의 남쪽에 위치한 아름다운 마을이 있었습니다. 그곳은 사계절이 뚜렷하고 자연이 풍요로운 곳이었죠. 봄에는 벚꽃이 만발하여 온 동네가 분홍빛으로 물들었고, 여름에는 푸른 논밭이 황금빛 파도를 이루었습니다. 가을이 되면 단풍이 산을 붉게 물들이고, 겨울에는 하얀 눈이 세상을 덮어 마치 동화 속 풍경 같았습니다. 마을 사람들은 대대로 이곳에서 살아왔으며, 서로 돕고 의지하며 공동체를 이루어 살았습니다. 할머니들은 마당에 모여 앉아 옛날이야기를 들려주었고, 아이들은 골목길을 뛰어다니며 즐겁게 놀았습니다.

시간이 흘러 세상은 많이 변했지만, 이 마을 사람들의 마음만큼은 여전히 따뜻했습니다. 젊은이들이 도시로 떠나고 인구가 줄어들었지만, 남아있는 사람들은 더욱 끈끈한 유대감으로 뭉쳤습니다. 그들은 전통을 지키면서도 새로운 변화를 받아들였습니다. 작은 카페가 생기고, 예술가들이 이곳에 정착하기 시작했습니다. 주말이면 도시에서 사람들이 찾아와 마을의 평화로운 분위기를 즐겼습니다. 마을 어르신들은 옛 방식대로 된장과 김치를 담그며 전통의 맛을 이어갔고, 젊은 세대는 인터넷을 통해 마을의 아름다움을 세상에 알렸습니다. 이렇게 과거와 현재가 조화롭게 공존하는 마을은 모두에게 희망과 위안을 주는 특별한 장소가 되었습니다."""


ids = sp.encode(test_text_chinese_korean)
pieces = [sp.id_to_piece(idx) for idx in ids]

# print(f"Text: '{test_text_chinese_korean}'")
print(f"Chinese and Korean test")
print(f"Token IDs: {ids}")
print(f"Token pieces: {pieces}")
print(f"Decoded: '{sp.decode(ids)}'")
