from collections import defaultdict
from tqdm import tqdm

class Tokenizer:
    @staticmethod
    def Encode(content, vocab):
        encoded = bytearray(content.encode('utf-8'))
        before = len(encoded)
        merges = {}

        def find_top_pairs(data):
            pair_counts = defaultdict(int)
            for i in range(0, len(data)-1, 2):
                pair = (data[i], data[i+1])
                pair_counts[pair] += 1
            return max(pair_counts.items(), key=lambda x: x[1], default=(None, 0))[0]

        def merge_pairs(data, pair, merge_id):
            merges={}
            new_data = []
            i = 0
            while i < len(data):
                if i < len(data)-1 and (data[i], data[i+1]) == pair:
                    new_data.append(merge_id)
                    merges[merge_id] = pair
                    i += 2
                else:
                    new_data.append(data[i])
                    i += 1
            return new_data, merges

        vocab = vocab
        diff = vocab - 256
        merged={}
        print("Encoding...")
        from tqdm import tqdm
        for i in tqdm(range(diff)):
            result = find_top_pairs(encoded)
            results, merges = merge_pairs(encoded, result, i)
            merged[list(merges.values())[0]]=list(merges.keys())[0]
            encoded = results
        after = len(encoded)
        print(f"{100.00 - ((after / before) * 100):.2f}% compressed vocab")
        return encoded, merged

    @staticmethod
    def Decode(merged, encoded):
        updated = []
        for item in reversed(list(merged.keys())): updated.append(item)
        reordered = {k: merged[k] for k in updated}
        print("Decoding...")
        for i in tqdm(range(len(list(reordered.keys())))):
            temp=[]
            for j in range(len(encoded)):
                if list(reordered.keys())[i] == encoded[j]:
                    temp.append(list(reordered.values())[i][0])
                    temp.append(list(reordered.values())[i][1])
                else:
                    temp.append(encoded[j])
            encoded=temp
        tokens = bytes(encoded)
        return str(tokens.decode("utf-8", errors="replace"))
