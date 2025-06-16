from collections import defaultdict
from tqdm import tqdm

class Tokenizer:

    def Encode(self, content, vocab):
        encoded = bytearray(content.encode('utf-8'))
        before = len(encoded)

        def find_top_pairs(data):
            pair_counts = defaultdict(int)
            for i in range(0, len(data)-1):
                pair = (data[i], data[i+1])
                pair_counts[pair] += 1
            return max(pair_counts.items(), key=lambda x: x[1], default=(None, 0))[0]

        def merge_pairs(data, pair, merge_id):
            new_data = []
            merges={}
            i = 0
            while i < len(data)-1:
                if i < len(data)-1 and (data[i], data[i+1]) == pair:
                    new_data.append(merge_id)
                    merges[merge_id] = pair
                    i += 2
                else:
                    new_data.append(data[i])
                    i += 1
            if len(data)%2!=0:
                new_data.append(data[-1])
            return new_data, merges

        vocab = vocab
        diff = vocab - 256
        merged={}
        print("Encoding...")
        from tqdm import tqdm
        for i in tqdm(range(diff)):
            result = find_top_pairs(encoded)
            results, merges = merge_pairs(encoded, result, 257+i)
            merged[list(merges.values())[0]]=list(merges.keys())[0]
            encoded = results
        after = len(encoded)
        print(f"{100.00 - ((after / before) * 100):.2f}% compressed vocab")
        return encoded, merged


    def Decode(merged, encoded):
        id_to_pair = {v: k for k, v in merged.items()}
        merge_ids = sorted(id_to_pair.keys(), reverse=True)
        print("Decoding...")
        for merge_id in tqdm(merge_ids):
            pair = id_to_pair[merge_id]
            temp = []
            i = 0
            while i < len(encoded):
                if encoded[i] == merge_id:
                    temp.extend(pair)
                    i += 1
                else:
                    temp.append(encoded[i])
                    i += 1
            encoded = temp
        
        tokens = bytes(encoded)
        return tokens.decode("utf-8", errors="replace")

    def build_vocab(merged, total_tokens):
        vocab = merged
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in merged.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        new_vocab = {v: k for k, v in vocab.items()}
        new_vocab["<EOS>"]=total_tokens+1
        new_vocab["<SOS>"]=total_tokens+2
        new_vocab["<EOS>"]=total_tokens+3
        new_vocab["<CLS>"]=total_tokens+4
        return new_vocab

    def split_data(self, data, splits):
        splitted_data=[]
        count=0
        temp=[]
        for i in tqdm(range(len(data))):
            if count==splits:
                count=0
                #final = ''.join(temp)
                splitted_data.append(temp)
                temp=[]
            temp.append(data[i])
            count+=1
        return splitted_data


class Data_process(Tokenizer):
    def Encode_data(self, data, vocab_size, splits):
        encoded,_ = self.Encode(data, vocab_size)
        clean_data = self.split_data(encoded, splits)
        #tokens = [vocab["<SOS>"]] + [vocab[w.encode("utf-8")] for w in data] + [vocab["<EOS>"]]
        return clean_data
