from collections import defaultdict
from tqdm import tqdm

class Tokenizer:
    
    def Encode(content, vocab):
        def find_top(encode):
            encoded = list(encode.encode("utf-8"))
            before = len(encoded)
            dicto = defaultdict(int)
            for i in range(0, len(encoded) - 1, 2):
                two = (encoded[i], encoded[i+1])
                dicto[two] += 1
            if not dicto:
                return "No merges possible"
            # Use built-in sorted for efficiency
            top = max(dicto.items(), key=lambda item: item[1])[0]
            return list(top)
    
        def merge_pairs(encoded, result, num):
            new_ids = []
            merges = {}
            i = 0
            while i < len(encoded) - 1:
                two = [encoded[i], encoded[i+1]]
                if two != result:
                    new_ids.extend(two)
                else:
                    new_ids.append(256 + num)
                    merges[(encoded[i], encoded[i+1])] = 256 + num
                i += 2
            if len(encoded) % 2 != 0:
                new_ids.append(encoded[-1])
            return new_ids, merges
    
    
        vocab = vocab
        diff = vocab - 256
        merged={}
        encoded=content
        print("Encoding...")
        from tqdm import tqdm
        for i in tqdm(range(diff)):
            result = find_top(encoded)
            print("Found top pair")
            results, merges = merge_pairs(encoded, result, i)
            print("Merged pairs")
            merged[list(merges.values())[0]]=list(merges.keys())[0]
            encoded = results
            print("Restarted")
        after = len(encoded)
        print(f"{100.00 - ((after / before) * 100):.2f}% compressed vocab")
        return merged, encoded

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
