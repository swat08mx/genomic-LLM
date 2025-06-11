class Tokenizer:

    def Encode(content, vocab):
        #text = "Indian autism simplex families   were enrolled in our study. Out of 72 samples originally sequenced, we had chosen 57 samples for our study, the other had to be eliminated due to presence of neuronal disorder history in their families. We performed WES (Whole Exome Sequencing) on the samples from 57 individuals. The vcf files of all the samples were merged in a single vcf file after removing the duplicates, which contained ~2500000 mutations. On an average ~150000 variants were identified per sample. In order to understand  the burden of the variants at the chromosomal level, the total number of variants in each chromosome in each sample were plotted in a box plot (Supplementary figure). Box plot indicates that chromosome 1 is showing to have more variant burden and this was due to the default size of the chromosome. In contrast,  when the size normalized burden plot (Figure 1, A) was plotted, we observed  that chromosome 19 has more variants per base pair, and the data spread is fairly smaller when compared to the other chromosomes, indicating its consistency in the number of variants in each sample. After filtering for rare variants with minor allele frequency (MAF) < 0.01 in all annotated populations in the gnomAD database, we had ~10000 variants per sample. We found that a total of 3814 (1.6% of the total mutations) variants were not reported in any of the gnomAD populations and were novel mutations across the cohort. Principal component analysis (PCA) was applied to reduce the dimensionality of rare variant data, followed by clustering to visualize the heterogeneity between variants (Figure 1B). The resulting plot revealed two distinct groupings: a dominant cluster localized to the right quadrant and a spread of sparse data points distributed across the left quadrant. Ultra-rare variants (minor allele frequency [MAF] < 1×10⁻⁶ in the gnomAD database [v3.1.2] across all populations) are highlighted in green and pathogenic rare variants (at least 5 out of 9 tools predicted it to be damaging) are highlighted in red. Strikingly, 98% of ultra-rare variants and 100% of the pathogenic variants co-localized with the high-density cluster on the right, contrasting with the sparse distribution of relatively common variants."
        text = content
        encoded = list(text.encode("utf-8"))
        before = len(encoded)
        def find_top(encoded):
          dicto={}
          try:
            for i in range(0, len(encoded), 2):
              two = (encoded[i], encoded[i+1])
              if two not in dicto:
                dicto[two] = 1
              else:
                dicto[two] +=1
          except:
            pass
          if not dicto:
            return "No merges possible"
        
          def bubble_sort_dict_by_values(dictionary):
              items_list = list(dictionary.items())
              n = len(items_list)
              for i in range(n):
                  for j in range(0, n - i - 1):
                      if items_list[j][1] < items_list[j + 1][1]:
                          items_list[j], items_list[j + 1] = items_list[j + 1], items_list[j]
              sorted_dict = dict(items_list)
              return sorted_dict
        
          result = bubble_sort_dict_by_values(dicto)
          top = list(list(result.keys())[0])
          return top
        
        def merge_pairs(encoded, result, num):
          new_ids=[]
          merges={}
          i=0
          while i < len(encoded) -1:
            two = list((encoded[i], encoded[i+1]))
            if two != result:
              new_ids.append(encoded[i])
              new_ids.append(encoded[i+1])
            else:
              new_ids.append(256+num)
              merges[(encoded[i]), encoded[i+1]] = 256+num
            i+=2
          if len(encoded)%2!=0:
            new_ids.append(encoded[-1])
          return new_ids, merges
        
        vocab = vocab
        diff = vocab - 256
        merged={}
        print("Encoding...")
        from tqdm import tqdm
        for i in tqdm(range(diff)):
          result = find_top(encoded)
          results, merges = merge_pairs(encoded, result, i)
          merged[list(merges.values())[0]]=list(merges.keys())[0]
          encoded = results
        after = len(encoded)
        print(f"{100.00 - ((after / before) * 100):.2f}% compressed vocab")

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
        print(str(tokens.decode("utf-8", errors="replace")))
