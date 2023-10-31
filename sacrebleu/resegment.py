import sys
import numpy as np
from importlib import import_module
from tqdm import tqdm
_TOKENIZERS = {
    'none': 'tokenizer_none.NoneTokenizer',
    'zh': 'tokenizer_zh.TokenizerZh',
    '13a': 'tokenizer_13a.Tokenizer13a',
    'intl': 'tokenizer_intl.TokenizerV14International',
    'char': 'tokenizer_char.TokenizerChar',
    'ja-mecab': 'tokenizer_ja_mecab.TokenizerJaMecab',
    'ko-mecab': 'tokenizer_ko_mecab.TokenizerKoMecab',
    'spm': 'tokenizer_spm.TokenizerSPM',
    'flores101': 'tokenizer_spm.Flores101Tokenizer',
    'flores200': 'tokenizer_spm.Flores200Tokenizer',
}


def _get_tokenizer(name: str):
    """Dynamically import tokenizer as importing all is slow."""
    module_name, class_name = _TOKENIZERS[name].rsplit('.', 1)
    return getattr(
        import_module(f'.tokenizers.{module_name}', 'sacrebleu'),
        class_name)

class Resegment():
    """A class for resegmenting SLT to fit reference sentence segmentation"""
    TOKENIZERS = _TOKENIZERS.keys()
    # mteval-v13a.pl tokenizer unless Chinese or Japanese is provided
    TOKENIZER_DEFAULT = '13a'

    _TOKENIZER_MAP = {
        'zh': 'zh',
        'ja': 'ja-mecab',
        'ko': 'ko-mecab',
    }

    def __init__(self,tokenize=None):
        # If the tokenizer wasn't specified, choose it according to the
        # following logic. We use 'v13a' except for ZH and JA. Note that
        # this logic can only be applied when sacrebleu knows the target
        # language, which is only the case for builtin datasets.
        if tokenize is None:
            best_tokenizer = self.TOKENIZER_DEFAULT

            # # Set `zh` or `ja-mecab` or `ko-mecab` if target language is provided
            # if self.trg_lang in self._TOKENIZER_MAP:
            #     best_tokenizer = self._TOKENIZER_MAP[self.trg_lang]
        else:
            best_tokenizer = tokenize

        # Create the tokenizer
        self.tokenizer = _get_tokenizer(best_tokenizer)()

        # Build the signature
        self.tokenizer_signature = self.tokenizer.signature()

    def align(self,ref,hyp):
        """First align tokenized hypothesis to tokenized reference
            Then align tokenized hypothesis to original hypothesis
        """
        refLines = [self.tokenizer(l.strip().lower()) for l in ref]
        hypLines = [l.strip() for l in hyp]
        # print(hypLines)
        hypLinesTok = [self.tokenizer(l.lower()).strip() for l in hyp]

        hypLinesTok = self.calcAlignment(refLines,hypLinesTok,500,2,True);

        hypLines = self.calcAlignment(hypLinesTok,hypLines,250,2,False);
        # print('after align')
        # print(hypLines)
        return hypLines


    def calcAlignment(self,refLines,hypLines,beam,replaceCost,words):
        """Calculate alignment
           Then align remaing word to segments
        """
        refString = " ".join(refLines)
        hypString = " ".join(hypLines)

        if(words):
            refWords = refString.split()
            hypWords = hypString.split()
        else:
            refWords = refString
            hypWords = hypString



        #Matrix stores estimate alignment points based on length ratio to perform beam search
        matrix = np.zeros((len(refWords)+1), dtype=np.int32);

        i=0
        while(i < len(refWords)):
            matrix[i]=int(1.0*i/len(refWords)*len(hypWords))
            i+=1

        matrix[len(refWords)] = matrix[len(refWords)-1]


        #Calcuate alignment via minimal edit distance
        operations = self.matches(refWords,hypWords,matrix,beam,replaceCost);


        #iterate through text an align the non-matched words
        op=0
        length=0
        hyp_length=0
        result = []

        for i in range(len(refLines)):
            if(words):
                    length += len(refLines[i].split())
            else:
                    #plus space between lines
                    length += len(refLines[i])+1

            #print ("Ref:",refLines[i])

            #print ("Links:",end=" ")
            #find last matching block in line
            while(op+1 < len(operations) and operations[op+1][0] < length):
                #print (operations[op+1][0],refWords[operations[op+1][0]],end=" ")
                op += 1;
            #print ("")

            #match is across reference regment boundaries
            if (operations[op][0] + operations[op][2] >=length):
                #take split at the appropriate point
                matchingSequence = length - operations[op][0]
                end = operations[op][1] + matchingSequence
            else:
                #start of nonmatching
                start_ref = operations[op][0] + operations[op][2]
                start_hyp = operations[op][1] + operations[op][2]

                if(op +1 < len(operations)):
                    next_ref = operations[op+1][0]
                    next_hyp = operations[op+1][1]
                else:
                    next_ref = len(refWords)
                    next_hyp = len(hypWords)
                #take same ratio in source and target of non-matches
                ratio = 1.0*(length - start_ref) / (next_ref - start_ref)
                end = int(ratio * (next_hyp - start_hyp))+start_hyp
            
            #only split on spaces
            if(not words):
                p1 = hypWords.find(" ",end)
                p2 = hypWords.rfind(" ",hyp_length,end)
                diff1 = p1-end
                diff2 = end-p2
                minDiff = 0;
                if(p1 == -1):
                    if(p2 != -1):
                        end = p2+1
                else:
                    if(p2 == -1 or diff1 < diff2):
                        end = p1+1
                    else:
                        end = p2+1

            if (words):
                result.append(" ".join(hypWords[hyp_length:end]))
                #print("Hype:"," ".join(hypWords[hyp_length:end]))
            else:
                result.append(hypWords[hyp_length:end])
            hyp_length = end

        assert(len(result) == len(refLines));
        return result;



    def matches(self,s1,s2,anchor,beam=100,replaceCost=2):
        l1=len(s1)
        l2=len(s2)

        #store matrix for optimal path
        matrix = np.zeros((l1+1,2*beam+1), dtype=np.int32)
        #store backpointers
        backx=np.zeros((l1+1,2*beam+1), dtype=np.int32)
        backy=np.zeros((l1+1,2*beam+1), dtype=np.int32)

        hits=np.zeros((l1+1), dtype=np.int32)
        hits_sum=np.zeros((l1+1), dtype=np.int32)

        for i in range(l1+1): 
            # if(i <= 5):
            #     anchor[i] = 1.0*i/l1*l2
            # else:
            #     sum = 0;
            #     for j in range(1,6):
            #         sum += hits[i-j]+j/l1*l2
            #     anchor[i] = int(sum/5)

            for j in range(2*beam+1): 
                

                if j == 0: 
                    y=anchor[i]-beam+j
                    matrix[i][j] = i+y
                    backx[i][j]=i-1
                    backy[i][j]=0
                elif i == 0:
                    y=anchor[i]-beam+j                
                    matrix[i][j] = y
                    backx[i][j]=0
                    backy[i][j]=j-1
                else: 

                    y=anchor[i]-beam+j
                    #anchor of previous position might be different
                    prevJ=y-anchor[i-1]+beam
                    

                    #step to the right
                    matrix[i][j]  = matrix[i][j-1] + 1
                    backx[i][j]=i
                    backy[i][j]=j-1
                    
                    #replacement or match
                    if(prevJ > 0 and prevJ < 2*beam+1):
                        jump = matrix[i-1][prevJ-1] + replaceCost
                        if y > 0 and y <= l2 and s1[i-1].lower() == s2[y-1].lower():
                            #print("Match",i-1,y-1)
                            jump = matrix[i-1][prevJ-1]
                            hits_sum[i] +=1;
                            hits[i] += y;
                        if(jump < matrix[i][j]):
                            matrix[i][j] = jump
                            backx[i][j]=i-1
                            backy[i][j]=prevJ-1
                    
                    #step down
                    if(prevJ >= 0 and prevJ < 2*beam+1 and matrix[i-1][prevJ] + 1 < matrix[i][j]):
                        matrix[i][j] =  matrix[i-1][prevJ] + 1
                        backx[i][j]=i-1
                        backy[i][j]=prevJ
            if(hits_sum[i] > 0):
                hits[i] /= hits_sum[i]
            elif(i == 0):
                hits[i] = 0
            else:
                hits[i] = hits[i-1]+l2/l1
        matches = []
        i=l1
        j=2*beam

        #output matches
        while(i >0 or j > 0):
            y=anchor[i]-beam+j
            prevJ=y-anchor[i-1]+beam
            if(y > 0 and y <= l2 and backx[i][j] == i-1 and backy[i][j] == prevJ-1 and s1[i-1].lower() == s2[y-1].lower()):
                matches.append((i-1,y-1,1))
            ii=backx[i][j]
            jj=backy[i][j]
            i=ii
            j=jj
        matches.reverse()
        matches.append((l1,l2,1))
        return matches