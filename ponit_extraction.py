# ltp模块
import sys, os
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import pandas as pd


# segmentor.release()  # 释放模型

class ltp_api(object):
    def __init__(self, MODELDIR, exword_path=None):
        self.MODELDIR = MODELDIR
        self.output = {}
        self.words = None
        self.postags = None
        self.netags = None
        self.arcs = None
        self.exword_path = exword_path  # e.x: '/data1/research/matt/ltp/exwords.txt'
        # 分词
        self.segmentor = Segmentor()
        if not self.exword_path:
            # 是否加载额外词典
            self.segmentor.load(os.path.join(self.MODELDIR, "cws.model"))
        else:
            self.segmentor.load_with_lexicon(os.path.join(self.MODELDIR, "cws.model"), self.exword_path)

        # 词性标注
        self.postagger = Postagger()
        self.postagger.load(os.path.join(self.MODELDIR, "pos.model"))
        # 依存句法
        self.parser = Parser()
        self.parser.load(os.path.join(self.MODELDIR, "parser.model"))
        # 命名实体识别
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(self.MODELDIR, "ner.model"))
        # 语义角色
        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(MODELDIR, "pisrl.model"))

    # 分词
    def ltp_segmentor(self, sentence):
        words = self.segmentor.segment(sentence)
        return words

    # 词性标注
    def ltp_postagger(self, words):
        postags = self.postagger.postag(words)
        return postags

    # 依存语法
    def ltp_parser(self, words, postags):
        arcs = self.parser.parse(words, postags)
        return arcs

    # 命名实体识别
    def ltp_recognizer(self, words, postags):
        netags = self.recognizer.recognize(words, postags)
        return netags

    # 语义角色识别
    def ltp_labeller(self, words, postags, arcs):
        output = []
        roles = self.labeller.label(words, postags, arcs)
        for role in roles:
            output.append([(role.index, arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
        return output

    def release(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
        self.recognizer.release()
        self.labeller.release()

    def get_result(self, sentence):
        self.words = self.ltp_segmentor(sentence)
        self.postags = self.ltp_postagger(self.words)
        self.arcs = self.ltp_parser(self.words, self.postags)
        self.netags = self.ltp_recognizer(self.words, self.postags)
        self.output['role'] = self.ltp_labeller(self.words, self.postags, self.arcs)

        # 载入output
        self.output['words'] = list(self.words)
        self.output['postags'] = list(self.postags)
        self.output['arcs'] = [(arc.head, arc.relation) for arc in self.arcs]
        self.output['netags'] = list(self.netags)


# 解析模块
def get_tuples_word(word_list1, n1, word_list2, n2):
    # 按照顺序，拼接词
    result = []
    for i, n1s, j, n2s in zip(word_list1, n1, word_list2, n2):
        if n1s < n2s:
            result.append(''.join([i, j]))
        else:  # n1s > n2s
            result.append(''.join([j, i]))
    return result


def Parser2dataframe(words, postags, arcs):
    '''
    把依存句法解构成dataframe
    '''
    word_dict = dict(enumerate(words))
    match_word = []
    relation = []
    pos = []
    match_word_n = []
    # 解读
    for n, arc in enumerate(arcs):
        relation_word = 'root ' if arc.head - 1 < 0 else word_dict[arc.head - 1]  # 核心词，root，为空
        match_word.append(relation_word)
        relation.append(arc.relation)
        pos.append(postags[n])
        match_word_n.append(0 if arc.head - 1 < 0 else arc.head - 1)

    tuples_words = pd.DataFrame({'word': list(word_dict.values()), 'word_n': list(word_dict.keys()), \
                                 'match_word': match_word, 'relation': relation, 'pos': pos,
                                 'match_word_n': match_word_n})
    tuples_words['tuples_words'] = get_tuples_word(tuples_words['word'], tuples_words['word_n'], \
                                                   tuples_words['match_word'], tuples_words['match_word_n'])
    return tuples_words


# 实体名词搭配
def FindEntityCollocation(tuples_words, neg_words=['是', '又', '而且', 'root']):
    return [wo for wo in list(tuples_words['tuples_words'][tuples_words['pos'] == 'n']) if 'root' not in wo]


# 通用内容搭配
def FindCollocation(tuples_words, neg_words=['是', '又', '而且']):
    SBV_output, ADJ_output = '', ''
    if sum(tuples_words['relation'] == 'COO') > 0:
        first_word = tuples_words['match_word'][tuples_words['relation'] == 'SBV']
        second_word = tuples_words['word'][tuples_words['relation'] == 'SBV']
        SBV_output = [wo for wo in list(zip(second_word, first_word)) if len(set(neg_words) & set(wo)) == 0]

    if (sum(tuples_words['relation'] == 'ADV') > 0) or (sum(tuples_words['relation'] == 'ATT') > 0):
        # ADV部分
        first_word = tuples_words['match_word'][tuples_words['relation'] == 'ADV']
        second_word = tuples_words['word'][tuples_words['relation'] == 'ADV']
        ADJ_output_1 = [wo for wo in list(zip(second_word, first_word)) if len(set(neg_words) & set(wo)) == 0]
        # ATT部分
        first_word = tuples_words['match_word'][tuples_words['relation'] == 'ATT']
        second_word = tuples_words['word'][tuples_words['relation'] == 'ATT']
        ADJ_output_2 = [wo for wo in list(zip(second_word, first_word)) if len(set(neg_words) & set(wo)) == 0]
        # 相连
        ADJ_output = ADJ_output_1 + ADJ_output_2
    return SBV_output, ADJ_output


# 并列名词查找
def FindSynonym(tuples_words, neg_words=['是', '又', '而且']):
    output = ''
    if sum(tuples_words['relation'] == 'COO') > 0:
        first_word = tuples_words['match_word'][tuples_words['relation'] == 'COO']
        second_word = tuples_words['word'][tuples_words['relation'] == 'COO']
        output = [wo for wo in list(zip(second_word, first_word)) if len(set(neg_words) & set(wo)) == 0]
    return output


# 总结核心
# 以：主 + 谓 + 宾为核心
# sentense = '全书有数百个具体的例子，并被组织成了紧密的实用概念框架，能够适用于各个层次上的经理人与创业者。'
def includeSth(sth, list_sth):
    return [i in sth for i in list(list_sth)]


def includeSBV_VOB(list_sth):
    return True if sum([i in list(list_sth) for i in ['SBV', 'VOB']]) == 2 else False


def SBV_VOB_bind(core_data, core_n, words):
    SBV_VOB_n = list(core_data[includeSth(['SBV', 'VOB'], core_data['relation'])]['word_n'])
    SBV_VOB_n.extend(list(core_n))
    center_words = ''
    for i in sorted(SBV_VOB_n):
        center_words = ''.join([center_words, words[i]])
    return center_words


def CoreExtraction(tuples_words):
    core_n = tuples_words[tuples_words['relation'] == 'HED']['word_n']
    core_data = tuples_words[tuples_words['match_word_n'] == int(core_n)]
    core = ''
    if includeSBV_VOB(core_data['relation']):
        # SBV_VOB构成主谓宾，就是自动摘要了，最好两个都有
        # print (SBV_VOB_bind(core_data,words))
        core = SBV_VOB_bind(core_data, core_n, words)
    elif sum(includeSth(['SBV'], core_data['relation'])) > 0:
        # 主谓关系
        # print (list(core_data[includeSth(['SBV'],core_data['relation'])]['tuples_words']))
        core = list(core_data[includeSth(['SBV'], core_data['relation'])]['tuples_words'])
    elif sum(includeSth(['VOB'], core_data['relation'])) > 0:
        # 动宾关系
        # print (list(core_data[includeSth(['VOB'],core_data['relation'])]['tuples_words']))
        core = list(core_data[includeSth(['VOB'], core_data['relation'])]['tuples_words'])
    elif sum(tuples_words['relation'] == 'HED') > 0:
        core = list(tuples_words['word'][tuples_words['relation'] == 'HED'])
    return core


if __name__ == "__main__":
    pass
    '''
    ltp初始化，但是有个问题就是一定需要释放，不能重复初始化，不然会出现爆内存的情况
    '''
    #  模型文件
    MODELDIR = '/home/amei/Data/NLP/ltp_data_v3.4.0'
    ltp = ltp_api(MODELDIR)
    # ltp.release()
    paragraph = ''''''
    '''paragraph = """环境很好，位置独立性很强，比较安静很切合店名，半闲居，偷得半日闲。点了比较经典的菜品，味道果然不错！烤乳鸽，
    超级赞赞赞，脆皮焦香，肉质细嫩，超好吃。艇仔粥料很足，香葱自己添加，很贴心。金钱肚味道不错，不过没有在广州吃的烂，牙口不好的慎点。
    凤爪很火候很好，推荐。最惊艳的是长寿菜，菜料十足，很新鲜，清淡又不乏味道，而且没有添加调料的味道，搭配的非常不错！"""
    '''
    sentences = SentenceSplitter.split(paragraph)

    for sentence in sentences:
        if sentence:
            print('\n===================== 原句 =====================\n')
            print(sentence)
            # 第一种：类内计算
            words = ltp.ltp_segmentor(sentence)  # 分词
            postags = ltp.ltp_postagger(words)  # 词性
            print(postags[0])
            for i in range(0,len(words)):
                if postags[i] == 'ns':
                    print(words[i])
            '''
            arcs = ltp.ltp_parser(words, postags)  # 依存
            netags = ltp.ltp_recognizer(words, postags)  # 命名实体识别
            labeller = ltp.ltp_labeller(words, postags, arcs)  # 语义角色

            # ltp.get_result(sentence)
            # output = ltp.output
            # arcs = output['arcs']
            # netags = output['netags']
            # postags = output['postags']
            # labeller = output['role']
            # words = output['words']
            tuples_words = Parser2dataframe(words, postags, arcs)

            print('\n----- 搭配用语查找 -----\n')
            print(FindCollocation(tuples_words))
            print('\n----- 并列词查找 -----\n')
            print(FindSynonym(tuples_words))
            print('\n----- 核心观点抽取 -----\n')
            print(CoreExtraction(tuples_words))
            print('\n----- 实体名词搭配 -----\n')
            print(FindEntityCollocation(tuples_words))
            '''
