# -*- coding: utf-8 -*-
"""
@author: Ricardo Alvarez
@class: COMP 550 
"""


from nltk import CFG
import nltk
import pandas as pd
import re
from nltk.treeprettyprinter import TreePrettyPrinter

with open (r'C:\Users\Ricardo\My Drive\McGill\COMP 550\Assignment 2\french-grammar.txt') as myfile:
    grammar = myfile.read()

french_grammar = CFG.fromstring(grammar)

class CYK:
    def __init__(self, grammar, sentence):
        self.grammar = grammar
        self.original_grammar = grammar
        self.sentence = sentence 
        
    def CNF(self):
        # A -> B C D
        # ------ binarize --------
        # A -> B C_D
        # C_D -> C D 
        
        binary_rules = []
        
        for rule in (self.grammar.productions()):
            # As we must only handle non binary rules
            if len(rule.rhs()) > 2:
                #print(type(rule))
                lhs = rule.lhs()
                # As we must not alter the symbols of the last two rule items
                for i in range(0, len(rule.rhs())- 2):
                    rule_item = rule.rhs()[i]
                    #print(rule_item)
                    new_rhs1 = rule.rhs()[i+1]
                    new_rhs2 = rule.rhs()[i+2]
                    new_rule_item = nltk.grammar.Nonterminal(new_rhs1.symbol()\
                                                    + "-" + new_rhs2.symbol())
                    #print(new_rule_item)
                    new_rule = nltk.grammar.Production(lhs, (rule_item, new_rule_item))
                    lhs = new_rule_item
                    binary_rules.append(new_rule)
                # To not forget the last rule, equivalent of C_D -> C D
                last_rule = nltk.grammar.Production(lhs, (rule.rhs()[-2:]))
                binary_rules.append(last_rule)
                
            else:
                binary_rules.append(rule)

        # To avoid repetitions in parsing trees
        binary_rules = list(set(binary_rules))
        self.grammar.productions = list(set(self.grammar.productions()))
        
        
        binary_grammar = CFG(self.grammar.start(), binary_rules)
        
        # Convert unitary rules 
        # A -> B 
        # B -> C D
        # -------- Remove unitary -------
        # A -> C D 
    
        unitary_rules = []
        nonunitary_rules = []
        
        for rule in binary_grammar.productions():
            if len(rule.rhs()) == 1 and rule.is_nonlexical():
                unitary_rules.append(rule)
            else:
                nonunitary_rules.append(rule)
                
        while len(unitary_rules) != 0:
            current = unitary_rules.pop(0)
            # print(current)
            # NP -> NP3S
            # NP3S -> DTM NMS
            #---------------
            # NP -> DTM NMS
            
            
            for binary_rule in binary_grammar.productions():
                # To equalize symbols and not tuples: rhs()[0]
                if binary_rule.lhs() == current.rhs()[0]:
                    #print(binary_rule.lhs())
                    #print(current.rhs()[0])
                    new_non_unitary_rule = nltk.grammar.\
                        Production(current.lhs(), (binary_rule.rhs()))    
                    
                    # To be able to reach lexicals:
                    # NP -> NP3S
                    # NP3S -> PN 
                    # PN -> Montreal
                    # ------------------------
                    # NP -> Montreal
                    # NP3S -> Montreal
                    # PN -> Montreal
                    
                    # And to keep original rules that are not lexical
                    # but non-unitary such as NP -> DTM NMS   
                    
                    #print(len(new_non_unitary_rule))
                    
                    if len(new_non_unitary_rule.rhs()) != 1 or \
                       new_non_unitary_rule.is_lexical():
                        nonunitary_rules.append(new_non_unitary_rule)
                    else:
                        unitary_rules.append(new_non_unitary_rule)
                        #print(new_non_unitary_rule)
                        
                else:
                    continue
        
        # To avoid repetitions in parsing trees
        nonunitary_rules = list(set(nonunitary_rules))
        binary_grammar.productions = list(set(binary_grammar.productions()))
        
        CNF_grammar = CFG(binary_grammar.start(), nonunitary_rules)

        # Verify that it is in CNF and number of productions is the same
        #print(CNF_grammar.is_chomsky_normal_form()) 
        #print(self.grammar.chomsky_normal_form())
        

        self.grammar = CNF_grammar
       
        return CNF_grammar
    

    def convert_to_tree(self, dp_table, coord, interpretation, head):
        
        l = coord[0]
        r = coord[1]
                             
        
        tree_list = []
        
        if dp_table[r][l]["bp"] == []:
            return [self.sentence[r]]
        else:
            (coordl1, coordr1) = dp_table[r][l]["bp"][interpretation][0]
            (coordl2, coordr2) = dp_table[r][l]["bp"][interpretation][1]
        
        #print(coordl1, coordr1)
        #print(coordl2, coordr2)
        non_term_list1 = dp_table[coordr1][coordl1]["non-term"]
        non_term_list2 = dp_table[coordr2][coordl2]["non-term"]        
        
        for non_term1 in non_term_list1:
            for non_term2 in non_term_list2:
                #print(non_term.symbol())
                #print(dp_table[r1][l1]["non-term"][interpretation].symbol())
                for rule in self.grammar.productions():
                    #print(rule.lhs().symbol())
                    #print(rule.rhs())
                    #print(rule)  
                    #dp_table[r][l]["non-term"][interpretation].symbol() == rule.lhs().symbol() and\ 
                    if rule.is_nonlexical() and\
                        head == rule.lhs().symbol() and\
                        non_term1.symbol() == rule.rhs()[0].symbol() and\
                            non_term2.symbol() == rule.rhs()[1].symbol():
                           
                           #print(non_term1.symbol())
                           #print(non_term2.symbol())
                           tree_list.append(nltk.tree.Tree(non_term1.symbol(),\
                                self.convert_to_tree(dp_table, (coordl1, coordr1), interpretation, non_term1.symbol())))
                           tree_list.append(nltk.tree.Tree(non_term2.symbol(),\
                                self.convert_to_tree(dp_table, (coordl2 ,coordr2), interpretation, non_term2.symbol())))
                               
                           #print(tree_list)
                           
                    else:
                        continue
            
            
            #print(dp_table[r1][l1]["bp"][interpretation][coord])
        
        
        return tree_list
    
    def tree_not_CNF(self, tree):
        
        new_children = []
        
        for subtree in tree:
            #print(subtree)
            #print(parent)
            if isinstance(subtree, nltk.tree.Tree):
                if "-" in subtree.label():
                    
                    for subtreei in range(len(subtree)): 
                        #print(subtreei)
                        #new_children.append(nltk.tree.Tree(subtree[subtreei].label(), subtree[subtreei]))
                        new_children.append(self.tree_not_CNF(subtree[subtreei]))
                else:
                    #new_children.append(nltk.tree.Tree(subtree.label(), subtree))
                    new_children.append(self.tree_not_CNF(subtree))
                     
                new_tree = nltk.tree.Tree(tree.label(), new_children)
            
            else:
                return nltk.tree.Tree(tree.label(), [subtree])
               
        return new_tree
    
    def parse(self, sentence):
        
        self.CNF()
        
        # Keep hypens in proper nouns
        sentence = re.sub(r'[^\w\s\-]','', sentence)
        
        # Handle negation
        s_NOT = re.split(" (ne) ", sentence)
        
        #print(s_NOT)
        if len(s_NOT) != 1:
            s1 = re.split(" (pas) ", s_NOT[2])
            s2 = s_NOT[0].split(" ") 
            s3 = s1[-1].split(" ")
            v = ["ne " + s1[-3] + " pas"]
            s2.append(v[0])
            
            for i in range(len(s3)):
                s2.append(s3[i])
            
            #print(s2)
            sentence = list(filter(None, s2))

        else:
            sentence = [sentence][0].split(' ')
        
        self.sentence = sentence
        
        n = len(sentence)
        #print(n)
        table_struc = [[ {'non-term': [], 'bp':[] } \
                        for j in range(n) ] for i in range(n)]
        # Not the most efficient, but easier and not evaluated on O(n)
        # For debugging
        global dp_table 
        dp_table = pd.DataFrame.from_dict(table_struc)
            
        #print(sentence)      
        l = 0 
        r = 0
        
        for word in range(len(sentence)):
            #for k in range(len(sentence) - word): 
            l = 0
            r = word
            while r >= l and (r <= (len(sentence) - 1)): 
                #print(str(l) + "-" + str(r))
                                
                # Base Case
                if l == r:
                    for rule in self.grammar.productions():
                        # Proper names in the sentence and grammar are handled
                        if rule.is_lexical() and\
                           sentence[l].lower() == rule.rhs()[0].lower():
                            #print(rule)
                            dp_table[r][l]["non-term"].append(rule.lhs())
                            #dp_table[word][word]["bp"]
                
                # Recursive Step
                else: 
                    for breakp in range(l, r):
                        #print((l, breakp))
                        #print((breakp + 1, r))
                        #print(breakp)
                        for rule in self.grammar.productions():  
                            for non_term1 in dp_table[breakp][l]["non-term"]:
                                for non_term2 in dp_table[r][breakp + 1]["non-term"]:
                                    if (non_term1, non_term2) == rule.rhs():
                                        #print((non_term1, non_term2))
                                        #print(rule.rhs())
                                        #print(rule.lhs())
                                        dp_table[r][l]["non-term"].append(rule.lhs())
                                        dp_table[r][l]["bp"].append(((l, breakp) , (breakp + 1, r)))                                            
                    #print("-------")            
                
                l += 1
                r += 1
                
        #print(self.grammar)
        
        tree_list = []
        
        for interpretation in range(len(dp_table[n-1][0]["non-term"])):
            # Interpretation must be a valid sentence
            if dp_table[n-1][0]["non-term"][interpretation].symbol() == 'S':
                tree = nltk.tree.Tree('S', self.convert_to_tree(dp_table, (0, n-1), interpretation, 'S'))           
                #print(tree)
                
            
            else:
                continue 
            
            tree_list.append(tree)
        
        final_list = []
        
        for tr in tree_list:
            tpp = TreePrettyPrinter(tr)
            print(tpp.text())
            new_t = self.tree_not_CNF(tr)
            final_list.append(new_t)
               
        return final_list  
    

sentence = "Le chat ne mange pas le poisson"
    
cyk = CYK(french_grammar, sentence)
# Les belles Iles-de-la-Madelaine regardent la television.
# Je regarde la television.
# Le beau Jonathan ne mange pas les chattes.
# Le chat regarde la television.
# Les jolies Nations-Unies ne regardent pas les anciens Etats-Unis!
# Les jolies Nations-Unies ne regardent pas les Etats-Unis anciens!
# Jonathan la regarde.
# Les Canadiens les mangent. 
# Nathan ne mange pas les poissons noirs.
# Je mange

# Mangeons!
# Le chats mangent la television.
# Breaks my code: Je pas mange ne la television.

tl = cyk.parse(sentence)


for tree in tl:
    tpp = TreePrettyPrinter(tree)
    print(tpp.text())
    

