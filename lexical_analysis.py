"""This module contains the code to process, extract, and store the lexical information 
gathered from target methods (i.e., `source_code_analysis.models.CodeMethod`).

In particular, two classes are provided:
    `LinsenNormalizer`: class to invoke the LINSEN algorithm to normalize code tokens'
    `LexicalAnalyzer`: class to boost the whole lexical analysis process, and to store
                       `CodeLexiconInfo` instances associates to each method.
"""
# Author: Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

from shlex import split as shlex_split
from subprocess import PIPE, Popen
import os
import subprocess

# Import NLTK
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

# Import PyTrie (Python Prefix Tree Implementation)
# --------------
# The `pytrie` module is used to create the list of identifiers that should be fed to the LINSEN algorithm
# 
# In particular, every single word that does not correspond to the prefix of any word in the English dictionary, 
# is considered a target for the LINSEN normalizer algorith.
# 
# The list of candidate identifiers will be then stored into a textual file whose path will be passed to 
# the LINSEN algorithm via the `target_identifiers_list` entry in the `TEST_DATA_CONF` configuration file.

from pytrie import SortedStringTrie as trie


# ----------------------
# LINSEN Execution Setup
# ----------------------

LINSEN_CONFIG_FOLDER_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'linsen')
CONFIGURATION_FILES_FOLDER = os.path.join(LINSEN_CONFIG_FOLDER_PATH, 'ConfigurationFiles')

TEST_DATA_CONF = '''
root_project={test_src_folder}
output_test_set={nbdata_folder}/TestLinsen/TestSetDictionary/
path_token_file_test_set=
path_stopwords_test_set=
acronyms_file={config_folder}/Acronyms.txt
abbreviations_file={config_folder}/Abbreviations.txt
root_report={report_folder}
oracle_path=
extension_indexing_file={extensions}
target_file_path={target_filepath}
target_identifiers_list={identifiers_list_filepath}
'''
TEST_DATA_CONF_FILENAME = 'ConfigurationTestSet.properties'

LINSEN_DATA_CONF = '''
output_dictionary={config_folder}/Structure/{project_name}/Dictionary/
path_token_file_dictionary={config_folder}/Informatics.txt;{config_folder}/English.txt
root_project_dictionary={src_folder}
acronyms_file={config_folder}/Acronyms.txt
abbreviations_file={config_folder}/Abbreviations.txt
extension_indexing_file={extensions}
path_stopwords_dictionary={config_folder}/StopWord.txt
'''
LINSEN_DATA_CONF_FILENAME = 'ConfigurationDictionaryforSimilarSplit.properties'

LINSEN_MAIN_FOLDER = os.path.join(LINSEN_CONFIG_FOLDER_PATH, 'Linsen')
CLASSPATH_FOLDER = os.path.join(LINSEN_MAIN_FOLDER, 'jars')
CLASSPATH = ':'.join([os.path.join(CLASSPATH_FOLDER, match) for match in 
                     filter(lambda f: os.path.splitext(f)[1] == '.jar', 
                            os.listdir(CLASSPATH_FOLDER))])

LINSEN_MAIN_CLASS = 'NormalizeMain'
LINSEN_CMD = 'java -cp {cp}:{wd}: {jclass}'

# ---------------
# LINSEN Analyzer
# ---------------

class LINSENnormalizer:
    """
    This class handles the execution of the LINSEN Normalization Process.
    In particular the `run` method will be in charge of:
    - setting up all the configuration files and the execution environment
    - invoke the java command
    - get the results
    - purge all the report and created files in order to keep the execution
        environment ready for next executions.
    """
    
    def __init__(self, code_method, target_identifiers):
        """
        Parameter:
        ----------
        code_method: `source_code_analysis.models.CodeMethod` instance.
            The target code method to the processed.
            
        target_identifiers: list
            The list of target identifiers to be normalized.
        """
        sw_project = code_method.project  # get the SoftwareProject instance
        
        self._src_folder = os.path.join(settings.MEDIA_ROOT, sw_project.src_folder_path)
        if not self._src_folder.endswith(os.path.sep):
            self._src_folder += os.path.sep  # Append final slash - java SUCKS!!!!
        
        file_extensions = regexp_tokenize(sw_project.file_extensions, pattern='\w+')        
        self._extensions = ','.join(file_extensions)
        
        # The folder where "target" source files are going to be created 
        # (see test_src_folder in TEST_DATA_CONF)
        self._target_folderpath = os.path.join(self._src_folder, 'target_sourcefiles/')
        if not os.path.exists(self._target_folderpath):
            os.makedirs(self._target_folderpath)
        
        self._target_method = code_method
        self._identifiers_list = target_identifiers
        
        from collections import defaultdict
        self._normalization_map = defaultdict(list)  # This will store LINSEN Normalization results
        
    def _create_fake_target_file(self):
        """
        Create a fake source file containing the code and the comment 
        of the target method.
        
        Return:
        -------
            The path of the newly created fake file. This path
            will be set into the configuration file as for the `target_file_path`
            entry.
        """
        # Get file extension - the first one available.
        file_ext = self._extensions.split(',')[0]
        # The name of this fake file will correspond to the method `name`.
        filename = '{0}.{1}'.format(self._target_method.method_name.lower(), file_ext)
        
        filepath = os.path.join(self._target_folderpath, filename)
        with open(filepath, 'w') as target_file:
            target_file.write(self._target_method.comment)
            target_file.write('\n')
            target_file.write(self._target_method.code_fragment)
        return filepath  # return the path to the fake file.
    
    def _create_target_identifiers_list(self):
        """
        Create a file containing one target identifier per line.
        
        Returns:
        --------
            The path to the list file.
        """
        filename = 'target_identifiers.txt'
        filepath = os.path.join(self._src_folder, filename)
        with open(filepath, 'w') as target_ids_list:
            for sw_id in self._identifiers_list:
                target_ids_list.write('{0}\n'.format(sw_id))
        return filepath
    
    def _create_configuration_files(self, target_filepath, target_list_path, report_folderpath):
        """
        Create the Configuration Files necessary to setup and run the 
        LINSEN (java) program.
        
        Parameters:
        -----------
        target_filepath: string
            The filepath to the target "fake" file containing the code and the comment
            of the target method. This fake file is necessary to instruct LINSEN to 
            process the lexical information gathered exclusively from the **target method**.

        target_list_path: string
            The path to the file containing the list of code identifiers to be processed.
            
        report_folderpath: string
            Path to the folder containing report files (output of each execution)
            
        Returns:
        --------
            The path to the newly created configuration files.
        """
        
        # Creating TEST Data File
        test_data_conf_content = TEST_DATA_CONF.format(test_src_folder=self._target_folderpath,
                                                       nbdata_folder=LINSEN_CONFIG_FOLDER_PATH,
                                                       config_folder=CONFIGURATION_FILES_FOLDER,
                                                       report_folder=report_folderpath,
                                                       extensions=self._extensions,
                                                       target_filepath=target_filepath,
                                                       identifiers_list_filepath=target_list_path)
        
        test_data_conf_filepath = os.path.join(LINSEN_MAIN_FOLDER, TEST_DATA_CONF_FILENAME)
        with open(test_data_conf_filepath, 'w') as test_data_conf_file:
            test_data_conf_file.write(test_data_conf_content.strip())
            
        
        # Creating LINSEN execution Data File
        linsen_data_conf_content = LINSEN_DATA_CONF.format(src_folder=self._src_folder,
                                                           config_folder=CONFIGURATION_FILES_FOLDER,
                                                           extensions=self._extensions,
                                                           project_name=self._target_method.project.name.upper().strip())

        linsen_data_conf_filepath = os.path.join(LINSEN_MAIN_FOLDER, LINSEN_DATA_CONF_FILENAME)
        with open(linsen_data_conf_filepath, 'w') as linsen_data_conf_file:
            linsen_data_conf_file.write(linsen_data_conf_content.strip())
            
        return test_data_conf_filepath, linsen_data_conf_filepath
    
    def _run_LINSEN_process(self):
        """
        Invoke the Java code via a new process.
        """
        
        current_wd = os.path.abspath(os.path.curdir)
        os.chdir(LINSEN_MAIN_FOLDER)
        command = LINSEN_CMD.format(cp=CLASSPATH, wd=LINSEN_MAIN_FOLDER, jclass=LINSEN_MAIN_CLASS)
        with subprocess.Popen(args=shlex_split(command), stderr=PIPE) as proc:
            from pprint import pprint
            pprint(proc.stderr.read())
        os.chdir(current_wd)
        
    def _restore_context(self, target_fp, list_fp, tdc_fp, lc_fp, report_fp):
        """
        Restore the execution context by removing files and folder created
        throught the six processing steps.
        
        Parameters:
        -----------
        target_fp: string (path)
            Path to the fake source files created in Step 1
        list_fp: string (path)
            Path to the list of target identifiers created in Step 2
        tdc_fp: string (path)
            Path to the "test_data" configuration file (Step 4)
        lc_fp: string (path)
            Path to the "linsen" configuration file (Step 4)
        report_fp: string (path)
            Path to the reportfilepath created as output of the LINSEN execution.
        """
        
        os.remove(target_fp)  # Remove target (fake) file
        os.remove(list_fp)  # Remove target identifiers list
        os.remove(report_fp)  # Remove output report file
        os.remove(tdc_fp)  # Remove first configuration file
        os.remove(lc_fp)  # Remove second configuration file
        
        # Finally: Remove the (Lucene) index folder - to avoid issue in next computations
        # See TEST_DATA_CONF: {nbdata_folder}/TestLinsen/TestSetDictionary/
        output_folder = os.path.join(LINSEN_CONFIG_FOLDER_PATH, 'TestLinsen', 'TestSetDictionary')
        remove_list = [os.path.join(output_folder, f) for f in os.listdir(output_folder)]
        for path in remove_list:
            os.remove(path)
        os.removedirs(output_folder)
        
    def normalize(self):
        """
        Setup and Run the LINSEN Normalization algorithm.
        """
        
        # Step 1: Create the "fake" source file and get the filepath
        # ----------------------------------------------------------
        target_filepath = self._create_fake_target_file()
        
        # Step 2: Create the target identifiers list file and get the resulting path
        # --------------------------------------------------------------------------
        target_list_path = self._create_target_identifiers_list()
        
        # Step 3: Create (if it does not exist) the Output Report folder
        # --------------------------------------------------------------
        
        # -- Report filename will be `report_folder_path+'/'+method_name.lower()+'.txt'`
        report_folder_path = os.path.join(self._src_folder, 'LINSEN_report_files/')
        if not os.path.exists(report_folder_path):
            os.makedirs(report_folder_path)
        
        # Step 4: Create Configuration files
        # ----------------------------------
        test_data_conf_fp, linsen_conf_fp = self._create_configuration_files(target_filepath, 
                                                                             target_list_path,
                                                                             report_folder_path)
        # Step 5: Run LINSEN
        # ------------------
        self._run_LINSEN_process()
        
        # Step 6: Gather Normalization Results
        # ------------------------------------
        
        # -- Check if the report_file has been created
        report_filename = '{0}.txt'.format(self._target_method.method_name.lower())
        report_filepath = os.path.join(report_folder_path, report_filename)
        
        if not os.path.exists(report_filepath):
            raise RuntimeError('Report filepath {0} does not exists!'.format(report_filepath))
        
        # -- Fed the `self._normalization_map` attribute
        with open(report_filepath) as report_file:
            for line in report_file:
                line = line.strip().strip(',')  # remove last useless comma in line
                identifier, results = line.split(':=')
                self._normalization_map[identifier] = results.split(',')
                
        # Step 7: Restore the Context
        # ---------------------------
        self._restore_context(target_filepath, target_list_path, test_data_conf_fp, 
                              linsen_conf_fp, report_filepath)
        
    @property
    def normalization_map(self):
        return self._normalization_map
    

# ---------------
# Lexical Analyzer
# ---------------

# Path to the files containing the list of Java reserved words
JAVA_RESERVED_WORDS = os.path.join(CONFIGURATION_FILES_FOLDER, 'java_keywords.txt') 
ENGLISH_DICTIONARY = os.path.join(CONFIGURATION_FILES_FOLDER, 'English.txt')

class LexicalAnalyzer:
    """
    This class is responsible to apply the actual analysis process to the textual information 
    gathered from the `code` and the `comment` fields of the given `CodeMethod` instance.
    
    The analysis process encompasses the following steps:
    - (Tokens Extraction)
        - The textual data are chunked into tokens (thanks to `nltk`)
    - (Tokens Normalization)
        - Most common (english) stopwords are removed, as well as Java language reserved keywords;
        - Each non-english token is processed by the **LINSEN** algorithm;
        - Each token (a.k.a, *lexeme*) is turned into lowercase letters;
        - Resulting tokens are finally stemmed.
        
    Note: the same pipeline process applies to both code and comments.
    
    Once the process has completed, the Jaccard Coefficient is computed in order to 
    calculate the (lexical) overlap between the two "zones" (i.e., code and comments).
    
    Finally, analysis results may be saved in an `CodeLexicalInfo` instance and stored into the database
    (see method `save_results`).
    """
    
    def __init__(self, code_method):
        self._code_method = code_method
        
        # Create Analysis Data Structures (stopwords and trie)
        self._java_keywords = self._fetch_java_keywords_from_file()
        self._english_dictionary = self._create_prefix_trie()
        self._stopwords = stopwords.words('english')
        
        # CodeLexicalInfo instance fields will store all the analysis results
        self._code_lexical_info_instance = CodeLexiconInfo()
        self._code_lexical_info_instance.reference_method = self._code_method

    def _fetch_java_keywords_from_file(self):
        """
        Fetches all the keywords in the `JAVA_RESERVED_WORDS`
        file and stores them into a Python set.
        """
        keywords_set = set()
        with open(JAVA_RESERVED_WORDS) as keywords_list:
            for kword in keywords_list:
                kword = kword.strip().lower()
                keywords_set.add(kword)
        return keywords_set
    
    def _create_prefix_trie(self):
        """
        Fetches all the keywords in the `JAVA_RESERVED_WORDS`
        file and stores them into a SortedStringTrie (from `pytrie`).
        """
        english_dictionary_word_set = set()
        with open(ENGLISH_DICTIONARY) as dictionary:
            for word in dictionary:
                word = word.strip().lower()
                english_dictionary_word_set.add(word)
        english_dictionary_word_set = sorted(english_dictionary_word_set)
        english_dictionary_word_set = [(word, i) for i, word in enumerate(english_dictionary_word_set)]
        english_dictionary = trie(english_dictionary_word_set)
        return english_dictionary

    def _tokenize_text(self, text):
        """
        Tokenize the input text into a list of words.
        
        Parameter:
        ----------
        text: string
            The text to tokenize.
            
        Returns:
        --------
            A list of words, resulting from the `nltk.regexp_tokenize` invocation.
        """
        return regexp_tokenize(text, pattern='\w+')  # Tokenize all words.
    
    def _turn_words_to_lowercase(self, words):
        """
        Transform the input words into lowercase.
        
        Parameter:
        ----------
        words: list
            The list containing all the words to turn into lowercase
        
        Return:
        -------
            A list containing the words transformed.
        """
        return [w.lower() for w in words]
    
    def _filter_stopwords(self, words, language_keywords=True):
        """
        Filters stopwords from the given word list.
        
        Parameters:
        -----------
        words: list
            The list of words to be checked
        language_keywords: bool, True by default.
            If True (default), language reserverd words (so far, only Java is supported)
            will be checked and removed as well.
            If False, only English stopwords will be removed from the input list.
            
        Return:
            A filter object (iterable in Py3)
        """
        
        # NOTE:
        # -----
        # Before checking if a given word `w` is a stopword, it is turned into lowercase
        # letters. This is because in the aforementioned pipeline, lowercase transformations
        # will be applied only after the execution of the LINSEN normalization algorithm, as 
        # the use of different cases in identifiers represents a valuable info for the 
        # splitting algorithm.
        if not language_keywords:
            return list(filter(lambda w: w.lower() not in self._stopwords, words))
        return list(filter(lambda w: w.lower() not in self._stopwords and \
                                w.lower() not in self._java_keywords, words))
        
    def _normalize_identifiers(self, words_in_code, words_in_comment, log=False):
        """
        This function is responsible to normalize identifiers by calling the LINSEN
        algorithm. In particular, only non-english tokens (i.e., words not being the 
        prefix of any entry in the english dictionary).
        
        NOTE: 
        Two iterables are passed to this function in order to be able to invoke
        LINSEN only once for each method.
        This is because LINSEN is going to parse and analyse identifiers appearing in a target 
        file and checked against a list of target identifiers to process.
        This target file is going to be a "fake" file created ad-hoc containing only the code
        (and the comment) of the target `CodeMethod` instance.
        See `LINSENnormalizer` for further details.
        
        Parameter:
        ----------
        words_in_code: filter object (iterable)
            The iterable containing all the words to be checked gathered from the `code` field
            of the target `CodeMethod` instance.
            This iterable is (usually) expected to the the output of the stopwords
            filtering invocation.
            
        words_in_comment: filter object (iterable)
            The iterable containing all the words to be checked gathered from the `comment` field
            of the target `CodeMethod` instance.
            This iterable is (usually) expected to the the output of the stopwords
            filtering invocation.
            
        Returns:
        --------
            Two iterables containing all the resulting words after the normalization with
            LINSEN has been applied. In particular, english words remain unchanged, while
            normalized identifiers will be replaced by LINSEN results.
        """
        
        # Step 1: Analyze words in Comment
        # --------------------------------
        comment_non_eng_identifiers = list()
        for word in words_in_comment:
            if word.lower() not in self._english_dictionary:
                comment_non_eng_identifiers.append(word)
        # -- Store this information into the `CodeLexiconInfo` instance
        self._code_lexical_info_instance.non_eng_tokens_comments = ','.join(comment_non_eng_identifiers)
        
        # Step 2: Analyze words in Source Code
        # ------------------------------------
        code_non_eng_identifiers = list()
        for word in words_in_code:
            if word.lower() not in self._english_dictionary:
                code_non_eng_identifiers.append(word)
        # -- Store this information into the `CodeLexiconInfo` instance
        self._code_lexical_info_instance.non_eng_tokens_code = ','.join(code_non_eng_identifiers)
        
        # Step 3: Remove Duplicates
        # -------------------------
        comment_non_eng_identifiers = set(comment_non_eng_identifiers)
        code_non_eng_identifiers = set(code_non_eng_identifiers)
        target_identifiers = comment_non_eng_identifiers.union(code_non_eng_identifiers)
        
        # Step 4: LINSEN Normalizer
        # -------------------------
        normalizer = LINSENnormalizer(self._code_method, target_identifiers)
        normalizer.normalize()
        norm_map = normalizer.normalization_map
        
        if log:
            print('After LINSEN! Normalization Map: ', norm_map)
        
        # Step 5: Gather Results
        # ----------------------
        
        def normalized_words(words, normalization_map):
            """
            Generate a sequence of words by replacing terms in normalization_map
            with the corresponding normalization result.
            """
            for w in words:
                if w not in normalization_map:
                    yield w
                else:
                    for nw in normalization_map[w]:
                        yield nw
        
        words_in_code = normalized_words(words_in_code, norm_map)
        words_in_comment = normalized_words(words_in_comment, norm_map)
        return words_in_code, words_in_comment
    
    def _apply_stemming(self, words):
        """
        Applies English Stemmer to each word in the input list
        
        Parameter:
        ----------
        words: list
            The list of words to be stemmed.
            
        Returns:
        --------
            The list of stemmed words
        """
        stemmer = EnglishStemmer()
        return [stemmer.stem(w) for w in words]
    
    def _jaccard_coefficient(self, comment_lexemes, code_lexemes):
        """
        Calculate the `Jaccard Coefficient` amont resulting Code and 
        Comments
        """
        
        comment_set = set(comment_lexemes)
        code_set = set(code_lexemes)
        
        jaccard_coeff = len(comment_set.intersection(code_set)) / len(comment_set.union(code_set))
        self._code_lexical_info_instance.jaccard_coeff = jaccard_coeff
        
    def analyse_textual_information(self, save_results=True, show_log=False):
        """
        This method actually implements the analysis pipeline processing.    
        
        Parameter:
        ----------
        save_results: bool, default=True
            Decide whether or not analysis results should be saved and stored into the DB
        """
        
        # Step 1: Tokenization
        # --------------------
        comment_tokens = self._tokenize_text(self._code_method.comment)
        code_tokens = self._tokenize_text(self._code_method.code_fragment)
        
        if show_log:
            print('Step 1: Tokenization')
            print([t for t in comment_tokens])
            print([t for t in code_tokens])
        
        # Step 2: Stopword Filtering
        # ---------------------------
        comment_tokens = self._filter_stopwords(comment_tokens)
        code_tokens = self._filter_stopwords(code_tokens)
        
        if show_log:
            print('Step 2: Stopword Filtering')
            print([t for t in comment_tokens])
            print([t for t in code_tokens])
        
        # Step 3: Identifier Normalization (LINSEN)
        # -----------------------------------------
        code_tokens, comment_tokens = self._normalize_identifiers(code_tokens, comment_tokens, log=show_log)
        
        code_tokens = list(code_tokens)
        comment_tokens = list(comment_tokens)
            
        if show_log:
            print('Step 3: Normalization')
            print([t for t in comment_tokens])
            print([t for t in code_tokens])
        
        # Step 4: Lowercase
        # -----------------
        comment_tokens = self._turn_words_to_lowercase(comment_tokens)
        code_tokens = self._turn_words_to_lowercase(code_tokens)

        if show_log:
            print('Step 4:')
            print([t for t in comment_tokens])
            print([t for t in code_tokens])
        
        # Step 5: Stemming
        # ----------------
        comment_tokens = self._apply_stemming(comment_tokens)
        code_tokens = self._apply_stemming(code_tokens)

        if show_log:
            print('Step 5:')
            print([t for t in comment_tokens])
            print([t for t in code_tokens])
        
        # End of Normalization Operations
        # -------------------------------
        
        # -- Update the `CodeLexiconInfo` instance
        self._code_lexical_info_instance.normalized_comment = ' '.join(comment_tokens)
        self._code_lexical_info_instance.normalized_code = ' '.join(code_tokens)
        
        # Step 6: Jaccard Index
        # ---------------------
        self._jaccard_coefficient(comment_tokens, code_tokens)
        
        if save_results:
            self.save_results()
        
    @property
    def code_lexical_info(self):
        return self._code_lexical_info_instance
    
    def save_results(self):
        """
        Save the information stored into `self._code_lexical_info_instance`
        attribute into the database.
        """
        self._code_lexical_info_instance.save()
        
