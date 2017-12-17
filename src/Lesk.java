/**
 * Implement the Lesk algorithm for Word Sense Disambiguation (WSD)
 */
import java.util.*;
import java.io.*;
import javafx.util.Pair;

import edu.mit.jwi.*;
import edu.mit.jwi.Dictionary;
import edu.mit.jwi.item.*; 

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;

public class Lesk {

	//keynote: testCorpus, ambiguousLocations and groundTruths = arrayList


	/** 
	 * Each entry is a sentence where there is at least a word to be disambiguate.
	 * E.g.,
	 * 		testCorpus.get(0) is Sentence object representing
	 * 			"It is a full scale, small, but efficient house that can become a year' round retreat complete in every detail."
	 **/
	private ArrayList<Sentence> testCorpus = new ArrayList<Sentence>();
	
	/** Each entry is a list of locations (integers) where a word needs to be disambiguate.
	 * The index here is in accordance to testCorpus.
	 * E.g.,
	 * 		ambiguousLocations.get(0) is a list [13]
	 * 		ambiguousLocations.get(1) is a list [10, 28]
	 **/
	private ArrayList<ArrayList<Integer> > ambiguousLocations = new ArrayList<ArrayList<Integer> >();
	
	/**
	 * Each entry is a list of pairs, where each pair is the lemma and POS tag of an ambiguous word.
	 * E.g.,
	 * 		ambiguousWords.get(0) is [(become, VERB)]
	 * 		ambiguousWords.get(1) is [(take, VERB), (apply, VERB)]
	 */
	private ArrayList<ArrayList<Pair<String, String> > > ambiguousWords = new ArrayList<ArrayList<Pair<String, String> > > (); 
	
	/**
	 * Each entry is a list of maps, each of which maps from a sense key to similarity(context, signature)
	 * E.g.,
	 * 		predictions.get(1) = [{take%2:30:01:: -> 0.9, take%2:38:09:: -> 0.1}, {apply%2:40:00:: -> 0.1}]
	 */
	private ArrayList<ArrayList<HashMap<String, Double> > > predictions = new ArrayList<ArrayList<HashMap<String, Double> > >();
	
	/**
	 * Each entry is a list of ground truth senses for the ambiguous locations.
	 * Each String object can contain multiple synset ids, separated by comma.
	 * E.g.,
	 * 		groundTruths.get(0) is a list of strings ["become%2:30:00::,become%2:42:01::"]
	 * 		groundTruths.get(1) is a list of strings ["take%2:30:01::,take%2:38:09::,take%2:38:10::,take%2:38:11::,take%2:42:10::", "apply%2:40:00::"]
	 */
	private ArrayList<ArrayList<String> > groundTruths = new ArrayList<ArrayList<String> >();
	
	/* This section contains the NLP tools */
	
	private Set<String> POS = new HashSet<String>(Arrays.asList("ADJECTIVE", "ADVERB", "NOUN", "VERB"));
	
	private IDictionary wordnetdict;
	
	private StanfordCoreNLP pipeline;

	private Set<String> stopwords;
	
	public Lesk() {

		/**
		 * The constructor initializes any WordNet/NLP tools and reads the stopwords.
		 */

		//read the stopwords from /data/stopwords.txt

		stopwords = new HashSet<String>();

		String fileName = "/data/stopwords.txt";

		String line = null;

		try {
			// FileReader reads text files in the default encoding.
			FileReader fileReader =
					new FileReader(fileName);

			// Always wrap FileReader in BufferedReader.
			BufferedReader bufferedReader =
					new BufferedReader(fileReader);

			while((line = bufferedReader.readLine()) != null) {
//				System.out.println(line);
				stopwords.add(line);
			}

			// Always close files.
			bufferedReader.close();
		}
		catch(FileNotFoundException ex) {
			System.out.println(
					"Unable to open file '" +
							fileName + "'");
		}
		catch(IOException ex) {
			System.out.println(
					"Error reading file '"
							+ fileName + "'");
			// Or we could just do this:
			// ex.printStackTrace();
		}


		//initializes any WordNet/NLP tools
		//ref:https://stanfordnlp.github.io/CoreNLP/api.html

		// creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
//		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		pipeline = new StanfordCoreNLP(props);  //globe private verib

		// read some text in the text variable
//		String text = "...";

		// create an empty Annotation just with the given text
//		Annotation document = new Annotation(text);

		// run all Annotators on this text
//		pipeline.annotate(document);

	}
	
	/**
	 * Convert a pos tag in the input file to a POS tag that WordNet can recognize (JWI needs this).
	 * We only handle adjectives, adverbs, nouns and verbs.
	 * @param pos: a POS tag from an input file.
	 * @return JWI POS tag.
	 */
	private String toJwiPOS(String pos) {
		if (pos.equals("ADJ")) {
			return "ADJECTIVE";
		} else if (pos.equals("ADV")) {
			return "ADVERB";
		} else if (pos.equals("NOUN") || pos.equals("VERB")) {
			return pos;
		} else {
			return null;
		}
	}

	/**
	 * This function fills up testCorpus, ambiguousLocations and groundTruths lists
	 * @param filename
	 * example
	 * #2 further ADV further%4:02:02::,further%4:02:03::
	 */
	public void readTestData(String filename) throws Exception {
		String line = null;

		try {
			// FileReader reads text files in the default encoding.
			FileReader fileReader =
					new FileReader(filename);

			// Always wrap FileReader in BufferedReader.
			BufferedReader bufferedReader =
					new BufferedReader(fileReader);

			// read some text in the text variable
			String text = "...";

//			// create an empty Annotation just with the given text
//			Annotation document = new Annotation(text);

//			// run all Annotators on this text
//			pipeline.annotate(document);

//			int line_counter = 0;

			//Interpreting the output to testCorpus, ambiguousLocations and groundTruths lists
			while((line = bufferedReader.readLine()) != null) {

				/**
				 * read the first line
				 * example :The Fulton County Grand Jury said Friday an investigation of Atlanta 's recent primary election produced " no evidence " that any irregularities took place .
				 */
//				if(line_counter == 0){

					// create an empty Annotation just with the given text
					Annotation document = new Annotation(line);
					// run all Annotators on this text
					pipeline.annotate(document);

					// these are all the sentences in this document
					// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
					List<CoreMap> sentences = document.get(SentencesAnnotation.class);

					for(CoreMap sentence: sentences) {
						//only one line of sentence

						// traversing the words in the current sentence
						// a CoreLabel is a CoreMap with additional token-specific methods
						Sentence all_word = new Sentence();
						for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
							// this is the text of the token
							String w = token.get(TextAnnotation.class);
							Word word = new Word(w);
							all_word.addWord(word);

							// this is the POS tag of the token
//						String pos = token.get(PartOfSpeechAnnotation.class);
//						// this is the NER label of the token
//						String ne = token.get(NamedEntityTagAnnotation.class);
						}
						testCorpus.add(all_word);
					}
//				}

				/**
				 * read the the ambiguous word number
				 */
				int number = Integer.parseInt(bufferedReader.readLine());

                /**
                 * read the the ambiguous word list [location, targets, groundTruth]
                 */

                ArrayList<Integer> locations = new ArrayList<>();
                ArrayList<Pair<String, String>> targets = new ArrayList<>();
                ArrayList<String> ground_truth = new ArrayList<>();

				for (int i = 0; i < number; ++i){
				    String read_ambiguous = bufferedReader.readLine();

                    String[] read_in = read_ambiguous.split(" ");
                    // read_in[0] = location
                    // read_in[1] + read_in[2] = targets <word, POS>
                    // read_in[3] = ground Truth



                    locations.add(Integer.parseInt(read_in[0]));
                    ground_truth.add(read_in[3]);

                    targets.add(new Pair<>(read_in[1], read_in[2]));
                }

                /**
                 * ArrayList<ArrayList<Integer> > ambiguousLocations
                 * ArrayList<ArrayList<Pair<String, String> > > ambiguousWords
                 * ArrayList<ArrayList<String> > groundTruths
                 */

                ambiguousLocations.add(locations);
                ambiguousWords.add(targets);
                groundTruths.add(ground_truth);
//				}
//				line_counter++;
			}
			// Always close files.
			bufferedReader.close();
		}
		catch(FileNotFoundException ex) {
			System.out.println(
					"Unable to open file '" +
							fileName + "'");
		}
		catch(IOException ex) {
			System.out.println(
					"Error reading file '"
							+ fileName + "'");
			// Or we could just do this:
			// ex.printStackTrace();
		}
	}
	
	/**
	 * Create signatures of the senses of a pos-tagged word.
	 * 
	 * 1. use lemma and pos to look up IIndexWord using Dictionary.getIndexWord()
	 * 2. use IIndexWord.getWordIDs() to find a list of word ids pertaining to this (lemma, pos) combination.
	 * 3. Each word id identifies a sense/synset in WordNet: use Dictionary's getWord() to find IWord
	 * 4. Use the getSynset() api of IWord to find ISynset
	 *    Use the getSenseKey() api of IWord to find ISenseKey (such as charge%1:04:00::)
	 * 5. Use the getGloss() api of the ISynset interface to get the gloss String
	 * 6. Use the Dictionary.getSenseEntry(ISenseKey).getTagCount() to find the frequencies of the synset.d
	 * 
	 * @param args
	 * lemma: word form to be disambiguated
	 * pos_name: POS tag of the wordform, must be in {ADJECTIVE, ADVERB, NOUN, VERB}.
	 * 
	 */
	private Map<String, Pair<String, Integer> > getSignatures(String lemma, String pos_name) {
	}
	
	/**
	 * Create a bag-of-words representation of a document (a sentence/phrase/paragraph/etc.)
	 * @param str: input string
	 * @return a list of strings (words, punctuation, etc.)
	 */
	private ArrayList<String> str2bow(String str) {
	}
	
	/**
	 * compute similarity between two bags-of-words.
	 * @param bag1 first bag of words
	 * @param bag2 second bag of words
	 * @param sim_opt COSINE or JACCARD similarity
	 * @return similarity score
	 */
	private double similarity(ArrayList<String> bag1, ArrayList<String> bag2, String sim_opt) {
	}
	
	/**
	 * This is the WSD function that prediction what senses are more likely.
	 * @param context_option: one of {ALL_WORDS, ALL_WORDS_R, WINDOW, POS}
	 * @param window_size: an odd positive integer > 1
	 * @param sim_option: one of {COSINE, JACCARD}
	 */
	public void predict(String context_option, int window_size, String sim_option) {
	}
	

	/**
	 * Multiple senses are concatenated using comma ",". Separate them out.
	 * @param senses
	 * @return
	 */
	private ArrayList<String> parseSenseKeys(String senseStr) {
		ArrayList<String> senses = new ArrayList<String>();
		String[] items = senseStr.split(",");
		for (String item : items) {
			senses.add(item);
		}
		return senses;
	}
	
	/**
	 * Precision/Recall/F1-score at top K positions
	 * @param groundTruths: a list of sense id strings, such as [become%2:30:00::, become%2:42:01::]
	 * @param predictions: a map from sense id strings to the predicted similarity
	 * @param K
	 * @return a list of [top K precision, top K recall, top K F1]
	 */
	private ArrayList<Double> evaluate(ArrayList<String> groundTruths, HashMap<String, Double> predictions, int K) {
	}
	
	/**
	 * Test the prediction performance on all test sentences
	 * @param K Top-K precision/recall/f1
	 */
	public ArrayList<Double> evaluate(int K) {
	}

	/**
	 * @param args[0] file name of a test corpus
	 */
	public static void main(String[] args) {
		Lesk model = new Lesk();
		try {
			model.readTestData(args[0]);
		} catch (Exception e) {
			System.out.println(args[0]);
			e.printStackTrace();
		}
		String context_opt = "ALL_WORDS";
		int window_size = 3;
		String sim_opt = "JACCARD";
		
		model.predict(context_opt, window_size, sim_opt);
		
		ArrayList<Double> res = model.evaluate(1);
		System.out.print(args[0]);
		System.out.print("\t");
		System.out.print(res.get(0));
		System.out.print("\t");
		System.out.print(res.get(1));
		System.out.print("\t");
		System.out.println(res.get(2));
	}
}
