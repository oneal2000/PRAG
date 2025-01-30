# Prompt Design for Our Work

This repository contains all the prompts involved in our work, categorized and explained for better understanding. The prompts are organized into the following sections:

- **Prompt for Document Augmentation (Section 3.2.1):**
  Specific prompts for augmenting documents, including tasks like rewriting and generating question-answer pairs.



- **Prompt for Experimental Datasets:**
  Specific prompts for the datasets used in our experiments, including 2WikiMultihopQA, HotpotQA, PopQA, and ComplexWebQuestions.





## Prompt for Document Augmentation

### Document Rewriting
Details of the prompts used to rewrite or transform documents for augmentation purposes.

```plain
Rewrite the following passage. While keeping the entities, proper nouns, and key details such as names, locations, and terminology intact, create a new version of the text that expresses the same ideas in a different way. Make sure the revised passage is distinct from the original one, but preserves the core meaning and relevant information.
{passage}
```

### QA Generation
Explanation of the prompts used to generate question-answer pairs for document augmentation.

```plain
I will provide a passage of text, and you need to generate three different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.
You need to generate the question and answer in the following format:
[
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
        "full_answer": "The capital of France is Paris."
    }, 
]
This list should have at least three elements. You only need to output this list in the above format.
Passage:
{passage}
```

## Prompt for Experimental Datasets

Following prior works such as FLARE, DRAGIN, SEAKR, and DRAD, we adopt the few-shot prompting template introduced in IR-CoT, which includes exemplars to guide the reasoning process of LLMs. However, IR-CoT only provides prompt templates with few-shot examples for 2WikiMultihopQA and HotpotQA. Consequently, to be consistent with previous works, we employ the same few-shot prompting strategy for these two datasets.

In contrast, IR-CoT does not provide predefined few-shot templates for PopQA and ComplexWebQuestions. We use zero-shot prompting for these datasets to ensure a fair comparison without introducing dataset-specific prompt engineering.

**Furthermore, we ensured that P-RAG and all the baselines share the same prompt template within each dataset, ensuring a fair evaluation.**



### 2WikiMultihopQA
Description of the prompts designed specifically for 2WikiMultihopQA dataset tasks.

```plain
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.
Here are some examples about how to answer the questions.
Question: When did the director of film Hypocrite (Film) die?
Answer: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is 19 June 2013.

Question: Are both Kurram Garhi and Trojkrsti located in the same country?
Answer: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is no.

Question: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
Answer: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality. So the answer is no.

Question: Who is Boraqchin (Wife Of Ögedei)'s father-in-law?
Answer: Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan. So the answer is Genghis Khan.

Question: Who was born first out of Martin Hodge and Ivania Martinich?
Answer: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is Martin Hodge.

Question: When did the director of film Laughter In Hell die?
Answer: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is August 25, 1963.

Question: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
Answer: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two. So the answer is Twenty Plus Two.

Question: Who is the grandchild of Krishna Shah (Nepalese Royal)?
Answer: Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah. So the answer is Prithvipati Shah.

Here are some reference.
Passage 1: Polish-Russian War (film) Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska. The film's events take place over several days and they are set in the present time in a large Polish city. The main character is a bandit, a Polish dres (a Polish chav) called "Strong" (Borys Szyc), who does not work or study, and who frequently gets into conflict with the law and is in love with Magda (Roma Gąsiorowska). The relationship is not going well. "Strong" is insanely jealous of
Passage 2: film was shot between May 6 and 18 June 2008 in locations of Warsaw, Wejherowo, Sopot and Gdynia outskirts. The film premiered on May 22, 2009. The budget of Polish-Russian War amounted to approx. 4 million zlotys. The creators of the music for the film are Jan Komar, Filip Kuncewicz, Liroy, Mateusz Łapot and Jarosław Karczmarczyk. The soundtrack also included the following songs: Polish-Russian War (film) Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by Xawery Żuławski based on the novel Polish-Russian War under the white-red flag by Dorota Masłowska. The film's events take place over several days
Passage 3: she has been in contact with Olga's dead mother and she asks the Attorney to participate in a seance. Body (2015 Polish film) Body () is a 2015 Polish drama film directed by Małgorzata Szumowska. It was screened in the main competition section of the 65th Berlin International Film Festival where Szumowska won the Silver Bear for Best Director. The film also received the Golden Lions Award at the 2015 Gdynia Film Festival and the People's Choice Award at the 2016 European Film Awards. Olga who struggles with anorexia is sent to psychiatric hospital by the Attorney, where she is


Let's think step by step. Answer the questions in the same format as above.
Question: Who is the mother of the director of film Polish-Russian War (Film)?<|im_end|>
<|im_start|>assistant
Answer: 
```

### HotpotQA
Details about the prompts tailored for the HotpotQA dataset.

```plain
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.
Here are some examples about how to answer the questions.
Question: Jeremy Theobald and Christopher Nolan share what profession?
Answer: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is producer.

Question: What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?
Answer: Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau. So the answer is The Phantom Hour..

Question: How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?
Answer: The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is 20.

Question: Were Lonny and Allure both founded in the 1990s?
Answer: Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s. So the answer is no.

Question: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
Answer: The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn. So the answer is Scott Glenn.

Question: What was the 2014 population of the city where Lake Wales Medical Center is located?
Answer: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is 15,140.

Question: Who was born first? Jan de Bont or Raoul Walsh?
Answer: Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is Raoul Walsh.

Question: In what country was Lost Gravity manufactured?
Answer: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is Germany.

Question: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
Answer: The debut album of the band "Hot Hot Heat" was "Make Up the Breakdown". The debut album of the band "The Operation M.D." was "We Have an Emergency". So the answer is The Operation M.D..

Question: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
Answer: The singer of "A Girl Like Me" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award. So the answer is one.

Question: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
Answer: The actor that stars as Joe Proctor on the series "Power" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante. So the answer is Assante.

Here are some reference.
Passage 1: and has two children. Critical, public and commercial reception to films Derrickson has directed as of November 13, 2016. Scott Derrickson Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. Derrickson is best known for directing numerous horror films, such as "The Exorcism of Emily Rose" (2005), "Sinister" (2012), and "Deliver Us From Evil" (2014), as well as the Marvel Cinematic Universe superhero film "Doctor Strange" (2016). Derrickson grew up in Denver, Colorado. He graduated from Biola University with a B.A. in Humanities, with an emphasis on literature and philosophy,
Passage 2: Scott Derrickson Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. Derrickson is best known for directing numerous horror films, such as "The Exorcism of Emily Rose" (2005), "Sinister" (2012), and "Deliver Us From Evil" (2014), as well as the Marvel Cinematic Universe superhero film "Doctor Strange" (2016). Derrickson grew up in Denver, Colorado. He graduated from Biola University with a B.A. in Humanities, with an emphasis on literature and philosophy, and a B.A. in communications, with an emphasis on film, and a minor in theological studies. He earned his
Passage 3: M.A. in film production from USC School of Cinematic Arts. Derrickson co-wrote and directed the film "The Exorcism of Emily Rose" which was loosely based on a true story about Anneliese Michel. The film won the 2005 Saturn Award for Best Horror or Thriller Film and in 2006 was named in the Chicago Film Critics Association's list of the "Top 100 Scariest Films Ever Made." Theatrical box office gross for "The Exorcism of Emily Rose" was over $144 million worldwide. That same year, Derrickson wrote "Land of Plenty" for director Wim Wenders, an independent drama starring Michelle Williams. Derrickson next


Let's think step by step. Answer the questions in the same format as above.
Question: Were Scott Derrickson and Ed Wood of the same nationality?<|im_end|>
<|im_start|>assistant
Answer: 
```

### PopQA
Explanation of the prompts created for PopQA dataset experiments.

```plain
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You should answer the question by referring to the knowledge provided below and integrating your own knowledge.
Passage 1: George Rankin Major General George James Rankin, (1 May 1887 – 28 December 1957) was an Australian soldier and politician. He served in both the House of Representatives and the Senate, representing the Country Party of Australia. Rankin was born at Bamawm, Victoria, the tenth child of Irish farmer James Rankin and Sarah, née Gallagher. He attended the local state school and became a farmer. In 1907, he joined the Militia, and was commissioned in the 9th Light Horse Regiment in 1909. He married Annie Isabella Oliver at Rochester, Victoria on 7 July 1912. In 1914, he was appointed a
Passage 2: was buried, and was survived by his wife. George Rankin Major General George James Rankin, (1 May 1887 – 28 December 1957) was an Australian soldier and politician. He served in both the House of Representatives and the Senate, representing the Country Party of Australia. Rankin was born at Bamawm, Victoria, the tenth child of Irish farmer James Rankin and Sarah, née Gallagher. He attended the local state school and became a farmer. In 1907, he joined the Militia, and was commissioned in the 9th Light Horse Regiment in 1909. He married Annie Isabella Oliver at Rochester, Victoria on 7
Passage 3: George Claus Rankin Sir George Claus Rankin PC (12 August 1877 – 8 April 1946) was a British judge in India. Rankin was born in Lamington, Lanarkshire, the son of Rev. Robert Rankin. He was educated at the University of Edinburgh and Trinity College, Cambridge. He as admitted at Lincoln's Inn and called to the bar in 1904. He served in the First World War with the Royal Garrison Artillery. He went to India in 1918 and served first as a puisne judge of the High Court of Calcutta, and then as Chief Justice, from 1926 to 1934. While in


Question: What is George Rankin's occupation?<|im_end|>
<|im_start|>assistant
The answer is 
```

### ComplexWebQuestions
Information on the prompts designed for the ComplexWebQuestions dataset.

```plain
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
You should answer the question by referring to the knowledge provided below and integrating your own knowledge.
Passage 1: party such as the United Nations. The 1974 Interim Constitution Act was passed by the 48-member Azad Jammu and Kashmir unicameral assembly. Azad Jammu and Kashmir (AJK) is a self-governing state under Pakistani control, but under Pakistan's constitution the state is informally part of the country. Pakistan is administering the region as a self-governing territory rather than incorporating it in the federation since the UN-mandated ceasefire. Azad Kashmir has its own elected President, Prime Minister, Legislative Assembly, High Court, with Azam Khan as its present chief justice, and official flag. Azad Kashmir's financial matters, i.e., budget and tax affairs, are
Passage 2: and new research programmes have already been launched. However Institute of Geology is ranked 2nd in whole country based upon diverse field work and researches in Masters and Ph.D programmes. University of Azad Jammu and Kashmir The University of Azad Jammu and Kashmir is a university at Muzaffarabad, Azad Jammu and Kashmir, Pakistan. It was established in 1980, and is currently ranked at No.14 in HEC ranking of General category universities in Pakistan. The University of Azad Jammu and Kashmir is a multi-campus, multi-discipline university. The University of Azad Jammu and Kashmir has been making steady progress in both academic
Passage 3: organisations as "Pakistan administered Kashmir". Azad Kashmir is one-sixth of the size of Gilgit-Baltistan. The territory also borders Pakistan's Punjab province to the south and Khyber Pakhtunkhwa province to the west. To the east, Azad Kashmir is separated from the state of Jammu and Kashmir by the Line of Control, the "de facto" border between India and Pakistan. Azad Kashmir has a total area of , and a total population of 4,045,366 as per the 2017 Census. The territory has a parliamentary form of government modeled after the Westminster system, with its capital located at Muzaffarabad. The President is the


Question: Who was the president in 1980 of the country that has Azad Kashmir?<|im_end|>
<|im_start|>assistant
The answer is 
```