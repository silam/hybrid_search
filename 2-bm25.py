from pathlib import Path
import pandas as pd
import bm25s



DATA_DIR = Path(__file__).parent / "data" / "fiqa" / "fiqa"
INDEX_DIR = DATA_DIR / "indexes" / "bm25"

"""
	_id	title	text
0	3		I'm not saying I don't like the idea of on-the...
1	31		So nothing preventing false ratings besides ad...
2	56		You can never use a health FSA for individual ...
3	59		Samsung created the LCD and other flat screen ...
4	63		Here are the SEC requirements: The federal s
"""
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")


"""
['3',
 '31',
 '56',
 '59',
 '63',
 '100',
 '108',
 '112',
 '115',
 '120',
 '125',
 '130',
 '135',
 '140',
 '145',
 '150',
 '155',
 '160',
 '165',
 '170',
 '175',
 '180',
 '185',
 '190',
 '195',
 '200',
"""
corpus_doc_ids = corpus['_id'].tolist()


"""
["I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job - they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything.",
 "So nothing preventing false ratings besides additional scrutiny from the market/investors, but there are some newer controls in place to prevent institutions from using them. Under the DFA banks can no longer solely rely on credit ratings as due diligence to buy a financial instrument, so that's a plus. The intent being that if financial institutions do their own leg work then *maybe* they'll figure out that a certain CDO is garbage or not.  Edit: lead in",
 "You can never use a health FSA for individual health insurance premiums.  Moreover, FSA plan sponsors can limit what they are will to reimburse.  While you can't use a health FSA for premiums, you could previously use a 125 cafeteria plan to pay premiums, but it had to be a separate election from the health FSA. However, under N. 2013-54, even using a cafeteria plan to pay for indivdiual premiums is effectively prohibited.",
 'Samsung created the LCD and other flat screen technology like OLED. a few years ago every flat screen came from Samsung factories and were reshelled. I think the 21 Hanns screen I am looking at now is Samsung and it is only a couple of years old. Samsung seem to be a good company.',
 'Here are the SEC requirements: The federal securities laws define the term accredited investor in   Rule 501 of Regulation D as: a bank, insurance company, registered investment company, business development company, or small business investment company; an employee benefit plan, within the meaning of the Employee Retirement Income Security Act, if a bank, insurance company, or   registered investment adviser makes the investment decisions, or if   the plan has total assets in excess of $5 million; a charitable organization, corporation, or partnership with assets exceeding $5 million; a director, executive officer, or general partner of the company selling the securities; a business in which all the equity owners are accredited investors; a natural person who has individual net worth, or joint net worth with the person’s spouse, that exceeds $1 million at the time of the   purchase, excluding the value of the primary residence of such person; a natural person with income exceeding $200,000 in each of the two most recent years or joint income with a spouse exceeding $300,000 for   those years and a reasonable expectation of the same income level in   the current year; or a trust with assets in excess of $5 million, not formed to acquire the securities offered, whose purchases a sophisticated person makes. No citizenship/residency requirements.',
 '"Only relevant to those with fantasy economy teams. Seriously, Rand\'s fictional works never translate well into reality because, no matter how hard people try, that ""fiction"" element just can\'t be ignored.  Test it yourself: Strip John Galt and his followers of everything they have which was created by or within the ""society"" they so revile, drop them in the desert -- and they\'ll all be dead of exposure and starvation in less than two weeks because they will be naked, without tools and without food.  The only reason the libertarians get away with pushing their tripe as a rational philosophy is because no one will point out what it is wrong with their thinking. Why? Well, for most of my lifetime, their ""philosophy"" was considered nuttery in line with the John Birchers and so why bother. It\'s only with the ascendency of these billionaire-funded politicians that this crap thinking has become acceptable, and even then, only to them."',
 "Futures contracts are a member of a larger class of financial assets called derivatives. Derivatives are called such because their payoffs depend on the price of other assets (financial or real). Other kinds of derivatives are call options, put options. Fixed income assets that mimic the behavior of derivatives are callable bonds, puttable bonds etc.  A futures contract is a contract that specifies the following: Just like with any other contract, there are two parties involved. One party commits to delivering the underlying asset to the other party on expiration date in exchange for the futures price. The other party commits to paying the futures price in exchange for the asset. There is no price that any of the two parties pay upfront to engage in the contract. The language used is so that the agent committing to receiving the delivery of the underlying asset is said to have bought the contract. The agent that commits to make the delivery is said to have sold the contract.  So answer your question, buying on June 1 a futures contract at the futures price of $100, with a maturity date on August 1 means you commit to paying $100 for the underlying asset on August 1. You don't have to pay anything upfront. Futures price is simply what the contract prescribes the underlying asset will exchange hands for.",
 'This month when you join Scentsy you get a free defuser with your kit!   This has never been done before.  You also get spring / summer and Fall / Winter testers plus all your kit items!    Be your own boss!  You choose what hours you work, when and where you work them.   Join my Scentsy family today!  [Amanda C. Robar Scentsy Business ](http://www.amandacrobar.scentsy.ca)',
 'Whenever you pay or withdraw some fund from your account, paypal takes approx 3% of the current currency value along with the fees. i.e. If you are paying/withdraw 100 unit of US Dollars to British pounds and if the current convertion rate is 1$=0.82GBP, then consider reducing 3% of the actual currency rate. So, the approximate magnitude will be 0.82*97% (100-3=97) = 0.7954. So, 1$=0.7954GBP. This formula will not give you 100% accurate value but will help of course. Captain',
 'So you asked him in 2010 how he was gong to compete with DVD rental distributors like Netflix (which is what Netflix primarily was at the time) and Lovefilm and you were surprised that he was he said they were going to continue to compete as a DVD rental distributor just like the mentioned competitors?',
 'In the US, I would say the risk is exactly the same.  If your accounts are withing the FDIC amount (currently $250,000) your balance is 100% covered in case of a failure. You are giving up a larger network of ATMs in some cases.  You are also perhaps giving up the number of branches you can visit, the hours the bank is open and maybe how well the website works.  The features might be less, but the protection for your deposits is the same.',
 "I don't know what the fee is, I just made up $5 as an example. It doesn't surprise me at all that Comcast is charging $20.  I don't really mind that I don't have on demand, it doesn't bother me. But after the disaster that CableCards were, there was no chance that tru2way was going to work if the industry had any say in it. They forced the failure of CableCards, and none of their reasons for opposing them went away.",
 '"At any given moment, one can tally the numbers used for NAV. It\'s math, and little more. The Market Cap, which as you understand is a result of share value. Share value (stock price) is what the market will pay today for the shares. It\'s not only based on NAV today, but on future expectations. And expectations aren\'t the same for each of us. Which is why there are always sellers for the buyers of a stock, and vice-versa.  From your question, we agree that NAV can be measured, it\'s the result of adding up things that are all known. (For now, let\'s ignore things such as ""goodwill."") Rarely is a stock price simply equal to the NAV divided by the number of shares. Often, it\'s quite higher. The simplest way to look at it is that the stock price not only reflects the NAV, but investors\' expectations looking into the future.  If you look for two companies with identical NAV per share but quite different share prices, you\'ll see that the companies differ in that one might be a high growth company, the other, a solid one but with a market that\'s not in such a growth mode."',
 'Chinese suppliers can quote their price in CNY rather than USD (as has been typical), and thus avoid the exchange risk from US dollar volatility- the CNY has been generally appreciating so committing to receive payments in US dollars when their costs are in CNY means they are typically on the losing end of the equation and they have to pad their prices a bit.  Canadian importers will have to buy RMB (typically with CAD) to pay for their orders and Canadian exporters can take payment in RMB if they wish, or set prices in CAD. By avoiding the US dollar middleman the transactions are made less risky and incur less costs. Japan did this many decades ago (they, too, used to price their products in USD).  This is important in transactions of large amounts, not so much for the tiny amounts associated with tourism. Two-way annual trade between China and Canada is in excess of $70bn.  Of course Forex trading may greatly exceed the actual amounts required for trade- the world Forex market is at least an order of magnitude greater than size of real international trade. All that trading in currency and financial instruments means more jobs on Bay Street and more money flowing into a very vital part of the Canadian economy.  Recent article from the (liberal) Toronto Star here.',
 '"I read the post.  I read what **they said** about it.   Yet, that part makes **no sense**.  They keep talking about ""only avilable to members of the Florida Bar"" but stop and think about how that doesn\'t make sense.  Are they in a *florida* court or in a *federal* court?  Or an IP/administrative setting?    Some things they wrote suggests they\'re in a *federal* court.  But others things suggest they\'re in an administrative court related to IP.  For example, they talk about needing an attorney who can represent them in the Trademark/IP courts.  That sounds like patent prosecution - but that\'s typically just about getting the patent/IP stuff in the first place - and not so much after-the-fact stuff.  But then again, I\'m not at all familiar with the area so I could be way off base.  The problem is: I shouldn\'t have to be speculating so much because their story sucked.  But patent litigation (for infringement for example) as far as I was aware, occurs in traditional district courts and so far as I am aware you don\'t need a special trademark/ip exam to sue on it.  By contrast, patent *prosecution*, which is getting a patent through the offices, requires that the attorney take the USPTO exam - to become a \'patent\' attorney.  [link](http://en.wikipedia.org/wiki/USPTO_registration_examination).  Federal courts, if that\'s where they are, typically use PACER ECF/CM (case management web portal).  Which, I\'ll have to assume is the system they are referring to.  https://www.pacer.gov/psco/cgi-bin/regform.pl  That\'s where anyone can fucking sign up for it.  Then they can look at dockets and all that jazz.  Maybe there\'s another system I\'m not familiar with OR another system that I can\'t discern from the way they wrote the story.  But so far.  More BS or just confusing or wrong information.  This is why they should have hired a lawyer."',
 "&gt; monopoly  &gt; names 3 giant companies who compete with each other in almost every industry possible  What happened to /economy :(. It used to be actual business minded people, now it's becoming another socialist echo chamber... Sad to see.",
 'What happens would depend on company culture.  At the very least they could pass it on to their manager.  Or maybe the company just pays support staff by number of answered emails in which case you get the quickest pre-canned answer they can provide.  But the net effect is they will continue to lose to alternatives because of this issue.  They have been given customer feedback.  And their culture is to ignore it.  The results are self inflicted.',
 "We write them off as cheap knock offs because that's what they are. They are no match when they come up against the superior technology of the US and her allies - they are only a threat when these countries use them against their own people to stifle dissent.",
 "My super fund and I would say many other funds give you one free switch of strategies per year.  Some suggest you should change from high growth option to a more balance option once you are say about 10 to 15 years from retirement, and then change to a more capital guaranteed option a few years from retirement. This is a more passive approach and has benefits as well as disadvantages. The benefit is that there is not much work involved, you just change your investment option based on your life stage, 2 to 3 times during your lifetime. This allows you to take more risk when you are young to aim for higher returns, take a balanced approach with moderate risk and returns during the middle part of your working life, and take less risk with lower returns (above inflation) during the latter part of your working life. A possible disadvantage of this strategy is you may be in the higher risk/ higher growth option during a market correction and then change to a more balanced option just when the market starts to pick up again. So your funds will be hit with large losses whilst the market is in retreat and just when things look to be getting better you change to a more balanced portfolio and miss out on the big gains. A second more active approach would be to track the market and change investment option as the market changes. One approach which shouldn't take much time is to track the index such as the ASX200 (if you investment option is mainly invested in the Australian stock market) with a 200 day Simple Moving Average (SMA). The concept is that if the index crosses above the 200 day SMA the market is bullish and if it crosses below it is bearish. See the chart below:  This strategy will work well when the market is trending up or down but not very well when the market is going sideways, as you will be changing from aggressive to balanced and back too often. Possibly a more appropriate option would be a combination of the two. Use the first passive approach to change investment option from aggressive to balanced to capital guaranteed with your life stages, however use the second active approach to time the change. For example, if you were say in your late 40s now and were looking to change from aggressive to balanced in the near future, you could wait until the ASX200 crosses below the 200 day SMA before making the change. This way you could capture the majority of the uptrend (which could go on for years) before changing from the high growth/aggressive option to the balanced option. If you where after more control over your superannuation assets another option open to you is to start a SMSF, however I would recommend having at least $300K to $400K in assets before starting a SMSF, or else the annual costs would be too high as a percentage of your total super assets.",
 '"&gt; You do realize that investors are protected from being sued right?  Not usually, no. If the investor is a partner in the company then they\'re just as responsible for the debts of their business as any other partner. If they\'re just a capital investor with no interest in the company business then sure - you can\'t sue them. But then my contract wouldn\'t be with them, would it?  &gt; So if the company you work for tanks, you can sue them, but they don\'t have any money so what do you expect to get?  The registered owners of the company can also be held liable for it\'s debts if it\'s a corporation. Or you can always just have them sign as guarantor for your back pay.  And don\'t say ""Nobody will ever do that!!"", because if your service is valuable enough then **YES THEY WILL**. I\'ve pushed for a personal liability guarantee from a company owner before and got it."',
 '"So you\'re basically saying that average market fluctuations have an affect on individual stocks, because individual stocks are often priced in relation to the growth of the market as a whole?  Also, what kinds of investments would be considered ""risk free"" in this nomenclature?"',
 "US government bonds are where money goes when the markets are turbulent and investors are fleeing from risk, and that applies even if the risk is a downgrade of the US credit rating, because there's simply nowhere else to put your money if you're in search of safety. Most AAA-rated governments have good credit ratings because they don't borrow much money (and most of them also have fairly small economies compared with the US), meaning that there's poor liquidity in their scarce bonds.",
 "Nonsense. In places where people have most kids, it doesn't matter how expensive baby care products are because most babies are born where they aren't used nearly as much, or at all. Sure, making them more expensive will help having less children in the first world, but that would change squat WRT the total number.",
 "Either way the govt is going to get to a breaking point and it's going to be shitty and miserable until they get there (and obviously when they get there it will get worse before it gets better). This bond trade just prolongs the shitty/miserableness even longer. Who knows how many more people will die from govt sponsored thugs and hunger because this trade pushed the breaking point out a few months/years?",
 "Yeah, I insti
"""

corpus_doc_texts = corpus['text'].tolist()

print(f"Index Directory: {INDEX_DIR}")
print(f"Number of Documents:       {len(corpus_doc_texts)} ")

# Tokenize the document texts using BM25's tokenizer with English stopwords
# The resulting `tokens` variable will be a list of tokenized documents, ready for BM25 indexing.   

"""
Tokenized(
  "ids": [
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
    1: [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, ...]
    2: [7, 8, 86, 87, 88, 89, 90, 88, 91, 92, ...]
    3: [115, 116, 117, 118, 119, 120, 121, 2, 122, 123, ...]
    4: [142, 143, 144, 145, 146, 147, 148, 149, 150, 151, ...]
    5: [137, 228, 212, 229, 230, 231, 232, 233, 234, 235, ...]
    6: [298, 299, 300, 301, 302, 70, 173, 303, 304, 304, ...]
    7: [359, 360, 7, 361, 362, 7, 273, 363, 364, 347, ...]
    8: [388, 7, 104, 389, 52, 390, 49, 347, 391, 392, ...]
    9: [41, 7, 421, 422, 423, 241, 424, 425, 426, 427, ...]
    ... (total 57638 docs)
  ],
  "vocab": [
    '': 80283
    '00': 3166
    '000': 206
    '0000': 15200
    '00000': 59299
    '0000000': 68778
    '00000000': 66734
    '00000000000': 17835
    '0000000000001': 71845
    '000000000001': 23084
    ... (total 80284 tokens)
  ],
"""
tokens = bm25s.tokenize(corpus_doc_texts, stopwords="en")


"""
[[0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  5,
  12,
  4,
  13,
  14,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
  22,
...
  1002,
  13,
  279,
  215],
"""
print(tokens.ids[:1])

"""
{"saying":0,"don":1,"like":2,"idea":3,"job":4,"training":5,"too":6,
"""
print(tokens.vocab.items())

retriever = bm25s.BM25()
#print(f"Retriever created with {len(retriever.tokens.ids)} documents.")
retriever.index(tokens)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(INDEX_DIR)
print(f"Index saved to {INDEX_DIR}")
(INDEX_DIR/"doc_ids.txt").write_text("\n".join(corpus_doc_ids))
print(f"Document IDs saved to {INDEX_DIR/'doc_ids.txt'}")


def search_bm25(query: str, k: int=10) -> list[tuple[str, float]]:

    # Tokenize the query and retrieve the top_k results from the BM25 index.

    """
    Tokenized(
  "ids": [
    0: [0, 1, 2, 3, 4, 5, 6]
  ],
  "vocab": [
    'day': 5
    'fund': 6
    'my': 3
    'park': 2
    'rainy': 4
    'should': 1
    'where': 0
  ],
)
    """
    query_tokens = bm25s.tokenize([query], stopwords="en")

    # Ensure the query tokens are in the correct format for retrieval.
    # Retrieve the top_k documents based on the query tokens.
    # `indices` contains the document indices, and `scores` contains the corresponding BM25 scores.
    # Combine the document IDs with their corresponding scores to form the final results list.


    """
    (array([[36224,  3157, 39987, 55681, 55690, 47916, 24501, 33316, 24910,
           763]]),
 array([[13.760674 ,  8.884806 ,  7.9142647,  7.830278 ,  7.5942163,
          7.1871395,  6.8798385,  6.7713947,  6.7225423,  6.572762 ]],
       dtype=float32))
    """

    indices, scores = retriever.retrieve(query_tokens, k=k)
    results = [(corpus_doc_ids[index_doc_id], float(scores[0][index])) for index, index_doc_id in enumerate(indices[0].tolist())]
    return results


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    k = 10

    """
    [('376148', 13.760674476623535),
 ('32833', 8.884805679321289),
 ('416189', 7.914264678955078),
 ('579848', 7.830277919769287),
 ('580025', 7.594216346740723),
 ('497993', 7.187139511108398),
 ('253614', 6.879838466644287),
 ('345428', 6.771394729614258),
 ('257921', 6.7225422859191895),
 ('8088', 6.5727620124816895)]
    """
    results = search_bm25(query, k=k)


    """
    Document ID: 376148, Score: 13.760674476623535
    Text: Bond aren't necessarily any safer than the stock market.  Ultimately, there is no such thing as a low risk mutual fund. You want something that will allow you get at your money relatively quickly. In other words, CDs (since you you can pick a definite time period for your money to be tied up), money market account or just a plain old savings account. Basically, you want to match inflation and have easy access to the money.  Any other returns on top of that are gravy, but don't fret too much about it. See also: Where can I park my rainy-day / emergency fund? Savings accounts don’t generate much interest. Where should I park my rainy-day / emergency fund?

    Document ID: 32833, Score: 8.884805679321289
    Text: In addition to the issues discussed in BrenBarn's answer, I think you need to consider your medium term saving needs and existing savings. In particular, do you have a sufficient rainy day fund, a fund you will spend if things go wrong? For example, if you are dependent on a vehicle that is not covered by a guarantee or service plan, you should have enough money saved for a couple of major repairs. Depending on how secure your job is, whether it carries sick leave and long term disability, and how easy or difficult it would be to find another job in the event e.g. of your employer going bankrupt due to a downturn in your industry, you should have months to years of minimal living expenses in your rainy day fund. If you don't have those things covered, you should urgently save as much as you can until they are covered. If you do, then the next savings priority is to put money by for retirement. Of course, if all goes well the rainy day fund will ultimately get folded into retirement, but it needs to exist now, in a form you can access quickly.

    Document ID: 416189, Score: 7.914264678955078
    Text: For me there are two issues.  So, what to do? You have the basics of a very strong position coming together. A good salary in a good city. I'd be patient and work on consolidating my position for another year to 18 months (including building a rainy day fund) and look to buy then.

    Document ID: 579848, Score: 7.830277919769287
    Text: You are only 33, with plenty of time to generate long-term returns. A correction is an opportunity for you to buy more at a lower price. My advice: invest about half in an index fund (S&P 500, for example) and half in a target fund (pick your retirement age), keep saving regularly, and in 30 years you'll be very happy you did. 5-10K a month is fine until all of the money is invested. (You said this is a retirement account, so invest the entire amount. Keep a rainy day fund in your non-retirement savings.) You can fiddle around with how you invest the money and I'm sure you'll get variations on my answer. My above suggestions may not be the absolute best option, but they are certainly not the worst option. Given that you are very risk-averse, keep in mind that you are in this for the long term and should be investing in something where you can safely ignore short-term downswings.

    Document ID: 580025, Score: 7.594216346740723
    Text: "I don't know Canada very well, but can offer some general points when considering where to park your emergency fund. Savings rates are currently low, but then so is inflation. Always bear in mind that inflation decreases the value of your money, so if you're getting 4% interest and inflation is 2%, you're making 2% gross in real terms. If you're getting 2% and inflation is close to zero, you're actually earning a similar amount, it's just the numbers are going up more slowly. Obviously when and how much tax you pay affects the actual return, it's just worth bearing in mind that low interest and low inflation are actually not that bad a savings environment as they first appear. For an emergency fund the key thing is ease of access, consider keeping some portion of your savings in an instant access account for those emergencies that happen when the banks are closed. In the UK there are various tax-free savings options, I'm guessing Canada has a few too, if so you should explore those options. While these may not have attractive headline rates, you don't pay tax on the interest, this can make them much more competitive (4% tax free is the same as 5% gross if you would have to pay tax at 20%). Normally tax free investments have caps so once you've invested a set amount you can't add anymore. This may be a consideration if you regularly dip into your emergency fund as you might not easily be able to build it up again. My approach is to have about 90% of my ""rainy day"" fund in easily accessible but tax free savings. This discourages me from spending it unless I really need to. I then keep a slush fund sufficient to cover every day disasters (boiler packing up, needing a hire car for a week etc) in instant access accounts ."

    """
    for i, (doc_id, score) in enumerate(results):
        text = corpus.loc[corpus['_id'] == doc_id, 'text'].iloc[0]
        print(f"Document ID: {doc_id}, Score: {score}")
        print(f"Text: {text}\n")