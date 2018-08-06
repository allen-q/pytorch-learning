# Introduction
This iPython Notebook was inspired by Andrej Karpathy' blog: The Unreasonable Effectiveness of Recurrent Neural Networks link: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

In his original post, Andrej published an vanilla implementation of the char rnn model in pure Python and numpy. See https://gist.github.com/karpathy/d4dee566867f8291f086

I took his idea and re-implemented the Char RNN model in Pytorch and trained a model using Jin Yong's famous Wu Xia novel "The Legend of The Condor Heroes" in an attempt to extend this great book.

The performance of the model was quite impressive. With a two layer LSTM RNN model and a few hours training, the model was able to generate some very interesting text. Some examples are shown below:


* 穆念慈认得那人只得远远跟着后再摇头，待华筝可是识破，于是大冷的叫道：“人是不肯我玩儿。”

* 穆念慈道：“回走！”穆念慈心中怨苦，告影不错。黄蓉奇道：“娶可你恶甚么好出京。”穆念慈抬头道：“你如此得了他们真实，他就无理，哪敢要害毛骨事？”郭靖道：“我们不内我的笑话，招术笨，老下也接你老人家首坐。那不是，是听定是老人家教求你？要是我们手不会肯传朱聪修习练肚，便不见到。

* 黄蓉骂道：“你一句‘梁子翁’这两下武艺，这一下叫他是笑弥陀究武中金国亲大的民不高人之中，武功已然不出，当下慢慢想起计嘻甚傻，说道：“靖哥哥了好，先立誓。”穆念慈叹道：“想不到宝贝呢？你可跪下去远近，说来跟他们一边皇帝，你们要过不见好，你托跪必有过招术。”

* 洪七公道：“多谢过你。爹爹又好，身边素会便了。”穆念慈从不意，摆了黄蓉道：“我这么忧，天下了无数时也没有他们再说。你要杀了你！我走破了可，叫化一作有徒儿，但统的听我喊扯，要原刚我若悲武艺，实是非成啦？于何他？”穆念慈道：“我也不是意思，这才杂毛我肉外，老毒物耳闻大的听不上七公，不可多言黄蓉比得你这女娃娃再救你。”欧阳克抢到道：“真是我的这自友虽然十未作眨我，却有实不指点无穷。”黄蓉笑道：“你们胆敢去罢，我就胡闹。罢你好玩儿。”

* 黄蓉哈哈大笑，微微一笑，沉吟道：“这些女子动手的。”格的一声，说道：“嗯，神夜侠义，今日我教了一个吃！那姓穆的时也是其实。”

* 黄药师是我的踪影，去杨门的野外，只听得我爹爹女子，你们死！”黄蓉道：“快势快说，却不是决不会有这么郑重的道理？”

* 洪七公道：“那怎么办？”穆念慈道：“只道不过不奸，但帮手对付他们对这许多局想在干干人边。这番独事，的却是在江南六侠三分好险，们就不到。”

* 朱聪道：“跃长了声音呼叱，只盼洪七公击在蛇身之上。两人挺了起来，她一招“法子尾”. 第一眼拂中，不追这面前微微笑容，抢步群蛇，一时在洪七公胸口逼出，笑问：“怎么事在这毒蛇记起小记、和我！”

You should be able to use this notebook to train your own model using any text data.