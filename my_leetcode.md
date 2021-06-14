# 算法题大全



### 存在重复元素

> 给定一个整数数组，判断是否存在重复元素。
>
> 如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。
>
>  示例 1:
>
> 输入: [1,2,3,1]
>输出: true
> 示例 2:
> 
> 输入: [1,2,3,4]
>输出: false
> 示例 3:
> 
> 输入: [1,1,1,3,3,4,3,2,4,2]
>输出: true
> 
> 来源：力扣（LeetCode）
>链接：https://leetcode-cn.com/problems/contains-duplicate

```python
思路：比较该数组原本的长度和去重之后的长度，如果两者相等，证明没有重复值

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
      return len(nums) != len(set(nums))
    	# return len(nums) > len(set(nums)s)
```





### 两个数组的交集II

> 给定两个数组，编写一个函数来计算它们的交集。
>
>  示例 1：
>
> 输入：nums1 = [1,2,2,1], nums2 = [2,2]
>输出：[2,2]
> 示例 2:
> 
> 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
>输出：[4,9]
> 
> 
>说明：
>
> 输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
>我们可以不考虑输出结果的顺序。
> 进阶：
> 
> 如果给定的数组已经排好序呢？你将如何优化你的算法？
>如果 nums1 的大小比 nums2 小很多，哪种方法更优？
> 如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有的元素到内存中，你该怎么办？
> 
> 来源：力扣（LeetCode）
>链接：https://leetcode-cn.com/problems/intersection-of-two-arrays-ii

```python
python思路1：双指针
先按照大小排序，然后双指针找相同的数来保留

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = sorted(nums1)
        nums2 = sorted(nums2)
        first, second = 0, 0
        temp = []
        while first<len(nums1) and second<len(nums2):
            if nums1[first] < nums2[second]:
                first+=1
            elif nums1[first] == nums2[second]:
                temp.append(nums1[first])
                first+=1
                second+=1
            else:
                second+=1
        return temp
      
      
#当然还可以加入判断列表是否为空
if not all([num1, num2]): #all(),any()里面只接受literable
  return []
```



```python
python思路2：
先找传统意义上的交集，然后再计算交集内每个数字应该出现的次数

class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums = list(set(nums1)&set(nums2))
        temp = []
        for num in nums:
            temp += [num] * min(nums1.count(num), nums2.count(num))
        return temp
```



### 重新排列句子中的单词

> 「句子」是一个用空格分隔单词的字符串。给你一个满足下述格式的句子 text :
>
> 句子的首字母大写
> text 中的每个单词都用单个空格分隔。
> 请你重新排列 text 中的单词，使所有单词按其长度的升序排列。如果两个单词的长度相同，则保留其在原句子中的相对顺序。
>
> 请同样按上述格式返回新的句子。
>
>  
>
> 示例 1：
>
> 输入：text = "Leetcode is cool"
> 输出："Is cool leetcode"
> 解释：句子中共有 3 个单词，长度为 8 的 "Leetcode" ，长度为 2 的 "is" 以及长度为 4 的 "cool" 。
> 输出需要按单词的长度升序排列，新句子中的第一个单词首字母需要大写。
> 示例 2：
>
> 输入：text = "Keep calm and code on"
> 输出："On and keep calm code"
> 解释：输出的排序情况如下：
> "On" 2 个字母。
> "and" 3 个字母。
> "keep" 4 个字母，因为存在长度相同的其他单词，所以它们之间需要保留在原句子中的相对顺序。
> "calm" 4 个字母。
> "code" 4 个字母。
> 示例 3：
>
> 输入：text = "To be or not to be"
> 输出："To be or to be not"
>
>
> 提示：
>
> text 以大写字母开头，然后包含若干小写字母以及单词间的单个空格。
> 1 <= text.length <= 10^5





```python
#python解法


#v1
def arrangeWords(text:str) -> str:
    text = text.lower() #小写首字母
    text = text.split() #借空格分隔成数组
    text.sort(key=lambda str:len(str)) #按元素的长度排序
    return  ' '.join(text).capitalize() #大写首字母
  
#v2:更加简洁
def arrangeWords(text):
    return ' '.join(sorted(text.lower().split(' '), key=len)).capitalize()
```



sorted(list)和list.sort()区别：

```
sort(self, /, *, key=None, reverse=False)
  Sort the list in ascending order and return None.
  The sort is in-place (i.e. the list itself is modified) and stable (i.e. the order of two equal elements is maintained).


sorted(iterable, /, *, key=None, reverse=False)
  Return a new list containing all items from the iterable in ascending order.
    

```



### 只出现一次的数



> 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
>
> 说明：
>
> 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
>
> 示例 1:
>
> 输入: [2,2,1]
> 输出: 1
> 示例 2:
>
> 输入: [4,1,2,1,2]
> 输出: 4



```python
  #java解法：利用for循环，采用异或
  #异或运算满足交换律，a^b^a=a^a^b=b,因此ans相当于nums[0]^nums[1]^nums[2]^nums[3]^nums[4]..... 然后再根据交换律把相等的合并到一块儿进行异或（结果为0），然后再与只出现过一次的元素进行异或，这样最后的结果就是，只出现过一次的元素（0^任意值=任意值
    public static int sigleNumber(int[] nums){
        int result = 0;
        for(int i =0; i<nums.length; i++){
            result ^= nums[i];
        }
        return result;
    }
  
  
  #java解法：stream + lambda一行搞定
      public static int sigleNumber(int[] nums){
        return Arrays.stream(nums).reduce(0,(x,y) -> x^y);
    }
  
  
  
  #python:异或
from typing import List
class Solution:
    def singleNumber(self, nums: List[int]):
        result = 0
        for num in nums:
            result ^= num
        return result
  
```







### 只出现一次的数II+

> 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
>
> 说明：
>
> 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
>
> 示例 1:
>
> 输入: [2,2,3,2]
> 输出: 3
> 示例 2:
>
> 输入: [0,1,0,1,0,1,99]
> 输出: 99

正解在这，目前还没看懂，所以写了最简单的暴力算法

https://leetcode-cn.com/problems/single-number-ii/solution/single-number-ii-mo-ni-san-jin-zhi-fa-by-jin407891/

```python
#python:暴力解法
class Solution:
    def singleNumber(self, *nums):
        return (sum(set(nums))*3 - sum(nums))//2
      
#python:效率低下的解法
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        nums_set = set(nums)
        for n in nums_set:
            if nums.count(n) == 1:
                return n
```



### 环形链表+

> 给定一个链表，判断链表中是否有环。
>







```java
//java快慢指针

/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        //至少要有两个才能形成环
        if(head == null || head.next == null){
            return false;
        }
        //假设slow每次走一步，fast可以走两步
        //它们的起始点其实是一样的，这里是假设都走了一次的情况下
        //上面的这个if已经保证了这走的一次的可能性
        ListNode slow = head;
        ListNode fast = head.next;
        while(slow != fast){
            //如果fast无路可走了，也就是说dda到达了链的终点，根本没有环
            if(fast == null || fast.next == null){
                return false;
            }
            //否则的话，继续往前走
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}

```



```python
#python快慢指针
class Solution(object):
	def hasCycle(self, head):
		"""
		:type head: ListNode
		:rtype: bool
		"""
		if not (head and head.next):
			return False
		#定义两个指针i和j，i为慢指针，j为快指针
		i,j = head,head.next
		while j and j.next:
			if i==j:
				return True
			# i每次走一步，j每次走两步
			i,j = i.next, j.next.next
		return False
```







### 快乐数

> 编写一个算法来判断一个数 n 是不是快乐数。
>
> 「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，那么这个数就是快乐数。
>
> 如果 n 是快乐数就返回 True ；不是，则返回 False 。
>
>  
>
> 示例：
>
> 输入：19
> 输出：true
> 解释：
> 12 + 92 = 82
> 82 + 22 = 68
> 62 + 82 = 100
> 12 + 02 + 02 = 1

https://leetcode-cn.com/problems/happy-number/

```python
#python最简单的方法（但是有致命弊端：用集合记录每次的计算结果来判断是否进入循环，因为这个集合可能大到无法存储；）

class Solution:
    def isHappy(self, n: int) -> bool: #输入12
        keep_nums = set()
        while True:
            n = [int(i)**2 for i in str(n)] #得到[1, 4]
            n = sum(n) #5
            if n == 1:
                return True
            #如果这个数在keep_nums里面已经存在过了，那就说明它是无限循环的
            elif n in keep_nums:
                return False
            else:
                keep_nums.add(n)
```

```python
# python快慢指针(推荐)
# 1. 创建一个慢指针，一次走一步，再创建一个快指针，一次走两步。
# 2. 当快慢指针相遇，代表形参环路，该数不是快乐数。
# 3. 若指针移动过程中，找到了 1（肯定是快指针最先找到），则当前数是一个快乐数。

class Solution:
  def is_happy(self, n:int)->bool:
    def get_next(num:int)->int:
      total_sum = 0
      while num > 0:
        num, digital = divmod(num, 10)
        total_sum += digital**2
      return total_sum
    
    slow_runner = n
    faste_runner = get_next(n) # 其实快针是可以等于慢针的，但是因为下面while的条件之一就是 slow_runner != fast_runner，为了能进入该循环，特此写成不相等
    while slow_runner != fast_runner and fast_runner != 1:
      slow_runner = get_next(slow_runner)
      fast_runner = get_next(get_next(fast_runner))
    return fast_runner == 1
  
        

```

没懂

```java
//java快慢针解法
class Solution {
    public boolean isHappy(int n) {
        int fast=n;
        int slow=n;
        do{
            slow=squareSum(slow);
            fast=squareSum(fast);
            fast=squareSum(fast);
        }while(slow!=fast);
        if(fast==1)
            return true;
        else return false;
    }
    
    private int squareSum(int m){
        int squaresum=0;
        while(m!=0){
           squaresum+=(m%10)*(m%10);
            m/=10;
        }
        return squaresum;
    }
}
```



### Excel表格



> 给定一个正整数，返回它在 Excel 表中相对应的列名称。
>
> 例如，
>
>     1 -> A
>     2 -> B
>     3 -> C
>     ...
>     26 -> Z
>     27 -> AA
>     28 -> AB 
>     ...
> 示例 1:
>
> 输入: 1
> 输出: "A"
> 示例 2:
>
> 输入: 28
> 输出: "AB"
> 示例 3:
>
> 输入: 701
> 输出: "ZY"
>
> 

```python
#python：两个解法，方法基本一致
class Solution:
    def convertToTitle(self, n: int) -> str:
        if n <= 0:
            raise ValueError("doit donner un chiffre >= 1")

        result = ''
        A = ord('A')
        # len
        while n != 0 :
            rest = (n-1) % 26 # n-1, parce que 0 ne correspond à rien
            result = chr(rest + A ) + result
            n = (n-1) // 26
        return result


      
      
      
      
      
      
      
class Solution:
    def convertToTitle(self, n: int) -> str:
        A = ord('A') #A的assic码的值
        result = ''
        while n:
            n -= 1 #为了让0对应A，而不是1对应A
            n, rest = divmod(n, 26) # n=n//26; rest = n%26
            result = chr(A + rest) + result
        return result
```







```java
//java

    public String convertToTitle(int n) {
        StringBuilder stringBuilder = new StringBuilder();
        while (n != 0) {
            n --;//这里稍作处理，因为它是从1开始
            stringBuilder.append((char)(n % 26 + 'A'));
            n /= 26;
        }
        return stringBuilder.reverse().toString();
    }
```





### Excel表格II

> 给定一个Excel表格中的列名称，返回其相应的列序号。
>
> 例如，
>
>     A -> 1
>     B -> 2
>     C -> 3
>     ...
>     Z -> 26
>     AA -> 27
>     AB -> 28 
>     ...
> 示例 1:
>
> 输入: "A"
> 输出: 1
> 示例 2:
>
> 输入: "AB"
> 输出: 28
> 示例 3:
>
> 输入: "ZY"
> 输出: 701



```python
#python
其实这题就是26进制的算法
2019用10进制可以写成
2*10**3 + 0*10**2 + 1*10**1 + 9
那么26进制的话把10改成26就可以了啊

class Solution:
    def titleToNumber(self, s: str) -> int:
        bit = 1 #其实是26的0次方，用来计算个位数
        res = 0
        for i in s[::-1]: #把字符串倒过来取
            res += (ord(i) - ord('A') + 1) * bit
            bit *= 26
        return res
```



```python
#python 来膜拜吧，和上面一样的思路，但是用法好高明啊啊啊

class Solution:
    def titleToNumber(self, s: str) -> int:
        return sum((ord(letter) - ord('A') + 1) * (26**i) for i,letter in enumerate(s[::-1]))

```



### 加一

> 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
>
> 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
>
> 你可以假设除了整数 0 之外，这个整数不会以零开头。
>
> 示例 1:
>
> 输入: [1,2,3]
> 输出: [1,2,4]
> 解释: 输入数组表示数字 123。
> 示例 2:
>
> 输入: [4,3,2,9]
> 输出: [4,3,3,0]
> 解释: 输入数组表示数字 4330。

```python
#python一句话搞定，稍微有点绕

from typing import List

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int,list(str(int("".join(map(str, digits)))+1))))
         

        #假设传入参数是[1, 9, 9]
        #map(str, digits) => <map object at 0x101859520>
        #"".join(map(str, digits)) => 199, 但是是str类型。如果str(这个199会得到['1', '9', '9']，并不是我们想要的
        #int("".join(map(str, digits))) => 199，但是已经是int类型了。但是却不可以list(这里的int199)，因为TypeError: 'int' object is not iterable
        #int("".join(map(str, digits)))+1  => 200，趁热打铁马上加1
        #list(str(int("".join(map(str, digits)))+1)) => ['2', '0', '0']
        #map(int,list(str(int("".join(map(str, digits)))+1))) => <map object at 0x106c1a4c0>
        #list(map(int,list(str(int("".join(map(str, digits)))+1)))) => [2, 0, 0]。巧妙的把原来的['2', '0', '0']，变成了int的[2, 0, 0]

      
```





```python
#python超强震撼版！！

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        return [int(j) for j in str(int(''.join('%s' % i for i in digits))+1)]
      
      
      #假设传入参数是[1, 9, 9]
				#''.join('%s' % i for i in digits) => 199, %s是把i转换成str
        #int(''.join('%s' % i for i in digits))+1 => 200 (int类型)
        #[int(j) for j in str(int(''.join('%s' % i for i in digits))+1)] =>[2,0,0]
```





```java
//java也很精彩 


class Solution {
        public int[] plusOne(int[] digits) {
            for (int i = digits.length - 1; i >= 0; i--) {
                digits[i]++;
                digits[i] = digits[i] % 10;
                if (digits[i] != 0) return digits;
            }
            //下面是针对9，99，999这种类型的数
            digits = new int[digits.length + 1];
            digits[0] = 1;
            return digits;
        }
    }
```







### 单词规律

> 给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。
>
> 这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。
>
> 示例1:
>
> 输入: pattern = "abba", str = "dog cat cat dog"
> 输出: true
> 示例 2:
>
> 输入:pattern = "abba", str = "dog cat cat fish"
> 输出: false
> 示例 3:
>
> 输入: pattern = "aaaa", str = "dog cat cat dog"
> 输出: false
> 示例 4:
>
> 输入: pattern = "abba", str = "dog dog dog dog"
> 输出: false
> 说明:
> 你可以假设 pattern 只包含小写字母， str 包含了由单个空格分隔的小写字母。    



```python
#python:膜拜大神的思路，利用index来描绘每个单词/字母的位置

class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
       res = str.split()
       return list(map(pattern.index, pattern))== list(map(res.index, res))

      	#模拟Solution().wordPattern("abba", "dog cat cat dog")
        #res = str.split() =>['dog', 'cat', 'cat', 'dog'], 这步必须，否则直接map(str.index, str)的话，它会被拆分为单个字母（d, o,g,c...）,而不是单词（dog, cat ...）
        #list(map(res.index, res)) => [0, 1, 1, 0]
        #list(map(pattern.index, pattern)) => [0, 1, 1, 0], 和str一样的操作，只不过，不需要有split这一步
       
```





### 单词大写

> 给定一个单词，你需要判断单词的大写使用是否正确。
>
> 我们定义，在以下情况时，单词的大写用法是正确的：
>
> 全部字母都是大写，比如"USA"。
> 单词中所有字母都不是大写，比如"leetcode"。
> 如果单词不只含有一个字母，只有首字母大写， 比如 "Google"。
> 否则，我们定义这个单词没有正确使用大写字母。
>
> 示例 1:
>
> 输入: "USA"
> 输出: True
> 示例 2:
>
> 输入: "FlaG"
> 输出: False
> 注意: 输入是由大写和小写拉丁字母组成的非空单词。



```python
#python: 第一次没有参考别人的思路自己写的，虽然不是最优解，但是值得纪念
思路：把输入的单词分别转成全部大写的，全部小写的，和仅仅大写首字母的，然后进行对比，如果不符合其中任何一个，则返回False


class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        word_cap = word.capitalize()
        word_upper = word.upper()
        word_lower = word.lower()
        if word == word_cap or word == word_lower or word == word_upper:
            return True
        return False
```



```java
//java 和上面的python一样的思路，也是自己写的哦
public class Solution {
    public Boolean detectCapitalUse(String word) {
        String wordUpper = word.toUpperCase();
        String wordLower = word.toLowerCase();
        String wordCap = word.substring(0,1).toUpperCase() + word.substring(1).toLowerCase();
        System.out.println(wordCap);

        if (word.equals(wordUpper) || word.equals(wordLower) || word.equals(wordCap)){
            return true;
        }
        return false;
}
}
```



```python
#python:更巧妙的使用各种方法来判断

class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.isupper() or word.islower() or word.istitle()

```







### 最后一块石头的重量

> 有一堆石头，每块石头的重量都是正整数。
>
> 每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
>
> 如果 x == y，那么两块石头都会被完全粉碎；
> 如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
> 最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。
>
>  
>
> 示例：
>
> 输入：[2,7,4,1,8,1]
> 输出：1
> 解释：
> 先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
> 再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
> 接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
> 最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。
>
>
> 提示：
>
> 1 <= stones.length <= 30
> 1 <= stones[i] <= 1000
>
> 

```python
#python 自己做的呢！

from typing import List

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:

        while len(stones)>=2:
            # 这种写法错误stones_sorted = stones.sort()，因为sort方法直接修改list本身
            stones.sort()
            #取出最大的两个值
            x = stones[-1]
            y = stones[-2]
            #如果两值相等，直接去掉后两位
            if x == y :
                stones = stones[:-2]
            #两者不等，去掉后两位，并把两值之差的绝对值加入
            else:
                num = abs(x - y)
                stones = stones[:-2]
                stones.append(num)
        #如果最后为空，返回0；否则返回list里面剩下的这个数值
        if stones :
            return stones[0]
        else:
            return 0
```







```java
//java:就是排序，每次都让最后末尾两个相减，一个就变成0，一个就变成差，再排序，直到最后一个石头

public class Solution {
    public int lastStoneWeight(int[] stones) {
        int index = stones.length - 1;
        for(int i = 0; i < stones.length - 1; i++){     //通过stones.length来判断需要操作的次数。（不用将stones.length == 1的情况单独考虑）
            Arrays.sort(stones);                        //将sort放在循环体的开始。（避免在循环体外再写一次重复的sort（））
            stones[index] -= stones[index-1];           //两种不同情况使用同一表达式处理。（）
            stones[index-1] = 0;
        }
        return stones[stones.length-1];
    }
    
}
```





### 缺失数字

> 



```
#python:又是自己写出来的，哈哈哈
思路：
对nums进行从小到大排序（有一个数的缺失），然后再建一个list包含0到n(完整的) => 用两个list去互相做减法，第一个两数之差为1的那一对就是有问题的！
例如：排序好的有缺失的list是 [0,1,3,4,5]
		          完整的list是[0,1,2,3,4,5]
那么会在3和2处出现第一次差为1的情况，这时返回完整list中的那个数，也就是2,即可。

为了防止出现输入为[0,1]，期待结果为[0,1,2]的情况，特意将不完整的List的末尾添加了一个0，让两个List的长度一致。否则会出现



from typing import List

class Solution:
    def missingNumber(self, nums: List[int]) -> int:

        nums.sort()
        nums.append(0)
        
        nums_right = [i for i in range(0,len(nums))]

        for num, num_right in zip(nums, nums_right):
                if num != num_right :
                    return num_right
                else:
                    continue



print(Solution().missingNumber([0,1,2]))
print(Solution().missingNumber([0]))
print(Solution().missingNumber([1,0,2,5,4]))
```



```python
#python:大神的异或解法，很方便啊
思路：假设输入为[0,1,2,3]

下标	0	1	2	3
数字	0	1	3	4
可以将结果的初始值设为 nn，再对数组中的每一个数以及它的下标进行一个异或运算，即：
  
=4∧(0∧0)∧(1∧1)∧(2∧3)∧(3∧4)
=(4∧4)∧(0∧0)∧(1∧1)∧(3∧3)∧2
=0∧0∧0∧0∧2
=2


class Solution:
    def missingNumber(self, nums):
        missing = len(nums)
        for i, num in enumerate(nums):
            missing ^= i ^ num
        return missing

```





> 给定一个整数数组，判断是否存在重复元素。
>
> 如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。
>
>  
>
> 示例 1:
>
> 输入: [1,2,3,1]
> 输出: true
> 示例 2:
>
> 输入: [1,2,3,4]
> 输出: false
> 示例 3:
>
> 输入: [1,1,1,3,3,4,3,2,4,2]
> 输出: true

```python
#python:自己写的，比较该数组和变成set后的长度
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if len(set(nums)) ==  len(nums): return False
        return True
        
 
 #更简单的写法
 class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))
```





```java
//java: 该数组和变成set后的长度
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>(nums.length);
        for(int i : nums){
            set.add(i);
        }
        return nums.length != set.size();
    }
}
```



```java
//java: 先排序，再比较相邻的两数是否相等
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        for(int i=0; i < nums.length-1; i++){ //这里一定要小于nums.length-1，避免到时候nums[i+1]根本取不到
            if(nums[i] ==  nums[i+1]){
                return true;
            }
        }
        return false;
    }
}
```



### 最长的公共前缀

> 编写一个函数来查找字符串数组中的最长公共前缀。
>
> 如果不存在公共前缀，返回空字符串 ""。
>
> 示例 1:
>
> 输入: ["flower","flow","flight"]
> 输出: "fl"
> 示例 2:
>
> 输入: ["dog","racecar","car"]
> 输出: ""
> 解释: 输入不存在公共前缀。
> 说明:
>
> 所有输入只包含小写字母 a-z 。
>



```python
思路：使用 zip 根据字符串下标合并成数组，判断合并后数组里元素是否都相同
zip使用例子https://www.runoob.com/python/python-func-zip.html
  
  
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for str in zip(*strs):
            if len(set(str)) == 1:
                res += str[0]
            else:
                break
        return res
```





> 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
>
> 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
>
> 示例:
>
> 输入："23"
> 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
>
>
> 链接：https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number
>



```python
#python巧妙用法
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        conversion = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        if len(digits) == 0 : return []

        res = ['']
        for key in digits:
            res = [ i+j for i in res for j in conversion[key]]
        return res
```









```
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)
```





### 爬楼梯

> 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
>
> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
>
> 注意：给定 n 是一个正整数。
>
> 示例 1：
>
> 输入： 2
> 输出： 2
> 解释： 有两种方法可以爬到楼顶。
> 1.  1 阶 + 1 阶
> 2.  2 阶
> 示例 2：
>
> 输入： 3
> 输出： 3
> 解释： 有三种方法可以爬到楼顶。
> 1.  1 阶 + 1 阶 + 1 阶
> 2.  1 阶 + 2 阶
> 3.  2 阶 + 1 阶
>



```python
#python：动态规划
假设楼梯有i阶，那么最后一次迈步只有两种情况，要么停留在了i-1阶，要么在i-2阶。那么爬到i阶的方法就是爬到i-1和i-2阶的总和

class Solution:
    def climbStairs(self, n: int) -> int:
        climb = {}

        climb[0] = 0
        climb[1] = 1
        climb[2] = 2
        for i in range(3, n+1):
            climb[i] = climb[i-1] + climb[i-2]
        return climb[n]
```

```java
//java:同样是动态规划的思路

class Solution {
    public int climbStairs(int n) {
        if(n==1){
            return 1;
        } //这个if判断不可少，否则n=1时，下面的dp[2]就Index 2 out of bounds for length 2
        int[] dp = new int[n+1];
        dp[1] = 1;
        dp[2] = 2;
        for(int i=3; i <= n; i++){
            dp[i] =  dp[i-1] + dp[i-2];
        }
        return dp[n];
    }
}
```







### 独特的电子邮箱地址

> 每封电子邮件都由一个本地名称和一个域名组成，以 @ 符号分隔。
>
> 例如，在 alice@leetcode.com中， alice 是本地名称，而 leetcode.com 是域名。
>
> 除了小写字母，这些电子邮件还可能包含 '.' 或 '+'。
>
> 如果在电子邮件地址的本地名称部分中的某些字符之间添加句点（'.'），则发往那里的邮件将会转发到本地名称中没有点的同一地址。例如，"alice.z@leetcode.com” 和 “alicez@leetcode.com” 会转发到同一电子邮件地址。 （请注意，此规则不适用于域名。）
>
> 如果在本地名称中添加加号（'+'），则会忽略第一个加号后面的所有内容。这允许过滤某些电子邮件，例如 m.y+name@email.com 将转发到 my@email.com。 （同样，此规则不适用于域名。）
>
> 可以同时使用这两个规则。
>
> 给定电子邮件列表 emails，我们会向列表中的每个地址发送一封电子邮件。实际收到邮件的不同地址有多少？
>
>  
>
> 示例：
>
> 输入：["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
> 输出：2
> 解释：实际收到邮件的是 "testemail@leetcode.com" 和 "testemail@lee.tcode.com"。
>
>
> 提示：
>
> 1 <= emails[i].length <= 100
> 1 <= emails.length <= 100
> 每封 emails[i] 都包含有且仅有一个 '@' 字符。
>
> 来源：力扣（LeetCode）
> 链接：https://leetcode-cn.com/problems/unique-email-addresses



```python
#解法1
思路：
根据@的位置分为前后两部分local和domain
local里面再根据+的位置（如果有）分为几个部分，然后仅仅保留第一个+前面的内容
去掉保留内容里面的所有点.
再把local中保留的部分，和没有任何变动的domain部分用@连接起来


class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        mail_set = set()
        for mail in emails:
            local, domain = mail.split('@')
            if '+' in local: #这个判断是必要的，因为如果没有+的话，local.index('+')会报错
                local = local[:local.index('+')]
            mail_set.add(local.replace('.','') + '@' + domain)
        return len(mail_set)
```





```python
#解法2：
思路和解法1完全一致
只是对解法1的local.index('+')做出了修改

class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        mail_set = set()
        for mail in emails:
            local, domain = mail.split('@')
            local = local.split('+')[0] #无论有多少个+都无所谓，仅仅截取的是第一个+号前面的部分哦！
            local = local.replace('.', '')
            res = local + '@' + domain
            mail_set.add(res)
            print(local)
            
        return len(mail_set)
```







### 旋转数字

> 我们称一个数 X 为好数, 如果它的每位数字逐个地被旋转 180 度后，我们仍可以得到一个有效的，且和 X 不同的数。要求每位数字都要被旋转。
>
> 如果一个数的每位数字被旋转以后仍然还是一个数字， 则这个数是有效的。0, 1, 和 8 被旋转后仍然是它们自己；2 和 5 可以互相旋转成对方（在这种情况下，它们以不同的方向旋转，换句话说，2 和 5 互为镜像）；6 和 9 同理，除了这些以外其他的数字旋转以后都不再是有效的数字。
>
> 现在我们有一个正整数 N, 计算从 1 到 N 中有多少个数 X 是好数？
>
>  
>
> 示例：
>
> 输入: 10
> 输出: 4
> 解释: 
> 在[1, 10]中有四个好数： 2, 5, 6, 9。
> 注意 1 和 10 不是好数, 因为他们在旋转之后不变。
>
>
> 提示：
>
> N 的取值范围是 [1, 10000]。
>
> 来源：力扣（LeetCode）
> 链接：https://leetcode-cn.com/problems/rotated-digits



```python
#解法1：暴力解法
成为好数字有两个条件
1.里面一定不可以有3，4或7
2.里面一定至少有2，5，6，7中的其中一个（避免全部数字都是由0，1，8组成的，它们只能变成它们自己）


class Solution:
    def rotatedDigits(self, N: int) -> int:
        res = 0
        for n in range(1, N+1):
            num = str(n)
            #all要全部是True才是True,any只要其中一个为True则为True
            #int + boolean是可以的，True=1, False=0
            res += (all(d not in '347' for d in num) and any(d in '2569' for d in num))
        return res
```







### 种花问题

> 假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
>
> 给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 n 。能否在不打破种植规则的情况下种入 n 朵花？能则返回True，不能则返回False。
>
> 示例 1:
>
> 输入: flowerbed = [1,0,0,0,1], n = 1
> 输出: True
> 示例 2:
>
> 输入: flowerbed = [1,0,0,0,1], n = 2
> 输出: False
> 注意:
>
> 数组内已种好的花不会违反种植规则。
> 输入的数组长度范围为 [1, 20000]。
> n 是非负整数，且不会超过输入数组的大小。
>
> 来源：力扣（LeetCode）
> 链接：https://leetcode-cn.com/problems/can-place-flowers
>
> 

```

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    	tmp = [0]+flowerbed+[0]
    	for i in range(1, len(tmp)-1):
    		if tmp[i-1] == tmp[i] ==tmp[i+1] ==0:
    			flowerbed[i] = 1
    			n -= 1
    	return n<=0
    
    
```







### 唯一摩斯密码词

> 国际摩尔斯密码定义一种标准编码方式，将每个字母对应于一个由一系列点和短线组成的字符串， 比如: "a" 对应 ".-", "b" 对应 "-...", "c" 对应 "-.-.", 等等。
>
> 为了方便，所有26个英文字母对应摩尔斯密码表如下：
>
> [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
> 给定一个单词列表，每个单词可以写成每个字母对应摩尔斯密码的组合。例如，"cab" 可以写成 "-.-..--..."，(即 "-.-." + ".-" + "-..." 字符串的结合)。我们将这样一个连接过程称作单词翻译。
>
> 返回我们可以获得所有词不同单词翻译的数量。
>
> 例如:
> 输入: words = ["gin", "zen", "gig", "msg"]
> 输出: 2
> 解释: 
> 各单词翻译如下:
> "gin" -> "--...-."
> "zen" -> "--...-."
> "gig" -> "--...--."
> "msg" -> "--...--."
>
> 共有 2 种不同翻译, "--...-." 和 "--...--.".
>
>
> 注意:
>
> 单词列表words 的长度不会超过 100。
> 每个单词 words[i]的长度范围为 [1, 12]。
> 每个单词 words[i]只包含小写字母。

```python
python解法：

class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
    	morse_list= [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
    	res_list = []
    	for word in words:
    		res = ''
    		for letter in word:
          # 利用字母的ascii码,不要直接写成morse_list[ord(letter)-97]，尽管ord('a')却是是等于97的
    			res += morse_list[ord(letter)-ord('a')]
    		res_list.append(res)
    	return len(set(res_list))
    						
```



### 正数的各位积和之差

> Given an integer number n, return the difference between the product of its digits and the sum of its digits.
>
>
> Example 1:
>
> Input: n = 234
> Output: 15 
> Explanation: 
> Product of digits = 2 * 3 * 4 = 24 
> Sum of digits = 2 + 3 + 4 = 9 
> Result = 24 - 9 = 15
> Example 2:
>
> Input: n = 4421
> Output: 21
> Explanation: 
> Product of digits = 4 * 4 * 2 * 1 = 32 
> Sum of digits = 4 + 4 + 2 + 1 = 11 
> Result = 32 - 11 = 21
>
>
> Constraints:
>
> 1 <= n <= 10^5
>
> 来源：力扣（LeetCode）
> 链接：https://leetcode-cn.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer



```python
python解法：
这题乍看超级简单，但是却涉及到int和str的转换。
我自己写的思路正确，但是别人的写法更精炼简洁

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        #先转换为str才能变成list
        numbers = list(str(n))
        print(numbers) #["1","2","3"]
        #然后又必须变成int后面才能做计算
        numbers = list(map(int, numbers))
        print(numbers) #[1,2,3]
        n_sum = sum(number for number in numbers)
        n_product = reduce(lambda x,y: x*y, numbers)
        return n_product - n_sum
```







```python
python解法：和上面的思路相差无几，但是更精炼简洁

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
    		multyply = 1
    		total = 0
    		for number in str(n):
    				multiply *= int(number)
    				total += int(number)
    		return multiply - total
```





### 生成验证码

验证码必须可以包含大小写字母和0-9数字

```python
import random
def get_random_letters(quantity:int)->str:
        chars = []
        for i in range(quantity):
            random_letter_lower = chr(random.randint(97,122))
            random_letter_upper = chr(random.randint(65, 90))
            random_num = str(random.randint(0,9))
            char = random.choice([random_num, random_letter_lower, random_letter_upper])
            chars.append(char)
        return ' '.join(chars)

```

