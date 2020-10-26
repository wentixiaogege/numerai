#### [数值的n次方](https://www.nowcoder.com/practice/1a834e5e3e1a4b7ba251417554e07c00?tpId=13&tqId=11165&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：，用二分

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if exponent ==0:
            return 1
        if exponent ==1:
            return base
        ans,pos=1,1
        if exponent <0:
            pos=-1
            exponent = - exponent
        if exponent %2 ==0: # 偶数
            ans = self.Power(base,exponent//2)*self.Power(base,exponent//2)
        else: # 奇数
            ans = base * self.Power(base,exponent//2)*self.Power(base,exponent//2)
        return ans if pos==1 else 1.0/ans
```

#### [二分法和牛顿迭代法求平方根](https://blog.csdn.net/ycf74514/article/details/48996383?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)

![img](https://pic2.zhimg.com/80/v2-5b9cb84f2f326c5739ea5e0ae8e9411d_720w.jpg)

```
##二分法
import math
from math import sqrt
 
def sqrt_binary(num):
	x=sqrt(num)
	y=num/2.0
	low=0.0
	up=num*1.0
	count=1
	while abs(y-x)>0.00000001:
		print count,y
		count+=1		
		if (y*y>num):
			up=y
			y=low+(y-low)/2
		else:
			low=y
			y=up-(up-y)/2
	return y
 
print(sqrt_binary(5))
## 牛顿法 二次方
## https://zhuanlan.zhihu.com/p/111598542
def sqrt_newton(num):
	x=sqrt(num)
	y=num/2.0
	count=1
	while abs(y-x)>0.00000001:
		print count,y
		count+=1
		y=((y*1.0)+(1.0*num)/y)/2.0000
	return y
 
print(sqrt_newton(5))
print(sqrt(5))

### 牛顿法 三次方
def cube_newton(num):
	x=num/3.0
	y=0
	count=1
	while abs(x-y)>0.00000001:
		print count,x
		count+=1
		y=x
		x=(2.0/3.0)*x+(num*1.0)/(x*x*3.0)
	return x
 
print(cube_newton(27))
```



#### [翻转链表](https://www.nowcoder.com/practice/75e878df47f24fdc9dc3e400ec6058ca?tpId=13&tqId=11168&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

https://blog.csdn.net/songyunli1111/article/details/79416684

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        last = None   #指向上一个节点
        while pHead:
            #先用tmp保存pHead的下一个节点的信息，
            #保证单链表不会因为失去pHead节点的next而就此断裂
            tmp = pHead.next   
            #保存完next，就可以让pHead的next指向last了
            pHead.next = last
            #让last，pHead依次向后移动一个节点，继续下一次的指针反转
            last = pHead
            pHead = tmp
        return last
```

####  [二分查找（九章算法)](https://www.jiuzhang.com/problem/binary-search/#tag-lang-python)

题目：给定一个排好序的证书数组nums，和一个整数target，寻找target在nums中任何一个/第一次出现/最后一次出现的位置，不存在return-1。

思路：基本上看到时间复杂度要求O（logN）基本就是要用二分法，二分法的本质是保留有解的一半。

**3.通用的二分法模板(四个要素)**

①start+1<end

②start+(end-start)/2

③A[mid]==,<,>

④A[start] A[end] ? target 

```python
class Solution:
    # @param nums: The integer array
    # @param target: Target number to find
    # @return the first position of target in nums, position start from 0 
    def binarySearch(self, nums, target):
        if not nums:
            return -1
            
        start, end = 0, len(nums) - 1
        # 用 start + 1 < end 而不是 start < end 的目的是为了避免死循环
        # 在 first position of target 的情况下不会出现死循环
        # 但是在 last position of target 的情况下会出现死循环
        # 样例：nums=[1，1] target = 1
        # 为了统一模板，我们就都采用 start + 1 < end，就保证不会出现死循环
        while start + 1 < end:
            # python 没有 overflow 的问题，直接 // 2 就可以了
            # java和C++ 最好写成 mid = start + (end - start) / 2
            # 防止在 start = 2^31 - 1, end = 2^31 - 1 的情况下出现加法 overflow
            mid = (start + end) // 2
            
            # > , =, < 的逻辑先分开写，然后在看看 = 的情况是否能合并到其他分支里
            if nums[mid] < target:
                # 写作 start = mid + 1 也是正确的
                # 只是可以偷懒不写，因为不写也没问题，不会影响时间复杂度
                # 不写的好处是，万一你不小心写成了 mid - 1 你就错了
                start = mid
            elif nums[mid] == target:
                end = mid
            else: 
                # 写作 end = mid - 1 也是正确的
                # 只是可以偷懒不写，因为不写也没问题，不会影响时间复杂度
                # 不写的好处是，万一你不小心写成了 mid + 1 你就错了
                end = mid
                
        # 因为上面的循环退出条件是 start + 1 < end
        # 因此这里循环结束的时候，start 和 end 的关系是相邻关系（1和2，3和4这种）
        # 因此需要再单独判断 start 和 end 这两个数谁是我们要的答案
        # 如果是找 first position of target 就先看 start，否则就先看 end
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        
        return -1
```

[查找range](https://www.jiuzhang.com/problem/find-first-and-last-position-of-element-in-sorted-array/#tag-lang-python)

```python
class Solution:
    """
    @param nums: the array of integers
    @param target: 
    @return: the starting and ending position
    """
    def searchRange(self, nums, target):
        # Write your code here.
        if not nums:
            return [-1, -1]
            
        first_index =  self.binarySearch(nums, target, True)
        if first_index == -1:
            return [-1, -1]
        last_index =  self.binarySearch(nums, target, False)
        return [first_index, last_index]
    
    def binarySearch(self, nums, target, isFindFirst):
        start, end = 0, len(nums) - 1 
        while start + 1 < end:
            mid = (start + end) // 2 
            if target > nums[mid]:
                start = mid
            elif target < nums[mid]:
                end = mid
            else:
                if isFindFirst:
                    end = mid
                else:
                    start = mid
        if isFindFirst:
            if nums[start] == target:
                return start 
            if nums[end] == target:
                return end 
            return -1
        
        else:
            if nums[end] == target:
                return end
            if nums[start] == target:
                return start
            return -1
```



#### [ 旋转数组的最小数字](https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=13&tqId=11159&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：二分判断

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if rotateArray == []:
            return 0
        _len = len(rotateArray)
        left = 0
        right = _len - 1
        while left <= right:
            mid = int((left + right) >> 1)
            if rotateArray[mid]<rotateArray[mid-1]:
                return rotateArray[mid]
            if rotateArray[mid] >= rotateArray[right]:
                # 说明在【mid，right】之间
                left = mid + 1
            else:
                # 说明在【left，mid】之间
                right = mid - 1
        return rotateArray[mid]
    
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, nums):
        # write code here
        if not nums:
            return -1
            
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
                
        return min(nums[start], nums[end])
```



####  [矩形覆盖](https://www.nowcoder.com/practice/72a5a919508a4251859fb2cfb987a0e6?tpId=13&tqId=11163&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
思考： 2*1 1 种； 2*2 2种 2*3 3种 2*4 5种

```python
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here

        if number<=2:
            return number
        dp = [1,2]
        for i in range(number-2):
            dp.append(dp[-1]+dp[-2])
        return dp[-1]
```



#### 把字符串转换成整数
将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0
思考：如果有正负号，需要在数字之前，出现其他字符或者字符串为空都非法返回0

```python
class Solution:
    def StrToInt(self, s):
        # write code here
        flag = True  # 是否出现过 + - 符号
        pos = 1 # 正负号
        ret = None # 返回值
        if s=='':
            return 0
        for i in s:
            if i=='+' or i=='-':
                if flag:
                    pos = -1 if i=='-' else 1
                    flag = False
                else:
                    return 0
            elif i>='0' and i<='9':
                flag = False
                if ret == None:
                    ret = int(i)
                else:
                    ret = ret*10+int(i)
            else:
                return 0
        return pos*ret if ret else 0
```

#### [二叉树的先序、中序、后续遍历 递归和非递归](https://www.cnblogs.com/icekx/p/9127569.html)

```python
# 先序打印二叉树（递归）
def preOrderTraverse(node):
    if node is None:
        return None
    print(node.val)
    preOrderTraverse(node.left)
    preOrderTraverse(node.right)
# 先序打印二叉树（非递归）
def preOrderTravese(node):
    stack = [node]
    while len(stack) > 0:
        print(node.val)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
        node = stack.pop()
        
# 中序打印二叉树（递归）
def inOrderTraverse(node):
    if node is None:
        return None
    inOrderTraverse(node.left)
    print(node.val)
    inOrderTraverse(node.right)
# 中序打印二叉树（非递归）
def inOrderTraverse(node):
    stack = []
    pos = node
    while pos is not None or len(stack) > 0:
        if pos is not None:
            stack.append(pos)
            pos = pos.left
        else:
            pos = stack.pop()
            print(pos.val)
            pos = pos.right
            
# 后序打印二叉树（递归）
def postOrderTraverse(node):
    if node is None:
        return None
    postOrderTraverse(node.left)
    postOrderTraverse(node.right)
    print(node.val)
# 后序打印二叉树（非递归）
# 使用两个栈结构
# 第一个栈进栈顺序：左节点->右节点->跟节点
# 第一个栈弹出顺序： 跟节点->右节点->左节点(先序遍历栈弹出顺序：跟->左->右)
# 第二个栈存储为第一个栈的每个弹出依次进栈
# 最后第二个栈依次出栈
def postOrderTraverse(node):
    stack = [node]
    stack2 = []
    while len(stack) > 0:
        node = stack.pop()
        stack2.append(node)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    while len(stack2) > 0:
        print(stack2.pop().val)
        
# 先进先出选用队列结构
import queue
def layerTraverse(head):
    if not head:
        return None
    que = queue.Queue()      # 创建先进先出队列
    que.put(head)
    while not que.empty():
        head = que.get()    # 弹出第一个元素并打印
        print(head.val)
        if head.left:       # 若该节点存在左子节点,则加入队列（先push左节点）
            que.put(head.left)
        if head.right:      # 若该节点存在右子节点,则加入队列（再push右节点）
            que.put(head.right)
# 求二叉树节点个数
def treeNodenums(node):
    if node is None:
        return 0
    nums = treeNodenums(node.left)
    nums += treeNodenums(node.right)
    return nums + 1
```



#### [平衡二叉树的判断](https://www.nowcoder.com/practice/8b3b95850edb4115918ecebdf1b4d222?tpId=13&tqId=11192&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

思考：BST的定义为，原问题拆分为计算树高度和判断高度差

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def IsBalanced_Solution(self, pRoot):
        if not pRoot: return True
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        if abs(left - right) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and   self.IsBalanced_Solution(pRoot.right)
    
    def TreeDepth(self, pRoot):
        if not pRoot: return 0
        left = self.TreeDepth(pRoot.left)
        right = self.TreeDepth(pRoot.right)
        return max(left, right) + 1
```

#### [二叉树的深度](https://www.nowcoder.com/practice/435fb86331474282a3499955f0a41e8b?tpId=13&tqId=11191&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot == None:
            return 0
        if pRoot.left == None and pRoot.right==None:
            return 1
        return max(self.TreeDepth(pRoot.left),self.TreeDepth(pRoot.right))+1
```


#### [二叉树的下一个结点](https://www.nowcoder.com/practice/9023a0c988684a53960365b889ceaf5e?tpId=13&tqId=11210&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
思路：中序遍历的顺序为LVR
则有以下三种情况：
a. 如果该结点存在右子结点，那么该结点的下一个结点是右子结点树上最左子结点
b. 如果该结点不存在右子结点，且它是它父结点的左子结点，那么该结点的下一个结点是它的父结点
c. 如果该结点既不存在右子结点，且也不是它父结点的左子结点，则需要一路向祖先结点搜索，直到找到一个结点，该结点是其父亲结点的左子结点。如果这样的结点存在，那么该结点的父亲结点就是我们要找的下一个结点。

```python
class Solution:
    def GetNext(self, pNode):
        # write code here
        # left root right
        if pNode == None: # 空节点
            return None
        if pNode.right: # 有右节点
            tmp = pNode.right
            while(tmp.left):
                tmp = tmp.left
            return tmp
        p = pNode.next # 没有右节点
        while(p and p.right==pNode):
            pNode = p
            p = p.next
        return p
```



#### [对称的二叉树](https://www.nowcoder.com/practice/ff05d44dfdb04e1d83bdbdab320efbcb?tpId=13&tqId=11211&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
[Leetcode 101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/)
判断一棵树是不是左右对称的树

```python
class Solution:
    def Symmetrical(self,Lnode,Rnode):
        if Lnode == None and Rnode == None:
            return True
        if Lnode and Rnode:
            return Lnode.val == Rnode.val and self.Symmetrical(Lnode.right,Rnode.left) and self.Symmetrical(Lnode.left,Rnode.right)
        else:
            return False
    def isSymmetrical(self, pRoot):
        # write code here
        if pRoot == None:
            return True
        return self.Symmetrical(pRoot.left,pRoot.right)
```

#### [将二叉树按照层级转化为链表](https://blog.csdn.net/majichen95/article/details/86630492)



```python
    1
   / \
  2   3
 /
4
[
  1->null,
  2->3->null,
  4->null
]
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
"""
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists(self, root):
        # Write your code here
        #二叉树的层次遍历
        res_list = []
        if not root:
            return []
        
        q = [root]
        while q:
            new_q = []
            dummy = ListNode(0)
            pre = dummy
            
            for node in q:
                if node.left:
                    new_q.append(node.left)
                if node.right:
                    new_q.append(node.right)
                
                pre.next = ListNode(node.val)
                pre = pre.next
            
            q = new_q
            res_list.append(dummy.next)
        
        return res_list
```



#### [把二叉树打印成多行](https://www.nowcoder.com/practice/445c44d982d04483b04a54f298796288?tpId=13&tqId=11213&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

```python
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if pRoot == None:
            return []
        stack = [pRoot]
        ret = []

        while(stack):
            tmpstack = []
            tmp = []
            for node in stack:
                tmp.append(node.val)
                if node.left:
                    tmpstack.append(node.left)
                if node.right:
                    tmpstack.append(node.right)
            ret.append(tmp[:])
            stack = tmpstack[:]
        return ret
```



#### [之字形打印二叉树](https://www.nowcoder.com/practice/91b69814117f4e8097390d107d2efbe0?tpId=13&tqId=11212&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```python
class Solution:
    def Print(self, pRoot):
        # write code here
        if pRoot == None:
            return []
        stack = [pRoot]
        step = 1
        ret = []
        while(stack):
            tmpstack = []
            tmp = []
            for node in stack:
                tmp+=[node.val]
                if node.left:
                    tmpstack.append(node.left)
                if node.right:
                    tmpstack.append(node.right)
            if step%2==0:
                tmp.reverse()
            ret.append(tmp)
            step += 1
            stack = tmpstack[:]
        return ret 
```



#### [序列化和反序列化二叉树](https://www.nowcoder.com/practice/cf7e25aa97c04cc1a68c8f040e71fb84?tpId=13&tqId=11214&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
[Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/)

```python
class Solution:
    def Serialize(self, root):
        # write code here
        def doit(node):
            if node: # 先序遍历的方式
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                vals.append('#')
        vals = []
        doit(root)
        return ' '.join(vals)

    def Deserialize(self, s):
        # write code here
        def doit():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node
        vals = iter(s.split())
        return doit()
```



#### 二叉平衡树中的第k小数

[二叉搜索树中的第k大结点](https://www.nowcoder.com/practice/ef068f602dde4d28aab2b210e859150a?tpId=13&tqId=11215&tPage=4&rp=4&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
[Leetcode 230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
思路：BST的中序遍历就是一个有序数组，需要注意到Leetcode中限制了k在[1,树结点个数]而牛客网没有，所以需要考虑k的值有没有超出

```python
class Solution: # 中序遍历 + cnt 计数
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        stack = []
        node = pRoot
        while node:
            stack.append(node)
            node = node.left
        cnt = 1
        while(stack and cnt<=k):
            node = stack.pop()
            right = node.right
            while right:
                stack.append(right)
                right = right.left
            cnt+=1
        if node and k==cnt-1:
            return node
        return None
   
```

#### [重建二叉树](https://www.nowcoder.com/practice/8a19cbe657394eeaac2f6ea9b0f6fcf6?tpId=13&tqId=11157&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
[Leetcode 105. Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)
根据先序、中序来构建二叉树

```python
class Solution(object):
    def buildTree(self, pre, tin):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if pre==[]:
            return None
        val = pre[0]
        idx = tin.index(val)
        ltin = tin[0:idx]
        rtin = tin[idx+1:]
        lpre = pre[1:1+len(ltin)]
        rpre = pre[1+len(ltin):]
        root = TreeNode(val)
        root.left = self.buildTree(lpre,ltin)
        root.right = self.buildTree(rpre,rtin)
        return root
```

[Leetcode 106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)
根据中序和后序构建二叉树

```python
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if postorder == []:
            return None
        val = postorder[-1]
        idx = inorder.index(val)
        lin = inorder[0:idx]
        rin = inorder[idx+1:]
        lpos = postorder[0:len(lin)]
        rpos = postorder[len(lin):-1]
        root = TreeNode(val)
        root.left = self.buildTree(lin,lpos)
        root.right = self.buildTree(rin,rpos)
        return root
```





#### [和为S的连续正数序列](https://www.nowcoder.com/practice/c451a3fd84b64cb19485dad758a55ebe?tpId=13&tqId=11194&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
思考：

解法：1.穷举法，2.[滑动窗口法](https://blog.csdn.net/shansusu/article/details/50211519)

 ，3.数字规律法(往下看)

S%奇数==0 或者S%偶数==偶数／2 就说明有这个连续序列，但是注意是正数序列，可能会出现越界情况

```python
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        k = 2
        ret = []
        for k in range(2,tsum):
            if k%2==1 and tsum%k==0:
                tmp = []
                mid = tsum/k
                if mid-k/2>0:
                    for i in range(mid-k/2,mid+k/2+1):
                        tmp.append(i)
                    ret.append(tmp[:])
            elif k%2==0 and (tsum%k)*2==k:
                mid = tsum/k
                tmp = []
                if mid-k/2+1>0:
                    for i in range(mid-k/2+1,mid+k/2+1):
                        tmp.append(i)
                    ret.append(tmp[:])
        ret.sort()
        return ret
```

#### [左旋转字符串](https://www.nowcoder.com/practice/12d959b108cb42b1ab72cef4d36af5ec?tpId=13&tqId=11196&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
思考：需要先K= K%len(S)

```python
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if s == '':
            return s
        n = n%len(s)
        return s[n:]+s[0:n]
```

#### 数字在排序数组中出现的次数
[数字在排序数组中出现的次数](https://www.nowcoder.com/practice/70610bf967994b22bb1c26f9ae901fa2?tpId=13&tqId=11190&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：原来是可以用hash做的，但是因为是排序数组，所以可以用二分查找

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        start = 0
        end = len(data)-1
        while(start<=end):
            mid = (start+end)/2
            if data[mid]==k:
                cnt = 0
                tmp = mid
                while(tmp>=0 and data[tmp]==k):
                    cnt+=1
                    tmp-=1
                tmp = mid+1
                while(tmp<len(data) and data[tmp]==k):
                    cnt+=1
                    tmp+=1
                return cnt
            elif data[mid]>k:
                end = mid-1
            else:
                start = mid+1
        return 0
```

#### [数组中只出现一次的数字](https://www.nowcoder.com/practice/e02fdb54d7524710a7d664d082bb7811?tpId=13&tqId=11193&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字
思考：用hash；或者位运算
首先利用0 ^ a = a; a^a = 0的性质
两个不相等的元素在位级表示上必定会有一位存在不同，
将数组的所有元素异或得到的结果为不存在重复的两个元素异或的结果，
据异或的结果1所在的最低位，把数字分成两半，每一半里都还有一个出现一次的数据和其他成对出现的数据,
问题就转化为了两个独立的子问题“数组中只有一个数出现一次，其他数都出现了2次，找出这个数字”。

```python
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        ans,a1,a2,flag= 0,0,0,1
        for num in array:
            ans = ans ^ num
        while(ans):
            if ans%2 == 0:
                ans = ans >>1 
                flag = flag <<1
            else:
                break
        for num in array:
            if num & flag:
                a1 = a1 ^ num
            else:
                a2 = a2 ^ num
        return a1,a2
```

#### [翻转单词顺序列](https://www.nowcoder.com/practice/3194a4f4cf814f63919d0790578d51f3?tpId=13&tqId=11197&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        ret = s.split(" ")
        ret.reverse()
        return ' '.join(ret)
```

#### [和为S的两个数字](https://www.nowcoder.com/practice/390da4f7a00f44bea7c2f3d19491311b?tpId=13&tqId=11195&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
hash

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        memorys= {}
        ret = []
        for num in array:
            if tsum-num in memorys:
                if ret == []:
                    ret = [tsum-num,num]
                elif ret and ret[0]*ret[1]>(tsum-num)*num:
                    ret = [tsum-num,num]
            else:
                memorys[num] = 1
        return ret

```

#### [顺时针打印矩阵](https://www.nowcoder.com/practice/9b4c81a02cd34f76be2659fa0d54342a?tpId=13&tqId=11172&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        ans=[]
        m=len(matrix)
        if m==0:
            return ans
        n=len(matrix[0])
		# 边界范围
        upper_i =0;lower_i=m-1;left_j=0;right_j=n-1
        # 整体计数，确定何时结束
        num=1
        # 轮询
        i=0;j=0
        # 轮询方向
        right_pointer=1
        down_pointer=0
        while(num<=m*n):
            ans.append(matrix[i][j])
            if right_pointer==1: # 判断朝向
                if j<right_j: # 没有到边界
                    j=j+1
                else: # 到边界
                    right_pointer=0
                    down_pointer=1
                    upper_i = upper_i+1
                    i = i+1
            elif down_pointer == 1:
                if i<lower_i:
                    i = i+1
                else:
                    right_pointer=-1
                    down_pointer=0
                    right_j = right_j -1
                    j = j-1
            elif right_pointer ==-1:
                if j > left_j:
                    j=j-1
                else:
                    right_pointer=0
                    down_pointer=-1
                    lower_i =lower_i-1
                    i = i-1
            elif down_pointer == -1:
                if i > upper_i:
                    i=i-1
                else:
                    right_pointer=1
                    down_pointer=0
                    left_j = left_j +1
                    j = j+1
            num=num+1
        return ans
```

#### [数据流中的中位数](https://www.nowcoder.com/practice/9be0172896bd43948f8a32fb954e1be1?tpId=13&tqId=11216&tPage=4&rp=4&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
[Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/description/)

```python
from heapq import *
#heapq默认就是最小堆，large 是最小堆，但是都是正数所以比较大的最小堆，small是最大堆，但是都是负数，所以是比较小的最大堆；
#1.首先将large 跟当前num做判断，然后弹出最小的那个，放入最大堆；这里large的个数并没有增加，只是增加了小的，
#2.如果large个数比较小，就在向large里面放，
#3.中位数如果是奇数个，那么它一定在large里面；如果是偶数个，那就求个平均数；
class MedianFinder:

    def __init__(self):
        self.heaps = [], []

    def addNum(self, num):
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small):
            heappush(large, -heappop(small))

    def findMedian(self):
        small, large = self.heaps
        if len(large) > len(small):
            return float(large[0])
        return (large[0] - small[0]) / 2.0
```

#### [滑动窗口的最大值](https://blog.csdn.net/qq_20141867/article/details/81088705)
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
思考：假设当前窗口起始位置为start,结束位置为end，我们要构造一个stack, 使得stack[0]为区间[start,end]的最大值。

窗口的滑动过程中数字的进出类似一个队列中元素的出队入队，这里我们采用一个队列queue存储**可能成为最大值的元素下标**(之所以队列中存元素下标而不是元素值本身，是因为队列并不存储所有元素，而我们需要知道什么时候队首元素已经离开滑动窗口)。当遇到一个新数时，将它与队尾元素比较，如果大于队尾元素，则丢掉队尾元素，继续重复比较，**直到新数小于队尾元素，或者队列为空为止**，将新数下标放入队列。同时需要根据滑动窗口的移动**判断队首元素是否已经离开**。

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        # 存放可能是最大值的下标
        maxqueue = []
        # 存放窗口中最大值
        maxlist = []
        n = len(num)
        # 参数检验
        if n == 0 or size == 0 or size > n:
            return maxlist
        for i in range(n):
            # 判断队首下标对应的元素是否已经滑出窗口
            if len(maxqueue) > 0 and i - size >= maxqueue[0]:
                maxqueue.pop(0)
            # 判断新入的是否比第0 index的要大，如果是就要全部替换掉；
            while len(maxqueue) > 0 and num[i] > num[maxqueue[-1]]:
                maxqueue.pop()
            maxqueue.append(i)
            if i >= size - 1:
                maxlist.append(num[maxqueue[0]])
        return maxlist
```

#### [用两个栈实现队列](https://www.nowcoder.com/practice/54275ddae22f475981afa2244dd448c6?tpId=13&tqId=11158&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：栈FILO，队列FIFO

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)

    def pop(self):
        # return xx
        if len(self.stack2):
            return self.stack2.pop()
        while(self.stack1):
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()
```

#### [丑数](https://www.nowcoder.com/practice/6aa9e04fc3794f68acf8778237ba065b?tpId=13&tqId=11186&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
只包含因子2、3和5的数称作丑数（Ugly Number），求按从小到大的顺序的第N个丑数。因为丑数只包含质因子2，3，5，假设我们已经有n-1个丑数，按照顺序排列，且第n-1的丑数为M。那么第n个丑数一定是由这n-1个丑数分别乘以2，3，5，得到的所有大于M的结果中，最小的那个数。

事实上我们不需要每次都计算前面所有丑数乘以2，3，5的结果，然后再比较大小。因为在已存在的丑数中，一定存在某个数T2T2，在它之前的所有数乘以2都小于已有丑数，而T2×2T2×2的结果一定大于M，同理，也存在这样的数T3，T5T3，T5，我们只需要标记这三个数即可。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index == 0:
            return 0
        # 1作为特殊数直接保存
        baselist = [1]
        min2 = min3 = min5 = 0
        curnum = 1
        while curnum < index:
            minnum = min(baselist[min2] * 2, baselist[min3] * 3, baselist[min5] * 5)
            baselist.append(minnum)
            # 找到第一个乘以2的结果大于当前最大丑数M的数字，也就是T2
            while baselist[min2] * 2 <= minnum:
                min2 += 1
            # 找到第一个乘以3的结果大于当前最大丑数M的数字，也就是T3
            while baselist[min3] * 3 <= minnum:
                min3 += 1
            # 找到第一个乘以5的结果大于当前最大丑数M的数字，也就是T5
            while baselist[min5] * 5 <= minnum:
                min5 += 1
            curnum += 1
        return baselist[-1]
```

#### [两个链表的第一个公共结点](https://www.nowcoder.com/practice/6ab1d9a29e88450685099d45c9e31e46?tpId=13&tqId=11189&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：设链表pHead1的长度为a,到公共结点的长度为l1；链表pHead2的长度为b,到公共结点的长度为l2，有a+l2 = b+l1

```python
### 思路：第一次循环 while(pa!=pb)会将长度自动对齐；
### 第二次循环，就正常去判断了；一个trick
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if pHead1== None or pHead2 == None:
            return None
        pa = pHead1
        pb = pHead2 
        while(pa!=pb):
            pa = pHead2 if pa is None else pa.next
            pb = pHead1 if pb is None else pb.next
        return pa
```

#### [第一个只出现一次的字符](https://www.nowcoder.com/practice/1c82e8cf713b4bbeb2a5b31cf5b0417c?tpId=13&tqId=11187&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：用dict 最多256个来空间换时间

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if s=='':
            return -1
        ret = [0,-1]
        chars = {}
        for i,c in enumerate(s):
            if c in chars:
                chars[c][0] = chars[c][0] +1
            else:
                chars[c] =[1,i]
        for k,v in chars.items():
            if v[0] ==1:
                if ret[1] ==-1:
                    ret = v
                elif v[1] <= ret[1]:
                    ret = v
        return ret[1]
```

#### [数组中的逆序对](https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=13&tqId=11188&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
方法1：思考：这边的python会超时，但是思路是对的
时间复杂度O(nlogn),空间复杂度O(n)。
先将数组逆转，构建一个新的数组L，将num二分插入到L中，所插入的位置i，代表有i个数字比当前这个数字小

```python
import bisect
class Solution:
    def InversePairs(self, data):
        data.reverse()
        L = []
        ret = 0
        for d in data:
            pos = bisect.bisect_left(L,d)
            L.insert(pos,d)
            ret+= pos
            ret = ret % 1000000007
        return ret % 1000000007
```

方法2：[归并排序的思路](https://blog.csdn.net/dlengong/article/details/7594919)


#### [连续子数组的最大和](https://www.nowcoder.com/practice/459bd355da1549fa8a49e350bf3df484?tpId=13&tqId=11183&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

可以用动态规划的思想来分析这个问题。如果用函数f(i)表示以第i个数字结尾的子数组的最大和，那么我们需要求出max[f(i)]，其中0 <= i < n。我们可用如下边归公式求f(i):

![这里写图片描述](https://img-blog.csdn.net/20150703073321619)

这个公式的意义：当以第i-1 个数字结尾的子数组中所有数字的和小于0时，如果把这个负数与第i个数累加，得到的结果比第i个数字本身还要小，所以这种情况下以第i个数字结尾的子数组就是第i个数字本身。如果以第i-1 个数字结尾的子数组中所有数字的和大于0 ,与第i 个数字累加就得到以第i个数字结尾的子数组中所有数字的和。

```python
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if len(array)==1:
            return array[0]
        cur = pos = array[0]
        for i in range(1,len(array)):
            pos = max(pos+array[i],array[i])
            cur = max(cur,pos)
        return cur
```



#### [最小的K个数](https://www.nowcoder.com/practice/6a296eb82cf844ca8539b57c23e6e9bf?tpId=13&tqId=11182&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

```python
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        heaps = []
        ret = []
        for num in tinput:
            heapq.heappush(heaps,num)
        if k>len(heaps):
            return []
        for i in range(k):
            ret.append(heapq.heappop(heaps))
        return ret
```

#### [数组中出现次数超过一半的数字](https://www.nowcoder.com/practice/e8a1b01a2df14cb2b228b30ee6a92163?tpId=13&tqId=11181&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：摩尔投票法

```python
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if numbers == []:
            return 0
        val,cnt = None,0
        for num in numbers:
            if cnt==0:
                val,cnt = num,1
            elif val == num:
                cnt+=1
            else:
                cnt-=1
        return val if numbers.count(val)*2>len(numbers) else 0
```

#### [从1到n整数中1出现的次数](https://www.nowcoder.com/practice/bd7f978302044eee894445e244c7eee6?tpId=13&tqId=11184&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
思考：可以从n的每个位上入手，pos来记录位，ans来记录当前1的个数，last来记录前面的数（这样讲好复杂，举个例子好了）
xxxxYzzzz （假设9位）
在Y上1的个数由xxxx,zzzz和Y来决定
首先至少有xxxx0000个
其次看Y
如果Y大于1那么会多了10000个
如果Y等于1那么会多了（zzzz+1）个

**[举例：](https://zhuanlan.zhihu.com/p/39712892)**

设N = abcde ,其中abcde分别为十进制中各位上的数字。
如果要计算百位上1出现的次数，它要受到3方面的影响：百位上的数字，百位以下（低位）的数字，百位以上（高位）的数字。
① 如果百位上数字为0，百位上可能出现1的次数由更高位决定。比如：12013，则可以知道百位出现1的情况可能是：100~199，1100~1199,2100~2199，，...，11100~11199，一共1200个。可以看出是由更高位数字（12）决定，并且等于更高位数字（12）乘以 当前位数（100）。
② 如果百位上数字为1，百位上可能出现1的次数不仅受更高位影响还受低位影响。比如：12113，则可以知道百位受高位影响出现的情况是：100~199，1100~1199,2100~2199，，....，11100~11199，一共1200个。和上面情况一样，并且等于更高位数字（12）乘以 当前位数（100）。但同时它还受低位影响，百位出现1的情况是：12100~12113,一共114个，等于低位数字（113）+1。
③ 如果百位上数字大于1（2~9），则百位上出现1的情况仅由更高位决定，比如12213，则百位出现1的情况是：100~199,1100~1199，2100~2199，...，11100~11199,12100~12199,一共有1300个，并且等于更高位数字+1（12+1）乘以当前位数（100）。
——参考牛客网@藍裙子的百合魂

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        if n<1:  return 0
        if n==1: return 1
        last,ans,pos = 0,0,1 # pos来记录位，ans来记录当前1的个数，last来记录前面的数(右边的数)
        while(n):
            num = n%10
            n = n/10
            ans += pos*n
            if num>1:
                ans+=pos
            elif num==1:
                ans+=(last+1)
            last = last+num*pos
            pos*=10
        return ans
```

#### [把数组排成最小的数](https://www.nowcoder.com/practice/8fecd3f8ba334add803bf2a06af1b993?tpId=13&tqId=11185&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
str 化

```python
链接：https://www.nowcoder.com/questionTerminal/8fecd3f8ba334add803bf2a06af1b993
来源：牛客网
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:return ""
        numbers = list(map(str,numbers))
        numbers.sort(cmp=lambda x,y:cmp(x+y,y+x))
        return '0' if numbers[0]=='0' else ''.join(numbers)
```

#### [数组中重复的数字](https://www.nowcoder.com/practice/623a5ac0ea5b4e5f95552655361ae0a8?tpId=13&tqId=11203&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中第一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

返回描述：

如果数组中有重复的数字，函数返回true，否则返回false。

如果数组中有重复的数字，把重复的数字放到参数duplication[0]中。（ps:duplication已经初始化，可以直接赋值使用。）

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        dup = dict()
        for num in numbers:
            if num not in dup:
                dup[num] = True
            else:
                duplication[0]=num
                return True
```

#### [构造乘积数组](https://www.nowcoder.com/practice/94a4d381a68b47b7a8bed86f2975db46?tpId=13&tqId=11204&tPage=2&rp=2&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
B[i]=A[0]*A[1]*…*A[i-1]\*A[i+1]*…*A[n-1]
思考：分为下三角和上三角DP计算B
下三角
上三角（从最后往前）

```python
class Solution:
    def multiply(self, A):
        # write code here
        size = len(A)
        B = [1]*size
        for i in range(1,size):
            B[i] = B[i-1]*A[i-1]
        tmp = 1
        for i in range(size-2,-1,-1):
            tmp = tmp*A[i+1]
            B[i] = B[i]*tmp
        return B
```

#### [二维数组中的查找](https://www.nowcoder.com/practice/abc3fe2ce8e146608e868a70efebf62e?tpId=13&tqId=11154&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
思考：从0,n-1出开始，小了往下，大了往左

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array)==0 or len(array[0])==0:
            return False
        i = 0
        j = len(array[0])-1
        while(i<len(array) and j>=0):
            if array[i][j]==target:
                return True
            elif array[i][j]>target:
                j-=1
            else:
                i+=1
        return False
```

#### [扑克牌顺子](https://www.nowcoder.com/practice/762836f4d43d43ca9deb273b3de8e1f4?tpId=13&tqId=11198&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)…他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子…..LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0。

```python
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return 0
        numbers.sort()
        zeros = numbers.count(0)
        for i, v in enumerate(numbers[:-1]):
            if v!=0:
                if numbers[i+1]==v:
                    return False
                zeros -= (numbers[i+1]-numbers[i]-1)
                if zeros<0:
                    return False
        return True
```

#### [孩子们的游戏](https://www.nowcoder.com/practice/f78a359491e64a50bce2d89cff857eb6?tpId=13&tqId=11199&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

每年六一儿童节,我都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

如果没有小朋友，请返回-1



为了讨论方便，先把问题稍微改变一下，并不影响原意：
我们知道第一个人（编号一定是（m-1）) 出列之后，剩下的n-1个人组成了一个新的约瑟夫环（以编号为k=m mod n的人开始）：

> k, k+1, k+2, … n-2,n-1,0,1,2,… k-2

并且从k开始报0。
我们把他们的编号做一下转换：

```
k --> 0
k+1 --> 1
k+2 --> 2
...
...
k-2 --> n-2
```

变换后就完完全全成为了（n-1）个人报数的子问题，假如我们知道这个子问题的解：例如x是最终的胜利者，那么根据上面这个表把这个x变回去不刚好就是n个人情况的解吗？！！变回去的公式很简单，相信大家都可以推出来：x’=(x+k) mod n

如何知道（n-1）个人报数的问题的解？对，只要知道（n-2）个人的解就行了。（n-2）个人的解呢？当然是先求（n-3）的情况 —- 这显然就是一个倒推问题！好了，思路出来了，下面写递推公式：

令f表示i个人玩游戏报m退出最后胜利者的编号，最后的结果自然是f[n]

递推公式

> 让f[i]为i个人玩游戏报m退出最后的胜利者的编号，最后的结果自然是f[n]
>
> f[1] = 0;
>
> f[i] = (f[i - 1] + m) mod i;

有了这个公式，我们要做的就是从1-n顺序算出f的数值，最后结果是f[n]。

```python
# 正经思路
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n<1: return -1
        final,start = -1,0
        cnt = [i for i in range(n)]
        while cnt:
            k = (start+m-1)%n
            final = cnt.pop(k)
            n-=1
            start = k
        return final 
    
# 数学思路
class Solution{
def LastRemaining_Solution(self,n,m):
    if n <1 : return -1
    ans=0;
    for i in range(1,n+1):
       ans = (ans + m) %i
    return ans
}
#include<cstdio>
int main(){
    int n,m;
    while(scanf("%d %d",&n,&m)==2&&n&&m){
        int ans = 0;
        for(int i = 1;i <= n;++i){
            ans = (ans + m) % i;
        }
        printf("总人数:%d 每次出列的人喊的号数:%d 最后一个出列的人的序号:%d\n",n,m,ans+1);
    }
    return 0;
}

```



#### [正则表达式匹配](https://www.nowcoder.com/practice/45327ae22b7b413ea21df13ee7d6429c?tpId=13&tqId=11205&tPage=3&rp=3&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
请实现一个函数用来匹配包括'.'和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（包含0次）。  在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串”aaa”与模式”a.a”和”ab*ac*a”匹配，但是与”aa.a”和”ab\*a”均不匹配
思考：
第一种情况，当p == ” 时 return s==”
当len(p)==1时 要满足len(s)==1 AND (p[0]==s[0] OR p[0] == ‘.’)
当len(p)>1时，要讨论p[1] 是不是为’*’ ，因为如果p[1]==’*’ 时候 可能会是p[2:] 和 s 匹配情况
但当p[1]!=’*’ 时候 意味着 必须要关注是否 p[0]==s[0] 或者 p[0]==’.’
那么这两个可以合并为
IF len(p) == 0 or p[1]!=’*’
返回 len(s) AND match(p[1:],s[1:]) AND (p[0]==s[0] OR p[0] == ‘.’)
然后最复杂的一种情况p[1] == ‘*’
p = ‘b*bbacd’ s = ‘bbbbbacd’
很明显的是如果p[0]!=s[0] 且 p[0]!=’.’ 那么 看p[2:] 和 s 的匹配情况
如果p[0] == s[0] 或者 p[0] == ‘.’ ， 可以判断p[2:] 和 s[1:] … p[2:] 和 s[2:] … p[2:] 和 s[3:] … 搞个循环 就可以

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def __init__(self):
        self.dic = {}
    def match(self, s, p):
        # write code here
        if (s,p) in self.dic:
            return self.dic[(s,p)]
        if p == '':
            return s==''
        if len(p)==1 or p[1]!='*':
            self.dic[(s[1:],p[1:])] = self.match(s[1:],p[1:])
            return len(s)>0 and (p[0]=='.' or p[0]==s[0]) and self.dic[(s[1:],p[1:])]
        while(len(s) and (p[0]=='.' or p[0]==s[0])):
            self.dic[(s,p[2:])] = self.match(s,p[2:])
            if self.match(s[:],p[2:]):
                return True
            s = s[1:]
        self.dic[(s,p[2:])] = self.match(s,p[2:])
        return self.dic[(s,p[2:])]
```

