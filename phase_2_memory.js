// 使用新版模块化导入
import { ChatOpenAI } from "@langchain/openai";
// ConversationChain 与 BufferMemory 需要从主包导入，社区包未导出该子路径
import { HumanMessage } from "@langchain/core/messages";
import * as dotenv from 'dotenv';
dotenv.config();

// 1. 初始化模型
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  temperature: 0.5, // 适当降低随机性，让对话更稳定
});

// 2. 简单的对话记忆：用数组手动维护消息历史
const history = [];

async function chat(input) {
  history.push(new HumanMessage(input));
  const aiMsg = await model.invoke(history);
  history.push(aiMsg);
  return aiMsg;
}

async function main() {
  console.log('=== 开始对话 ===\n');

  // 第一轮：告诉AI你的名字
  const response1 = await chat("你好，请叫我小王。");
  console.log(`你: 你好，请叫我小王。`);
  console.log(`AI: ${response1.content}\n`);

  // 第二轮：AI应该记得“小王”
  const response2 = await chat("你还记得我的名字吗？");
  console.log(`你: 你还记得我的名字吗？`);
  console.log(`AI: ${response2.content}\n`);

  // 第三轮：基于上下文的连续提问
  const response3 = await chat("用我的名字造一个句子。");
  console.log(`你: 用我的名字造一个句子。`);
  console.log(`AI: ${response3.content}\n`);

  // 第四轮：更复杂的上下文依赖问题
  const response4 = await chat("我们最开始是怎么打招呼的？");
  console.log(`你: 我们最开始是怎么打招呼的？`);
  console.log(`AI: ${response4.content}`);
}

main().catch(console.error);