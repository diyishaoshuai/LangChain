// index.js
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import * as dotenv from 'dotenv';

dotenv.config();

// 初始化 OpenAI GPT-3.5 模型
const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY, // 使用 OpenAI API 密钥
  model: "gpt-3.5-turbo", // 使用 GPT-3.5-turbo
  temperature: 0.7,
  // 自定义兼容代理的基础地址
  configuration: {
    baseURL: "https://api.chatanywhere.tech/v1",
  },
});

async function main() {
  console.log("🧠 正在通过OpenAI兼容接口调用DeepSeek...");
  try {
    const response = await model.invoke([
      new HumanMessage("用中文简单介绍下自己，不超过30字。")
    ]);
    console.log("\n💬 回复：", response.content);
  } catch (error) {
    console.error('❌ 请求失败：', error.message);
  }
}   
main();