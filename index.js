// index.js
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import * as dotenv from 'dotenv';

dotenv.config();

const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY, // 确保是DeepSeek的Key
  model: "deepseek-chat",
  temperature: 0.7,
  configuration: {
    baseURL: "https://api.deepseek.com/v1", // 关键：指向DeepSeek
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