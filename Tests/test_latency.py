import asyncio
import time
from openai import AsyncOpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  
client = AsyncOpenAI() 

async def measure_latency():
    times = []
    
    for _ in range(20):
        start = time.perf_counter()
        
        await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Reply only 0 or 1."},
                {"role": "user", "content": "I want a red truck"}
            ],
            max_tokens=1,
            temperature=0,
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"{elapsed:.0f}ms")
    
    times.sort()
    print(f"\nmedian:  {times[10]:.0f}ms")
    print(f"p90:     {times[18]:.0f}ms")   # 90% of calls faster than this
    print(f"p99:     {times[19]:.0f}ms")   # 99% of calls faster than this
    print(f"worst:   {times[-1]:.0f}ms")

asyncio.run(measure_latency())