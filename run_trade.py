import requests
import time

# =========================
# 基本配置
# =========================

BASE_URL = "https://lumid.market/trading"
API_TOKEN = "086059ced9ac841c1cf1ce82a2b571d351266ef7588a1cff775c31aee8509f2f"

SYMBOL = "STRN"       # 改成你要交易的标的
LOOKBACK = 3          # 3日动量
INTERVAL = 60         # 每60秒执行一次

HEADERS = {
    "X-API-Token": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

price_history = []


# =========================
# API函数
# =========================

def check_account():
    url = f"{BASE_URL}/api/custom/trading/account"
    r = requests.get(url, headers=HEADERS)
    return r.json()


def get_price(symbol):
    url = f"{BASE_URL}/api/custom/trading/price"
    r = requests.get(url, headers=HEADERS, params={"symbol": symbol})
    return r.json()


def submit_order(symbol, direction, volume):
    url = f"{BASE_URL}/api/custom/trading/order"
    payload = {
        "symbol": symbol,
        "direction": direction,
        "volume": volume
    }
    r = requests.post(url, headers=HEADERS, json=payload)
    return r.json()


def get_position_size(symbol):
    url = f"{BASE_URL}/api/custom/trading/positions"
    r = requests.get(url, headers=HEADERS)
    data = r.json()

    if data["ret_code"] != 0:
        return 0

    for pos in data["data"]:
        if pos["symbol"] == symbol:
            return pos["position_size"]

    return 0


# =========================
# 动量策略逻辑
# =========================

def generate_signal():
    if len(price_history) < LOOKBACK + 1:
        return None

    past_price = price_history[-(LOOKBACK + 1)]
    current_price = price_history[-1]

    ret = current_price / past_price - 1

    print(f"3-period return: {ret:.4f}")

    if ret > 0:
        return "Buy"
    elif ret < 0:
        return "Sell"
    else:
        return None


# =========================
# 主循环
# =========================

if __name__ == "__main__":

    print("Checking account...")

    account = check_account()
    if account["ret_code"] != 0:
        print("Account error:", account)
        exit()

    print("Account connected successfully.")
    print("Starting strategy...\n")

    while True:

        try:
            price_data = get_price(SYMBOL)

            if price_data["ret_code"] != 0:
                print("Price error:", price_data)
                time.sleep(INTERVAL)
                continue

            current_price = price_data["data"]["price"]
            price_history.append(current_price)

            print(f"Current price: {current_price}")

            signal = generate_signal()

            position = get_position_size(SYMBOL)
            print(f"Current position: {position}")

            if signal == "Buy" and position == 0:
                print(">>> Executing BUY")
                order = submit_order(SYMBOL, "Buy", 1)
                print("Order result:", order)

            elif signal == "Sell" and position > 0:
                print(">>> Executing SELL (close position)")
                order = submit_order(SYMBOL, "Sell", position)
                print("Order result:", order)

            else:
                print("No trade executed.")

            print("-" * 40)

            time.sleep(INTERVAL)

        except Exception as e:
            print("Unexpected error:", e)
            time.sleep(INTERVAL)