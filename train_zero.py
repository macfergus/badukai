import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-input', required=True)
    parser.add_argument('--bot', required=True)
    parser.add_argument('--bot-out', required=True)
    args = parser.parse_args()

    bot_file = args.bot
    bot = badukai.bots.load_bot(bot_file)

    game_records = badukai.bots.zero.load_game_records(open(args.games_input))
    bot.train(game_records)

    badukai.bots.save_bot(bot, args.bot_out)


if __name__ == '__main__':
    main()
