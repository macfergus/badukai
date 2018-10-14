import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-input', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--bot', required=True)
    parser.add_argument('--bot-out', required=True)
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    with badukai.io.get_input(args.games_input) as games_file:
        game_records = badukai.bots.zero.load_game_records(open(games_file))

    print('Training on {} game records'.format(len(game_records)))
    bot.train(game_records, num_epochs=args.epochs)

    with badukai.io.open_output_filename(args.bot_out) as output_filename:
        badukai.bots.save_bot(bot, output_filename)


if __name__ == '__main__':
    main()
