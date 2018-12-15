import argparse

import badukai


def chunk(generator, chunk_size):
    unit = []
    for item in generator:
        unit.append(item)
        if len(unit) >= chunk_size:
            yield unit
            unit = []
    if unit:
        yield unit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-input', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--bot', required=True)
    parser.add_argument('--bot-out', required=True)
    parser.add_argument('--chunk-size', type=int, default=825)
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    for i in range(args.epochs):
        print('Epoch {}/{}...'.format(i + 1, args.epochs))
        with badukai.io.get_input(args.games_input) as games_file:
            game_records = badukai.bots.zero.generate_game_records(
                open(games_file))
            for games in chunk(game_records, args.chunk_size):
                print('Training on {} game records'.format(len(games)))
                bot.train(games, num_epochs=1)

    with badukai.io.open_output_filename(args.bot_out) as output_filename:
        badukai.bots.save_bot(bot, output_filename)


if __name__ == '__main__':
    main()
