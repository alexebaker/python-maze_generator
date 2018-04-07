from corn_maze_generator.field import Field


def main(size=(4, 2), diameter=5, complexity=0.75, density=0.75):
    field = Field(size=size, diameter=diameter, complexity=complexity, density=density)
    field.generate()
    field.display()
    return


if __name__ == '__main__':
    pass
