function multiple = round_up_to_multiple( num, multiple_of )
    multiple = num - 1 - mod(num - 1, multiple_of) + multiple_of;
end

