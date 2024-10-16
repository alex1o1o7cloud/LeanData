import Mathlib

namespace NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l1416_141633

def f (x : ℝ) : ℝ := |x|

theorem abs_is_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l1416_141633


namespace NUMINAMATH_CALUDE_symmetric_lines_l1416_141699

/-- Given two lines symmetric about y = x, if one line is y = 2x - 3, 
    then the other line is y = (1/2)x + (3/2) -/
theorem symmetric_lines (x y : ℝ) : 
  (y = 2 * x - 3) ↔ 
  (∃ (x' y' : ℝ), y' = (1/2) * x' + (3/2) ∧ 
    (x + x') / 2 = (y + y') / 2 ∧
    y = x) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_lines_l1416_141699


namespace NUMINAMATH_CALUDE_fraction_of_66_l1416_141635

theorem fraction_of_66 (x : ℚ) (h : x = 22.142857142857142) : 
  ((((x + 5) * 7) / 5) - 5) = 66 * (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_66_l1416_141635


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1416_141664

/-- The coefficient of x^2 in the expansion of (1+2x)^5 is 40 -/
theorem coefficient_x_squared_in_expansion : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * 2^k * if k = 2 then 1 else 0) = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1416_141664


namespace NUMINAMATH_CALUDE_amys_remaining_money_is_56_04_l1416_141670

/-- Calculates the amount of money Amy has left after her purchases --/
def amys_remaining_money (initial_amount : ℚ) (doll_price : ℚ) (doll_count : ℕ)
  (board_game_price : ℚ) (board_game_count : ℕ) (comic_book_price : ℚ) (comic_book_count : ℕ)
  (board_game_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let doll_cost := doll_price * doll_count
  let board_game_cost := board_game_price * board_game_count
  let comic_book_cost := comic_book_price * comic_book_count
  let discounted_board_game_cost := board_game_cost * (1 - board_game_discount)
  let total_before_tax := doll_cost + discounted_board_game_cost + comic_book_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  initial_amount - total_after_tax

/-- Theorem stating that Amy has $56.04 left after her purchases --/
theorem amys_remaining_money_is_56_04 :
  amys_remaining_money 100 1.25 3 12.75 2 3.50 4 0.10 0.08 = 56.04 := by
  sorry

end NUMINAMATH_CALUDE_amys_remaining_money_is_56_04_l1416_141670


namespace NUMINAMATH_CALUDE_maries_trip_l1416_141606

theorem maries_trip (total_distance : ℚ) 
  (h1 : total_distance / 4 + 15 + total_distance / 6 = total_distance) : 
  total_distance = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_maries_trip_l1416_141606


namespace NUMINAMATH_CALUDE_three_m_plus_n_equals_46_l1416_141669

theorem three_m_plus_n_equals_46 (m n : ℕ) 
  (h1 : m > n) 
  (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 
  3 * m + n = 46 := by
sorry

end NUMINAMATH_CALUDE_three_m_plus_n_equals_46_l1416_141669


namespace NUMINAMATH_CALUDE_melanie_turnips_count_l1416_141694

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The total number of turnips Melanie and Benny grew together -/
def total_turnips : ℕ := 252

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := total_turnips - benny_turnips

theorem melanie_turnips_count : melanie_turnips = 139 := by
  sorry

end NUMINAMATH_CALUDE_melanie_turnips_count_l1416_141694


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1416_141656

/-- Represents a pitcher with a capacity and an amount of orange juice --/
structure Pitcher where
  capacity : ℚ
  juice : ℚ
  h_juice_nonneg : 0 ≤ juice
  h_juice_le_capacity : juice ≤ capacity

/-- Calculates the fraction of orange juice in the final mixture --/
def orange_juice_fraction (p1 p2 : Pitcher) : ℚ :=
  (p1.juice + p2.juice) / (p1.capacity + p2.capacity)

theorem orange_juice_mixture_fraction :
  let p1 : Pitcher := ⟨500, 250, by norm_num, by norm_num⟩
  let p2 : Pitcher := ⟨700, 420, by norm_num, by norm_num⟩
  orange_juice_fraction p1 p2 = 67 / 120 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1416_141656


namespace NUMINAMATH_CALUDE_vocabulary_increase_l1416_141679

def words_per_day : ℕ := 10
def years : ℕ := 2
def days_per_year : ℕ := 365
def initial_vocabulary : ℕ := 14600

theorem vocabulary_increase :
  let total_new_words := words_per_day * years * days_per_year
  let final_vocabulary := initial_vocabulary + total_new_words
  let percentage_increase := (total_new_words : ℚ) / (initial_vocabulary : ℚ) * 100
  percentage_increase = 50 := by sorry

end NUMINAMATH_CALUDE_vocabulary_increase_l1416_141679


namespace NUMINAMATH_CALUDE_smallest_perfect_square_tiling_l1416_141615

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length. -/
structure Square where
  side : ℕ

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The area of a square. -/
def Square.area (s : Square) : ℕ := s.side * s.side

/-- A rectangle fits in a square if its width and height are both less than or equal to the square's side length. -/
def fits_in (r : Rectangle) (s : Square) : Prop := r.width ≤ s.side ∧ r.height ≤ s.side

/-- A square is perfectly tiled by rectangles if the sum of the areas of the rectangles equals the area of the square. -/
def perfectly_tiled (s : Square) (rs : List Rectangle) : Prop :=
  (rs.map Rectangle.area).sum = s.area

theorem smallest_perfect_square_tiling :
  ∃ (s : Square) (rs : List Rectangle),
    (∀ r ∈ rs, r.width = 3 ∧ r.height = 4) ∧
    perfectly_tiled s rs ∧
    (∀ r ∈ rs, fits_in r s) ∧
    rs.length = 12 ∧
    s.side = 12 ∧
    (∀ (s' : Square) (rs' : List Rectangle),
      (∀ r ∈ rs', r.width = 3 ∧ r.height = 4) →
      perfectly_tiled s' rs' →
      (∀ r ∈ rs', fits_in r s') →
      s'.side ≥ s.side) := by
  sorry

#check smallest_perfect_square_tiling

end NUMINAMATH_CALUDE_smallest_perfect_square_tiling_l1416_141615


namespace NUMINAMATH_CALUDE_initial_population_proof_l1416_141610

/-- Proves that the initial population is 10000 given the conditions --/
theorem initial_population_proof (P : ℝ) : 
  (P * (1 + 0.2)^2 = 14400) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_proof_l1416_141610


namespace NUMINAMATH_CALUDE_solve_system_l1416_141678

theorem solve_system (x y : ℝ) :
  (x / 6) * 12 = 10 ∧ (y / 4) * 8 = x → x = 5 ∧ y = (5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solve_system_l1416_141678


namespace NUMINAMATH_CALUDE_signal_count_l1416_141618

/-- Represents the three flag colors -/
inductive FlagColor
  | Red
  | Yellow
  | Blue

/-- Represents a signal as a list of flag colors -/
def Signal := List FlagColor

/-- Returns true if the signal is valid (contains 1, 2, or 3 flags) -/
def isValidSignal (s : Signal) : Bool :=
  1 ≤ s.length ∧ s.length ≤ 3

/-- Returns all possible valid signals -/
def allValidSignals : List Signal :=
  (List.map (λ c => [c]) [FlagColor.Red, FlagColor.Yellow, FlagColor.Blue]) ++
  (List.map (λ (c1, c2) => [c1, c2]) [(FlagColor.Red, FlagColor.Yellow), (FlagColor.Red, FlagColor.Blue), (FlagColor.Yellow, FlagColor.Red), (FlagColor.Yellow, FlagColor.Blue), (FlagColor.Blue, FlagColor.Red), (FlagColor.Blue, FlagColor.Yellow)]) ++
  (List.map (λ (c1, c2, c3) => [c1, c2, c3]) [(FlagColor.Red, FlagColor.Yellow, FlagColor.Blue), (FlagColor.Red, FlagColor.Blue, FlagColor.Yellow), (FlagColor.Yellow, FlagColor.Red, FlagColor.Blue), (FlagColor.Yellow, FlagColor.Blue, FlagColor.Red), (FlagColor.Blue, FlagColor.Red, FlagColor.Yellow), (FlagColor.Blue, FlagColor.Yellow, FlagColor.Red)])

theorem signal_count : (allValidSignals.filter isValidSignal).length = 15 := by
  sorry

end NUMINAMATH_CALUDE_signal_count_l1416_141618


namespace NUMINAMATH_CALUDE_find_divisor_l1416_141620

theorem find_divisor (n : ℕ) (s : ℕ) (d : ℕ) : 
  n = 724946 →
  s = 6 →
  d ∣ (n - s) →
  (∀ k < s, ¬(d ∣ (n - k))) →
  d = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1416_141620


namespace NUMINAMATH_CALUDE_inequality_solutions_l1416_141626

theorem inequality_solutions (x : ℝ) : 
  ((-x^2 + x + 6 ≤ 0) ↔ (x ≤ -2 ∨ x ≥ 3)) ∧
  ((x^2 - 2*x - 5 < 2*x) ↔ (-1 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1416_141626


namespace NUMINAMATH_CALUDE_friday_occurs_five_times_in_september_l1416_141657

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- Structure representing a year with specific properties -/
structure Year where
  julySundayCount : Nat
  februaryLeap : Bool
  septemberDayCount : Nat

/-- Function to determine the day that occurs five times in September -/
def dayOccurringFiveTimesInSeptember (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that Friday occurs five times in September under given conditions -/
theorem friday_occurs_five_times_in_september (y : Year) 
    (h1 : y.julySundayCount = 5)
    (h2 : y.februaryLeap = true)
    (h3 : y.septemberDayCount = 30) :
    dayOccurringFiveTimesInSeptember y = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_friday_occurs_five_times_in_september_l1416_141657


namespace NUMINAMATH_CALUDE_merchant_max_profit_optimal_selling_price_l1416_141659

/-- Represents the merchant's profit function -/
def profit (x : ℝ) : ℝ := -10 * x^2 + 80 * x + 200

/-- The optimal price increase that maximizes profit -/
def optimal_increase : ℝ := 4

/-- The maximum achievable profit -/
def max_profit : ℝ := 360

theorem merchant_max_profit :
  (∀ x, 0 ≤ x → x < 10 → profit x ≤ max_profit) ∧
  profit optimal_increase = max_profit :=
sorry

theorem optimal_selling_price :
  optimal_increase + 10 = 14 :=
sorry

end NUMINAMATH_CALUDE_merchant_max_profit_optimal_selling_price_l1416_141659


namespace NUMINAMATH_CALUDE_dvd_average_price_l1416_141631

/-- Calculates the average price of DVDs bought from two different price groups -/
theorem dvd_average_price (n1 : ℕ) (p1 : ℚ) (n2 : ℕ) (p2 : ℚ) : 
  n1 = 10 → p1 = 2 → n2 = 5 → p2 = 5 → 
  (n1 * p1 + n2 * p2) / (n1 + n2 : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_dvd_average_price_l1416_141631


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l1416_141697

theorem pascal_triangle_15th_row_5th_number :
  Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l1416_141697


namespace NUMINAMATH_CALUDE_max_dominoes_9x10_board_l1416_141651

/-- Represents a chessboard with given dimensions -/
structure Chessboard where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of dominoes that can be placed on a chessboard -/
def max_dominoes (board : Chessboard) (domino : Domino) : ℕ :=
  sorry

/-- Theorem stating the maximum number of 6x1 dominoes on a 9x10 chessboard -/
theorem max_dominoes_9x10_board :
  let board := Chessboard.mk 9 10
  let domino := Domino.mk 6 1
  max_dominoes board domino = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_dominoes_9x10_board_l1416_141651


namespace NUMINAMATH_CALUDE_first_player_wins_l1416_141650

/-- Represents a chessboard --/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a piece on the chessboard --/
structure Piece :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a player in the game --/
inductive Player
| First
| Second

/-- Defines the game state --/
structure GameState :=
  (board : Chessboard)
  (firstPiece : Piece)
  (secondPiece : Piece)
  (currentPlayer : Player)

/-- Defines a winning strategy for a player --/
def WinningStrategy (player : Player) (game : GameState) : Prop :=
  ∃ (strategy : GameState → ℕ × ℕ), 
    ∀ (opponent_move : ℕ × ℕ), 
      player = game.currentPlayer → 
      ∃ (next_state : GameState), 
        next_state.currentPlayer ≠ player ∧ 
        (∃ (final_state : GameState), final_state.currentPlayer = player ∧ ¬∃ (move : ℕ × ℕ), true)

/-- The main theorem stating that the first player has a winning strategy --/
theorem first_player_wins (game : GameState) : 
  game.board.rows = 3 ∧ 
  game.board.cols = 1000 ∧ 
  game.firstPiece.width = 1 ∧ 
  game.firstPiece.height = 2 ∧ 
  game.secondPiece.width = 2 ∧ 
  game.secondPiece.height = 1 ∧ 
  game.currentPlayer = Player.First → 
  WinningStrategy Player.First game :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1416_141650


namespace NUMINAMATH_CALUDE_candy_count_l1416_141601

/-- Given the total number of treats, chewing gums, and chocolate bars,
    prove that the number of candies of different flavors is 40. -/
theorem candy_count (total_treats chewing_gums chocolate_bars : ℕ) 
  (h1 : total_treats = 155)
  (h2 : chewing_gums = 60)
  (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l1416_141601


namespace NUMINAMATH_CALUDE_law_school_applicants_l1416_141634

/-- The number of applicants to a law school satisfying certain conditions -/
theorem law_school_applicants 
  (ps : ℕ)  -- number of applicants who majored in political science
  (gpa_high : ℕ)  -- number of applicants with GPA > 3.0
  (not_ps_low_gpa : ℕ)  -- number of applicants not in PS and GPA ≤ 3.0
  (ps_and_high_gpa : ℕ)  -- number of applicants in PS and GPA > 3.0
  (h1 : ps = 15)
  (h2 : gpa_high = 20)
  (h3 : not_ps_low_gpa = 10)
  (h4 : ps_and_high_gpa = 5) :
  ps + gpa_high - ps_and_high_gpa + not_ps_low_gpa = 40 :=
by sorry

end NUMINAMATH_CALUDE_law_school_applicants_l1416_141634


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l1416_141649

/-- The number of children on a bus after a stop -/
theorem children_on_bus_after_stop 
  (initial_children : ℕ) 
  (children_who_got_on : ℕ) 
  (h1 : initial_children = 18) 
  (h2 : children_who_got_on = 7) :
  initial_children + children_who_got_on = 25 := by
  sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l1416_141649


namespace NUMINAMATH_CALUDE_expression_value_l1416_141625

theorem expression_value (a : ℚ) (h : a = 1/3) : 
  (3 * a⁻¹ + (2 * a⁻¹) / 3) / a = 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1416_141625


namespace NUMINAMATH_CALUDE_min_sum_a_b_l1416_141607

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y = 1 → a + b ≤ x + y ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l1416_141607


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l1416_141643

theorem crazy_silly_school_series (total : ℕ) (books_read : ℕ) (movies_watched : ℕ) 
  (h1 : total = books_read + movies_watched)
  (h2 : books_read = movies_watched + 1)
  (h3 : total = 13) : 
  movies_watched = 6 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l1416_141643


namespace NUMINAMATH_CALUDE_subtract_point_five_from_forty_seven_point_two_l1416_141693

theorem subtract_point_five_from_forty_seven_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_forty_seven_point_two_l1416_141693


namespace NUMINAMATH_CALUDE_circle_C_properties_l1416_141600

/-- The circle C passing through A(4,1) and tangent to x-y-1=0 at B(2,1) -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 4}

/-- Point A -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B -/
def point_B : ℝ × ℝ := (2, 1)

/-- The line x-y-1=0 -/
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_C ∧ tangent_line p → p = point_B :=
by sorry

end NUMINAMATH_CALUDE_circle_C_properties_l1416_141600


namespace NUMINAMATH_CALUDE_wedge_product_formula_l1416_141609

/-- The wedge product of two 2D vectors -/
def wedge_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

/-- Theorem: The wedge product of two 2D vectors (a₁, a₂) and (b₁, b₂) is equal to a₁b₂ - a₂b₁ -/
theorem wedge_product_formula (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 := by
  sorry

end NUMINAMATH_CALUDE_wedge_product_formula_l1416_141609


namespace NUMINAMATH_CALUDE_B_fair_share_l1416_141677

-- Define the total rent
def total_rent : ℕ := 841

-- Define the number of horses and months for each person
def horses_A : ℕ := 12
def months_A : ℕ := 8
def horses_B : ℕ := 16
def months_B : ℕ := 9
def horses_C : ℕ := 18
def months_C : ℕ := 6

-- Calculate the total horse-months
def total_horse_months : ℕ := horses_A * months_A + horses_B * months_B + horses_C * months_C

-- Calculate B's horse-months
def B_horse_months : ℕ := horses_B * months_B

-- Theorem: B's fair share of the rent is 348
theorem B_fair_share : 
  (total_rent : ℚ) * B_horse_months / total_horse_months = 348 := by
  sorry

end NUMINAMATH_CALUDE_B_fair_share_l1416_141677


namespace NUMINAMATH_CALUDE_CaO_weight_calculation_l1416_141680

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := calcium_weight + oxygen_weight

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

theorem CaO_weight_calculation : total_weight_CaO = 392.56 := by
  sorry

end NUMINAMATH_CALUDE_CaO_weight_calculation_l1416_141680


namespace NUMINAMATH_CALUDE_louie_last_match_goals_l1416_141675

/-- The number of goals Louie scored in the last match -/
def last_match_goals : ℕ := sorry

/-- The number of seasons Louie's brother has played -/
def brothers_seasons : ℕ := 3

/-- The number of games in each season -/
def games_per_season : ℕ := 50

/-- The total number of goals scored by both brothers -/
def total_goals : ℕ := 1244

/-- The number of goals Louie scored in previous matches -/
def previous_goals : ℕ := 40

theorem louie_last_match_goals : 
  last_match_goals = 4 ∧
  brothers_seasons * games_per_season * (2 * last_match_goals) + 
  previous_goals + last_match_goals = total_goals :=
by sorry

end NUMINAMATH_CALUDE_louie_last_match_goals_l1416_141675


namespace NUMINAMATH_CALUDE_expression_bounds_l1416_141628

theorem expression_bounds (x y z w : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
                    Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ∧
  Real.sqrt (x^2 + (1-y)^2) + Real.sqrt (y^2 + (1-z)^2) + 
  Real.sqrt (z^2 + (1-w)^2) + Real.sqrt (w^2 + (1-x)^2) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l1416_141628


namespace NUMINAMATH_CALUDE_investment_problem_l1416_141673

/-- The investment problem --/
theorem investment_problem (vishal trishul raghu : ℝ) : 
  vishal = 1.1 * trishul →  -- Vishal invested 10% more than Trishul
  raghu = 2500 →  -- Raghu invested Rs. 2500
  vishal + trishul + raghu = 7225 →  -- Total sum of investments
  trishul < raghu →  -- Trishul invested less than Raghu
  (raghu - trishul) / raghu * 100 = 10 :=  -- Percentage difference
by sorry

end NUMINAMATH_CALUDE_investment_problem_l1416_141673


namespace NUMINAMATH_CALUDE_green_blue_difference_l1416_141695

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  colorCount : DiskColor → Nat

/-- The theorem stating the difference between green and blue disks -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8])
  (h_color_count : ∀ c, bag.colorCount c = (bag.total / (bag.ratio 0 + bag.ratio 1 + bag.ratio 2)) * match c with
    | DiskColor.Blue => bag.ratio 0
    | DiskColor.Yellow => bag.ratio 1
    | DiskColor.Green => bag.ratio 2) :
  bag.colorCount DiskColor.Green - bag.colorCount DiskColor.Blue = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l1416_141695


namespace NUMINAMATH_CALUDE_water_servings_difference_l1416_141611

/-- Proves the difference in servings for Simeon's water consumption --/
theorem water_servings_difference (total_water : ℕ) (old_serving : ℕ) (new_serving : ℕ)
  (h1 : total_water = 64)
  (h2 : old_serving = 8)
  (h3 : new_serving = 16)
  (h4 : old_serving > 0)
  (h5 : new_serving > 0) :
  (total_water / old_serving) - (total_water / new_serving) = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_servings_difference_l1416_141611


namespace NUMINAMATH_CALUDE_salmon_population_increase_l1416_141655

/-- Calculates the total number of salmon after an increase. -/
def total_salmon_after_increase (initial_salmon : ℕ) (increase_factor : ℕ) : ℕ :=
  initial_salmon * increase_factor

/-- Theorem stating that given an initial salmon population of 500 and an increase factor of 10,
    the total number of salmon after the increase is 5000. -/
theorem salmon_population_increase :
  total_salmon_after_increase 500 10 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_salmon_population_increase_l1416_141655


namespace NUMINAMATH_CALUDE_number_of_dogs_l1416_141648

/-- The number of dogs at a farm, given the number of fish, cats, and total pets. -/
theorem number_of_dogs (fish : ℕ) (cats : ℕ) (total_pets : ℕ) : 
  fish = 72 → cats = 34 → total_pets = 149 → total_pets = fish + cats + 43 :=
by sorry

end NUMINAMATH_CALUDE_number_of_dogs_l1416_141648


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l1416_141674

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem even_odd_sum_difference : 
  (sumFirstEvenIntegers 25) - (sumFirstOddIntegers 20) = 250 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l1416_141674


namespace NUMINAMATH_CALUDE_tan_product_equals_two_l1416_141698

theorem tan_product_equals_two :
  (∀ x y z : ℝ, x = 100 ∧ y = 35 ∧ z = 135 →
    Real.tan (z * π / 180) = -1 →
    (1 - Real.tan (x * π / 180)) * (1 - Real.tan (y * π / 180)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_tan_product_equals_two_l1416_141698


namespace NUMINAMATH_CALUDE_q_contribution_l1416_141681

/-- Represents the contribution and time in the business for a partner -/
structure Partner where
  contribution : ℕ
  time : ℕ

/-- Calculates the weighted contribution of a partner -/
def weightedContribution (p : Partner) : ℕ := p.contribution * p.time

/-- Represents the business scenario -/
structure Business where
  p : Partner
  q : Partner
  profitRatio : Fraction

theorem q_contribution (b : Business) : b.q.contribution = 9000 :=
  sorry

end NUMINAMATH_CALUDE_q_contribution_l1416_141681


namespace NUMINAMATH_CALUDE_total_practice_hours_l1416_141624

def monday_hours : ℕ := 6
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := 5
def thursday_hours : ℕ := 7
def friday_hours : ℕ := 3

def total_scheduled_hours : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def practice_days : ℕ := 5

def player_a_missed_hours : ℕ := 2
def player_b_missed_hours : ℕ := 3

def rainy_day_hours : ℕ := total_scheduled_hours / practice_days

theorem total_practice_hours :
  total_scheduled_hours - (rainy_day_hours + player_a_missed_hours + player_b_missed_hours) = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_hours_l1416_141624


namespace NUMINAMATH_CALUDE_aquarium_visitors_l1416_141627

-- Define the constants
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group_size : ℕ := 10
def total_earnings : ℕ := 240

-- Define the function to calculate the number of people who only went to the aquarium
def people_only_aquarium : ℕ :=
  (total_earnings - group_size * (admission_fee + tour_fee)) / admission_fee

-- Theorem to prove
theorem aquarium_visitors :
  people_only_aquarium = 5 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l1416_141627


namespace NUMINAMATH_CALUDE_min_waves_to_21_l1416_141666

/-- Represents the direction of a wand wave -/
inductive WaveDirection
  | Up
  | Down

/-- Calculates the number of open flowers after a single wave -/
def wave (n : ℕ) (d : WaveDirection) : ℕ :=
  match d with
  | WaveDirection.Up => if n > 0 then n - 1 else 0
  | WaveDirection.Down => 2 * n

/-- Calculates the number of open flowers after a sequence of waves -/
def waveSequence (initial : ℕ) (waves : List WaveDirection) : ℕ :=
  waves.foldl wave initial

/-- Checks if a sequence of waves results in the target number of flowers -/
def isValidSequence (initial target : ℕ) (waves : List WaveDirection) : Prop :=
  waveSequence initial waves = target

/-- Theorem: The minimum number of waves to reach 21 flowers from 3 flowers is 6 -/
theorem min_waves_to_21 :
  ∃ (waves : List WaveDirection),
    waves.length = 6 ∧
    isValidSequence 3 21 waves ∧
    ∀ (other : List WaveDirection),
      isValidSequence 3 21 other → waves.length ≤ other.length :=
by sorry

end NUMINAMATH_CALUDE_min_waves_to_21_l1416_141666


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l1416_141689

theorem cos_2alpha_plus_4pi_3 (α : ℝ) (h : Real.sqrt 3 * Real.sin α + Real.cos α = 1/2) :
  Real.cos (2 * α + 4 * Real.pi / 3) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_4pi_3_l1416_141689


namespace NUMINAMATH_CALUDE_carla_initial_marbles_l1416_141663

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has after buying -/
def total_marbles : ℕ := 187

/-- The number of marbles Carla started with -/
def initial_marbles : ℕ := total_marbles - marbles_bought

theorem carla_initial_marbles :
  initial_marbles = 53 := by sorry

end NUMINAMATH_CALUDE_carla_initial_marbles_l1416_141663


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1416_141686

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a : V)

theorem vector_equation_solution (x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1416_141686


namespace NUMINAMATH_CALUDE_red_cows_produce_more_milk_l1416_141632

/-- The daily milk production of a black cow -/
def black_cow_milk : ℝ := sorry

/-- The daily milk production of a red cow -/
def red_cow_milk : ℝ := sorry

/-- The total milk production of 4 black cows and 3 red cows in 5 days -/
def milk_production_1 : ℝ := 5 * (4 * black_cow_milk + 3 * red_cow_milk)

/-- The total milk production of 3 black cows and 5 red cows in 4 days -/
def milk_production_2 : ℝ := 4 * (3 * black_cow_milk + 5 * red_cow_milk)

theorem red_cows_produce_more_milk :
  milk_production_1 = milk_production_2 → red_cow_milk > black_cow_milk := by
  sorry

end NUMINAMATH_CALUDE_red_cows_produce_more_milk_l1416_141632


namespace NUMINAMATH_CALUDE_polynomial_value_l1416_141652

theorem polynomial_value : (3 : ℝ)^6 - 7 * 3 = 708 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1416_141652


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1416_141660

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.cos α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1416_141660


namespace NUMINAMATH_CALUDE_jinho_remaining_money_l1416_141658

theorem jinho_remaining_money (initial_amount : ℕ) (eraser_cost pencil_cost : ℕ) 
  (eraser_count pencil_count : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 2500 →
  eraser_cost = 120 →
  pencil_cost = 350 →
  eraser_count = 5 →
  pencil_count = 3 →
  remaining_amount = initial_amount - (eraser_cost * eraser_count + pencil_cost * pencil_count) →
  remaining_amount = 850 := by
sorry

end NUMINAMATH_CALUDE_jinho_remaining_money_l1416_141658


namespace NUMINAMATH_CALUDE_calculation_proof_l1416_141668

theorem calculation_proof : (1 / 6 : ℚ) * (-6 : ℚ) / (-1 / 6 : ℚ) * 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1416_141668


namespace NUMINAMATH_CALUDE_calculation_proof_l1416_141619

theorem calculation_proof : 4 * 6 * 8 + 24 / 4 + 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1416_141619


namespace NUMINAMATH_CALUDE_inequality_proof_l1416_141684

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a + b)) + (b / (b + c)) + (c / (c + a)) ≤ 3 / (1 + Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1416_141684


namespace NUMINAMATH_CALUDE_ratio_after_adding_water_l1416_141647

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratio (m : Mixture) : ℝ × ℝ :=
  (m.alcohol, m.water)

/-- Adds water to a mixture -/
def add_water (m : Mixture) (amount : ℝ) : Mixture :=
  { alcohol := m.alcohol, water := m.water + amount }

/-- The initial mixture -/
def initial_mixture : Mixture :=
  { alcohol := 4, water := 3 }

/-- The amount of water added -/
def water_added : ℝ := 8

/-- Theorem stating that adding water changes the ratio to 4:11 -/
theorem ratio_after_adding_water :
  ratio (add_water initial_mixture water_added) = (4, 11) := by
  sorry

end NUMINAMATH_CALUDE_ratio_after_adding_water_l1416_141647


namespace NUMINAMATH_CALUDE_tip_amount_is_24_l1416_141639

-- Define the cost of haircuts
def womens_haircut_cost : ℚ := 48
def childrens_haircut_cost : ℚ := 36

-- Define the number of each type of haircut
def num_womens_haircuts : ℕ := 1
def num_childrens_haircuts : ℕ := 2

-- Define the tip percentage
def tip_percentage : ℚ := 20 / 100

-- Theorem statement
theorem tip_amount_is_24 :
  let total_cost := womens_haircut_cost * num_womens_haircuts + childrens_haircut_cost * num_childrens_haircuts
  tip_percentage * total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_tip_amount_is_24_l1416_141639


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1416_141688

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (4, -1)
def d : ℝ × ℝ := (2, -5)

theorem parallel_lines_distance : 
  let line1 := fun (t : ℝ) => a + t • d
  let line2 := fun (s : ℝ) => b + s • d
  (∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ∀ (x y : ℝ × ℝ), x ∈ Set.range line1 → y ∈ Set.range line2 → 
      ‖p - q‖ ≤ ‖x - y‖) →
  ∃ (p q : ℝ × ℝ), p ∈ Set.range line1 ∧ q ∈ Set.range line2 ∧ 
    ‖p - q‖ = (5 * Real.sqrt 29) / 29 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1416_141688


namespace NUMINAMATH_CALUDE_sum_of_solutions_x_squared_36_l1416_141616

theorem sum_of_solutions_x_squared_36 (x : ℝ) (h : x^2 = 36) :
  ∃ (y : ℝ), y^2 = 36 ∧ x + y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_x_squared_36_l1416_141616


namespace NUMINAMATH_CALUDE_vacation_expense_balance_l1416_141644

/-- Vacation expense balancing problem -/
theorem vacation_expense_balance :
  let total_spent := 160 + 120 + 140 + 200
  let individual_share := total_spent / 4
  let alice_balance := 160 - individual_share
  let bob_balance := 120 - individual_share
  let alice_gives_dave := max 0 (-alice_balance)
  let bob_gives_dave := max 0 (-bob_balance)
  alice_gives_dave - bob_gives_dave = -35 :=
by sorry

end NUMINAMATH_CALUDE_vacation_expense_balance_l1416_141644


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1416_141637

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31 / 13 - (1 / 13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1416_141637


namespace NUMINAMATH_CALUDE_divisors_odd_iff_perfect_square_l1416_141646

theorem divisors_odd_iff_perfect_square (n : ℕ) : 
  Odd (Finset.card (Nat.divisors n)) ↔ ∃ m : ℕ, n = m ^ 2 := by
sorry

end NUMINAMATH_CALUDE_divisors_odd_iff_perfect_square_l1416_141646


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l1416_141623

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_calculation (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l1416_141623


namespace NUMINAMATH_CALUDE_min_omega_for_50_maxima_l1416_141672

theorem min_omega_for_50_maxima (ω : ℝ) : ω > 0 → (∀ x ∈ Set.Icc 0 1, ∃ y, y = Real.sin (ω * x)) →
  (∃ (maxima : Finset ℝ), maxima.card ≥ 50 ∧ 
    ∀ t ∈ maxima, t ∈ Set.Icc 0 1 ∧ 
    (∀ h ∈ Set.Icc 0 1, Real.sin (ω * t) ≥ Real.sin (ω * h))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end NUMINAMATH_CALUDE_min_omega_for_50_maxima_l1416_141672


namespace NUMINAMATH_CALUDE_floor_width_calculation_l1416_141661

def tile_length : ℝ := 65
def tile_width : ℝ := 25
def floor_length : ℝ := 150
def max_tiles : ℕ := 36

theorem floor_width_calculation (floor_width : ℝ) 
  (h1 : floor_length = 150)
  (h2 : tile_length = 65)
  (h3 : tile_width = 25)
  (h4 : max_tiles = 36)
  (h5 : 2 * tile_length ≤ floor_length)
  (h6 : floor_width = (max_tiles / 2 : ℝ) * tile_width) :
  floor_width = 450 := by
sorry

end NUMINAMATH_CALUDE_floor_width_calculation_l1416_141661


namespace NUMINAMATH_CALUDE_odd_function_sum_l1416_141640

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_sum (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-1) = 2) :
  f 0 + f 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l1416_141640


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1416_141687

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : x * y = 18)
  (h2 : x * z = 3)
  (h3 : y * z = 6) :
  x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1416_141687


namespace NUMINAMATH_CALUDE_separation_theorem_l1416_141692

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a line in a plane
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

-- Define the set of white points and black points
def WhitePoints : Set Point := sorry
def BlackPoints : Set Point := sorry

-- Define a function to check if a point is on one side of a line
def onSide (p : Point) (l : Line) : Bool := sorry

-- Define a function to check if a line separates two sets of points
def separates (l : Line) (s1 s2 : Set Point) : Prop :=
  ∀ p1 ∈ s1, ∀ p2 ∈ s2, onSide p1 l ≠ onSide p2 l

-- State the theorem
theorem separation_theorem :
  (∀ p1 p2 p3 p4 : Point, p1 ∈ WhitePoints ∪ BlackPoints →
    p2 ∈ WhitePoints ∪ BlackPoints → p3 ∈ WhitePoints ∪ BlackPoints →
    p4 ∈ WhitePoints ∪ BlackPoints →
    ∃ l : Line, separates l (WhitePoints ∩ {p1, p2, p3, p4}) (BlackPoints ∩ {p1, p2, p3, p4})) →
  ∃ l : Line, separates l WhitePoints BlackPoints :=
sorry

end NUMINAMATH_CALUDE_separation_theorem_l1416_141692


namespace NUMINAMATH_CALUDE_linear_regression_at_6_l1416_141654

/-- Linear regression equation -/
def linear_regression (b a x : ℝ) : ℝ := b * x + a

theorem linear_regression_at_6 (b a : ℝ) (h1 : linear_regression b a 4 = 50) (h2 : b = -2) :
  linear_regression b a 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_at_6_l1416_141654


namespace NUMINAMATH_CALUDE_optimal_plan_is_best_three_valid_plans_l1416_141604

/-- Represents a purchasing plan for machines --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions --/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 6 ∧
  7 * p.typeA + 5 * p.typeB ≤ 34 ∧
  100 * p.typeA + 60 * p.typeB ≥ 380

/-- Calculates the total cost of a purchase plan --/
def totalCost (p : PurchasePlan) : ℕ :=
  7 * p.typeA + 5 * p.typeB

/-- The optimal purchase plan --/
def optimalPlan : PurchasePlan :=
  { typeA := 1, typeB := 5 }

/-- Theorem stating that the optimal plan is valid and minimizes cost --/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

/-- Theorem stating that there are exactly 3 valid purchase plans --/
theorem three_valid_plans :
  ∃! (plans : List PurchasePlan),
    plans.length = 3 ∧
    ∀ p : PurchasePlan, isValidPlan p ↔ p ∈ plans :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_is_best_three_valid_plans_l1416_141604


namespace NUMINAMATH_CALUDE_meet_after_four_turns_l1416_141638

-- Define the number of points on the circular track
def num_points : ℕ := 15

-- Define Alice's clockwise movement per turn
def alice_move : ℕ := 4

-- Define Bob's counterclockwise movement per turn
def bob_move : ℕ := 11

-- Define the starting point for both Alice and Bob
def start_point : ℕ := 15

-- Function to calculate the new position after a move
def new_position (current : ℕ) (move : ℕ) : ℕ :=
  ((current + move - 1) % num_points) + 1

-- Function to calculate Alice's position after n turns
def alice_position (n : ℕ) : ℕ :=
  new_position start_point (n * alice_move)

-- Function to calculate Bob's position after n turns
def bob_position (n : ℕ) : ℕ :=
  new_position start_point (n * (num_points - bob_move))

-- Theorem stating that Alice and Bob meet after 4 turns
theorem meet_after_four_turns :
  ∃ n : ℕ, n = 4 ∧ alice_position n = bob_position n :=
sorry

end NUMINAMATH_CALUDE_meet_after_four_turns_l1416_141638


namespace NUMINAMATH_CALUDE_not_always_parallel_lines_l1416_141682

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (α β : Plane) (m n : Line) 
  (h1 : parallel_line_plane m α) 
  (h2 : parallel_plane_plane α β) 
  (h3 : line_in_plane n β) : 
  ¬(∀ m n, parallel_line_line m n) := by
sorry


end NUMINAMATH_CALUDE_not_always_parallel_lines_l1416_141682


namespace NUMINAMATH_CALUDE_phone_number_proof_l1416_141612

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def transform (n : ℕ) : ℕ :=
  2 * 10000000 + (n / 100000) * 1000000 + 800000 + (n % 100000)

theorem phone_number_proof (x : ℕ) (h1 : is_six_digit x) (h2 : transform x = 81 * x) :
  x = 260000 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_proof_l1416_141612


namespace NUMINAMATH_CALUDE_intersection_implies_range_l1416_141671

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}

def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_implies_range (a : ℝ) : A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_range_l1416_141671


namespace NUMINAMATH_CALUDE_g_inv_f_10_l1416_141630

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- Assume f and g are bijective
variable (hf : Function.Bijective f)
variable (hg : Function.Bijective g)

-- Define the relationship between f and g
axiom fg_relation : ∀ x, f_inv (g x) = 3 * x - 1

-- Define the inverse functions
axiom f_inverse : Function.LeftInverse f_inv f ∧ Function.RightInverse f_inv f
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g

-- State the theorem
theorem g_inv_f_10 : g_inv (f 10) = 11 / 3 := by sorry

end NUMINAMATH_CALUDE_g_inv_f_10_l1416_141630


namespace NUMINAMATH_CALUDE_problem_solution_l1416_141614

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1416_141614


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_seventeen_l1416_141603

theorem largest_negative_congruent_to_two_mod_seventeen :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 2 [ZMOD 17] → n ≤ -1001 :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_seventeen_l1416_141603


namespace NUMINAMATH_CALUDE_power_difference_mod_eight_l1416_141617

theorem power_difference_mod_eight : 
  (47^1235 - 22^1235) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_mod_eight_l1416_141617


namespace NUMINAMATH_CALUDE_solve_equation_for_A_l1416_141641

theorem solve_equation_for_A : ∃ A : ℝ,
  (1 / ((5 / (1 + (24 / A))) - 5 / 9)) * (3 / (2 + (5 / 7))) / (2 / (3 + (3 / 4))) + 2.25 = 4 ∧ A = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_A_l1416_141641


namespace NUMINAMATH_CALUDE_adjacent_even_numbers_l1416_141667

theorem adjacent_even_numbers (n : ℕ) : 
  Odd (2*n + 1) ∧ 
  Even (2*n) ∧ 
  Even (2*n + 2) ∧
  (2*n + 1) - 1 = 2*n ∧
  (2*n + 1) + 1 = 2*n + 2 := by
sorry

end NUMINAMATH_CALUDE_adjacent_even_numbers_l1416_141667


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_dimensions_l1416_141665

def is_valid_dimension (x y z : ℕ) : Prop :=
  2 * (x * y + y * z + z * x) = x * y * z

def valid_dimensions : List (ℕ × ℕ × ℕ) :=
  [(6,6,6), (5,5,10), (4,8,8), (3,12,12), (3,7,42), (3,8,24), (3,9,18), (3,10,15), (4,5,20), (4,6,12)]

theorem rectangular_parallelepiped_dimensions (x y z : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 → is_valid_dimension x y z → (x, y, z) ∈ valid_dimensions := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_dimensions_l1416_141665


namespace NUMINAMATH_CALUDE_speed_conversion_correct_l1416_141653

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 19.445999999999998

/-- Converts speed from m/s to km/h -/
def convert_speed (s : ℝ) : ℝ := s * mps_to_kmph

theorem speed_conversion_correct : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0005 ∧ |convert_speed speed_mps - 70.006| < ε :=
sorry

end NUMINAMATH_CALUDE_speed_conversion_correct_l1416_141653


namespace NUMINAMATH_CALUDE_cake_eaten_after_four_trips_l1416_141642

def eat_cake (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem cake_eaten_after_four_trips :
  eat_cake 4 = 40/81 := by
  sorry

end NUMINAMATH_CALUDE_cake_eaten_after_four_trips_l1416_141642


namespace NUMINAMATH_CALUDE_exact_exponent_equality_l1416_141608

theorem exact_exponent_equality (n k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ m : ℕ, (2^(2^n) + 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(2^n) + 1 = p^(k+1) * l)) →
  (∃ m : ℕ, (2^(p-1) - 1 = p^k * m) ∧ ¬(∃ l : ℕ, 2^(p-1) - 1 = p^(k+1) * l)) :=
by sorry

end NUMINAMATH_CALUDE_exact_exponent_equality_l1416_141608


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1416_141629

theorem decimal_to_fraction (x : ℚ) (h : x = 3.36) : 
  ∃ (n d : ℕ), d ≠ 0 ∧ x = n / d ∧ n = 84 ∧ d = 25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1416_141629


namespace NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l1416_141691

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun son_age man_age =>
    (son_age = 22) →
    (man_age + 2 = 2 * (son_age + 2)) →
    (man_age - son_age = 24)

-- The proof is omitted
theorem man_son_age_difference_proof : ∃ (son_age man_age : ℕ), man_son_age_difference son_age man_age :=
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l1416_141691


namespace NUMINAMATH_CALUDE_exam_time_allocation_l1416_141683

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time minutes : ℕ) (total_questions type_A_questions : ℕ) : ℕ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := 2  -- Type A takes twice as long as Type B
  let total_time_units := type_A_questions * time_ratio + type_B_questions
  (total_time * minutes * type_A_questions * time_ratio) / total_time_units

/-- Theorem: Given the exam conditions, the time spent on type A problems is 120 minutes -/
theorem exam_time_allocation :
  time_on_type_A 3 60 200 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l1416_141683


namespace NUMINAMATH_CALUDE_sam_remaining_seashells_l1416_141613

def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

theorem sam_remaining_seashells : 
  initial_seashells - seashells_given_away = 17 := by sorry

end NUMINAMATH_CALUDE_sam_remaining_seashells_l1416_141613


namespace NUMINAMATH_CALUDE_work_completion_time_l1416_141605

theorem work_completion_time (work : ℝ) (time_renu : ℝ) (time_suma : ℝ) 
  (h1 : time_renu = 8) 
  (h2 : time_suma = 8) 
  (h3 : work > 0) :
  let rate_renu := work / time_renu
  let rate_suma := work / time_suma
  let combined_rate := rate_renu + rate_suma
  work / combined_rate = 4 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l1416_141605


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l1416_141621

/-- Given a current speed and a speed against the current, calculates the speed with the current -/
def speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem: Given the specified conditions, the man's speed with the current is 15 km/hr -/
theorem mans_speed_with_current :
  let current_speed : ℝ := 2.8
  let speed_against_current : ℝ := 9.4
  speed_with_current current_speed speed_against_current = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l1416_141621


namespace NUMINAMATH_CALUDE_lemons_per_glass_l1416_141676

/-- Given that 9 glasses of lemonade can be made with 18 lemons,
    prove that the number of lemons needed per glass is 2. -/
theorem lemons_per_glass (total_glasses : ℕ) (total_lemons : ℕ) 
  (h1 : total_glasses = 9) (h2 : total_lemons = 18) :
  total_lemons / total_glasses = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemons_per_glass_l1416_141676


namespace NUMINAMATH_CALUDE_hcd_6432_132_minus_8_l1416_141690

theorem hcd_6432_132_minus_8 : Nat.gcd 6432 132 - 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_hcd_6432_132_minus_8_l1416_141690


namespace NUMINAMATH_CALUDE_race_finish_orders_l1416_141685

theorem race_finish_orders (n : ℕ) : n = 4 → Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l1416_141685


namespace NUMINAMATH_CALUDE_janets_freelance_rate_janets_freelance_rate_is_33_75_l1416_141696

/-- Calculates Janet's hourly rate as a freelancer given her current job details and additional costs --/
theorem janets_freelance_rate (current_hourly_rate : ℝ) 
  (weekly_hours : ℝ) (weeks_per_month : ℝ) (extra_fica_per_week : ℝ) 
  (healthcare_premium : ℝ) (additional_monthly_income : ℝ) : ℝ :=
  let current_monthly_income := current_hourly_rate * weekly_hours * weeks_per_month
  let additional_costs := extra_fica_per_week * weeks_per_month + healthcare_premium
  let freelance_income := current_monthly_income + additional_monthly_income
  let net_freelance_income := freelance_income - additional_costs
  let monthly_hours := weekly_hours * weeks_per_month
  net_freelance_income / monthly_hours

/-- Proves that Janet's freelance hourly rate is $33.75 given the specified conditions --/
theorem janets_freelance_rate_is_33_75 : 
  janets_freelance_rate 30 40 4 25 400 1100 = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_janets_freelance_rate_janets_freelance_rate_is_33_75_l1416_141696


namespace NUMINAMATH_CALUDE_exactly_two_false_l1416_141662

-- Define the types
def Quadrilateral : Type := sorry
def Square : Quadrilateral → Prop := sorry
def Rectangle : Quadrilateral → Prop := sorry

-- Define the propositions
def P1 : Prop := ∀ q : Quadrilateral, Square q → Rectangle q
def P2 : Prop := ∀ q : Quadrilateral, Rectangle q → Square q
def P3 : Prop := ∀ q : Quadrilateral, ¬(Square q) → ¬(Rectangle q)
def P4 : Prop := ∀ q : Quadrilateral, ¬(Rectangle q) → ¬(Square q)

-- The theorem to prove
theorem exactly_two_false : 
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨ 
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_false_l1416_141662


namespace NUMINAMATH_CALUDE_special_parallelogram_side_ratio_l1416_141622

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- Adjacent sides of the parallelogram
  a : ℝ
  b : ℝ
  -- Diagonals of the parallelogram
  d1 : ℝ
  d2 : ℝ
  -- Conditions
  a_pos : 0 < a
  b_pos : 0 < b
  d1_pos : 0 < d1
  d2_pos : 0 < d2
  acute_angle : Real.cos (60 * π / 180) = 1 / 2
  diag_ratio : d1^2 / d2^2 = 1 / 3
  diag1_eq : d1^2 = a^2 + b^2 - a * b
  diag2_eq : d2^2 = a^2 + b^2 + a * b

/-- Theorem: In a special parallelogram, the ratio of adjacent sides is 1:1 -/
theorem special_parallelogram_side_ratio (p : SpecialParallelogram) : p.a = p.b := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_side_ratio_l1416_141622


namespace NUMINAMATH_CALUDE_max_value_theorem_l1416_141636

theorem max_value_theorem (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  ∀ a b c : ℝ, 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → c = 2 * b → 
  10 * x + 3 * y + 12 * z ≥ 10 * a + 3 * b + 12 * c ∧
  ∃ x₀ y₀ z₀ : ℝ, 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ z₀ = 2 * y₀ ∧
  10 * x₀ + 3 * y₀ + 12 * z₀ = Real.sqrt 253 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1416_141636


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1416_141602

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  n = 3^19 + 6^21 → (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q) ∧
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → p ≤ q) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1416_141602


namespace NUMINAMATH_CALUDE_leahs_coins_value_l1416_141645

theorem leahs_coins_value (n p : ℕ) : 
  n + p = 13 →                   -- Total number of coins is 13
  n + 1 = p →                    -- One more nickel would equal pennies
  5 * n + p = 37                 -- Total value in cents
  := by sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l1416_141645
