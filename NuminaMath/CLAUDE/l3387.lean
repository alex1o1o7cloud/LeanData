import Mathlib

namespace NUMINAMATH_CALUDE_bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l3387_338765

-- Define a type for figures
structure Figure where
  isBounded : Bool

-- Define a type for sets of points
structure PointSet where
  isFinite : Bool

-- Define a function to count centers of symmetry
def countCentersOfSymmetry (f : Figure) : Nat :=
  sorry

-- Define a function to count almost centers of symmetry
def countAlmostCentersOfSymmetry (s : PointSet) : Nat :=
  sorry

-- Theorem 1: A bounded figure has at most one center of symmetry
theorem bounded_figure_at_most_one_center (f : Figure) (h : f.isBounded = true) :
  countCentersOfSymmetry f ≤ 1 :=
sorry

-- Theorem 2: No figure can have exactly two centers of symmetry
theorem no_figure_exactly_two_centers (f : Figure) :
  countCentersOfSymmetry f ≠ 2 :=
sorry

-- Theorem 3: A finite set of points has at most 3 almost centers of symmetry
theorem finite_set_at_most_three_almost_centers (s : PointSet) (h : s.isFinite = true) :
  countAlmostCentersOfSymmetry s ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_bounded_figure_at_most_one_center_no_figure_exactly_two_centers_finite_set_at_most_three_almost_centers_l3387_338765


namespace NUMINAMATH_CALUDE_second_player_cannot_win_l3387_338784

-- Define the game of tic-tac-toe
structure TicTacToe :=
  (board : Matrix (Fin 3) (Fin 3) (Option Bool))
  (current_player : Bool)

-- Define optimal play
def optimal_play (game : TicTacToe) : Bool := sorry

-- Define the winning condition
def is_win (game : TicTacToe) (player : Bool) : Prop := sorry

-- Define the draw condition
def is_draw (game : TicTacToe) : Prop := sorry

-- Theorem: If the first player plays optimally, the second player cannot win
theorem second_player_cannot_win (game : TicTacToe) :
  optimal_play game → ¬(is_win game false) :=
by sorry

end NUMINAMATH_CALUDE_second_player_cannot_win_l3387_338784


namespace NUMINAMATH_CALUDE_find_certain_number_l3387_338797

theorem find_certain_number : ∃ x : ℝ,
  (20 + 40 + 60) / 3 = ((10 + x + 16) / 3 + 8) ∧ x = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l3387_338797


namespace NUMINAMATH_CALUDE_special_function_is_even_l3387_338785

/-- A function satisfying the given functional equation -/
structure SpecialFunction (f : ℝ → ℝ) : Prop where
  functional_eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y
  nonzero_at_zero : f 0 ≠ 0

/-- The main theorem: if f is a SpecialFunction, then it is even -/
theorem special_function_is_even (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_even_l3387_338785


namespace NUMINAMATH_CALUDE_luke_sticker_count_l3387_338727

/-- Represents the number of stickers Luke has at different stages -/
structure StickerCount where
  initial : ℕ
  afterBuying : ℕ
  afterGift : ℕ
  afterGiving : ℕ
  afterUsing : ℕ
  final : ℕ

/-- Theorem stating the relationship between Luke's initial and final sticker counts -/
theorem luke_sticker_count (s : StickerCount) 
  (hbuy : s.afterBuying = s.initial + 12)
  (hgift : s.afterGift = s.afterBuying + 20)
  (hgive : s.afterGiving = s.afterGift - 5)
  (huse : s.afterUsing = s.afterGiving - 8)
  (hfinal : s.final = s.afterUsing)
  (h_final_count : s.final = 39) :
  s.initial = 20 := by
  sorry

end NUMINAMATH_CALUDE_luke_sticker_count_l3387_338727


namespace NUMINAMATH_CALUDE_set_membership_equivalence_l3387_338768

theorem set_membership_equivalence (C M N : Set α) (x : α) :
  x ∈ C ∪ (M ∩ N) ↔ (x ∈ C ∪ M ∨ x ∈ C ∪ N) := by sorry

end NUMINAMATH_CALUDE_set_membership_equivalence_l3387_338768


namespace NUMINAMATH_CALUDE_premium_rate_calculation_l3387_338747

theorem premium_rate_calculation (total_investment dividend_rate share_face_value total_dividend : ℚ)
  (h1 : total_investment = 14400)
  (h2 : dividend_rate = 5 / 100)
  (h3 : share_face_value = 100)
  (h4 : total_dividend = 576) :
  ∃ premium_rate : ℚ,
    premium_rate = 25 ∧
    total_dividend = dividend_rate * share_face_value * (total_investment / (share_face_value + premium_rate)) :=
by sorry


end NUMINAMATH_CALUDE_premium_rate_calculation_l3387_338747


namespace NUMINAMATH_CALUDE_relationship_abc_l3387_338771

theorem relationship_abc : 
  let a : ℝ := (0.6 : ℝ) ^ (2/5 : ℝ)
  let b : ℝ := (0.4 : ℝ) ^ (2/5 : ℝ)
  let c : ℝ := (0.4 : ℝ) ^ (3/5 : ℝ)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3387_338771


namespace NUMINAMATH_CALUDE_fraction_equality_l3387_338798

theorem fraction_equality (p q : ℚ) : 
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 → p / q = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3387_338798


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l3387_338716

theorem sin_three_pi_halves : Real.sin (3 * π / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l3387_338716


namespace NUMINAMATH_CALUDE_survey_result_l3387_338740

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : Nat
  yes_responses : Nat
  id_range : Nat × Nat

/-- Calculates the expected number of students who have cheated -/
def expected_cheaters (data : SurveyData) : Nat :=
  let odd_ids := (data.id_range.2 - data.id_range.1 + 1) / 2
  let expected_yes_to_odd := odd_ids / 2
  let expected_cheaters_half := data.yes_responses - expected_yes_to_odd
  2 * expected_cheaters_half

/-- Theorem stating the expected number of cheaters based on the survey data -/
theorem survey_result (data : SurveyData) 
  (h1 : data.total_students = 2000)
  (h2 : data.yes_responses = 510)
  (h3 : data.id_range = (1, 2000)) :
  expected_cheaters data = 20 := by
  sorry


end NUMINAMATH_CALUDE_survey_result_l3387_338740


namespace NUMINAMATH_CALUDE_dany_farm_cows_l3387_338736

/-- Represents the number of cows on Dany's farm -/
def num_cows : ℕ := 4

/-- Represents the number of sheep on Dany's farm -/
def num_sheep : ℕ := 3

/-- Represents the number of chickens on Dany's farm -/
def num_chickens : ℕ := 7

/-- Represents the number of bushels a sheep eats per day -/
def sheep_bushels : ℕ := 2

/-- Represents the number of bushels a chicken eats per day -/
def chicken_bushels : ℕ := 3

/-- Represents the number of bushels a cow eats per day -/
def cow_bushels : ℕ := 2

/-- Represents the total number of bushels needed for all animals per day -/
def total_bushels : ℕ := 35

theorem dany_farm_cows :
  num_cows * cow_bushels + num_sheep * sheep_bushels + num_chickens * chicken_bushels = total_bushels :=
by sorry

end NUMINAMATH_CALUDE_dany_farm_cows_l3387_338736


namespace NUMINAMATH_CALUDE_monthly_snake_feeding_cost_l3387_338712

/-- Proves that the monthly cost per snake is $10, given Harry's pet ownership and feeding costs. -/
theorem monthly_snake_feeding_cost (num_geckos num_iguanas num_snakes : ℕ)
  (gecko_cost iguana_cost : ℚ) (total_annual_cost : ℚ) :
  num_geckos = 3 →
  num_iguanas = 2 →
  num_snakes = 4 →
  gecko_cost = 15 →
  iguana_cost = 5 →
  total_annual_cost = 1140 →
  (num_geckos * gecko_cost + num_iguanas * iguana_cost + num_snakes * 10) * 12 = total_annual_cost :=
by sorry

end NUMINAMATH_CALUDE_monthly_snake_feeding_cost_l3387_338712


namespace NUMINAMATH_CALUDE_max_boxes_is_240_l3387_338791

/-- Represents the weight of a box in pounds -/
inductive BoxWeight
  | light : BoxWeight  -- 10-pound box
  | heavy : BoxWeight  -- 40-pound box

/-- Calculates the total weight of a pair of boxes (one light, one heavy) -/
def pairWeight : ℕ := 50

/-- Represents the maximum weight capacity of a truck in pounds -/
def truckCapacity : ℕ := 2000

/-- Represents the number of trucks available for delivery -/
def numTrucks : ℕ := 3

/-- Calculates the maximum number of boxes that can be shipped in each delivery -/
def maxBoxesPerDelivery : ℕ := 
  (truckCapacity / pairWeight) * 2 * numTrucks

/-- Theorem stating that the maximum number of boxes that can be shipped in each delivery is 240 -/
theorem max_boxes_is_240 : maxBoxesPerDelivery = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_boxes_is_240_l3387_338791


namespace NUMINAMATH_CALUDE_cafe_tables_l3387_338744

-- Define the seating capacity in base 8
def seating_capacity_base8 : ℕ := 312

-- Define the number of people per table
def people_per_table : ℕ := 3

-- Define the function to convert from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Theorem statement
theorem cafe_tables :
  (base8_to_base10 seating_capacity_base8) / people_per_table = 67 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_l3387_338744


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l3387_338756

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- The sequence of common elements between the arithmetic and geometric progressions -/
def common_sequence (n : ℕ) : ℕ := 40 * 4^n

theorem sum_of_common_elements :
  (Finset.range 10).sum common_sequence = 13981000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l3387_338756


namespace NUMINAMATH_CALUDE_slope_problem_l3387_338767

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m) : m = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_problem_l3387_338767


namespace NUMINAMATH_CALUDE_jame_tear_frequency_l3387_338719

/-- Represents the number of times Jame tears cards per week -/
def tear_frequency (cards_per_tear : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) (num_weeks : ℕ) : ℕ :=
  (cards_per_deck * num_decks) / (cards_per_tear * num_weeks)

/-- Theorem stating that Jame tears cards 3 times a week given the conditions -/
theorem jame_tear_frequency :
  let cards_per_tear := 30
  let cards_per_deck := 55
  let num_decks := 18
  let num_weeks := 11
  tear_frequency cards_per_tear cards_per_deck num_decks num_weeks = 3 := by
  sorry


end NUMINAMATH_CALUDE_jame_tear_frequency_l3387_338719


namespace NUMINAMATH_CALUDE_sqrt_14_plus_2_bounds_l3387_338753

theorem sqrt_14_plus_2_bounds : 5 < Real.sqrt 14 + 2 ∧ Real.sqrt 14 + 2 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_plus_2_bounds_l3387_338753


namespace NUMINAMATH_CALUDE_count_even_one_matrices_l3387_338706

/-- The number of m × n matrices with entries 0 or 1, where the number of 1's in each row and column is even -/
def evenOneMatrices (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem stating that the number of m × n matrices with entries 0 or 1, 
    where the number of 1's in each row and column is even, is 2^((m-1)(n-1)) -/
theorem count_even_one_matrices (m n : ℕ) :
  evenOneMatrices m n = 2^((m-1)*(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_count_even_one_matrices_l3387_338706


namespace NUMINAMATH_CALUDE_point_p_coordinates_l3387_338746

/-- Given points A, B, C in ℝ³ and a point P such that vector AP is half of vector CB,
    prove that P has the specified coordinates. -/
theorem point_p_coordinates (A B C P : ℝ × ℝ × ℝ) : 
  A = (2, -1, 2) → 
  B = (4, 5, -1) → 
  C = (-2, 2, 3) → 
  P - A = (1/2 : ℝ) • (B - C) → 
  P = (5, 1/2, 0) := by
sorry


end NUMINAMATH_CALUDE_point_p_coordinates_l3387_338746


namespace NUMINAMATH_CALUDE_sisyphus_earning_zero_l3387_338766

/-- Represents the state of the stone boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  x : ℤ  -- Sisyphus's earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AB : Move  -- from A to B
  | AC : Move  -- from A to C
  | BA : Move  -- from B to A
  | BC : Move  -- from B to C
  | CA : Move  -- from C to A
  | CB : Move  -- from C to B

/-- Applies a move to a BoxState -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      x := state.x + (state.a - state.b - 1) }
  | Move.AC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      x := state.x + (state.a - state.c - 1) }
  | Move.BA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      x := state.x + (state.b - state.a - 1) }
  | Move.BC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      x := state.x + (state.b - state.c - 1) }
  | Move.CA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      x := state.x + (state.c - state.a - 1) }
  | Move.CB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      x := state.x + (state.c - state.b - 1) }

/-- Theorem: The greatest possible earning of Sisyphus is 0 -/
theorem sisyphus_earning_zero 
  (initial : BoxState) 
  (moves : List Move) 
  (h1 : moves.length = 24 * 365 * 1000) -- 1000 years of hourly moves
  (h2 : (moves.foldl applyMove initial).a = initial.a) -- stones return to initial state
  (h3 : (moves.foldl applyMove initial).b = initial.b)
  (h4 : (moves.foldl applyMove initial).c = initial.c) :
  (moves.foldl applyMove initial).x ≤ 0 :=
sorry

#check sisyphus_earning_zero

end NUMINAMATH_CALUDE_sisyphus_earning_zero_l3387_338766


namespace NUMINAMATH_CALUDE_prime_power_sum_product_l3387_338741

theorem prime_power_sum_product (p : ℕ) : 
  Prime p → 
  (∃ x y z : ℕ, ∃ q r s : ℕ, 
    Prime q ∧ Prime r ∧ Prime s ∧ 
    q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    x^p + y^p + z^p - x - y - z = q * r * s) ↔ 
  p = 2 ∨ p = 3 ∨ p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_product_l3387_338741


namespace NUMINAMATH_CALUDE_unique_dataset_l3387_338788

def is_valid_dataset (x : Fin 4 → ℕ) : Prop :=
  (∀ i, x i > 0) ∧
  (x 0 ≤ x 1) ∧ (x 1 ≤ x 2) ∧ (x 2 ≤ x 3) ∧
  (x 0 + x 1 + x 2 + x 3 = 8) ∧
  ((x 1 + x 2) / 2 = 2) ∧
  ((x 0 - 2)^2 + (x 1 - 2)^2 + (x 2 - 2)^2 + (x 3 - 2)^2 = 4)

theorem unique_dataset :
  ∀ x : Fin 4 → ℕ, is_valid_dataset x → (x 0 = 1 ∧ x 1 = 1 ∧ x 2 = 3 ∧ x 3 = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_dataset_l3387_338788


namespace NUMINAMATH_CALUDE_odd_power_function_l3387_338730

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m

theorem odd_power_function (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) m, f m x = -f m (-x)) →
  f m (m + 1) = 8 :=
by sorry

end NUMINAMATH_CALUDE_odd_power_function_l3387_338730


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l3387_338708

/-- An arithmetic sequence {a_n} with a common ratio q -/
def ArithmeticSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_increasing : ∀ n, a (n + 1) > a n)
  (h_positive : a 1 > 0)
  (h_condition : ∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1))
  (h_arithmetic : ArithmeticSequence a q) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l3387_338708


namespace NUMINAMATH_CALUDE_find_divisor_l3387_338723

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) : 
  dividend = quotient * divisor + remainder → 
  dividend = 22 → 
  quotient = 7 → 
  remainder = 1 → 
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3387_338723


namespace NUMINAMATH_CALUDE_dodecagon_enclosure_l3387_338758

theorem dodecagon_enclosure (m : ℕ) (n : ℕ) : 
  m = 12 →
  (360 : ℝ) / n = (180 : ℝ) - (m - 2 : ℝ) * 180 / m →
  n = 6 :=
sorry

end NUMINAMATH_CALUDE_dodecagon_enclosure_l3387_338758


namespace NUMINAMATH_CALUDE_percentage_calculation_l3387_338763

theorem percentage_calculation (x y z : ℝ) 
  (hx : 0.2 * x = 200)
  (hy : 0.3 * y = 150)
  (hz : 0.4 * z = 80) :
  (0.9 * x - 0.6 * y) + 0.5 * (x + y + z) = 1450 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3387_338763


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3387_338715

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 10) = 12 → y = 134 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3387_338715


namespace NUMINAMATH_CALUDE_steve_height_l3387_338770

/-- Converts feet and inches to total inches -/
def feet_to_inches (feet : ℕ) (inches : ℕ) : ℕ := feet * 12 + inches

/-- Calculates final height after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_to_inches initial_feet initial_inches + growth

theorem steve_height :
  final_height 5 6 6 = 72 := by sorry

end NUMINAMATH_CALUDE_steve_height_l3387_338770


namespace NUMINAMATH_CALUDE_pen_count_is_31_l3387_338702

/-- The number of pens after a series of events --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Theorem stating that given the initial conditions, the final number of pens is 31 --/
theorem pen_count_is_31 : final_pen_count 5 20 2 19 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_count_is_31_l3387_338702


namespace NUMINAMATH_CALUDE_sum_coordinates_point_D_l3387_338705

/-- Given a point N which is the midpoint of segment CD, and point C,
    prove that the sum of coordinates of point D is 5. -/
theorem sum_coordinates_point_D (N C D : ℝ × ℝ) : 
  N = (3, 5) →
  C = (1, 10) →
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_point_D_l3387_338705


namespace NUMINAMATH_CALUDE_accidental_addition_l3387_338720

theorem accidental_addition (x : ℕ) : x + 65 = 125 → x + 95 = 155 := by
  sorry

end NUMINAMATH_CALUDE_accidental_addition_l3387_338720


namespace NUMINAMATH_CALUDE_sum_of_digits_n_l3387_338773

/-- The least 7-digit number that leaves a remainder of 4 when divided by 5, 850, 35, 27, and 90 -/
def n : ℕ := sorry

/-- Condition: n is a 7-digit number -/
axiom n_seven_digits : 1000000 ≤ n ∧ n < 10000000

/-- Condition: n leaves a remainder of 4 when divided by 5 -/
axiom n_mod_5 : n % 5 = 4

/-- Condition: n leaves a remainder of 4 when divided by 850 -/
axiom n_mod_850 : n % 850 = 4

/-- Condition: n leaves a remainder of 4 when divided by 35 -/
axiom n_mod_35 : n % 35 = 4

/-- Condition: n leaves a remainder of 4 when divided by 27 -/
axiom n_mod_27 : n % 27 = 4

/-- Condition: n leaves a remainder of 4 when divided by 90 -/
axiom n_mod_90 : n % 90 = 4

/-- Condition: n is the least number satisfying all the above conditions -/
axiom n_least : ∀ m : ℕ, (1000000 ≤ m ∧ m < 10000000 ∧ 
                          m % 5 = 4 ∧ m % 850 = 4 ∧ m % 35 = 4 ∧ m % 27 = 4 ∧ m % 90 = 4) → n ≤ m

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of n is 22 -/
theorem sum_of_digits_n : sum_of_digits n = 22 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_n_l3387_338773


namespace NUMINAMATH_CALUDE_min_hours_is_eight_l3387_338783

/-- Represents Biff's expenses and earnings during the bus trip -/
structure BusTrip where
  ticket : ℕ
  snacks : ℕ
  headphones : ℕ
  lunch : ℕ
  dinner : ℕ
  accommodation : ℕ
  hourly_rate : ℕ
  day_wifi_rate : ℕ
  night_wifi_rate : ℕ

/-- Calculates the total fixed expenses for the trip -/
def total_fixed_expenses (trip : BusTrip) : ℕ :=
  trip.ticket + trip.snacks + trip.headphones + trip.lunch + trip.dinner + trip.accommodation

/-- Calculates the minimum number of hours needed to break even -/
def min_hours_to_break_even (trip : BusTrip) : ℕ :=
  (total_fixed_expenses trip + trip.night_wifi_rate - 1) / (trip.hourly_rate - trip.night_wifi_rate) + 1

/-- Theorem stating that the minimum number of hours to break even is 8 -/
theorem min_hours_is_eight (trip : BusTrip)
  (h1 : trip.ticket = 11)
  (h2 : trip.snacks = 3)
  (h3 : trip.headphones = 16)
  (h4 : trip.lunch = 8)
  (h5 : trip.dinner = 10)
  (h6 : trip.accommodation = 35)
  (h7 : trip.hourly_rate = 12)
  (h8 : trip.day_wifi_rate = 2)
  (h9 : trip.night_wifi_rate = 1) :
  min_hours_to_break_even trip = 8 := by
  sorry

#eval min_hours_to_break_even {
  ticket := 11,
  snacks := 3,
  headphones := 16,
  lunch := 8,
  dinner := 10,
  accommodation := 35,
  hourly_rate := 12,
  day_wifi_rate := 2,
  night_wifi_rate := 1
}

end NUMINAMATH_CALUDE_min_hours_is_eight_l3387_338783


namespace NUMINAMATH_CALUDE_inequality_proof_l3387_338726

theorem inequality_proof (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) ≥ 12 ∧
  ((a * b / (c - 1)) + (b * c / (a - 1)) + (c * a / (b - 1)) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3387_338726


namespace NUMINAMATH_CALUDE_writing_time_for_three_books_l3387_338764

/-- Calculates the number of days required to write multiple books given the daily writing rate and book length. -/
def days_to_write_books (pages_per_day : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  (pages_per_book * num_books) / pages_per_day

/-- Theorem stating that it takes 60 days to write 3 books of 400 pages each at a rate of 20 pages per day. -/
theorem writing_time_for_three_books :
  days_to_write_books 20 400 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_writing_time_for_three_books_l3387_338764


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3387_338774

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b)
  (h4 : (a + b) / 2 = 7 / 2) (h5 : Real.sqrt (a * b) = 2 * Real.sqrt 3) :
  let c := Real.sqrt (a^2 + b^2)
  Real.sqrt (c^2 - b^2) / b = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3387_338774


namespace NUMINAMATH_CALUDE_songs_ratio_l3387_338752

def initial_songs : ℕ := 54
def deleted_songs : ℕ := 9

theorem songs_ratio :
  let kept_songs := initial_songs - deleted_songs
  let ratio := kept_songs / deleted_songs
  ratio = 5 := by sorry

end NUMINAMATH_CALUDE_songs_ratio_l3387_338752


namespace NUMINAMATH_CALUDE_find_n_l3387_338737

theorem find_n : ∃ n : ℝ, (256 : ℝ)^(1/4) = 4^n ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_find_n_l3387_338737


namespace NUMINAMATH_CALUDE_factor_expression_value_l3387_338759

theorem factor_expression_value (k m n : ℕ) : 
  k > 1 → m > 1 → n > 1 →
  (∃ (Z : ℕ), Z = 2^k * 3^m * 5^n ∧ (2^60 * 3^35 * 5^20 * 7^7) % Z = 0) →
  ∃ (k' m' n' : ℕ), k' > 1 ∧ m' > 1 ∧ n' > 1 ∧
    2^k' + 3^m' + k'^3 * m'^n' - n' = 43 :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_value_l3387_338759


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3387_338794

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) :
  (360 : ℝ) / n = 60 → n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3387_338794


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3387_338718

/-- The y-intercept of the line 2x + 7y = 35 is (0, 5) -/
theorem y_intercept_of_line (x y : ℝ) :
  2 * x + 7 * y = 35 → y = 5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3387_338718


namespace NUMINAMATH_CALUDE_polynomial_division_result_l3387_338799

theorem polynomial_division_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (25 * x^2 * y - 5 * x * y^2) / (5 * x * y) = 5 * x - y := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l3387_338799


namespace NUMINAMATH_CALUDE_line_slope_l3387_338772

/-- A straight line in the xy-plane with y-intercept 4 and passing through (199, 800) has slope 4 -/
theorem line_slope (m : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + 4) ∧ f 199 = 800) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l3387_338772


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3387_338721

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 1) < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -2 < x ∧ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3387_338721


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3387_338795

/-- 
Given an arithmetic progression where a₁₂, a₁₃, a₁₅ are the 12th, 13th, and 15th terms respectively,
and their squares form a geometric progression with common ratio q,
prove that q must be one of: 4, 4 - 2√3, 4 + 2√3, or 9/25.
-/
theorem arithmetic_geometric_progression (a₁₂ a₁₃ a₁₅ d q : ℝ) : 
  (a₁₃ = a₁₂ + d ∧ a₁₅ = a₁₃ + 2*d) →  -- arithmetic progression condition
  (a₁₃^2)^2 = a₁₂^2 * a₁₅^2 →  -- geometric progression condition
  (q = (a₁₃^2 / a₁₂^2)) →  -- definition of q
  (q = 4 ∨ q = 4 - 2*Real.sqrt 3 ∨ q = 4 + 2*Real.sqrt 3 ∨ q = 9/25) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l3387_338795


namespace NUMINAMATH_CALUDE_arithmetic_seq_problem_l3387_338722

def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_seq_problem (a : ℕ → ℚ) :
  is_arithmetic_seq a →
  a 4 + a 5 + a 6 + a 7 = 56 →
  a 4 * a 7 = 187 →
  ((∃ a₁ d, ∀ n, a n = a₁ + (n - 1) * d) ∧ 
   ((a 1 = 5 ∧ ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 2) ∨
    (a 1 = 23 ∧ ∃ d, ∀ n, a n = 23 + (n - 1) * d ∧ d = -2))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_seq_problem_l3387_338722


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3387_338780

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the areas of the smaller triangles
def small_triangle_areas (T : Triangle) : ℕ × ℕ × ℕ := (16, 25, 64)

-- Define the theorem
theorem triangle_area_theorem (T : Triangle) : 
  let (a1, a2, a3) := small_triangle_areas T
  (a1 : ℝ) + a2 + a3 > 0 →
  (∃ (l1 l2 l3 : ℝ), l1 > 0 ∧ l2 > 0 ∧ l3 > 0 ∧ 
    l1^2 = a1 ∧ l2^2 = a2 ∧ l3^2 = a3) →
  (∃ (A : ℝ), A = (l1 + l2 + l3)^2 * (a1 + a2 + a3) / (l1^2 + l2^2 + l3^2)) →
  A = 30345 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l3387_338780


namespace NUMINAMATH_CALUDE_sqrt_6_irrational_l3387_338751

theorem sqrt_6_irrational : Irrational (Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_irrational_l3387_338751


namespace NUMINAMATH_CALUDE_nell_initial_cards_l3387_338792

/-- Nell's initial number of baseball cards -/
def initial_cards : ℕ := sorry

/-- Number of cards Nell gave to Jeff -/
def cards_given : ℕ := 28

/-- Number of cards Nell has left -/
def cards_left : ℕ := 276

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : initial_cards = 304 := by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l3387_338792


namespace NUMINAMATH_CALUDE_no_rational_roots_for_odd_m_n_l3387_338724

theorem no_rational_roots_for_odd_m_n (m n : ℤ) (hm : Odd m) (hn : Odd n) :
  ∀ x : ℚ, x^2 + 2*m*x + 2*n ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_odd_m_n_l3387_338724


namespace NUMINAMATH_CALUDE_todd_repayment_l3387_338789

/-- Calculates the amount Todd repaid his brother --/
def amount_repaid (loan : ℝ) (ingredients_cost : ℝ) (snow_cones_sold : ℕ) (price_per_snow_cone : ℝ) (remaining_money : ℝ) : ℝ :=
  (snow_cones_sold : ℝ) * price_per_snow_cone - ingredients_cost + loan - remaining_money

/-- Proves that Todd repaid his brother $110 --/
theorem todd_repayment : 
  amount_repaid 100 75 200 0.75 65 = 110 := by
  sorry

#eval amount_repaid 100 75 200 0.75 65

end NUMINAMATH_CALUDE_todd_repayment_l3387_338789


namespace NUMINAMATH_CALUDE_lattice_triangle_area_bound_l3387_338731

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Checks if a point is on the edge of a triangle -/
def isOnEdge (p : LatticePoint) (t : LatticeTriangle) : Prop := sorry

/-- Counts the number of lattice points inside a triangle -/
def interiorPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Counts the number of lattice points on the edges of a triangle -/
def boundaryPointCount (t : LatticeTriangle) : ℕ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : LatticeTriangle) : ℚ := sorry

/-- Theorem: The area of a lattice triangle with exactly one interior lattice point is at most 9/2 -/
theorem lattice_triangle_area_bound (t : LatticeTriangle) 
  (h : interiorPointCount t = 1) : 
  triangleArea t ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_lattice_triangle_area_bound_l3387_338731


namespace NUMINAMATH_CALUDE_fraction_simplification_l3387_338734

theorem fraction_simplification :
  (2 - 4 + 8 - 16 + 32 - 64 + 128 - 256) / (4 - 8 + 16 - 32 + 64 - 128 + 256 - 512) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3387_338734


namespace NUMINAMATH_CALUDE_root_value_l3387_338781

theorem root_value (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 2*x + a = 0 ↔ x = x₁ ∨ x = x₂) → 
  (x₁ + 2*x₂ = 3 - Real.sqrt 2) →
  x₂ = 1 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_root_value_l3387_338781


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l3387_338742

theorem sqrt_sum_comparison : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l3387_338742


namespace NUMINAMATH_CALUDE_solutions_equality_l3387_338704

-- Define a as a positive real number
variable (a : ℝ) (ha : a > 0)

-- Define the condition that 10 < a^x < 100 has exactly five solutions in natural numbers
def has_five_solutions (a : ℝ) : Prop :=
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (10 < a^x ∧ a^x < 100))

-- Theorem statement
theorem solutions_equality (h : has_five_solutions a) :
  ∃ (s : Finset ℕ), s.card = 5 ∧ ∀ x : ℕ, x ∈ s ↔ (100 < a^x ∧ a^x < 1000) :=
sorry

end NUMINAMATH_CALUDE_solutions_equality_l3387_338704


namespace NUMINAMATH_CALUDE_cross_product_example_l3387_338710

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => v 1 * w 2 - v 2 * w 1
  | 1 => v 2 * w 0 - v 0 * w 2
  | 2 => v 0 * w 1 - v 1 * w 0

/-- The first vector -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => -3
  | 1 => 4
  | 2 => 5

/-- The second vector -/
def w : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 2
  | 1 => -1
  | 2 => 4

theorem cross_product_example : cross_product v w = fun i =>
  match i with
  | 0 => 21
  | 1 => 22
  | 2 => -5 := by sorry

end NUMINAMATH_CALUDE_cross_product_example_l3387_338710


namespace NUMINAMATH_CALUDE_balloon_count_l3387_338750

/-- The number of blue balloons after a series of events --/
def total_balloons (joan_initial : ℕ) (joan_popped : ℕ) (jessica_initial : ℕ) (jessica_inflated : ℕ) (peter_initial : ℕ) (peter_deflated : ℕ) : ℕ :=
  (joan_initial - joan_popped) + (jessica_initial + jessica_inflated) + (peter_initial - peter_deflated)

/-- Theorem stating the total number of balloons after the given events --/
theorem balloon_count :
  total_balloons 9 5 2 3 4 2 = 11 := by
  sorry

#eval total_balloons 9 5 2 3 4 2

end NUMINAMATH_CALUDE_balloon_count_l3387_338750


namespace NUMINAMATH_CALUDE_exists_larger_area_figure_l3387_338739

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  -- We don't need to define the internal structure for this problem
  area : ℝ
  perimeter : ℝ

/-- A chord of a convex figure -/
structure Chord (F : ConvexFigure) where
  -- We don't need to define the internal structure for this problem
  dividesPerimeterInHalf : Bool
  dividesAreaUnequally : Bool

/-- Theorem: If a convex figure has a chord that divides its perimeter in half
    and its area unequally, then there exists another figure with the same
    perimeter but larger area -/
theorem exists_larger_area_figure (F : ConvexFigure) 
  (h : ∃ c : Chord F, c.dividesPerimeterInHalf ∧ c.dividesAreaUnequally) :
  ∃ G : ConvexFigure, G.perimeter = F.perimeter ∧ G.area > F.area :=
by sorry

end NUMINAMATH_CALUDE_exists_larger_area_figure_l3387_338739


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l3387_338775

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

theorem dog_tricks_conversion :
  base9ToBase10 [1, 2, 5] = 424 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l3387_338775


namespace NUMINAMATH_CALUDE_line_equation_correct_l3387_338733

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation 3x - y + 4 = 0 represents a line with slope 3 and y-intercept 4 -/
theorem line_equation_correct (l : Line) (eq : LineEquation) :
  l.slope = 3 ∧ l.intercept = 4 ∧ 
  eq.a = 3 ∧ eq.b = -1 ∧ eq.c = 4 →
  eq.a * x + eq.b * y + eq.c = 0 ↔ y = l.slope * x + l.intercept :=
by sorry

end NUMINAMATH_CALUDE_line_equation_correct_l3387_338733


namespace NUMINAMATH_CALUDE_amount_equals_scientific_notation_l3387_338754

/-- Represents the amount in yuan -/
def amount : ℝ := 2.51e6

/-- Represents the scientific notation of the amount -/
def scientific_notation : ℝ := 2.51 * (10 ^ 6)

/-- Theorem stating that the amount is equal to its scientific notation representation -/
theorem amount_equals_scientific_notation : amount = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_amount_equals_scientific_notation_l3387_338754


namespace NUMINAMATH_CALUDE_log_equation_solution_l3387_338749

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution :
  ∃ (x : ℝ), log (2^x) (3^20) = log (2^(x+3)) (3^2020) → x = 3/100 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3387_338749


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l3387_338714

theorem geometric_sequence_b_value (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ r : ℝ, r > 0 ∧ b = 30 * r ∧ 9/4 = b * r) : 
  b = 3 * Real.sqrt 30 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l3387_338714


namespace NUMINAMATH_CALUDE_stamps_per_page_l3387_338713

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end NUMINAMATH_CALUDE_stamps_per_page_l3387_338713


namespace NUMINAMATH_CALUDE_profit_percentage_formula_l3387_338755

theorem profit_percentage_formula (C S M n : ℝ) (P : ℝ) 
  (h1 : S > 0) 
  (h2 : C > 0)
  (h3 : n > 0)
  (h4 : M = (2 / n) * C) 
  (h5 : P = (M / S) * 100) :
  P = 200 / (n + 2) := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_formula_l3387_338755


namespace NUMINAMATH_CALUDE_fixed_point_on_curve_l3387_338701

-- Define the curve equation
def curve_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + (4*k + 10)*y + 10*k + 20 = 0

-- Theorem statement
theorem fixed_point_on_curve :
  ∀ k : ℝ, k ≠ -1 → curve_equation k 1 (-3) :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_curve_l3387_338701


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3387_338761

theorem ac_squared_gt_bc_squared_implies_a_gt_b (a b c : ℝ) :
  a * c^2 > b * c^2 → a > b := by sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_implies_a_gt_b_l3387_338761


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l3387_338790

/-- Proves that the rate of a stream is 5 km/hr given the boat's speed in still water,
    distance traveled downstream, and time taken. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 16 →
  distance = 84 →
  time = 4 →
  ∃ stream_rate : ℝ, 
    stream_rate = 5 ∧
    distance = (boat_speed + stream_rate) * time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_rate_calculation_l3387_338790


namespace NUMINAMATH_CALUDE_container_volume_maximized_l3387_338769

/-- The total length of the steel bar used to make the container frame -/
def total_length : ℝ := 14.8

/-- The function representing the volume of the container -/
def volume (width : ℝ) : ℝ :=
  width * (width + 0.5) * (3.2 - 2 * width)

/-- The width that maximizes the container's volume -/
def optimal_width : ℝ := 1

theorem container_volume_maximized :
  ∀ w : ℝ, 0 < w → w < 1.6 → volume w ≤ volume optimal_width :=
sorry

end NUMINAMATH_CALUDE_container_volume_maximized_l3387_338769


namespace NUMINAMATH_CALUDE_can_meet_in_three_jumps_l3387_338776

-- Define the grid
def Grid := ℤ × ℤ

-- Define the color of a square
inductive Color
| Red
| White

-- Define the coloring function
def coloring : Grid → Color := sorry

-- Define the grasshopper's position
def grasshopper : Grid := sorry

-- Define the flea's position
def flea : Grid := sorry

-- Define a valid jump
def valid_jump (start finish : Grid) (c : Color) : Prop :=
  (coloring start = c ∧ coloring finish = c) ∧
  (start.1 = finish.1 ∨ start.2 = finish.2)

-- Define adjacent squares
def adjacent (a b : Grid) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) = 1)

-- Main theorem
theorem can_meet_in_three_jumps :
  ∃ (g1 g2 g3 f1 f2 f3 : Grid),
    (valid_jump grasshopper g1 Color.Red ∨ g1 = grasshopper) ∧
    (valid_jump g1 g2 Color.Red ∨ g2 = g1) ∧
    (valid_jump g2 g3 Color.Red ∨ g3 = g2) ∧
    (valid_jump flea f1 Color.White ∨ f1 = flea) ∧
    (valid_jump f1 f2 Color.White ∨ f2 = f1) ∧
    (valid_jump f2 f3 Color.White ∨ f3 = f2) ∧
    adjacent g3 f3 :=
  sorry


end NUMINAMATH_CALUDE_can_meet_in_three_jumps_l3387_338776


namespace NUMINAMATH_CALUDE_triangle_property_l3387_338725

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : (3 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 2) : 
  Real.cos t.A = 1/3 ∧ 
  ∃ (p : ℝ), p = t.a + t.b + t.c ∧ 
  p ≥ 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ∧
  (p = 2 * Real.sqrt 6 + 2 * Real.sqrt 2 ↔ t.a = t.b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3387_338725


namespace NUMINAMATH_CALUDE_opposite_of_four_l3387_338738

-- Define the concept of opposite number
def opposite (x : ℝ) : ℝ := -x

-- Theorem stating that the opposite of 4 is -4
theorem opposite_of_four : opposite 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_four_l3387_338738


namespace NUMINAMATH_CALUDE_game_winnable_iff_k_leq_n_minus_one_l3387_338782

/-- Represents the game state -/
structure GameState :=
  (k : ℕ)
  (n : ℕ)
  (h1 : 2 ≤ k)
  (h2 : k ≤ n)

/-- Predicate to determine if the game is winnable -/
def is_winnable (g : GameState) : Prop :=
  g.k ≤ g.n - 1

/-- Theorem stating the condition for the game to be winnable -/
theorem game_winnable_iff_k_leq_n_minus_one (g : GameState) :
  is_winnable g ↔ g.k ≤ g.n - 1 :=
sorry

end NUMINAMATH_CALUDE_game_winnable_iff_k_leq_n_minus_one_l3387_338782


namespace NUMINAMATH_CALUDE_factorial_ratio_l3387_338700

theorem factorial_ratio (N : ℕ) (h : N > 1) :
  (Nat.factorial (N^2 - 1)) / ((Nat.factorial (N + 1))^2) = 
  (Nat.factorial (N - 1)) / (N + 1) :=
sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3387_338700


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3387_338777

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_angle_C (t : Triangle) 
  (h : t.b * (2 * sin t.B + sin t.A) + (2 * t.a + t.b) * sin t.A = 2 * t.c * sin t.C) :
  t.C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3387_338777


namespace NUMINAMATH_CALUDE_min_value_and_location_l3387_338760

theorem min_value_and_location (f : ℝ → ℝ) :
  (∀ x, f x = 2 * Real.sin x ^ 4 + 2 * Real.cos x ^ 4 + Real.cos (2 * x) ^ 2 - 3) →
  (∃ x_min ∈ Set.Icc (π / 16) (3 * π / 16), 
    (∀ x ∈ Set.Icc (π / 16) (3 * π / 16), f x_min ≤ f x) ∧
    x_min = 3 * π / 16 ∧
    f x_min = -(Real.sqrt 2 + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_location_l3387_338760


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3387_338728

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 2*x + 5)}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 4}

-- Define the universal set U
def U : Type := ℝ

-- State the theorem
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = Set.Ioo (-1 : ℝ) (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3387_338728


namespace NUMINAMATH_CALUDE_test_results_l3387_338745

/-- The probability of exactly two people meeting the standard in a test where
    A, B, and C have independent probabilities of 2/5, 3/4, and 1/2 respectively. -/
def prob_two_meet_standard : ℚ := 17/40

/-- The most likely number of people to meet the standard in the test. -/
def most_likely_number : ℕ := 2

/-- Probabilities of A, B, and C meeting the standard -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/2

theorem test_results :
  (prob_two_meet_standard = prob_A * prob_B * (1 - prob_C) +
                            prob_A * (1 - prob_B) * prob_C +
                            (1 - prob_A) * prob_B * prob_C) ∧
  (most_likely_number = 2) := by
  sorry

end NUMINAMATH_CALUDE_test_results_l3387_338745


namespace NUMINAMATH_CALUDE_area_of_triangle_DEF_l3387_338735

-- Define the square PQRS
def square_PQRS : Real := 36

-- Define the side length of the smaller squares
def small_square_side : Real := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : Real
  EF : Real
  isIsosceles : DE = DF

-- Define the folding property
def folding_property (t : Triangle_DEF) : Prop :=
  ∃ (center : Real), 
    center = (square_PQRS.sqrt / 2) ∧
    t.DE = center + 2 * small_square_side

-- Theorem statement
theorem area_of_triangle_DEF : 
  ∀ (t : Triangle_DEF), 
    folding_property t → 
    (1/2 : Real) * t.EF * t.DE = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_DEF_l3387_338735


namespace NUMINAMATH_CALUDE_road_length_calculation_l3387_338793

/-- Given a map scale and a road length on the map, calculate the actual road length in kilometers. -/
def actual_road_length (map_scale : ℚ) (map_length : ℚ) : ℚ :=
  map_length * map_scale / 100000

/-- Theorem stating that for a map scale of 1:2500000 and a road length of 6 cm on the map,
    the actual length of the road is 150 km. -/
theorem road_length_calculation :
  let map_scale : ℚ := 2500000
  let map_length : ℚ := 6
  actual_road_length map_scale map_length = 150 := by
  sorry

#eval actual_road_length 2500000 6

end NUMINAMATH_CALUDE_road_length_calculation_l3387_338793


namespace NUMINAMATH_CALUDE_number_to_add_divisibility_l3387_338748

theorem number_to_add_divisibility (p q : ℕ) (n m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p = 563 → q = 839 → n = 1398547 → m = 18284 →
  (p * q) ∣ (n + m) :=
by sorry

end NUMINAMATH_CALUDE_number_to_add_divisibility_l3387_338748


namespace NUMINAMATH_CALUDE_mountain_trail_length_l3387_338732

/-- Represents the hike of Phoenix on the Mountain Trail --/
structure MountainTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of Phoenix's hike --/
def HikeConditions (hike : MountainTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day4 + hike.day5 = 34 ∧
  hike.day1 + hike.day3 = 32

/-- Theorem: The total length of the Mountain Trail is 94 miles --/
theorem mountain_trail_length (hike : MountainTrail) 
  (h : HikeConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 94 := by
  sorry


end NUMINAMATH_CALUDE_mountain_trail_length_l3387_338732


namespace NUMINAMATH_CALUDE_tangent_slope_circle_l3387_338786

/-- Given a circle with center (2,3) and a point (8,7) on the circle,
    the slope of the line tangent to the circle at (8,7) is -3/2. -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (2, 3) →
  point = (8, 7) →
  (((point.2 - center.2) / (point.1 - center.1)) * (-1 / ((point.2 - center.2) / (point.1 - center.1)))) = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_circle_l3387_338786


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_negative_l3387_338762

/-- Given a triangle with side lengths a, b, and c, 
    the expression a^2 - c^2 - 2ab + b^2 is always negative. -/
theorem triangle_inequality_expression_negative 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - c^2 - 2*a*b + b^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_negative_l3387_338762


namespace NUMINAMATH_CALUDE_cubic_sum_coefficients_l3387_338729

/-- A cubic function f(x) = ax^3 + bx^2 + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_sum_coefficients (a b c d : ℝ) :
  (∀ x, cubic_function a b c d (x + 2) = 2 * x^3 - x^2 + 5 * x + 3) →
  a + b + c + d = -5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_coefficients_l3387_338729


namespace NUMINAMATH_CALUDE_sanchez_sum_problem_l3387_338778

theorem sanchez_sum_problem (x y : ℕ+) : x - y = 5 → x * y = 84 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_sanchez_sum_problem_l3387_338778


namespace NUMINAMATH_CALUDE_function_types_l3387_338717

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2
def p (x : ℝ) (a : ℝ) : ℝ := a * x^2

-- State the theorem
theorem function_types (a x : ℝ) (ha : a ≠ 0) (hx : x ≠ 0) :
  (∃ b c : ℝ, ∀ x, f a x = x^2 + b*x + c) ∧
  (∃ m b : ℝ, ∀ a, p x a = m*a + b) :=
sorry

end NUMINAMATH_CALUDE_function_types_l3387_338717


namespace NUMINAMATH_CALUDE_function_value_determines_parameter_l3387_338787

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3^x + 1 else x^2 + a*x

theorem function_value_determines_parameter (a : ℝ) : f a (f a 0) = 6 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_determines_parameter_l3387_338787


namespace NUMINAMATH_CALUDE_all_lines_through_single_point_l3387_338707

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for lines
structure Line where
  color : Color

-- Define a type for points
structure Point where

-- Define the plane
structure Plane where
  lines : Finset Line
  points : Set Point

-- Define the property that no lines are parallel
def NoParallelLines (p : Plane) : Prop :=
  ∀ l1 l2 : Line, l1 ∈ p.lines → l2 ∈ p.lines → l1 ≠ l2 → ∃ pt : Point, pt ∈ p.points

-- Define the property that through each intersection point of same-color lines passes a line of the other color
def IntersectionProperty (p : Plane) : Prop :=
  ∀ pt : Point, pt ∈ p.points →
    ∀ l1 l2 : Line, l1 ∈ p.lines → l2 ∈ p.lines → l1.color = l2.color →
      ∃ l3 : Line, l3 ∈ p.lines ∧ l3.color ≠ l1.color

-- The main theorem
theorem all_lines_through_single_point (p : Plane) 
  (h1 : Finite p.lines)
  (h2 : NoParallelLines p)
  (h3 : IntersectionProperty p) :
  ∃ pt : Point, ∀ l : Line, l ∈ p.lines → pt ∈ p.points :=
sorry

end NUMINAMATH_CALUDE_all_lines_through_single_point_l3387_338707


namespace NUMINAMATH_CALUDE_sara_pumpkins_l3387_338796

def pumpkins_grown : ℕ := 43
def pumpkins_eaten : ℕ := 23

theorem sara_pumpkins : pumpkins_grown - pumpkins_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_sara_pumpkins_l3387_338796


namespace NUMINAMATH_CALUDE_bailey_shot_percentage_l3387_338703

theorem bailey_shot_percentage (total_shots : ℕ) (scored_shots : ℕ) 
  (h1 : total_shots = 8) (h2 : scored_shots = 6) : 
  (1 - scored_shots / total_shots) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bailey_shot_percentage_l3387_338703


namespace NUMINAMATH_CALUDE_hash_solution_l3387_338711

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that 2 is the number that satisfies x # 3 = 12 -/
theorem hash_solution : ∃ x : ℝ, hash x 3 = 12 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_hash_solution_l3387_338711


namespace NUMINAMATH_CALUDE_smallest_share_amount_l3387_338709

def total_amount : ℝ := 500
def give_away_percentage : ℝ := 0.60
def friend_count : ℕ := 5
def shares : List ℝ := [0.30, 0.25, 0.20, 0.15, 0.10]

theorem smallest_share_amount :
  let amount_to_distribute := total_amount * give_away_percentage
  let smallest_share := shares.minimum?
  smallest_share.map (λ s => s * amount_to_distribute) = some 30 := by sorry

end NUMINAMATH_CALUDE_smallest_share_amount_l3387_338709


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3387_338743

def plane_equation (x y z : ℝ) : ℝ := 3 * x - y + 2 * z - 11

theorem plane_equation_correct :
  ∃ (A B C D : ℤ),
    (∀ (s t : ℝ),
      plane_equation (2 + 2*s - 2*t) (3 - 2*s) (4 - s + 3*t) = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    ∀ (x y z : ℝ), A * x + B * y + C * z + D = plane_equation x y z := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3387_338743


namespace NUMINAMATH_CALUDE_sugar_in_recipe_l3387_338779

/-- Given a cake recipe with specific flour requirements and a relation between
    sugar and remaining flour, this theorem proves the amount of sugar needed. -/
theorem sugar_in_recipe (total_flour remaining_flour sugar : ℕ) : 
  total_flour = 14 →
  remaining_flour = total_flour - 4 →
  remaining_flour = sugar + 1 →
  sugar = 9 := by
  sorry

end NUMINAMATH_CALUDE_sugar_in_recipe_l3387_338779


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3387_338757

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset Int),
    (∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0) ∧
    (∀ n, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 → n ∈ S) ∧
    S.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3387_338757
