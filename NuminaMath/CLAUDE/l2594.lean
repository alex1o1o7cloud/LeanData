import Mathlib

namespace contrapositive_square_sum_zero_l2594_259461

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_square_sum_zero_l2594_259461


namespace jills_herd_sale_fraction_l2594_259434

/-- Represents the number of llamas in Jill's herd -/
structure LlamaHerd where
  initial : ℕ
  single_births : ℕ
  twin_births : ℕ
  traded_calves : ℕ
  traded_adults : ℕ
  final : ℕ

/-- Calculates the fraction of the herd sold at the market -/
def fraction_sold (herd : LlamaHerd) : ℚ :=
  let total_calves := herd.single_births + 2 * herd.twin_births
  let before_trade := herd.initial + total_calves
  let after_trade := before_trade - herd.traded_calves + herd.traded_adults
  let sold := after_trade - herd.final
  sold / before_trade

/-- Theorem stating the fraction of the herd Jill sold at the market -/
theorem jills_herd_sale_fraction : 
  ∀ (herd : LlamaHerd), 
  herd.single_births = 9 → 
  herd.twin_births = 5 → 
  herd.traded_calves = 8 → 
  herd.traded_adults = 2 → 
  herd.final = 18 → 
  fraction_sold herd = 4 / 13 := by
  sorry


end jills_herd_sale_fraction_l2594_259434


namespace haris_capital_contribution_l2594_259453

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℕ
  months : ℕ

/-- Calculates the effective capital based on the amount and months invested -/
def effectiveCapital (c : Capital) : ℕ := c.amount * c.months

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  first : ℕ
  second : ℕ

theorem haris_capital_contribution 
  (praveens_capital : Capital)
  (haris_join_month : ℕ)
  (total_months : ℕ)
  (profit_ratio : ProfitRatio)
  (h1 : praveens_capital.amount = 3360)
  (h2 : praveens_capital.months = total_months)
  (h3 : haris_join_month = 5)
  (h4 : total_months = 12)
  (h5 : profit_ratio.first = 2)
  (h6 : profit_ratio.second = 3)
  : ∃ (haris_capital : Capital), 
    haris_capital.amount = 8640 ∧ 
    haris_capital.months = total_months - haris_join_month ∧
    effectiveCapital praveens_capital * profit_ratio.second = 
    effectiveCapital haris_capital * profit_ratio.first :=
sorry

end haris_capital_contribution_l2594_259453


namespace cube_volume_increase_l2594_259463

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let new_edge := 1.6 * s
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry

end cube_volume_increase_l2594_259463


namespace candy_bar_profit_l2594_259424

/-- Calculates the profit from selling candy bars --/
def calculate_profit (
  total_bars : ℕ
  ) (buy_rate : ℚ × ℚ)
    (sell_rate : ℚ × ℚ)
    (discount_rate : ℕ × ℚ) : ℚ :=
  let cost_per_bar := buy_rate.2 / buy_rate.1
  let sell_per_bar := sell_rate.2 / sell_rate.1
  let total_cost := cost_per_bar * total_bars
  let total_revenue := sell_per_bar * total_bars
  let total_discounts := (total_bars / discount_rate.1) * discount_rate.2
  total_revenue - total_discounts - total_cost

theorem candy_bar_profit :
  calculate_profit 1200 (3, 1.5) (4, 3) (100, 2) = 276 := by
  sorry

end candy_bar_profit_l2594_259424


namespace cereal_serving_size_l2594_259483

def cereal_box_problem (total_cups : ℕ) (total_servings : ℕ) : Prop :=
  total_cups ≠ 0 ∧ total_servings ≠ 0 → total_cups / total_servings = 2

theorem cereal_serving_size : cereal_box_problem 18 9 := by
  sorry

end cereal_serving_size_l2594_259483


namespace root_product_plus_one_l2594_259450

theorem root_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
sorry

end root_product_plus_one_l2594_259450


namespace problems_left_to_grade_l2594_259487

/-- Given the number of total worksheets, graded worksheets, and problems per worksheet,
    calculate the number of problems left to grade. -/
theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : problems_per_worksheet = 3)
  (h4 : graded_worksheets ≤ total_worksheets) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 24 :=
by sorry

end problems_left_to_grade_l2594_259487


namespace remainder_three_power_45_plus_4_mod_5_l2594_259440

theorem remainder_three_power_45_plus_4_mod_5 : (3^45 + 4) % 5 = 2 := by
  sorry

end remainder_three_power_45_plus_4_mod_5_l2594_259440


namespace modified_chessboard_cannot_be_tiled_l2594_259413

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (black_squares : Nat)

/-- Represents a domino tile -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Defines the properties of a standard 8x8 chessboard with opposite corners removed -/
def standard_modified_chessboard : ModifiedChessboard :=
  { size := 8,
    total_squares := 62,
    white_squares := 32,
    black_squares := 30 }

/-- Defines the properties of a 1x2 domino -/
def standard_domino : Domino :=
  { length := 1,
    width := 2 }

/-- Checks if a chessboard can be tiled with dominoes -/
def can_be_tiled (board : ModifiedChessboard) (tile : Domino) : Prop :=
  board.white_squares = board.black_squares

/-- Theorem stating that the modified 8x8 chessboard cannot be tiled with 1x2 dominoes -/
theorem modified_chessboard_cannot_be_tiled :
  ¬(can_be_tiled standard_modified_chessboard standard_domino) :=
by
  sorry


end modified_chessboard_cannot_be_tiled_l2594_259413


namespace nadine_garage_sale_spend_l2594_259462

/-- The amount Nadine spent at the garage sale -/
def garage_sale_total (table_price chair_price num_chairs : ℕ) : ℕ :=
  table_price + chair_price * num_chairs

/-- Theorem: Nadine spent $56 at the garage sale -/
theorem nadine_garage_sale_spend :
  garage_sale_total 34 11 2 = 56 := by
  sorry

end nadine_garage_sale_spend_l2594_259462


namespace factor_expression_l2594_259428

theorem factor_expression (x : ℝ) : x * (x + 3) - 2 * (x + 3) = (x + 3) * (x - 2) := by
  sorry

end factor_expression_l2594_259428


namespace average_xyz_in_terms_of_k_l2594_259455

theorem average_xyz_in_terms_of_k (x y z k : ℝ) 
  (eq1 : 2 * x + y - z = 26)
  (eq2 : x + 2 * y + z = 10)
  (eq3 : x - y + z = k) :
  (x + y + z) / 3 = (36 + k) / 6 := by
  sorry

end average_xyz_in_terms_of_k_l2594_259455


namespace base4_equals_base2_l2594_259439

-- Define a function to convert a number from base 4 to base 10
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

-- Define a function to convert a number from base 2 to base 10
def base2ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (2 ^ i)) 0

-- Theorem statement
theorem base4_equals_base2 :
  base4ToDecimal [0, 1, 0, 1] = base2ToDecimal [0, 0, 1, 0, 0, 0, 1] := by
  sorry

end base4_equals_base2_l2594_259439


namespace negative_fractions_comparison_l2594_259474

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end negative_fractions_comparison_l2594_259474


namespace f_min_value_l2594_259486

noncomputable def f (x : ℝ) := Real.exp x + 3 * x^2 - x + 2011

theorem f_min_value :
  ∃ (min : ℝ), min = 2012 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end f_min_value_l2594_259486


namespace square_area_ratio_l2594_259427

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_perimeter : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l2594_259427


namespace integer_fraction_implication_l2594_259430

theorem integer_fraction_implication (m n p q : ℕ) (h1 : m ≠ p) 
  (h2 : ∃ k : ℤ, k = (m * n + p * q) / (m - p)) : 
  ∃ l : ℤ, l = (m * q + n * p) / (m - p) := by
  sorry

end integer_fraction_implication_l2594_259430


namespace unique_function_divisibility_l2594_259445

theorem unique_function_divisibility (k : ℕ) :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, (f m + f n) ∣ (m + n)^k :=
by
  sorry

end unique_function_divisibility_l2594_259445


namespace sum_complex_exp_argument_l2594_259466

/-- The argument of the sum of five complex exponentials -/
theorem sum_complex_exp_argument :
  let z : ℂ := Complex.exp (11 * Real.pi * Complex.I / 100) +
               Complex.exp (31 * Real.pi * Complex.I / 100) +
               Complex.exp (51 * Real.pi * Complex.I / 100) +
               Complex.exp (71 * Real.pi * Complex.I / 100) +
               Complex.exp (91 * Real.pi * Complex.I / 100)
  0 ≤ Complex.arg z ∧ Complex.arg z < 2 * Real.pi →
  Complex.arg z = 51 * Real.pi / 100 :=
by sorry

end sum_complex_exp_argument_l2594_259466


namespace proposition_equivalences_l2594_259496

theorem proposition_equivalences (x y : ℝ) : 
  (((Real.sqrt (x - 2) + (y + 1)^2 = 0) → (x = 2 ∧ y = -1)) ↔
   ((x = 2 ∧ y = -1) → (Real.sqrt (x - 2) + (y + 1)^2 = 0))) ∧
  (((Real.sqrt (x - 2) + (y + 1)^2 ≠ 0) → (x ≠ 2 ∨ y ≠ -1)) ↔
   ((x ≠ 2 ∨ y ≠ -1) → (Real.sqrt (x - 2) + (y + 1)^2 ≠ 0))) :=
by sorry

#check proposition_equivalences

end proposition_equivalences_l2594_259496


namespace power_of_two_plus_one_l2594_259481

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_prime_divisors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k :=
by sorry

end power_of_two_plus_one_l2594_259481


namespace set_intersection_problem_l2594_259419

theorem set_intersection_problem :
  let A : Set ℤ := {-2, -1, 0, 1}
  let B : Set ℤ := {-1, 0, 1, 2}
  A ∩ B = {-1, 0, 1} := by
  sorry

end set_intersection_problem_l2594_259419


namespace quadratic_inequality_theorem_l2594_259464

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the theorem
theorem quadratic_inequality_theorem (a c : ℝ) 
  (h : ∀ x, f a c x > 0 ↔ x ∈ solution_set a c) :
  a = -1/4 ∧ c = -3/4 ∧ 
  ∀ m : ℝ, (∀ x, -1/4 * x^2 + 2*x - 3 > 0 → x + m > 0) → m ≥ -2 :=
sorry

end quadratic_inequality_theorem_l2594_259464


namespace intersects_both_branches_iff_l2594_259454

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a line with slope k passing through a point -/
structure Line where
  k : ℝ

/-- Predicate indicating if a line intersects both branches of a hyperbola -/
def intersects_both_branches (h : Hyperbola) (l : Line) : Prop := sorry

/-- The necessary and sufficient condition for a line to intersect both branches of a hyperbola -/
theorem intersects_both_branches_iff (h : Hyperbola) (l : Line) :
  intersects_both_branches h l ↔ -h.b / h.a < l.k ∧ l.k < h.b / h.a := by sorry

end intersects_both_branches_iff_l2594_259454


namespace product_trailing_zeros_l2594_259451

def max_num : ℕ := 2020
def multiples_of_5 : ℕ := 404

-- Function to calculate the number of trailing zeros
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

-- Theorem statement
theorem product_trailing_zeros :
  trailing_zeros max_num = 503 :=
sorry

end product_trailing_zeros_l2594_259451


namespace greatest_4digit_base9_divisible_by_7_l2594_259404

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 --/
def base10_to_base9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is_4digit_base9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∀ n : ℕ, is_4digit_base9 n →
    (base9_to_base10 n) % 7 = 0 →
    n ≤ 8050 :=
by sorry

end greatest_4digit_base9_divisible_by_7_l2594_259404


namespace lottery_winning_probability_l2594_259421

/-- The number of balls for MegaBall selection -/
def megaBallCount : ℕ := 30

/-- The number of balls for WinnerBall selection -/
def winnerBallCount : ℕ := 45

/-- The number of WinnerBalls to be drawn -/
def winnerBallDrawCount : ℕ := 6

/-- The probability of winning the lottery -/
def winningProbability : ℚ := 1 / 244351800

/-- Theorem stating the probability of winning the lottery -/
theorem lottery_winning_probability :
  (1 / megaBallCount) * (1 / (winnerBallCount.choose winnerBallDrawCount)) = winningProbability := by
  sorry

end lottery_winning_probability_l2594_259421


namespace sufficient_condition_absolute_value_necessary_condition_inequality_l2594_259495

-- Statement ③
theorem sufficient_condition_absolute_value (a b : ℝ) :
  a^2 ≠ b^2 → |a| = |b| :=
sorry

-- Statement ④
theorem necessary_condition_inequality (a b c : ℝ) :
  a * c^2 < b * c^2 → a < b :=
sorry

end sufficient_condition_absolute_value_necessary_condition_inequality_l2594_259495


namespace valid_numbers_l2594_259499

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A ≥ 1 ∧ A ≤ 9 ∧ B ≥ 0 ∧ B ≤ 9 ∧ C ≥ 0 ∧ C ≤ 9 ∧
    n = 100001 * A + 10010 * B + 1100 * C ∧
    n % 7 = 0 ∧
    (100 * A + 10 * B + C) % 7 = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {168861, 259952, 861168, 952259} :=
by sorry

end valid_numbers_l2594_259499


namespace opposite_number_pairs_l2594_259425

theorem opposite_number_pairs : 
  (-(-(3 : ℤ)) = -(-|(-(3 : ℤ))|)) ∧ 
  ((-(2 : ℤ))^4 = -(2^4)) ∧ 
  ¬((-(2 : ℤ))^3 = -((-(3 : ℤ))^2)) ∧ 
  ¬((-(2 : ℤ))^3 = -(2^3)) := by
  sorry

end opposite_number_pairs_l2594_259425


namespace candy_has_nine_pencils_l2594_259484

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the pencil problem -/
def PencilProblem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

/-- The theorem stating that under the given conditions, Candy has 9 pencils -/
theorem candy_has_nine_pencils (p : PencilCounts) (h : PencilProblem p) : p.candy = 9 := by
  sorry

end candy_has_nine_pencils_l2594_259484


namespace brownie_pan_dimensions_l2594_259447

theorem brownie_pan_dimensions :
  ∀ m n : ℕ,
    m * n = 48 →
    (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4) →
    ((m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6)) :=
by sorry

end brownie_pan_dimensions_l2594_259447


namespace largest_m_value_l2594_259417

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem largest_m_value :
  ∀ m x y : ℕ,
    m ≥ 1000 →
    m < 10000 →
    is_prime x →
    is_prime y →
    is_prime (10 * x + y) →
    x < 10 →
    y < 10 →
    x > y →
    m = x * y * (10 * x + y) →
    m ≤ 1533 :=
sorry

end largest_m_value_l2594_259417


namespace function_identity_l2594_259400

theorem function_identity (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end function_identity_l2594_259400


namespace cost_per_side_of_square_l2594_259475

/-- The cost of fencing each side of a square, given the total cost --/
theorem cost_per_side_of_square (total_cost : ℝ) (h : total_cost = 276) : 
  ∃ (side_cost : ℝ), side_cost * 4 = total_cost ∧ side_cost = 69 := by
  sorry

end cost_per_side_of_square_l2594_259475


namespace events_mutually_exclusive_not_complementary_l2594_259411

-- Define the sample space
def SampleSpace := Finset (Fin 6 × Fin 6)

-- Define the events
def event_W (s : SampleSpace) : Prop := sorry
def event_1 (s : SampleSpace) : Prop := sorry
def event_2 (s : SampleSpace) : Prop := sorry

-- Define mutually exclusive
def mutually_exclusive (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, ¬(A s ∧ B s)

-- Define complementary
def complementary (A B : SampleSpace → Prop) : Prop :=
  ∀ s : SampleSpace, (A s ∨ B s) ∧ ¬(A s ∧ B s)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutually_exclusive event_W event_1 ∧
  mutually_exclusive event_W event_2 ∧
  ¬complementary event_W event_1 ∧
  ¬complementary event_W event_2 :=
sorry

end events_mutually_exclusive_not_complementary_l2594_259411


namespace arithmetic_sequence_first_term_range_l2594_259438

theorem arithmetic_sequence_first_term_range (a : ℕ → ℝ) (d : ℝ) (h1 : d = π / 8) :
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (a 10 ≤ 0) →
  (a 11 ≥ 0) →
  -5 * π / 4 ≤ a 1 ∧ a 1 ≤ -9 * π / 8 :=
by sorry

end arithmetic_sequence_first_term_range_l2594_259438


namespace billy_ice_cubes_l2594_259449

/-- The number of ice cubes in each tray -/
def cubes_per_tray : ℕ := 25

/-- The number of trays Billy has -/
def number_of_trays : ℕ := 15

/-- The total number of ice cubes Billy can make -/
def total_ice_cubes : ℕ := cubes_per_tray * number_of_trays

theorem billy_ice_cubes : total_ice_cubes = 375 := by sorry

end billy_ice_cubes_l2594_259449


namespace quadratic_roots_property_l2594_259402

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 122 := by
  sorry

end quadratic_roots_property_l2594_259402


namespace chess_tournament_games_l2594_259490

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 24 → 
  total_games = 276 → 
  total_games = n * (n - 1) / 2 → 
  ∃ (games_per_participant : ℕ), 
    games_per_participant = n - 1 ∧ 
    games_per_participant = 23 := by
  sorry

end chess_tournament_games_l2594_259490


namespace oh_squared_value_l2594_259478

/-- Given a triangle ABC with circumcenter O, orthocenter H, side lengths a, b, c, and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The squared distance between the circumcenter and orthocenter -/
def OH_squared (t : Triangle) : ℝ := 9 * t.R^2 - (t.a^2 + t.b^2 + t.c^2)

theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 5) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  OH_squared t = 175 := by
  sorry

end oh_squared_value_l2594_259478


namespace remaining_cooking_time_l2594_259460

def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def cooking_time_per_potato : ℕ := 5

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 := by
  sorry

end remaining_cooking_time_l2594_259460


namespace rocky_training_totals_l2594_259418

/-- Rocky's training schedule over three days -/
structure TrainingSchedule where
  initial_distance : ℝ
  initial_elevation : ℝ
  day2_distance_multiplier : ℝ
  day2_elevation_multiplier : ℝ
  day3_distance_multiplier : ℝ
  day3_elevation_multiplier : ℝ

/-- Calculate total distance and elevation gain over three days -/
def calculate_totals (schedule : TrainingSchedule) : ℝ × ℝ :=
  let day1_distance := schedule.initial_distance
  let day1_elevation := schedule.initial_elevation
  let day2_distance := day1_distance * schedule.day2_distance_multiplier
  let day2_elevation := day1_elevation * schedule.day2_elevation_multiplier
  let day3_distance := day2_distance * schedule.day3_distance_multiplier
  let day3_elevation := day2_elevation * schedule.day3_elevation_multiplier
  (day1_distance + day2_distance + day3_distance,
   day1_elevation + day2_elevation + day3_elevation)

/-- Theorem stating the total distance and elevation gain for Rocky's training -/
theorem rocky_training_totals :
  let schedule := TrainingSchedule.mk 4 100 2 1.5 4 2
  calculate_totals schedule = (44, 550) := by
  sorry

#eval calculate_totals (TrainingSchedule.mk 4 100 2 1.5 4 2)

end rocky_training_totals_l2594_259418


namespace second_exponent_base_l2594_259470

theorem second_exponent_base (x b : ℕ) (h1 : b > 0) (h2 : (18^6) * (x^17) = (2^6) * (3^b)) : x = 3 := by
  sorry

end second_exponent_base_l2594_259470


namespace smallest_three_digit_multiple_of_17_l2594_259432

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end smallest_three_digit_multiple_of_17_l2594_259432


namespace power_equality_l2594_259420

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end power_equality_l2594_259420


namespace correct_arrangement_count_l2594_259406

/-- The number of ways to arrange 8 students and 2 teachers in a line,
    where the teachers cannot stand next to each other -/
def arrangement_count : ℕ :=
  Nat.factorial 8 * 9 * 8

/-- Theorem stating that the number of valid arrangements is correct -/
theorem correct_arrangement_count :
  arrangement_count = Nat.factorial 8 * 9 * 8 := by
  sorry

end correct_arrangement_count_l2594_259406


namespace vector_magnitude_problem_l2594_259401

/-- Given vectors a and b in ℝ², if |a + 2b| = |a - 2b|, then |b| = 2√5 -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  a = (-1, -2) →
  b.1 = m →
  b.2 = 2 →
  ‖a + 2 • b‖ = ‖a - 2 • b‖ →
  ‖b‖ = 2 * Real.sqrt 5 := by
  sorry

end vector_magnitude_problem_l2594_259401


namespace fifteen_fishers_tomorrow_l2594_259435

/-- Represents the fishing pattern in the coastal village -/
structure FishingPattern :=
  (daily : ℕ)
  (everyOtherDay : ℕ)
  (everyThreeDay : ℕ)
  (yesterday : ℕ)
  (today : ℕ)

/-- Calculates the number of people fishing tomorrow given the fishing pattern -/
def fishersTomorrow (pattern : FishingPattern) : ℕ :=
  pattern.daily + pattern.everyThreeDay + (pattern.everyOtherDay - (pattern.yesterday - pattern.daily))

/-- Theorem stating that given the specific fishing pattern, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow : 
  ∀ (pattern : FishingPattern), 
  pattern.daily = 7 ∧ 
  pattern.everyOtherDay = 8 ∧ 
  pattern.everyThreeDay = 3 ∧
  pattern.yesterday = 12 ∧
  pattern.today = 10 →
  fishersTomorrow pattern = 15 := by
  sorry


end fifteen_fishers_tomorrow_l2594_259435


namespace smallest_number_with_all_factors_l2594_259441

def alice_number : ℕ := 24

-- Function to check if a number has all prime factors of another number
def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n) → (p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧ 
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end smallest_number_with_all_factors_l2594_259441


namespace max_exterior_elements_sum_l2594_259491

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismWithPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_base_edges : ℕ

/-- Calculate the total number of exterior elements after fusion -/
def total_exterior_elements (shape : PrismWithPyramid) : ℕ :=
  let new_faces := shape.prism_faces - 1 + shape.pyramid_base_edges
  let new_edges := shape.prism_edges + shape.pyramid_base_edges
  let new_vertices := shape.prism_vertices + 1
  new_faces + new_edges + new_vertices

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_exterior_elements_sum :
  ∀ shape : PrismWithPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_base_edges = 4 →
  total_exterior_elements shape = 34 := by
  sorry


end max_exterior_elements_sum_l2594_259491


namespace smallest_seating_arrangement_l2594_259457

/-- Represents a circular seating arrangement -/
structure CircularSeating :=
  (total_chairs : ℕ)
  (seated_people : ℕ)

/-- Checks if the seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  seating.seated_people > 0 ∧
  seating.seated_people ≤ seating.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < seating.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < seating.total_chairs ∧
      (new_seat = (occupied_seat + 1) % seating.total_chairs ∨
       new_seat = (occupied_seat + seating.total_chairs - 1) % seating.total_chairs)

/-- The main theorem to be proved -/
theorem smallest_seating_arrangement :
  ∃ (n : ℕ), n = 18 ∧
    satisfies_condition ⟨72, n⟩ ∧
    ∀ (m : ℕ), m < n → ¬satisfies_condition ⟨72, m⟩ :=
sorry

end smallest_seating_arrangement_l2594_259457


namespace max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l2594_259488

-- Define the line l: mx - y - 3m + 4 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 4 = 0

-- Define the circle O: x^2 + y^2 = 4
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the fixed point M(3,4) on line l
def point_M : ℝ × ℝ := (3, 4)

-- Theorem 1: Maximum distance from circle O to line l
theorem max_distance_circle_to_line :
  ∃ (m : ℝ), ∀ (x y : ℝ), circle_O x y →
    ∃ (max_dist : ℝ), max_dist = 7 ∧
      ∀ (x' y' : ℝ), line_l m x' y' →
        Real.sqrt ((x - x')^2 + (y - y')^2) ≤ max_dist :=
sorry

-- Theorem 2: Tangent line condition
theorem tangent_line_condition :
  ∃ (m : ℝ), m = (12 + 2 * Real.sqrt 21) / 5 ∨ m = (12 - 2 * Real.sqrt 21) / 5 →
    ∀ (x y : ℝ), line_l m x y →
      (∃! (x' y' : ℝ), circle_O x' y' ∧ x = x' ∧ y = y') :=
sorry

-- Theorem 3: Chord length condition
theorem chord_length_condition :
  ∃ (m : ℝ), m = (6 + Real.sqrt 6) / 4 ∨ m = (6 - Real.sqrt 6) / 4 →
    ∃ (x1 y1 x2 y2 : ℝ),
      line_l m x1 y1 ∧ line_l m x2 y2 ∧
      circle_O x1 y1 ∧ circle_O x2 y2 ∧
      Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 3 :=
sorry

end max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l2594_259488


namespace inequality_proof_l2594_259448

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b ≥ 0) :
  Real.sqrt (a^2 + b^2) + (a^3 + b^3)^(1/3) + (a^4 + b^4)^(1/4) ≤ 3*a + b := by
  sorry

end inequality_proof_l2594_259448


namespace sum_of_roots_quadratic_l2594_259436

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (∀ x, -x^2 + 2*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l2594_259436


namespace set_equality_l2594_259431

def positive_integers : Set ℕ := {n : ℕ | n > 0}

def set_a : Set ℕ := {x ∈ positive_integers | x - 3 < 2}
def set_b : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_a = set_b := by sorry

end set_equality_l2594_259431


namespace robbie_afternoon_rice_l2594_259458

/-- Represents the number of cups of rice Robbie eats at different times of the day and the fat content --/
structure RiceIntake where
  morning : ℕ
  evening : ℕ
  fat_per_cup : ℕ
  total_fat_per_week : ℕ

/-- Calculates the number of cups of rice Robbie eats in the afternoon --/
def afternoon_rice_cups (intake : RiceIntake) : ℕ :=
  (intake.total_fat_per_week - 7 * (intake.morning + intake.evening) * intake.fat_per_cup) / (7 * intake.fat_per_cup)

/-- Theorem stating that given the conditions, Robbie eats 14 cups of rice in the afternoon --/
theorem robbie_afternoon_rice 
  (intake : RiceIntake) 
  (h_morning : intake.morning = 3)
  (h_evening : intake.evening = 5)
  (h_fat_per_cup : intake.fat_per_cup = 10)
  (h_total_fat : intake.total_fat_per_week = 700) :
  afternoon_rice_cups intake = 14 := by
  sorry

end robbie_afternoon_rice_l2594_259458


namespace catchup_time_correct_l2594_259403

/-- Represents a person walking on the triangle -/
structure Walker where
  speed : ℝ  -- speed in meters per minute
  startVertex : ℕ  -- starting vertex (0, 1, or 2)

/-- Represents the triangle and walking scenario -/
structure TriangleWalk where
  sideLength : ℝ
  walkerA : Walker
  walkerB : Walker
  vertexDelay : ℝ  -- delay at each vertex in seconds

/-- Calculates the time when walker A catches up with walker B -/
def catchUpTime (tw : TriangleWalk) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem catchup_time_correct (tw : TriangleWalk) : 
  tw.sideLength = 200 ∧ 
  tw.walkerA = ⟨100, 0⟩ ∧ 
  tw.walkerB = ⟨80, 1⟩ ∧ 
  tw.vertexDelay = 15 → 
  catchUpTime tw = 1470 :=
sorry

end catchup_time_correct_l2594_259403


namespace no_x4_term_implies_a_zero_l2594_259479

theorem no_x4_term_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, ∃ b c d : ℝ, -5 * x^3 * (x^2 + a * x + 5) = b * x^5 + c * x^3 + d) →
  a = 0 :=
by sorry

end no_x4_term_implies_a_zero_l2594_259479


namespace digits_of_product_l2594_259405

theorem digits_of_product (n : ℕ) : n = 2^10 * 5^7 * 3^2 → (Nat.digits 10 n).length = 10 :=
by
  sorry

end digits_of_product_l2594_259405


namespace triangle_inradius_l2594_259437

/-- Given a triangle with perimeter 48 and area 60, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : P = 48) 
    (h2 : A = 60) 
    (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end triangle_inradius_l2594_259437


namespace probability_one_boy_one_girl_l2594_259468

def num_boys : ℕ := 3
def num_girls : ℕ := 2
def num_participants : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_participants

def favorable_outcomes : ℕ := num_boys.choose 1 * num_girls.choose 1

theorem probability_one_boy_one_girl :
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 := by sorry

end probability_one_boy_one_girl_l2594_259468


namespace line_segments_form_triangle_l2594_259408

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem line_segments_form_triangle :
  can_form_triangle 5 6 10 := by
  sorry


end line_segments_form_triangle_l2594_259408


namespace second_number_is_72_l2594_259498

theorem second_number_is_72 
  (sum : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (third : ℝ) 
  (h1 : sum = 264) 
  (h2 : first = 2 * second) 
  (h3 : third = (1/3) * first) 
  (h4 : first + second + third = sum) : second = 72 := by
sorry

end second_number_is_72_l2594_259498


namespace contractor_daily_wage_l2594_259489

/-- Represents the contractor's payment scenario -/
structure ContractorPayment where
  totalDays : ℕ
  finePerAbsence : ℚ
  totalPayment : ℚ
  absentDays : ℕ

/-- Calculates the daily wage of the contractor -/
def dailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerAbsence * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating the contractor's daily wage is 25 -/
theorem contractor_daily_wage :
  let c := ContractorPayment.mk 30 (7.5) 425 10
  dailyWage c = 25 := by sorry

end contractor_daily_wage_l2594_259489


namespace roof_dimension_difference_l2594_259415

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 900 →
  length - width = 45 := by
sorry

end roof_dimension_difference_l2594_259415


namespace smallest_steps_l2594_259409

theorem smallest_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 3 →
  n ≥ 59 :=
by sorry

end smallest_steps_l2594_259409


namespace cats_remaining_l2594_259422

/-- The number of cats remaining after a sale in a pet store -/
theorem cats_remaining (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 19 → house = 45 → sold = 56 → siamese + house - sold = 8 := by
  sorry

end cats_remaining_l2594_259422


namespace function_minimum_and_integer_bound_l2594_259493

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a + Real.log x)

theorem function_minimum_and_integer_bound :
  (∃ a : ℝ, ∀ x > 0, f a x ≥ -Real.exp (-2) ∧ ∃ x₀ > 0, f a x₀ = -Real.exp (-2)) →
  (∃ a : ℝ, a = 1 ∧
    ∀ k : ℤ, (∀ x > 1, ↑k < (f a x) / (x - 1)) →
      k ≤ 3 ∧ (∃ x > 1, 3 < (f a x) / (x - 1))) :=
by sorry

end function_minimum_and_integer_bound_l2594_259493


namespace negation_of_all_squares_nonnegative_l2594_259412

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end negation_of_all_squares_nonnegative_l2594_259412


namespace unique_solution_l2594_259426

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_solution :
  ∃! A : ℕ, ∃ B : ℕ,
    4 * A + (10 * B + 3) = 68 ∧
    is_two_digit (4 * A) ∧
    is_two_digit (10 * B + 3) ∧
    A ≤ 9 ∧ B ≤ 9 :=
by sorry

end unique_solution_l2594_259426


namespace dina_has_60_dolls_l2594_259485

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collectors edition dolls Ivy has -/
def ivy_collectors : ℕ := 20

theorem dina_has_60_dolls :
  (2 * ivy_dolls = dina_dolls) →
  (2 * ivy_collectors = 3 * ivy_dolls) →
  (ivy_collectors = 20) →
  dina_dolls = 60 := by
  sorry

end dina_has_60_dolls_l2594_259485


namespace max_t_for_tangent_slope_l2594_259494

/-- Given t > 0 and f(x) = x²(x - t), prove that the maximum value of t for which
    the slope of the tangent line to f(x) is always greater than or equal to -1
    when x is in (0, 1] is 3/2. -/
theorem max_t_for_tangent_slope (t : ℝ) (h_t : t > 0) :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → (3 * x^2 - 2 * t * x) ≥ -1) ↔ t ≤ 3/2 := by
  sorry

end max_t_for_tangent_slope_l2594_259494


namespace shyne_plants_l2594_259456

/-- The number of plants Shyne can grow from her seed packets -/
def total_plants (eggplant_per_packet : ℕ) (sunflower_per_packet : ℕ) 
                 (eggplant_packets : ℕ) (sunflower_packets : ℕ) : ℕ :=
  eggplant_per_packet * eggplant_packets + sunflower_per_packet * sunflower_packets

/-- Proof that Shyne can grow 116 plants -/
theorem shyne_plants : 
  total_plants 14 10 4 6 = 116 := by
  sorry

end shyne_plants_l2594_259456


namespace equal_roots_C_value_l2594_259477

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4 * eq.a * eq.c

/-- Checks if a quadratic equation has equal roots -/
def hasEqualRoots (eq : QuadraticEquation) : Prop :=
  discriminant eq = 0

/-- The specific quadratic equation from the problem -/
def problemEquation (k C : ℝ) : QuadraticEquation where
  a := 2 * k
  b := 6 * k
  c := C

/-- The theorem to be proved -/
theorem equal_roots_C_value :
  ∃ C : ℝ, hasEqualRoots (problemEquation 0.4444444444444444 C) ∧ C = 2 := by
  sorry

end equal_roots_C_value_l2594_259477


namespace union_of_M_and_N_l2594_259480

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end union_of_M_and_N_l2594_259480


namespace point_on_line_product_of_y_coordinates_l2594_259482

theorem point_on_line_product_of_y_coordinates :
  ∀ y₁ y₂ : ℝ,
  ((-3 - 3)^2 + (-1 - y₁)^2 = 13^2) →
  ((-3 - 3)^2 + (-1 - y₂)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -132 := by sorry

end point_on_line_product_of_y_coordinates_l2594_259482


namespace fifteen_factorial_representation_l2594_259476

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_representation (X Y Z : ℕ) :
  X < 10 ∧ Y < 10 ∧ Z < 10 →
  factorial 15 = 1307674300000000 + X * 100000000 + Y * 10000 + Z * 100 →
  X + Y + Z = 0 := by
sorry

end fifteen_factorial_representation_l2594_259476


namespace coat_drive_total_l2594_259414

theorem coat_drive_total (high_school_coats : ℕ) (elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 := by
  sorry

end coat_drive_total_l2594_259414


namespace fraction_calculation_l2594_259442

theorem fraction_calculation : (3/8) / (4/9) + 1/6 = 97/96 := by
  sorry

end fraction_calculation_l2594_259442


namespace quadratic_properties_l2594_259423

/-- A quadratic function passing through specific points -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  f 0 = -7/2 ∧ f 1 = 1/2 ∧ f (3/2) = 1 ∧ f 2 = 1/2

theorem quadratic_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧  -- f is quadratic
  f 0 = -7/2 ∧  -- y-axis intersection
  (∀ x, f (3/2 - x) = f (3/2 + x)) ∧  -- axis of symmetry
  (∀ x, f x ≤ f (3/2)) ∧  -- vertex
  (∀ x, f x = -2 * (x - 3/2)^2 + 1)  -- analytical expression
  := by sorry

end quadratic_properties_l2594_259423


namespace largest_angle_of_special_quadrilateral_l2594_259492

/-- A convex quadrilateral is rude if there exists a convex quadrilateral inside or on its sides
    with a larger sum of diagonals. -/
def IsRude (Q : Set (ℝ × ℝ)) : Prop := sorry

/-- The largest angle of a quadrilateral -/
def LargestAngle (Q : Set (ℝ × ℝ)) : ℝ := sorry

/-- A convex quadrilateral -/
def ConvexQuadrilateral (Q : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_angle_of_special_quadrilateral 
  (A B C D : ℝ × ℝ) 
  (r : ℝ)
  (h_convex : ConvexQuadrilateral {A, B, C, D})
  (h_not_rude : ¬IsRude {A, B, C, D})
  (h_r_positive : r > 0)
  (h_nearby_rude : ∀ A', A' ≠ A → dist A' A ≤ r → IsRude {A', B, C, D}) :
  LargestAngle {A, B, C, D} = 150 * π / 180 := by sorry

end largest_angle_of_special_quadrilateral_l2594_259492


namespace substitution_result_l2594_259473

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end substitution_result_l2594_259473


namespace pat_to_mark_ratio_project_hours_ratio_l2594_259444

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the problem --/
def satisfiesConditions (hours : ProjectHours) : Prop :=
  hours.pat + hours.kate + hours.mark = 117 ∧
  hours.pat = 2 * hours.kate ∧
  hours.mark = hours.kate + 65

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (hours : ProjectHours) 
  (h : satisfiesConditions hours) : 
  hours.pat * 3 = hours.mark * 1 := by
  sorry

/-- Main theorem proving the ratio is 1:3 --/
theorem project_hours_ratio : 
  ∃ hours : ProjectHours, satisfiesConditions hours ∧ hours.pat * 3 = hours.mark * 1 := by
  sorry

end pat_to_mark_ratio_project_hours_ratio_l2594_259444


namespace subtraction_of_squares_l2594_259407

theorem subtraction_of_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end subtraction_of_squares_l2594_259407


namespace inequality_iff_in_interval_l2594_259472

/-- The roots of the quadratic equation x^2 - (16/5)x - 8 = 0 --/
def a : ℝ := sorry
def b : ℝ := sorry

axiom a_lt_b : a < b
axiom b_lt_zero : b < 0
axiom roots_property : ∀ x : ℝ, x^2 - (16/5) * x - 8 = 0 ↔ (x = a ∨ x = b)

/-- The main theorem stating the equivalence between the inequality and the solution interval --/
theorem inequality_iff_in_interval (x : ℝ) : 
  1 / (x^2 + 2) + 1 / 2 > 5 / x + 21 / 10 ↔ (x < a ∨ (b < x ∧ x < 0)) :=
sorry

end inequality_iff_in_interval_l2594_259472


namespace school_water_cases_l2594_259469

theorem school_water_cases : 
  ∀ (bottles_per_case : ℕ) 
    (bottles_used_first_game : ℕ) 
    (bottles_used_second_game : ℕ) 
    (bottles_left : ℕ),
  bottles_per_case = 20 →
  bottles_used_first_game = 70 →
  bottles_used_second_game = 110 →
  bottles_left = 20 →
  (bottles_used_first_game + bottles_used_second_game + bottles_left) / bottles_per_case = 10 := by
sorry

end school_water_cases_l2594_259469


namespace sqrt_8_times_sqrt_18_l2594_259459

theorem sqrt_8_times_sqrt_18 : Real.sqrt 8 * Real.sqrt 18 = 12 := by
  sorry

end sqrt_8_times_sqrt_18_l2594_259459


namespace quadratic_root_difference_l2594_259465

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x : ℝ, x^2 + p*x + q = 0 → 
    ∃ y : ℝ, y^2 + p*y + q = 0 ∧ |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end quadratic_root_difference_l2594_259465


namespace double_dimensions_volume_l2594_259429

/-- A cylindrical container with volume, height, and radius. -/
structure CylindricalContainer where
  volume : ℝ
  height : ℝ
  radius : ℝ
  volume_formula : volume = Real.pi * radius^2 * height

/-- Given a cylindrical container of 5 gallons, doubling its dimensions results in a 40-gallon container -/
theorem double_dimensions_volume (c : CylindricalContainer) 
  (h_volume : c.volume = 5) :
  let new_container : CylindricalContainer := {
    volume := Real.pi * (2 * c.radius)^2 * (2 * c.height),
    height := 2 * c.height,
    radius := 2 * c.radius,
    volume_formula := by sorry
  }
  new_container.volume = 40 := by
  sorry

end double_dimensions_volume_l2594_259429


namespace christines_dog_weight_l2594_259497

/-- The weight of Christine's dog given the weights of her two cats -/
def dogs_weight (cat1_weight cat2_weight : ℕ) : ℕ :=
  2 * (cat1_weight + cat2_weight)

/-- Theorem stating that Christine's dog weighs 34 pounds -/
theorem christines_dog_weight :
  dogs_weight 7 10 = 34 := by
  sorry

end christines_dog_weight_l2594_259497


namespace max_square_plots_l2594_259467

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 2400

/-- Calculates the number of square plots along the field's width -/
def numPlotsWidth (field : FieldDimensions) : ℕ :=
  20

/-- Calculates the number of square plots along the field's length -/
def numPlotsLength (field : FieldDimensions) : ℕ :=
  30

/-- Calculates the total number of square plots -/
def totalPlots (field : FieldDimensions) : ℕ :=
  numPlotsWidth field * numPlotsLength field

/-- Calculates the length of internal fencing used -/
def usedFence (field : FieldDimensions) : ℕ :=
  field.width * (numPlotsLength field - 1) + field.length * (numPlotsWidth field - 1)

/-- Theorem stating that 600 is the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h1 : field.width = 40) 
    (h2 : field.length = 60) : 
    totalPlots field = 600 ∧ 
    usedFence field ≤ availableFence ∧ 
    ∀ n m : ℕ, n * m > 600 → 
      field.width * (m - 1) + field.length * (n - 1) > availableFence :=
  sorry

#check max_square_plots

end max_square_plots_l2594_259467


namespace factorization_x_squared_minus_one_l2594_259410

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_x_squared_minus_one_l2594_259410


namespace black_to_grey_ratio_in_square_with_circles_l2594_259471

/-- The ratio of black to grey areas in a square with four inscribed circles -/
theorem black_to_grey_ratio_in_square_with_circles (s : ℝ) (h : s > 0) :
  let r := s / 4
  let circle_area := π * r^2
  let total_square_area := s^2
  let remaining_area := total_square_area - 4 * circle_area
  let black_area := remaining_area / 4
  let grey_area := 3 * black_area
  black_area / grey_area = 1 / 3 := by sorry

end black_to_grey_ratio_in_square_with_circles_l2594_259471


namespace probability_to_reach_target_l2594_259452

-- Define the robot's position as a pair of integers
def Position := ℤ × ℤ

-- Define the possible directions
inductive Direction
| Left
| Right
| Up
| Down

-- Define a step as a movement in a direction
def step (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Left  => (pos.1 - 1, pos.2)
  | Direction.Right => (pos.1 + 1, pos.2)
  | Direction.Up    => (pos.1, pos.2 + 1)
  | Direction.Down  => (pos.1, pos.2 - 1)

-- Define the probability of each direction
def directionProbability : ℚ := 1 / 4

-- Define the maximum number of steps
def maxSteps : ℕ := 6

-- Define the target position
def target : Position := (3, 1)

-- Define the function to calculate the probability of reaching the target
noncomputable def probabilityToReachTarget : ℚ := sorry

-- State the theorem
theorem probability_to_reach_target :
  probabilityToReachTarget = 37 / 512 := by sorry

end probability_to_reach_target_l2594_259452


namespace simplify_expression_l2594_259446

theorem simplify_expression (x y : ℚ) (hx : x = 10) (hy : y = -1/25) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = 2/5 := by
  sorry

end simplify_expression_l2594_259446


namespace yellow_balloons_count_l2594_259433

/-- The number of yellow balloons -/
def yellow_balloons : ℕ := 3414

/-- The number of black balloons -/
def black_balloons : ℕ := yellow_balloons + 1762

/-- The total number of balloons -/
def total_balloons : ℕ := yellow_balloons + black_balloons

theorem yellow_balloons_count : yellow_balloons = 3414 :=
  by
  have h1 : black_balloons = yellow_balloons + 1762 := rfl
  have h2 : total_balloons / 10 = 859 := by sorry
  sorry


end yellow_balloons_count_l2594_259433


namespace intersecting_rectangles_area_l2594_259443

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The total shaded area of two intersecting rectangles -/
def totalShadedArea (r1 r2 overlap : Rectangle) : ℝ :=
  r1.area + r2.area - overlap.area

theorem intersecting_rectangles_area :
  let r1 : Rectangle := ⟨4, 12⟩
  let r2 : Rectangle := ⟨5, 10⟩
  let overlap : Rectangle := ⟨4, 5⟩
  totalShadedArea r1 r2 overlap = 78 := by
  sorry

end intersecting_rectangles_area_l2594_259443


namespace no_integer_root_pairs_l2594_259416

theorem no_integer_root_pairs (n : ℕ) : ¬ ∃ (a b : Fin 5 → ℤ),
  (∀ k : Fin 5, ∃ (x y : ℤ), x^2 + a k * x + b k = 0 ∧ y^2 + a k * y + b k = 0) ∧
  (∀ k : Fin 5, ∃ m : ℤ, a k = 2 * n + 2 * k + 2 ∨ a k = 2 * n + 2 * k + 4) ∧
  (∀ k : Fin 5, ∃ m : ℤ, b k = 2 * n + 2 * k + 2 ∨ b k = 2 * n + 2 * k + 4) :=
by sorry

end no_integer_root_pairs_l2594_259416
