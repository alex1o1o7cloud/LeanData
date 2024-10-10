import Mathlib

namespace cos_40_plus_sqrt3_tan_10_eq_1_l2225_222502

theorem cos_40_plus_sqrt3_tan_10_eq_1 : 
  Real.cos (40 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 := by
  sorry

end cos_40_plus_sqrt3_tan_10_eq_1_l2225_222502


namespace highlighters_count_l2225_222564

/-- The number of pink highlighters in the desk -/
def pink_highlighters : ℕ := 7

/-- The number of yellow highlighters in the desk -/
def yellow_highlighters : ℕ := 4

/-- The number of blue highlighters in the desk -/
def blue_highlighters : ℕ := 5

/-- The number of green highlighters in the desk -/
def green_highlighters : ℕ := 3

/-- The number of orange highlighters in the desk -/
def orange_highlighters : ℕ := 6

/-- The total number of highlighters in the desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + orange_highlighters

theorem highlighters_count : total_highlighters = 25 := by
  sorry

end highlighters_count_l2225_222564


namespace cubic_root_simplification_l2225_222584

theorem cubic_root_simplification (s : ℝ) : s = 1 / (2 - Real.rpow 3 (1/3)) → s = 2 + Real.rpow 3 (1/3) := by
  sorry

end cubic_root_simplification_l2225_222584


namespace curve_transformation_l2225_222526

theorem curve_transformation (x : ℝ) : 
  Real.sin (2 * x + 2 * Real.pi / 3) = Real.cos (2 * (x - Real.pi / 12)) := by
  sorry

end curve_transformation_l2225_222526


namespace select_gloves_count_l2225_222545

/-- The number of ways to select 4 gloves from 5 pairs of gloves with exactly one pair of the same color -/
def select_gloves (n : ℕ) : ℕ :=
  let total_pairs := 5
  let select_size := 4
  let pair_combinations := Nat.choose total_pairs 1
  let remaining_gloves := 2 * (total_pairs - 1)
  let other_combinations := Nat.choose remaining_gloves 2
  let same_color_pair := Nat.choose (total_pairs - 1) 1
  pair_combinations * (other_combinations - same_color_pair)

/-- Theorem stating that the number of ways to select 4 gloves from 5 pairs of gloves 
    with exactly one pair of the same color is 120 -/
theorem select_gloves_count : select_gloves 5 = 120 := by
  sorry

end select_gloves_count_l2225_222545


namespace problem_statement_l2225_222527

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 2) 
  (h3 : a * x + b * y = 5) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = -5 := by
  sorry

end problem_statement_l2225_222527


namespace parallel_vectors_k_value_l2225_222562

/-- Given two 2D vectors, find the value of k that makes one vector parallel to another. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = 1/4 ∧ 
  ∃ c : ℝ, c • (2 • a + b) = (1/2 • a + k • b) := by
  sorry

end parallel_vectors_k_value_l2225_222562


namespace quadratic_sum_l2225_222575

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x) ∧  -- minimum exists
  (quadratic a b c 3 = 0) ∧  -- passes through (3,0)
  (quadratic a b c 7 = 0) ∧  -- passes through (7,0)
  (∀ (x : ℝ), quadratic a b c x ≥ 36) ∧  -- minimum value is 36
  (∃ (x : ℝ), quadratic a b c x = 36)  -- minimum value is achieved
  →
  a + b + c = -108 :=
by sorry

end quadratic_sum_l2225_222575


namespace waiting_time_is_correct_l2225_222501

/-- The total waiting time in minutes for Mark's vaccine appointments -/
def total_waiting_time : ℕ :=
  let days_first_vaccine := 4
  let days_second_vaccine := 20
  let days_first_secondary := 30 + 10  -- 1 month and 10 days
  let days_second_secondary := 14 + 3  -- 2 weeks and 3 days
  let days_full_effectiveness := 3 * 7 -- 3 weeks
  let total_days := days_first_vaccine + days_second_vaccine + days_first_secondary +
                    days_second_secondary + days_full_effectiveness
  let minutes_per_day := 24 * 60
  total_days * minutes_per_day

/-- Theorem stating that the total waiting time is 146,880 minutes -/
theorem waiting_time_is_correct : total_waiting_time = 146880 := by
  sorry

end waiting_time_is_correct_l2225_222501


namespace company_plants_1500_trees_l2225_222580

/-- Represents the number of trees chopped down in the first half of the year -/
def trees_chopped_first_half : ℕ := 200

/-- Represents the number of trees chopped down in the second half of the year -/
def trees_chopped_second_half : ℕ := 300

/-- Represents the number of trees to be planted for each tree chopped down -/
def trees_planted_per_chopped : ℕ := 3

/-- Calculates the total number of trees that need to be planted -/
def trees_to_plant : ℕ := (trees_chopped_first_half + trees_chopped_second_half) * trees_planted_per_chopped

/-- Theorem stating that the company needs to plant 1500 trees -/
theorem company_plants_1500_trees : trees_to_plant = 1500 := by
  sorry

end company_plants_1500_trees_l2225_222580


namespace tens_digit_of_nine_to_1010_l2225_222589

theorem tens_digit_of_nine_to_1010 :
  (9 : ℕ) ^ 1010 % 100 = 1 :=
sorry

end tens_digit_of_nine_to_1010_l2225_222589


namespace hotel_rooms_available_l2225_222536

theorem hotel_rooms_available (total_floors : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) :
  total_floors = 10 →
  rooms_per_floor = 10 →
  unavailable_floors = 1 →
  (total_floors - unavailable_floors) * rooms_per_floor = 90 := by
  sorry

end hotel_rooms_available_l2225_222536


namespace perimeter_of_specific_hexagon_l2225_222518

-- Define the hexagon ABCDEF
structure RightAngledHexagon where
  AB : ℝ
  BC : ℝ
  EF : ℝ

-- Define the perimeter function
def perimeter (h : RightAngledHexagon) : ℝ :=
  2 * (h.AB + h.EF) + 2 * h.BC

-- Theorem statement
theorem perimeter_of_specific_hexagon :
  ∃ (h : RightAngledHexagon), h.AB = 8 ∧ h.BC = 15 ∧ h.EF = 5 ∧ perimeter h = 56 := by
  sorry

end perimeter_of_specific_hexagon_l2225_222518


namespace monotonic_increasing_condition_non_negative_condition_two_roots_condition_l2225_222563

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

/-- Theorem for part I -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ≥ 4, ∀ y ≥ 4, x < y → f a x < f a y) ↔ a ≥ -3 :=
sorry

/-- Theorem for part II -/
theorem non_negative_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ -1 ≤ a ∧ a ≤ 5 :=
sorry

/-- Theorem for part III -/
theorem two_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ -5/4 < a ∧ a < -1 :=
sorry

end monotonic_increasing_condition_non_negative_condition_two_roots_condition_l2225_222563


namespace red_note_rows_l2225_222550

theorem red_note_rows (red_notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ) (total_notes : ℕ) :
  red_notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  total_notes = 100 →
  ∃ (rows : ℕ), rows * red_notes_per_row + rows * red_notes_per_row * blue_notes_per_red + additional_blue_notes = total_notes ∧ rows = 5 :=
by
  sorry

end red_note_rows_l2225_222550


namespace quadratic_minimum_l2225_222557

theorem quadratic_minimum (p q : ℝ) : 
  (∀ x, x^2 + p*x + q ≥ 1) → q = 1 + p^2/4 := by
  sorry

end quadratic_minimum_l2225_222557


namespace highway_scenario_solution_l2225_222551

/-- Represents the scenario of a person walking along a highway with buses passing by -/
structure HighwayScenario where
  personSpeed : ℝ
  busSpeed : ℝ
  busDepartureInterval : ℝ
  oncomingBusInterval : ℝ
  overtakingBusInterval : ℝ
  busDistance : ℝ

/-- Checks if the given scenario satisfies all conditions -/
def isValidScenario (s : HighwayScenario) : Prop :=
  s.personSpeed > 0 ∧
  s.busSpeed > s.personSpeed ∧
  s.oncomingBusInterval * (s.busSpeed + s.personSpeed) = s.busDistance ∧
  s.overtakingBusInterval * (s.busSpeed - s.personSpeed) = s.busDistance ∧
  s.busDepartureInterval = s.busDistance / s.busSpeed

/-- The main theorem stating the unique solution to the highway scenario -/
theorem highway_scenario_solution :
  ∃! s : HighwayScenario, isValidScenario s ∧
    s.oncomingBusInterval = 4 ∧
    s.overtakingBusInterval = 6 ∧
    s.busDistance = 1200 ∧
    s.personSpeed = 50 ∧
    s.busSpeed = 250 ∧
    s.busDepartureInterval = 4.8 := by
  sorry


end highway_scenario_solution_l2225_222551


namespace increasing_sine_function_bound_l2225_222535

open Real

/-- A function f is increasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem: If f(x) = x + a*sin(x) is increasing on ℝ, then -1 ≤ a ≤ 1 -/
theorem increasing_sine_function_bound (a : ℝ) :
  IncreasingOn (fun x => x + a * sin x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end increasing_sine_function_bound_l2225_222535


namespace absolute_value_v_l2225_222546

theorem absolute_value_v (u v : ℂ) : 
  u * v = 20 - 15 * I → Complex.abs u = 5 → Complex.abs v = 5 := by
  sorry

end absolute_value_v_l2225_222546


namespace gold_coin_distribution_l2225_222594

theorem gold_coin_distribution (x y : ℤ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25*(x - y)) : x + y = 25 := by
  sorry

end gold_coin_distribution_l2225_222594


namespace sqrt_sum_comparison_l2225_222532

theorem sqrt_sum_comparison : Real.sqrt 2 + Real.sqrt 10 < 2 * Real.sqrt 6 := by
  sorry

end sqrt_sum_comparison_l2225_222532


namespace five_digit_palindromes_count_l2225_222512

/-- A function that counts the number of 5-digit palindromes -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

end five_digit_palindromes_count_l2225_222512


namespace largest_number_with_property_l2225_222505

/-- Checks if a four-digit number satisfies the property that each of the last two digits
    is equal to the sum of the two preceding digits. -/
def satisfiesProperty (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 100 = (n / 100 % 10 + n / 10 % 10) % 10) ∧
  (n / 10 % 10 = (n / 1000 + n / 100 % 10) % 10)

/-- Theorem stating that 9099 is the largest four-digit number satisfying the property. -/
theorem largest_number_with_property :
  satisfiesProperty 9099 ∧ ∀ m : ℕ, satisfiesProperty m → m ≤ 9099 :=
sorry

end largest_number_with_property_l2225_222505


namespace data_set_properties_l2225_222595

def data_set : List ℕ := [3, 5, 4, 5, 6, 7]

def mode (list : List ℕ) : ℕ := sorry

def median (list : List ℕ) : ℚ := sorry

def mean (list : List ℕ) : ℚ := sorry

theorem data_set_properties :
  mode data_set = 5 ∧ 
  median data_set = 5 ∧ 
  mean data_set = 5 := by sorry

end data_set_properties_l2225_222595


namespace max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l2225_222598

/-- Represents a chessboard configuration -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a chessboard configuration is valid according to the given constraints -/
def is_valid_configuration (board : Chessboard) : Prop :=
  -- At most 4 tokens per row
  (∀ row, (Finset.filter (λ col => board row col) Finset.univ).card ≤ 4) ∧
  -- At most 4 tokens per column
  (∀ col, (Finset.filter (λ row => board row col) Finset.univ).card ≤ 4)

/-- Checks if a chessboard configuration is valid including diagonal constraints -/
def is_valid_configuration_with_diagonals (board : Chessboard) : Prop :=
  is_valid_configuration board ∧
  -- At most 4 tokens on main diagonal
  (Finset.filter (λ i => board i i) Finset.univ).card ≤ 4 ∧
  -- At most 4 tokens on anti-diagonal
  (Finset.filter (λ i => board i (7 - i)) Finset.univ).card ≤ 4

/-- The total number of tokens on the board -/
def token_count (board : Chessboard) : Nat :=
  (Finset.filter (λ p => board p.1 p.2) (Finset.univ.product Finset.univ)).card

theorem max_tokens_on_chessboard :
  (∃ board : Chessboard, is_valid_configuration board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration board → token_count board ≤ 32) :=
sorry

theorem max_tokens_on_chessboard_with_diagonals :
  (∃ board : Chessboard, is_valid_configuration_with_diagonals board ∧ token_count board = 32) ∧
  (∀ board : Chessboard, is_valid_configuration_with_diagonals board → token_count board ≤ 32) :=
sorry

end max_tokens_on_chessboard_max_tokens_on_chessboard_with_diagonals_l2225_222598


namespace half_dollars_in_tip_jar_l2225_222514

def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def half_dollar_value : ℚ := 50 / 100

def nickels_shining : ℕ := 3
def dimes_shining : ℕ := 13
def dimes_tip : ℕ := 7
def total_amount : ℚ := 665 / 100

theorem half_dollars_in_tip_jar :
  ∃ (half_dollars : ℕ),
    (nickels_shining : ℚ) * nickel_value +
    (dimes_shining : ℚ) * dime_value +
    (dimes_tip : ℚ) * dime_value +
    (half_dollars : ℚ) * half_dollar_value = total_amount ∧
    half_dollars = 9 :=
by sorry

end half_dollars_in_tip_jar_l2225_222514


namespace line_equation_proof_l2225_222585

/-- The parabola y^2 = (5/2)x -/
def parabola (x y : ℝ) : Prop := y^2 = (5/2) * x

/-- Point O is the origin (0,0) -/
def O : ℝ × ℝ := (0, 0)

/-- Point through which the line passes -/
def P : ℝ × ℝ := (2, 1)

/-- Predicate to check if a point is on the line -/
def on_line (x y : ℝ) : Prop := 2*x + y - 5 = 0

/-- Two points are perpendicular with respect to the origin -/
def perpendicular_to_origin (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = 0

theorem line_equation_proof :
  ∃ (A B : ℝ × ℝ),
    A ≠ O ∧ B ≠ O ∧
    A ≠ B ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    perpendicular_to_origin A B ∧
    on_line A.1 A.2 ∧
    on_line B.1 B.2 ∧
    on_line P.1 P.2 :=
sorry

end line_equation_proof_l2225_222585


namespace sqrt_equation_solution_l2225_222513

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = x + y - 1 ↔ (x = 1 ∧ y ≥ 0) ∨ (y = 1 ∧ x ≥ 0) := by
  sorry

end sqrt_equation_solution_l2225_222513


namespace count_seven_digit_phone_numbers_l2225_222515

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def sevenDigitPhoneNumbers : ℕ := 9 * (10 ^ 6)

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 * 10^6 -/
theorem count_seven_digit_phone_numbers :
  sevenDigitPhoneNumbers = 9 * (10 ^ 6) := by
  sorry

end count_seven_digit_phone_numbers_l2225_222515


namespace trimmed_square_area_l2225_222534

/-- The area of a rectangle formed by trimming a square --/
theorem trimmed_square_area (original_side : ℝ) (trim1 : ℝ) (trim2 : ℝ) 
  (h1 : original_side = 18)
  (h2 : trim1 = 4)
  (h3 : trim2 = 3) :
  (original_side - trim1) * (original_side - trim2) = 210 := by
sorry

end trimmed_square_area_l2225_222534


namespace compute_expression_l2225_222547

theorem compute_expression : 
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2/3) = 41.65 := by sorry

end compute_expression_l2225_222547


namespace contract_completion_problem_l2225_222506

/-- Represents the contract completion problem -/
theorem contract_completion_problem (total_days : ℕ) (initial_hours_per_day : ℕ) 
  (days_worked : ℕ) (work_completed_fraction : ℚ) (additional_men : ℕ) 
  (new_hours_per_day : ℕ) :
  total_days = 46 →
  initial_hours_per_day = 8 →
  days_worked = 33 →
  work_completed_fraction = 4/7 →
  additional_men = 81 →
  new_hours_per_day = 9 →
  ∃ (initial_men : ℕ), 
    (initial_men * days_worked * initial_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = work_completed_fraction ∧
    ((initial_men + additional_men) * (total_days - days_worked) * new_hours_per_day : ℚ) / (total_days * initial_hours_per_day) = 1 - work_completed_fraction ∧
    initial_men = 117 :=
by sorry

end contract_completion_problem_l2225_222506


namespace decreasing_f_implies_a_range_l2225_222529

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry


end decreasing_f_implies_a_range_l2225_222529


namespace largest_band_size_l2225_222599

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (original : BandFormation) (total : ℕ) : Prop :=
  total < 100 ∧
  total = original.rows * original.membersPerRow + 3 ∧
  total = (original.rows - 3) * (original.membersPerRow + 1)

/-- Finds the largest valid band formation --/
def largestValidFormation : Option (BandFormation × ℕ) :=
  sorry

theorem largest_band_size :
  ∀ bf : BandFormation,
  ∀ m : ℕ,
  isValidFormation bf m →
  m ≤ 75 :=
sorry

end largest_band_size_l2225_222599


namespace marble_probability_l2225_222577

theorem marble_probability (red green white blue : ℕ) 
  (h_red : red = 5)
  (h_green : green = 4)
  (h_white : white = 12)
  (h_blue : blue = 2) :
  let total := red + green + white + blue
  (red / total) * (blue / (total - 1)) = 5 / 253 := by
  sorry

end marble_probability_l2225_222577


namespace sum_of_2008th_powers_l2225_222559

theorem sum_of_2008th_powers (a b c : ℝ) 
  (sum_eq_3 : a + b + c = 3) 
  (sum_squares_eq_3 : a^2 + b^2 + c^2 = 3) : 
  a^2008 + b^2008 + c^2008 = 3 := by
sorry

end sum_of_2008th_powers_l2225_222559


namespace difference_of_squares_special_case_l2225_222553

theorem difference_of_squares_special_case : (503 : ℤ) * 503 - 502 * 504 = 1 := by
  sorry

end difference_of_squares_special_case_l2225_222553


namespace julie_total_earnings_l2225_222504

/-- Calculates Julie's total earnings for September and October based on her landscaping business rates and hours worked. -/
theorem julie_total_earnings (
  -- September hours
  small_lawn_sept : ℕ) (large_lawn_sept : ℕ) (simple_garden_sept : ℕ) (complex_garden_sept : ℕ)
  (small_tree_sept : ℕ) (large_tree_sept : ℕ) (mulch_sept : ℕ)
  -- Rates
  (small_lawn_rate : ℕ) (large_lawn_rate : ℕ) (simple_garden_rate : ℕ) (complex_garden_rate : ℕ)
  (small_tree_rate : ℕ) (large_tree_rate : ℕ) (mulch_rate : ℕ)
  -- Given conditions
  (h1 : small_lawn_sept = 10) (h2 : large_lawn_sept = 15) (h3 : simple_garden_sept = 2)
  (h4 : complex_garden_sept = 1) (h5 : small_tree_sept = 5) (h6 : large_tree_sept = 5)
  (h7 : mulch_sept = 5)
  (h8 : small_lawn_rate = 4) (h9 : large_lawn_rate = 6) (h10 : simple_garden_rate = 8)
  (h11 : complex_garden_rate = 10) (h12 : small_tree_rate = 10) (h13 : large_tree_rate = 15)
  (h14 : mulch_rate = 12) :
  -- Theorem statement
  (small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
   simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
   small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
   mulch_rate * mulch_sept) +
  ((small_lawn_rate * small_lawn_sept + large_lawn_rate * large_lawn_sept +
    simple_garden_rate * simple_garden_sept + complex_garden_rate * complex_garden_sept +
    small_tree_rate * small_tree_sept + large_tree_rate * large_tree_sept +
    mulch_rate * mulch_sept) * 3 / 2) = 8525/10 := by
  sorry

end julie_total_earnings_l2225_222504


namespace football_tournament_l2225_222596

theorem football_tournament (n : ℕ) (k : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (n * (n - 1)) / 2 + k * n = 77 →  -- Total matches equation
  2 * n = 14  -- Prove that the initial number of teams is 14
  := by sorry

end football_tournament_l2225_222596


namespace equation_solution_l2225_222516

theorem equation_solution : ∃ x : ℝ, (17.28 / x) / (3.6 * 0.2) = 2 ∧ x = 12 := by
  sorry

end equation_solution_l2225_222516


namespace five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l2225_222582

-- Define the types of tiles
inductive Tile
| Domino    -- 2x1 tile
| Tetromino -- 4x1 tile

-- Define a board
structure Board :=
(rows : ℕ)
(cols : ℕ)
(removed_cells : List (ℕ × ℕ)) -- List of removed cells' coordinates

-- Define a function to check if a board can be tiled
def can_be_tiled (b : Board) (t : Tile) : Prop :=
  match t with
  | Tile.Domino    => sorry
  | Tile.Tetromino => sorry

-- Theorem 1: A 5x7 board cannot be tiled with dominoes
theorem five_by_seven_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [] } Tile.Domino :=
sorry

-- Theorem 2: A 5x7 board with bottom left corner removed can be tiled with dominoes
theorem five_by_seven_minus_corner_domino :
  can_be_tiled { rows := 5, cols := 7, removed_cells := [(1, 1)] } Tile.Domino :=
sorry

-- Theorem 3: A 5x7 board with leftmost cell on second row removed cannot be tiled with dominoes
theorem five_by_seven_minus_second_row_domino :
  ¬ can_be_tiled { rows := 5, cols := 7, removed_cells := [(2, 1)] } Tile.Domino :=
sorry

-- Theorem 4: A 6x6 board can be tiled with tetrominoes
theorem six_by_six_tetromino :
  can_be_tiled { rows := 6, cols := 6, removed_cells := [] } Tile.Tetromino :=
sorry

end five_by_seven_domino_five_by_seven_minus_corner_domino_five_by_seven_minus_second_row_domino_six_by_six_tetromino_l2225_222582


namespace remainder_is_perfect_square_l2225_222510

theorem remainder_is_perfect_square (n : ℕ+) : ∃ k : ℤ, (10^n.val - 1) % 37 = k^2 := by
  sorry

end remainder_is_perfect_square_l2225_222510


namespace complex_equation_solution_l2225_222517

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l2225_222517


namespace parallelogram_area_specific_vectors_l2225_222592

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : ℝ × ℝ) : ℝ :=
  |v.1 * w.2 - v.2 * w.1|

theorem parallelogram_area_specific_vectors :
  let v : ℝ × ℝ := (8, -5)
  let w : ℝ × ℝ := (14, -3)
  parallelogramArea v w = 46 := by
sorry

end parallelogram_area_specific_vectors_l2225_222592


namespace alternating_series_ratio_l2225_222591

theorem alternating_series_ratio : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128) / 
  (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2) = 1 / 113 := by
  sorry

end alternating_series_ratio_l2225_222591


namespace apple_preference_percentage_l2225_222586

theorem apple_preference_percentage (total_responses : ℕ) (apple_responses : ℕ) 
  (h1 : total_responses = 300) (h2 : apple_responses = 70) :
  (apple_responses : ℚ) / (total_responses : ℚ) * 100 = 23 := by
  sorry

end apple_preference_percentage_l2225_222586


namespace podcast5_duration_theorem_l2225_222561

def total_drive_time : ℕ := 6 * 60  -- in minutes
def podcast1_duration : ℕ := 45     -- in minutes
def podcast2_duration : ℕ := 2 * podcast1_duration
def podcast3_duration : ℕ := 105    -- 1 hour and 45 minutes in minutes
def podcast4_duration : ℕ := 60     -- 1 hour in minutes

def total_podcast_time : ℕ := podcast1_duration + podcast2_duration + podcast3_duration + podcast4_duration

theorem podcast5_duration_theorem :
  total_drive_time - total_podcast_time = 60 := by sorry

end podcast5_duration_theorem_l2225_222561


namespace prime_quadratic_roots_l2225_222533

theorem prime_quadratic_roots (p : ℕ) : 
  Nat.Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) → 
  p = 37 :=
sorry

end prime_quadratic_roots_l2225_222533


namespace fifth_power_minus_fifth_power_equals_sixteen_product_l2225_222554

theorem fifth_power_minus_fifth_power_equals_sixteen_product (m n : ℤ) :
  m^5 - n^5 = 16*m*n ↔ (m = 0 ∧ n = 0) ∨ (m = -2 ∧ n = 2) :=
sorry

end fifth_power_minus_fifth_power_equals_sixteen_product_l2225_222554


namespace monster_family_eyes_l2225_222548

/-- A monster family with a specific number of eyes for each member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  kid_eyes : ℕ

/-- Calculate the total number of eyes in a monster family -/
def total_eyes (family : MonsterFamily) : ℕ :=
  family.mom_eyes + family.dad_eyes + family.num_kids * family.kid_eyes

/-- Theorem: The total number of eyes in the given monster family is 16 -/
theorem monster_family_eyes :
  ∃ (family : MonsterFamily),
    family.mom_eyes = 1 ∧
    family.dad_eyes = 3 ∧
    family.num_kids = 3 ∧
    family.kid_eyes = 4 ∧
    total_eyes family = 16 := by
  sorry

end monster_family_eyes_l2225_222548


namespace simplify_expression_l2225_222528

theorem simplify_expression :
  81 * ((5 + 1/3) - (3 + 1/4)) / ((4 + 1/2) + (2 + 2/5)) = 225/92 := by
  sorry

end simplify_expression_l2225_222528


namespace interval_change_l2225_222583

/-- Represents the interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℚ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem interval_change (total_time : ℕ) :
  total_time = 2 * 21 →
  interval 2 total_time = 21 →
  interval 3 total_time = 14 := by
  sorry

#eval interval 3 42  -- Should output 14

end interval_change_l2225_222583


namespace roots_of_quadratic_equation_l2225_222576

theorem roots_of_quadratic_equation :
  let equation := fun (x : ℂ) => x^2 + 4
  ∃ (r₁ r₂ : ℂ), r₁ = -2*I ∧ r₂ = 2*I ∧ equation r₁ = 0 ∧ equation r₂ = 0 :=
by sorry

end roots_of_quadratic_equation_l2225_222576


namespace root_approximation_l2225_222544

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_approximation (root : ℕ+) :
  (f root = 0) →
  (f 1 = -2) →
  (f 1.5 = 0.625) →
  (f 1.25 = -0.984) →
  (f 1.375 = -0.260) →
  (f 1.4375 = 0.162) →
  (f 1.40625 = -0.054) →
  ∃ x : ℝ, x ∈ (Set.Ioo 1.375 1.4375) ∧ f x = 0 :=
by sorry

end root_approximation_l2225_222544


namespace class_vision_most_suitable_l2225_222538

/-- Represents a survey option -/
inductive SurveyOption
  | SleepTimeNationwide
  | RiverWaterQuality
  | PocketMoneyCity
  | ClassVision

/-- Checks if a survey option is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.ClassVision => true
  | _ => false

/-- Theorem stating that investigating the vision of all classmates in a class
    is the most suitable for a comprehensive survey -/
theorem class_vision_most_suitable :
  isSuitableForComprehensiveSurvey SurveyOption.ClassVision ∧
  (∀ (option : SurveyOption),
    isSuitableForComprehensiveSurvey option →
    option = SurveyOption.ClassVision) :=
by
  sorry

#check class_vision_most_suitable

end class_vision_most_suitable_l2225_222538


namespace similar_triangles_height_l2225_222565

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio ∧ h_large = 15 :=
by sorry

end similar_triangles_height_l2225_222565


namespace function_positivity_implies_m_range_l2225_222503

theorem function_positivity_implies_m_range 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = 2 * m * x^2 - 2 * (4 - m) * x + 1) 
  (h_g : ∀ x, g x = m * x) 
  (h_pos : ∀ x, f x > 0 ∨ g x > 0) : 
  0 < m ∧ m < 8 := by
sorry

end function_positivity_implies_m_range_l2225_222503


namespace prob_less_than_one_third_l2225_222566

/-- The probability that a number randomly selected from (0, 1/2) is less than 1/3 is 2/3. -/
theorem prob_less_than_one_third : 
  ∀ (P : Set ℝ → ℝ) (Ω : Set ℝ),
    (∀ a b, a < b → P (Set.Ioo a b) = b - a) →  -- P is a uniform probability measure
    Ω = Set.Ioo 0 (1/2) →                       -- Ω is the interval (0, 1/2)
    P {x ∈ Ω | x < 1/3} / P Ω = 2/3 := by
  sorry

end prob_less_than_one_third_l2225_222566


namespace recliner_sales_increase_l2225_222552

theorem recliner_sales_increase 
  (price_reduction : ℝ) 
  (gross_increase : ℝ) 
  (sales_increase : ℝ) : 
  price_reduction = 0.2 → 
  gross_increase = 0.4400000000000003 → 
  sales_increase = (1 + gross_increase) / (1 - price_reduction) - 1 →
  sales_increase = 0.8 := by
sorry

end recliner_sales_increase_l2225_222552


namespace pauls_lost_crayons_l2225_222574

/-- Given that Paul initially had 110 crayons, gave 90 crayons to his friends,
    and lost 322 more crayons than those he gave to his friends,
    prove that Paul lost 412 crayons. -/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (crayons_given : ℕ)
  (extra_lost_crayons : ℕ)
  (h1 : initial_crayons = 110)
  (h2 : crayons_given = 90)
  (h3 : extra_lost_crayons = 322)
  : crayons_given + extra_lost_crayons = 412 := by
  sorry

end pauls_lost_crayons_l2225_222574


namespace line_vector_proof_l2225_222500

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 3)) →
  (line_vector 5 = (12, -37)) →
  (line_vector (-3) = (-4, 27)) :=
by sorry

end line_vector_proof_l2225_222500


namespace different_color_prob_l2225_222578

def bag_prob (p_red_red p_white_white : ℚ) : Prop :=
  p_red_red = 2/15 ∧ p_white_white = 1/3

theorem different_color_prob (p_red_red p_white_white : ℚ) 
  (h : bag_prob p_red_red p_white_white) : 
  1 - (p_red_red + p_white_white) = 8/15 :=
sorry

end different_color_prob_l2225_222578


namespace latin_speakers_l2225_222530

/-- In a group of people, given the total number, the number of French speakers,
    the number of people speaking neither Latin nor French, and the number of people
    speaking both Latin and French, we can determine the number of Latin speakers. -/
theorem latin_speakers (total : ℕ) (french : ℕ) (neither : ℕ) (both : ℕ) :
  total = 25 →
  french = 15 →
  neither = 6 →
  both = 9 →
  ∃ latin : ℕ, latin = 13 ∧ latin + french - both = total - neither :=
by sorry

end latin_speakers_l2225_222530


namespace six_foldable_configurations_l2225_222542

/-- Represents a square in the puzzle -/
inductive Square
| A | B | C | D | E | F | G | H

/-- Represents the T-shaped figure -/
structure TShape :=
  (squares : Finset Square)
  (h_count : squares.card = 4)

/-- Represents a configuration of the puzzle -/
structure Configuration :=
  (base : TShape)
  (added : Square)

/-- Predicate to check if a configuration can be folded into a topless cubical box -/
def is_foldable (c : Configuration) : Prop :=
  sorry  -- Definition of foldability

/-- The main theorem statement -/
theorem six_foldable_configurations :
  ∃ (valid_configs : Finset Configuration),
    valid_configs.card = 6 ∧
    (∀ c ∈ valid_configs, is_foldable c) ∧
    (∀ c : Configuration, is_foldable c → c ∈ valid_configs) :=
  sorry

end six_foldable_configurations_l2225_222542


namespace basketball_shooting_averages_l2225_222508

/-- Represents the average number of successful shots -/
structure ShootingAverage where
  male : ℝ
  female : ℝ

/-- Represents the number of students -/
structure StudentCount where
  male : ℝ
  female : ℝ

/-- The theorem stating the average number of successful shots for male and female students -/
theorem basketball_shooting_averages 
  (avg : ShootingAverage) 
  (count : StudentCount) 
  (h1 : avg.male = 1.25 * avg.female) 
  (h2 : count.female = 1.25 * count.male) 
  (h3 : (avg.male * count.male + avg.female * count.female) / (count.male + count.female) = 4) :
  avg.male = 4.5 ∧ avg.female = 3.6 := by
  sorry

#check basketball_shooting_averages

end basketball_shooting_averages_l2225_222508


namespace chocolate_distribution_chocolate_problem_l2225_222581

theorem chocolate_distribution (initial_bars : ℕ) (sisters : ℕ) (father_ate : ℕ) (father_left : ℕ) : ℕ :=
  let total_people := sisters + 1
  let bars_per_person := initial_bars / total_people
  let bars_given_to_father := (bars_per_person / 2) * total_people
  let bars_father_had := bars_given_to_father - father_ate
  bars_father_had - father_left

theorem chocolate_problem : 
  chocolate_distribution 20 4 2 5 = 3 := by sorry

end chocolate_distribution_chocolate_problem_l2225_222581


namespace team_a_win_probability_l2225_222587

def number_of_matches : ℕ := 7
def wins_required : ℕ := 4
def win_probability : ℚ := 1/2

theorem team_a_win_probability :
  (number_of_matches.choose wins_required) * win_probability ^ wins_required * (1 - win_probability) ^ (number_of_matches - wins_required) = 35/128 := by
  sorry

end team_a_win_probability_l2225_222587


namespace business_investment_problem_l2225_222549

/-- Proves that A's investment is 16000, given the conditions of the business problem -/
theorem business_investment_problem (b_investment c_investment : ℕ) 
  (b_profit : ℕ) (profit_difference : ℕ) :
  b_investment = 10000 →
  c_investment = 12000 →
  b_profit = 1400 →
  profit_difference = 560 →
  ∃ (a_investment : ℕ), 
    a_investment * b_profit = b_investment * (a_investment * b_profit / b_investment - c_investment * b_profit / b_investment + profit_difference) ∧ 
    a_investment = 16000 := by
  sorry

end business_investment_problem_l2225_222549


namespace no_valid_arrangement_l2225_222524

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Checks if two positions in the grid are adjacent -/
def isAdjacent (x1 y1 x2 y2 : Fin 3) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y2 = y1 + 1)) ∨
  (y1 = y2 ∧ (x1 = x2 + 1 ∨ x2 = x1 + 1))

/-- Checks if a grid arrangement is valid according to the problem conditions -/
def isValidArrangement (g : Grid) : Prop :=
  (∀ x y : Fin 3, g x y ∈ Finset.range 9) ∧
  (∀ x1 y1 x2 y2 : Fin 3, isAdjacent x1 y1 x2 y2 → isPrime (g x1 y1 + g x2 y2)) ∧
  (∀ n : Fin 9, ∃ x y : Fin 3, g x y = n + 1)

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬∃ g : Grid, isValidArrangement g := by
  sorry

end no_valid_arrangement_l2225_222524


namespace x_values_l2225_222558

theorem x_values (A : Set ℝ) (x : ℝ) (h1 : A = {0, 1, x^2 - 5*x}) (h2 : -4 ∈ A) :
  x = 1 ∨ x = 4 := by
sorry

end x_values_l2225_222558


namespace line_through_points_l2225_222593

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line_equation a b 6 = 7) →
  (line_equation a b 10 = 23) →
  a + b = -13 := by
  sorry

end line_through_points_l2225_222593


namespace cookie_cost_claire_cookie_cost_l2225_222572

/-- The cost of a cookie given Claire's spending habits and gift card balance --/
theorem cookie_cost (gift_card : ℝ) (latte_cost : ℝ) (croissant_cost : ℝ) 
  (days : ℕ) (num_cookies : ℕ) (remaining_balance : ℝ) : ℝ :=
  let daily_treat_cost := latte_cost + croissant_cost
  let weekly_treat_cost := daily_treat_cost * days
  let total_spent := gift_card - remaining_balance
  let cookie_total_cost := total_spent - weekly_treat_cost
  cookie_total_cost / num_cookies

/-- Proof that each cookie costs $1.25 given Claire's spending habits --/
theorem claire_cookie_cost : 
  cookie_cost 100 3.75 3.50 7 5 43 = 1.25 := by
  sorry

end cookie_cost_claire_cookie_cost_l2225_222572


namespace tabitha_honey_days_l2225_222537

/-- Represents the number of days Tabitha can enjoy honey in her tea --/
def honey_days (servings_per_cup : ℕ) (evening_cups : ℕ) (morning_cups : ℕ) 
               (container_ounces : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_ounces * servings_per_ounce) / (servings_per_cup * (evening_cups + morning_cups))

/-- Theorem stating that Tabitha can enjoy honey in her tea for 32 days --/
theorem tabitha_honey_days : 
  honey_days 1 2 1 16 6 = 32 := by
  sorry

end tabitha_honey_days_l2225_222537


namespace power_of_32_l2225_222571

theorem power_of_32 (n : ℕ) : 
  2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n → n = 81 := by
  sorry

end power_of_32_l2225_222571


namespace product_equals_fraction_l2225_222511

/-- The repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by
  sorry

end product_equals_fraction_l2225_222511


namespace f_composition_of_three_l2225_222597

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_composition_of_three : f (f (f 3)) = 107 := by
  sorry

end f_composition_of_three_l2225_222597


namespace congruent_triangles_equal_perimeters_l2225_222570

/-- Two triangles are congruent if they have the same shape and size -/
def CongruentTriangles (T1 T2 : Set (ℝ × ℝ)) : Prop := sorry

/-- The perimeter of a triangle is the sum of the lengths of its sides -/
def Perimeter (T : Set (ℝ × ℝ)) : ℝ := sorry

/-- If two triangles are congruent, then their perimeters are equal -/
theorem congruent_triangles_equal_perimeters (T1 T2 : Set (ℝ × ℝ)) :
  CongruentTriangles T1 T2 → Perimeter T1 = Perimeter T2 := by sorry

end congruent_triangles_equal_perimeters_l2225_222570


namespace log_inequality_l2225_222573

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality (a x₁ x₂ : ℝ) (ha : a > 0 ∧ a ≠ 1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  (a > 1 → (f a x₁ + f a x₂) / 2 ≤ f a ((x₁ + x₂) / 2)) ∧
  (0 < a ∧ a < 1 → (f a x₁ + f a x₂) / 2 ≥ f a ((x₁ + x₂) / 2)) :=
by sorry

end log_inequality_l2225_222573


namespace outfit_count_l2225_222521

/-- The number of red shirts -/
def red_shirts : ℕ := 7

/-- The number of blue shirts -/
def blue_shirts : ℕ := 7

/-- The number of pairs of pants -/
def pants : ℕ := 10

/-- The number of green hats -/
def green_hats : ℕ := 9

/-- The number of red hats -/
def red_hats : ℕ := 9

/-- Each piece of clothing is distinct -/
axiom distinct_clothing : red_shirts + blue_shirts + pants + green_hats + red_hats = red_shirts + blue_shirts + pants + green_hats + red_hats

/-- The number of outfits where the shirt and hat are never the same color -/
def num_outfits : ℕ := red_shirts * pants * green_hats + blue_shirts * pants * red_hats

theorem outfit_count : num_outfits = 1260 := by
  sorry

end outfit_count_l2225_222521


namespace practice_time_is_three_l2225_222507

/-- Calculates the practice time per minute of singing given the performance duration,
    tantrum time per minute of singing, and total time. -/
def practice_time_per_minute (performance_duration : ℕ) (tantrum_time_per_minute : ℕ) (total_time : ℕ) : ℕ :=
  ((total_time - performance_duration) / performance_duration) - tantrum_time_per_minute

/-- Proves that given a 6-minute performance, 5 minutes of tantrums per minute of singing,
    and a total time of 54 minutes, the practice time per minute of singing is 3 minutes. -/
theorem practice_time_is_three :
  practice_time_per_minute 6 5 54 = 3 := by
  sorry

#eval practice_time_per_minute 6 5 54

end practice_time_is_three_l2225_222507


namespace triangle_area_scaling_l2225_222568

theorem triangle_area_scaling (original_area new_area : ℝ) : 
  new_area = 54 → 
  new_area = 9 * original_area → 
  original_area = 6 := by
sorry

end triangle_area_scaling_l2225_222568


namespace crop_planting_problem_l2225_222520

/-- Cost function for planting crops -/
def cost_function (x : ℝ) : ℝ := x^2 + 5*x + 10

/-- Revenue function for planting crops -/
def revenue_function (x : ℝ) : ℝ := 15*x

/-- Profit function for planting crops -/
def profit_function (x : ℝ) : ℝ := revenue_function x - cost_function x

theorem crop_planting_problem :
  (cost_function 1 = 16 ∧ cost_function 3 = 34) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ cost_function x₁ / x₁ = 12 ∧ cost_function x₂ / x₂ = 12) ∧
  (∃ x_max : ℝ, x_max = 5 ∧ 
    ∀ x : ℝ, profit_function x ≤ profit_function x_max ∧ 
    profit_function x_max = 15) :=
by sorry

#check crop_planting_problem

end crop_planting_problem_l2225_222520


namespace last_season_episodes_l2225_222531

/-- The number of seasons before the announcement -/
def previous_seasons : ℕ := 9

/-- The number of episodes in each regular season -/
def episodes_per_season : ℕ := 22

/-- The duration of each episode in hours -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all seasons in hours -/
def total_watch_time : ℚ := 112

/-- The additional episodes in the last season compared to regular seasons -/
def additional_episodes : ℕ := 4

theorem last_season_episodes (last_season_episodes : ℕ) :
  last_season_episodes = episodes_per_season + additional_episodes ∧
  (previous_seasons * episodes_per_season + last_season_episodes) * episode_duration = total_watch_time :=
by sorry

end last_season_episodes_l2225_222531


namespace line_parallel_perpendicular_implies_planes_perpendicular_l2225_222522

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → planes_perpendicular α β :=
by sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l2225_222522


namespace tan_alpha_plus_pi_fourth_l2225_222519

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h : 6 * Real.sin α * Real.cos α = 1 + Real.cos (2 * α)) : 
  Real.tan (α + π/4) = 2 ∨ Real.tan (α + π/4) = -1 := by
  sorry

end tan_alpha_plus_pi_fourth_l2225_222519


namespace square_root_sum_equals_four_root_six_l2225_222588

theorem square_root_sum_equals_four_root_six :
  Real.sqrt (16 - 8 * Real.sqrt 6) + Real.sqrt (16 + 8 * Real.sqrt 6) = 4 * Real.sqrt 6 := by
  sorry

end square_root_sum_equals_four_root_six_l2225_222588


namespace tan_five_pi_four_equals_one_l2225_222543

theorem tan_five_pi_four_equals_one : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_four_equals_one_l2225_222543


namespace exists_two_equal_types_l2225_222555

/-- Represents the types of sweets -/
inductive SweetType
  | Blackberry
  | Coconut
  | Chocolate

/-- Represents the number of sweets for each type -/
structure Sweets where
  blackberry : Nat
  coconut : Nat
  chocolate : Nat

/-- The initial number of sweets -/
def initialSweets : Sweets :=
  { blackberry := 7, coconut := 6, chocolate := 3 }

/-- The number of sweets Sofia eats -/
def eatenSweets : Nat := 2

/-- Checks if two types of sweets have the same number -/
def hasTwoEqualTypes (s : Sweets) : Prop :=
  (s.blackberry = s.coconut) ∨ (s.blackberry = s.chocolate) ∨ (s.coconut = s.chocolate)

/-- Theorem: It's possible for grandmother to receive the same number of sweets for two varieties -/
theorem exists_two_equal_types :
  ∃ (finalSweets : Sweets),
    finalSweets.blackberry + finalSweets.coconut + finalSweets.chocolate =
      initialSweets.blackberry + initialSweets.coconut + initialSweets.chocolate - eatenSweets ∧
    finalSweets.blackberry ≤ initialSweets.blackberry ∧
    finalSweets.coconut ≤ initialSweets.coconut ∧
    finalSweets.chocolate ≤ initialSweets.chocolate ∧
    hasTwoEqualTypes finalSweets :=
  sorry

end exists_two_equal_types_l2225_222555


namespace parallel_vectors_x_value_l2225_222569

/-- Given two parallel vectors a and b in R³, where a = (2, -1, 2) and b = (-4, 2, x),
    prove that x = -4. -/
theorem parallel_vectors_x_value (a b : ℝ × ℝ × ℝ) (x : ℝ) :
  a = (2, -1, 2) →
  b = (-4, 2, x) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  x = -4 := by sorry

end parallel_vectors_x_value_l2225_222569


namespace watch_loss_percentage_l2225_222539

theorem watch_loss_percentage (CP : ℝ) (SP : ℝ) : 
  CP = 1357.142857142857 →
  SP + 190 = CP * (1 + 4 / 100) →
  (CP - SP) / CP * 100 = 10 := by
sorry

end watch_loss_percentage_l2225_222539


namespace log_equation_solution_l2225_222567

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 := by
  sorry

end log_equation_solution_l2225_222567


namespace min_value_and_y_l2225_222556

theorem min_value_and_y (x y z : ℝ) (h : 2*x - 3*y + z = 3) :
  ∃ (min_val : ℝ), 
    (∀ x' y' z' : ℝ, 2*x' - 3*y' + z' = 3 → x'^2 + (y'-1)^2 + z'^2 ≥ min_val) ∧
    (x^2 + (y-1)^2 + z^2 = min_val) ∧
    (min_val = 18/7) ∧
    (y = -2/7) := by
  sorry

end min_value_and_y_l2225_222556


namespace triangle_inequality_l2225_222523

/-- Given a triangle with circumradius R, inradius r, side lengths a, b, c, and semiperimeter p,
    prove that 20Rr - 4r^2 ≤ ab + bc + ca ≤ 4(R + r)^2 -/
theorem triangle_inequality (R r a b c p : ℝ) (hR : R > 0) (hr : r > 0)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hp : p = (a + b + c) / 2)
    (hcirc : R = a * b * c / (4 * p * r)) (hinr : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
    20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 4 * (R + r)^2 := by
  sorry

end triangle_inequality_l2225_222523


namespace book_purchase_problem_l2225_222579

theorem book_purchase_problem :
  ∀ (total_A total_B only_A only_B both : ℕ),
    total_A = 2 * total_B →
    both = 500 →
    both = 2 * only_B →
    total_A = only_A + both →
    total_B = only_B + both →
    only_A = 1000 := by
  sorry

end book_purchase_problem_l2225_222579


namespace sequence_formula_and_sum_bound_l2225_222525

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula_and_sum_bound :
  (∀ n : ℕ+, a n = S n - S (n-1)) ∧
  (∃ m : ℕ+, (∀ n : ℕ+, T n < m / 20) ∧
             (∀ k : ℕ+, k < m → ∃ n : ℕ+, T n ≥ k / 20)) :=
sorry

end sequence_formula_and_sum_bound_l2225_222525


namespace books_sold_and_remaining_l2225_222541

/-- Given that a person sells 45 books and has 6 books remaining, prove that they initially had 51 books. -/
theorem books_sold_and_remaining (books_sold : ℕ) (books_remaining : ℕ) : 
  books_sold = 45 → books_remaining = 6 → books_sold + books_remaining = 51 :=
by sorry

end books_sold_and_remaining_l2225_222541


namespace unwashed_shirts_l2225_222560

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) : 
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end unwashed_shirts_l2225_222560


namespace quadratic_roots_property_l2225_222590

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ,
  x₁^2 - 4*x₁ + 2 = 0 →
  x₂^2 - 4*x₂ + 2 = 0 →
  x₁ + x₂ - x₁*x₂ = 2 := by
  sorry

end quadratic_roots_property_l2225_222590


namespace problem_solution_l2225_222540

theorem problem_solution (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -5) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -10.528 := by
  sorry

end problem_solution_l2225_222540


namespace triangle_angles_from_radii_relations_l2225_222509

/-- Given a triangle with excircle radii r_a, r_b, r_c, and circumcircle radius R,
    if r_a + r_b = 3R and r_b + r_c = 2R, then the angles of the triangle are 90°, 60°, and 30°. -/
theorem triangle_angles_from_radii_relations (r_a r_b r_c R : ℝ) 
    (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
    ∃ (α β γ : ℝ),
      α = π / 2 ∧ β = π / 6 ∧ γ = π / 3 ∧
      α + β + γ = π ∧
      0 < α ∧ 0 < β ∧ 0 < γ :=
by sorry

end triangle_angles_from_radii_relations_l2225_222509
