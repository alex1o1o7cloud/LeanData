import Mathlib

namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1128_112830

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1128_112830


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1128_112867

theorem inequality_solution_set (x : ℝ) :
  (3 / (x + 2) + 4 / (x + 8) ≥ 3 / 4) ↔ 
  (x ∈ Set.Icc (-10.125) (-8) ∪ Set.Ico (-2) 4.125) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1128_112867


namespace NUMINAMATH_CALUDE_fifth_element_row_20_value_l1128_112888

/-- Pascal's triangle element -/
def pascal_triangle_element (n k : ℕ) : ℕ := Nat.choose n k

/-- The fifth element in Row 20 of Pascal's triangle -/
def fifth_element_row_20 : ℕ := pascal_triangle_element 20 4

theorem fifth_element_row_20_value : fifth_element_row_20 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_value_l1128_112888


namespace NUMINAMATH_CALUDE_max_candy_leftover_l1128_112852

theorem max_candy_leftover (x : ℕ) (h : x > 11) : 
  ∃ (q r : ℕ), x = 11 * q + r ∧ r > 0 ∧ r ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l1128_112852


namespace NUMINAMATH_CALUDE_tim_seashells_l1128_112827

/-- The number of seashells Tim found initially -/
def initial_seashells : ℕ := 679

/-- The number of seashells Tim gave to Sara -/
def seashells_given : ℕ := 172

/-- The number of seashells Tim has after giving some to Sara -/
def remaining_seashells : ℕ := initial_seashells - seashells_given

theorem tim_seashells : remaining_seashells = 507 := by sorry

end NUMINAMATH_CALUDE_tim_seashells_l1128_112827


namespace NUMINAMATH_CALUDE_quadratic_polynomial_root_sum_l1128_112877

theorem quadratic_polynomial_root_sum (Q : ℝ → ℝ) (a b c : ℝ) :
  (∀ x : ℝ, Q x = a * x^2 + b * x + c) →
  (∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) →
  (∃ r₁ r₂ : ℝ, ∀ x : ℝ, Q x = a * (x - r₁) * (x - r₂)) →
  r₁ + r₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_root_sum_l1128_112877


namespace NUMINAMATH_CALUDE_hexagon_division_l1128_112892

/-- Given a regular hexagon with area 21.12 square centimeters divided into 6 equal pieces,
    prove that the area of each piece is 3.52 square centimeters. -/
theorem hexagon_division (hexagon_area : ℝ) (num_pieces : ℕ) (piece_area : ℝ) :
  hexagon_area = 21.12 ∧ num_pieces = 6 ∧ piece_area = hexagon_area / num_pieces →
  piece_area = 3.52 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_division_l1128_112892


namespace NUMINAMATH_CALUDE_medicine_weight_l1128_112849

/-- Represents the weight system used for measurement -/
inductive WeightSystem
  | Ancient
  | Modern

/-- Represents a weight measurement -/
structure Weight where
  jin : ℕ
  liang : ℕ
  system : WeightSystem

/-- Converts a Weight to grams -/
def Weight.toGrams (w : Weight) : ℕ :=
  match w.system with
  | WeightSystem.Ancient => w.jin * 600 + w.liang * (600 / 16)
  | WeightSystem.Modern => w.jin * 500 + w.liang * (500 / 10)

/-- The theorem to be proved -/
theorem medicine_weight (w₁ w₂ : Weight) 
  (h₁ : w₁.system = WeightSystem.Ancient)
  (h₂ : w₂.system = WeightSystem.Modern)
  (h₃ : w₁.jin + w₂.jin = 5)
  (h₄ : w₁.liang + w₂.liang = 68)
  (h₅ : w₁.jin * 16 + w₁.liang = w₁.liang)
  (h₆ : w₂.jin * 10 + w₂.liang = w₂.liang) :
  w₁.toGrams + w₂.toGrams = 2800 := by
  sorry


end NUMINAMATH_CALUDE_medicine_weight_l1128_112849


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1128_112812

theorem cube_plus_reciprocal_cube (a : ℝ) (h : (a + 1/a)^2 = 3) :
  a^3 + 1/a^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l1128_112812


namespace NUMINAMATH_CALUDE_asian_games_mascot_sales_l1128_112883

/-- Represents the sales situation of Asian Games mascots -/
theorem asian_games_mascot_sales 
  (initial_sales : ℕ) 
  (total_sales_next_two_days : ℕ) 
  (growth_rate : ℝ) :
  initial_sales = 5000 →
  total_sales_next_two_days = 30000 →
  (initial_sales : ℝ) * (1 + growth_rate) + (initial_sales : ℝ) * (1 + growth_rate)^2 = total_sales_next_two_days :=
by sorry

end NUMINAMATH_CALUDE_asian_games_mascot_sales_l1128_112883


namespace NUMINAMATH_CALUDE_train_capacity_l1128_112824

/-- Proves that given a train with 4 carriages, each initially having 25 seats
    and can accommodate 10 more passengers, the total number of passengers
    that would fill up 3 such trains is 420. -/
theorem train_capacity (initial_seats : Nat) (additional_seats : Nat) 
  (carriages_per_train : Nat) (number_of_trains : Nat) :
  initial_seats = 25 →
  additional_seats = 10 →
  carriages_per_train = 4 →
  number_of_trains = 3 →
  (initial_seats + additional_seats) * carriages_per_train * number_of_trains = 420 := by
  sorry

#eval (25 + 10) * 4 * 3  -- Should output 420

end NUMINAMATH_CALUDE_train_capacity_l1128_112824


namespace NUMINAMATH_CALUDE_string_average_length_l1128_112806

theorem string_average_length : 
  let strings : List ℚ := [2, 5, 7]
  (strings.sum / strings.length : ℚ) = 14/3 := by
sorry

end NUMINAMATH_CALUDE_string_average_length_l1128_112806


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1128_112829

theorem algebraic_expression_equality (x y : ℝ) : 
  x + 2*y + 1 = 3 → 2*x + 4*y + 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1128_112829


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1128_112828

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (1 : ℂ) / ((1 + Complex.I)^2 + 1) + Complex.I
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1128_112828


namespace NUMINAMATH_CALUDE_exam_average_l1128_112876

theorem exam_average (students1 : ℕ) (average1 : ℚ) (students2 : ℕ) (average2 : ℚ) : 
  students1 = 15 → 
  average1 = 75 / 100 → 
  students2 = 10 → 
  average2 = 95 / 100 → 
  (students1 * average1 + students2 * average2) / (students1 + students2) = 83 / 100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l1128_112876


namespace NUMINAMATH_CALUDE_largest_n_with_difference_seven_l1128_112826

/-- A function that checks if a list of natural numbers contains two numbers with a difference of 7 -/
def hasDifferenceSeven (list : List Nat) : Prop :=
  ∃ x y, x ∈ list ∧ y ∈ list ∧ x ≠ y ∧ (x - y = 7 ∨ y - x = 7)

/-- A function that checks if all selections of 50 numbers from 1 to n satisfy the difference condition -/
def allSelectionsHaveDifferenceSeven (n : Nat) : Prop :=
  ∀ list : List Nat, list.Nodup → list.length = 50 → (∀ x ∈ list, x ≥ 1 ∧ x ≤ n) →
    hasDifferenceSeven list

/-- The main theorem stating that 98 is the largest number satisfying the condition -/
theorem largest_n_with_difference_seven :
  allSelectionsHaveDifferenceSeven 98 ∧
  ¬(allSelectionsHaveDifferenceSeven 99) := by
  sorry

#check largest_n_with_difference_seven

end NUMINAMATH_CALUDE_largest_n_with_difference_seven_l1128_112826


namespace NUMINAMATH_CALUDE_factorial_divisibility_l1128_112869

theorem factorial_divisibility (n : ℕ) (h : 1 ≤ n ∧ n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^3 - 1)) = k * (Nat.factorial n)^(n + 1) :=
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l1128_112869


namespace NUMINAMATH_CALUDE_tony_fish_problem_l1128_112825

/-- The number of fish Tony has after a given number of years -/
def fish_count (initial : ℕ) (years : ℕ) : ℕ :=
  initial + years

theorem tony_fish_problem (x : ℕ) :
  fish_count x 5 = 7 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_problem_l1128_112825


namespace NUMINAMATH_CALUDE_triangle_side_length_theorem_l1128_112810

def triangle_side_length (a : ℝ) : Set ℝ :=
  if a < Real.sqrt 3 / 2 then
    ∅
  else if a = Real.sqrt 3 / 2 then
    {1/2}
  else if a < 1 then
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2, (1 - Real.sqrt (4 * a^2 - 3)) / 2}
  else
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2}

theorem triangle_side_length_theorem (a : ℝ) :
  let A : ℝ := 60 * π / 180
  let AB : ℝ := 1
  let BC : ℝ := a
  ∀ AC ∈ triangle_side_length a,
    AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos A :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_theorem_l1128_112810


namespace NUMINAMATH_CALUDE_calculate_expression_l1128_112878

theorem calculate_expression : -1^4 - 1/4 * (2 - (-3)^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1128_112878


namespace NUMINAMATH_CALUDE_equation_solution_l1128_112835

theorem equation_solution :
  ∃ y : ℚ, (y + 1/3 = 3/8 - 1/4) ∧ (y = -5/24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1128_112835


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1128_112808

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1128_112808


namespace NUMINAMATH_CALUDE_g_one_equals_three_l1128_112814

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the given equations
axiom eq1 : f (-1) + g 1 = 2
axiom eq2 : f 1 + g (-1) = 4

-- State the theorem to be proved
theorem g_one_equals_three : g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l1128_112814


namespace NUMINAMATH_CALUDE_quadratic_negative_range_l1128_112832

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_negative_range (a b c : ℝ) :
  (quadratic a b c (-1) = 4) →
  (quadratic a b c 0 = 0) →
  (∃ n, quadratic a b c 1 = n) →
  (∃ m, quadratic a b c 2 = m) →
  (quadratic a b c 3 = 4) →
  (∀ x : ℝ, quadratic a b c x < 0 ↔ 0 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_range_l1128_112832


namespace NUMINAMATH_CALUDE_correct_average_weight_l1128_112889

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.9 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l1128_112889


namespace NUMINAMATH_CALUDE_april_coffee_expenditure_l1128_112822

/-- Calculates the total expenditure on coffee for a month given the number of coffees per day, 
    cost per coffee, and number of days in the month. -/
def coffee_expenditure (coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_month : ℕ) : ℕ :=
  coffees_per_day * cost_per_coffee * days_in_month

theorem april_coffee_expenditure :
  coffee_expenditure 2 2 30 = 120 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_expenditure_l1128_112822


namespace NUMINAMATH_CALUDE_survey_result_l1128_112898

theorem survey_result (total : ℕ) (radio_dislike_percent : ℚ) (music_dislike_percent : ℚ)
  (h_total : total = 1500)
  (h_radio : radio_dislike_percent = 40 / 100)
  (h_music : music_dislike_percent = 15 / 100) :
  (total : ℚ) * radio_dislike_percent * music_dislike_percent = 90 :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l1128_112898


namespace NUMINAMATH_CALUDE_range_of_m_l1128_112887

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - 1 - m) * (x - 1 + m) ≤ 0}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (m > 0 ∧ A ⊂ B m) → m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1128_112887


namespace NUMINAMATH_CALUDE_floor_painting_overlap_l1128_112881

theorem floor_painting_overlap (red green blue : ℝ) 
  (h_red : red = 0.75) 
  (h_green : green = 0.7) 
  (h_blue : blue = 0.65) : 
  1 - (1 - red + 1 - green + 1 - blue) ≥ 0.1 := by sorry

end NUMINAMATH_CALUDE_floor_painting_overlap_l1128_112881


namespace NUMINAMATH_CALUDE_triangle_angle_from_sides_l1128_112842

theorem triangle_angle_from_sides (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (ha : a = Real.sqrt 3) (hb : b = 1) (hc : c = 2) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_from_sides_l1128_112842


namespace NUMINAMATH_CALUDE_remainder_of_7_pow_2023_mod_17_l1128_112847

theorem remainder_of_7_pow_2023_mod_17 : 7^2023 % 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7_pow_2023_mod_17_l1128_112847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1128_112821

theorem arithmetic_sequence_problem (n : ℕ) (min max sum : ℚ) (h_n : n = 150) 
  (h_min : min = 20) (h_max : max = 90) (h_sum : sum = 9000) :
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L := avg - (29 * d)
  let G := avg + (29 * d)
  G - L = 7140 / 149 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1128_112821


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l1128_112813

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 :
  ∃ (N : ℕ), sum_of_divisors 360 = N ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → p ≤ 13) ∧
  13 ∣ N ∧ Nat.Prime 13 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l1128_112813


namespace NUMINAMATH_CALUDE_linear_system_solution_l1128_112837

theorem linear_system_solution :
  ∃! (x y : ℝ), (x - y = 1) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1128_112837


namespace NUMINAMATH_CALUDE_music_library_avg_mb_per_hour_l1128_112834

/-- Represents a digital music library -/
structure MusicLibrary where
  days : ℕ
  space : ℕ

/-- Calculates the average megabytes per hour for a given music library -/
def avgMBPerHour (lib : MusicLibrary) : ℚ :=
  lib.space / (lib.days * 24)

/-- Theorem stating that for a music library with 15 days of music and 21,600 MB of space,
    the average megabytes per hour is 60 -/
theorem music_library_avg_mb_per_hour :
  let lib : MusicLibrary := { days := 15, space := 21600 }
  avgMBPerHour lib = 60 := by
  sorry

end NUMINAMATH_CALUDE_music_library_avg_mb_per_hour_l1128_112834


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l1128_112809

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l1128_112809


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1128_112839

def total_arrangements : ℕ := Nat.choose 6 2

def non_adjacent_arrangements : ℕ := Nat.choose 5 2

theorem zeros_not_adjacent_probability :
  (non_adjacent_arrangements : ℚ) / total_arrangements = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1128_112839


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l1128_112857

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a c : ℝ, (x - m) * (x - 3) = a * x^2 + c) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l1128_112857


namespace NUMINAMATH_CALUDE_base_10_729_equals_base_7_2061_l1128_112816

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Theorem: 729 in base-10 is equal to 2061 in base-7 --/
theorem base_10_729_equals_base_7_2061 :
  729 = base7ToBase10 [1, 6, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_10_729_equals_base_7_2061_l1128_112816


namespace NUMINAMATH_CALUDE_line_intercepts_minimum_minimum_sum_of_intercepts_l1128_112886

theorem line_intercepts_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) : 
  (b / a) + (a / b) ≥ 2 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = a * b ∧ x / a + y / b = 2) :=
by sorry

theorem minimum_sum_of_intercepts (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_line : a + b = a * b) :
  a + b ≥ 4 ∧ (a + b = 4 ↔ a = 2 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_minimum_minimum_sum_of_intercepts_l1128_112886


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l1128_112895

/-- The probability of all Russian players being paired with each other in a tennis tournament -/
theorem russian_players_pairing_probability
  (total_players : ℕ)
  (russian_players : ℕ)
  (h1 : total_players = 10)
  (h2 : russian_players = 4)
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / (total_players.choose 2) = 1 / 21 :=
sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l1128_112895


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1128_112861

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b = 2 * a →
  c = 3 * a →
  a + b + c = 180 →
  c = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l1128_112861


namespace NUMINAMATH_CALUDE_odd_monotone_function_range_theorem_l1128_112896

/-- A function that is odd and monotonically increasing on non-negative reals -/
def OddMonotoneFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

/-- The theorem statement -/
theorem odd_monotone_function_range_theorem (f : ℝ → ℝ) (h : OddMonotoneFunction f) :
  {x : ℝ | f (x^2 - x - 1) < f 5} = Set.Ioo (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_monotone_function_range_theorem_l1128_112896


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1128_112884

theorem cousins_ages_sum (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 →  -- single-digit ages
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct ages
  (a * b = 20 ∧ c * d = 36) ∨ (a * c = 20 ∧ b * d = 36) ∨ 
  (a * d = 20 ∧ b * c = 36) →  -- product conditions
  a + b + c + d = 21 := by
sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1128_112884


namespace NUMINAMATH_CALUDE_find_principal_amount_l1128_112891

/-- Given compound and simple interest for 2 years, find the principal amount -/
theorem find_principal_amount (compound_interest simple_interest : ℚ) : 
  compound_interest = 11730 → 
  simple_interest = 10200 → 
  ∃ (principal rate : ℚ), 
    principal > 0 ∧ 
    rate > 0 ∧ 
    rate < 100 ∧
    compound_interest = principal * ((1 + rate / 100) ^ 2 - 1) ∧
    simple_interest = principal * rate * 2 / 100 ∧
    principal = 1700 :=
by sorry

end NUMINAMATH_CALUDE_find_principal_amount_l1128_112891


namespace NUMINAMATH_CALUDE_red_cars_count_l1128_112894

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  ∃ (red_cars : ℕ), red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l1128_112894


namespace NUMINAMATH_CALUDE_exists_x0_implies_a_value_l1128_112811

noncomputable def f (a x : ℝ) : ℝ := Real.exp (x + a) + x

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 3) - 4 * Real.exp (-x - a)

theorem exists_x0_implies_a_value (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 2) → a = 2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_x0_implies_a_value_l1128_112811


namespace NUMINAMATH_CALUDE_books_checked_out_wednesday_l1128_112840

theorem books_checked_out_wednesday (initial_books : ℕ) (thursday_returned : ℕ) 
  (thursday_checked_out : ℕ) (friday_returned : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  thursday_returned = 23 →
  thursday_checked_out = 5 →
  friday_returned = 7 →
  final_books = 80 →
  ∃ (wednesday_checked_out : ℕ),
    wednesday_checked_out = 43 ∧
    final_books = initial_books - wednesday_checked_out + thursday_returned - 
      thursday_checked_out + friday_returned :=
by
  sorry

end NUMINAMATH_CALUDE_books_checked_out_wednesday_l1128_112840


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l1128_112831

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ :=
  a / b

-- Axioms for the diamond operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) * c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- Theorem to prove
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 504 (diamond 12 x) = 50 → x = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l1128_112831


namespace NUMINAMATH_CALUDE_profit_and_maximum_l1128_112865

noncomputable section

-- Define the sales volume function
def p (x : ℝ) : ℝ := 3 - 2 / (x + 1)

-- Define the profit function
def y (x : ℝ) : ℝ := 16 - 4 / (x + 1) - x

-- Theorem for the profit function and its maximum
theorem profit_and_maximum (a : ℝ) (h_a : a > 0) :
  -- The profit function
  (∀ x, 0 ≤ x ∧ x ≤ a → y x = 16 - 4 / (x + 1) - x) ∧
  -- Maximum profit when a ≥ 1
  (a ≥ 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 13 ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) ∧
  -- Maximum profit when a < 1
  (a < 1 → ∃ x, 0 ≤ x ∧ x ≤ a ∧ y x = 16 - 4 / (a + 1) - a ∧ ∀ x', 0 ≤ x' ∧ x' ≤ a → y x' ≤ y x) :=
sorry

end

end NUMINAMATH_CALUDE_profit_and_maximum_l1128_112865


namespace NUMINAMATH_CALUDE_nine_friends_with_pears_l1128_112864

/-- The number of friends carrying pears -/
def friends_with_pears (total_friends orange_friends : ℕ) : ℕ :=
  total_friends - orange_friends

/-- Proof that 9 friends were carrying pears -/
theorem nine_friends_with_pears :
  friends_with_pears 15 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_friends_with_pears_l1128_112864


namespace NUMINAMATH_CALUDE_max_n_value_l1128_112872

theorem max_n_value (n : ℤ) (h1 : 101 * n^2 ≤ 2525) (h2 : n ≤ 5) : n ≤ 5 ∧ ∃ m : ℤ, m = 5 ∧ 101 * m^2 ≤ 2525 := by
  sorry

end NUMINAMATH_CALUDE_max_n_value_l1128_112872


namespace NUMINAMATH_CALUDE_chameleon_count_denis_chameleons_l1128_112817

theorem chameleon_count : ℕ → Prop :=
  fun total : ℕ =>
    ∃ (initial_brown : ℕ),
      let initial_red : ℕ := 5 * initial_brown
      let final_brown : ℕ := initial_brown - 2
      let final_red : ℕ := initial_red + 2
      (final_red = 8 * final_brown) ∧
      (total = initial_brown + initial_red) ∧
      (total = 36)

theorem denis_chameleons : chameleon_count 36 := by
  sorry

end NUMINAMATH_CALUDE_chameleon_count_denis_chameleons_l1128_112817


namespace NUMINAMATH_CALUDE_impossibleColoring_l1128_112853

def Color := Bool

def isRed (c : Color) : Prop := c = true
def isBlue (c : Color) : Prop := c = false

theorem impossibleColoring :
  ¬∃(f : ℕ → Color),
    (∀ n : ℕ, n > 1000 → (isRed (f n) ∨ isBlue (f n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m ≠ n → isRed (f m) → isRed (f n) → isBlue (f (m * n))) ∧
    (∀ m n : ℕ, m > 1000 → n > 1000 → m = n + 1 → ¬(isBlue (f m) ∧ isBlue (f n))) :=
by
  sorry

end NUMINAMATH_CALUDE_impossibleColoring_l1128_112853


namespace NUMINAMATH_CALUDE_system_solution_transformation_l1128_112819

theorem system_solution_transformation (x y : ℝ) : 
  (2 * x + 3 * y = 19 ∧ 3 * x + 4 * y = 26) → 
  (2 * (2 * x + 4) + 3 * (y + 3) = 19 ∧ 3 * (2 * x + 4) + 4 * (y + 3) = 26) → 
  (x = 2 ∧ y = 5) → 
  (x = -1 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_transformation_l1128_112819


namespace NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l1128_112838

/-- A police emergency number is a positive integer that ends with 133 in decimal representation. -/
def is_police_emergency_number (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_has_large_prime_divisor (n : ℕ) :
  is_police_emergency_number n → ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_police_emergency_number_has_large_prime_divisor_l1128_112838


namespace NUMINAMATH_CALUDE_offspring_ratio_l1128_112863

-- Define the traits
inductive Cotyledon
| Green
| Brown

inductive SeedShape
| Round
| Kidney

-- Define the genotypes
structure Genotype where
  cotyledon : Bool  -- True for Y, False for y
  seedShape : Bool  -- True for R, False for r

-- Define the phenotypes
def phenotype (g : Genotype) : Cotyledon × SeedShape :=
  (if g.cotyledon then Cotyledon.Green else Cotyledon.Brown,
   if g.seedShape then SeedShape.Round else SeedShape.Kidney)

-- Define the inheritance rule
def inherit (parent1 parent2 : Genotype) : Genotype :=
  { cotyledon := parent1.cotyledon || parent2.cotyledon
  , seedShape := parent1.seedShape || parent2.seedShape }

-- Define the parental combinations
def parent1 : Genotype := { cotyledon := true, seedShape := true }   -- YyRr
def parent2 : Genotype := { cotyledon := false, seedShape := false } -- yyrr
def parent3 : Genotype := { cotyledon := true, seedShape := false }  -- Yyrr
def parent4 : Genotype := { cotyledon := false, seedShape := true }  -- yyRr

-- Theorem statement
theorem offspring_ratio 
  (independent_inheritance : ∀ p1 p2, inherit p1 p2 = inherit p2 p1) :
  (∃ (f : Genotype → Genotype → Fin 4), 
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Round → f parent1 parent2 = 0) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Kidney → f parent1 parent2 = 1) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Round → f parent1 parent2 = 2) ∧
    (∀ g, (g = inherit parent1 parent2 ∨ g = inherit parent2 parent1) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Kidney → f parent1 parent2 = 3)) ∧
  (∃ (f : Genotype → Genotype → Fin 4), 
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Round → f parent3 parent4 = 0) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Green ∧ (phenotype g).2 = SeedShape.Kidney → f parent3 parent4 = 1) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Round → f parent3 parent4 = 2) ∧
    (∀ g, (g = inherit parent3 parent4 ∨ g = inherit parent4 parent3) → 
      (phenotype g).1 = Cotyledon.Brown ∧ (phenotype g).2 = SeedShape.Kidney → f parent3 parent4 = 3)) :=
by sorry

end NUMINAMATH_CALUDE_offspring_ratio_l1128_112863


namespace NUMINAMATH_CALUDE_arianna_position_l1128_112845

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance between Ethan and Arianna when Ethan finished, in meters -/
def distance_between : ℝ := 816

/-- Arianna's distance from the start line when Ethan finished -/
def arianna_distance : ℝ := race_length - distance_between

theorem arianna_position : arianna_distance = 184 := by
  sorry

end NUMINAMATH_CALUDE_arianna_position_l1128_112845


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1128_112899

theorem point_not_on_graph : ¬(2 / (2 + 2) = 2 / 3) := by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1128_112899


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1128_112803

/-- The positive difference between the two largest prime factors of 159137 is 14 -/
theorem largest_prime_factors_difference (n : Nat) : n = 159137 → 
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p ∣ n ∧ q ∣ n ∧ 
  (∀ (r : Nat), Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 14 := by
sorry

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1128_112803


namespace NUMINAMATH_CALUDE_prime_power_plus_one_mod_240_l1128_112843

theorem prime_power_plus_one_mod_240 (n : ℕ+) (h : Nat.Prime (2^n.val + 1)) :
  (2^n.val + 1) % 240 = 17 ∨ (2^n.val + 1) % 240 = 3 ∨ (2^n.val + 1) % 240 = 5 :=
by sorry

end NUMINAMATH_CALUDE_prime_power_plus_one_mod_240_l1128_112843


namespace NUMINAMATH_CALUDE_cow_count_is_fifteen_l1128_112890

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The main theorem stating that the number of cows is 15 -/
theorem cow_count_is_fifteen :
  ∃ (ac : AnimalCount), totalLegs ac = 2 * totalHeads ac + 30 ∧ ac.cows = 15 := by
  sorry

#check cow_count_is_fifteen

end NUMINAMATH_CALUDE_cow_count_is_fifteen_l1128_112890


namespace NUMINAMATH_CALUDE_pizza_order_count_l1128_112807

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 2) (h2 : total_slices = 28) :
  total_slices / slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_count_l1128_112807


namespace NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l1128_112873

/-- The area of a sector with a central angle of 120° and a radius of √3 is π. -/
theorem sector_area_120_deg_sqrt3_radius (π : Real) : 
  let central_angle : Real := 120 * π / 180  -- Convert 120° to radians
  let radius : Real := Real.sqrt 3
  let sector_area : Real := (1/2) * radius^2 * central_angle
  sector_area = π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_120_deg_sqrt3_radius_l1128_112873


namespace NUMINAMATH_CALUDE_max_volume_pyramid_l1128_112836

/-- A triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure TriangularPrism where
  volume : ℝ
  AA₁ : ℝ
  BB₁ : ℝ
  CC₁ : ℝ

/-- Points M, N, K on edges AA₁, BB₁, CC₁ respectively -/
structure PrismPoints (prism : TriangularPrism) where
  M : ℝ
  N : ℝ
  K : ℝ
  h_M : M ≤ prism.AA₁
  h_N : N ≤ prism.BB₁
  h_K : K ≤ prism.CC₁

/-- Theorem stating the maximum volume of pyramid MNKP -/
theorem max_volume_pyramid (prism : TriangularPrism) (points : PrismPoints prism) :
  prism.volume = 35 →
  points.M / prism.AA₁ = 5 / 6 →
  points.N / prism.BB₁ = 6 / 7 →
  points.K / prism.CC₁ = 2 / 3 →
  (∃ (P : ℝ), (P ≥ 0 ∧ P ≤ prism.AA₁) ∨ (P ≥ 0 ∧ P ≤ prism.BB₁) ∨ (P ≥ 0 ∧ P ≤ prism.CC₁)) →
  ∃ (pyramid_volume : ℝ), pyramid_volume ≤ 10 ∧ 
    ∀ (other_volume : ℝ), other_volume ≤ pyramid_volume := by
  sorry

end NUMINAMATH_CALUDE_max_volume_pyramid_l1128_112836


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1128_112875

theorem sin_cos_difference_equals_half : 
  Real.sin (36 * π / 180) * Real.cos (6 * π / 180) - 
  Real.sin (54 * π / 180) * Real.cos (84 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1128_112875


namespace NUMINAMATH_CALUDE_factors_imply_absolute_value_l1128_112874

def polynomial (h k : ℝ) (x : ℝ) : ℝ := 3 * x^4 - h * x^2 + k

theorem factors_imply_absolute_value (h k : ℝ) :
  (∀ x : ℝ, (x + 1 = 0 ∨ x - 2 = 0 ∨ x + 3 = 0) → polynomial h k x = 0) →
  |3 * h - 4 * k| = 3 := by
  sorry

end NUMINAMATH_CALUDE_factors_imply_absolute_value_l1128_112874


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l1128_112804

/-- The sum of the first n terms of an arithmetic progression -/
def arithmeticSum (a d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a + (n - 1) * d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms
    is three times the sum of the first 8 terms, the ratio of the first term
    to the common difference is 7:3 -/
theorem arithmetic_progression_ratio (a d : ℚ) :
  arithmeticSum a d 15 = 3 * arithmeticSum a d 8 → a / d = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l1128_112804


namespace NUMINAMATH_CALUDE_prove_equation_l1128_112882

/-- Given that (x + y) / 3 = 1.888888888888889 and 2x + y = 7, prove that x + y = 5.666666666666667 
    is the equation that, when combined with 2x + y = 7, gives the correct value for (x + y) / 3. -/
theorem prove_equation (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.888888888888889)
  (h2 : 2 * x + y = 7) :
  x + y = 5.666666666666667 := by
sorry

end NUMINAMATH_CALUDE_prove_equation_l1128_112882


namespace NUMINAMATH_CALUDE_certain_number_exists_l1128_112851

theorem certain_number_exists : ∃ x : ℝ, 
  3500 - (1000 / x) = 3451.2195121951218 ∧ 
  abs (x - 20.5) < 0.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1128_112851


namespace NUMINAMATH_CALUDE_five_digit_numbers_count_correct_l1128_112879

/-- Counts five-digit numbers with specific digit conditions -/
def count_five_digit_numbers : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let all_identical := 9
  let two_different := 1215
  let three_different := 16200
  let four_different := 45360
  let five_different := 27216
  (all_identical, two_different, three_different, four_different, five_different)

/-- The first digit of a five-digit number cannot be zero -/
axiom first_digit_nonzero : ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 → n / 10000 ≠ 0

/-- The sum of all cases equals the total number of five-digit numbers -/
theorem five_digit_numbers_count_correct :
  let (a, b, c, d, e) := count_five_digit_numbers
  a + b + c + d + e = 90000 :=
sorry

end NUMINAMATH_CALUDE_five_digit_numbers_count_correct_l1128_112879


namespace NUMINAMATH_CALUDE_problem_solution_l1128_112823

theorem problem_solution (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * a - (1/3 : ℝ)
  let B := -a^2 + (1/2 : ℝ) * a * b + (2/3 : ℝ)
  (a + 1)^2 + |b + 2| = 0 → 4 * A - (3 * A - 2 * B) = 11 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1128_112823


namespace NUMINAMATH_CALUDE_video_game_spending_ratio_l1128_112846

theorem video_game_spending_ratio (initial_amount : ℚ) (video_game_cost : ℚ) (remaining : ℚ) :
  initial_amount = 100 →
  remaining = initial_amount - video_game_cost - (1/5) * (initial_amount - video_game_cost) →
  remaining = 60 →
  video_game_cost / initial_amount = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spending_ratio_l1128_112846


namespace NUMINAMATH_CALUDE_space_diagonals_count_l1128_112855

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Definition of a space diagonal in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem stating the number of space diagonals in the given polyhedron -/
theorem space_diagonals_count (Q : ConvexPolyhedron) 
  (h1 : Q.vertices = 30)
  (h2 : Q.edges = 58)
  (h3 : Q.faces = 36)
  (h4 : Q.triangular_faces = 26)
  (h5 : Q.quadrilateral_faces = 10)
  (h6 : Q.triangular_faces + Q.quadrilateral_faces = Q.faces) :
  space_diagonals Q = 357 := by
  sorry


end NUMINAMATH_CALUDE_space_diagonals_count_l1128_112855


namespace NUMINAMATH_CALUDE_daves_monday_hours_l1128_112868

/-- 
Given:
- Dave's hourly rate is $6
- Dave worked on Monday and Tuesday
- On Tuesday, Dave worked 2 hours
- Dave made $48 in total for both days

Prove: Dave worked 6 hours on Monday
-/
theorem daves_monday_hours 
  (hourly_rate : ℕ) 
  (tuesday_hours : ℕ) 
  (total_earnings : ℕ) 
  (h1 : hourly_rate = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_earnings = 48) : 
  ∃ (monday_hours : ℕ), 
    hourly_rate * (monday_hours + tuesday_hours) = total_earnings ∧ 
    monday_hours = 6 := by
  sorry

#check daves_monday_hours

end NUMINAMATH_CALUDE_daves_monday_hours_l1128_112868


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l1128_112885

/-- Given two integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u,v)
    B is the reflection of A across y = x
    C is the reflection of B across y = -x
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 615, then u + v = 45. -/
theorem pentagon_area_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v) : 
  let A := (u, v)
  let B := (v, u)
  let C := (-u, v)
  let D := (-u, -v)
  let E := (u, -v)
  let area := u^2 + 3*u*v
  area = 615 → u + v = 45 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l1128_112885


namespace NUMINAMATH_CALUDE_arrangement_of_six_objects_l1128_112805

theorem arrangement_of_six_objects (n : ℕ) (h : n = 6) : 
  Nat.factorial n = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_arrangement_of_six_objects_l1128_112805


namespace NUMINAMATH_CALUDE_complex_power_sum_l1128_112850

theorem complex_power_sum (z : ℂ) (h : z = -(1 - Complex.I) / Real.sqrt 2) : 
  z^100 + z^50 + 1 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1128_112850


namespace NUMINAMATH_CALUDE_cos_75_degrees_l1128_112871

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l1128_112871


namespace NUMINAMATH_CALUDE_storeroom_items_proof_l1128_112897

/-- Calculates the number of items in the storeroom given the number of restocked items,
    sold items, and total items left in the store. -/
def items_in_storeroom (restocked : ℕ) (sold : ℕ) (total_left : ℕ) : ℕ :=
  total_left - (restocked - sold)

/-- Proves that the number of items in the storeroom is 575 given the specific conditions. -/
theorem storeroom_items_proof :
  items_in_storeroom 4458 1561 3472 = 575 := by
  sorry

#eval items_in_storeroom 4458 1561 3472

end NUMINAMATH_CALUDE_storeroom_items_proof_l1128_112897


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_two_minus_two_l1128_112893

theorem absolute_value_sqrt_two_minus_two :
  (1 : ℝ) < Real.sqrt 2 ∧ Real.sqrt 2 < 2 →
  |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_two_minus_two_l1128_112893


namespace NUMINAMATH_CALUDE_quiz_max_percentage_l1128_112858

/-- Represents the maximum percentage of points a single student can earn in a quiz -/
def max_percentage (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (k : ℚ) / (n : ℚ) * p

/-- Theorem stating the maximum percentage of points a single student can earn in the given quiz scenario -/
theorem quiz_max_percentage : 
  max_percentage 100 66 (1/2) = 1/4 := by sorry

end NUMINAMATH_CALUDE_quiz_max_percentage_l1128_112858


namespace NUMINAMATH_CALUDE_spider_sock_shoe_arrangements_l1128_112856

/-- The number of legs the spider has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_legs)

/-- Theorem stating the number of valid arrangements for the spider to put on socks and shoes -/
theorem spider_sock_shoe_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_legs) :=
sorry

end NUMINAMATH_CALUDE_spider_sock_shoe_arrangements_l1128_112856


namespace NUMINAMATH_CALUDE_island_length_l1128_112860

/-- Represents the dimensions of a rectangular island. -/
structure IslandDimensions where
  width : ℝ
  length : ℝ
  perimeter : ℝ

/-- Theorem: Given a rectangular island with width 4 miles and perimeter 22 miles, its length is 7 miles. -/
theorem island_length (island : IslandDimensions) 
    (h_width : island.width = 4)
    (h_perimeter : island.perimeter = 22)
    (h_rectangle : island.perimeter = 2 * (island.length + island.width)) :
  island.length = 7 := by
  sorry


end NUMINAMATH_CALUDE_island_length_l1128_112860


namespace NUMINAMATH_CALUDE_min_green_surface_fraction_l1128_112802

/-- Represents a cube with given edge length -/
structure Cube where
  edge : ℕ
  deriving Repr

/-- Represents the composition of a large cube -/
structure CubeComposition where
  large_cube : Cube
  small_cube : Cube
  blue_count : ℕ
  green_count : ℕ
  deriving Repr

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge^2

/-- Calculates the volume of a cube -/
def volume (c : Cube) : ℕ := c.edge^3

/-- Theorem: Minimum green surface area fraction -/
theorem min_green_surface_fraction (cc : CubeComposition) 
  (h1 : cc.large_cube.edge = 4)
  (h2 : cc.small_cube.edge = 1)
  (h3 : volume cc.large_cube = cc.blue_count + cc.green_count)
  (h4 : cc.blue_count = 50)
  (h5 : cc.green_count = 14) :
  (6 : ℚ) / surface_area cc.large_cube = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_min_green_surface_fraction_l1128_112802


namespace NUMINAMATH_CALUDE_original_number_not_800_l1128_112841

theorem original_number_not_800 : ¬(∃ x : ℝ, x * 10 = x + 720 ∧ x = 800) := by
  sorry

end NUMINAMATH_CALUDE_original_number_not_800_l1128_112841


namespace NUMINAMATH_CALUDE_divisible_by_91_l1128_112854

theorem divisible_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n+2) + 10^(2*n+1) = 91 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_91_l1128_112854


namespace NUMINAMATH_CALUDE_not_divisible_by_61_l1128_112859

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_61_l1128_112859


namespace NUMINAMATH_CALUDE_unique_solution_system_l1128_112801

theorem unique_solution_system (x y : ℚ) : 
  (3 * x + 2 * y = 7 ∧ 6 * x - 5 * y = 4) ↔ (x = 43/27 ∧ y = 10/9) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1128_112801


namespace NUMINAMATH_CALUDE_mean_score_all_students_l1128_112820

/-- The mean score of all students given specific class conditions --/
theorem mean_score_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (class_ratio : ℚ)
  (additional_group_score : ℝ)
  (additional_group_ratio : ℚ)
  (h1 : morning_mean = 85)
  (h2 : afternoon_mean = 72)
  (h3 : class_ratio = 4/5)
  (h4 : additional_group_score = 68)
  (h5 : additional_group_ratio = 1/4) :
  ∃ (total_mean : ℝ), total_mean = 87 ∧
    total_mean = (morning_mean * class_ratio + 
                  afternoon_mean * (1 - additional_group_ratio) +
                  additional_group_score * additional_group_ratio) /
                 (class_ratio + 1) := by
  sorry

end NUMINAMATH_CALUDE_mean_score_all_students_l1128_112820


namespace NUMINAMATH_CALUDE_floor_product_equals_twenty_l1128_112866

theorem floor_product_equals_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < (21/4) :=
sorry

end NUMINAMATH_CALUDE_floor_product_equals_twenty_l1128_112866


namespace NUMINAMATH_CALUDE_homework_time_ratio_l1128_112862

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Represents the ratio of time spent on two subjects -/
structure TimeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of time spent on geography to history -/
def calculateRatio (time : HomeworkTime) : TimeRatio :=
  { numerator := time.geography, denominator := time.history }

theorem homework_time_ratio (time : HomeworkTime) :
  time.biology = 20 →
  time.history = 2 * time.biology →
  time.geography > time.history →
  time.geography > time.biology →
  time.biology + time.history + time.geography = 180 →
  calculateRatio time = { numerator := 3, denominator := 1 } := by
  sorry

#check homework_time_ratio

end NUMINAMATH_CALUDE_homework_time_ratio_l1128_112862


namespace NUMINAMATH_CALUDE_april_earnings_l1128_112848

/-- Calculates the total money earned from selling flowers -/
def total_money_earned (rose_price tulip_price daisy_price : ℕ) 
                       (roses_sold tulips_sold daisies_sold : ℕ) : ℕ :=
  rose_price * roses_sold + tulip_price * tulips_sold + daisy_price * daisies_sold

/-- Proves that April earned $78 from selling flowers -/
theorem april_earnings : 
  total_money_earned 4 3 2 9 6 12 = 78 := by sorry

end NUMINAMATH_CALUDE_april_earnings_l1128_112848


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1128_112870

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = Real.sqrt (3 * x + 4)
def q (x : ℝ) : Prop := x^2 = 3 * x + 4

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1128_112870


namespace NUMINAMATH_CALUDE_systematic_sample_valid_l1128_112815

/-- Checks if a list of integers forms a valid systematic sample -/
def is_valid_systematic_sample (population_size : ℕ) (sample_size : ℕ) (sample : List ℕ) : Prop :=
  let interval := population_size / sample_size
  sample.length = sample_size ∧
  ∀ i j, i < j → j < sample.length →
    sample[j]! - sample[i]! = (j - i) * interval

theorem systematic_sample_valid :
  let population_size := 50
  let sample_size := 5
  let sample := [3, 13, 23, 33, 43]
  is_valid_systematic_sample population_size sample_size sample := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_valid_l1128_112815


namespace NUMINAMATH_CALUDE_point_coordinates_l1128_112800

/-- A point in a plane rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of a plane rectangular coordinate system -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : fourth_quadrant p) 
  (h2 : distance_to_x_axis p = 2) 
  (h3 : distance_to_y_axis p = 3) : 
  p = Point.mk 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1128_112800


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1128_112833

/-- Enumeration of sampling methods -/
inductive SamplingMethod
  | LotteryMethod
  | StratifiedSampling
  | RandomNumberMethod
  | SystematicSampling

/-- Scenario with total population and sample size -/
structure Scenario where
  total_population : ℕ
  sample_size : ℕ
  has_strata : Bool

/-- Function to determine the correct sampling method based on scenario -/
def correct_sampling_method (s : Scenario) : SamplingMethod :=
  if s.has_strata then
    SamplingMethod.StratifiedSampling
  else if s.total_population ≤ 30 then
    SamplingMethod.LotteryMethod
  else if s.sample_size ≤ 10 then
    SamplingMethod.RandomNumberMethod
  else
    SamplingMethod.SystematicSampling

/-- Theorem stating the correct sampling methods for given scenarios -/
theorem correct_sampling_methods :
  (correct_sampling_method ⟨30, 10, false⟩ = SamplingMethod.LotteryMethod) ∧
  (correct_sampling_method ⟨30, 10, true⟩ = SamplingMethod.StratifiedSampling) ∧
  (correct_sampling_method ⟨300, 10, false⟩ = SamplingMethod.RandomNumberMethod) ∧
  (correct_sampling_method ⟨300, 50, false⟩ = SamplingMethod.SystematicSampling) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1128_112833


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_verify_solution_l1128_112844

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given a SwimmerSpeed and a direction. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 10 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed)
  (h_downstream : effectiveSpeed s true * 3 = 45)
  (h_upstream : effectiveSpeed s false * 3 = 15) : 
  s.swimmer = 10 := by
  sorry

/-- Verifies that the solution satisfies the given conditions. -/
theorem verify_solution : 
  let s : SwimmerSpeed := ⟨10, 5⟩
  effectiveSpeed s true * 3 = 45 ∧ 
  effectiveSpeed s false * 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_verify_solution_l1128_112844


namespace NUMINAMATH_CALUDE_range_of_a_l1128_112880

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1 / (x - 2) ≥ 1 → |x - a| < 1) ∧ 
   ∃ x, (|x - a| < 1 ∧ 1 / (x - 2) < 1)) →
  a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1128_112880


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l1128_112818

/-- Given three collinear points A(-1, 1), B(2, -4), and C(x, -9), prove that x = 5 -/
theorem collinear_points_x_value : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -4)
  let C : ℝ × ℝ := (x, -9)
  (∀ t : ℝ, (1 - t) * A.1 + t * B.1 = C.1 ∧ (1 - t) * A.2 + t * B.2 = C.2) →
  x = 5 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l1128_112818
