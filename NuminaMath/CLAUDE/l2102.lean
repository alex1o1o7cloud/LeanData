import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l2102_210235

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a - 1)) - (a / (a - 1)) = a :=
sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h : x ≠ -1) :
  (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l2102_210235


namespace NUMINAMATH_CALUDE_square_equality_l2102_210287

theorem square_equality (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l2102_210287


namespace NUMINAMATH_CALUDE_corner_sum_9x9_board_l2102_210204

/- Define the size of the checkerboard -/
def boardSize : Nat := 9

/- Define the total number of squares -/
def totalSquares : Nat := boardSize * boardSize

/- Define the positions of the corner and adjacent numbers -/
def cornerPositions : List Nat := [1, 2, 8, 9, 73, 74, 80, 81]

/- Theorem statement -/
theorem corner_sum_9x9_board :
  (List.sum cornerPositions) = 328 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_9x9_board_l2102_210204


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l2102_210254

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) →
  (3 * q ^ 2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l2102_210254


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l2102_210208

/-- An isosceles triangle with side lengths 3 and 7 has a base of length 3. -/
theorem isosceles_triangle_base (a b : ℝ) (h1 : a = 3 ∨ a = 7) (h2 : b = 3 ∨ b = 7) (h3 : a ≠ b) :
  ∃ (x y : ℝ), x = y ∧ x + y > b ∧ x = 7 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l2102_210208


namespace NUMINAMATH_CALUDE_copy_pages_proof_l2102_210294

/-- The cost in cents to copy a single page -/
def cost_per_page : ℚ := 25/10

/-- The amount of money available in dollars -/
def available_money : ℚ := 20

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of pages that can be copied with the available money -/
def pages_copied : ℕ := 800

theorem copy_pages_proof : 
  (available_money * cents_per_dollar) / cost_per_page = pages_copied := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_proof_l2102_210294


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_equality_holds_largest_n_is_six_l2102_210231

theorem largest_binomial_coefficient (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem equality_holds : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_equality_holds_largest_n_is_six_l2102_210231


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2102_210262

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2102_210262


namespace NUMINAMATH_CALUDE_construction_rearrangements_l2102_210225

def word : String := "CONSTRUCTION"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  vowels.length.factorial

def consonant_arrangements : ℕ :=
  consonants.length.factorial / ((consonants.countP (· = 'C')).factorial *
                                 (consonants.countP (· = 'T')).factorial *
                                 (consonants.countP (· = 'N')).factorial)

theorem construction_rearrangements :
  vowel_arrangements * consonant_arrangements = 30240 := by
  sorry

end NUMINAMATH_CALUDE_construction_rearrangements_l2102_210225


namespace NUMINAMATH_CALUDE_prob_red_or_green_l2102_210224

/-- The probability of drawing a red or green marble from a bag -/
theorem prob_red_or_green (red green yellow : ℕ) (h : red = 4 ∧ green = 3 ∧ yellow = 6) :
  (red + green : ℚ) / (red + green + yellow) = 7 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_or_green_l2102_210224


namespace NUMINAMATH_CALUDE_state_A_selection_percentage_l2102_210207

theorem state_A_selection_percentage : 
  ∀ (total_candidates : ℕ) (state_B_percentage : ℚ) (extra_selected : ℕ),
    total_candidates = 8000 →
    state_B_percentage = 7 / 100 →
    extra_selected = 80 →
    ∃ (state_A_percentage : ℚ),
      state_A_percentage * total_candidates + extra_selected = state_B_percentage * total_candidates ∧
      state_A_percentage = 6 / 100 := by
  sorry

end NUMINAMATH_CALUDE_state_A_selection_percentage_l2102_210207


namespace NUMINAMATH_CALUDE_triangle_height_inequality_l2102_210255

/-- Given a triangle ABC with sides a, b, c and heights h_a, h_b, h_c, 
    the sum of squares of heights divided by squares of sides is at most 9/2. -/
theorem triangle_height_inequality (a b c h_a h_b h_c : ℝ) 
    (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_pos_ha : h_a > 0) (h_pos_hb : h_b > 0) (h_pos_hc : h_c > 0)
    (h_triangle : a * h_a = b * h_b ∧ b * h_b = c * h_c) : 
    (h_b^2 + h_c^2) / a^2 + (h_c^2 + h_a^2) / b^2 + (h_a^2 + h_b^2) / c^2 ≤ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_inequality_l2102_210255


namespace NUMINAMATH_CALUDE_count_distinct_n_values_l2102_210256

/-- Given a quadratic equation x² - nx + 36 = 0 with integer roots,
    there are exactly 10 distinct possible values for n. -/
theorem count_distinct_n_values : ∃ (S : Finset ℤ),
  (∀ n ∈ S, ∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) ∧
  (∀ n : ℤ, (∃ x₁ x₂ : ℤ, x₁ * x₂ = 36 ∧ x₁ + x₂ = n) → n ∈ S) ∧
  Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_count_distinct_n_values_l2102_210256


namespace NUMINAMATH_CALUDE_floor_sqrt_33_squared_l2102_210214

theorem floor_sqrt_33_squared : ⌊Real.sqrt 33⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_33_squared_l2102_210214


namespace NUMINAMATH_CALUDE_annual_growth_rate_is_30_percent_l2102_210218

-- Define the initial number of users and the number after 2 years
def initial_users : ℝ := 1000000
def users_after_2_years : ℝ := 1690000

-- Define the time period
def years : ℝ := 2

-- Define the growth rate as a function
def growth_rate (x : ℝ) : Prop :=
  initial_users * (1 + x)^years = users_after_2_years

-- Theorem statement
theorem annual_growth_rate_is_30_percent :
  ∃ (x : ℝ), x > 0 ∧ growth_rate x ∧ x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_is_30_percent_l2102_210218


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l2102_210293

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 6 > Real.sqrt 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l2102_210293


namespace NUMINAMATH_CALUDE_max_sum_cubes_l2102_210205

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l2102_210205


namespace NUMINAMATH_CALUDE_percentage_problem_l2102_210277

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.8 * N = 240) 
  (h2 : (P / 100) * N = 60) : 
  P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2102_210277


namespace NUMINAMATH_CALUDE_rectangle_area_l2102_210279

/-- A rectangle with one side of length 4 and a diagonal of length 5 has an area of 12. -/
theorem rectangle_area (w l d : ℝ) (hw : w = 4) (hd : d = 5) (h_pythagorean : w^2 + l^2 = d^2) : w * l = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2102_210279


namespace NUMINAMATH_CALUDE_exactly_three_valid_sequences_l2102_210282

/-- An arithmetic sequence with the given properties -/
structure ValidSequence where
  a₁ : ℕ
  d : ℕ
  h_a₁_single_digit : a₁ < 10
  h_100_in_seq : ∃ n : ℕ, a₁ + (n - 1) * d = 100
  h_3103_in_seq : ∃ m : ℕ, a₁ + (m - 1) * d = 3103
  h_max_terms : ∀ k : ℕ, a₁ + (k - 1) * d ≤ 3103 → k ≤ 240

/-- The set of all valid sequences -/
def validSequences : Set ValidSequence := {s | s.a₁ + 239 * s.d ≥ 3103}

theorem exactly_three_valid_sequences :
  ∃! (s₁ s₂ s₃ : ValidSequence),
    validSequences = {s₁, s₂, s₃} ∧
    s₁.a₁ = 9 ∧ s₁.d = 13 ∧
    s₂.a₁ = 1 ∧ s₂.d = 33 ∧
    s₃.a₁ = 9 ∧ s₃.d = 91 :=
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sequences_l2102_210282


namespace NUMINAMATH_CALUDE_joans_initial_balloons_count_l2102_210221

/-- The number of blue balloons Joan had initially -/
def joans_initial_balloons : ℕ := 9

/-- The number of balloons Sally popped -/
def popped_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessicas_balloons : ℕ := 2

/-- The total number of blue balloons they have now -/
def total_balloons_now : ℕ := 6

theorem joans_initial_balloons_count : 
  joans_initial_balloons = popped_balloons + (total_balloons_now - jessicas_balloons) :=
by sorry

end NUMINAMATH_CALUDE_joans_initial_balloons_count_l2102_210221


namespace NUMINAMATH_CALUDE_model_height_is_58_l2102_210275

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Empire State Building in feet -/
def actual_height : ℕ := 1454

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The height of the scale model rounded to the nearest foot -/
def model_height : ℕ := (round_to_nearest ((actual_height : ℚ) / scale_ratio)).natAbs

theorem model_height_is_58 : model_height = 58 := by sorry

end NUMINAMATH_CALUDE_model_height_is_58_l2102_210275


namespace NUMINAMATH_CALUDE_calculation_proof_l2102_210227

theorem calculation_proof : (30 / (8 + 2 - 5)) * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2102_210227


namespace NUMINAMATH_CALUDE_expected_red_balls_l2102_210253

/-- Given a bag of balls with some red and some white, prove that the expected
    number of red balls is proportional to the number of red draws in a series
    of random draws with replacement. -/
theorem expected_red_balls
  (total_balls : ℕ)
  (total_draws : ℕ)
  (red_draws : ℕ)
  (h_total_balls : total_balls = 12)
  (h_total_draws : total_draws = 200)
  (h_red_draws : red_draws = 50) :
  (total_balls : ℚ) * (red_draws : ℚ) / (total_draws : ℚ) = 3 :=
sorry

end NUMINAMATH_CALUDE_expected_red_balls_l2102_210253


namespace NUMINAMATH_CALUDE_greatest_divisor_630_under_60_and_factor_90_l2102_210295

def is_greatest_divisor (n : ℕ) : Prop :=
  n ∣ 630 ∧ n < 60 ∧ n ∣ 90 ∧
  ∀ m : ℕ, m ∣ 630 → m < 60 → m ∣ 90 → m ≤ n

theorem greatest_divisor_630_under_60_and_factor_90 :
  is_greatest_divisor 45 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_630_under_60_and_factor_90_l2102_210295


namespace NUMINAMATH_CALUDE_ferry_travel_time_l2102_210212

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 2

/-- The speed of Ferry P in kilometers per hour -/
def speed_p : ℝ := 8

/-- The speed of Ferry Q in kilometers per hour -/
def speed_q : ℝ := speed_p + 4

/-- The distance traveled by Ferry P in kilometers -/
def distance_p : ℝ := speed_p * t

/-- The distance traveled by Ferry Q in kilometers -/
def distance_q : ℝ := 3 * distance_p

/-- The travel time of Ferry Q in hours -/
def time_q : ℝ := t + 2

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧
  t = 2 := by sorry

end NUMINAMATH_CALUDE_ferry_travel_time_l2102_210212


namespace NUMINAMATH_CALUDE_range_of_a_when_P_or_Q_false_l2102_210265

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Define proposition P
def P (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- Define the set of a values where either P or Q is false
def A : Set ℝ := {a | -1 < a ∧ a < 0 ∨ 0 < a ∧ a < 1}

-- Theorem statement
theorem range_of_a_when_P_or_Q_false :
  ∀ a : ℝ, (¬P a ∨ ¬Q a) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_P_or_Q_false_l2102_210265


namespace NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2102_210210

theorem consecutive_even_integers_cube_sum : 
  ∀ a : ℕ, 
    a > 0 → 
    (2*a - 2) * (2*a) * (2*a + 2) = 12 * (6*a) → 
    (2*a - 2)^3 + (2*a)^3 + (2*a + 2)^3 = 8568 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_cube_sum_l2102_210210


namespace NUMINAMATH_CALUDE_calculation_equality_algebraic_simplification_l2102_210267

-- Part 1
theorem calculation_equality : (-(1/3))⁻¹ + (2015 - Real.sqrt 3)^0 - 4 * Real.sin (60 * π / 180) + |(- Real.sqrt 12)| = -2 := by sorry

-- Part 2
theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  ((1 / (a + b) - 1 / (a - b)) / (b / (a^2 - 2*a*b + b^2))) = -2*(a - b)/(a + b) := by sorry

end NUMINAMATH_CALUDE_calculation_equality_algebraic_simplification_l2102_210267


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l2102_210283

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ 
    (n.choose 3) * x^(n - 3) * a^3 = 210 * k ∧
    (n.choose 4) * x^(n - 4) * a^4 = 420 * k ∧
    (n.choose 5) * x^(n - 5) * a^5 = 630 * k) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l2102_210283


namespace NUMINAMATH_CALUDE_june_election_win_l2102_210238

/-- The minimum percentage of boys required for June to win the election -/
def min_boys_percentage : ℝ :=
  -- We'll define this later in the proof
  sorry

theorem june_election_win (total_students : ℕ) (boys_vote_percentage : ℝ) (girls_vote_percentage : ℝ) 
  (h_total : total_students = 200)
  (h_boys_vote : boys_vote_percentage = 67.5)
  (h_girls_vote : girls_vote_percentage = 25)
  (h_win_threshold : ∀ x : ℝ, x > 50 → x ≥ (total_students : ℝ) / 2 + 0.5) :
  ∃ ε > 0, abs (min_boys_percentage - 60) < ε ∧ 
  ∀ boys_percentage : ℝ, boys_percentage ≥ min_boys_percentage →
    (boys_percentage * boys_vote_percentage + (100 - boys_percentage) * girls_vote_percentage) / 100 > 50 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l2102_210238


namespace NUMINAMATH_CALUDE_quadratic_equation_real_root_l2102_210292

theorem quadratic_equation_real_root (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_root_l2102_210292


namespace NUMINAMATH_CALUDE_round_robin_tournament_l2102_210239

theorem round_robin_tournament (x : ℕ) : x > 0 → (x * (x - 1)) / 2 = 15 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_l2102_210239


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2102_210206

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ x^2 - 4*a*x + 3*a^2 = 0 ∧ y^2 - 4*a*y + 3*a^2 = 0) →
  (1/3 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2102_210206


namespace NUMINAMATH_CALUDE_jimmy_cookies_count_l2102_210232

/-- Given that:
  - Crackers contain 15 calories each
  - Cookies contain 50 calories each
  - Jimmy eats 10 crackers
  - Jimmy consumes a total of 500 calories
  Prove that Jimmy eats 7 cookies -/
theorem jimmy_cookies_count :
  let cracker_calories : ℕ := 15
  let cookie_calories : ℕ := 50
  let crackers_eaten : ℕ := 10
  let total_calories : ℕ := 500
  let cookies_eaten : ℕ := (total_calories - cracker_calories * crackers_eaten) / cookie_calories
  cookies_eaten = 7 :=
by sorry

end NUMINAMATH_CALUDE_jimmy_cookies_count_l2102_210232


namespace NUMINAMATH_CALUDE_f_max_min_l2102_210220

-- Define the function f(x) = 2x² - x⁴
def f (x : ℝ) : ℝ := 2 * x^2 - x^4

-- Theorem statement
theorem f_max_min :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ 1) ∧
  (∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l2102_210220


namespace NUMINAMATH_CALUDE_divisibility_condition_l2102_210249

theorem divisibility_condition (M : ℕ) : 
  0 < M ∧ M < 10 → (5 ∣ 1989^M + M^1889 ↔ M = 1 ∨ M = 4) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2102_210249


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2102_210243

/-- The complex number z = i / (1 - i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2102_210243


namespace NUMINAMATH_CALUDE_equal_trout_division_l2102_210201

theorem equal_trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → total_trout / num_people = trout_per_person → trout_per_person = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_trout_division_l2102_210201


namespace NUMINAMATH_CALUDE_range_of_x_l2102_210237

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2102_210237


namespace NUMINAMATH_CALUDE_gcd_16_12_l2102_210271

def operation : List (ℕ × ℕ) := [(16, 12), (12, 4), (8, 4), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_l2102_210271


namespace NUMINAMATH_CALUDE_rowing_problem_l2102_210288

/-- A rowing problem in a river with current and headwind -/
theorem rowing_problem (downstream_speed current_speed headwind_reduction : ℝ) 
  (h1 : downstream_speed = 22)
  (h2 : current_speed = 4.5)
  (h3 : headwind_reduction = 1.5) :
  let still_water_speed := downstream_speed - current_speed
  still_water_speed - current_speed - headwind_reduction = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_rowing_problem_l2102_210288


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l2102_210274

/-- The total shaded area on a square carpet -/
theorem carpet_shaded_area (S T : ℝ) : 
  S > 0 ∧ T > 0 ∧ (12 : ℝ) / S = 4 ∧ S / T = 2 →
  S^2 + 4 * T^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l2102_210274


namespace NUMINAMATH_CALUDE_root_product_expression_l2102_210263

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - p*α + 2 = 0) → 
  (β^2 - p*β + 2 = 0) → 
  (γ^2 + q*γ - 2 = 0) → 
  (δ^2 + q*δ - 2 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = -2*(p-q)^2 - 4*p*q + 4*q^2 + 16 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l2102_210263


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2102_210245

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (x y : ℝ) :
  (3 * x - 6 * y = 12) → 
  (∃ (m b : ℝ), y = m * x + b ∧ m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2102_210245


namespace NUMINAMATH_CALUDE_average_study_time_difference_l2102_210251

/-- The differences in study times (Mia - Liam) for each day of the week --/
def study_time_differences : List Int := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week --/
def days_in_week : Nat := 7

/-- Theorem: The average difference in study time per day is 7 minutes --/
theorem average_study_time_difference :
  (study_time_differences.sum : ℚ) / days_in_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l2102_210251


namespace NUMINAMATH_CALUDE_integer_difference_l2102_210286

theorem integer_difference (x y : ℕ+) : 
  x > y → x + y = 5 → x^3 - y^3 = 63 → x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l2102_210286


namespace NUMINAMATH_CALUDE_bank_teller_coins_l2102_210247

theorem bank_teller_coins (rolls_per_teller : ℕ) (coins_per_roll : ℕ) (num_tellers : ℕ) :
  rolls_per_teller = 10 →
  coins_per_roll = 25 →
  num_tellers = 4 →
  rolls_per_teller * coins_per_roll * num_tellers = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_teller_coins_l2102_210247


namespace NUMINAMATH_CALUDE_prob_at_least_twice_eq_target_l2102_210296

/-- The probability of hitting a target in one shot -/
def p : ℝ := 0.6

/-- The number of shots taken -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots -/
def prob_at_least_twice (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * p^2 * (1 - p) + (n.choose 3) * p^3

theorem prob_at_least_twice_eq_target : 
  prob_at_least_twice p n = 0.648 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_twice_eq_target_l2102_210296


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2102_210264

/-- Two angles are supplementary if their measures sum to 180 degrees -/
def Supplementary (a b : ℝ) : Prop := a + b = 180

theorem angle_measure_proof (A B : ℝ) 
  (h1 : Supplementary A B) 
  (h2 : A = 8 * B) : 
  A = 160 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2102_210264


namespace NUMINAMATH_CALUDE_max_volume_l2102_210281

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- AB is perpendicular to BC and CD
  ab_perp_bc : True
  ab_perp_cd : True
  -- Length of BC is 2
  bc_length : ℝ
  bc_eq_two : bc_length = 2
  -- Dihedral angle between AB and CD is 60°
  dihedral_angle : ℝ
  dihedral_angle_eq_sixty : dihedral_angle = 60
  -- Circumradius is √5
  circumradius : ℝ
  circumradius_eq_sqrt_five : circumradius = Real.sqrt 5

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- The maximum possible volume of the tetrahedron -/
theorem max_volume (t : Tetrahedron) : volume t ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_l2102_210281


namespace NUMINAMATH_CALUDE_extra_fruit_calculation_l2102_210203

theorem extra_fruit_calculation (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : students = 9) : 
  red_apples + green_apples - students = 40 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruit_calculation_l2102_210203


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l2102_210230

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  games : Fin num_teams → Fin num_teams → Bool
  
/-- Tournament conditions --/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 7 ∧
  (∀ i j, i ≠ j → (t.games i j ↔ ¬t.games j i)) ∧
  (∀ i, ¬t.games i i)

/-- Number of wins for a team --/
def wins (t : Tournament) (team : Fin t.num_teams) : Nat :=
  (Finset.univ.filter (λ j => t.games team j)).card

/-- Maximum number of wins in the tournament --/
def max_wins (t : Tournament) : Nat :=
  Finset.univ.sup (λ team => wins t team)

/-- Number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : Nat :=
  (Finset.univ.filter (λ team => wins t team = max_wins t)).card

/-- The main theorem --/
theorem max_teams_tied_for_most_wins (t : Tournament) 
  (h : valid_tournament t) : 
  num_teams_with_max_wins t ≤ 6 ∧ 
  ∃ t' : Tournament, valid_tournament t' ∧ num_teams_with_max_wins t' = 6 := by
  sorry


end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l2102_210230


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l2102_210242

def g (x : ℝ) := 10 * x^4 - 16 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l2102_210242


namespace NUMINAMATH_CALUDE_inverse_sum_of_cube_function_l2102_210233

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_of_cube_function :
  g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_of_cube_function_l2102_210233


namespace NUMINAMATH_CALUDE_lea_purchases_cost_l2102_210223

/-- The cost of a single book -/
def book_cost : ℕ := 16

/-- The cost of a single binder -/
def binder_cost : ℕ := 2

/-- The number of binders bought -/
def num_binders : ℕ := 3

/-- The cost of a single notebook -/
def notebook_cost : ℕ := 1

/-- The number of notebooks bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := book_cost + (binder_cost * num_binders) + (notebook_cost * num_notebooks)

theorem lea_purchases_cost : total_cost = 28 := by
  sorry

end NUMINAMATH_CALUDE_lea_purchases_cost_l2102_210223


namespace NUMINAMATH_CALUDE_eating_contest_l2102_210259

/-- Eating contest problem -/
theorem eating_contest (hotdog_weight burger_weight pie_weight : ℕ)
  (jacob_pies noah_burgers mason_hotdogs : ℕ)
  (h1 : hotdog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : jacob_pies + 3 = noah_burgers)
  (h5 : mason_hotdogs = 3 * jacob_pies)
  (h6 : mason_hotdogs * hotdog_weight = 30) :
  noah_burgers = 8 := by
  sorry

end NUMINAMATH_CALUDE_eating_contest_l2102_210259


namespace NUMINAMATH_CALUDE_matrix_power_2023_l2102_210268

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 1; 0, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 2023; 0, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l2102_210268


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2102_210270

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = -(a + 2) / (a - 2) ∧
  (-(1 + 2) / (1 - 2) = 3) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2102_210270


namespace NUMINAMATH_CALUDE_frank_has_two_ten_dollar_bills_l2102_210211

-- Define the problem parameters
def one_dollar_bills : ℕ := 7
def five_dollar_bills : ℕ := 4
def twenty_dollar_bill : ℕ := 1
def peanut_price_per_pound : ℕ := 3
def change : ℕ := 4
def daily_peanut_consumption : ℕ := 3
def days_in_week : ℕ := 7

-- Define the function to calculate the number of ten-dollar bills
def calculate_ten_dollar_bills : ℕ := 
  let total_without_tens : ℕ := one_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bill
  let total_peanuts_bought : ℕ := daily_peanut_consumption * days_in_week
  let total_spent : ℕ := peanut_price_per_pound * total_peanuts_bought
  let amount_from_tens : ℕ := total_spent - total_without_tens + change
  amount_from_tens / 10

-- Theorem stating that Frank has exactly 2 ten-dollar bills
theorem frank_has_two_ten_dollar_bills : calculate_ten_dollar_bills = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_has_two_ten_dollar_bills_l2102_210211


namespace NUMINAMATH_CALUDE_initial_bees_count_l2102_210269

/-- Given a hive where 8 bees fly in and the total becomes 24, prove that there were initially 16 bees. -/
theorem initial_bees_count (initial_bees : ℕ) : initial_bees + 8 = 24 → initial_bees = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_bees_count_l2102_210269


namespace NUMINAMATH_CALUDE_expedition_max_distance_l2102_210298

/-- Represents the state of the expedition --/
structure ExpeditionState where
  participants : Nat
  distance : Nat
  fuel_per_car : Nat

/-- Calculates the maximum distance the expedition can travel --/
def max_distance (initial_state : ExpeditionState) : Nat :=
  sorry

/-- Theorem stating the maximum distance the expedition can travel --/
theorem expedition_max_distance :
  let initial_state : ExpeditionState := {
    participants := 9,
    distance := 0,
    fuel_per_car := 10  -- 1 gallon in tank + 9 additional cans
  }
  max_distance initial_state = 360 := by
  sorry

end NUMINAMATH_CALUDE_expedition_max_distance_l2102_210298


namespace NUMINAMATH_CALUDE_correct_distribution_l2102_210244

/-- Represents the jellybean distribution problem --/
structure JellybeanDistribution where
  total_jellybeans : ℕ
  num_nephews : ℕ
  num_nieces : ℕ
  nephew_ratio : ℕ
  niece_ratio : ℕ

/-- Calculates the maximum number of jellybeans each nephew and niece can receive --/
def max_distribution (jd : JellybeanDistribution) : ℕ × ℕ :=
  let total_parts := jd.num_nephews * jd.nephew_ratio + jd.num_nieces * jd.niece_ratio
  let jellybeans_per_part := jd.total_jellybeans / total_parts
  (jellybeans_per_part * jd.nephew_ratio, jellybeans_per_part * jd.niece_ratio)

/-- Theorem stating the correct distribution for the given problem --/
theorem correct_distribution (jd : JellybeanDistribution) 
  (h1 : jd.total_jellybeans = 537)
  (h2 : jd.num_nephews = 4)
  (h3 : jd.num_nieces = 3)
  (h4 : jd.nephew_ratio = 2)
  (h5 : jd.niece_ratio = 1) :
  max_distribution jd = (96, 48) ∧ 
  96 * jd.num_nephews + 48 * jd.num_nieces ≤ jd.total_jellybeans :=
by
  sorry

#eval max_distribution {
  total_jellybeans := 537,
  num_nephews := 4,
  num_nieces := 3,
  nephew_ratio := 2,
  niece_ratio := 1
}

end NUMINAMATH_CALUDE_correct_distribution_l2102_210244


namespace NUMINAMATH_CALUDE_collinearity_condition_for_linear_combination_l2102_210213

/-- Given points O, A, B are not collinear, and vector OP = m * vector OA + n * vector OB,
    points A, P, B are collinear if and only if m + n = 1 -/
theorem collinearity_condition_for_linear_combination
  (O A B P : EuclideanSpace ℝ (Fin 3))
  (m n : ℝ)
  (h_not_collinear : ¬ Collinear ℝ {O, A, B})
  (h_linear_combination : P - O = m • (A - O) + n • (B - O)) :
  Collinear ℝ {A, P, B} ↔ m + n = 1 := by sorry

end NUMINAMATH_CALUDE_collinearity_condition_for_linear_combination_l2102_210213


namespace NUMINAMATH_CALUDE_age_difference_l2102_210276

theorem age_difference (A B : ℕ) : B = 39 → A + 10 = 2 * (B - 10) → A - B = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2102_210276


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l2102_210222

noncomputable def point_to_line_distance (θ : ℝ) : ℝ :=
  |3 * Real.cos θ + 4 * Real.sin θ - 4| / 5

theorem max_distance_point_to_line :
  ∃ (θ : ℝ), ∀ (φ : ℝ), point_to_line_distance θ ≥ point_to_line_distance φ ∧
  point_to_line_distance θ = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_point_to_line_l2102_210222


namespace NUMINAMATH_CALUDE_most_cost_effective_option_l2102_210248

/-- Represents the cost calculation for tea sets and bowls under different offers -/
def cost_calculation (tea_set_price : ℕ) (tea_bowl_price : ℕ) (num_sets : ℕ) (num_bowls : ℕ) : ℕ → ℕ
| 1 => tea_set_price * num_sets + tea_bowl_price * (num_bowls - num_sets)  -- Offer 1
| 2 => (tea_set_price * num_sets * 95 + tea_bowl_price * num_bowls * 95) / 100  -- Offer 2
| _ => 0  -- Invalid offer

theorem most_cost_effective_option 
  (tea_set_price : ℕ) 
  (tea_bowl_price : ℕ) 
  (num_sets : ℕ) 
  (num_bowls : ℕ) 
  (h1 : tea_set_price = 200)
  (h2 : tea_bowl_price = 20)
  (h3 : num_sets = 30)
  (h4 : num_bowls = 40)
  (h5 : num_bowls > num_sets) :
  let offer1_cost := cost_calculation tea_set_price tea_bowl_price num_sets num_bowls 1
  let offer2_cost := cost_calculation tea_set_price tea_bowl_price num_sets num_bowls 2
  let combined_offer_cost := tea_set_price * num_sets + 
                             (cost_calculation tea_set_price tea_bowl_price (num_bowls - num_sets) (num_bowls - num_sets) 2)
  combined_offer_cost < min offer1_cost offer2_cost ∧ combined_offer_cost = 6190 :=
by sorry

end NUMINAMATH_CALUDE_most_cost_effective_option_l2102_210248


namespace NUMINAMATH_CALUDE_delta_value_l2102_210209

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2102_210209


namespace NUMINAMATH_CALUDE_art_performance_probability_l2102_210299

def artDepartment : Finset Nat := {1, 2, 3, 4}
def firstGrade : Finset Nat := {1, 2}
def secondGrade : Finset Nat := {3, 4}

theorem art_performance_probability :
  let totalSelections := Finset.powerset artDepartment |>.filter (λ s => s.card = 2)
  let differentGradeSelections := totalSelections.filter (λ s => s ∩ firstGrade ≠ ∅ ∧ s ∩ secondGrade ≠ ∅)
  (differentGradeSelections.card : ℚ) / totalSelections.card = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_art_performance_probability_l2102_210299


namespace NUMINAMATH_CALUDE_distance_PQ_is_2_25_l2102_210216

/-- The distance between two points on a ruler -/
def distance_on_ruler (p q : ℚ) : ℚ := q - p

/-- The position of point P on the ruler -/
def P : ℚ := 1/2

/-- The position of point Q on the ruler -/
def Q : ℚ := 2 + 3/4

theorem distance_PQ_is_2_25 : distance_on_ruler P Q = 2.25 := by sorry

end NUMINAMATH_CALUDE_distance_PQ_is_2_25_l2102_210216


namespace NUMINAMATH_CALUDE_power_inequality_l2102_210234

/-- Proof of inequality involving powers -/
theorem power_inequality (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 5) :
  a^n + b^n + c^n ≥ a^(n-5) * b^3 * c^2 + b^(n-5) * c^3 * a^2 + c^(n-5) * a^3 * b^2 := by
  sorry

#check power_inequality

end NUMINAMATH_CALUDE_power_inequality_l2102_210234


namespace NUMINAMATH_CALUDE_total_score_is_38_l2102_210202

/-- Represents the scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the game and the total score calculation. -/
def game_result (s : Scores) : Prop :=
  s.marius = s.darius + 3 ∧
  s.matt = s.darius + 5 ∧
  s.darius = 10 ∧
  s.darius + s.matt + s.marius = 38

/-- Theorem stating that under the given conditions, the total score is 38. -/
theorem total_score_is_38 : ∃ s : Scores, game_result s :=
  sorry

end NUMINAMATH_CALUDE_total_score_is_38_l2102_210202


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l2102_210246

theorem friend_lunch_cost (total : ℕ) (difference : ℕ) (friend_cost : ℕ) : 
  total = 19 →
  difference = 3 →
  friend_cost = total / 2 + difference →
  friend_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l2102_210246


namespace NUMINAMATH_CALUDE_min_distance_squared_l2102_210280

noncomputable def e : ℝ := Real.exp 1

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : b = a - 2 * e^a) 
  (h2 : c + d = 4) : 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2102_210280


namespace NUMINAMATH_CALUDE_mary_initial_weight_l2102_210226

/-- Mary's weight changes and final weight --/
structure WeightChanges where
  initial_loss : ℕ
  final_weight : ℕ

/-- Calculate Mary's initial weight given her weight changes --/
def calculate_initial_weight (changes : WeightChanges) : ℕ :=
  changes.final_weight         -- Start with final weight
  + changes.initial_loss * 3   -- Add back the triple loss
  - changes.initial_loss * 2   -- Subtract the double gain
  - 6                          -- Subtract the final gain
  + changes.initial_loss       -- Add back the initial loss

/-- Theorem stating that Mary's initial weight was 99 pounds --/
theorem mary_initial_weight :
  let changes : WeightChanges := { initial_loss := 12, final_weight := 81 }
  calculate_initial_weight changes = 99 := by
  sorry


end NUMINAMATH_CALUDE_mary_initial_weight_l2102_210226


namespace NUMINAMATH_CALUDE_equation_solution_l2102_210250

theorem equation_solution : ∃ x : ℝ, (2 / (x + 5) = 1 / (3 * x)) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2102_210250


namespace NUMINAMATH_CALUDE_rubber_bands_total_l2102_210219

theorem rubber_bands_total (harper_bands : ℕ) (brother_difference : ℕ) : 
  harper_bands = 15 → 
  brother_difference = 6 → 
  harper_bands + (harper_bands - brother_difference) = 24 := by
sorry

end NUMINAMATH_CALUDE_rubber_bands_total_l2102_210219


namespace NUMINAMATH_CALUDE_polynomial_value_at_3_l2102_210200

-- Define a monic polynomial of degree 4
def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_value_at_3 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_4 p) 
  (h1 : p 1 = 1) 
  (h2 : p (-1) = -1) 
  (h3 : p 2 = 2) 
  (h4 : p (-2) = -2) : 
  p 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_3_l2102_210200


namespace NUMINAMATH_CALUDE_ways_to_buy_three_items_eq_646_l2102_210261

/-- Represents the inventory of a store --/
structure Inventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy three items (headphones, keyboard, mouse) --/
def ways_to_buy_three_items (inv : Inventory) : Nat :=
  inv.keyboard_mouse_sets * inv.headphones +
  inv.headphone_mouse_sets * inv.keyboards +
  inv.headphones * inv.mice * inv.keyboards

/-- The theorem stating the number of ways to buy three items --/
theorem ways_to_buy_three_items_eq_646 (inv : Inventory) 
  (h1 : inv.headphones = 9)
  (h2 : inv.mice = 13)
  (h3 : inv.keyboards = 5)
  (h4 : inv.keyboard_mouse_sets = 4)
  (h5 : inv.headphone_mouse_sets = 5) :
  ways_to_buy_three_items inv = 646 := by
  sorry


end NUMINAMATH_CALUDE_ways_to_buy_three_items_eq_646_l2102_210261


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l2102_210289

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating the mass of the man in the given problem. -/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 8
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01  -- 1 cm in meters
  let water_density : ℝ := 1000     -- kg/m³
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 240 := by
  sorry


end NUMINAMATH_CALUDE_man_mass_on_boat_l2102_210289


namespace NUMINAMATH_CALUDE_train_length_l2102_210285

theorem train_length (pole_time : ℝ) (tunnel_length tunnel_time : ℝ) :
  pole_time = 20 →
  tunnel_length = 500 →
  tunnel_time = 40 →
  ∃ (train_length : ℝ),
    train_length = pole_time * (train_length + tunnel_length) / tunnel_time ∧
    train_length = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2102_210285


namespace NUMINAMATH_CALUDE_connor_score_l2102_210278

theorem connor_score (connor amy jason : ℕ) : 
  (amy = connor + 4) →
  (jason = 2 * amy) →
  (connor + amy + jason = 20) →
  connor = 2 := by sorry

end NUMINAMATH_CALUDE_connor_score_l2102_210278


namespace NUMINAMATH_CALUDE_fencing_length_l2102_210228

theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 8 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 178 := by
  sorry

end NUMINAMATH_CALUDE_fencing_length_l2102_210228


namespace NUMINAMATH_CALUDE_village_male_population_l2102_210217

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_parts : ℕ
  male_parts : ℕ

/-- Calculates the number of males in the village -/
def num_males (v : Village) : ℕ :=
  v.total_population * v.male_parts / v.num_parts

theorem village_male_population (v : Village) 
  (h1 : v.total_population = 600)
  (h2 : v.num_parts = 4)
  (h3 : v.male_parts = 2) : 
  num_males v = 300 := by
  sorry

#check village_male_population

end NUMINAMATH_CALUDE_village_male_population_l2102_210217


namespace NUMINAMATH_CALUDE_distance_between_points_l2102_210258

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (10, 8)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2102_210258


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l2102_210273

/-- Calculates the new concentration of alcohol in a mixture --/
theorem alcohol_concentration_in_mixture
  (vessel1_capacity : ℝ)
  (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_concentration : ℝ)
  (total_liquid : ℝ)
  (new_vessel_capacity : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_concentration = 0.4)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_concentration = 0.6)
  (h5 : total_liquid = 8)
  (h6 : new_vessel_capacity = 10)
  (h7 : total_liquid ≤ new_vessel_capacity) :
  let alcohol1 := vessel1_capacity * vessel1_concentration
  let alcohol2 := vessel2_capacity * vessel2_concentration
  let total_alcohol := alcohol1 + alcohol2
  let water_added := new_vessel_capacity - total_liquid
  let new_concentration := total_alcohol / new_vessel_capacity
  new_concentration = 0.44 := by
  sorry

#check alcohol_concentration_in_mixture

end NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l2102_210273


namespace NUMINAMATH_CALUDE_sum_product_inequality_l2102_210252

theorem sum_product_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l2102_210252


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2102_210297

theorem quadratic_factorization (x : ℝ) : 9 * x^2 - 6 * x + 1 = (3 * x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2102_210297


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2102_210240

/-- A type representing the colors of segments -/
inductive Color
| Red
| Blue

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function type representing the coloring of segments -/
def Coloring := (Point × Point) → Color

/-- Theorem: Given 6 points in a plane with all segments colored either red or blue,
    there exists a triangle whose sides are all the same color -/
theorem monochromatic_triangle_exists (points : Fin 6 → Point) (coloring : Coloring) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (coloring (points i, points j) = coloring (points j, points k) ∧
     coloring (points j, points k) = coloring (points k, points i)) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l2102_210240


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2102_210241

/-- For a quadratic equation x^2 + 2(k-1)x + k^2 - 1 = 0, 
    the equation has real roots if and only if k ≤ 1 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(k-1)*x + k^2 - 1 = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2102_210241


namespace NUMINAMATH_CALUDE_y1_gt_y2_l2102_210236

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : k < 0

/-- The function corresponding to the LinearFunctionNotInThirdQuadrant -/
def f (l : LinearFunctionNotInThirdQuadrant) (x : ℝ) : ℝ :=
  l.k * x + l.b

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_gt_y2 (l : LinearFunctionNotInThirdQuadrant) (y₁ y₂ : ℝ)
    (h1 : f l (-1) = y₁)
    (h2 : f l 1 = y₂) :
    y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_l2102_210236


namespace NUMINAMATH_CALUDE_coincidence_time_l2102_210266

-- Define the movement pattern
def move_distance (n : ℕ) : ℤ := if n % 2 = 0 then -n else n

-- Define the position after n moves
def position (n : ℕ) : ℤ := (List.range n).map move_distance |>.sum

-- Define the total distance traveled after n moves
def total_distance (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the speed
def speed : ℕ := 4

-- Define the position of point A
def point_A : ℤ := -24

-- Theorem to prove
theorem coincidence_time :
  ∃ n : ℕ, position n = point_A ∧ (total_distance n / speed : ℚ) = 294 := by
  sorry


end NUMINAMATH_CALUDE_coincidence_time_l2102_210266


namespace NUMINAMATH_CALUDE_volume_ratio_is_one_over_three_root_three_l2102_210257

/-- A right circular cone -/
structure RightCircularCone where
  radius : ℝ
  height : ℝ

/-- A plane cutting the cone -/
structure CuttingPlane where
  tangent_to_base : Bool
  passes_through_midpoint : Bool

/-- The ratio of volumes -/
def volume_ratio (cone : RightCircularCone) (plane : CuttingPlane) : ℝ := 
  sorry

/-- Theorem statement -/
theorem volume_ratio_is_one_over_three_root_three 
  (cone : RightCircularCone) 
  (plane : CuttingPlane) 
  (h1 : plane.tangent_to_base = true) 
  (h2 : plane.passes_through_midpoint = true) : 
  volume_ratio cone plane = 1 / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_one_over_three_root_three_l2102_210257


namespace NUMINAMATH_CALUDE_value_of_expression_l2102_210291

theorem value_of_expression (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2102_210291


namespace NUMINAMATH_CALUDE_dealership_sales_expectation_l2102_210260

/-- The number of trucks expected to be sold -/
def expected_trucks : ℕ := 30

/-- The number of vans expected to be sold -/
def expected_vans : ℕ := 15

/-- The ratio of trucks to SUVs -/
def truck_suv_ratio : ℚ := 3 / 5

/-- The ratio of SUVs to vans -/
def suv_van_ratio : ℚ := 2 / 1

/-- The number of SUVs the dealership should expect to sell -/
def expected_suvs : ℕ := 30

theorem dealership_sales_expectation :
  (expected_trucks : ℚ) / truck_suv_ratio ≥ expected_suvs ∧
  suv_van_ratio * expected_vans = expected_suvs :=
sorry

end NUMINAMATH_CALUDE_dealership_sales_expectation_l2102_210260


namespace NUMINAMATH_CALUDE_uranus_appearance_time_l2102_210229

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def add_minutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Calculates the difference in minutes between two times -/
def minutes_difference (t1 t2 : Time) : ℕ :=
  sorry

theorem uranus_appearance_time 
  (mars_disappearance : Time)
  (jupiter_delay : ℕ)
  (uranus_delay : ℕ)
  (h_mars : mars_disappearance = ⟨0, 10, sorry, sorry⟩)  -- 12:10 AM
  (h_jupiter : jupiter_delay = 2 * 60 + 41)  -- 2 hours and 41 minutes
  (h_uranus : uranus_delay = 3 * 60 + 16)  -- 3 hours and 16 minutes
  : 
  let jupiter_appearance := add_minutes mars_disappearance jupiter_delay
  let uranus_appearance := add_minutes jupiter_appearance uranus_delay
  minutes_difference ⟨6, 0, sorry, sorry⟩ uranus_appearance = 7 :=
sorry

end NUMINAMATH_CALUDE_uranus_appearance_time_l2102_210229


namespace NUMINAMATH_CALUDE_unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l2102_210272

/-- Definition of a "maintaining value" interval for a function f on [a,b] --/
def is_maintaining_value_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The square function --/
def f (x : ℝ) : ℝ := x^2

/-- The square function with constant --/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + m

/-- Theorem: [0,1] is the only "maintaining value" interval for f(x) = x^2 --/
theorem unique_maintaining_value_interval_for_square :
  ∀ a b : ℝ, is_maintaining_value_interval f a b ↔ a = 0 ∧ b = 1 :=
sorry

/-- Theorem: Characterization of "maintaining value" intervals for g(x) = x^2 + m --/
theorem maintaining_value_intervals_for_square_plus_constant :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_maintaining_value_interval (g m) a b) ↔ 
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioc 0 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l2102_210272


namespace NUMINAMATH_CALUDE_parabola_translation_l2102_210284

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- Theorem stating that the translated parabola is correct
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2102_210284


namespace NUMINAMATH_CALUDE_dads_dimes_l2102_210215

theorem dads_dimes (initial : ℕ) (from_mother : ℕ) (total : ℕ) : 
  initial = 7 → from_mother = 4 → total = 19 → 
  total - (initial + from_mother) = 8 := by
sorry

end NUMINAMATH_CALUDE_dads_dimes_l2102_210215


namespace NUMINAMATH_CALUDE_equation_value_l2102_210290

theorem equation_value (x y : ℝ) (h : x^2 - 3*y - 5 = 0) : 2*x^2 - 6*y - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l2102_210290
