import Mathlib

namespace NUMINAMATH_CALUDE_supermarket_spending_l2592_259293

theorem supermarket_spending (total : ℚ) :
  (1/2 : ℚ) * total +
  (1/3 : ℚ) * total +
  (1/10 : ℚ) * total +
  8 = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2592_259293


namespace NUMINAMATH_CALUDE_zoe_spent_30_dollars_l2592_259265

/-- The price of a single flower in dollars -/
def flower_price : ℕ := 3

/-- The number of roses Zoe bought -/
def roses_bought : ℕ := 8

/-- The number of daisies Zoe bought -/
def daisies_bought : ℕ := 2

/-- Theorem: Given the conditions, Zoe spent 30 dollars -/
theorem zoe_spent_30_dollars : 
  (roses_bought + daisies_bought) * flower_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_zoe_spent_30_dollars_l2592_259265


namespace NUMINAMATH_CALUDE_train_platform_passage_time_train_platform_passage_time_specific_l2592_259230

/-- Calculates the time taken for a train to pass a platform given its speed, 
    the platform length, and the time taken to pass a stationary man. -/
theorem train_platform_passage_time 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * time_pass_man
  let total_distance := platform_length + train_length
  let time_pass_platform := total_distance / train_speed_ms
  time_pass_platform

/-- Proves that given the specific conditions, the time taken to pass 
    the platform is approximately 30 seconds. -/
theorem train_platform_passage_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_platform_passage_time 54 150.012 20 - 30| < ε :=
sorry

end NUMINAMATH_CALUDE_train_platform_passage_time_train_platform_passage_time_specific_l2592_259230


namespace NUMINAMATH_CALUDE_happy_point_range_l2592_259275

theorem happy_point_range (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) (-3/2), a * x^2 - 2*x - 2*a - 3/2 = -x) →
  a ∈ Set.Icc (-1/4 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_happy_point_range_l2592_259275


namespace NUMINAMATH_CALUDE_additional_cards_proof_l2592_259294

/-- The number of cards in the original deck -/
def original_deck : ℕ := 52

/-- The number of players -/
def num_players : ℕ := 3

/-- The number of cards each player has after splitting the deck -/
def cards_per_player : ℕ := 18

/-- The number of additional cards added to the deck -/
def additional_cards : ℕ := (num_players * cards_per_player) - original_deck

theorem additional_cards_proof :
  additional_cards = 2 := by sorry

end NUMINAMATH_CALUDE_additional_cards_proof_l2592_259294


namespace NUMINAMATH_CALUDE_fifth_root_unity_sum_l2592_259220

theorem fifth_root_unity_sum (x : ℂ) : x^5 = 1 → 1 + x^4 + x^8 + x^12 + x^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_unity_sum_l2592_259220


namespace NUMINAMATH_CALUDE_five_squared_minus_nine_over_five_minus_three_equals_eight_l2592_259204

theorem five_squared_minus_nine_over_five_minus_three_equals_eight :
  (5^2 - 9) / (5 - 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_squared_minus_nine_over_five_minus_three_equals_eight_l2592_259204


namespace NUMINAMATH_CALUDE_tim_picked_five_pears_l2592_259258

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 6

/-- The total number of pears picked by Sara and Tim -/
def total_pears : ℕ := 11

/-- The number of pears Tim picked -/
def tim_pears : ℕ := total_pears - sara_pears

theorem tim_picked_five_pears : tim_pears = 5 := by
  sorry

end NUMINAMATH_CALUDE_tim_picked_five_pears_l2592_259258


namespace NUMINAMATH_CALUDE_find_a_l2592_259222

/-- The system of equations -/
def system (a b m : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y = 2 ∧ m * x - 7 * y = -8

/-- Xiao Li's solution -/
def solution_li (a b : ℝ) : Prop :=
  a * (-2) + b * 3 = 2

/-- Xiao Zhang's solution -/
def solution_zhang (a b : ℝ) : Prop :=
  a * (-2) + b * 2 = 2

/-- Theorem stating that if both solutions satisfy the first equation, then a = -1 -/
theorem find_a (a b m : ℝ) : solution_li a b ∧ solution_zhang a b → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2592_259222


namespace NUMINAMATH_CALUDE_valid_sequences_l2592_259200

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 8 ∧
  s.count 1 = 2 ∧
  s.count 2 = 2 ∧
  s.count 3 = 2 ∧
  s.count 4 = 2 ∧
  ∃ i j, i + 2 = j ∧ s[i]? = some 1 ∧ s[j]? = some 1 ∧
  ∃ i j, i + 3 = j ∧ s[i]? = some 2 ∧ s[j]? = some 2 ∧
  ∃ i j, i + 4 = j ∧ s[i]? = some 3 ∧ s[j]? = some 3 ∧
  ∃ i j, i + 5 = j ∧ s[i]? = some 4 ∧ s[j]? = some 4

theorem valid_sequences :
  is_valid_sequence [4, 1, 3, 1, 2, 4, 3, 2] ∧
  is_valid_sequence [2, 3, 4, 2, 1, 3, 1, 4] :=
by sorry

end NUMINAMATH_CALUDE_valid_sequences_l2592_259200


namespace NUMINAMATH_CALUDE_power_exceeds_million_l2592_259281

theorem power_exceeds_million : ∃ (n₁ n₂ n₃ : ℕ+),
  (1.01 : ℝ) ^ (n₁ : ℕ) > 1000000 ∧
  (1.001 : ℝ) ^ (n₂ : ℕ) > 1000000 ∧
  (1.000001 : ℝ) ^ (n₃ : ℕ) > 1000000 := by
  sorry

end NUMINAMATH_CALUDE_power_exceeds_million_l2592_259281


namespace NUMINAMATH_CALUDE_equation_solutions_l2592_259254

def solution_set : Set ℝ := {12, 1, -1, -12}

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 3*x - 18) + 1 / (x^2 - 15*x - 12) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2592_259254


namespace NUMINAMATH_CALUDE_no_perfect_square_polynomial_l2592_259250

theorem no_perfect_square_polynomial (n : ℕ) : ∃ (m : ℕ), n^6 + 3*n^5 - 5*n^4 - 15*n^3 + 4*n^2 + 12*n + 3 ≠ m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_polynomial_l2592_259250


namespace NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_term_l2592_259211

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fifth_and_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_roots : a 3 * a 10 = 5 ∧ a 3 + a 10 = 3) :
  a 5 + a 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fifth_and_eighth_term_l2592_259211


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_numbers_l2592_259213

theorem sum_of_four_consecutive_even_numbers : 
  let n : ℕ := 32
  let sum := n + (n + 2) + (n + 4) + (n + 6)
  sum = 140 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_numbers_l2592_259213


namespace NUMINAMATH_CALUDE_largest_non_sum_of_three_distinct_composites_l2592_259262

/-- A number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ¬Nat.Prime n

/-- A function that checks if a natural number can be expressed as the sum of three distinct composite numbers -/
def IsSumOfThreeDistinctComposites (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), IsComposite a ∧ IsComposite b ∧ IsComposite c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = n

/-- The theorem stating that 17 is the largest integer that cannot be expressed as the sum of three distinct composite numbers -/
theorem largest_non_sum_of_three_distinct_composites :
  (∀ n > 17, IsSumOfThreeDistinctComposites n) ∧
  ¬IsSumOfThreeDistinctComposites 17 ∧
  (∀ n < 17, ¬IsSumOfThreeDistinctComposites n → ¬IsSumOfThreeDistinctComposites 17) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_three_distinct_composites_l2592_259262


namespace NUMINAMATH_CALUDE_sin_cos_cube_difference_squared_l2592_259242

theorem sin_cos_cube_difference_squared (θ : Real) 
  (h : Real.sin θ - Real.cos θ = (Real.sqrt 6 - Real.sqrt 2) / 2) : 
  24 * (Real.sin θ ^ 3 - Real.cos θ ^ 3) ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_difference_squared_l2592_259242


namespace NUMINAMATH_CALUDE_num_factors_of_given_number_l2592_259279

/-- The number of distinct, natural-number factors of 4³ * 5⁴ * 6² -/
def num_factors : ℕ := 135

/-- The given number -/
def given_number : ℕ := 4^3 * 5^4 * 6^2

theorem num_factors_of_given_number :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_num_factors_of_given_number_l2592_259279


namespace NUMINAMATH_CALUDE_nine_digit_integers_count_l2592_259214

theorem nine_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 8) = 800000000 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_integers_count_l2592_259214


namespace NUMINAMATH_CALUDE_negation_of_all_nonnegative_squares_l2592_259240

theorem negation_of_all_nonnegative_squares (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_nonnegative_squares_l2592_259240


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_conditions_l2592_259286

theorem sufficient_but_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) :=
by sorry


end NUMINAMATH_CALUDE_sufficient_but_not_necessary_conditions_l2592_259286


namespace NUMINAMATH_CALUDE_quadratic_root_implies_t_value_l2592_259241

theorem quadratic_root_implies_t_value (a t : ℝ) :
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (a + 3 * Complex.I : ℂ) ^ 2 - 4 * (a + 3 * Complex.I : ℂ) + t = 0 →
  t = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_t_value_l2592_259241


namespace NUMINAMATH_CALUDE_membership_ratio_is_three_to_one_l2592_259236

/-- Represents the monthly costs and sign-up fees for two gym memberships --/
structure GymMemberships where
  cheap_monthly : ℚ
  cheap_signup : ℚ
  expensive_signup_months : ℚ
  total_first_year : ℚ

/-- Calculates the ratio of expensive gym's monthly cost to cheap gym's monthly cost --/
def membership_ratio (g : GymMemberships) : ℚ :=
  let cheap_yearly := g.cheap_signup + 12 * g.cheap_monthly
  let expensive_yearly := g.total_first_year - cheap_yearly
  let expensive_monthly := expensive_yearly / (g.expensive_signup_months + 12)
  expensive_monthly / g.cheap_monthly

/-- Theorem stating that the membership ratio is 3:1 for the given conditions --/
theorem membership_ratio_is_three_to_one (g : GymMemberships)
    (h1 : g.cheap_monthly = 10)
    (h2 : g.cheap_signup = 50)
    (h3 : g.expensive_signup_months = 4)
    (h4 : g.total_first_year = 650) :
    membership_ratio g = 3 := by
  sorry

end NUMINAMATH_CALUDE_membership_ratio_is_three_to_one_l2592_259236


namespace NUMINAMATH_CALUDE_max_d_value_l2592_259274

def a (n : ℕ+) : ℕ := 80 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ n : ℕ+, d n = 5) ∧ (∀ n : ℕ+, d n ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2592_259274


namespace NUMINAMATH_CALUDE_square_measurement_error_l2592_259266

theorem square_measurement_error (actual_side : ℝ) (measured_side : ℝ) 
  (h : measured_side ^ 2 = 1.0816 * actual_side ^ 2) : 
  (measured_side - actual_side) / actual_side = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l2592_259266


namespace NUMINAMATH_CALUDE_solution_days_is_forty_l2592_259298

/-- The number of days required to solve all problems given the conditions -/
def solution_days (a b c : ℕ) : ℕ :=
  let total_problems := 5 * (11 * a + 7 * b + 9 * c)
  40

/-- The theorem stating that the solution_days function returns 40 given the problem conditions -/
theorem solution_days_is_forty (a b c : ℕ) :
  (5 * (11 * a + 7 * b + 9 * c) = 16 * (4 * a + 2 * b + 3 * c)) →
  solution_days a b c = 40 := by
  sorry

#check solution_days_is_forty

end NUMINAMATH_CALUDE_solution_days_is_forty_l2592_259298


namespace NUMINAMATH_CALUDE_equation_solutions_l2592_259277

theorem equation_solutions (a b : ℝ) (h : a + b = 0) :
  (∃! x : ℝ, a * x + b = 0) ∨ (∀ x : ℝ, a * x + b = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2592_259277


namespace NUMINAMATH_CALUDE_sum_of_perimeters_l2592_259217

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * x + 4 * y = 4 * (Real.sqrt 63 + Real.sqrt 22) := by
sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_l2592_259217


namespace NUMINAMATH_CALUDE_fraction_transformation_impossibility_l2592_259219

theorem fraction_transformation_impossibility : ¬ ∃ (f : ℚ), (
  f = 5/8 ∧
  (∀ (n : ℕ), f = (f.num + n) / (f.den + n) ∨ f = (f.num * n) / (f.den * n)) ∧
  f = 3/5
) := by sorry

end NUMINAMATH_CALUDE_fraction_transformation_impossibility_l2592_259219


namespace NUMINAMATH_CALUDE_min_sum_squares_with_real_solution_l2592_259201

theorem min_sum_squares_with_real_solution (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_with_real_solution_l2592_259201


namespace NUMINAMATH_CALUDE_workshop_workers_l2592_259224

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (non_technician_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 8000) 
  (h2 : technician_salary = 10000) (h3 : non_technician_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 14 ∧ 
  (num_technicians : ℝ) * technician_salary + 
  ((total_workers - num_technicians) : ℝ) * non_technician_salary = 
  (total_workers : ℝ) * average_salary :=
sorry

end NUMINAMATH_CALUDE_workshop_workers_l2592_259224


namespace NUMINAMATH_CALUDE_expression_evaluation_l2592_259208

theorem expression_evaluation :
  let a : ℤ := -1
  (a^2 + 1) - 3*a*(a - 1) + 2*(a^2 + a - 1) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2592_259208


namespace NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l2592_259263

theorem triangle_angle_expression_minimum (A B C : Real) 
  (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) : 
  (1 / (Real.sin A)^2) + (1 / (Real.sin B)^2) + (4 / (1 + Real.sin C)) ≥ 16 - 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_expression_minimum_l2592_259263


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2592_259273

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 0 = 2048) ∧ 
  (a 1 = 512) ∧ 
  (a 2 = 128) ∧ 
  (a 5 = 2) ∧ 
  ∀ n, a (n + 1) = a n * (a 1 / a 0)

/-- The sum of the fourth and fifth terms in the sequence is 40 -/
theorem sum_of_fourth_and_fifth_terms (a : ℕ → ℚ) 
  (h : geometric_sequence a) : a 3 + a 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l2592_259273


namespace NUMINAMATH_CALUDE_custom_mult_identity_value_l2592_259226

/-- Custom multiplication operation -/
def custom_mult (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mult_identity_value (a b c : ℝ) :
  (custom_mult a b c 1 2 = 4) →
  (custom_mult a b c 2 3 = 6) →
  (∃ m : ℝ, m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x) →
  ∃ m : ℝ, m = 13 ∧ m ≠ 0 ∧ ∀ x : ℝ, custom_mult a b c x m = x :=
by sorry

end NUMINAMATH_CALUDE_custom_mult_identity_value_l2592_259226


namespace NUMINAMATH_CALUDE_product_local_abs_value_l2592_259228

/-- The local value of a digit in a number -/
def localValue (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def absValue (n : ℕ) : ℕ := n

/-- The given number -/
def givenNumber : ℕ := 564823

/-- The digit we're focusing on -/
def focusDigit : ℕ := 4

/-- The position of the focus digit (0-indexed from right) -/
def digitPosition : ℕ := 4

theorem product_local_abs_value : 
  localValue givenNumber focusDigit digitPosition * absValue focusDigit = 160000 := by
  sorry

end NUMINAMATH_CALUDE_product_local_abs_value_l2592_259228


namespace NUMINAMATH_CALUDE_students_exceed_rabbits_l2592_259268

theorem students_exceed_rabbits :
  let classrooms : ℕ := 5
  let students_per_classroom : ℕ := 23
  let rabbits_per_classroom : ℕ := 3
  let total_students : ℕ := classrooms * students_per_classroom
  let total_rabbits : ℕ := classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
sorry

end NUMINAMATH_CALUDE_students_exceed_rabbits_l2592_259268


namespace NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l2592_259272

/-- Given that some typists can type 38 letters in 20 minutes and 30 typists working at the same rate can complete 171 letters in 1 hour, prove that the number of typists in the initial group is 20. -/
theorem initial_typists_count : ℕ :=
  let initial_letters : ℕ := 38
  let initial_time : ℕ := 20
  let second_typists : ℕ := 30
  let second_letters : ℕ := 171
  let second_time : ℕ := 60
  20

/-- Proof of the theorem -/
theorem initial_typists_count_proof : initial_typists_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_initial_typists_count_initial_typists_count_proof_l2592_259272


namespace NUMINAMATH_CALUDE_calculation_proof_l2592_259296

theorem calculation_proof :
  ((-4)^2 * ((-3/4) + (-5/8)) = -22) ∧
  (-2^2 - (1 - 0.5) * (1/3) * (2 - (-4)^2) = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2592_259296


namespace NUMINAMATH_CALUDE_round_robin_tournament_games_l2592_259243

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of combinations of n things taken k at a time -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem round_robin_tournament_games :
  num_games 6 = binom 6 2 := by sorry

end NUMINAMATH_CALUDE_round_robin_tournament_games_l2592_259243


namespace NUMINAMATH_CALUDE_complement_of_M_in_S_l2592_259271

def S : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_S :
  S \ M = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_S_l2592_259271


namespace NUMINAMATH_CALUDE_prime_factors_count_l2592_259290

/-- The total number of prime factors in the given expression -/
def total_prime_factors : ℕ :=
  (2 * 17) + (2 * 13) + (3 * 7) + (5 * 3) + (7 * 19)

/-- The theorem stating that the total number of prime factors in the given expression is 229 -/
theorem prime_factors_count :
  total_prime_factors = 229 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l2592_259290


namespace NUMINAMATH_CALUDE_third_roll_greater_probability_l2592_259292

def roll_count : ℕ := 3
def sides : ℕ := 8

def favorable_outcomes (sides : ℕ) : ℕ := 
  (sides - 1) * (sides - 1) + (sides - 1)

theorem third_roll_greater_probability (sides : ℕ) (h : sides > 0) :
  (favorable_outcomes sides : ℚ) / (sides ^ roll_count) = 7 / 64 :=
sorry

end NUMINAMATH_CALUDE_third_roll_greater_probability_l2592_259292


namespace NUMINAMATH_CALUDE_sin_cos_power_12_range_l2592_259225

theorem sin_cos_power_12_range (x : ℝ) : 
  (1 : ℝ) / 32 ≤ Real.sin x ^ 12 + Real.cos x ^ 12 ∧ 
  Real.sin x ^ 12 + Real.cos x ^ 12 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_power_12_range_l2592_259225


namespace NUMINAMATH_CALUDE_inradius_value_l2592_259227

/-- Given a triangle with perimeter p and area A, its inradius r satisfies A = r * p / 2 -/
axiom inradius_formula (p A r : ℝ) : A = r * p / 2

/-- The perimeter of the triangle -/
def p : ℝ := 42

/-- The area of the triangle -/
def A : ℝ := 105

/-- The inradius of the triangle -/
def r : ℝ := 5

theorem inradius_value : r = 5 := by sorry

end NUMINAMATH_CALUDE_inradius_value_l2592_259227


namespace NUMINAMATH_CALUDE_initial_rate_is_36_l2592_259270

/-- Represents the production of cogs on an assembly line with two phases -/
def cog_production (initial_rate : ℝ) : Prop :=
  let initial_order := 60
  let second_order := 60
  let increased_rate := 60
  let total_cogs := initial_order + second_order
  let initial_time := initial_order / initial_rate
  let second_time := second_order / increased_rate
  let total_time := initial_time + second_time
  let average_output := 45
  (total_cogs / total_time) = average_output

/-- The theorem stating that the initial production rate is 36 cogs per hour -/
theorem initial_rate_is_36 : 
  ∃ (rate : ℝ), cog_production rate ∧ rate = 36 :=
sorry

end NUMINAMATH_CALUDE_initial_rate_is_36_l2592_259270


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l2592_259251

theorem gold_coin_distribution (x y : ℕ) (h1 : x > y) (h2 : x + y = 49) :
  ∃ (k : ℕ), x^2 - y^2 = k * (x - y) → k = 49 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_distribution_l2592_259251


namespace NUMINAMATH_CALUDE_triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l2592_259231

def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

theorem triangular_number_gcd_bound (n : ℕ+) : 
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

theorem triangular_number_gcd_achieves_three : 
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l2592_259231


namespace NUMINAMATH_CALUDE_circles_intersect_l2592_259289

/-- Definition of Circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- Definition of Circle O₂ -/
def circle_O₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- Center of Circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 1)

/-- Center of Circle O₂ -/
def center_O₂ : ℝ × ℝ := (1, 2)

/-- Radius of Circle O₁ -/
def radius_O₁ : ℝ := 1

/-- Radius of Circle O₂ -/
def radius_O₂ : ℝ := 2

/-- Theorem: Circles O₁ and O₂ are intersecting -/
theorem circles_intersect : 
  (radius_O₁ + radius_O₂ > Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2)) ∧
  (Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2) > |radius_O₂ - radius_O₁|) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_l2592_259289


namespace NUMINAMATH_CALUDE_inequality_solution_l2592_259202

theorem inequality_solution (x : ℝ) :
  x > -1 ∧ x ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔
  (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2592_259202


namespace NUMINAMATH_CALUDE_hyperbola_area_ratio_l2592_259239

noncomputable def hyperbola_ratio (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) : Prop :=
  let x := λ p : ℝ × ℝ => p.1
  let y := λ p : ℝ × ℝ => p.2
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((x p - x q)^2 + (y p - y q)^2)
  let area := λ p q r : ℝ × ℝ => abs ((x q - x p) * (y r - y p) - (x r - x p) * (y q - y p)) / 2
  (∀ p : ℝ × ℝ, (x p)^2 / a^2 - (y p)^2 / b^2 = 1 → 
    (x p - x F₁) * (x p - x F₂) + (y p - y F₁) * (y p - y F₂) = a^2 - b^2) ∧
  (x F₁ = -Real.sqrt (a^2 + b^2) ∧ y F₁ = 0) ∧
  (x F₂ = Real.sqrt (a^2 + b^2) ∧ y F₂ = 0) ∧
  ((x A)^2 / a^2 - (y A)^2 / b^2 = 1) ∧
  ((x B)^2 / a^2 - (y B)^2 / b^2 = 1) ∧
  (y B - y A) * (x A - x F₁) = (x B - x A) * (y A - y F₁) ∧
  dist A F₁ / dist A F₂ = 1/2 →
  area A F₁ F₂ / area A B F₂ = 4/9

theorem hyperbola_area_ratio : 
  hyperbola_ratio 3 4 (-5, 0) (5, 0) (-27/5, 8*Real.sqrt 14/5) (0, 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_area_ratio_l2592_259239


namespace NUMINAMATH_CALUDE_frisbee_price_problem_l2592_259278

theorem frisbee_price_problem (total_frisbees : ℕ) (total_revenue : ℕ) 
  (price_some : ℕ) (min_sold_at_price_some : ℕ) :
  total_frisbees = 64 →
  total_revenue = 200 →
  price_some = 4 →
  min_sold_at_price_some = 8 →
  ∃ (price_others : ℕ), 
    price_others = 3 ∧
    ∃ (num_at_price_some : ℕ),
      num_at_price_some ≥ min_sold_at_price_some ∧
      price_some * num_at_price_some + price_others * (total_frisbees - num_at_price_some) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_frisbee_price_problem_l2592_259278


namespace NUMINAMATH_CALUDE_flagpole_height_l2592_259221

-- Define the given conditions
def flagpoleShadowLength : ℝ := 45
def buildingShadowLength : ℝ := 65
def buildingHeight : ℝ := 26

-- Define the theorem
theorem flagpole_height :
  ∃ (h : ℝ), h / flagpoleShadowLength = buildingHeight / buildingShadowLength ∧ h = 18 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_height_l2592_259221


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2592_259237

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 20) % 15 = 0 ∧ (n - 20) % 30 = 0 ∧ (n - 20) % 45 = 0 ∧ (n - 20) % 60 = 0

theorem smallest_number_proof :
  is_divisible_by_all 200 ∧ ∀ m : ℕ, m < 200 → ¬is_divisible_by_all m :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2592_259237


namespace NUMINAMATH_CALUDE_bushel_weight_is_56_l2592_259234

/-- The weight of a bushel of corn in pounds -/
def bushel_weight : ℝ := 56

/-- The weight of an individual ear of corn in pounds -/
def ear_weight : ℝ := 0.5

/-- The number of bushels Clyde picked -/
def bushels_picked : ℕ := 2

/-- The number of individual corn cobs Clyde picked -/
def cobs_picked : ℕ := 224

/-- Theorem: The weight of a bushel of corn is 56 pounds -/
theorem bushel_weight_is_56 : 
  bushel_weight = (ear_weight * cobs_picked) / bushels_picked :=
sorry

end NUMINAMATH_CALUDE_bushel_weight_is_56_l2592_259234


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2592_259203

theorem largest_divisor_of_difference_of_squares (m n : ℕ) :
  (∃ k l : ℕ, m = 2 * k ∧ n = 2 * l) →  -- m and n are even
  n < m →  -- n is less than m
  (∃ d : ℕ, d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) →
  (∃ d : ℕ, d = 16 ∧ d > 0 ∧ 
    (∀ a b : ℕ, (∃ i j : ℕ, a = 2 * i ∧ b = 2 * j) → b < a → 
      d ∣ (a^2 - b^2)) ∧
    (∀ e : ℕ, e > d → 
      ∃ x y : ℕ, (∃ p q : ℕ, x = 2 * p ∧ y = 2 * q) ∧ y < x ∧ ¬(e ∣ (x^2 - y^2)))) :=
by sorry


end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l2592_259203


namespace NUMINAMATH_CALUDE_people_not_buying_coffee_l2592_259288

theorem people_not_buying_coffee (total_people : ℕ) (coffee_ratio : ℚ) 
  (h1 : total_people = 25) 
  (h2 : coffee_ratio = 3/5) : 
  total_people - (coffee_ratio * total_people).floor = 10 := by
  sorry

end NUMINAMATH_CALUDE_people_not_buying_coffee_l2592_259288


namespace NUMINAMATH_CALUDE_oranges_for_24_apples_value_l2592_259285

/-- The number of oranges that can be bought for the price of 24 apples -/
def oranges_for_24_apples (apple_price banana_price cucumber_price orange_price : ℚ) : ℚ :=
  24 * apple_price / orange_price

/-- Theorem stating the number of oranges that can be bought for the price of 24 apples -/
theorem oranges_for_24_apples_value
  (h1 : 12 * apple_price = 6 * banana_price)
  (h2 : 3 * banana_price = 5 * cucumber_price)
  (h3 : 2 * cucumber_price = orange_price)
  : oranges_for_24_apples apple_price banana_price cucumber_price orange_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_oranges_for_24_apples_value_l2592_259285


namespace NUMINAMATH_CALUDE_negation_existence_false_l2592_259209

theorem negation_existence_false : ¬(∀ x : ℝ, 2^x + x^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_false_l2592_259209


namespace NUMINAMATH_CALUDE_apple_distribution_l2592_259280

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 5 * x + 12) → 
  (total_apples < 8 * x) →
  (0 ≤ 5 * x + 12 - 8 * (x - 1) ∧ 5 * x + 12 - 8 * (x - 1) < 8) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l2592_259280


namespace NUMINAMATH_CALUDE_complex_equation_difference_l2592_259247

theorem complex_equation_difference (x y : ℝ) : 
  (x : ℂ) + y * I = 1 + 2 * x * I → x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l2592_259247


namespace NUMINAMATH_CALUDE_number_solution_l2592_259218

theorem number_solution : ∃ x : ℝ, 45 - 3 * x = 12 ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_number_solution_l2592_259218


namespace NUMINAMATH_CALUDE_unpacked_boxes_correct_l2592_259260

-- Define the cookie types
inductive CookieType
  | LemonChaletCremes
  | ThinMints
  | Samoas
  | Trefoils

-- Define the function for boxes per case
def boxesPerCase (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 12
  | CookieType.ThinMints => 15
  | CookieType.Samoas => 10
  | CookieType.Trefoils => 18

-- Define the function for boxes sold
def boxesSold (c : CookieType) : ℕ :=
  match c with
  | CookieType.LemonChaletCremes => 31
  | CookieType.ThinMints => 26
  | CookieType.Samoas => 17
  | CookieType.Trefoils => 44

-- Define the function for unpacked boxes
def unpackedBoxes (c : CookieType) : ℕ :=
  boxesSold c % boxesPerCase c

-- Theorem statement
theorem unpacked_boxes_correct (c : CookieType) :
  unpackedBoxes c =
    match c with
    | CookieType.LemonChaletCremes => 7
    | CookieType.ThinMints => 11
    | CookieType.Samoas => 7
    | CookieType.Trefoils => 8
  := by sorry

end NUMINAMATH_CALUDE_unpacked_boxes_correct_l2592_259260


namespace NUMINAMATH_CALUDE_sequence_exceeds_1994_l2592_259229

/-- A sequence satisfying the given conditions -/
def SpecialSequence (x : ℕ → ℝ) (k : ℝ) : Prop :=
  (x 0 = 1) ∧
  (x 1 = 1 + k) ∧
  (k > 0) ∧
  (∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1)) ∧
  (∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2))

/-- The main theorem stating that the sequence eventually exceeds 1994 -/
theorem sequence_exceeds_1994 {x : ℕ → ℝ} {k : ℝ} (h : SpecialSequence x k) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
sorry

end NUMINAMATH_CALUDE_sequence_exceeds_1994_l2592_259229


namespace NUMINAMATH_CALUDE_max_sum_is_1120_l2592_259245

/-- Represents a splitting operation on a pile of coins -/
structure Split :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a > 1)
  (h2 : b ≥ 1)
  (h3 : c ≥ 1)
  (h4 : a = b + c)

/-- Represents the state of the coin piles -/
structure PileState :=
  (piles : List ℕ)
  (board_sum : ℕ)

/-- Performs a single split operation on a pile state -/
def split_pile (state : PileState) (split : Split) : PileState :=
  sorry

/-- Checks if the splitting process is complete -/
def is_complete (state : PileState) : Bool :=
  state.piles.length == 15 && state.piles.all (· == 1)

/-- Finds the maximum possible board sum after splitting 15 coins into 15 piles -/
def max_board_sum : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_sum_is_1120 :
  max_board_sum = 1120 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_1120_l2592_259245


namespace NUMINAMATH_CALUDE_sum_base5_equals_l2592_259212

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

/-- The theorem to be proved -/
theorem sum_base5_equals : 
  decimalToBase5 (base5ToDecimal [1, 2, 3] + base5ToDecimal [4, 3, 2] + base5ToDecimal [2, 1, 4]) = 
  [1, 3, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_sum_base5_equals_l2592_259212


namespace NUMINAMATH_CALUDE_total_limes_picked_l2592_259233

theorem total_limes_picked (fred_limes alyssa_limes nancy_limes david_limes eileen_limes : ℕ)
  (h1 : fred_limes = 36)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35)
  (h4 : david_limes = 42)
  (h5 : eileen_limes = 50) :
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_picked_l2592_259233


namespace NUMINAMATH_CALUDE_least_months_to_double_debt_l2592_259252

def initial_amount : ℝ := 1500
def monthly_rate : ℝ := 0.06

def amount_owed (t : ℕ) : ℝ :=
  initial_amount * (1 + monthly_rate) ^ t

theorem least_months_to_double_debt :
  (∀ n < 12, amount_owed n ≤ 2 * initial_amount) ∧
  amount_owed 12 > 2 * initial_amount :=
sorry

end NUMINAMATH_CALUDE_least_months_to_double_debt_l2592_259252


namespace NUMINAMATH_CALUDE_min_value_theorem_l2592_259206

theorem min_value_theorem (n : ℕ+) (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a^n.val + x^n.val) * (a + x)^n.val / x^n.val ≥ 2^(n.val + 1) * a^n.val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2592_259206


namespace NUMINAMATH_CALUDE_share_of_A_l2592_259269

theorem share_of_A (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  a = (2 / 3) * b →
  b = (1 / 4) * c →
  total = a + b + c →
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_share_of_A_l2592_259269


namespace NUMINAMATH_CALUDE_f_min_value_l2592_259253

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := (x^2 + 4*x + 5)*(x^2 + 4*x + 2) + 2*x^2 + 8*x + 1

/-- Theorem stating that the minimum value of f(x) is -9 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -9 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l2592_259253


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l2592_259207

/-- Calculates the total weight of remaining chocolate eggs after one box is discarded -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  (total_eggs * egg_weight) - (total_eggs / num_boxes * egg_weight) = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l2592_259207


namespace NUMINAMATH_CALUDE_equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l2592_259299

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop := ∃ (k : ℝ), line_equation a k 0 ∧ line_equation a 0 k

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop := ∀ (x y : ℝ), line_equation a x y → (x > 0 → y ≤ 0)

-- Theorem 1: Equal intercepts condition
theorem equal_intercepts_iff_specific_equation :
  ∀ (a : ℝ), equal_intercepts a ↔ (∀ (x y : ℝ), x + y + 4 = 0 ↔ line_equation a x y) :=
sorry

-- Theorem 2: Not passing through second quadrant condition
theorem not_in_second_quadrant_iff_a_leq_neg_one :
  ∀ (a : ℝ), not_in_second_quadrant a ↔ a ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_iff_specific_equation_not_in_second_quadrant_iff_a_leq_neg_one_l2592_259299


namespace NUMINAMATH_CALUDE_square_of_1024_l2592_259215

theorem square_of_1024 : (1024 : ℕ)^2 = 1048576 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1024_l2592_259215


namespace NUMINAMATH_CALUDE_hockey_season_length_l2592_259284

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The total number of hockey games in the season -/
def total_games : ℕ := 182

/-- The number of months in the hockey season -/
def season_length : ℕ := total_games / games_per_month

theorem hockey_season_length :
  season_length = 14 :=
sorry

end NUMINAMATH_CALUDE_hockey_season_length_l2592_259284


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2592_259264

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2592_259264


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_four_l2592_259256

theorem no_solution_iff_k_equals_four :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_four_l2592_259256


namespace NUMINAMATH_CALUDE_pedestrian_speed_theorem_l2592_259244

theorem pedestrian_speed_theorem :
  ∃ (v : ℝ), v > 0 ∧
  (∀ (t : ℝ), 0 ≤ t ∧ t ≤ 2.5 →
    (if t % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 +
    (if (t + 0.5) % 1 < 0.5 then (5 + v) else (5 - v)) * 0.5 = 5) ∧
  ((4 * (5 + v) * 0.5 + 3 * (5 - v) * 0.5) / 3.5 > 5) :=
by sorry

end NUMINAMATH_CALUDE_pedestrian_speed_theorem_l2592_259244


namespace NUMINAMATH_CALUDE_rohan_farm_earnings_l2592_259291

/-- Represents a coconut farm with given characteristics -/
structure CoconutFarm where
  size : ℕ  -- farm size in square meters
  trees_per_sqm : ℕ  -- number of trees per square meter
  coconuts_per_tree : ℕ  -- number of coconuts per tree
  harvest_frequency : ℕ  -- harvest frequency in months
  price_per_coconut : ℚ  -- price per coconut in dollars
  time_period : ℕ  -- time period in months

/-- Calculates the earnings from a coconut farm over a given time period -/
def calculate_earnings (farm : CoconutFarm) : ℚ :=
  let total_trees := farm.size * farm.trees_per_sqm
  let total_coconuts_per_harvest := total_trees * farm.coconuts_per_tree
  let number_of_harvests := farm.time_period / farm.harvest_frequency
  let total_coconuts := total_coconuts_per_harvest * number_of_harvests
  total_coconuts * farm.price_per_coconut

/-- Theorem stating that the earnings from Rohan's coconut farm after 6 months is $240 -/
theorem rohan_farm_earnings :
  let farm : CoconutFarm := {
    size := 20,
    trees_per_sqm := 2,
    coconuts_per_tree := 6,
    harvest_frequency := 3,
    price_per_coconut := 1/2,
    time_period := 6
  }
  calculate_earnings farm = 240 := by sorry

end NUMINAMATH_CALUDE_rohan_farm_earnings_l2592_259291


namespace NUMINAMATH_CALUDE_ben_marbles_count_l2592_259255

theorem ben_marbles_count (ben_marbles : ℕ) (leo_marbles : ℕ) : 
  (leo_marbles = ben_marbles + 20) →
  (ben_marbles + leo_marbles = 132) →
  (ben_marbles = 56) := by
sorry

end NUMINAMATH_CALUDE_ben_marbles_count_l2592_259255


namespace NUMINAMATH_CALUDE_greater_than_implies_greater_than_scaled_and_shifted_l2592_259210

theorem greater_than_implies_greater_than_scaled_and_shifted {a b : ℝ} (h : a > b) : 3*a + 5 > 3*b + 5 := by
  sorry

end NUMINAMATH_CALUDE_greater_than_implies_greater_than_scaled_and_shifted_l2592_259210


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l2592_259249

/-- Proves that adding a specific amount of pure alcohol to a given solution results in the desired alcohol percentage -/
theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 100 →
  initial_percentage = 0.2 →
  added_alcohol = 14.285714285714286 →
  final_percentage = 0.3 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = final_percentage := by
  sorry

#check alcohol_solution_proof

end NUMINAMATH_CALUDE_alcohol_solution_proof_l2592_259249


namespace NUMINAMATH_CALUDE_five_divides_x_l2592_259259

theorem five_divides_x (x y : ℕ) (hx : x > 1) (heq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_five_divides_x_l2592_259259


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_prism_inscribed_l2592_259248

/-- Given a rectangular prism with adjacent face areas of 2, 3, and 6,
    and all vertices lying on the same spherical surface,
    prove that the surface area of this sphere is 14π. -/
theorem sphere_surface_area_of_prism_inscribed (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b = 6 → b * c = 2 → a * c = 3 →
  (4 : ℝ) * Real.pi * ((a^2 + b^2 + c^2) / 4) = 14 * Real.pi := by
  sorry

#check sphere_surface_area_of_prism_inscribed

end NUMINAMATH_CALUDE_sphere_surface_area_of_prism_inscribed_l2592_259248


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2592_259223

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- A geometric sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  q : ℝ
  h_geometric : ∀ n, b (n + 1) = b n * q

/-- The main theorem -/
theorem arithmetic_geometric_sequence_relation
  (a : ArithmeticSequence)
  (b : GeometricSequence)
  (h_consecutive : b.b 1 = a.a 5 ∧ b.b 2 = a.a 8 ∧ b.b 3 = a.a 13)
  (h_b2 : b.b 2 = 5) :
  ∀ n, b.b n = 5 * (5/3)^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l2592_259223


namespace NUMINAMATH_CALUDE_valid_numbers_count_l2592_259232

/-- Converts a base-10 number to base-12 --/
def toBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a base-12 number uses only digits 0-9 --/
def usesOnlyDigits0to9 (n : ℕ) : Bool := sorry

/-- Counts numbers up to n (base-10) that use only digits 0-9 in base-12 --/
def countValidNumbers (n : ℕ) : ℕ := sorry

theorem valid_numbers_count :
  countValidNumbers 1200 = 90 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l2592_259232


namespace NUMINAMATH_CALUDE_adam_remaining_candy_l2592_259287

/-- Calculates the number of candy pieces Adam has left after giving some boxes away. -/
def remaining_candy_pieces (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - given_away_boxes) * pieces_per_box

/-- Proves that Adam has 36 pieces of candy left. -/
theorem adam_remaining_candy :
  remaining_candy_pieces 13 7 6 = 36 := by
  sorry

#eval remaining_candy_pieces 13 7 6

end NUMINAMATH_CALUDE_adam_remaining_candy_l2592_259287


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l2592_259276

/-- An inverse proportion function passing through (3, -2) lies in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ (k : ℝ), k ≠ 0 →
  (∃ (f : ℝ → ℝ), (∀ x, x ≠ 0 → f x = k / x) ∧ f 3 = -2) →
  (∀ x y, (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l2592_259276


namespace NUMINAMATH_CALUDE_jake_alcohol_consumption_l2592_259257

-- Define the given constants
def total_shots : ℚ := 8
def ounces_per_shot : ℚ := 3/2
def alcohol_percentage : ℚ := 1/2

-- Define Jake's share of shots
def jakes_shots : ℚ := total_shots / 2

-- Define the function to calculate pure alcohol consumed
def pure_alcohol_consumed : ℚ :=
  jakes_shots * ounces_per_shot * alcohol_percentage

-- Theorem statement
theorem jake_alcohol_consumption :
  pure_alcohol_consumed = 3 := by sorry

end NUMINAMATH_CALUDE_jake_alcohol_consumption_l2592_259257


namespace NUMINAMATH_CALUDE_emily_square_subtraction_l2592_259205

theorem emily_square_subtraction : 49^2 = 50^2 - 99 := by
  sorry

end NUMINAMATH_CALUDE_emily_square_subtraction_l2592_259205


namespace NUMINAMATH_CALUDE_max_value_of_product_l2592_259267

theorem max_value_of_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  x^2 * y^3 * z ≤ 1 / 3888 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l2592_259267


namespace NUMINAMATH_CALUDE_equation_A_is_linear_l2592_259235

/-- An equation is linear in two variables if it can be written in the form ax + by + c = 0,
    where a, b, and c are constants, and a and b are not both zero. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y + c = 0

/-- The equation (2y-1)/5 = 2 - (3x-2)/4 -/
def equation_A (x y : ℝ) : Prop :=
  (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

theorem equation_A_is_linear :
  is_linear_equation_in_two_variables equation_A :=
sorry

end NUMINAMATH_CALUDE_equation_A_is_linear_l2592_259235


namespace NUMINAMATH_CALUDE_passengers_left_is_200_l2592_259216

/-- The number of minutes between train arrivals -/
def train_interval : ℕ := 5

/-- The number of passengers each train takes -/
def passengers_taken : ℕ := 320

/-- The total number of different passengers stepping on and off trains in one hour -/
def total_passengers : ℕ := 6240

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of passengers each train leaves at the station -/
def passengers_left : ℕ := (total_passengers - (minutes_per_hour / train_interval * passengers_taken)) / (minutes_per_hour / train_interval)

theorem passengers_left_is_200 : passengers_left = 200 := by
  sorry

end NUMINAMATH_CALUDE_passengers_left_is_200_l2592_259216


namespace NUMINAMATH_CALUDE_square_room_carpet_area_l2592_259297

theorem square_room_carpet_area (room_side : ℝ) (sq_yard_to_sq_feet : ℝ) : 
  room_side = 9 → sq_yard_to_sq_feet = 9 → (room_side * room_side) / sq_yard_to_sq_feet = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_room_carpet_area_l2592_259297


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2592_259283

noncomputable def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

theorem functional_equation_solutions (f : ℝ → ℝ) :
  FunctionalEquation f →
  (∀ x : ℝ, f x = 0) ∨
  (∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → f x = 1) ∧ f 0 = a) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2592_259283


namespace NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l2592_259238

/-- Represents an angle in the coordinate system -/
structure Angle where
  value : ℝ

/-- Represents the terminal side of an angle -/
structure TerminalSide where
  x : ℝ
  y : ℝ

/-- Returns the terminal side of an angle -/
noncomputable def terminalSide (a : Angle) : TerminalSide :=
  { x := Real.cos a.value, y := Real.sin a.value }

/-- Theorem: Equal angles have the same terminal side -/
theorem equal_angles_same_terminal_side (a b : Angle) :
  a = b → terminalSide a = terminalSide b := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_same_terminal_side_l2592_259238


namespace NUMINAMATH_CALUDE_probability_of_specific_match_l2592_259246

/-- Calculates the probability of two specific players facing each other in a tournament. -/
theorem probability_of_specific_match (n : ℕ) (h : n = 26) : 
  (n - 1 : ℚ) / (n * (n - 1) / 2) = 1 / 13 := by
  sorry

#check probability_of_specific_match

end NUMINAMATH_CALUDE_probability_of_specific_match_l2592_259246


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l2592_259282

/-- A function f is monotonic on an interval [a,b] if it is either 
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The statement of the problem -/
theorem monotonic_quadratic (a : ℝ) :
  IsMonotonic (fun x => x^2 + (1-a)*x + 3) 1 4 ↔ a ≥ 9 ∨ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l2592_259282


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2592_259261

/-- A hyperbola with foci at (-3,0) and (3,0), and a vertex at (2,0) has the equation x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let foci_1 : ℝ × ℝ := (-3, 0)
  let foci_2 : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (2, 0)
  (x^2 / 4 - y^2 / 5 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      vertex.1 = a ∧
      (foci_2.1 - foci_1.1)^2 / 4 = a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2592_259261


namespace NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l2592_259295

/-- Hyperbola C with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

/-- Line passing through a point and intersecting the hyperbola -/
structure IntersectingLine where
  passing_point : ℝ × ℝ
  intersection_point1 : ℝ × ℝ
  intersection_point2 : ℝ × ℝ

/-- Theorem stating that the intersection point P lies on a fixed line -/
theorem intersection_point_on_fixed_line (C : Hyperbola) (L : IntersectingLine) : 
  C.center = (0, 0) →
  C.left_focus = (-2 * Real.sqrt 5, 0) →
  C.eccentricity = Real.sqrt 5 →
  C.left_vertex = (-2, 0) →
  C.right_vertex = (2, 0) →
  L.passing_point = (-4, 0) →
  L.intersection_point1.1 < 0 ∧ L.intersection_point1.2 > 0 → -- M in second quadrant
  ∃ (P : ℝ × ℝ), P.1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l2592_259295
