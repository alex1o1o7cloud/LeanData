import Mathlib

namespace NUMINAMATH_CALUDE_normal_transform_theorem_l3642_364214

/-- Transforms a standard normal random variable to a normal distribution with given mean and standard deviation -/
def transform (x : ℝ) (μ σ : ℝ) : ℝ := σ * x + μ

/-- The four standard normal random variables -/
def X₁ : ℝ := 0.06
def X₂ : ℝ := -1.10
def X₃ : ℝ := -1.52
def X₄ : ℝ := 0.83

/-- The mean of the target normal distribution -/
def μ : ℝ := 2

/-- The standard deviation of the target normal distribution -/
def σ : ℝ := 3

/-- Theorem stating that the transformation of the given standard normal random variables
    results in the specified values for the target normal distribution -/
theorem normal_transform_theorem :
  (transform X₁ μ σ, transform X₂ μ σ, transform X₃ μ σ, transform X₄ μ σ) =
  (2.18, -1.3, -2.56, 4.49) := by
  sorry

end NUMINAMATH_CALUDE_normal_transform_theorem_l3642_364214


namespace NUMINAMATH_CALUDE_locus_of_vertices_is_parabola_l3642_364205

/-- The locus of vertices of a family of parabolas forms a parabola -/
theorem locus_of_vertices_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    (∀ t : ℝ, ∃ (x y : ℝ),
      (y = a * x^2 + (2 * t + 1) * x + c) ∧
      (x = -(2 * t + 1) / (2 * a)) ∧
      (y = A * x^2 + B * x + C)) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertices_is_parabola_l3642_364205


namespace NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l3642_364271

/-- Sum of two arithmetic sequences with specific properties -/
theorem sum_of_two_arithmetic_sequences : 
  let seq1 := [2, 14, 26, 38, 50]
  let seq2 := [6, 18, 30, 42, 54]
  (seq1.sum + seq2.sum) = 280 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_arithmetic_sequences_l3642_364271


namespace NUMINAMATH_CALUDE_horner_method_v2_l3642_364217

def f (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def horner_v2 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  v₁ * x + a₄

theorem horner_method_v2 :
  horner_v2 2 3 0 5 6 7 8 2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l3642_364217


namespace NUMINAMATH_CALUDE_wheel_center_travel_distance_l3642_364275

/-- The distance traveled by the center of a wheel rolling one complete revolution -/
theorem wheel_center_travel_distance (r : ℝ) (h : r = 1) :
  let circumference := 2 * Real.pi * r
  circumference = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_wheel_center_travel_distance_l3642_364275


namespace NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l3642_364278

-- Define an arithmetic sequence
def is_arithmetic_seq (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = m

-- Define an equal difference of squares sequence
def is_equal_diff_squares_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1)^2 - a n^2 = d

theorem m_zero_sufficient_not_necessary :
  (∀ b : ℕ → ℝ, is_arithmetic_seq b 0 → is_equal_diff_squares_seq b) ∧
  (∃ b : ℕ → ℝ, ∃ m : ℝ, m ≠ 0 ∧ is_arithmetic_seq b m ∧ is_equal_diff_squares_seq b) :=
by sorry

end NUMINAMATH_CALUDE_m_zero_sufficient_not_necessary_l3642_364278


namespace NUMINAMATH_CALUDE_theater_probability_ratio_l3642_364277

theorem theater_probability_ratio : 
  let n : ℕ := 4  -- number of sections and acts
  let p : ℝ := 1 / 4  -- probability of moving in a given act
  let q : ℝ := 1 - p  -- probability of not moving in a given act
  let prob_move_once : ℝ := n * p * q^(n-1)  -- probability of moving exactly once
  let prob_move_twice : ℝ := (n.choose 2) * p^2 * q^(n-2)  -- probability of moving exactly twice
  prob_move_twice / prob_move_once = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_theater_probability_ratio_l3642_364277


namespace NUMINAMATH_CALUDE_triangle_sum_equals_nine_l3642_364234

def triangle_operation (a b c : ℤ) : ℤ := a * b - c

theorem triangle_sum_equals_nine : 
  triangle_operation 3 4 5 + triangle_operation 1 2 4 + triangle_operation 2 5 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_nine_l3642_364234


namespace NUMINAMATH_CALUDE_bryans_annual_travel_time_l3642_364263

/-- Represents the time in minutes for each leg of Bryan's journey --/
structure JourneyTime where
  walkToBus : ℕ
  busRide : ℕ
  walkToJob : ℕ

/-- Calculates the total annual travel time in hours --/
def annualTravelTime (j : JourneyTime) (daysWorked : ℕ) : ℕ :=
  let totalDailyMinutes := 2 * (j.walkToBus + j.busRide + j.walkToJob)
  (totalDailyMinutes * daysWorked) / 60

/-- Theorem stating that Bryan spends 365 hours per year traveling to and from work --/
theorem bryans_annual_travel_time :
  let j : JourneyTime := { walkToBus := 5, busRide := 20, walkToJob := 5 }
  annualTravelTime j 365 = 365 := by
  sorry

end NUMINAMATH_CALUDE_bryans_annual_travel_time_l3642_364263


namespace NUMINAMATH_CALUDE_friend_payment_percentage_l3642_364253

def adoption_fee : ℝ := 200
def james_payment : ℝ := 150

theorem friend_payment_percentage : 
  (adoption_fee - james_payment) / adoption_fee * 100 = 25 := by sorry

end NUMINAMATH_CALUDE_friend_payment_percentage_l3642_364253


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l3642_364249

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (50/17, 24/17)

/-- First line equation: 2y = 3x - 6 -/
def line1 (x y : ℚ) : Prop := 2 * y = 3 * x - 6

/-- Second line equation: x + 5y = 10 -/
def line2 (x y : ℚ) : Prop := x + 5 * y = 10

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations : 
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_unique_intersection_point_l3642_364249


namespace NUMINAMATH_CALUDE_orange_buckets_problem_l3642_364262

/-- The problem of calculating the number of oranges and their total weight -/
theorem orange_buckets_problem :
  let bucket1 : ℝ := 22.5
  let bucket2 : ℝ := 2 * bucket1 + 3
  let bucket3 : ℝ := bucket2 - 11.5
  let bucket4 : ℝ := 1.5 * (bucket1 + bucket3)
  let weight1 : ℝ := 0.3
  let weight3 : ℝ := 0.4
  let weight4 : ℝ := 0.35
  let total_oranges : ℝ := bucket1 + bucket2 + bucket3 + bucket4
  let total_weight : ℝ := weight1 * bucket1 + weight3 * bucket3 + weight4 * bucket4
  total_oranges = 195.5 ∧ total_weight = 52.325 := by
  sorry


end NUMINAMATH_CALUDE_orange_buckets_problem_l3642_364262


namespace NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l3642_364203

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population_size : Nat
  sample_size : Nat
  last_sample : Nat

/-- Calculates the interval between samples -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.population_size / s.sample_size

/-- Calculates the first sampled number -/
def first_sample (s : SystematicSampling) : Nat :=
  s.last_sample % (sample_interval s)

/-- Calculates the second sampled number -/
def second_sample (s : SystematicSampling) : Nat :=
  first_sample s + sample_interval s

/-- Theorem stating the first two sampled numbers for the given scenario -/
theorem systematic_sampling_first_two_samples
  (s : SystematicSampling)
  (h1 : s.population_size = 8000)
  (h2 : s.sample_size = 50)
  (h3 : s.last_sample = 7900) :
  first_sample s = 60 ∧ second_sample s = 220 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_first_two_samples_l3642_364203


namespace NUMINAMATH_CALUDE_smallest_whole_number_larger_than_sum_l3642_364236

def mixed_to_fraction (whole : ℤ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num : ℚ) / (denom : ℚ)

def sum_of_mixed_numbers : ℚ :=
  mixed_to_fraction 1 2 3 +
  mixed_to_fraction 2 1 4 +
  mixed_to_fraction 3 3 8 +
  mixed_to_fraction 4 1 6

theorem smallest_whole_number_larger_than_sum :
  (⌈sum_of_mixed_numbers⌉ : ℤ) = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_larger_than_sum_l3642_364236


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l3642_364274

theorem right_triangle_area_perimeter_relation : 
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    c^2 = a^2 + b^2 ∧
    (a * b : ℚ) / 2 = 4 * (a + b + c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_relation_l3642_364274


namespace NUMINAMATH_CALUDE_one_third_of_390_l3642_364204

theorem one_third_of_390 : (1 / 3 : ℚ) * 390 = 130 := by sorry

end NUMINAMATH_CALUDE_one_third_of_390_l3642_364204


namespace NUMINAMATH_CALUDE_fruit_cost_problem_l3642_364239

/-- The cost of fruits problem -/
theorem fruit_cost_problem (apple_price pear_price mango_price : ℝ) 
  (h1 : 5 * apple_price + 4 * pear_price = 48)
  (h2 : 2 * apple_price + 3 * mango_price = 33)
  (h3 : mango_price = pear_price + 2.5) :
  3 * apple_price + 3 * pear_price = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_problem_l3642_364239


namespace NUMINAMATH_CALUDE_initial_men_count_l3642_364283

/-- Represents the initial number of men -/
def initialMen : ℕ := 200

/-- Represents the initial food duration in days -/
def initialDuration : ℕ := 20

/-- Represents the number of days after which some men leave -/
def daysBeforeLeaving : ℕ := 15

/-- Represents the number of men who leave -/
def menWhoLeave : ℕ := 100

/-- Represents the remaining food duration after some men leave -/
def remainingDuration : ℕ := 10

theorem initial_men_count :
  initialMen * daysBeforeLeaving = (initialMen - menWhoLeave) * remainingDuration ∧
  initialMen * initialDuration = initialMen * daysBeforeLeaving + (initialMen - menWhoLeave) * remainingDuration :=
by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3642_364283


namespace NUMINAMATH_CALUDE_consecutive_odd_product_sum_l3642_364286

theorem consecutive_odd_product_sum (a b c : ℤ) : 
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧  -- a, b, c are odd
  (b = a + 2) ∧ (c = b + 2) ∧                -- a, b, c are consecutive
  (a * b * c = 9177) →                       -- their product is 9177
  (a + b + c = 63) :=                        -- their sum is 63
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_sum_l3642_364286


namespace NUMINAMATH_CALUDE_election_win_probability_l3642_364230

/-- Represents the state of an election --/
structure ElectionState :=
  (total_voters : ℕ)
  (votes_a : ℕ)
  (votes_b : ℕ)

/-- Calculates the probability of candidate A winning given the current state --/
noncomputable def win_probability (state : ElectionState) : ℚ :=
  sorry

/-- The main theorem stating the probability of the initially leading candidate winning --/
theorem election_win_probability :
  let initial_state : ElectionState := ⟨2019, 2, 1⟩
  win_probability initial_state = 1513 / 2017 :=
sorry

end NUMINAMATH_CALUDE_election_win_probability_l3642_364230


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_zero_l3642_364215

theorem quadratic_inequality_implies_zero (a b x y : ℤ) 
  (h1 : a > b^2) 
  (h2 : a^2 * x^2 + 2*a*b * x*y + (b^2 + 1) * y^2 < b^2 + 1) : 
  x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_zero_l3642_364215


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3642_364259

theorem largest_solution_of_equation :
  ∃ (x : ℚ), x = -10/9 ∧ 
  5*(9*x^2 + 9*x + 10) = x*(9*x - 40) ∧
  ∀ (y : ℚ), 5*(9*y^2 + 9*y + 10) = y*(9*y - 40) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3642_364259


namespace NUMINAMATH_CALUDE_function_relationship_l3642_364219

theorem function_relationship (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  ¬(∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3 →
    ∀ x y : ℝ, x < y → a^x > a^y) :=
by sorry

end NUMINAMATH_CALUDE_function_relationship_l3642_364219


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l3642_364284

/-- Given a cubic function f(x) = ax^3 - 3x^2 + x + b with a ≠ 0, 
    if the tangent line at x = 1 is 2x + y + 1 = 0, 
    then f(x) = x^3 - 3x^2 + x - 2 -/
theorem cubic_function_uniqueness (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * x^2 + x + b
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 - 6 * x + 1
  (f' 1 = -2 ∧ f 1 = -3) → f = λ x ↦ x^3 - 3 * x^2 + x - 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l3642_364284


namespace NUMINAMATH_CALUDE_wendy_count_problem_l3642_364231

theorem wendy_count_problem (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 28) 
  (h2 : total_legs = 92) : 
  ∃ (people animals : ℕ), 
    people + animals = total_heads ∧ 
    2 * people + 4 * animals = total_legs ∧ 
    people = 10 := by
  sorry

end NUMINAMATH_CALUDE_wendy_count_problem_l3642_364231


namespace NUMINAMATH_CALUDE_equation_solution_l3642_364200

theorem equation_solution : ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3642_364200


namespace NUMINAMATH_CALUDE_complex_modulus_l3642_364291

theorem complex_modulus (x y : ℝ) : 
  (1 : ℂ) + x * Complex.I = (2 - y) - 3 * Complex.I → 
  Complex.abs (x + y * Complex.I) = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l3642_364291


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3642_364237

/-- In a pentagon FGHIJ, given the following conditions:
  - Angle F measures 50°
  - Angle G measures 75°
  - Angles H and I are equal
  - Angle J is 10° more than twice angle H
  Prove that the largest angle measures 212.5° -/
theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 50 ∧ 
  G = 75 ∧ 
  H = I ∧ 
  J = 2 * H + 10 ∧ 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 212.5 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l3642_364237


namespace NUMINAMATH_CALUDE_S_equals_formula_S_2k_minus_1_is_polynomial_l3642_364296

-- Define S as a function of n and k
def S (n k : ℕ) : ℚ := sorry

-- Define S_{2k-1}(n) as a function
def S_2k_minus_1 (n k : ℕ) : ℚ := sorry

-- Theorem 1: S equals (n^k * (n+1)^k) / 2
theorem S_equals_formula (n k : ℕ) : 
  S n k = (n^k * (n+1)^k : ℚ) / 2 := by sorry

-- Theorem 2: S_{2k-1}(n) is a polynomial of degree k in (n(n+1))/2
theorem S_2k_minus_1_is_polynomial (n k : ℕ) :
  ∃ (p : Polynomial ℚ), 
    (S_2k_minus_1 n k = p.eval ((n * (n+1) : ℕ) / 2 : ℚ)) ∧ 
    (p.degree = k) := by sorry

end NUMINAMATH_CALUDE_S_equals_formula_S_2k_minus_1_is_polynomial_l3642_364296


namespace NUMINAMATH_CALUDE_clayton_first_game_score_l3642_364229

def clayton_basketball_score (game1 : ℝ) : Prop :=
  let game2 : ℝ := 14
  let game3 : ℝ := 6
  let game4 : ℝ := (game1 + game2 + game3) / 3
  let total : ℝ := 40
  (game1 + game2 + game3 + game4 = total) ∧ (game1 = 10)

theorem clayton_first_game_score :
  ∃ (game1 : ℝ), clayton_basketball_score game1 :=
sorry

end NUMINAMATH_CALUDE_clayton_first_game_score_l3642_364229


namespace NUMINAMATH_CALUDE_relay_race_distance_per_member_l3642_364223

theorem relay_race_distance_per_member 
  (total_distance : ℕ) 
  (team_size : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_size = 5) :
  total_distance / team_size = 30 := by
sorry

end NUMINAMATH_CALUDE_relay_race_distance_per_member_l3642_364223


namespace NUMINAMATH_CALUDE_A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l3642_364282

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := x^2 - 2 * x - y + x * y - 5

-- Theorem 1: A - 3B = 5x + 5y - 7xy + 15
theorem A_minus_3B_formula (x y : ℝ) :
  A x y - 3 * B x y = 5 * x + 5 * y - 7 * x * y + 15 := by sorry

-- Theorem 2: A - 3B = 26 when (x + y - 4/5)^2 + |xy + 1| = 0
theorem A_minus_3B_value (x y : ℝ) 
  (h : (x + y - 4/5)^2 + |x * y + 1| = 0) :
  A x y - 3 * B x y = 26 := by sorry

-- Theorem 3: x = 5/7 when the coefficient of y in A - 3B is zero
theorem x_value_when_independent_of_y (x : ℝ) 
  (h : ∀ y : ℝ, 5 - 7 * x = 0) :
  x = 5/7 := by sorry

end NUMINAMATH_CALUDE_A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l3642_364282


namespace NUMINAMATH_CALUDE_incorrect_mark_calculation_l3642_364293

theorem incorrect_mark_calculation (correct_mark : ℤ) (num_pupils : ℕ) 
  (h1 : correct_mark = 63)
  (h2 : num_pupils = 40) :
  ∃ (incorrect_mark : ℤ), 
    (incorrect_mark - correct_mark) * num_pupils = num_pupils * 2 ∧ 
    incorrect_mark = 83 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_mark_calculation_l3642_364293


namespace NUMINAMATH_CALUDE_possible_m_values_l3642_364243

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l3642_364243


namespace NUMINAMATH_CALUDE_factorization_equality_l3642_364280

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3642_364280


namespace NUMINAMATH_CALUDE_natural_fraction_pairs_l3642_364206

def is_valid_pair (x y : ℕ) : Prop :=
  (∃ k : ℕ, (x + 1) = k * y) ∧ (∃ m : ℕ, (y + 1) = m * x)

theorem natural_fraction_pairs :
  ∀ x y : ℕ, is_valid_pair x y ↔ 
    ((x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_natural_fraction_pairs_l3642_364206


namespace NUMINAMATH_CALUDE_sum_of_square_root_differences_l3642_364287

theorem sum_of_square_root_differences (S : ℝ) : 
  S = 1 / (4 - Real.sqrt 9) - 1 / (Real.sqrt 9 - Real.sqrt 8) + 
      1 / (Real.sqrt 8 - Real.sqrt 7) - 1 / (Real.sqrt 7 - Real.sqrt 6) + 
      1 / (Real.sqrt 6 - 3) → 
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_root_differences_l3642_364287


namespace NUMINAMATH_CALUDE_monotonically_decreasing_iff_a_leq_neg_three_l3642_364216

/-- A function f is monotonically decreasing on an interval [a, b] if for all x, y in [a, b],
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

/-- The quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- Theorem: f is monotonically decreasing on [-2, 4] if and only if a ≤ -3 -/
theorem monotonically_decreasing_iff_a_leq_neg_three (a : ℝ) :
  MonotonicallyDecreasing (f a) (-2) 4 ↔ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_iff_a_leq_neg_three_l3642_364216


namespace NUMINAMATH_CALUDE_cosA_value_l3642_364294

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- State the theorem
theorem cosA_value (t : Triangle) 
  (h : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C) : 
  Real.cos t.A = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosA_value_l3642_364294


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3642_364290

/-- Given a triangle with sides a, b, and c, where a = 3, b = 5, and c is a root of x^2 - 5x + 4 = 0
    that satisfies the triangle inequality, prove that the perimeter is 12. -/
theorem triangle_perimeter (a b c : ℝ) : 
  a = 3 → b = 5 → c^2 - 5*c + 4 = 0 → 
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3642_364290


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3642_364269

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → x^2 + 4*y^2 + z^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3642_364269


namespace NUMINAMATH_CALUDE_digit_sum_property_l3642_364251

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A proposition stating that a number is a 1962-digit number -/
def is1962DigitNumber (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : is1962DigitNumber n) 
  (h2 : n % 9 = 0) : 
  sumOfDigits (sumOfDigits (sumOfDigits n)) = 9 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3642_364251


namespace NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3642_364241

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12

/-- Theorem stating that for a 5-inch by 7-inch picture, enlarged and bordered as described, 
    the minimum framing needed is 6 feet. -/
theorem min_framing_for_specific_picture : 
  min_framing_feet 5 7 3 = 6 := by
  sorry

#eval min_framing_feet 5 7 3

end NUMINAMATH_CALUDE_min_framing_for_specific_picture_l3642_364241


namespace NUMINAMATH_CALUDE_complex_power_2013_l3642_364245

theorem complex_power_2013 : (((1 + Complex.I) / (1 - Complex.I)) ^ 2013 : ℂ) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2013_l3642_364245


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3642_364255

theorem unique_three_digit_number (g : ℕ) (h : ℕ) (hg : g > 2) (hh : h = g + 1 ∨ h = g - 1) :
  ∃! (a b c : ℕ),
    0 ≤ a ∧ a < g ∧
    0 ≤ b ∧ b < g ∧
    0 ≤ c ∧ c < g ∧
    g^2 * a + g * b + c = h^2 * c + h * b + a ∧
    a = (g + 1) / 2 ∧
    b = (g - 1) / 2 ∧
    c = (g - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3642_364255


namespace NUMINAMATH_CALUDE_eunji_reading_pages_l3642_364276

theorem eunji_reading_pages (pages_tuesday pages_thursday total_pages : ℕ) 
  (h1 : pages_tuesday = 18)
  (h2 : pages_thursday = 23)
  (h3 : total_pages = 60)
  (h4 : pages_tuesday + pages_thursday + (total_pages - pages_tuesday - pages_thursday) = total_pages) :
  total_pages - pages_tuesday - pages_thursday = 19 := by
  sorry

end NUMINAMATH_CALUDE_eunji_reading_pages_l3642_364276


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3642_364226

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line y = ax + 1 -/
def slope₁ (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope₂ (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is a necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope₁ a) (slope₂ a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3642_364226


namespace NUMINAMATH_CALUDE_bell_rings_count_l3642_364227

def number_of_classes : Nat := 5

def current_class : Nat := 5

def bell_rings_per_class : Nat := 2

theorem bell_rings_count (n : Nat) (c : Nat) (r : Nat) 
  (h1 : n = number_of_classes) 
  (h2 : c = current_class) 
  (h3 : r = bell_rings_per_class) 
  (h4 : c ≤ n) : 
  (c - 1) * r + 1 = 9 := by
  sorry

#check bell_rings_count

end NUMINAMATH_CALUDE_bell_rings_count_l3642_364227


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l3642_364281

theorem number_with_specific_remainders : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → n % k = k - 1) ∧ n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l3642_364281


namespace NUMINAMATH_CALUDE_expression_evaluation_l3642_364288

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = 1) : 
  (2 * m^2 - 3 * m * n + 8) - (5 * m * n - 4 * m^2 + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3642_364288


namespace NUMINAMATH_CALUDE_march_largest_drop_l3642_364285

/-- Represents the months in the first half of 1994 -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- Returns the price change for a given month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 0.50
  | Month.March    => -3.00
  | Month.April    => 2.00
  | Month.May      => -1.50
  | Month.June     => -0.75

/-- Determines if a given month has the largest price drop -/
def has_largest_drop (m : Month) : Prop :=
  ∀ (other : Month), price_change m ≤ price_change other

theorem march_largest_drop :
  has_largest_drop Month.March :=
sorry

end NUMINAMATH_CALUDE_march_largest_drop_l3642_364285


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_seven_mod_thirteen_l3642_364254

theorem smallest_five_digit_congruent_to_seven_mod_thirteen : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n % 13 = 7 → n ≥ 10004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_seven_mod_thirteen_l3642_364254


namespace NUMINAMATH_CALUDE_fraction_value_l3642_364295

theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  x * z / (y * w) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l3642_364295


namespace NUMINAMATH_CALUDE_max_sum_pair_contains_96420_l3642_364224

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ 
  a ≥ 10000 ∧ a < 100000 ∧ 
  b ≥ 10000 ∧ b < 100000 ∧
  (∀ d : ℕ, d < 10 → 
    (∃! i : ℕ, i < 5 ∧ (a / 10^i) % 10 = d) ∨
    (∃! i : ℕ, i < 5 ∧ (b / 10^i) % 10 = d))

def is_max_sum_pair (a b : ℕ) : Prop :=
  is_valid_pair a b ∧
  ∀ c d : ℕ, is_valid_pair c d → a + b ≥ c + d

theorem max_sum_pair_contains_96420 :
  ∃ n : ℕ, is_max_sum_pair 96420 n ∨ is_max_sum_pair n 96420 :=
sorry

end NUMINAMATH_CALUDE_max_sum_pair_contains_96420_l3642_364224


namespace NUMINAMATH_CALUDE_sequence_formula_l3642_364258

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 3 + 2 * a n

theorem sequence_formula (a : ℕ → ℝ) (h : ∀ n, sequence_sum a n = 3 + 2 * a n) :
  ∀ n, a n = -3 * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l3642_364258


namespace NUMINAMATH_CALUDE_power_set_of_S_l3642_364209

def S : Set ℕ := {0, 1}

theorem power_set_of_S :
  𝒫 S = {∅, {0}, {1}, {0, 1}} := by
  sorry

end NUMINAMATH_CALUDE_power_set_of_S_l3642_364209


namespace NUMINAMATH_CALUDE_distance_inequality_l3642_364268

/-- Given five points A, B, C, D, E on a plane, 
    the sum of distances AB + CD + DE + EC 
    is less than or equal to 
    the sum of distances AC + AD + AE + BC + BD + BE -/
theorem distance_inequality (A B C D E : EuclideanSpace ℝ (Fin 2)) :
  dist A B + dist C D + dist D E + dist E C ≤ 
  dist A C + dist A D + dist A E + dist B C + dist B D + dist B E := by
  sorry

end NUMINAMATH_CALUDE_distance_inequality_l3642_364268


namespace NUMINAMATH_CALUDE_hayleys_friends_l3642_364267

def total_stickers : ℕ := 72
def stickers_per_friend : ℕ := 8

theorem hayleys_friends :
  total_stickers / stickers_per_friend = 9 :=
by sorry

end NUMINAMATH_CALUDE_hayleys_friends_l3642_364267


namespace NUMINAMATH_CALUDE_inclination_angle_of_line_l3642_364222

/-- Given a function f(x) = a*sin(x) - b*cos(x) with symmetry axis x = π/4,
    prove that the inclination angle of the line ax - by + c = 0 is 3π/4 -/
theorem inclination_angle_of_line (a b c : ℝ) :
  (∀ x, a * Real.sin (π/4 + x) - b * Real.cos (π/4 + x) = 
        a * Real.sin (π/4 - x) - b * Real.cos (π/4 - x)) →
  Real.arctan (a / b) = 3 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_of_line_l3642_364222


namespace NUMINAMATH_CALUDE_tangent_slope_ratio_l3642_364270

-- Define the function f(x) = ax² + b
def f (a b x : ℝ) : ℝ := a * x^2 + b

-- Define the derivative of f
def f_derivative (a b x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_ratio (a b : ℝ) :
  f_derivative a b 1 = 2 ∧ f a b 1 = 3 → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_ratio_l3642_364270


namespace NUMINAMATH_CALUDE_y_value_l3642_364246

theorem y_value (x y : ℝ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l3642_364246


namespace NUMINAMATH_CALUDE_gold_silver_weight_problem_l3642_364244

theorem gold_silver_weight_problem (x y : ℝ) : 
  (9 * x = 11 * y) ∧ ((10 * y + x) - (8 * x + y) = 13) ↔ 
  (9 * x = 11 * y ∧ 
   ∃ (gold_bag silver_bag : ℝ),
     gold_bag = 9 * x ∧
     silver_bag = 11 * y ∧
     gold_bag = silver_bag ∧
     (silver_bag + x - y) - (gold_bag - x + y) = 13) :=
by sorry

end NUMINAMATH_CALUDE_gold_silver_weight_problem_l3642_364244


namespace NUMINAMATH_CALUDE_completing_square_result_l3642_364220

theorem completing_square_result (x : ℝ) :
  x^2 - 4*x + 2 = 0 → (x - 2)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_completing_square_result_l3642_364220


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3642_364212

theorem intersection_nonempty_implies_a_value (a : ℝ) : 
  let P : Set ℝ := {0, a}
  let Q : Set ℝ := {1, 2}
  (P ∩ Q).Nonempty → a = 1 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3642_364212


namespace NUMINAMATH_CALUDE_lucy_doll_collection_l3642_364257

/-- Represents Lucy's doll collection problem -/
theorem lucy_doll_collection (X : ℕ) (Z : ℕ) : 
  (X : ℚ) * (1 + 1/5) = X + 5 → -- 20% increase after adding 5 dolls
  Z = (X + 5 + (X + 5) / 10 : ℚ).floor → -- 10% more dolls from updated collection
  X = 25 ∧ Z = 33 := by
  sorry

end NUMINAMATH_CALUDE_lucy_doll_collection_l3642_364257


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3642_364247

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ [1, 2] → x^2 < 4) ↔ (∃ x : ℝ, x ∈ [1, 2] ∧ x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3642_364247


namespace NUMINAMATH_CALUDE_min_value_abc_minus_b_l3642_364208

def S : Finset Int := {-10, -7, -3, 0, 4, 6, 9}

theorem min_value_abc_minus_b :
  (∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a * b * c - b = -546) ∧
  (∀ (a b c : Int), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    a * b * c - b ≥ -546) :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_minus_b_l3642_364208


namespace NUMINAMATH_CALUDE_max_value_cos_squared_minus_sin_l3642_364202

open Real

theorem max_value_cos_squared_minus_sin (x : ℝ) : 
  ∃ (M : ℝ), M = (5 : ℝ) / 4 ∧ ∀ x, cos x ^ 2 - sin x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_cos_squared_minus_sin_l3642_364202


namespace NUMINAMATH_CALUDE_tv_set_selection_count_l3642_364260

def total_sets : ℕ := 9
def type_a_sets : ℕ := 4
def type_b_sets : ℕ := 5
def sets_to_select : ℕ := 3

theorem tv_set_selection_count :
  (Nat.choose total_sets sets_to_select) -
  (Nat.choose type_a_sets sets_to_select) -
  (Nat.choose type_b_sets sets_to_select) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_set_selection_count_l3642_364260


namespace NUMINAMATH_CALUDE_unique_a_value_l3642_364242

def A (a : ℤ) : Set ℤ := {1, 2, a + 3}
def B (a : ℤ) : Set ℤ := {a, 5}

theorem unique_a_value (a : ℤ) : A a ∪ B a = A a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3642_364242


namespace NUMINAMATH_CALUDE_bus_travel_time_l3642_364218

/-- Proves that a bus traveling 35 miles at a speed of 50 mph takes 42 minutes -/
theorem bus_travel_time (distance : ℝ) (speed : ℝ) (time_minutes : ℝ) : 
  distance = 35 → 
  speed = 50 → 
  time_minutes = (distance / speed) * 60 → 
  time_minutes = 42 := by
sorry

end NUMINAMATH_CALUDE_bus_travel_time_l3642_364218


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3642_364228

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^4 * (3 : ℝ)^x = 81 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3642_364228


namespace NUMINAMATH_CALUDE_det_A_squared_l3642_364235

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 7, 2]

theorem det_A_squared : (Matrix.det A)^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_l3642_364235


namespace NUMINAMATH_CALUDE_coeff_x_cubed_in_product_l3642_364248

def p (x : ℝ) : ℝ := x^5 - 4*x^3 + 3*x^2 - 2*x + 1
def q (x : ℝ) : ℝ := 3*x^3 - 2*x^2 + x + 5

theorem coeff_x_cubed_in_product (x : ℝ) :
  ∃ (a b c d e : ℝ), p x * q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + (p 0 * q 0) ∧ c = -10 :=
sorry

end NUMINAMATH_CALUDE_coeff_x_cubed_in_product_l3642_364248


namespace NUMINAMATH_CALUDE_honey_lasts_16_days_l3642_364232

/-- Represents the number of days Tabitha can enjoy honey in her tea -/
def honey_days : ℕ :=
  let evening_servings : ℕ := 2 * 2  -- 2 cups with 2 servings each
  let morning_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let afternoon_servings : ℕ := 1 * 1  -- 1 cup with 1 serving
  let daily_servings : ℕ := evening_servings + morning_servings + afternoon_servings
  let container_ounces : ℕ := 16
  let servings_per_ounce : ℕ := 6
  let total_servings : ℕ := container_ounces * servings_per_ounce
  total_servings / daily_servings

theorem honey_lasts_16_days : honey_days = 16 := by
  sorry

end NUMINAMATH_CALUDE_honey_lasts_16_days_l3642_364232


namespace NUMINAMATH_CALUDE_expression_value_l3642_364299

theorem expression_value (x y z : ℝ) (hx : x = 3) (hy : y = 2) (hz : z = 1) :
  3 * x^2 - 4 * y + 2 * z = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3642_364299


namespace NUMINAMATH_CALUDE_chicken_selling_price_l3642_364250

/-- Represents the problem of determining the selling price of chickens --/
theorem chicken_selling_price 
  (num_chickens : ℕ) 
  (profit : ℚ) 
  (feed_per_chicken : ℚ) 
  (feed_bag_weight : ℚ) 
  (feed_bag_cost : ℚ) :
  num_chickens = 50 →
  profit = 65 →
  feed_per_chicken = 2 →
  feed_bag_weight = 20 →
  feed_bag_cost = 2 →
  ∃ (selling_price : ℚ), selling_price = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_chicken_selling_price_l3642_364250


namespace NUMINAMATH_CALUDE_min_cases_for_shirley_order_l3642_364264

/-- Represents the number of boxes of each cookie type sold -/
structure CookiesSold where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Represents the composition of each case -/
structure CaseComposition where
  trefoils : Nat
  samoas : Nat
  thinMints : Nat

/-- Calculates the minimum number of cases needed to fulfill the orders -/
def minCasesNeeded (sold : CookiesSold) (composition : CaseComposition) : Nat :=
  max
    (((sold.trefoils + composition.trefoils - 1) / composition.trefoils) : Nat)
    (max
      (((sold.samoas + composition.samoas - 1) / composition.samoas) : Nat)
      (((sold.thinMints + composition.thinMints - 1) / composition.thinMints) : Nat))

theorem min_cases_for_shirley_order :
  let sold : CookiesSold := { trefoils := 54, samoas := 36, thinMints := 48 }
  let composition : CaseComposition := { trefoils := 4, samoas := 3, thinMints := 5 }
  minCasesNeeded sold composition = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_cases_for_shirley_order_l3642_364264


namespace NUMINAMATH_CALUDE_alice_savings_l3642_364213

/-- Alice's savings problem -/
theorem alice_savings (total_days : ℕ) (total_dimes : ℕ) (dime_value : ℚ) (daily_savings : ℚ) : 
  total_days = 40 →
  total_dimes = 4 →
  dime_value = 1/10 →
  daily_savings = (total_dimes : ℚ) * dime_value / total_days →
  daily_savings = 1/100 := by
  sorry

#check alice_savings

end NUMINAMATH_CALUDE_alice_savings_l3642_364213


namespace NUMINAMATH_CALUDE_temperature_difference_l3642_364211

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 11) 
  (h2 : lowest = -11) : 
  highest - lowest = 22 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3642_364211


namespace NUMINAMATH_CALUDE_perimeter_increase_is_237_point_5_percent_l3642_364201

/-- Represents the side length ratio between consecutive triangles -/
def ratio : ℝ := 1.5

/-- Calculates the percent increase in perimeter from the first to the fourth triangle -/
def perimeter_increase : ℝ :=
  (ratio^3 - 1) * 100

/-- Theorem stating that the percent increase in perimeter is 237.5% -/
theorem perimeter_increase_is_237_point_5_percent :
  ∃ ε > 0, |perimeter_increase - 237.5| < ε :=
sorry

end NUMINAMATH_CALUDE_perimeter_increase_is_237_point_5_percent_l3642_364201


namespace NUMINAMATH_CALUDE_average_pencils_per_box_l3642_364261

theorem average_pencils_per_box : 
  let pencil_counts : List Nat := [12, 14, 14, 15, 15, 15, 16, 16, 17, 18]
  let total_boxes : Nat := pencil_counts.length
  let total_pencils : Nat := pencil_counts.sum
  (total_pencils : ℚ) / total_boxes = 15.2 := by
  sorry

end NUMINAMATH_CALUDE_average_pencils_per_box_l3642_364261


namespace NUMINAMATH_CALUDE_lemonade_solution_water_amount_l3642_364221

/-- The amount of lemonade syrup in the solution -/
def lemonade_syrup : ℝ := 7

/-- The amount of solution removed and replaced with water -/
def removed_amount : ℝ := 2.1428571428571423

/-- The desired concentration of lemonade syrup after adjustment -/
def desired_concentration : ℝ := 0.20

/-- The amount of water in the original solution -/
def water_amount : ℝ := 25.857142857142854

theorem lemonade_solution_water_amount :
  (lemonade_syrup / (lemonade_syrup + water_amount + removed_amount) = desired_concentration) :=
by sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_amount_l3642_364221


namespace NUMINAMATH_CALUDE_complement_of_A_is_zero_l3642_364279

def A : Set ℤ := {x | |x| ≥ 1}

theorem complement_of_A_is_zero : 
  (Set.univ : Set ℤ) \ A = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_is_zero_l3642_364279


namespace NUMINAMATH_CALUDE_exactly_one_event_probability_l3642_364273

theorem exactly_one_event_probability (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_event_probability_l3642_364273


namespace NUMINAMATH_CALUDE_large_posters_count_l3642_364207

theorem large_posters_count (total : ℕ) (small_fraction : ℚ) (medium_fraction : ℚ) : 
  total = 50 →
  small_fraction = 2 / 5 →
  medium_fraction = 1 / 2 →
  (total : ℚ) * small_fraction + (total : ℚ) * medium_fraction + 5 = total :=
by
  sorry

end NUMINAMATH_CALUDE_large_posters_count_l3642_364207


namespace NUMINAMATH_CALUDE_two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l3642_364256

-- Define the basic geometric objects
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the geometric relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Statement ②
theorem two_lines_perpendicular_to_plane_are_parallel 
  (p : Plane) (l1 l2 : Line) :
  perpendicular l1 p → perpendicular l2 p → parallel l1 l2 := by sorry

-- Statement ③
theorem two_planes_perpendicular_to_line_are_parallel 
  (l : Line) (p1 p2 : Plane) :
  perpendicular_plane p1 l → perpendicular_plane p2 l → parallel_plane p1 p2 := by sorry

end NUMINAMATH_CALUDE_two_lines_perpendicular_to_plane_are_parallel_two_planes_perpendicular_to_line_are_parallel_l3642_364256


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3642_364210

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₂ + a₄ = 41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3642_364210


namespace NUMINAMATH_CALUDE_probability_of_four_boys_l3642_364233

open BigOperators Finset

theorem probability_of_four_boys (total_students : ℕ) (total_boys : ℕ) (selected_students : ℕ) :
  total_students = 15 →
  total_boys = 7 →
  selected_students = 10 →
  (Nat.choose total_boys 4 * Nat.choose (total_students - total_boys) (selected_students - 4)) /
  Nat.choose total_students selected_students =
  Nat.choose 7 4 * Nat.choose 8 6 / Nat.choose 15 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_four_boys_l3642_364233


namespace NUMINAMATH_CALUDE_complex_modulus_l3642_364297

theorem complex_modulus (z : ℂ) (i : ℂ) (h : i * i = -1) (eq : z / (1 + i) = 2 * i) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l3642_364297


namespace NUMINAMATH_CALUDE_vector_properties_l3642_364289

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 4) ∧
  ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2 = 2) ∧
  ((a.1 + b.1) * c.1 + (a.2 + b.2) * c.2 = 0) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l3642_364289


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l3642_364298

/-- Given polynomials f and g with degrees 3 and 6 respectively,
    the degree of f(x^2) · g(x^3) is 24. -/
theorem degree_of_composed_product (f g : Polynomial ℝ) 
  (hf : Polynomial.degree f = 3)
  (hg : Polynomial.degree g = 6) :
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l3642_364298


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3642_364266

/-- Given a rectangular pen with perimeter 60 feet and one side length at least 15 feet,
    the maximum possible area is 225 square feet. -/
theorem max_area_rectangular_pen :
  ∀ (x y : ℝ),
    x > 0 ∧ y > 0 →
    x + y = 30 →
    (x ≥ 15 ∨ y ≥ 15) →
    x * y ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3642_364266


namespace NUMINAMATH_CALUDE_ticket_cost_is_18_l3642_364292

/-- The cost of a single ticket to an amusement park -/
def ticket_cost (total_people : ℕ) (snack_cost : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - total_people * snack_cost) / total_people

/-- Proof that the ticket cost is $18 given the problem conditions -/
theorem ticket_cost_is_18 :
  ticket_cost 4 5 92 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ticket_cost_is_18_l3642_364292


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l3642_364272

theorem arcade_spending_fraction (allowance : ℚ) (remaining : ℚ) 
  (h1 : allowance = 480 / 100)
  (h2 : remaining = 128 / 100)
  (h3 : remaining = (2/3) * (1 - (arcade_fraction : ℚ)) * allowance) :
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l3642_364272


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_l3642_364225

/-- Two lines with slopes that are the roots of x^2 - 3x - 1 = 0 are perpendicular -/
theorem perpendicular_lines_from_quadratic (k₁ k₂ : ℝ) : 
  k₁^2 - 3*k₁ - 1 = 0 → k₂^2 - 3*k₂ - 1 = 0 → k₁ ≠ k₂ → k₁ * k₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_l3642_364225


namespace NUMINAMATH_CALUDE_pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3642_364238

/-- A polygon in a plane -/
class Polygon :=
  (sides : ℕ)

/-- A pentagon is a polygon with 5 sides -/
def Pentagon : Polygon :=
  { sides := 5 }

/-- A triangle is a polygon with 3 sides -/
def Triangle : Polygon :=
  { sides := 3 }

/-- A quadrilateral is a polygon with 4 sides -/
def Quadrilateral : Polygon :=
  { sides := 4 }

/-- The maximum number of intersection points between the sides of two polygons -/
def maxIntersections (P Q : Polygon) : ℕ := sorry

/-- Theorem: Maximum intersections between a pentagon and a triangle -/
theorem pentagon_triangle_intersections :
  maxIntersections Pentagon Triangle = 10 := by sorry

/-- Theorem: Maximum intersections between a pentagon and a quadrilateral -/
theorem pentagon_quadrilateral_intersections :
  maxIntersections Pentagon Quadrilateral = 16 := by sorry

end NUMINAMATH_CALUDE_pentagon_triangle_intersections_pentagon_quadrilateral_intersections_l3642_364238


namespace NUMINAMATH_CALUDE_sundae_price_l3642_364240

/-- Given a caterer's order of ice-cream bars and sundaes, calculate the price of each sundae. -/
theorem sundae_price
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (ice_cream_bar_price : ℚ)
  (h1 : num_ice_cream_bars = 225)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 200)
  (h4 : ice_cream_bar_price = 0.60) :
  (total_price - (↑num_ice_cream_bars * ice_cream_bar_price)) / ↑num_sundaes = 0.52 := by
  sorry

end NUMINAMATH_CALUDE_sundae_price_l3642_364240


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l3642_364265

/-- Given a rectangle with sides measured with errors, calculate the error percent in the area --/
theorem rectangle_area_error_percent (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let measured_length := 1.05 * L
  let measured_width := 0.96 * W
  let actual_area := L * W
  let measured_area := measured_length * measured_width
  let error := measured_area - actual_area
  let error_percent := (error / actual_area) * 100
  error_percent = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l3642_364265


namespace NUMINAMATH_CALUDE_sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l3642_364252

theorem sum_of_three_squares_divisible_by_three_to_not_divisible_by_three
  (N : ℕ) (a b c : ℤ) (h1 : ∃ (a b c : ℤ), N = a^2 + b^2 + c^2)
  (h2 : 3 ∣ a) (h3 : 3 ∣ b) (h4 : 3 ∣ c) :
  ∃ (x y z : ℤ), N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_divisible_by_three_to_not_divisible_by_three_l3642_364252
