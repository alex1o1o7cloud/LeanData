import Mathlib

namespace NUMINAMATH_GPT_problem1_problem2_l1299_129946

section Problems

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x^3 - a * x + 1

-- Problem 1: Tangent line problem for a = 1
def tangent_line_eqn (x : ℝ) : Prop :=
  let a := 1
  let f := f a
  (∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b)

-- Problem 2: Minimum value problem
def min_value_condition (a : ℝ) : Prop :=
  f a (1 / 4) = (11 / 12)

theorem problem1 : tangent_line_eqn 0 :=
  sorry

theorem problem2 : min_value_condition (1 / 4) :=
  sorry

end Problems

end NUMINAMATH_GPT_problem1_problem2_l1299_129946


namespace NUMINAMATH_GPT_bobs_income_after_changes_l1299_129958

variable (initial_salary : ℝ) (february_increase_rate : ℝ) (march_reduction_rate : ℝ)

def february_salary (initial_salary : ℝ) (increase_rate : ℝ) : ℝ :=
  initial_salary * (1 + increase_rate)

def march_salary (february_salary : ℝ) (reduction_rate : ℝ) : ℝ :=
  february_salary * (1 - reduction_rate)

theorem bobs_income_after_changes (h1 : initial_salary = 2750)
  (h2 : february_increase_rate = 0.15)
  (h3 : march_reduction_rate = 0.10) :
  march_salary (february_salary initial_salary february_increase_rate) march_reduction_rate = 2846.25 := 
sorry

end NUMINAMATH_GPT_bobs_income_after_changes_l1299_129958


namespace NUMINAMATH_GPT_segments_divide_ratio_3_to_1_l1299_129930

-- Define points and segments
structure Point :=
  (x : ℝ) (y : ℝ)

structure Segment :=
  (A B : Point)

-- Define T-shaped figure consisting of 22 unit squares
noncomputable def T_shaped_figure : ℕ := 22

-- Define line p passing through point V
structure Line :=
  (p : Point → Point)
  (passes_through : Point)

-- Define equal areas condition
def equal_areas (white_area gray_area : ℝ) : Prop := 
  white_area = gray_area

-- Define the problem
theorem segments_divide_ratio_3_to_1
  (AB : Segment)
  (V : Point)
  (white_area gray_area : ℝ)
  (p : Line)
  (h1 : equal_areas white_area gray_area)
  (h2 : T_shaped_figure = 22)
  (h3 : p.passes_through = V) :
  ∃ (C : Point), (p.p AB.A = C) ∧ ((abs (AB.A.x - C.x)) / (abs (C.x - AB.B.x))) = 3 :=
sorry

end NUMINAMATH_GPT_segments_divide_ratio_3_to_1_l1299_129930


namespace NUMINAMATH_GPT_sum_of_fractions_and_decimal_l1299_129971

theorem sum_of_fractions_and_decimal :
  (6 / 5 : ℝ) + (1 / 10 : ℝ) + 1.56 = 2.86 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_and_decimal_l1299_129971


namespace NUMINAMATH_GPT_find_ab_l1299_129905

-- Define the statement to be proven
theorem find_ab (a b : ℕ) (h1 : (a + b) % 3 = 2)
                           (h2 : b % 5 = 3)
                           (h3 : (b - a) % 11 = 1) :
  10 * a + b = 23 := 
sorry

end NUMINAMATH_GPT_find_ab_l1299_129905


namespace NUMINAMATH_GPT_arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l1299_129978

-- Statement for Problem 1: Number of ways to draw five numbers forming an arithmetic progression
theorem arithmetic_progression_five_numbers :
  ∃ (N : ℕ), N = 968 :=
  sorry

-- Statement for Problem 2: Number of ways to draw four numbers forming an arithmetic progression with a fifth number being arbitrary
theorem arithmetic_progression_four_numbers :
  ∃ (N : ℕ), N = 111262 :=
  sorry

end NUMINAMATH_GPT_arithmetic_progression_five_numbers_arithmetic_progression_four_numbers_l1299_129978


namespace NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_8_l1299_129931

theorem product_of_consecutive_even_numbers_divisible_by_8 (n : ℤ) : 
  8 ∣ (2 * n) * (2 * n + 2) :=
by sorry

end NUMINAMATH_GPT_product_of_consecutive_even_numbers_divisible_by_8_l1299_129931


namespace NUMINAMATH_GPT_problem_l1299_129924

-- Definitions according to the conditions
def red_balls : ℕ := 1
def black_balls (n : ℕ) : ℕ := n
def total_balls (n : ℕ) : ℕ := red_balls + black_balls n
noncomputable def probability_red (n : ℕ) : ℚ := (red_balls : ℚ) / (total_balls n : ℚ)
noncomputable def variance (n : ℕ) : ℚ := (black_balls n : ℚ) / (total_balls n : ℚ)^2

-- The theorem we want to prove
theorem problem (n : ℕ) (h : 0 < n) : 
  (∀ m : ℕ, n < m → probability_red m < probability_red n) ∧ 
  (∀ m : ℕ, n < m → variance m < variance n) :=
sorry

end NUMINAMATH_GPT_problem_l1299_129924


namespace NUMINAMATH_GPT_scarves_per_yarn_correct_l1299_129956

def scarves_per_yarn (total_yarns total_scarves : ℕ) : ℕ :=
  total_scarves / total_yarns

theorem scarves_per_yarn_correct :
  scarves_per_yarn (2 + 6 + 4) 36 = 3 :=
by
  sorry

end NUMINAMATH_GPT_scarves_per_yarn_correct_l1299_129956


namespace NUMINAMATH_GPT_inequality_proof_l1299_129935

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a ^ 2 + 1)) + (b / Real.sqrt (b ^ 2 + 1)) + (c / Real.sqrt (c ^ 2 + 1)) ≤ (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1299_129935


namespace NUMINAMATH_GPT_find_range_of_a_l1299_129965

noncomputable def range_of_a (a : ℝ) : Prop :=
∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ (Real.pi / 2)) → 
  let α := (x + 3, x)
  let β := (2 * Real.sin θ * Real.cos θ, a * Real.sin θ + a * Real.cos θ)
  let sum := (α.1 + β.1, α.2 + β.2)
  (sum.1^2 + sum.2^2)^(1/2) ≥ Real.sqrt 2

theorem find_range_of_a : range_of_a a ↔ (a ≤ 1 ∨ a ≥ 5) :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l1299_129965


namespace NUMINAMATH_GPT_hoses_fill_time_l1299_129951

noncomputable def time_to_fill_pool {P A B C : ℝ} (h₁ : A + B = P / 3) (h₂ : A + C = P / 4) (h₃ : B + C = P / 5) : ℝ :=
  (120 / 47 : ℝ)

theorem hoses_fill_time {P A B C : ℝ} 
  (h₁ : A + B = P / 3) 
  (h₂ : A + C = P / 4) 
  (h₃ : B + C = P / 5) 
  : time_to_fill_pool h₁ h₂ h₃ = (120 / 47 : ℝ) :=
sorry

end NUMINAMATH_GPT_hoses_fill_time_l1299_129951


namespace NUMINAMATH_GPT_initial_cards_l1299_129906

theorem initial_cards (taken left initial : ℕ) (h1 : taken = 59) (h2 : left = 17) (h3 : initial = left + taken) : initial = 76 :=
by
  sorry

end NUMINAMATH_GPT_initial_cards_l1299_129906


namespace NUMINAMATH_GPT_weight_of_new_person_l1299_129907

theorem weight_of_new_person {avg_increase : ℝ} (n : ℕ) (p : ℝ) (w : ℝ) (h : n = 8) (h1 : avg_increase = 2.5) (h2 : w = 67):
  p = 87 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1299_129907


namespace NUMINAMATH_GPT_range_of_a_l1299_129936

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a^2 ≤ 0 → false) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1299_129936


namespace NUMINAMATH_GPT_dennis_rocks_left_l1299_129974

-- Definitions based on conditions:
def initial_rocks : ℕ := 10
def rocks_eaten_by_fish (initial : ℕ) : ℕ := initial / 2
def rocks_spat_out_by_fish : ℕ := 2

-- Total rocks left:
def total_rocks_left (initial : ℕ) (spat_out : ℕ) : ℕ :=
  (rocks_eaten_by_fish initial) + spat_out

-- Statement to be proved:
theorem dennis_rocks_left : total_rocks_left initial_rocks rocks_spat_out_by_fish = 7 :=
by
  -- Conclusion by calculation (Proved in steps)
  sorry

end NUMINAMATH_GPT_dennis_rocks_left_l1299_129974


namespace NUMINAMATH_GPT_puppies_left_l1299_129914

namespace AlyssaPuppies

def initPuppies : ℕ := 12
def givenAway : ℕ := 7
def remainingPuppies : ℕ := 5

theorem puppies_left (initPuppies givenAway remainingPuppies : ℕ) : 
  initPuppies - givenAway = remainingPuppies :=
by
  sorry

end AlyssaPuppies

end NUMINAMATH_GPT_puppies_left_l1299_129914


namespace NUMINAMATH_GPT_cube_roll_sums_l1299_129960

def opposite_faces_sum_to_seven (a b : ℕ) : Prop := a + b = 7

def valid_cube_faces : Prop := 
  opposite_faces_sum_to_seven 1 6 ∧
  opposite_faces_sum_to_seven 2 5 ∧
  opposite_faces_sum_to_seven 3 4

def max_min_sums : ℕ × ℕ := (342, 351)

theorem cube_roll_sums (faces_sum_seven : valid_cube_faces) : 
  ∃ cube_sums : ℕ × ℕ, cube_sums = max_min_sums := sorry

end NUMINAMATH_GPT_cube_roll_sums_l1299_129960


namespace NUMINAMATH_GPT_remainder_sum_mod9_l1299_129918

theorem remainder_sum_mod9 :
  ((2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_sum_mod9_l1299_129918


namespace NUMINAMATH_GPT_product_of_ys_l1299_129969

theorem product_of_ys (x y : ℤ) (h1 : x^3 + y^2 - 3 * y + 1 < 0)
                                     (h2 : 3 * x^3 - y^2 + 3 * y > 0) : 
  (y = 1 ∨ y = 2) → (1 * 2 = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_ys_l1299_129969


namespace NUMINAMATH_GPT_coin_probability_not_unique_l1299_129961

variables (p : ℝ) (w : ℝ)
def binomial_prob := 10 * p^3 * (1 - p)^2

theorem coin_probability_not_unique (h : binomial_prob p = 144 / 625) : 
  ∃ p1 p2, p1 ≠ p2 ∧ binomial_prob p1 = 144 / 625 ∧ binomial_prob p2 = 144 / 625 :=
by 
  sorry

end NUMINAMATH_GPT_coin_probability_not_unique_l1299_129961


namespace NUMINAMATH_GPT_contractor_initial_hire_l1299_129990

theorem contractor_initial_hire :
  ∃ (P : ℕ), 
    (∀ (total_work : ℝ), 
      (P * 20 = (1/4) * total_work) ∧ 
      ((P - 2) * 75 = (3/4) * total_work)) → 
    P = 10 :=
by
  sorry

end NUMINAMATH_GPT_contractor_initial_hire_l1299_129990


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_l1299_129908

theorem rhombus_longer_diagonal 
  (a b : ℝ) 
  (h₁ : a = 61) 
  (h₂ : b = 44) :
  ∃ d₂ : ℝ, d₂ = 2 * Real.sqrt (a * a - (b / 2) * (b / 2)) :=
sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_l1299_129908


namespace NUMINAMATH_GPT_prime_gt_3_divides_exp_l1299_129917

theorem prime_gt_3_divides_exp (p : ℕ) (hprime : Nat.Prime p) (hgt3 : p > 3) :
  42 * p ∣ 3^p - 2^p - 1 :=
sorry

end NUMINAMATH_GPT_prime_gt_3_divides_exp_l1299_129917


namespace NUMINAMATH_GPT_breadth_remains_the_same_l1299_129963

variable (L B : ℝ)

theorem breadth_remains_the_same 
  (A : ℝ) (hA : A = L * B) 
  (L_half : ℝ) (hL_half : L_half = L / 2) 
  (B' : ℝ)
  (A' : ℝ) (hA' : A' = L_half * B') 
  (hA_change : A' = 0.5 * A) : 
  B' = B :=
  sorry

end NUMINAMATH_GPT_breadth_remains_the_same_l1299_129963


namespace NUMINAMATH_GPT_sine_of_negative_90_degrees_l1299_129999

theorem sine_of_negative_90_degrees : Real.sin (-(Real.pi / 2)) = -1 := 
sorry

end NUMINAMATH_GPT_sine_of_negative_90_degrees_l1299_129999


namespace NUMINAMATH_GPT_parabola_intersection_probability_correct_l1299_129909

noncomputable def parabola_intersection_probability : ℚ := sorry

theorem parabola_intersection_probability_correct :
  parabola_intersection_probability = 209 / 216 := sorry

end NUMINAMATH_GPT_parabola_intersection_probability_correct_l1299_129909


namespace NUMINAMATH_GPT_find_a_and_b_l1299_129943

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x + 1))

theorem find_a_and_b
  (a b : ℝ)
  (h1 : a < b)
  (h2 : f a = f ((- (b + 1)) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = - 2 / 5 ∧ b = - 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_and_b_l1299_129943


namespace NUMINAMATH_GPT_find_number_l1299_129911

theorem find_number (x N : ℕ) (h₁ : x = 32) (h₂ : N - (23 - (15 - x)) = (12 * 2 / 1 / 2)) : N = 88 :=
sorry

end NUMINAMATH_GPT_find_number_l1299_129911


namespace NUMINAMATH_GPT_smallest_egg_count_l1299_129933

theorem smallest_egg_count : ∃ n : ℕ, n > 100 ∧ n % 12 = 10 ∧ n = 106 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_egg_count_l1299_129933


namespace NUMINAMATH_GPT_lcm_12_20_correct_l1299_129994

def lcm_12_20_is_60 : Nat := Nat.lcm 12 20

theorem lcm_12_20_correct : Nat.lcm 12 20 = 60 := by
  -- assumed factorization conditions as prerequisites
  have h₁ : Nat.primeFactors 12 = {2, 3} := sorry
  have h₂ : Nat.primeFactors 20 = {2, 5} := sorry
  -- the main proof goal
  exact sorry

end NUMINAMATH_GPT_lcm_12_20_correct_l1299_129994


namespace NUMINAMATH_GPT_chewbacca_gum_pieces_l1299_129922

theorem chewbacca_gum_pieces (y : ℚ)
  (h1 : ∀ x : ℚ, x ≠ 0 → (15 - y) = 15 * (25 + 2 * y) / 25) :
  y = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_chewbacca_gum_pieces_l1299_129922


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1299_129954

noncomputable def calculate_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a ^ 2 - b ^ 2)
  c / a

theorem eccentricity_of_ellipse : 
  (calculate_eccentricity 5 4) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1299_129954


namespace NUMINAMATH_GPT_half_of_number_l1299_129975

theorem half_of_number (N : ℝ)
  (h1 : (4 / 15) * (5 / 7) * N = (4 / 9) * (2 / 5) * N + 8) : 
  (N / 2) = 315 := 
sorry

end NUMINAMATH_GPT_half_of_number_l1299_129975


namespace NUMINAMATH_GPT_distance_from_neg2_l1299_129937

theorem distance_from_neg2 (x : ℝ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 := 
by sorry

end NUMINAMATH_GPT_distance_from_neg2_l1299_129937


namespace NUMINAMATH_GPT_machine_present_value_l1299_129925

theorem machine_present_value
  (rate_of_decay : ℝ) (n_periods : ℕ) (final_value : ℝ) (initial_value : ℝ)
  (h_decay : rate_of_decay = 0.25)
  (h_periods : n_periods = 2)
  (h_final_value : final_value = 225) :
  initial_value = 400 :=
by
  -- The proof would go here. 
  sorry

end NUMINAMATH_GPT_machine_present_value_l1299_129925


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1299_129926

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1299_129926


namespace NUMINAMATH_GPT_function_relation4_l1299_129993

open Set

section
  variable (M : Set ℤ) (N : Set ℤ)

  def relation1 (x : ℤ) := x ^ 2
  def relation2 (x : ℤ) := x + 1
  def relation3 (x : ℤ) := x - 1
  def relation4 (x : ℤ) := abs x

  theorem function_relation4 : 
    M = {-1, 1, 2, 4} →
    N = {1, 2, 4} →
    (∀ x ∈ M, relation4 x ∈ N) :=
  by
    intros hM hN
    simp [relation4]
    sorry
end

end NUMINAMATH_GPT_function_relation4_l1299_129993


namespace NUMINAMATH_GPT_quotient_product_larger_integer_l1299_129932

theorem quotient_product_larger_integer
  (x y : ℕ)
  (h1 : y / x = 7 / 3)
  (h2 : x * y = 189)
  : y = 21 := 
sorry

end NUMINAMATH_GPT_quotient_product_larger_integer_l1299_129932


namespace NUMINAMATH_GPT_fishing_probability_correct_l1299_129992

-- Definitions for probabilities
def P_sunny : ℝ := 0.3
def P_rainy : ℝ := 0.5
def P_cloudy : ℝ := 0.2

def P_fishing_given_sunny : ℝ := 0.7
def P_fishing_given_rainy : ℝ := 0.3
def P_fishing_given_cloudy : ℝ := 0.5

-- The total probability function
def P_fishing : ℝ :=
  P_sunny * P_fishing_given_sunny +
  P_rainy * P_fishing_given_rainy +
  P_cloudy * P_fishing_given_cloudy

theorem fishing_probability_correct : P_fishing = 0.46 :=
by 
  sorry -- Proof goes here

end NUMINAMATH_GPT_fishing_probability_correct_l1299_129992


namespace NUMINAMATH_GPT_expected_carrot_yield_l1299_129980

-- Condition definitions
def num_steps_width : ℕ := 16
def num_steps_length : ℕ := 22
def step_length : ℝ := 1.75
def avg_yield_per_sqft : ℝ := 0.75

-- Theorem statement
theorem expected_carrot_yield : 
  (num_steps_width * step_length) * (num_steps_length * step_length) * avg_yield_per_sqft = 808.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_carrot_yield_l1299_129980


namespace NUMINAMATH_GPT_Tammy_earnings_3_weeks_l1299_129940

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end NUMINAMATH_GPT_Tammy_earnings_3_weeks_l1299_129940


namespace NUMINAMATH_GPT_neg_prop1_true_neg_prop2_false_l1299_129913

-- Proposition 1: The logarithm of a positive number is always positive
def prop1 : Prop := ∀ x : ℝ, x > 0 → Real.log x > 0

-- Negation of Proposition 1: There exists a positive number whose logarithm is not positive
def neg_prop1 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0

-- Proposition 2: For all x in the set of integers Z, the last digit of x^2 is not 3
def prop2 : Prop := ∀ x : ℤ, (x * x % 10 ≠ 3)

-- Negation of Proposition 2: There exists an x in the set of integers Z such that the last digit of x^2 is 3
def neg_prop2 : Prop := ∃ x : ℤ, (x * x % 10 = 3)

-- Proof that the negation of Proposition 1 is true
theorem neg_prop1_true : neg_prop1 := 
  by sorry

-- Proof that the negation of Proposition 2 is false
theorem neg_prop2_false : ¬ neg_prop2 := 
  by sorry

end NUMINAMATH_GPT_neg_prop1_true_neg_prop2_false_l1299_129913


namespace NUMINAMATH_GPT_walt_age_l1299_129949

theorem walt_age (T W : ℕ) 
  (h1 : T = 3 * W)
  (h2 : T + 12 = 2 * (W + 12)) : 
  W = 12 :=
by
  sorry

end NUMINAMATH_GPT_walt_age_l1299_129949


namespace NUMINAMATH_GPT_probability_of_event_l1299_129989

noncomputable def drawing_probability : ℚ := 
  let total_outcomes := 81
  let successful_outcomes :=
    (9 + 9 + 9 + 9 + 9 + 7 + 5 + 3 + 1)
  successful_outcomes / total_outcomes

theorem probability_of_event :
  drawing_probability = 61 / 81 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_event_l1299_129989


namespace NUMINAMATH_GPT_find_x10_l1299_129929

theorem find_x10 (x : ℕ → ℝ) :
  x 1 = 1 ∧ x 2 = 1 ∧ (∀ n ≥ 2, x (n + 1) = (x n * x (n - 1)) / (x n + x (n - 1))) →
  x 10 = 1 / 55 :=
by sorry

end NUMINAMATH_GPT_find_x10_l1299_129929


namespace NUMINAMATH_GPT_problem1_problem2_l1299_129964

-- Define the first problem as a proof statement in Lean
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 25 → (x = 7 ∨ x = -3) := sorry

-- Define the second problem as a proof statement in Lean
theorem problem2 (x : ℝ) : (x - 5) ^ 2 = 2 * (5 - x) → (x = 5 ∨ x = 3) := sorry

end NUMINAMATH_GPT_problem1_problem2_l1299_129964


namespace NUMINAMATH_GPT_part_I_part_II_l1299_129977

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -4 ∨ x > -2}
def C (m : ℝ) : Set ℝ := {x | 3 - 2 * m ≤ x ∧ x ≤ 2 + m}
def D : Set ℝ := {y | y < -6 ∨ y > -5}

theorem part_I (m : ℝ) : (∀ x, x ∈ A ∧ x ∈ B → x ∈ C m) → m ≥ 5 / 2 :=
sorry

theorem part_II (m : ℝ) : 
  (B ∪ (C m) = Set.univ) ∧ 
  (C m ⊆ D) → 
  7 / 2 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1299_129977


namespace NUMINAMATH_GPT_clock_hands_angle_seventy_degrees_l1299_129921

theorem clock_hands_angle_seventy_degrees (t : ℝ) (h : t ≥ 0 ∧ t ≤ 60):
    let hour_angle := 210 + 30 * (t / 60)
    let minute_angle := 360 * (t / 60)
    let angle := abs (hour_angle - minute_angle)
    (angle = 70 ∨ angle = 290) ↔ (t = 25 ∨ t = 52) :=
by apply sorry

end NUMINAMATH_GPT_clock_hands_angle_seventy_degrees_l1299_129921


namespace NUMINAMATH_GPT_correct_average_of_15_numbers_l1299_129995

theorem correct_average_of_15_numbers
  (initial_average : ℝ)
  (num_numbers : ℕ)
  (incorrect1 incorrect2 correct1 correct2 : ℝ)
  (initial_average_eq : initial_average = 37)
  (num_numbers_eq : num_numbers = 15)
  (incorrect1_eq : incorrect1 = 52)
  (incorrect2_eq : incorrect2 = 39)
  (correct1_eq : correct1 = 64)
  (correct2_eq : correct2 = 27) :
  (initial_average * num_numbers - incorrect1 - incorrect2 + correct1 + correct2) / num_numbers = 37 :=
by
  rw [initial_average_eq, num_numbers_eq, incorrect1_eq, incorrect2_eq, correct1_eq, correct2_eq]
  sorry

end NUMINAMATH_GPT_correct_average_of_15_numbers_l1299_129995


namespace NUMINAMATH_GPT_trigonometric_identity_l1299_129991

theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (-1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1299_129991


namespace NUMINAMATH_GPT_twenty_twenty_third_term_l1299_129988

def sequence_denominator (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_numerator_pos (n : ℕ) : ℕ :=
  (n + 1) / 2

def sequence_numerator_neg (n : ℕ) : ℤ :=
  -((n + 1) / 2 : ℤ)

def sequence_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then 
    (sequence_numerator_pos n) / (sequence_denominator n) 
  else 
    (sequence_numerator_neg n : ℚ) / (sequence_denominator n)

theorem twenty_twenty_third_term :
  sequence_term 2023 = 1012 / 4045 := 
sorry

end NUMINAMATH_GPT_twenty_twenty_third_term_l1299_129988


namespace NUMINAMATH_GPT_systematic_sampling_correct_l1299_129984

-- Conditions as definitions
def total_bags : ℕ := 50
def num_samples : ℕ := 5
def interval (total num : ℕ) : ℕ := total / num
def correct_sequence : List ℕ := [5, 15, 25, 35, 45]

-- Statement
theorem systematic_sampling_correct :
  ∃ l : List ℕ, (l.length = num_samples) ∧ 
               (∀ i ∈ l, i ≤ total_bags) ∧
               (∀ i j, i < j → l.indexOf i < l.indexOf j → j - i = interval total_bags num_samples) ∧
               l = correct_sequence :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_correct_l1299_129984


namespace NUMINAMATH_GPT_near_square_qoutient_l1299_129996

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end NUMINAMATH_GPT_near_square_qoutient_l1299_129996


namespace NUMINAMATH_GPT_inequality_positive_real_numbers_l1299_129998

theorem inequality_positive_real_numbers
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_condition : a * b + b * c + c * a = 1) :
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ (3 / 2) :=
  sorry

end NUMINAMATH_GPT_inequality_positive_real_numbers_l1299_129998


namespace NUMINAMATH_GPT_zilla_savings_deposit_l1299_129953

-- Definitions based on problem conditions
def monthly_earnings (E : ℝ) : Prop :=
  0.07 * E = 133

def tax_deduction (E : ℝ) : ℝ :=
  E - 0.10 * E

def expenditure (earnings : ℝ) : ℝ :=
  133 +  0.30 * earnings + 0.20 * earnings + 0.12 * earnings

def savings_deposit (remaining_earnings : ℝ) : ℝ :=
  0.15 * remaining_earnings

-- The final proof statement
theorem zilla_savings_deposit (E : ℝ) (total_spent : ℝ) (earnings_after_tax : ℝ) (remaining_earnings : ℝ) : 
  monthly_earnings E →
  tax_deduction E = earnings_after_tax →
  expenditure earnings_after_tax = total_spent →
  remaining_earnings = earnings_after_tax - total_spent →
  savings_deposit remaining_earnings = 77.52 :=
by
  intros
  sorry

end NUMINAMATH_GPT_zilla_savings_deposit_l1299_129953


namespace NUMINAMATH_GPT_sin_sum_less_than_zero_l1299_129985

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ 0 < γ ∧ γ < Real.pi / 2

theorem sin_sum_less_than_zero (n : ℕ) :
  (∀ (α β γ : ℝ), is_acute_triangle α β γ → (Real.sin (n * α) + Real.sin (n * β) + Real.sin (n * γ) < 0)) ↔ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_sum_less_than_zero_l1299_129985


namespace NUMINAMATH_GPT_non_neg_int_solutions_l1299_129944

theorem non_neg_int_solutions (n : ℕ) (a b : ℤ) :
  n^2 = a + b ∧ n^3 = a^2 + b^2 → n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_non_neg_int_solutions_l1299_129944


namespace NUMINAMATH_GPT_smallest_n_fact_expr_l1299_129904

theorem smallest_n_fact_expr : ∃ n : ℕ, (∀ m : ℕ, m = 6 → n! = (n - 4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1)) ∧ n = 23 := by
  sorry

end NUMINAMATH_GPT_smallest_n_fact_expr_l1299_129904


namespace NUMINAMATH_GPT_find_number_of_valid_polynomials_l1299_129927

noncomputable def number_of_polynomials_meeting_constraints : Nat :=
  sorry

theorem find_number_of_valid_polynomials : number_of_polynomials_meeting_constraints = 11 :=
  sorry

end NUMINAMATH_GPT_find_number_of_valid_polynomials_l1299_129927


namespace NUMINAMATH_GPT_oil_spent_amount_l1299_129919

theorem oil_spent_amount :
  ∀ (P R M : ℝ), R = 25 → P = (R / 0.75) → ((M / R) - (M / P) = 5) → M = 500 :=
by
  intros P R M hR hP hOil
  sorry

end NUMINAMATH_GPT_oil_spent_amount_l1299_129919


namespace NUMINAMATH_GPT_intersection_eq_l1299_129968

def A : Set ℝ := {x : ℝ | (x - 2) / (x + 3) ≤ 0 }
def B : Set ℝ := {x : ℝ | x ≤ 1 }

theorem intersection_eq : A ∩ B = {x : ℝ | -3 < x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_intersection_eq_l1299_129968


namespace NUMINAMATH_GPT_distance_between_lines_l1299_129973

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines : 
  ∀ (x y : ℝ), line1 x y → line2 x y → (|1 - (-1)| / Real.sqrt (1^2 + (-1)^2)) = Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_lines_l1299_129973


namespace NUMINAMATH_GPT_value_of_a2_l1299_129967

theorem value_of_a2 (a0 a1 a2 a3 a4 : ℝ) (x : ℝ) 
  (h : x^4 = a0 + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3 + a4 * (x - 2)^4) :
  a2 = 24 :=
sorry

end NUMINAMATH_GPT_value_of_a2_l1299_129967


namespace NUMINAMATH_GPT_train_length_l1299_129915

theorem train_length (x : ℕ) (h1 : (310 + x) / 18 = x / 8) : x = 248 :=
  sorry

end NUMINAMATH_GPT_train_length_l1299_129915


namespace NUMINAMATH_GPT_dogs_not_liking_any_food_l1299_129987

-- Declare variables
variable (n w s ws c cs : ℕ)

-- Define problem conditions
def total_dogs := n
def dogs_like_watermelon := w
def dogs_like_salmon := s
def dogs_like_watermelon_and_salmon := ws
def dogs_like_chicken := c
def dogs_like_chicken_and_salmon_but_not_watermelon := cs

-- Define the statement proving the number of dogs that do not like any of the three foods
theorem dogs_not_liking_any_food : 
  n = 75 → 
  w = 15 → 
  s = 54 → 
  ws = 12 → 
  c = 20 → 
  cs = 7 → 
  (75 - ((w - ws) + (s - ws - cs) + (c - cs) + ws + cs) = 5) :=
by
  intros _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_dogs_not_liking_any_food_l1299_129987


namespace NUMINAMATH_GPT_trigonometric_simplification_l1299_129939

open Real

theorem trigonometric_simplification (α : ℝ) :
  (3.4113 * sin α * cos (3 * α) + 9 * sin α * cos α - sin (3 * α) * cos (3 * α) - 3 * sin (3 * α) * cos α) = 
  2 * sin (2 * α)^3 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_trigonometric_simplification_l1299_129939


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_45_l1299_129962

theorem smallest_positive_multiple_of_45 : ∃ x : ℕ, x > 0 ∧ (∃ (n : ℕ), n > 0 ∧ x = 45 * n) ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_45_l1299_129962


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1299_129957

section intersection_proof

-- Definitions of the sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x + 1 > 0}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := 
by {
  sorry
}

end intersection_proof

end NUMINAMATH_GPT_intersection_of_A_and_B_l1299_129957


namespace NUMINAMATH_GPT_part1_part2_l1299_129972

-- Define the quadratic equation
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation ax^2 + bx + c
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Part (1): Prove range for m
theorem part1 (m : ℝ) : (∃ x : ℝ, quadratic 1 (-5) m x = 0) ↔ m ≤ 25 / 4 := sorry

-- Part (2): Prove value of m given conditions on roots
theorem part2 (x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : 3 * x1 - 2 * x2 = 5) : 
  m = x1 * x2 → m = 6 := sorry

end NUMINAMATH_GPT_part1_part2_l1299_129972


namespace NUMINAMATH_GPT_no_prime_pair_summing_to_53_l1299_129950

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end NUMINAMATH_GPT_no_prime_pair_summing_to_53_l1299_129950


namespace NUMINAMATH_GPT_john_total_distance_l1299_129923

theorem john_total_distance :
  let speed := 55 -- John's speed in mph
  let time1 := 2 -- Time before lunch in hours
  let time2 := 3 -- Time after lunch in hours
  let distance1 := speed * time1 -- Distance before lunch
  let distance2 := speed * time2 -- Distance after lunch
  let total_distance := distance1 + distance2 -- Total distance

  total_distance = 275 :=
by
  sorry

end NUMINAMATH_GPT_john_total_distance_l1299_129923


namespace NUMINAMATH_GPT_train_length_is_400_l1299_129934

-- Conditions from a)
def train_speed_kmph : ℕ := 180
def crossing_time_sec : ℕ := 8

-- The corresponding length in meters
def length_of_train : ℕ := 400

-- The problem statement to prove
theorem train_length_is_400 :
  (train_speed_kmph * 1000 / 3600) * crossing_time_sec = length_of_train := by
  -- Proof is skipped as per the requirement
  sorry

end NUMINAMATH_GPT_train_length_is_400_l1299_129934


namespace NUMINAMATH_GPT_complementary_angle_difference_l1299_129901

theorem complementary_angle_difference (x : ℝ) (h : 3 * x + 5 * x = 90) : 
    abs ((5 * x) - (3 * x)) = 22.5 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_complementary_angle_difference_l1299_129901


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1299_129959

theorem distance_between_parallel_lines 
  (r : ℝ) (d : ℝ) 
  (h1 : 3 * (2 * r^2) = 722 + (19 / 4) * d^2) 
  (h2 : 3 * (2 * r^2) = 578 + (153 / 4) * d^2) : 
  d = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1299_129959


namespace NUMINAMATH_GPT_square_root_problem_l1299_129902

theorem square_root_problem
  (x : ℤ) (y : ℤ)
  (hx : x = Nat.sqrt 16)
  (hy : y^2 = 9) :
  x^2 + y^2 + x - 2 = 27 := by
  sorry

end NUMINAMATH_GPT_square_root_problem_l1299_129902


namespace NUMINAMATH_GPT_sum_first_five_terms_l1299_129983

-- Define the geometric sequence
noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^n

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n
  else a1 * (1 - q^(n + 1)) / (1 - q)

-- Given conditions
def a1 : ℝ := 1
def q : ℝ := 2
def n : ℕ := 5

-- The theorem to be proven
theorem sum_first_five_terms : sum_geometric_sequence a1 q (n-1) = 31 := by
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_l1299_129983


namespace NUMINAMATH_GPT_linda_needs_additional_batches_l1299_129900

theorem linda_needs_additional_batches:
  let classmates := 24
  let cookies_per_classmate := 10
  let dozen := 12
  let cookies_per_batch := 4 * dozen
  let cookies_needed := classmates * cookies_per_classmate
  let chocolate_chip_batches := 2
  let oatmeal_raisin_batches := 1
  let cookies_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let remaining_cookies := cookies_needed - cookies_made
  let additional_batches := remaining_cookies / cookies_per_batch
  additional_batches = 2 :=
by
  sorry

end NUMINAMATH_GPT_linda_needs_additional_batches_l1299_129900


namespace NUMINAMATH_GPT_fraction_equation_solution_l1299_129942

theorem fraction_equation_solution (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) : 
  (1 / (x - 2) = 3 / x) → x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_equation_solution_l1299_129942


namespace NUMINAMATH_GPT_new_milk_water_ratio_l1299_129955

theorem new_milk_water_ratio
  (original_milk : ℚ)
  (original_water : ℚ)
  (added_water : ℚ)
  (h_ratio : original_milk / original_water = 2 / 1)
  (h_milk_qty : original_milk = 45)
  (h_added_water : added_water = 10) :
  original_milk / (original_water + added_water) = 18 / 13 :=
by
  sorry

end NUMINAMATH_GPT_new_milk_water_ratio_l1299_129955


namespace NUMINAMATH_GPT_estimate_y_value_l1299_129997

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end NUMINAMATH_GPT_estimate_y_value_l1299_129997


namespace NUMINAMATH_GPT_value_of_f_l1299_129912

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom f_has_derivative : ∀ x, deriv f x = f' x
axiom f_equation : ∀ x, f x = 3 * x^2 + 2 * x * (f' 1)

-- Proof goal
theorem value_of_f'_at_3 : f' 3 = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_f_l1299_129912


namespace NUMINAMATH_GPT_bill_sun_vs_sat_l1299_129947

theorem bill_sun_vs_sat (B_Sat B_Sun J_Sun : ℕ) 
  (h1 : B_Sun = 6)
  (h2 : J_Sun = 2 * B_Sun)
  (h3 : B_Sat + B_Sun + J_Sun = 20) : 
  B_Sun - B_Sat = 4 :=
by
  sorry

end NUMINAMATH_GPT_bill_sun_vs_sat_l1299_129947


namespace NUMINAMATH_GPT_problem_solution_l1299_129938

theorem problem_solution (x : ℝ) (h : x - 29 = 63) : (x - 47 = 45) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1299_129938


namespace NUMINAMATH_GPT_factor_expression_l1299_129948

variable (b : ℤ)

theorem factor_expression : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1299_129948


namespace NUMINAMATH_GPT_expansion_of_binomials_l1299_129945

theorem expansion_of_binomials (a : ℝ) : (a + 2) * (a - 3) = a^2 - a - 6 :=
  sorry

end NUMINAMATH_GPT_expansion_of_binomials_l1299_129945


namespace NUMINAMATH_GPT_hyperbola_sum_l1299_129910

noncomputable def h : ℝ := -3
noncomputable def k : ℝ := 1
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := Real.sqrt 50
noncomputable def b : ℝ := Real.sqrt (c ^ 2 - a ^ 2)

theorem hyperbola_sum :
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_GPT_hyperbola_sum_l1299_129910


namespace NUMINAMATH_GPT_eval_expr1_eval_expr2_l1299_129981

theorem eval_expr1 : (1 / Real.sin (10 * Real.pi / 180) - Real.sqrt 3 / Real.cos (10 * Real.pi / 180)) = 4 :=
by
  -- proof goes here
  sorry

theorem eval_expr2 : (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) / (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180))) = Real.sqrt 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_eval_expr1_eval_expr2_l1299_129981


namespace NUMINAMATH_GPT_a_horses_is_18_l1299_129970

-- Definitions of given conditions
def total_cost : ℕ := 435
def b_share : ℕ := 180
def horses_b : ℕ := 16
def months_b : ℕ := 9
def cost_b : ℕ := horses_b * months_b

def horses_c : ℕ := 18
def months_c : ℕ := 6
def cost_c : ℕ := horses_c * months_c

def total_cost_eq (x : ℕ) : Prop :=
  x * 8 + cost_b + cost_c = total_cost

-- Statement of the proof problem
theorem a_horses_is_18 (x : ℕ) : total_cost_eq x → x = 18 := 
sorry

end NUMINAMATH_GPT_a_horses_is_18_l1299_129970


namespace NUMINAMATH_GPT_pear_sales_ratio_l1299_129976

theorem pear_sales_ratio : 
  ∀ (total_sold afternoon_sold morning_sold : ℕ), 
  total_sold = 420 ∧ afternoon_sold = 280 ∧ total_sold = afternoon_sold + morning_sold 
  → afternoon_sold / morning_sold = 2 :=
by 
  intros total_sold afternoon_sold morning_sold 
  intro h 
  have h_total : total_sold = 420 := h.1 
  have h_afternoon : afternoon_sold = 280 := h.2.1 
  have h_morning : total_sold = afternoon_sold + morning_sold := h.2.2
  sorry

end NUMINAMATH_GPT_pear_sales_ratio_l1299_129976


namespace NUMINAMATH_GPT_solve_fractional_eq_l1299_129979

-- Defining the fractional equation as a predicate
def fractional_eq (x : ℝ) : Prop :=
  (5 / x) = (7 / (x - 2))

-- The main theorem to be proven
theorem solve_fractional_eq : ∃ x : ℝ, fractional_eq x ∧ x = -5 := by
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1299_129979


namespace NUMINAMATH_GPT_tim_weekly_earnings_l1299_129982

-- Definitions based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- The theorem that we need to prove
theorem tim_weekly_earnings :
  (tasks_per_day * pay_per_task * days_per_week : ℝ) = 720 :=
by
  sorry -- Skipping the proof

end NUMINAMATH_GPT_tim_weekly_earnings_l1299_129982


namespace NUMINAMATH_GPT_minimum_and_maximum_S_l1299_129986

theorem minimum_and_maximum_S (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : a^2 + b^2 + c^2 + d^2 = 30) :
  3 * (a^3 + b^3 + c^3 + d^3) - 3 * a^2 - 3 * b^2 - 3 * c^2 - 3 * d^2 = 7.5 :=
sorry

end NUMINAMATH_GPT_minimum_and_maximum_S_l1299_129986


namespace NUMINAMATH_GPT_integer_sequence_perfect_square_l1299_129928

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ a 2 = 4 ∧ ∀ n ≥ 2, a n = (a (n - 1) * a (n + 1) + 1) ^ (1 / 2)

theorem integer_sequence {a : ℕ → ℝ} : 
  seq a → ∀ n, ∃ k : ℤ, a n = k := 
by sorry

theorem perfect_square {a : ℕ → ℝ} :
  seq a → ∀ n, ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k ^ 2 :=
by sorry

end NUMINAMATH_GPT_integer_sequence_perfect_square_l1299_129928


namespace NUMINAMATH_GPT_garden_area_difference_l1299_129920
-- Import the entire Mathlib

-- Lean Statement
theorem garden_area_difference :
  let length_Alice := 15
  let width_Alice := 30
  let length_Bob := 18
  let width_Bob := 28
  let area_Alice := length_Alice * width_Alice
  let area_Bob := length_Bob * width_Bob
  let difference := area_Bob - area_Alice
  difference = 54 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_difference_l1299_129920


namespace NUMINAMATH_GPT_inequality_flip_l1299_129941

theorem inequality_flip (a b : ℤ) (c : ℤ) (h1 : a < b) (h2 : c < 0) : 
  c * a > c * b :=
sorry

end NUMINAMATH_GPT_inequality_flip_l1299_129941


namespace NUMINAMATH_GPT_inequality_relationship_cannot_be_established_l1299_129903

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_relationship_cannot_be_established :
  ¬ (1 / (a - b) > 1 / a) :=
by sorry

end NUMINAMATH_GPT_inequality_relationship_cannot_be_established_l1299_129903


namespace NUMINAMATH_GPT_maximum_value_of_parabola_eq_24_l1299_129966

theorem maximum_value_of_parabola_eq_24 (x : ℝ) : 
  ∃ x, x = -2 ∧ (-2 * x^2 - 8 * x + 16) = 24 :=
by
  use -2
  sorry

end NUMINAMATH_GPT_maximum_value_of_parabola_eq_24_l1299_129966


namespace NUMINAMATH_GPT_weight_of_replaced_student_l1299_129916

-- Define the conditions as hypotheses
variable (W : ℝ)
variable (h : W - 46 = 40)

-- Prove that W = 86
theorem weight_of_replaced_student : W = 86 :=
by
  -- We should conclude the proof; for now, we leave a placeholder
  sorry

end NUMINAMATH_GPT_weight_of_replaced_student_l1299_129916


namespace NUMINAMATH_GPT_cubic_root_expression_l1299_129952

theorem cubic_root_expression (p q r : ℝ) (h1 : p + q + r = 0) (h2 : p * q + p * r + q * r = -2) (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -24 :=
sorry

end NUMINAMATH_GPT_cubic_root_expression_l1299_129952
