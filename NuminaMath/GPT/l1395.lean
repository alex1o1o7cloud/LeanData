import Mathlib

namespace NUMINAMATH_GPT_number_of_possible_values_for_a_l1395_139591

theorem number_of_possible_values_for_a 
  (a b c d : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a > b) (h6 : b > c) (h7 : c > d)
  (h8 : a + b + c + d = 2004)
  (h9 : a^2 - b^2 - c^2 + d^2 = 1004) : 
  ∃ n : ℕ, n = 500 :=
  sorry

end NUMINAMATH_GPT_number_of_possible_values_for_a_l1395_139591


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1395_139588

theorem min_value_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) : 
  ∃ (m : ℝ), m = 2 ∧ ∀ c d : ℝ, 0 < c ∧ 0 < d ∧ c + d = 2 → (1/c + 1/d) ≥ m := 
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1395_139588


namespace NUMINAMATH_GPT_one_and_one_third_of_what_number_is_45_l1395_139581

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end NUMINAMATH_GPT_one_and_one_third_of_what_number_is_45_l1395_139581


namespace NUMINAMATH_GPT_dragons_legs_l1395_139549

theorem dragons_legs :
  ∃ (n : ℤ), ∀ (x y : ℤ), x + 3 * y = 26
                       → 40 * x + n * y = 298
                       → n = 14 :=
by
  sorry

end NUMINAMATH_GPT_dragons_legs_l1395_139549


namespace NUMINAMATH_GPT_simplify_to_ellipse_l1395_139510

theorem simplify_to_ellipse (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_to_ellipse_l1395_139510


namespace NUMINAMATH_GPT_distance_symmetric_line_eq_l1395_139594

noncomputable def distance_from_point_to_line : ℝ :=
  let x0 := 2
  let y0 := -1
  let A := 2
  let B := 3
  let C := 0
  (|A * x0 + B * y0 + C|) / (Real.sqrt (A^2 + B^2))

theorem distance_symmetric_line_eq : distance_from_point_to_line = 1 / (Real.sqrt 13) := by
  sorry

end NUMINAMATH_GPT_distance_symmetric_line_eq_l1395_139594


namespace NUMINAMATH_GPT_XY_sum_l1395_139534

theorem XY_sum (A B C D X Y : ℕ) 
  (h1 : A + B + C + D = 22) 
  (h2 : X = A + B) 
  (h3 : Y = C + D) 
  : X + Y = 4 := 
  sorry

end NUMINAMATH_GPT_XY_sum_l1395_139534


namespace NUMINAMATH_GPT_solve_for_x_l1395_139598

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x (x : ℝ) : 2 * f x - 10 = f (x - 2) ↔ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1395_139598


namespace NUMINAMATH_GPT_num_ordered_pairs_l1395_139518

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end NUMINAMATH_GPT_num_ordered_pairs_l1395_139518


namespace NUMINAMATH_GPT_find_kn_l1395_139504

theorem find_kn (k n : ℕ) (h : k * n^2 - k * n - n^2 + n = 94) : k = 48 ∧ n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_kn_l1395_139504


namespace NUMINAMATH_GPT_find_ratio_l1395_139546

-- Given that the tangent of angle θ (inclination angle) is -2
def tan_theta (θ : Real) : Prop := Real.tan θ = -2

theorem find_ratio (θ : Real) (h : tan_theta θ) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_find_ratio_l1395_139546


namespace NUMINAMATH_GPT_sachin_younger_than_rahul_l1395_139525

theorem sachin_younger_than_rahul
  (S R : ℝ)
  (h1 : S = 24.5)
  (h2 : S / R = 7 / 9) :
  R - S = 7 := 
by sorry

end NUMINAMATH_GPT_sachin_younger_than_rahul_l1395_139525


namespace NUMINAMATH_GPT_number_of_students_l1395_139542

theorem number_of_students (S : ℕ) (hS1 : S ≥ 2) (hS2 : S ≤ 80) 
                          (hO : ∀ n : ℕ, (n * S) % 120 = 0) : 
    S = 40 :=
sorry

end NUMINAMATH_GPT_number_of_students_l1395_139542


namespace NUMINAMATH_GPT_original_plan_trees_average_l1395_139545

-- Definitions based on conditions
def original_trees_per_day (x : ℕ) := x
def increased_trees_per_day (x : ℕ) := x + 5
def time_to_plant_60_trees (x : ℕ) := 60 / (x + 5)
def time_to_plant_45_trees (x : ℕ) := 45 / x

-- The main theorem we need to prove
theorem original_plan_trees_average : ∃ x : ℕ, time_to_plant_60_trees x = time_to_plant_45_trees x ∧ x = 15 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_original_plan_trees_average_l1395_139545


namespace NUMINAMATH_GPT_parity_implies_even_sum_l1395_139556

theorem parity_implies_even_sum (n m : ℤ) (h : Even (n^2 + m^2 + n * m)) : ¬Odd (n + m) :=
sorry

end NUMINAMATH_GPT_parity_implies_even_sum_l1395_139556


namespace NUMINAMATH_GPT_last_score_is_65_l1395_139511

-- Define the scores and the problem conditions
def scores := [65, 72, 75, 80, 85, 88, 92]
def total_sum := 557
def remaining_sum (score : ℕ) : ℕ := total_sum - score

-- Define a property to check divisibility
def divisible_by (n d : ℕ) : Prop := n % d = 0

-- The main theorem statement
theorem last_score_is_65 :
  (∀ s ∈ scores, divisible_by (remaining_sum s) 6) ∧ divisible_by total_sum 7 ↔ scores = [65, 72, 75, 80, 85, 88, 92] :=
sorry

end NUMINAMATH_GPT_last_score_is_65_l1395_139511


namespace NUMINAMATH_GPT_negation_is_all_odd_or_at_least_two_even_l1395_139565

-- Define natural numbers a, b, and c.
variables {a b c : ℕ}

-- Define a predicate is_even which checks if a number is even.
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define the statement that exactly one of the natural numbers a, b, and c is even.
def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∨ is_even b ∨ is_even c) ∧
  ¬ (is_even a ∧ is_even b) ∧
  ¬ (is_even a ∧ is_even c) ∧
  ¬ (is_even b ∧ is_even c)

-- Define the negation of the statement that exactly one of the natural numbers a, b, and c is even.
def negation_of_exactly_one_even (a b c : ℕ) : Prop :=
  ¬ exactly_one_even a b c

-- State that the negation of exactly one even number among a, b, c is equivalent to all being odd or at least two being even.
theorem negation_is_all_odd_or_at_least_two_even :
  negation_of_exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) :=
sorry

end NUMINAMATH_GPT_negation_is_all_odd_or_at_least_two_even_l1395_139565


namespace NUMINAMATH_GPT_binomial_1300_2_eq_844350_l1395_139559

theorem binomial_1300_2_eq_844350 : Nat.choose 1300 2 = 844350 := 
by
  sorry

end NUMINAMATH_GPT_binomial_1300_2_eq_844350_l1395_139559


namespace NUMINAMATH_GPT_mark_less_than_kate_and_laura_l1395_139526

theorem mark_less_than_kate_and_laura (K : ℝ) (h : K + 2 * K + 3 * K + 4.5 * K = 360) :
  let Pat := 2 * K
  let Mark := 3 * K
  let Laura := 4.5 * K
  let Combined := K + Laura
  Mark - Combined = -85.72 :=
sorry

end NUMINAMATH_GPT_mark_less_than_kate_and_laura_l1395_139526


namespace NUMINAMATH_GPT_intersection_nonempty_implies_a_gt_neg1_l1395_139560

def A := {x : ℝ | -1 ≤ x ∧ x < 2}
def B (a : ℝ) := {x : ℝ | x < a}

theorem intersection_nonempty_implies_a_gt_neg1 (a : ℝ) : (A ∩ B a).Nonempty → a > -1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_implies_a_gt_neg1_l1395_139560


namespace NUMINAMATH_GPT_div3_of_div9_l1395_139569

theorem div3_of_div9 (u v : ℤ) (h : 9 ∣ (u^2 + u * v + v^2)) : 3 ∣ u ∧ 3 ∣ v :=
sorry

end NUMINAMATH_GPT_div3_of_div9_l1395_139569


namespace NUMINAMATH_GPT_complement_union_l1395_139572

open Set

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement relative to U
def complement (A B : Set ℕ) : Set ℕ := { x ∈ B | x ∉ A }

-- The theorem we need to prove
theorem complement_union :
  complement (M ∪ N) U = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1395_139572


namespace NUMINAMATH_GPT_sequences_count_l1395_139519

theorem sequences_count (a_n b_n c_n : ℕ → ℕ) :
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧ (c_n 1 = 1) ∧ 
  (∀ n : ℕ, a_n (n + 1) = a_n n + b_n n) ∧ 
  (∀ n : ℕ, b_n (n + 1) = a_n n + b_n n + c_n n) ∧ 
  (∀ n : ℕ, c_n (n + 1) = b_n n + c_n n) → 
  ∀ n : ℕ, a_n n + b_n n + c_n n = 
            (1/2 * ((1 + Real.sqrt 2)^(n+1) + (1 - Real.sqrt 2)^(n+1))) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sequences_count_l1395_139519


namespace NUMINAMATH_GPT_polynomial_expansion_sum_l1395_139524

theorem polynomial_expansion_sum :
  ∀ P Q R S : ℕ, ∀ x : ℕ, 
  (P = 4 ∧ Q = 10 ∧ R = 1 ∧ S = 21) → 
  ((x + 3) * (4 * x ^ 2 - 2 * x + 7) = P * x ^ 3 + Q * x ^ 2 + R * x + S) → 
  P + Q + R + S = 36 :=
by
  intros P Q R S x h1 h2
  sorry

end NUMINAMATH_GPT_polynomial_expansion_sum_l1395_139524


namespace NUMINAMATH_GPT_max_a1_value_l1395_139558

theorem max_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n+2) = a n + a (n+1))
    (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) : a 1 ≤ 11 :=
by 
  sorry

end NUMINAMATH_GPT_max_a1_value_l1395_139558


namespace NUMINAMATH_GPT_minimum_cans_needed_l1395_139516

theorem minimum_cans_needed (h : ∀ c, c * 10 ≥ 120) : ∃ c, c = 12 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cans_needed_l1395_139516


namespace NUMINAMATH_GPT_root_one_value_of_m_real_roots_range_of_m_l1395_139548

variables {m x : ℝ}

-- Part 1: Prove that if 1 is a root of 'mx^2 - 4x + 1 = 0', then m = 3
theorem root_one_value_of_m (h : m * 1^2 - 4 * 1 + 1 = 0) : m = 3 :=
  by sorry

-- Part 2: Prove that 'mx^2 - 4x + 1 = 0' has real roots iff 'm ≤ 4 ∧ m ≠ 0'
theorem real_roots_range_of_m : (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 0) :=
  by sorry

end NUMINAMATH_GPT_root_one_value_of_m_real_roots_range_of_m_l1395_139548


namespace NUMINAMATH_GPT_simplify_expression_l1395_139528

noncomputable def expr := (-1 : ℝ)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1 / 8) * Real.sqrt 32

theorem simplify_expression : expr = 3 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1395_139528


namespace NUMINAMATH_GPT_interest_difference_l1395_139586

-- Conditions
def principal : ℕ := 350
def rate : ℕ := 4
def time : ℕ := 8

-- Question rewritten as a statement to prove
theorem interest_difference :
  let SI := (principal * rate * time) / 100 
  let difference := principal - SI
  difference = 238 := by
  sorry

end NUMINAMATH_GPT_interest_difference_l1395_139586


namespace NUMINAMATH_GPT_sum_first_nine_primes_l1395_139514

theorem sum_first_nine_primes : 
  2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 = 100 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_nine_primes_l1395_139514


namespace NUMINAMATH_GPT_stratified_sampling_l1395_139509

-- Definition of the given variables and conditions
def total_students_grade10 : ℕ := 30
def total_students_grade11 : ℕ := 40
def selected_students_grade11 : ℕ := 8

-- Implementation of the stratified sampling proportion requirement
theorem stratified_sampling (x : ℕ) (hx : (x : ℚ) / total_students_grade10 = (selected_students_grade11 : ℚ) / total_students_grade11) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1395_139509


namespace NUMINAMATH_GPT_ratio_x_2y_l1395_139595

theorem ratio_x_2y (x y : ℤ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : x / (2 * y) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_x_2y_l1395_139595


namespace NUMINAMATH_GPT_ship_length_correct_l1395_139500

noncomputable def ship_length : ℝ :=
  let speed_kmh := 24
  let speed_mps := speed_kmh * 1000 / 3600
  let time := 202.48
  let bridge_length := 900
  let total_distance := speed_mps * time
  total_distance - bridge_length

theorem ship_length_correct : ship_length = 450.55 :=
by
  -- This is where the proof would be written, but we're skipping the proof as per instructions
  sorry

end NUMINAMATH_GPT_ship_length_correct_l1395_139500


namespace NUMINAMATH_GPT_remaining_insects_is_twenty_one_l1395_139538

-- Define the initial counts of each type of insect
def spiders := 3
def ants := 12
def ladybugs := 8

-- Define the number of ladybugs that flew away
def ladybugs_flew_away := 2

-- Define the total initial number of insects
def total_insects_initial := spiders + ants + ladybugs

-- Define the total number of insects that remain after some ladybugs fly away
def total_insects_remaining := total_insects_initial - ladybugs_flew_away

-- Theorem statement: proving that the number of insects remaining is 21
theorem remaining_insects_is_twenty_one : total_insects_remaining = 21 := sorry

end NUMINAMATH_GPT_remaining_insects_is_twenty_one_l1395_139538


namespace NUMINAMATH_GPT_line_slope_intercept_through_points_l1395_139523

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end NUMINAMATH_GPT_line_slope_intercept_through_points_l1395_139523


namespace NUMINAMATH_GPT_determine_c_l1395_139575

-- Define the points
def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (0, 4)

-- Define the direction vector calculation
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)

-- Define the target direction vector form
def target_direction_vector (c : ℝ) : ℝ × ℝ := (3, c)

-- Theorem stating that the calculated direction vector equals the target direction vector when c = 3
theorem determine_c : direction_vector = target_direction_vector 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_determine_c_l1395_139575


namespace NUMINAMATH_GPT_fraction_of_15_smaller_by_20_l1395_139566

/-- Define 80% of 40 -/
def eighty_percent_of_40 : ℝ := 0.80 * 40

/-- Define the fraction of 15 that we are looking for -/
def fraction_of_15 (x : ℝ) : ℝ := x * 15

/-- Define the problem statement -/
theorem fraction_of_15_smaller_by_20 : ∃ x : ℝ, fraction_of_15 x = eighty_percent_of_40 - 20 ∧ x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_15_smaller_by_20_l1395_139566


namespace NUMINAMATH_GPT_max_showers_l1395_139522

open Nat

variable (household water_limit water_for_drinking_and_cooking water_per_shower pool_length pool_width pool_height water_per_cubic_foot pool_leakage_rate days_in_july : ℕ)

def volume_of_pool (length width height: ℕ): ℕ :=
  length * width * height

def water_usage (drinking cooking pool leakage: ℕ): ℕ :=
  drinking + cooking + pool + leakage

theorem max_showers (h1: water_limit = 1000)
                    (h2: water_for_drinking_and_cooking = 100)
                    (h3: water_per_shower = 20)
                    (h4: pool_length = 10)
                    (h5: pool_width = 10)
                    (h6: pool_height = 6)
                    (h7: water_per_cubic_foot = 1)
                    (h8: pool_leakage_rate = 5)
                    (h9: days_in_july = 31) : 
  (water_limit - water_usage water_for_drinking_and_cooking
                                  (volume_of_pool pool_length pool_width pool_height) 
                                  ((pool_leakage_rate * days_in_july))) / water_per_shower = 7 := by
  sorry

end NUMINAMATH_GPT_max_showers_l1395_139522


namespace NUMINAMATH_GPT_symmetric_line_x_axis_l1395_139532

theorem symmetric_line_x_axis (y : ℝ → ℝ) (x : ℝ) :
  (∀ x, y x = 2 * x + 1) → (∀ x, -y x = 2 * x + 1) → y x = -2 * x -1 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_symmetric_line_x_axis_l1395_139532


namespace NUMINAMATH_GPT_rhind_papyrus_problem_l1395_139515

theorem rhind_papyrus_problem 
  (a1 a2 a3 a4 a5 : ℚ)
  (h1 : a2 = a1 + d)
  (h2 : a3 = a1 + 2 * d)
  (h3 : a4 = a1 + 3 * d)
  (h4 : a5 = a1 + 4 * d)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 60)
  (h_condition : (a4 + a5) / 2 = a1 + a2 + a3) :
  a1 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rhind_papyrus_problem_l1395_139515


namespace NUMINAMATH_GPT_find_m_l1395_139537

theorem find_m (m : ℕ) (h : m * (Nat.factorial m) + 2 * (Nat.factorial m) = 5040) : m = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1395_139537


namespace NUMINAMATH_GPT_range_of_a_l1395_139571

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (x^2 - x - 2) / x^3

theorem range_of_a (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1 → f a x1 ≥ g x2) ↔ a ≥ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1395_139571


namespace NUMINAMATH_GPT_silver_coin_worth_l1395_139539

theorem silver_coin_worth :
  ∀ (g : ℕ) (S : ℕ) (n_gold n_silver cash : ℕ), 
  g = 50 →
  n_gold = 3 →
  n_silver = 5 →
  cash = 30 →
  n_gold * g + n_silver * S + cash = 305 →
  S = 25 :=
by
  intros g S n_gold n_silver cash
  intros hg hng hnsi hcash htotal
  sorry

end NUMINAMATH_GPT_silver_coin_worth_l1395_139539


namespace NUMINAMATH_GPT_original_selling_price_l1395_139505

theorem original_selling_price (CP SP_original SP_loss : ℝ)
  (h1 : SP_original = CP * 1.25)
  (h2 : SP_loss = CP * 0.85)
  (h3 : SP_loss = 544) : SP_original = 800 :=
by
  -- The proof goes here, but we are skipping it with sorry
  sorry

end NUMINAMATH_GPT_original_selling_price_l1395_139505


namespace NUMINAMATH_GPT_x_can_be_any_sign_l1395_139508

theorem x_can_be_any_sign
  (x y z w : ℤ)
  (h1 : (y - 1) * (w - 2) ≠ 0)
  (h2 : (x + 2)/(y - 1) < - (z + 3)/(w - 2)) :
  ∃ x : ℤ, True :=
by
  sorry

end NUMINAMATH_GPT_x_can_be_any_sign_l1395_139508


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1395_139582

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (a_1 : ℚ) (d : ℚ) (m : ℕ) 
    (ha1 : a_1 = 2) 
    (ha2 : a 2 + a 8 = 24)
    (ham : 2 * a m = 24) 
    (h_sum : ∀ n, S n = (n * (2 * a_1 + (n - 1) * d)) / 2) 
    (h_an : ∀ n, a n = a_1 + (n - 1) * d) : 
    S (2 * m) = 265 / 2 :=
by
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1395_139582


namespace NUMINAMATH_GPT_known_number_l1395_139529

theorem known_number (A B : ℕ) (h_hcf : 1 / (Nat.gcd A B) = 1 / 15) (h_lcm : 1 / Nat.lcm A B = 1 / 312) (h_B : B = 195) : A = 24 :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_known_number_l1395_139529


namespace NUMINAMATH_GPT_length_reduction_by_50_percent_l1395_139530

variable (L B L' : ℝ)

def rectangle_dimension_change (L B : ℝ) (perc_area_change : ℝ) (new_breadth_factor : ℝ) : Prop :=
  let original_area := L * B
  let new_breadth := new_breadth_factor * B
  let new_area := L' * new_breadth
  let expected_new_area := (1 + perc_area_change) * original_area
  new_area = expected_new_area

theorem length_reduction_by_50_percent (L B : ℝ) (h1: rectangle_dimension_change L B L' 0.5 3) : 
  L' = 0.5 * L :=
by
  unfold rectangle_dimension_change at h1
  simp at h1
  sorry

end NUMINAMATH_GPT_length_reduction_by_50_percent_l1395_139530


namespace NUMINAMATH_GPT_fraction_calls_by_team_B_l1395_139570

-- Define the conditions
variables (A B C : ℝ)
axiom ratio_agents : A = (5 / 8) * B
axiom ratio_calls : ∀ (c : ℝ), c = (6 / 5) * C

-- Prove the fraction of the total calls processed by team B
theorem fraction_calls_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : ∀ (c : ℝ), c = (6 / 5) * C) :
  (B * C) / ((5 / 8) * B * (6 / 5) * C + B * C) = 4 / 7 :=
by {
  -- proof is omitted, so we use sorry
  sorry
}

end NUMINAMATH_GPT_fraction_calls_by_team_B_l1395_139570


namespace NUMINAMATH_GPT_complex_number_imaginary_l1395_139543

theorem complex_number_imaginary (x : ℝ) 
  (h1 : x^2 - 2*x - 3 = 0)
  (h2 : x + 1 ≠ 0) : x = 3 := sorry

end NUMINAMATH_GPT_complex_number_imaginary_l1395_139543


namespace NUMINAMATH_GPT_total_wet_surface_area_is_correct_l1395_139577

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_is_correct_l1395_139577


namespace NUMINAMATH_GPT_find_c_for_Q_l1395_139593

noncomputable def Q (c : ℚ) (x : ℚ) : ℚ := x^3 + 3*x^2 + c*x + 8

theorem find_c_for_Q (c : ℚ) : 
  (Q c 3 = 0) ↔ (c = -62 / 3) := by
  sorry

end NUMINAMATH_GPT_find_c_for_Q_l1395_139593


namespace NUMINAMATH_GPT_discount_rate_l1395_139564

theorem discount_rate (cost_price marked_price desired_profit_margin selling_price : ℝ)
  (h1 : cost_price = 160)
  (h2 : marked_price = 240)
  (h3 : desired_profit_margin = 0.2)
  (h4 : selling_price = cost_price * (1 + desired_profit_margin)) :
  marked_price * (1 - ((marked_price - selling_price) / marked_price)) = selling_price :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_l1395_139564


namespace NUMINAMATH_GPT_auston_height_l1395_139587

noncomputable def auston_height_in_meters (height_in_inches : ℝ) : ℝ :=
  let height_in_cm := height_in_inches * 2.54
  height_in_cm / 100

theorem auston_height : auston_height_in_meters 65 = 1.65 :=
by
  sorry

end NUMINAMATH_GPT_auston_height_l1395_139587


namespace NUMINAMATH_GPT_part_1_part_2a_part_2b_l1395_139590

namespace InequalityProofs

-- Definitions extracted from the problem
def quadratic_function (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

-- Lean statement for Part 1
theorem part_1 (m : ℝ) : (∀ x : ℝ, quadratic_function m x ≥ -2) ↔ m ∈ Set.Ici (1 / 3) :=
sorry

-- Lean statement for Part 2, breaking into separate theorems for different ranges of m
theorem part_2a (m : ℝ) (h : m < -1) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic (-1 / m) ∪ Set.Ici 1)) :=
sorry

theorem part_2b (m : ℝ) (h : -1 < m ∧ m < 0) :
  (∀ x : ℝ, quadratic_function m x < m - 1) → 
  (∀ x : ℝ, x ∈ (Set.Iic 1 ∪ Set.Ici (-1 / m))) :=
sorry

end InequalityProofs

end NUMINAMATH_GPT_part_1_part_2a_part_2b_l1395_139590


namespace NUMINAMATH_GPT_find_a_n_l1395_139563

theorem find_a_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_sum : ∀ n : ℕ, S n = 1/2 * (a n + 1 / (a n))) :
  ∀ n : ℕ, a n = Real.sqrt ↑n - Real.sqrt (↑n - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_n_l1395_139563


namespace NUMINAMATH_GPT_false_proposition_is_C_l1395_139544

theorem false_proposition_is_C : ¬ (∀ x : ℝ, x^3 > 0) :=
sorry

end NUMINAMATH_GPT_false_proposition_is_C_l1395_139544


namespace NUMINAMATH_GPT_equal_profits_at_20000_end_month_more_profit_50000_l1395_139513

noncomputable section

-- Define the conditions
def profit_beginning_month (x : ℝ) : ℝ := 0.15 * x + 1.15 * x * 0.1
def profit_end_month (x : ℝ) : ℝ := 0.3 * x - 700

-- Proof Problem 1: Prove that at x = 20000, the profits are equal
theorem equal_profits_at_20000 : profit_beginning_month 20000 = profit_end_month 20000 :=
by
  sorry

-- Proof Problem 2: Prove that at x = 50000, selling at end of month yields more profit than selling at beginning of month
theorem end_month_more_profit_50000 : profit_end_month 50000 > profit_beginning_month 50000 :=
by
  sorry

end NUMINAMATH_GPT_equal_profits_at_20000_end_month_more_profit_50000_l1395_139513


namespace NUMINAMATH_GPT_correct_calculation_l1395_139527

theorem correct_calculation (x : ℝ) : (2 * x^5) / (-x)^3 = -2 * x^2 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1395_139527


namespace NUMINAMATH_GPT_pascal_current_speed_l1395_139512

variable (v : ℝ)
variable (h₁ : v > 0) -- current speed is positive

-- Conditions
variable (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16)

-- Proving the speed
theorem pascal_current_speed (h₁ : v > 0) (h₂ : 96 / (v - 4) = 96 / (1.5 * v) + 16) : v = 8 :=
sorry

end NUMINAMATH_GPT_pascal_current_speed_l1395_139512


namespace NUMINAMATH_GPT_value_of_m_squared_plus_reciprocal_squared_l1395_139507

theorem value_of_m_squared_plus_reciprocal_squared 
  (m : ℝ) 
  (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 4 = 102 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_m_squared_plus_reciprocal_squared_l1395_139507


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1395_139501

variable {a_n : ℕ → ℕ}
variable (S_n : ℕ → ℕ)
variable (q : ℕ)
variable (a_1 : ℕ)

axiom h1 : a_n 2 = 2
axiom h2 : a_n 6 = 32
axiom h3 : ∀ n, S_n n = a_1 * (1 - q ^ n) / (1 - q)

theorem arithmetic_seq_sum : S_n 100 = 2^100 - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1395_139501


namespace NUMINAMATH_GPT_third_side_not_twelve_l1395_139553

theorem third_side_not_twelve (x : ℕ) (h1 : x > 5) (h2 : x < 11) (h3 : x % 2 = 0) : x ≠ 12 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_third_side_not_twelve_l1395_139553


namespace NUMINAMATH_GPT_geometric_sequence_new_product_l1395_139531

theorem geometric_sequence_new_product 
  (a r : ℝ) (n : ℕ) (h_even : n % 2 = 0)
  (P S S' : ℝ)
  (hP : P = a^n * r^(n * (n-1) / 2))
  (hS : S = a * (1 - r^n) / (1 - r))
  (hS' : S' = (1 - r^n) / (a * (1 - r))) :
  (2^n * a^n * r^(n * (n-1) / 2)) = (S * S')^(n / 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_new_product_l1395_139531


namespace NUMINAMATH_GPT_solve_quadratic_eqn_l1395_139580

theorem solve_quadratic_eqn : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eqn_l1395_139580


namespace NUMINAMATH_GPT_exists_pow_two_sub_one_divisible_by_odd_l1395_139589

theorem exists_pow_two_sub_one_divisible_by_odd {a : ℕ} (h_odd : a % 2 = 1) 
  : ∃ b : ℕ, (2^b - 1) % a = 0 :=
sorry

end NUMINAMATH_GPT_exists_pow_two_sub_one_divisible_by_odd_l1395_139589


namespace NUMINAMATH_GPT_soccer_tournament_solution_l1395_139592

-- Define the statement of the problem
theorem soccer_tournament_solution (k : ℕ) (n m : ℕ) (h1 : k ≥ 1) (h2 : n = (k+1)^2) (h3 : m = k*(k+1) / 2)
  (h4 : n > m) : 
  ∃ k : ℕ, n = (k + 1) ^ 2 ∧ m = k * (k + 1) / 2 ∧ k ≥ 1 := 
sorry

end NUMINAMATH_GPT_soccer_tournament_solution_l1395_139592


namespace NUMINAMATH_GPT_relationship_x2_ax_bx_l1395_139503

variable {x a b : ℝ}

theorem relationship_x2_ax_bx (h1 : x < a) (h2 : a < 0) (h3 : b > 0) : x^2 > ax ∧ ax > bx :=
by
  sorry

end NUMINAMATH_GPT_relationship_x2_ax_bx_l1395_139503


namespace NUMINAMATH_GPT_xiao_yun_age_l1395_139551

theorem xiao_yun_age (x : ℕ) (h1 : ∀ x, x + 25 = Xiao_Yun_fathers_current_age)
                     (h2 : ∀ x, Xiao_Yun_fathers_age_in_5_years = 2 * (x+5) - 10) :
  x = 30 := by
  sorry

end NUMINAMATH_GPT_xiao_yun_age_l1395_139551


namespace NUMINAMATH_GPT_dice_sum_to_11_l1395_139550

/-- Define the conditions for the outcomes of the dice rolls -/
def valid_outcomes (x : Fin 5 → ℕ) : Prop :=
  (∀ i, 1 ≤ x i ∧ x i ≤ 6) ∧ (x 0 + x 1 + x 2 + x 3 + x 4 = 11)

/-- Prove that there are exactly 205 ways to achieve a sum of 11 with five different colored dice -/
theorem dice_sum_to_11 : 
  (∃ (s : Finset (Fin 5 → ℕ)), (∀ x ∈ s, valid_outcomes x) ∧ s.card = 205) :=
  by
    sorry

end NUMINAMATH_GPT_dice_sum_to_11_l1395_139550


namespace NUMINAMATH_GPT_both_solve_correctly_l1395_139597

-- Define the probabilities of making an error for individuals A and B
variables (a b : ℝ)

-- Assuming a and b are probabilities, they must lie in the interval [0, 1]
axiom a_prob : 0 ≤ a ∧ a ≤ 1
axiom b_prob : 0 ≤ b ∧ b ≤ 1

-- Define the event that both individuals solve the problem correctly
theorem both_solve_correctly : (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by
  sorry

end NUMINAMATH_GPT_both_solve_correctly_l1395_139597


namespace NUMINAMATH_GPT_casey_nail_decorating_time_l1395_139517

theorem casey_nail_decorating_time :
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  total_time = 160 :=
by
  let coat_application_time := 20
  let coat_drying_time := 20
  let pattern_time := 40
  let total_time := 3 * (coat_application_time + coat_drying_time) + pattern_time
  trivial

end NUMINAMATH_GPT_casey_nail_decorating_time_l1395_139517


namespace NUMINAMATH_GPT_three_pow_2023_mod_17_l1395_139561

theorem three_pow_2023_mod_17 : (3 ^ 2023) % 17 = 7 := by
  sorry

end NUMINAMATH_GPT_three_pow_2023_mod_17_l1395_139561


namespace NUMINAMATH_GPT_simplify_expression_l1395_139541

variable (y : ℝ)

theorem simplify_expression :
  4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1395_139541


namespace NUMINAMATH_GPT_liquid_in_cylinders_l1395_139562

theorem liquid_in_cylinders (n : ℕ) (a : ℝ) (h1 : 2 ≤ n) :
  (∃ x : ℕ → ℝ, ∀ (k : ℕ), (1 ≤ k ∧ k ≤ n) → 
    (if k = 1 then 
      x k = a * n * (n - 2) / (n - 1) ^ 2 
    else if k = 2 then 
      x k = a * (n^2 - 2*n + 2) / (n - 1) ^ 2 
    else 
      x k = a)) :=
sorry

end NUMINAMATH_GPT_liquid_in_cylinders_l1395_139562


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1395_139584

variable {a b : ℝ}

theorem simplify_expr1 : 3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1 :=
by
  sorry

theorem simplify_expr2 : 2 * (5 * a - 3 * b) - 3 * (a ^ 2 - 2 * b) = 10 * a - 3 * a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1395_139584


namespace NUMINAMATH_GPT_tiles_count_l1395_139536

variable (c r : ℕ)

-- given: r = 10
def initial_rows_eq : Prop := r = 10

-- assertion: number of tiles is conserved after rearrangement
def tiles_conserved : Prop := c * r = (c - 2) * (r + 4)

-- desired: total number of tiles is 70
def total_tiles : Prop := c * r = 70

theorem tiles_count (h1 : initial_rows_eq r) (h2 : tiles_conserved c r) : total_tiles c r :=
by
  subst h1
  sorry

end NUMINAMATH_GPT_tiles_count_l1395_139536


namespace NUMINAMATH_GPT_unique_solution_of_system_of_equations_l1395_139574
open Set

variable {α : Type*} (A B X : Set α)

theorem unique_solution_of_system_of_equations :
  (X ∩ (A ∪ B) = X) ∧
  (A ∩ (B ∪ X) = A) ∧
  (B ∩ (A ∪ X) = B) ∧
  (X ∩ A ∩ B = ∅) →
  (X = (A \ B) ∪ (B \ A)) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_system_of_equations_l1395_139574


namespace NUMINAMATH_GPT_calculate_B_l1395_139583
open Real

theorem calculate_B 
  (A B : ℝ) 
  (a b : ℝ) 
  (hA : A = π / 6) 
  (ha : a = 1) 
  (hb : b = sqrt 3) 
  (h_sin_relation : sin B = (b * sin A) / a) : 
  (B = π / 3 ∨ B = 2 * π / 3) :=
sorry

end NUMINAMATH_GPT_calculate_B_l1395_139583


namespace NUMINAMATH_GPT_sqrt_diff_of_squares_l1395_139521

theorem sqrt_diff_of_squares : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end NUMINAMATH_GPT_sqrt_diff_of_squares_l1395_139521


namespace NUMINAMATH_GPT_point_P_in_Quadrant_II_l1395_139578

noncomputable def α : ℝ := (5 * Real.pi) / 8

theorem point_P_in_Quadrant_II : (Real.sin α > 0) ∧ (Real.tan α < 0) := sorry

end NUMINAMATH_GPT_point_P_in_Quadrant_II_l1395_139578


namespace NUMINAMATH_GPT_range_of_x_for_positive_function_value_l1395_139506

variable {R : Type*} [LinearOrderedField R]

def even_function (f : R → R) := ∀ x, f (-x) = f x

def monotonically_decreasing_on_nonnegatives (f : R → R) := ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem range_of_x_for_positive_function_value (f : R → R)
  (hf_even : even_function f)
  (hf_monotonic : monotonically_decreasing_on_nonnegatives f)
  (hf_at_2 : f 2 = 0)
  (hf_positive : ∀ x, f (x - 1) > 0) :
  ∀ x, -1 < x ∧ x < 3 := sorry

end NUMINAMATH_GPT_range_of_x_for_positive_function_value_l1395_139506


namespace NUMINAMATH_GPT_weighted_average_salary_l1395_139555

theorem weighted_average_salary :
  let num_managers := 9
  let salary_managers := 4500
  let num_associates := 18
  let salary_associates := 3500
  let num_lead_cashiers := 6
  let salary_lead_cashiers := 3000
  let num_sales_representatives := 45
  let salary_sales_representatives := 2500
  let total_salaries := 
    (num_managers * salary_managers) +
    (num_associates * salary_associates) +
    (num_lead_cashiers * salary_lead_cashiers) +
    (num_sales_representatives * salary_sales_representatives)
  let total_employees := 
    num_managers + num_associates + num_lead_cashiers + num_sales_representatives
  let weighted_avg_salary := total_salaries / total_employees
  weighted_avg_salary = 3000 := 
by
  sorry

end NUMINAMATH_GPT_weighted_average_salary_l1395_139555


namespace NUMINAMATH_GPT_share_of_C_l1395_139520

/-- Given the conditions:
  - Total investment is Rs. 120,000.
  - A's investment is Rs. 6,000 more than B's.
  - B's investment is Rs. 8,000 more than C's.
  - Profit distribution ratio among A, B, and C is 4:3:2.
  - Total profit is Rs. 50,000.
Prove that C's share of the profit is Rs. 11,111.11. -/
theorem share_of_C (total_investment : ℝ)
  (A_more_than_B : ℝ)
  (B_more_than_C : ℝ)
  (profit_distribution : ℝ)
  (total_profit : ℝ) :
  total_investment = 120000 →
  A_more_than_B = 6000 →
  B_more_than_C = 8000 →
  profit_distribution = 4 / 9 →
  total_profit = 50000 →
  ∃ (C_share : ℝ), C_share = 11111.11 :=
by
  sorry

end NUMINAMATH_GPT_share_of_C_l1395_139520


namespace NUMINAMATH_GPT_finite_solutions_l1395_139596

variable (a b : ℕ) (h1 : a ≠ b)

theorem finite_solutions (a b : ℕ) (h1 : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
  (x * y + z * w = a) ∧ (x * z + y * w = b) →
  (x, y, z, w) ∈ S :=
sorry

end NUMINAMATH_GPT_finite_solutions_l1395_139596


namespace NUMINAMATH_GPT_range_of_a_l1395_139533

-- Definition of sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B (a : ℝ) : Set ℝ := { x | x < a }

-- Condition of the union of A and B
theorem range_of_a (a : ℝ) : (A ∪ B a = { x | x < 1 }) ↔ -1 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1395_139533


namespace NUMINAMATH_GPT_sum_of_ages_in_10_years_l1395_139567

-- Define the initial conditions about Ann's and Tom's ages
def AnnCurrentAge : ℕ := 6
def TomCurrentAge : ℕ := 2 * AnnCurrentAge

-- Define their ages 10 years later
def AnnAgeIn10Years : ℕ := AnnCurrentAge + 10
def TomAgeIn10Years : ℕ := TomCurrentAge + 10

-- The proof statement
theorem sum_of_ages_in_10_years : AnnAgeIn10Years + TomAgeIn10Years = 38 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_in_10_years_l1395_139567


namespace NUMINAMATH_GPT_mia_socks_l1395_139540

-- Defining the number of each type of socks
variables {a b c : ℕ}

-- Conditions and constraints
def total_pairs (a b c : ℕ) : Prop := a + b + c = 15
def total_cost (a b c : ℕ) : Prop := 2 * a + 3 * b + 5 * c = 35
def at_least_one (a b c : ℕ) : Prop := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- Main theorem to prove the number of 2-dollar pairs of socks
theorem mia_socks : 
  ∀ (a b c : ℕ), 
  total_pairs a b c → 
  total_cost a b c → 
  at_least_one a b c → 
  a = 12 :=
by
  sorry

end NUMINAMATH_GPT_mia_socks_l1395_139540


namespace NUMINAMATH_GPT_area_of_figure_enclosed_by_curve_l1395_139552

theorem area_of_figure_enclosed_by_curve (θ : ℝ) : 
  ∃ (A : ℝ), A = 4 * Real.pi ∧ (∀ θ, (4 * Real.cos θ)^2 = (4 * Real.cos θ) * 4 * Real.cos θ) :=
sorry

end NUMINAMATH_GPT_area_of_figure_enclosed_by_curve_l1395_139552


namespace NUMINAMATH_GPT_g_value_at_49_l1395_139535

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value_at_49 :
  (∀ x y : ℝ, 0 < x → 0 < y → x * g y - y * g x = g (x^2 / y)) →
  g 49 = 0 :=
by
  -- Assuming the given condition holds for all positive real numbers x and y
  intro h
  -- sorry placeholder represents the proof process
  sorry

end NUMINAMATH_GPT_g_value_at_49_l1395_139535


namespace NUMINAMATH_GPT_no_valid_k_values_l1395_139547

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roots_are_primes (k : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 57 ∧ p * q = k

theorem no_valid_k_values : ∀ k : ℕ, ¬ roots_are_primes k := by
  sorry

end NUMINAMATH_GPT_no_valid_k_values_l1395_139547


namespace NUMINAMATH_GPT_find_m_l1395_139568

def U : Set ℕ := {1, 2, 3, 4}
def compl_U_A : Set ℕ := {1, 4}

theorem find_m (m : ℕ) (A : Set ℕ) (hA : A = {x | x ^ 2 - 5 * x + m = 0 ∧ x ∈ U}) :
  compl_U_A = U \ A → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1395_139568


namespace NUMINAMATH_GPT_area_of_triangle_ABC_is_1_l1395_139557

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (2, 1)

-- Define the function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove that the area of triangle ABC is 1
theorem area_of_triangle_ABC_is_1 : triangle_area A B C = 1 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_is_1_l1395_139557


namespace NUMINAMATH_GPT_part1_part2_l1395_139585

-- Define set A
def A : Set ℝ := {x | 3 < x ∧ x < 6}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set complement in ℝ
def CR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- First part of the problem
theorem part1 :
  (A ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (CR A ∪ CR B = {x | x ≤ 3 ∨ x ≥ 6}) :=
sorry

-- Define set C depending on a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Second part of the problem
theorem part2 (a : ℝ) (h : B ∪ C a = B) :
  a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1395_139585


namespace NUMINAMATH_GPT_fraction_value_l1395_139579

theorem fraction_value :
  (20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20) = -1 :=
by
  -- simplified proof omitted
  sorry

end NUMINAMATH_GPT_fraction_value_l1395_139579


namespace NUMINAMATH_GPT_disjoint_sets_l1395_139502

def P : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => 4 * x^3 + 3 * x
| (n + 1), x => (4 * x^2 + 2) * P n x - P (n - 1) x

def A (m : ℝ) : Set ℝ := {x | ∃ n : ℕ, P n m = x }

theorem disjoint_sets (m : ℝ) : Disjoint (A m) (A (m + 4)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_disjoint_sets_l1395_139502


namespace NUMINAMATH_GPT_sqrt_sqrt4_of_decimal_l1395_139576

theorem sqrt_sqrt4_of_decimal (h : 0.000625 = 625 / (10 ^ 6)) :
  Real.sqrt (Real.sqrt (Real.sqrt (Real.sqrt 625) / 1000)) = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sqrt4_of_decimal_l1395_139576


namespace NUMINAMATH_GPT_calculate_x_l1395_139554

def percentage (p : ℚ) (n : ℚ) := (p / 100) * n

theorem calculate_x : 
  (percentage 47 1442 - percentage 36 1412) + 65 = 234.42 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_x_l1395_139554


namespace NUMINAMATH_GPT_pencils_given_l1395_139599

theorem pencils_given (pencils_original pencils_left pencils_given : ℕ)
  (h1 : pencils_original = 142)
  (h2 : pencils_left = 111)
  (h3 : pencils_given = pencils_original - pencils_left) :
  pencils_given = 31 :=
by
  sorry

end NUMINAMATH_GPT_pencils_given_l1395_139599


namespace NUMINAMATH_GPT_some_number_value_l1395_139573

theorem some_number_value (a : ℕ) (some_number : ℕ) (h_a : a = 105)
  (h_eq : a ^ 3 = some_number * 25 * 35 * 63) : some_number = 7 := by
  sorry

end NUMINAMATH_GPT_some_number_value_l1395_139573
