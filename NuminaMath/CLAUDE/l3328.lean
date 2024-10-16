import Mathlib

namespace NUMINAMATH_CALUDE_green_ball_fraction_l3328_332814

theorem green_ball_fraction (total : ℕ) (green blue yellow white : ℕ) :
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  blue = 6 →
  green + blue + yellow + white = total →
  green = total / 4 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_fraction_l3328_332814


namespace NUMINAMATH_CALUDE_complete_square_existence_l3328_332864

theorem complete_square_existence :
  ∃ (k : ℤ) (a : ℝ), ∀ z : ℝ, z^2 - 6*z + 17 = (z + a)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_existence_l3328_332864


namespace NUMINAMATH_CALUDE_sum_a_plus_d_l3328_332847

theorem sum_a_plus_d (a b c d : ℤ) 
  (eq1 : a + b = 12) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_l3328_332847


namespace NUMINAMATH_CALUDE_trig_identity_l3328_332825

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3328_332825


namespace NUMINAMATH_CALUDE_bezout_identity_solutions_l3328_332837

theorem bezout_identity_solutions (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (u₀ v₀ : ℤ), ∀ (u v : ℤ),
    (a * u + b * v = Int.gcd a b) ↔ ∃ (k : ℤ), u = u₀ - k * b ∧ v = v₀ + k * a :=
by sorry

end NUMINAMATH_CALUDE_bezout_identity_solutions_l3328_332837


namespace NUMINAMATH_CALUDE_second_year_increase_is_fifteen_percent_l3328_332857

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  ((final_population : ℚ) / population_after_first_year - 1) * 100

/-- Theorem stating that given the specific conditions of the problem,
    the second year increase is 15%. -/
theorem second_year_increase_is_fifteen_percent :
  second_year_increase 800 (25 / 100) 1150 = 15 := by
  sorry

#eval second_year_increase 800 (25 / 100) 1150

end NUMINAMATH_CALUDE_second_year_increase_is_fifteen_percent_l3328_332857


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3328_332879

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ (x : ℤ), (x > m ∧ x < 8) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)) ↔
  (4 ≤ m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3328_332879


namespace NUMINAMATH_CALUDE_sum_of_squares_l3328_332887

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 2) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3328_332887


namespace NUMINAMATH_CALUDE_probability_is_point_six_l3328_332828

/-- Represents a company with a number of representatives -/
structure Company where
  representatives : ℕ

/-- Represents the meeting setup -/
structure Meeting where
  companies : Finset Company
  total_representatives : ℕ

/-- Calculates the probability of selecting 3 individuals from 3 different companies -/
def probability_three_different_companies (m : Meeting) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem probability_is_point_six (m : Meeting) 
  (h1 : m.companies.card = 4)
  (h2 : ∃ a ∈ m.companies, a.representatives = 2)
  (h3 : (m.companies.filter (λ c : Company => c.representatives = 1)).card = 3)
  (h4 : m.total_representatives = 5) :
  probability_three_different_companies m = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_probability_is_point_six_l3328_332828


namespace NUMINAMATH_CALUDE_power_of_two_subset_bound_l3328_332885

/-- The set of powers of 2 from 2^7 to 2^n -/
def S (n : ℕ) : Set ℕ := {x | ∃ k, 7 ≤ k ∧ k ≤ n ∧ x = 2^k}

/-- The subset of S where the sum of the last three digits is 8 -/
def A (n : ℕ) : Set ℕ := {x ∈ S n | (x % 1000 / 100 + x % 100 / 10 + x % 10) = 8}

/-- The number of elements in a finite set -/
def card {α : Type*} (s : Set α) : ℕ := sorry

theorem power_of_two_subset_bound (n : ℕ) (h : n ≥ 2009) :
  (28 : ℚ) / 2009 < (card (A n) : ℚ) / (card (S n)) ∧
  (card (A n) : ℚ) / (card (S n)) < 82 / 2009 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_subset_bound_l3328_332885


namespace NUMINAMATH_CALUDE_gcd_lcm_product_180_l3328_332810

def count_gcd_values (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ),
    (∀ a b : ℕ, (Nat.gcd a b) * (Nat.lcm a b) = n →
      (Nat.gcd a b) ∈ S) ∧
    S.card = 8

theorem gcd_lcm_product_180 :
  count_gcd_values 180 := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_180_l3328_332810


namespace NUMINAMATH_CALUDE_erdos_szekeres_101_l3328_332841

theorem erdos_szekeres_101 (σ : Fin 101 → Fin 101) :
  ∃ (s : Finset (Fin 101)) (f : Fin 11 → Fin 101),
    s.card = 11 ∧ 
    (∀ i j : Fin 11, i < j → (f i : ℕ) < (f j : ℕ) ∨ (f i : ℕ) > (f j : ℕ)) ∧
    (∀ i : Fin 11, f i ∈ s) :=
sorry

end NUMINAMATH_CALUDE_erdos_szekeres_101_l3328_332841


namespace NUMINAMATH_CALUDE_percentage_of_1000_l3328_332813

theorem percentage_of_1000 (x : ℝ) (h : x = 66.2) : 
  (x / 1000) * 100 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_1000_l3328_332813


namespace NUMINAMATH_CALUDE_quadratic_function_k_l3328_332835

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℚ) : ℚ := a * x^2 + b * x + c

theorem quadratic_function_k (a b c k : ℤ) : 
  g a b c (-1) = 0 → 
  (30 < g a b c 5 ∧ g a b c 5 < 40) → 
  (120 < g a b c 7 ∧ g a b c 7 < 130) → 
  (2000 * k < g a b c 50 ∧ g a b c 50 < 2000 * (k + 1)) → 
  k = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_k_l3328_332835


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l3328_332819

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) (D : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  0 < C ∧ C < Real.pi / 2 →
  A + B + C = Real.pi →
  f A = 3 / 2 →
  ∃ (AD BD : ℝ), AD = Real.sqrt 2 * BD ∧ AD = 2 →
  Real.cos C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l3328_332819


namespace NUMINAMATH_CALUDE_min_first_prize_l3328_332853

/-- Represents the prize structure and constraints for a competition --/
structure PrizeStructure where
  total_fund : ℕ
  first_prize : ℕ
  second_prize : ℕ
  third_prize : ℕ
  first_winners : ℕ
  second_winners : ℕ
  third_winners : ℕ

/-- Defines the conditions for a valid prize structure --/
def is_valid_structure (p : PrizeStructure) : Prop :=
  p.total_fund = 10800 ∧
  p.first_prize = 3 * p.second_prize ∧
  p.second_prize = 3 * p.third_prize ∧
  p.third_prize * p.third_winners > p.second_prize * p.second_winners ∧
  p.second_prize * p.second_winners > p.first_prize * p.first_winners ∧
  p.first_winners + p.second_winners + p.third_winners ≤ 20 ∧
  p.first_prize * p.first_winners + p.second_prize * p.second_winners + p.third_prize * p.third_winners = p.total_fund

/-- Theorem stating the minimum first prize amount --/
theorem min_first_prize (p : PrizeStructure) (h : is_valid_structure p) : 
  p.first_prize ≥ 2700 := by
  sorry

#check min_first_prize

end NUMINAMATH_CALUDE_min_first_prize_l3328_332853


namespace NUMINAMATH_CALUDE_B_power_101_l3328_332890

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 :
  B ^ 101 = ![![0, 0, 1],
              ![1, 0, 0],
              ![0, 1, 0]] := by
  sorry

end NUMINAMATH_CALUDE_B_power_101_l3328_332890


namespace NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l3328_332824

noncomputable def f (x : ℝ) := x / Real.exp x

theorem f_monotonicity_and_inequality :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_inequality_l3328_332824


namespace NUMINAMATH_CALUDE_tomatoes_left_l3328_332808

/-- Theorem: Given a farmer with 97 tomatoes who picks 83 tomatoes, the number of tomatoes left is equal to 14. -/
theorem tomatoes_left (total : ℕ) (picked : ℕ) (h1 : total = 97) (h2 : picked = 83) :
  total - picked = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l3328_332808


namespace NUMINAMATH_CALUDE_remainder_problem_l3328_332845

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (4 * n + 2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3328_332845


namespace NUMINAMATH_CALUDE_total_marks_math_physics_l3328_332807

/-- Proves that the total marks in mathematics and physics is 60 -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 40 →
  M + P = 60 := by
sorry

end NUMINAMATH_CALUDE_total_marks_math_physics_l3328_332807


namespace NUMINAMATH_CALUDE_train_passing_platform_l3328_332880

/-- Calculates the time for a train to pass a platform given its length, time to cross a tree, and platform length -/
theorem train_passing_platform (train_length : ℝ) (time_cross_tree : ℝ) (platform_length : ℝ) :
  train_length = 1200 ∧ time_cross_tree = 120 ∧ platform_length = 1100 →
  (train_length + platform_length) / (train_length / time_cross_tree) = 230 := by
sorry

end NUMINAMATH_CALUDE_train_passing_platform_l3328_332880


namespace NUMINAMATH_CALUDE_minoxidil_mixture_l3328_332849

theorem minoxidil_mixture (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_volume : ℝ) (added_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 70 ∧ 
  initial_concentration = 0.02 ∧ 
  added_volume = 35 ∧ 
  added_concentration = 0.05 ∧ 
  final_concentration = 0.03 →
  (initial_volume * initial_concentration + added_volume * added_concentration) / 
    (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_minoxidil_mixture_l3328_332849


namespace NUMINAMATH_CALUDE_square_of_1307_l3328_332802

theorem square_of_1307 : 1307 * 1307 = 1709849 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1307_l3328_332802


namespace NUMINAMATH_CALUDE_simplify_86_with_95_base_l3328_332839

/-- Simplifies a score based on a given base score. -/
def simplify_score (score : Int) (base : Int) : Int :=
  score - base

/-- The base score considered as excellent. -/
def excellent_score : Int := 95

/-- Theorem: Simplifying a score of 86 with 95 as the base results in -9. -/
theorem simplify_86_with_95_base :
  simplify_score 86 excellent_score = -9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_86_with_95_base_l3328_332839


namespace NUMINAMATH_CALUDE_football_club_balance_l3328_332883

/-- Calculates the final balance of a football club after player transactions -/
def final_balance (initial_balance : ℝ) (players_sold : ℕ) (selling_price : ℝ) 
  (players_bought : ℕ) (buying_price : ℝ) : ℝ :=
  initial_balance + players_sold * selling_price - players_bought * buying_price

/-- Theorem: The final balance of the football club is $60 million -/
theorem football_club_balance : 
  final_balance 100 2 10 4 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_football_club_balance_l3328_332883


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3328_332834

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3328_332834


namespace NUMINAMATH_CALUDE_base5_product_correct_l3328_332892

/-- Converts a base 5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The first number in base 5 --/
def num1 : List Nat := [3, 0, 2]

/-- The second number in base 5 --/
def num2 : List Nat := [4, 1]

/-- The expected product in base 5 --/
def expected_product : List Nat := [2, 0, 4, 3]

theorem base5_product_correct :
  toBase5 (toDecimal num1 * toDecimal num2) = expected_product := by
  sorry

end NUMINAMATH_CALUDE_base5_product_correct_l3328_332892


namespace NUMINAMATH_CALUDE_valid_pairings_eq_twenty_l3328_332859

/-- Represents the number of bowls -/
def num_bowls : ℕ := 5

/-- Represents the number of glasses -/
def num_glasses : ℕ := 4

/-- Represents the number of bowls with matching glasses -/
def num_matching_bowls : ℕ := 4

/-- Calculates the number of valid pairings between bowls and glasses -/
def valid_pairings : ℕ :=
  (num_matching_bowls * num_glasses) + (num_bowls - num_matching_bowls) * num_glasses

/-- Theorem stating that the number of valid pairings is 20 -/
theorem valid_pairings_eq_twenty : valid_pairings = 20 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairings_eq_twenty_l3328_332859


namespace NUMINAMATH_CALUDE_complex_absolute_value_squared_l3328_332822

theorem complex_absolute_value_squared (z : ℂ) (h : z + Complex.abs z = 3 + 7*I) : Complex.abs z ^ 2 = 841 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_squared_l3328_332822


namespace NUMINAMATH_CALUDE_connie_additional_money_l3328_332831

def additional_money_needed (savings : ℚ) (watch_cost : ℚ) (strap_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost_before_tax := watch_cost + strap_cost
  let tax_amount := total_cost_before_tax * tax_rate
  let total_cost_with_tax := total_cost_before_tax + tax_amount
  total_cost_with_tax - savings

theorem connie_additional_money :
  additional_money_needed 39 55 15 (8/100) = 366/10 := by
  sorry

end NUMINAMATH_CALUDE_connie_additional_money_l3328_332831


namespace NUMINAMATH_CALUDE_prime_sum_divisibility_l3328_332894

theorem prime_sum_divisibility (p : ℕ) : 
  Prime p → (7^p - 6^p + 2) % 43 = 0 → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisibility_l3328_332894


namespace NUMINAMATH_CALUDE_platform_length_platform_length_is_340_l3328_332818

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- The platform length is 340 meters -/
theorem platform_length_is_340 :
  platform_length 360 45 56 = 340 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_platform_length_is_340_l3328_332818


namespace NUMINAMATH_CALUDE_lawsuit_probability_difference_l3328_332881

def probability_win_lawsuit1 : ℝ := 0.3
def probability_win_lawsuit2 : ℝ := 0.5

theorem lawsuit_probability_difference :
  (1 - probability_win_lawsuit1) * (1 - probability_win_lawsuit2) - 
  (probability_win_lawsuit1 * probability_win_lawsuit2) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_lawsuit_probability_difference_l3328_332881


namespace NUMINAMATH_CALUDE_escalator_rate_calculation_l3328_332842

/-- The rate at which the escalator moves, in feet per second -/
def escalator_rate : ℝ := 11

/-- The length of the escalator, in feet -/
def escalator_length : ℝ := 140

/-- The rate at which the person walks, in feet per second -/
def person_walking_rate : ℝ := 3

/-- The time taken by the person to cover the entire length, in seconds -/
def time_taken : ℝ := 10

theorem escalator_rate_calculation :
  (person_walking_rate + escalator_rate) * time_taken = escalator_length :=
by sorry

end NUMINAMATH_CALUDE_escalator_rate_calculation_l3328_332842


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l3328_332863

theorem min_sum_absolute_values :
  ∀ x : ℝ, |x + 3| + |x + 6| + |x + 7| ≥ 4 ∧
  ∃ x : ℝ, |x + 3| + |x + 6| + |x + 7| = 4 :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l3328_332863


namespace NUMINAMATH_CALUDE_third_test_score_l3328_332893

def maria_scores (score3 : ℝ) : List ℝ := [80, 70, score3, 100]

theorem third_test_score (score3 : ℝ) : 
  (maria_scores score3).sum / (maria_scores score3).length = 85 → score3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_third_test_score_l3328_332893


namespace NUMINAMATH_CALUDE_jake_final_bitcoin_count_l3328_332862

def bitcoin_transactions (initial : ℕ) (investment : ℕ) (donation1 : ℕ) (debt : ℕ) (donation2 : ℕ) : ℕ :=
  let after_investment := initial - investment + 2 * investment
  let after_donation1 := after_investment - donation1
  let after_sharing := after_donation1 - (after_donation1 / 2)
  let after_debt_collection := after_sharing + debt
  let after_quadrupling := 4 * after_debt_collection
  after_quadrupling - donation2

theorem jake_final_bitcoin_count :
  bitcoin_transactions 120 40 25 5 15 = 277 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_bitcoin_count_l3328_332862


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l3328_332865

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : (angles 0) + (angles 1) + (angles 2) = 180
  all_positive : ∀ i, 0 < angles i

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) :
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) :=
sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l3328_332865


namespace NUMINAMATH_CALUDE_cab_driver_income_l3328_332898

theorem cab_driver_income (income1 income2 income3 income4 : ℕ) (average : ℚ) :
  income1 = 45 →
  income2 = 50 →
  income3 = 60 →
  income4 = 65 →
  average = 58 →
  ∃ income5 : ℕ, 
    (income1 + income2 + income3 + income4 + income5 : ℚ) / 5 = average ∧
    income5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3328_332898


namespace NUMINAMATH_CALUDE_min_value_theorem_l3328_332877

theorem min_value_theorem (x y a : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) :
  (∀ x y, x + 2*y = 1 → (3/x + a/y) ≥ 6*Real.sqrt 3) ∧
  (∃ x y, x + 2*y = 1 ∧ 3/x + a/y = 6*Real.sqrt 3) →
  (∀ x y, 1/x + 2/y = 1 → 3*x + a*y ≥ 6*Real.sqrt 3) ∧
  (∃ x y, 1/x + 2/y = 1 ∧ 3*x + a*y = 6*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3328_332877


namespace NUMINAMATH_CALUDE_print_shop_charges_l3328_332811

/-- 
Given:
- Print shop X charges $1.25 per color copy
- Print shop Y charges $60 more than print shop X for 40 color copies

Prove that print shop Y charges $2.75 per color copy
-/
theorem print_shop_charges (x y : ℝ) : 
  x = 1.25 → 
  40 * y = 40 * x + 60 → 
  y = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charges_l3328_332811


namespace NUMINAMATH_CALUDE_D_is_empty_l3328_332872

-- Define the set D
def D : Set ℝ := {x : ℝ | x^2 + 2 = 0}

-- Theorem stating that D is an empty set
theorem D_is_empty : D = ∅ := by sorry

end NUMINAMATH_CALUDE_D_is_empty_l3328_332872


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3328_332844

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3328_332844


namespace NUMINAMATH_CALUDE_tanning_time_proof_l3328_332870

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
def remaining_tanning_time (monthly_limit : ℕ) (week1_time : ℕ) (week2_time : ℕ) : ℕ :=
  monthly_limit - (week1_time + week2_time)

/-- Proves that given the specified tanning times, the remaining time is 45 minutes. -/
theorem tanning_time_proof : remaining_tanning_time 200 75 80 = 45 := by
  sorry

end NUMINAMATH_CALUDE_tanning_time_proof_l3328_332870


namespace NUMINAMATH_CALUDE_kettle_cannot_fill_claimed_cups_l3328_332829

-- Define the volume of water in the kettle in liters
def kettle_volume : ℝ := 2.5

-- Define the volume of each cup in milliliters
def cup_volume : ℝ := 250

-- Define the number of cups claimed to be filled
def claimed_cups : ℕ := 100

-- Define the conversion rate from liters to milliliters
def liters_to_milliliters : ℝ := 1000

-- Theorem stating that the kettle cannot fill the claimed number of cups
theorem kettle_cannot_fill_claimed_cups : 
  (kettle_volume * liters_to_milliliters) / cup_volume ≠ claimed_cups := by
  sorry

end NUMINAMATH_CALUDE_kettle_cannot_fill_claimed_cups_l3328_332829


namespace NUMINAMATH_CALUDE_lucy_calculation_l3328_332871

theorem lucy_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 13) 
  (h2 : x - y - z = -1) : 
  x - y = 6 := by sorry

end NUMINAMATH_CALUDE_lucy_calculation_l3328_332871


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3328_332876

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 - x^2/4 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = x/2 ∨ y = -x/2

-- Theorem statement
theorem hyperbola_properties :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a : ℝ), hyperbola_equation 0 a) ∧
  (∀ (x' y' : ℝ), x' ≠ 0 → asymptote_equation x' y' → 
    ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), δ > ε → 
      ∃ (x'' y'' : ℝ), hyperbola_equation x'' y'' ∧ 
      abs (x'' - x') < δ ∧ abs (y'' - y') < δ) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3328_332876


namespace NUMINAMATH_CALUDE_older_brother_age_l3328_332867

theorem older_brother_age (father_age : ℕ) (n : ℕ) (x : ℕ) : 
  father_age = 50 ∧ 
  2 * (x + n) = father_age + n ∧
  x + n ≤ father_age →
  x + n = 25 :=
by sorry

end NUMINAMATH_CALUDE_older_brother_age_l3328_332867


namespace NUMINAMATH_CALUDE_line_equation_l3328_332809

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point
def given_point : Point := { x := 3, y := -2 }

-- Theorem stating the line equation
theorem line_equation : 
  ∃ (l1 l2 : Line), 
    (point_on_line given_point l1 ∧ equal_intercepts l1) ∧
    (point_on_line given_point l2 ∧ equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = 3 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3328_332809


namespace NUMINAMATH_CALUDE_triangle_properties_l3328_332821

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 = t.c^2 + t.a * t.b)
  (h2 : Real.sqrt 3 * t.c = 14 * Real.sin t.C)
  (h3 : t.a + t.b = 13) :
  t.C = π/3 ∧ t.c = 7 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3328_332821


namespace NUMINAMATH_CALUDE_calculate_expression_l3328_332861

theorem calculate_expression : 3000 * (3000^2999 - 3000^2998) = 3000^2999 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3328_332861


namespace NUMINAMATH_CALUDE_sequence_length_l3328_332833

/-- The number of terms in the sequence 1, 2³, 2⁶, 2⁹, ..., 2³ⁿ⁺⁶ -/
def num_terms (n : ℕ) : ℕ := n + 3

/-- The exponent of the k-th term in the sequence -/
def exponent (k : ℕ) : ℕ := 3 * (k - 1)

theorem sequence_length (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ exponent k = 3 * n + 6) → 
  num_terms n = (Finset.range (n + 3)).card :=
sorry

end NUMINAMATH_CALUDE_sequence_length_l3328_332833


namespace NUMINAMATH_CALUDE_length_of_AB_l3328_332806

-- Define the line l: kx + y - 2 = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C: x² + y² - 6x + 2y + 9 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that line l is the axis of symmetry for circle C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x - x')^2 + (y - y')^2 = (x' - 3)^2 + (y' + 1)^2))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a tangent line from A to circle C
def exists_tangent (k : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C x y ∧ 
    ((x - 0)^2 + (y - k)^2) * ((x - 3)^2 + (y + 1)^2) = 1

-- Theorem statement
theorem length_of_AB (k : ℝ) : 
  is_axis_of_symmetry k → exists_tangent k → 
  ∃ x y : ℝ, circle_C x y ∧ 
    Real.sqrt ((x - 0)^2 + (y - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l3328_332806


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l3328_332843

def f (x : ℝ) := x^3 + 5*x^2 + 8*x + 4

theorem cubic_inequality_solution :
  {x : ℝ | f x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l3328_332843


namespace NUMINAMATH_CALUDE_function_symmetry_l3328_332804

theorem function_symmetry (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l3328_332804


namespace NUMINAMATH_CALUDE_bill_difference_l3328_332803

theorem bill_difference : 
  ∀ (sarah_bill linda_bill : ℝ),
  sarah_bill * 0.15 = 3 →
  linda_bill * 0.25 = 3 →
  sarah_bill - linda_bill = 8 := by
sorry

end NUMINAMATH_CALUDE_bill_difference_l3328_332803


namespace NUMINAMATH_CALUDE_wallpaper_overlap_l3328_332852

theorem wallpaper_overlap (total_area : ℝ) (wall_area : ℝ) (three_layer_area : ℝ)
  (h1 : total_area = 300)
  (h2 : wall_area = 180)
  (h3 : three_layer_area = 40) :
  total_area - wall_area - three_layer_area = 80 :=
by sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_l3328_332852


namespace NUMINAMATH_CALUDE_insurance_slogan_equivalence_l3328_332854

-- Define the universe of people
variable (Person : Type)

-- Define predicates
variable (happy : Person → Prop)
variable (has_it : Person → Prop)

-- Theorem stating the logical equivalence
theorem insurance_slogan_equivalence :
  (∀ p : Person, happy p → has_it p) ↔ (∀ p : Person, ¬has_it p → ¬happy p) :=
sorry

end NUMINAMATH_CALUDE_insurance_slogan_equivalence_l3328_332854


namespace NUMINAMATH_CALUDE_cricket_overs_played_l3328_332851

/-- Proves that the number of overs played initially in a cricket game is 10, given the specified conditions --/
theorem cricket_overs_played (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) : 
  target = 242 ∧ initial_rate = 3.2 ∧ required_rate = 5.25 →
  ∃ x : ℝ, x = 10 ∧ target - initial_rate * x = required_rate * (50 - x) := by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_played_l3328_332851


namespace NUMINAMATH_CALUDE_expand_product_l3328_332848

theorem expand_product (x : ℝ) : (3 * x - 2) * (2 * x + 4) = 6 * x^2 + 8 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3328_332848


namespace NUMINAMATH_CALUDE_nested_square_root_18_l3328_332882

theorem nested_square_root_18 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (18 + x) → x = 6 := by sorry

end NUMINAMATH_CALUDE_nested_square_root_18_l3328_332882


namespace NUMINAMATH_CALUDE_ageOfReplacedManIs42_l3328_332836

/-- Given a group of 6 men where:
    - The average age increases by 3 years when two women replace two men
    - One of the men is 26 years old
    - The average age of the women is 34
    This function calculates the age of the other man who was replaced. -/
def ageOfReplacedMan (averageIncrease : ℕ) (knownManAge : ℕ) (womenAverageAge : ℕ) : ℕ :=
  2 * womenAverageAge - knownManAge

/-- Theorem stating that under the given conditions, 
    the age of the other replaced man is 42 years. -/
theorem ageOfReplacedManIs42 :
  ageOfReplacedMan 3 26 34 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ageOfReplacedManIs42_l3328_332836


namespace NUMINAMATH_CALUDE_binomial_probability_l3328_332875

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_probability (ξ : BinomialVariable) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_l3328_332875


namespace NUMINAMATH_CALUDE_problem_statement_l3328_332891

theorem problem_statement (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a + b, 0, a^2} → a^2016 + b^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3328_332891


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l3328_332856

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem two_digit_number_theorem (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ 
  (10 * x + y) - (10 * y + x) = 81 ∧ 
  is_prime (x + y) → 
  x - y = 7 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l3328_332856


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3328_332858

/-- Given a geometric sequence {aₙ} where all terms are positive,
    prove that a₅a₇a₉ = 12 when a₂a₄a₆ = 6 and a₈a₁₀a₁₂ = 24 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  a 2 * a 4 * a 6 = 6 →
  a 8 * a 10 * a 12 = 24 →
  a 5 * a 7 * a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3328_332858


namespace NUMINAMATH_CALUDE_count_valid_sums_l3328_332874

/-- The number of valid ways to sum to 5750 using 5's, 55's, and 555's -/
def num_valid_sums : ℕ := 124

/-- Predicate for a valid sum configuration -/
def is_valid_sum (a b c : ℕ) : Prop :=
  a + 11 * b + 111 * c = 1150

/-- The length of the original string of 5's -/
def string_length (a b c : ℕ) : ℕ :=
  a + 2 * b + 3 * c

/-- Theorem stating that there are exactly 124 valid string lengths -/
theorem count_valid_sums :
  (∃ (S : Finset ℕ), S.card = num_valid_sums ∧
    (∀ n, n ∈ S ↔ ∃ a b c, is_valid_sum a b c ∧ string_length a b c = n)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_sums_l3328_332874


namespace NUMINAMATH_CALUDE_campers_fed_specific_l3328_332860

/-- The number of campers that can be fed given the caught fish --/
def campers_fed (trout_weight : ℕ) (bass_count bass_weight : ℕ) (salmon_count salmon_weight : ℕ) (consumption_per_camper : ℕ) : ℕ :=
  (trout_weight + bass_count * bass_weight + salmon_count * salmon_weight) / consumption_per_camper

/-- Theorem stating the number of campers that can be fed given the specific fishing scenario --/
theorem campers_fed_specific : campers_fed 8 6 2 2 12 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_campers_fed_specific_l3328_332860


namespace NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l3328_332846

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = 12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l3328_332846


namespace NUMINAMATH_CALUDE_min_value_2n_plus_k_l3328_332850

theorem min_value_2n_plus_k (n k : ℕ) : 
  (144 + n) * 2 = n * k → -- total coins after sharing
  n > 0 → -- at least one person joins
  k > 0 → -- each person carries at least one coin
  2 * n + k ≥ 50 ∧ ∃ (n' k' : ℕ), 2 * n' + k' = 50 ∧ (144 + n') * 2 = n' * k' ∧ n' > 0 ∧ k' > 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_2n_plus_k_l3328_332850


namespace NUMINAMATH_CALUDE_valid_three_digit_count_l3328_332873

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 90

/-- The total count of three-digit numbers -/
def total_three_digit : ℕ := 900

/-- The count of three-digit numbers with exactly two different non-adjacent digits -/
def excluded_count : ℕ := 810

/-- Theorem stating that the count of valid three-digit numbers is correct -/
theorem valid_three_digit_count :
  valid_count = total_three_digit - excluded_count :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_count_l3328_332873


namespace NUMINAMATH_CALUDE_sweeties_leftover_l3328_332817

theorem sweeties_leftover (m : ℕ) (h : m % 12 = 11) : (4 * m) % 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sweeties_leftover_l3328_332817


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3328_332878

theorem arithmetic_calculation : (2^3 * 3 * 5) + (18 / 2) = 129 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3328_332878


namespace NUMINAMATH_CALUDE_smallest_k_product_equals_sum_l3328_332800

theorem smallest_k_product_equals_sum (k : ℕ) : k = 10 ↔ 
  (k ≥ 3 ∧ 
   ∃ a b : ℕ, a ∈ Finset.range k ∧ b ∈ Finset.range k ∧ a ≠ b ∧
   a * b = (k * (k + 1) / 2) - a - b ∧
   ∀ m : ℕ, m ≥ 3 → m < k → 
     ¬∃ x y : ℕ, x ∈ Finset.range m ∧ y ∈ Finset.range m ∧ x ≠ y ∧
     x * y = (m * (m + 1) / 2) - x - y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_product_equals_sum_l3328_332800


namespace NUMINAMATH_CALUDE_existence_of_close_pair_l3328_332899

-- Define a type for numbers between 0 and 1
def UnitInterval := {x : ℝ | 0 < x ∧ x < 1}

-- State the theorem
theorem existence_of_close_pair :
  ∀ (x y z : UnitInterval), ∃ (a b : UnitInterval), |a.val - b.val| ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_existence_of_close_pair_l3328_332899


namespace NUMINAMATH_CALUDE_infinitely_many_primes_not_ending_in_one_l3328_332830

/-- The set of prime numbers whose last digit is not 1 is infinite. -/
theorem infinitely_many_primes_not_ending_in_one : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 10 ≠ 1} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_not_ending_in_one_l3328_332830


namespace NUMINAMATH_CALUDE_trig_identity_l3328_332805

theorem trig_identity (α : ℝ) : 
  (Real.sin (α / 2))^6 - (Real.cos (α / 2))^6 = ((Real.sin α)^2 - 4) / 4 * Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3328_332805


namespace NUMINAMATH_CALUDE_grapefruit_orchards_l3328_332823

/-- Represents the number of orchards for each type of citrus fruit -/
structure CitrusOrchards where
  total : ℕ
  lemons : ℕ
  oranges : ℕ
  limes : ℕ
  grapefruits : ℕ
  mandarins : ℕ

/-- Theorem stating the number of grapefruit orchards given the conditions -/
theorem grapefruit_orchards (c : CitrusOrchards) : c.grapefruits = 6 :=
  by
  have h1 : c.total = 40 := sorry
  have h2 : c.lemons = 15 := sorry
  have h3 : c.oranges = 2 * c.lemons / 3 := sorry
  have h4 : c.limes = c.grapefruits := sorry
  have h5 : c.mandarins = c.grapefruits / 2 := sorry
  have h6 : c.total = c.lemons + c.oranges + c.limes + c.grapefruits + c.mandarins := sorry
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_l3328_332823


namespace NUMINAMATH_CALUDE_library_visitors_l3328_332889

def sunday_visitors (avg_non_sunday : ℕ) (avg_total : ℕ) (days_in_month : ℕ) : ℕ :=
  let sundays := (days_in_month + 6) / 7
  let non_sundays := days_in_month - sundays
  ((avg_total * days_in_month) - (avg_non_sunday * non_sundays)) / sundays

theorem library_visitors :
  sunday_visitors 240 285 30 = 510 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l3328_332889


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l3328_332884

theorem multiplication_addition_equality : 21 * 47 + 21 * 53 = 2100 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l3328_332884


namespace NUMINAMATH_CALUDE_marbles_remaining_proof_l3328_332886

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of marbles remaining after sales -/
def remaining_marbles (initial : ℕ) (customers : ℕ) : ℕ :=
  initial - sum_to_n customers

theorem marbles_remaining_proof :
  remaining_marbles 2500 50 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_remaining_proof_l3328_332886


namespace NUMINAMATH_CALUDE_unique_d_for_single_solution_l3328_332895

theorem unique_d_for_single_solution :
  ∃! (d : ℝ), d ≠ 0 ∧
  (∃! (a : ℝ), a > 0 ∧
    (∃! (x : ℝ), x^2 + (a + 1/a) * x + d = 0)) ∧
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_d_for_single_solution_l3328_332895


namespace NUMINAMATH_CALUDE_fraction_equality_l3328_332815

theorem fraction_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_neq_xy : x ≠ y) (h_neq_yz : y ≠ z) (h_neq_xz : x ≠ z)
  (h_eq1 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h_eq2 : (y + 1) / (x + z) = (x + 1) / y) :
  (x + 1) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3328_332815


namespace NUMINAMATH_CALUDE_geometric_sequence_m_value_l3328_332832

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_m_value
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : q ≠ 1 ∧ q ≠ -1)
  (h3 : a 1 = -1)
  (h4 : ∃ m : ℕ, a m = a 1 * a 2 * a 3 * a 4 * a 5) :
  ∃ m : ℕ, m = 11 ∧ a m = a 1 * a 2 * a 3 * a 4 * a 5 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_m_value_l3328_332832


namespace NUMINAMATH_CALUDE_unique_positive_integer_cube_less_than_triple_l3328_332827

theorem unique_positive_integer_cube_less_than_triple :
  ∃! (n : ℕ), n > 0 ∧ n^3 < 3*n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_cube_less_than_triple_l3328_332827


namespace NUMINAMATH_CALUDE_exchange_rate_solution_l3328_332896

/-- Represents the exchange rate problem with Jack's currency amounts --/
def ExchangeRateProblem (pounds euros yen : ℕ) (yenPerPound : ℕ) (totalYen : ℕ) :=
  ∃ (poundsPerEuro : ℚ),
    (pounds : ℚ) * yenPerPound + euros * poundsPerEuro * yenPerPound + yen = totalYen ∧
    poundsPerEuro = 2

/-- Theorem stating that the exchange rate is 2 pounds per euro --/
theorem exchange_rate_solution :
  ExchangeRateProblem 42 11 3000 100 9400 :=
by
  sorry


end NUMINAMATH_CALUDE_exchange_rate_solution_l3328_332896


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3328_332855

theorem rational_equation_solution : ∃ x : ℚ, 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) ∧ 
  x = 55/13 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3328_332855


namespace NUMINAMATH_CALUDE_product_sum_relation_l3328_332868

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 12) → (b = 10) → (b - a = 6) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3328_332868


namespace NUMINAMATH_CALUDE_like_terms_sum_l3328_332820

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (a b : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, a i j ≠ 0 ∧ b i j ≠ 0 → i = j

theorem like_terms_sum (m n : ℕ) :
  like_terms (fun i j => if i = 2 ∧ j = n then 7 else 0)
             (fun i j => if i = m ∧ j = 3 then -5 else 0) →
  m + n = 5 := by
sorry

end NUMINAMATH_CALUDE_like_terms_sum_l3328_332820


namespace NUMINAMATH_CALUDE_unique_number_ratio_l3328_332826

theorem unique_number_ratio : ∃! x : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_unique_number_ratio_l3328_332826


namespace NUMINAMATH_CALUDE_sum_of_roots_l3328_332816

/-- The function f(x) = x^3 + 3x^2 + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) = 1 and f(b) = 19, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3328_332816


namespace NUMINAMATH_CALUDE_books_from_second_shop_l3328_332897

theorem books_from_second_shop
  (books_first_shop : ℕ)
  (cost_first_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : books_first_shop = 42)
  (h2 : cost_first_shop = 520)
  (h3 : cost_second_shop = 248)
  (h4 : average_price = 12)
  : ∃ (books_second_shop : ℕ),
    (cost_first_shop + cost_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_second_shop = 22 := by
  sorry

#check books_from_second_shop

end NUMINAMATH_CALUDE_books_from_second_shop_l3328_332897


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_real_line_l3328_332840

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_real_line 
  (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_real_line_l3328_332840


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3328_332801

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24365)
  (b_div_5 : b % 5 = 0)
  (b_div_10_eq_2a : b / 10 = 2 * a) :
  b - a = 19931 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3328_332801


namespace NUMINAMATH_CALUDE_number_puzzle_l3328_332812

theorem number_puzzle (x y : ℝ) : x = 95 → (x / 5 + y = 42) → y = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3328_332812


namespace NUMINAMATH_CALUDE_joshuas_skittles_l3328_332838

/-- Given that Joshua gave 40.0 Skittles to each of his 5.0 friends,
    prove that the total number of Skittles his friends have is 200.0. -/
theorem joshuas_skittles (skittles_per_friend : ℝ) (num_friends : ℝ) 
    (h1 : skittles_per_friend = 40.0)
    (h2 : num_friends = 5.0) : 
  skittles_per_friend * num_friends = 200.0 := by
  sorry

end NUMINAMATH_CALUDE_joshuas_skittles_l3328_332838


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3328_332869

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 24) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 8) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3328_332869


namespace NUMINAMATH_CALUDE_parallelogram_area_l3328_332866

/-- The area of a parallelogram with base length 3 and height 3 is 9 square units. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 3 → 
  height = 3 → 
  area = base * height → 
  area = 9 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3328_332866


namespace NUMINAMATH_CALUDE_max_distance_sum_l3328_332888

/-- Given m ∈ ℝ, and lines l₁ and l₂ passing through points A and B respectively,
    and intersecting at point P ≠ A, B, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, 3)
  let l₁ := {(x, y) : ℝ × ℝ | x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m * x - y - 2 * m + 3 = 0}
  ∀ P : ℝ × ℝ, P ∈ l₁ ∩ l₂ → P ≠ A → P ≠ B →
    ‖P - A‖ + ‖P - B‖ ≤ 2 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_max_distance_sum_l3328_332888
