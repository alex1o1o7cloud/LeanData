import Mathlib

namespace NUMINAMATH_GPT_rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l1762_176274

-- Define the digit constraints and the RD sum function
def is_digit (n : ℕ) : Prop := n < 10
def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def rd_sum (A B C D : ℕ) : ℕ :=
  let abcd := 1000 * A + 100 * B + 10 * C + D
  let dcba := 1000 * D + 100 * C + 10 * B + A
  abcd + dcba

-- Problem (a)
theorem rd_sum_4281 : rd_sum 4 2 8 1 = 6105 := sorry

-- Problem (b)
theorem rd_sum_formula (A B C D : ℕ) (hA : is_nonzero_digit A) (hD : is_nonzero_digit D) :
  ∃ m n, m = 1001 ∧ n = 110 ∧ rd_sum A B C D = m * (A + D) + n * (B + C) :=
  sorry

-- Problem (c)
theorem rd_sum_count_3883 :
  ∃ n, n = 18 ∧ ∃ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D ∧ rd_sum A B C D = 3883 :=
  sorry

-- Problem (d)
theorem count_self_equal_rd_sum : 
  ∃ n, n = 143 ∧ ∀ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D → (1001 * (A + D) + 110 * (B + C) ≤ 9999 → (1000 * A + 100 * B + 10 * C + D = rd_sum A B C D → 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ D ∧ D ≤ 9)) :=
  sorry

end NUMINAMATH_GPT_rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l1762_176274


namespace NUMINAMATH_GPT_negation_of_existence_implies_universal_l1762_176204

theorem negation_of_existence_implies_universal (x : ℝ) :
  (∀ x : ℝ, ¬(x^2 ≤ |x|)) ↔ (∀ x : ℝ, x^2 > |x|) :=
by 
  sorry

end NUMINAMATH_GPT_negation_of_existence_implies_universal_l1762_176204


namespace NUMINAMATH_GPT_polynomial_expansion_l1762_176265

-- Definitions of the polynomials
def p (w : ℝ) : ℝ := 3 * w^3 + 4 * w^2 - 7
def q (w : ℝ) : ℝ := 2 * w^3 - 3 * w^2 + 1

-- Statement of the theorem
theorem polynomial_expansion (w : ℝ) : 
  (p w) * (q w) = 6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l1762_176265


namespace NUMINAMATH_GPT_angle_E_degree_l1762_176241

-- Given conditions
variables {E F G H : ℝ} -- degrees of the angles in quadrilateral EFGH

-- Condition 1: The angles satisfy a specific ratio
axiom angle_ratio : E = 3 * F ∧ E = 2 * G ∧ E = 6 * H

-- Condition 2: The sum of the angles in the quadrilateral is 360 degrees
axiom angle_sum : E + (E / 3) + (E / 2) + (E / 6) = 360

-- Prove the degree measure of angle E is 180 degrees
theorem angle_E_degree : E = 180 :=
by
  sorry

end NUMINAMATH_GPT_angle_E_degree_l1762_176241


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1762_176290

theorem quadratic_inequality_solution (a b : ℝ)
  (h1 : ∀ x, (x > -1 ∧ x < 2) ↔ ax^2 + x + b > 0) :
  a + b = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1762_176290


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_5_mod_15_l1762_176238

theorem least_five_digit_congruent_to_5_mod_15 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 15 = 5 ∧ n = 10010 := by
  sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_5_mod_15_l1762_176238


namespace NUMINAMATH_GPT_division_quotient_l1762_176243

theorem division_quotient (dividend divisor remainder quotient : ℕ)
  (H1 : dividend = 190)
  (H2 : divisor = 21)
  (H3 : remainder = 1)
  (H4 : dividend = divisor * quotient + remainder) : quotient = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_division_quotient_l1762_176243


namespace NUMINAMATH_GPT_missing_digit_l1762_176235

theorem missing_digit (B : ℕ) (h : B < 10) : 
  (15 ∣ (200 + 10 * B)) ↔ B = 1 ∨ B = 4 :=
by sorry

end NUMINAMATH_GPT_missing_digit_l1762_176235


namespace NUMINAMATH_GPT_florist_first_picking_l1762_176208

theorem florist_first_picking (x : ℝ) (h1 : 37.0 + x + 19.0 = 72.0) : x = 16.0 :=
by
  sorry

end NUMINAMATH_GPT_florist_first_picking_l1762_176208


namespace NUMINAMATH_GPT_hypotenuse_length_l1762_176219

theorem hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = Real.sqrt 5) (h₂ : b = Real.sqrt 12) : c = Real.sqrt 17 :=
by
  -- Proof not required, hence skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1762_176219


namespace NUMINAMATH_GPT_polar_to_rectangular_l1762_176222

noncomputable def curve_equation (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x = curve_equation θ * Real.cos θ ∧ y = curve_equation θ * Real.sin θ) :=
sorry

end NUMINAMATH_GPT_polar_to_rectangular_l1762_176222


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1762_176295

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a ≠ 5) (h2 : b ≠ -5) : ¬((a + b ≠ 0) ↔ (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l1762_176295


namespace NUMINAMATH_GPT_diamond_two_three_l1762_176267

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_GPT_diamond_two_three_l1762_176267


namespace NUMINAMATH_GPT_number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l1762_176223

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end NUMINAMATH_GPT_number_of_ways_to_make_125_quacks_using_1_5_25_125_coins_l1762_176223


namespace NUMINAMATH_GPT_exists_root_in_interval_l1762_176234

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem exists_root_in_interval :
  f 1 < 0 → f 1.5 > 0 → f 1.25 < 0 → ∃ x ∈ (Set.Ioo 1.25 1.5), f x = 0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_exists_root_in_interval_l1762_176234


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1762_176252

namespace ProofProblem

def M := { x : ℝ | x^2 < 4 }
def N := { x : ℝ | x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end ProofProblem

end NUMINAMATH_GPT_intersection_of_M_and_N_l1762_176252


namespace NUMINAMATH_GPT_min_range_of_three_test_takers_l1762_176203

-- Proposition: The minimum possible range in scores of the 3 test-takers
-- where the ranges of their scores in the 5 practice tests are 18, 26, and 32, is 76.
theorem min_range_of_three_test_takers (r1 r2 r3: ℕ) 
  (h1 : r1 = 18) (h2 : r2 = 26) (h3 : r3 = 32) : 
  (r1 + r2 + r3) = 76 := by
  sorry

end NUMINAMATH_GPT_min_range_of_three_test_takers_l1762_176203


namespace NUMINAMATH_GPT_cos_tan_values_l1762_176253

theorem cos_tan_values (α : ℝ) (h : Real.sin α = -1 / 2) :
  (∃ (quadrant : ℕ), 
    (quadrant = 3 ∧ Real.cos α = -Real.sqrt 3 / 2 ∧ Real.tan α = Real.sqrt 3 / 3) ∨ 
    (quadrant = 4 ∧ Real.cos α = Real.sqrt 3 / 2 ∧ Real.tan α = -Real.sqrt 3 / 3)) :=
sorry

end NUMINAMATH_GPT_cos_tan_values_l1762_176253


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1762_176263

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a^2) / (b^2))

theorem hyperbola_eccentricity {b : ℝ} (hb_pos : b > 0)
  (h_area : b = 1) :
  eccentricity 1 b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1762_176263


namespace NUMINAMATH_GPT_trigonometric_identity_l1762_176229

theorem trigonometric_identity (t : ℝ) : 
  5.43 * Real.cos (22 * Real.pi / 180 - t) * Real.cos (82 * Real.pi / 180 - t) +
  Real.cos (112 * Real.pi / 180 - t) * Real.cos (172 * Real.pi / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1762_176229


namespace NUMINAMATH_GPT_light_distance_200_years_l1762_176242

-- Define the distance light travels in one year.
def distance_one_year := 5870000000000

-- Define the scientific notation representation for distance in one year
def distance_one_year_sci := 587 * 10^10

-- Define the distance light travels in 200 years.
def distance_200_years := distance_one_year * 200

-- Define the expected distance in scientific notation for 200 years.
def expected_distance := 1174 * 10^12

-- The theorem stating the given condition and the conclusion to prove
theorem light_distance_200_years : distance_200_years = expected_distance :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_light_distance_200_years_l1762_176242


namespace NUMINAMATH_GPT_proof_A_cap_complement_B_l1762_176298

variable (A B U : Set ℕ) (h1 : A ⊆ U) (h2 : B ⊆ U)
variable (h3 : U = {1, 2, 3, 4})
variable (h4 : (U \ (A ∪ B)) = {4}) -- \ represents set difference, complement in the universal set
variable (h5 : B = {1, 2})

theorem proof_A_cap_complement_B : A ∩ (U \ B) = {3} := by
  sorry

end NUMINAMATH_GPT_proof_A_cap_complement_B_l1762_176298


namespace NUMINAMATH_GPT_inequality_problem_l1762_176200

variable (a b c : ℝ)

theorem inequality_problem (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l1762_176200


namespace NUMINAMATH_GPT_sqrt_conjecture_l1762_176256

theorem sqrt_conjecture (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + (1 / (n + 2)))) = ((n + 1) * Real.sqrt (1 / (n + 2))) :=
sorry

end NUMINAMATH_GPT_sqrt_conjecture_l1762_176256


namespace NUMINAMATH_GPT_find_a_l1762_176291

theorem find_a (a : ℝ) (h_pos : 0 < a) 
(h : a + a^2 = 6) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1762_176291


namespace NUMINAMATH_GPT_intersection_is_empty_l1762_176270

open Finset

namespace ComplementIntersection

-- Define the universal set U, sets M and N
def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {2, 4, 5}

-- The complement of M with respect to U
def complement_U_M : Finset ℕ := U \ M

-- The complement of N with respect to U
def complement_U_N : Finset ℕ := U \ N

-- The intersection of the complements
def intersection_complements : Finset ℕ := complement_U_M ∩ complement_U_N

-- The proof statement
theorem intersection_is_empty : intersection_complements = ∅ :=
by sorry

end ComplementIntersection

end NUMINAMATH_GPT_intersection_is_empty_l1762_176270


namespace NUMINAMATH_GPT_constant_term_of_expansion_l1762_176215

open BigOperators

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_of_expansion :
  ∑ r in Finset.range (6 + 1), binomialCoeff 6 r * (2^r * (x : ℚ)^r) / (x^3 : ℚ) = 160 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_of_expansion_l1762_176215


namespace NUMINAMATH_GPT_locus_centers_of_circles_l1762_176279

theorem locus_centers_of_circles (P : ℝ × ℝ) (a : ℝ) (a_pos : 0 < a):
  {O : ℝ × ℝ | dist O P = a} = {O : ℝ × ℝ | dist O P = a} :=
by
  sorry

end NUMINAMATH_GPT_locus_centers_of_circles_l1762_176279


namespace NUMINAMATH_GPT_distinct_arrangements_of_beads_l1762_176294

noncomputable def factorial (n : Nat) : Nat := if h : n = 0 then 1 else n * factorial (n - 1)

theorem distinct_arrangements_of_beads : 
  ∃ (arrangements : Nat), arrangements = factorial 8 / (8 * 2) ∧ arrangements = 2520 := 
by
  -- Sorry to skip the proof, only requiring the statement.
  sorry

end NUMINAMATH_GPT_distinct_arrangements_of_beads_l1762_176294


namespace NUMINAMATH_GPT_prime_factor_of_difference_l1762_176287

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA9 : A ≤ 9) (hC : 1 ≤ C) (hC9 : C ≤ 9) (hA_ne_C : A ≠ C) :
  ∃ p : ℕ, Prime p ∧ p = 3 ∧ p ∣ 3 * (100 * A + 10 * B + C - (100 * C + 10 * B + A)) := by
  sorry

end NUMINAMATH_GPT_prime_factor_of_difference_l1762_176287


namespace NUMINAMATH_GPT_cost_of_soap_per_year_l1762_176296

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end NUMINAMATH_GPT_cost_of_soap_per_year_l1762_176296


namespace NUMINAMATH_GPT_value_to_add_l1762_176225

theorem value_to_add (a b c n m : ℕ) (h₁ : a = 510) (h₂ : b = 4590) (h₃ : c = 105) (h₄ : n = 627) (h₅ : m = Nat.lcm a (Nat.lcm b c)) :
  m - n = 31503 :=
by
  sorry

end NUMINAMATH_GPT_value_to_add_l1762_176225


namespace NUMINAMATH_GPT_cost_of_apples_l1762_176226

def cost_per_kilogram (m : ℝ) : ℝ := m
def number_of_kilograms : ℝ := 3

theorem cost_of_apples (m : ℝ) : cost_per_kilogram m * number_of_kilograms = 3 * m :=
by
  unfold cost_per_kilogram number_of_kilograms
  sorry

end NUMINAMATH_GPT_cost_of_apples_l1762_176226


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1762_176292

theorem distance_between_A_and_B
  (vA vB D : ℝ)
  (hvB : vB = (3/2) * vA)
  (second_meeting_distance : 20 = D * 2 / 5) : 
  D = 50 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1762_176292


namespace NUMINAMATH_GPT_average_first_8_matches_l1762_176268

/--
Assume we have the following conditions:
1. The average score for 12 matches is 48 runs.
2. The average score for the last 4 matches is 64 runs.
Prove that the average score for the first 8 matches is 40 runs.
-/
theorem average_first_8_matches (A1 A2 : ℕ) :
  (A1 / 12 = 48) → 
  (A2 / 4 = 64) →
  ((A1 - A2) / 8 = 40) :=
by
  sorry

end NUMINAMATH_GPT_average_first_8_matches_l1762_176268


namespace NUMINAMATH_GPT_range_of_m_l1762_176250

open Real

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → log x ≤ x * exp (m^2 - m - 1)

theorem range_of_m : 
  {m : ℝ | satisfies_inequality m} = {m : ℝ | m ≤ 0 ∨ m ≥ 1} :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1762_176250


namespace NUMINAMATH_GPT_probability_no_practice_l1762_176233

def prob_has_practice : ℚ := 5 / 8

theorem probability_no_practice : 
  1 - prob_has_practice = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_probability_no_practice_l1762_176233


namespace NUMINAMATH_GPT_scooter_gain_percent_l1762_176283

theorem scooter_gain_percent 
  (purchase_price : ℝ) (repair_costs : ℝ) (selling_price : ℝ) 
  (h1 : purchase_price = 800) (h2 : repair_costs = 200) (h3 : selling_price = 1200) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_scooter_gain_percent_l1762_176283


namespace NUMINAMATH_GPT_f_eq_for_neg_l1762_176273

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x * (2^(-x) + 1) else x * (2^x + 1)

-- Theorem to prove
theorem f_eq_for_neg (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, 0 ≤ x → f x = x * (2^(-x) + 1)) :
  ∀ x : ℝ, x < 0 → f x = x * (2^x + 1) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_f_eq_for_neg_l1762_176273


namespace NUMINAMATH_GPT_man_average_interest_rate_l1762_176205

noncomputable def average_rate_of_interest (total_investment : ℝ) (rate1 rate2 rate_average : ℝ) 
    (x : ℝ) (same_return : (rate1 * (total_investment - x) = rate2 * x)) : Prop :=
  (rate_average = ((rate1 * (total_investment - x) + rate2 * x) / total_investment))

theorem man_average_interest_rate
    (total_investment : ℝ) 
    (rate1 : ℝ)
    (rate2 : ℝ)
    (rate_average : ℝ)
    (x : ℝ)
    (same_return : rate1 * (total_investment - x) = rate2 * x) :
    total_investment = 4500 ∧ rate1 = 0.04 ∧ rate2 = 0.06 ∧ x = 1800 ∧ rate_average = 0.048 → 
    average_rate_of_interest total_investment rate1 rate2 rate_average x same_return := 
by
  sorry

end NUMINAMATH_GPT_man_average_interest_rate_l1762_176205


namespace NUMINAMATH_GPT_polynomial_degree_l1762_176246

variable {P : Polynomial ℝ}

theorem polynomial_degree (h1 : ∀ x : ℝ, (x - 4) * P.eval (2 * x) = 4 * (x - 1) * P.eval x) (h2 : P.eval 0 ≠ 0) : P.degree = 2 := 
sorry

end NUMINAMATH_GPT_polynomial_degree_l1762_176246


namespace NUMINAMATH_GPT_opposite_of_neg_five_halves_l1762_176207

theorem opposite_of_neg_five_halves : -(- (5 / 2: ℝ)) = 5 / 2 :=
by
    sorry

end NUMINAMATH_GPT_opposite_of_neg_five_halves_l1762_176207


namespace NUMINAMATH_GPT_complement_M_l1762_176251

section ComplementSet

variable (x : ℝ)

def M : Set ℝ := {x | 1 / x < 1}

theorem complement_M : {x | 0 ≤ x ∧ x ≤ 1} = Mᶜ := sorry

end ComplementSet

end NUMINAMATH_GPT_complement_M_l1762_176251


namespace NUMINAMATH_GPT_probability_of_y_gt_2x_l1762_176278

noncomputable def probability_y_gt_2x : ℝ := 
  (∫ x in (0:ℝ)..(1000:ℝ), ∫ y in (2*x)..(2000:ℝ), (1 / (1000 * 2000) : ℝ)) * (1000 * 2000)

theorem probability_of_y_gt_2x : probability_y_gt_2x = 0.5 := sorry

end NUMINAMATH_GPT_probability_of_y_gt_2x_l1762_176278


namespace NUMINAMATH_GPT_total_chairs_calculation_l1762_176236

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end NUMINAMATH_GPT_total_chairs_calculation_l1762_176236


namespace NUMINAMATH_GPT_range_of_b_l1762_176254

theorem range_of_b (a b c m : ℝ) (h_ge_seq : c = b * b / a) (h_sum : a + b + c = m) (h_pos_a : a > 0) (h_pos_m : m > 0) : 
  (-m ≤ b ∧ b < 0) ∨ (0 < b ∧ b ≤ m / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1762_176254


namespace NUMINAMATH_GPT_surface_area_hemisphere_l1762_176201

theorem surface_area_hemisphere
  (r : ℝ)
  (h₁ : 4 * Real.pi * r^2 = 4 * Real.pi * r^2)
  (h₂ : Real.pi * r^2 = 3) :
  3 * Real.pi * r^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_hemisphere_l1762_176201


namespace NUMINAMATH_GPT_y_paid_per_week_l1762_176284

variable (x y z : ℝ)

-- Conditions
axiom h1 : x + y + z = 900
axiom h2 : x = 1.2 * y
axiom h3 : z = 0.8 * y

-- Theorem to prove
theorem y_paid_per_week : y = 300 := by
  sorry

end NUMINAMATH_GPT_y_paid_per_week_l1762_176284


namespace NUMINAMATH_GPT_flower_total_l1762_176271

theorem flower_total (H C D : ℕ) (h1 : H = 34) (h2 : H = C - 13) (h3 : C = D + 23) : 
  H + C + D = 105 :=
by 
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_flower_total_l1762_176271


namespace NUMINAMATH_GPT_number_of_people_who_selected_dog_l1762_176239

theorem number_of_people_who_selected_dog 
  (total : ℕ) 
  (cat : ℕ) 
  (fish : ℕ) 
  (bird : ℕ) 
  (other : ℕ) 
  (h_total : total = 90) 
  (h_cat : cat = 25) 
  (h_fish : fish = 10) 
  (h_bird : bird = 15) 
  (h_other : other = 5) :
  (total - (cat + fish + bird + other) = 35) :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_who_selected_dog_l1762_176239


namespace NUMINAMATH_GPT_dinosaur_book_cost_l1762_176211

-- Define the constants for costs and savings/needs
def dict_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def needed : ℕ := 29
def total_cost : ℕ := savings + needed
def dino_cost : ℕ := 19

-- Mathematical statement to prove
theorem dinosaur_book_cost :
  dict_cost + dino_cost + cookbook_cost = total_cost :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_dinosaur_book_cost_l1762_176211


namespace NUMINAMATH_GPT_food_initially_meant_to_last_22_days_l1762_176247

variable (D : ℕ)   -- Denoting the initial number of days the food was meant to last
variable (m : ℕ := 760)  -- Initial number of men
variable (total_men : ℕ := 1520)  -- Total number of men after 2 days

-- The first condition derived from the problem: total amount of food
def total_food := m * D

-- The second condition derived from the problem: Remaining food after 2 days
def remaining_food_after_2_days := total_food - m * 2

-- The third condition derived from the problem: Remaining food to last for 10 more days
def remaining_food_to_last_10_days := total_men * 10

-- Statement to prove
theorem food_initially_meant_to_last_22_days :
  D - 2 = 10 →
  D = 22 :=
by
  sorry

end NUMINAMATH_GPT_food_initially_meant_to_last_22_days_l1762_176247


namespace NUMINAMATH_GPT_Harold_spending_l1762_176248

theorem Harold_spending
  (num_shirt_boxes : ℕ)
  (num_xl_boxes : ℕ)
  (wraps_shirt_boxes : ℕ)
  (wraps_xl_boxes : ℕ)
  (cost_per_roll : ℕ)
  (h1 : num_shirt_boxes = 20)
  (h2 : num_xl_boxes = 12)
  (h3 : wraps_shirt_boxes = 5)
  (h4 : wraps_xl_boxes = 3)
  (h5 : cost_per_roll = 4) :
  num_shirt_boxes / wraps_shirt_boxes + num_xl_boxes / wraps_xl_boxes * cost_per_roll = 32 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end NUMINAMATH_GPT_Harold_spending_l1762_176248


namespace NUMINAMATH_GPT_exists_hamiltonian_path_l1762_176249

theorem exists_hamiltonian_path (n : ℕ) (cities : Fin n → Type) (roads : ∀ (i j : Fin n), cities i → cities j → Prop) 
(road_one_direction : ∀ i j (c1 : cities i) (c2 : cities j), roads i j c1 c2 → ¬ roads j i c2 c1) :
∃ start : Fin n, ∃ path : Fin n → Fin n, ∀ i j : Fin n, i ≠ j → path i ≠ path j :=
sorry

end NUMINAMATH_GPT_exists_hamiltonian_path_l1762_176249


namespace NUMINAMATH_GPT_percentage_increase_in_sales_l1762_176231

theorem percentage_increase_in_sales (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  (∃ X : ℝ, (0.8 * (1 + X / 100) = 1.44) ∧ X = 80) :=
sorry

end NUMINAMATH_GPT_percentage_increase_in_sales_l1762_176231


namespace NUMINAMATH_GPT_people_believing_mostly_purple_l1762_176288

theorem people_believing_mostly_purple :
  ∀ (total : ℕ) (mostly_pink : ℕ) (both_mostly_pink_purple : ℕ) (neither : ℕ),
  total = 150 →
  mostly_pink = 80 →
  both_mostly_pink_purple = 40 →
  neither = 25 →
  (total - neither + both_mostly_pink_purple - mostly_pink) = 85 :=
by
  intros total mostly_pink both_mostly_pink_purple neither h_total h_mostly_pink h_both h_neither
  have people_identified_without_mostly_purple : ℕ := mostly_pink + both_mostly_pink_purple - mostly_pink + neither
  have leftover_people : ℕ := total - people_identified_without_mostly_purple
  have people_mostly_purple := both_mostly_pink_purple + leftover_people
  suffices people_mostly_purple = 85 by sorry
  sorry

end NUMINAMATH_GPT_people_believing_mostly_purple_l1762_176288


namespace NUMINAMATH_GPT_function_symmetry_l1762_176227

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 6))

theorem function_symmetry (ω : ℝ) (hω : ω > 0) (hT : (2 * Real.pi / ω) = 4 * Real.pi) :
  ∃ (k : ℤ), f ω (2 * k * Real.pi - Real.pi / 3) = f ω 0 := by
  sorry

end NUMINAMATH_GPT_function_symmetry_l1762_176227


namespace NUMINAMATH_GPT_inequality_holds_l1762_176293

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1762_176293


namespace NUMINAMATH_GPT_students_correct_answers_l1762_176259

theorem students_correct_answers
  (total_questions : ℕ)
  (correct_score per_question : ℕ)
  (incorrect_penalty : ℤ)
  (xiao_ming_score xiao_hong_score xiao_hua_score : ℤ)
  (xm_correct_answers xh_correct_answers xh_correct_answers : ℕ)
  (total : ℕ)
  (h_1 : total_questions = 10)
  (h_2 : correct_score = 10)
  (h_3 : incorrect_penalty = -3)
  (h_4 : xiao_ming_score = 87)
  (h_5 : xiao_hong_score = 74)
  (h_6 : xiao_hua_score = 9)
  (h_xm : xm_correct_answers = total_questions - (xiao_ming_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hong_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hua_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (expected : total = 20) :
  xm_correct_answers + xh_correct_answers + xh_correct_answers = total := 
sorry

end NUMINAMATH_GPT_students_correct_answers_l1762_176259


namespace NUMINAMATH_GPT_value_divided_by_3_l1762_176285

-- Given condition
def given_condition (x : ℕ) : Prop := x - 39 = 54

-- Correct answer we need to prove
theorem value_divided_by_3 (x : ℕ) (h : given_condition x) : x / 3 = 31 := 
by
  sorry

end NUMINAMATH_GPT_value_divided_by_3_l1762_176285


namespace NUMINAMATH_GPT_num_divisors_of_30_l1762_176277

theorem num_divisors_of_30 : 
  (∀ n : ℕ, n > 0 → (30 = 2^1 * 3^1 * 5^1) → (∀ k : ℕ, 0 < k ∧ k ∣ 30 → ∃ m : ℕ, k = 2^m ∧ k ∣ 30)) → 
  ∃ num_divisors : ℕ, num_divisors = 8 := 
by 
  sorry

end NUMINAMATH_GPT_num_divisors_of_30_l1762_176277


namespace NUMINAMATH_GPT_area_arccos_cos_eq_pi_sq_l1762_176212

noncomputable def area_bounded_by_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq_pi_sq :
  area_bounded_by_arccos_cos = Real.pi ^ 2 :=
sorry

end NUMINAMATH_GPT_area_arccos_cos_eq_pi_sq_l1762_176212


namespace NUMINAMATH_GPT_correct_factorization_l1762_176216

theorem correct_factorization : 
  (¬ (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3)) ∧ 
  (¬ (x^2 + 2 * x + 1 = x * (x^2 + 2) + 1)) ∧ 
  (¬ ((x + 2) * (x - 3) = x^2 - x - 6)) ∧ 
  (x^2 - 9 = (x - 3) * (x + 3)) :=
by 
  sorry

end NUMINAMATH_GPT_correct_factorization_l1762_176216


namespace NUMINAMATH_GPT_y_order_of_quadratic_l1762_176255

theorem y_order_of_quadratic (k : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = (-4)^2 + 4 * (-4) + k) → 
  (y2 = (-1)^2 + 4 * (-1) + k) → 
  (y3 = (1)^2 + 4 * (1) + k) → 
  y2 < y1 ∧ y1 < y3 :=
by
  intro hy1 hy2 hy3
  sorry

end NUMINAMATH_GPT_y_order_of_quadratic_l1762_176255


namespace NUMINAMATH_GPT_ordered_pairs_condition_l1762_176261

theorem ordered_pairs_condition (m n : ℕ) (hmn : m ≥ n) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 3 * m * n = 8 * (m + n - 1)) :
    (m, n) = (16, 3) ∨ (m, n) = (6, 4) := by
  sorry

end NUMINAMATH_GPT_ordered_pairs_condition_l1762_176261


namespace NUMINAMATH_GPT_find_circle_equation_l1762_176244

noncomputable def center_of_parabola : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

noncomputable def equation_of_circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1

theorem find_circle_equation 
  (center_c : ℝ × ℝ := center_of_parabola)
  (tangent : ∀ x y, tangent_line x y → (x - 1) ^ 2 + (y - 0) ^ 2 = 1) :
  equation_of_circle = (fun x y => sorry) :=
sorry

end NUMINAMATH_GPT_find_circle_equation_l1762_176244


namespace NUMINAMATH_GPT_multiplication_result_l1762_176206

theorem multiplication_result :
  10 * 9.99 * 0.999 * 100 = (99.9)^2 := 
by
  sorry

end NUMINAMATH_GPT_multiplication_result_l1762_176206


namespace NUMINAMATH_GPT_minimum_stamps_combination_l1762_176202

theorem minimum_stamps_combination (c f : ℕ) (h : 3 * c + 4 * f = 30) :
  c + f = 8 :=
sorry

end NUMINAMATH_GPT_minimum_stamps_combination_l1762_176202


namespace NUMINAMATH_GPT_find_n_l1762_176260

theorem find_n (x : ℝ) (h1 : x = 596.95) (h2 : ∃ n : ℝ, n + 11.95 - x = 3054) : ∃ n : ℝ, n = 3639 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1762_176260


namespace NUMINAMATH_GPT_calc_expression_solve_system_inequalities_l1762_176264

-- Proof Problem 1: Calculation
theorem calc_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2 / 3) ^ (-2 : ℤ) = -5 / 4 := 
by 
  sorry

-- Proof Problem 2: System of Inequalities Solution
variable (m : ℝ)
variable (x : ℝ)
  
theorem solve_system_inequalities (h : m < 0) : 
  (4 * x - 1 > x - 7) ∧ (-1 / 4 * x < 3 / 2 * m - 1) → x > 4 - 6 * m := 
by 
  sorry

end NUMINAMATH_GPT_calc_expression_solve_system_inequalities_l1762_176264


namespace NUMINAMATH_GPT_initial_contestants_proof_l1762_176257

noncomputable def initial_contestants (final_round : ℕ) : ℕ :=
  let fraction_remaining := 2 / 5
  let fraction_advancing := 1 / 2
  let fraction_final := fraction_remaining * fraction_advancing
  (final_round : ℕ) / fraction_final

theorem initial_contestants_proof : initial_contestants 30 = 150 :=
sorry

end NUMINAMATH_GPT_initial_contestants_proof_l1762_176257


namespace NUMINAMATH_GPT_coin_flip_sequences_l1762_176214

theorem coin_flip_sequences :
  let total_sequences := 2^10
  let sequences_starting_with_two_heads := 2^8
  total_sequences - sequences_starting_with_two_heads = 768 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_sequences_l1762_176214


namespace NUMINAMATH_GPT_sum_of_first_10_terms_of_arithmetic_sequence_l1762_176224

theorem sum_of_first_10_terms_of_arithmetic_sequence :
  ∀ (a n : ℕ) (a₁ : ℤ) (d : ℤ),
  (d = -2) →
  (a₇ : ℤ := a₁ + 6 * d) →
  (a₃ : ℤ := a₁ + 2 * d) →
  (a₁₀ : ℤ := a₁ + 9 * d) →
  (a₇ * a₇ = a₃ * a₁₀) →
  (S₁₀ : ℤ := 10 * a₁ + 45 * d) →
  S₁₀ = 270 :=
by
  intros a n a₁ d hd ha₇ ha₃ ha₁₀ hgm hS₁₀
  sorry

end NUMINAMATH_GPT_sum_of_first_10_terms_of_arithmetic_sequence_l1762_176224


namespace NUMINAMATH_GPT_max_value_of_n_l1762_176282

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variable (S_2015_pos : S 2015 > 0)
variable (S_2016_neg : S 2016 < 0)

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (S_2015_pos : S 2015 > 0)
  (S_2016_neg : S 2016 < 0) : 
  ∃ n, n = 1008 ∧ ∀ m, S m < S n := 
sorry

end NUMINAMATH_GPT_max_value_of_n_l1762_176282


namespace NUMINAMATH_GPT_square_area_twice_triangle_perimeter_l1762_176289

noncomputable def perimeter_of_triangle (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length * side_length

theorem square_area_twice_triangle_perimeter (a b c : ℕ) (h1 : perimeter_of_triangle a b c = 22) (h2 : a = 5) (h3 : b = 7) (h4 : c = 10) : area_of_square (side_length_of_square (2 * perimeter_of_triangle a b c)) = 121 :=
by
  sorry

end NUMINAMATH_GPT_square_area_twice_triangle_perimeter_l1762_176289


namespace NUMINAMATH_GPT_instantaneous_velocity_at_1_l1762_176237

noncomputable def S (t : ℝ) : ℝ := t^2 + 2 * t

theorem instantaneous_velocity_at_1 : (deriv S 1) = 4 :=
by 
  -- The proof is left as an exercise
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_1_l1762_176237


namespace NUMINAMATH_GPT_calculate_total_cost_l1762_176276

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

theorem calculate_total_cost :
  let total_items := num_sandwiches + num_sodas
  let cost_before_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  let discount := if total_items > discount_threshold then cost_before_discount * discount_rate else 0
  let final_cost := cost_before_discount - discount
  final_cost = 38.7 :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1762_176276


namespace NUMINAMATH_GPT_proof_problem_l1762_176217

noncomputable def problem_expression : ℝ :=
  50 * 39.96 * 3.996 * 500

theorem proof_problem : problem_expression = (3996 : ℝ)^2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1762_176217


namespace NUMINAMATH_GPT_crayon_count_l1762_176209

theorem crayon_count (initial_crayons eaten_crayons : ℕ) (h1 : initial_crayons = 62) (h2 : eaten_crayons = 52) : initial_crayons - eaten_crayons = 10 := 
by 
  sorry

end NUMINAMATH_GPT_crayon_count_l1762_176209


namespace NUMINAMATH_GPT_matt_homework_time_l1762_176299

variable (T : ℝ)
variable (h_math : 0.30 * T = math_time)
variable (h_science : 0.40 * T = science_time)
variable (h_others : math_time + science_time + 45 = T)

theorem matt_homework_time (h_math : 0.30 * T = math_time)
                             (h_science : 0.40 * T = science_time)
                             (h_others : math_time + science_time + 45 = T) :
  T = 150 := by
  sorry

end NUMINAMATH_GPT_matt_homework_time_l1762_176299


namespace NUMINAMATH_GPT_pythagorean_triple_example_l1762_176218

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_example :
  is_pythagorean_triple 7 24 25 :=
sorry

end NUMINAMATH_GPT_pythagorean_triple_example_l1762_176218


namespace NUMINAMATH_GPT_mean_after_removal_l1762_176240

variable {n : ℕ}
variable {S : ℝ}
variable {S' : ℝ}
variable {mean_original : ℝ}
variable {size_original : ℕ}
variable {x1 : ℝ}
variable {x2 : ℝ}

theorem mean_after_removal (h_mean_original : mean_original = 42)
    (h_size_original : size_original = 60)
    (h_x1 : x1 = 50)
    (h_x2 : x2 = 60)
    (h_S : S = mean_original * size_original)
    (h_S' : S' = S - (x1 + x2)) :
    S' / (size_original - 2) = 41.55 :=
by
  sorry

end NUMINAMATH_GPT_mean_after_removal_l1762_176240


namespace NUMINAMATH_GPT_solve_for_x2_minus_y2_minus_z2_l1762_176272

theorem solve_for_x2_minus_y2_minus_z2
  (x y z : ℝ)
  (h1 : x + y + z = 12)
  (h2 : x - y = 4)
  (h3 : y + z = 7) :
  x^2 - y^2 - z^2 = -12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x2_minus_y2_minus_z2_l1762_176272


namespace NUMINAMATH_GPT_intersect_at_0_intersect_at_180_intersect_at_90_l1762_176220

-- Define radii R and r, and the distance c
variables {R r c : ℝ}

-- Formalize the conditions and corresponding angles
theorem intersect_at_0 (h : c = R - r) : True := 
sorry

theorem intersect_at_180 (h : c = R + r) : True := 
sorry

theorem intersect_at_90 (h : c = Real.sqrt (R^2 + r^2)) : True := 
sorry

end NUMINAMATH_GPT_intersect_at_0_intersect_at_180_intersect_at_90_l1762_176220


namespace NUMINAMATH_GPT_pentagon_position_3010_l1762_176213

def rotate_72 (s : String) : String :=
match s with
| "ABCDE" => "EABCD"
| "EABCD" => "DCBAE"
| "DCBAE" => "EDABC"
| "EDABC" => "ABCDE"
| _ => s

def reflect_vertical (s : String) : String :=
match s with
| "EABCD" => "DCBAE"
| "DCBAE" => "EABCD"
| _ => s

def transform (s : String) (n : Nat) : String :=
match n % 5 with
| 0 => s
| 1 => reflect_vertical (rotate_72 s)
| 2 => rotate_72 (reflect_vertical (rotate_72 s))
| 3 => reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s)))
| 4 => rotate_72 (reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s))))
| _ => s

theorem pentagon_position_3010 :
  transform "ABCDE" 3010 = "ABCDE" :=
by 
  sorry

end NUMINAMATH_GPT_pentagon_position_3010_l1762_176213


namespace NUMINAMATH_GPT_volume_of_sphere_l1762_176275

theorem volume_of_sphere
    (area1 : ℝ) (area2 : ℝ) (distance : ℝ)
    (h1 : area1 = 9 * π)
    (h2 : area2 = 16 * π)
    (h3 : distance = 1) :
    ∃ R : ℝ, (4 / 3) * π * R ^ 3 = 500 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l1762_176275


namespace NUMINAMATH_GPT_solve_for_r_l1762_176280

theorem solve_for_r (r : ℤ) : 24 - 5 = 3 * r + 7 → r = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_r_l1762_176280


namespace NUMINAMATH_GPT_hyperbola_focus_l1762_176210

theorem hyperbola_focus (m : ℝ) :
  (∃ (F : ℝ × ℝ), F = (0, 5) ∧ F ∈ {P : ℝ × ℝ | ∃ x y : ℝ, 
  x = P.1 ∧ y = P.2 ∧ (y^2 / m - x^2 / 9 = 1)}) → 
  m = 16 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_l1762_176210


namespace NUMINAMATH_GPT_cos_double_alpha_proof_l1762_176281

theorem cos_double_alpha_proof (α : ℝ) (h1 : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = - 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_alpha_proof_l1762_176281


namespace NUMINAMATH_GPT_graduating_class_total_students_l1762_176221

theorem graduating_class_total_students (boys girls students : ℕ) (h1 : girls = boys + 69) (h2 : boys = 208) :
  students = boys + girls → students = 485 :=
by
  sorry

end NUMINAMATH_GPT_graduating_class_total_students_l1762_176221


namespace NUMINAMATH_GPT_find_x_l1762_176230

-- Define the condition as a Lean equation
def equation (x : ℤ) : Prop :=
  45 - (28 - (37 - (x - 19))) = 58

-- The proof statement: if the equation holds, then x = 15
theorem find_x (x : ℤ) (h : equation x) : x = 15 := by
  sorry

end NUMINAMATH_GPT_find_x_l1762_176230


namespace NUMINAMATH_GPT_total_spending_l1762_176228

theorem total_spending :
  let price_per_pencil := 0.20
  let tolu_pencils := 3
  let robert_pencils := 5
  let melissa_pencils := 2
  let tolu_cost := tolu_pencils * price_per_pencil
  let robert_cost := robert_pencils * price_per_pencil
  let melissa_cost := melissa_pencils * price_per_pencil
  let total_cost := tolu_cost + robert_cost + melissa_cost
  total_cost = 2.00 := by
  sorry

end NUMINAMATH_GPT_total_spending_l1762_176228


namespace NUMINAMATH_GPT_complete_square_transform_l1762_176297

theorem complete_square_transform (x : ℝ) :
  x^2 - 8 * x + 2 = 0 → (x - 4)^2 = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_transform_l1762_176297


namespace NUMINAMATH_GPT_find_n_from_binomial_variance_l1762_176262

variable (ξ : Type)
variable (n : ℕ)
variable (p : ℝ := 0.3)
variable (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p))

-- Given conditions
axiom binomial_distribution : p = 0.3 ∧ Var n p = 2.1

-- Prove n = 10
theorem find_n_from_binomial_variance (ξ : Type) (n : ℕ) (p : ℝ := 0.3) (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p)) :
  p = 0.3 ∧ Var n p = 2.1 → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_binomial_variance_l1762_176262


namespace NUMINAMATH_GPT_painter_total_cost_l1762_176266

-- Define the arithmetic sequence for house addresses
def south_side_arith_seq (n : ℕ) : ℕ := 5 + (n - 1) * 7
def north_side_arith_seq (n : ℕ) : ℕ := 6 + (n - 1) * 8

-- Define the counting of digits
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

-- Define the condition of painting cost for multiples of 10
def painting_cost (n : ℕ) : ℕ :=
  if n % 10 = 0 then 2 * digit_count n
  else digit_count n

-- Calculate total cost for side with given arithmetic sequence
def total_cost_for_side (side_arith_seq : ℕ → ℕ): ℕ :=
  List.range 25 |>.map (λ n => painting_cost (side_arith_seq (n + 1))) |>.sum

-- Main theorem to prove
theorem painter_total_cost : total_cost_for_side south_side_arith_seq + total_cost_for_side north_side_arith_seq = 147 := by
  sorry

end NUMINAMATH_GPT_painter_total_cost_l1762_176266


namespace NUMINAMATH_GPT_kite_diagonals_sum_l1762_176286

theorem kite_diagonals_sum (a b e f : ℝ) (h₁ : a ≥ b) 
    (h₂ : e < 2 * a) (h₃ : f < a + b) : 
    e + f < 2 * a + b := by 
    sorry

end NUMINAMATH_GPT_kite_diagonals_sum_l1762_176286


namespace NUMINAMATH_GPT_senior_year_allowance_more_than_twice_l1762_176269

noncomputable def middle_school_allowance : ℝ :=
  8 + 2

noncomputable def twice_middle_school_allowance : ℝ :=
  2 * middle_school_allowance

noncomputable def senior_year_increase : ℝ :=
  1.5 * middle_school_allowance

noncomputable def senior_year_allowance : ℝ :=
  middle_school_allowance + senior_year_increase

theorem senior_year_allowance_more_than_twice : 
  senior_year_allowance = twice_middle_school_allowance + 5 :=
by
  sorry

end NUMINAMATH_GPT_senior_year_allowance_more_than_twice_l1762_176269


namespace NUMINAMATH_GPT_lower_limit_tip_percentage_l1762_176245

namespace meal_tip

def meal_cost : ℝ := 35.50
def total_paid : ℝ := 40.825
def tip_limit : ℝ := 15

-- Define the lower limit tip percentage as the solution to the given conditions.
theorem lower_limit_tip_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 25 ∧ (meal_cost + (x / 100) * meal_cost = total_paid) → 
  x = tip_limit :=
sorry

end meal_tip

end NUMINAMATH_GPT_lower_limit_tip_percentage_l1762_176245


namespace NUMINAMATH_GPT_path_length_of_B_l1762_176232

noncomputable def lengthPathB (BC : ℝ) : ℝ :=
  let radius := BC
  let circumference := 2 * Real.pi * radius
  circumference

theorem path_length_of_B (BC : ℝ) (h : BC = 4 / Real.pi) : lengthPathB BC = 8 := by
  rw [lengthPathB, h]
  simp [Real.pi_ne_zero, div_mul_cancel]
  sorry

end NUMINAMATH_GPT_path_length_of_B_l1762_176232


namespace NUMINAMATH_GPT_find_y_eq_7_5_l1762_176258

theorem find_y_eq_7_5 (y : ℝ) (hy1 : 0 < y) (hy2 : ∃ z : ℤ, ((z : ℝ) ≤ y) ∧ (y < z + 1))
  (hy3 : (Int.floor y : ℝ) * y = 45) : y = 7.5 :=
sorry

end NUMINAMATH_GPT_find_y_eq_7_5_l1762_176258
