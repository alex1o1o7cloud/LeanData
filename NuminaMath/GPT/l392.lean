import Mathlib

namespace sequence_general_formula_l392_39290

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (rec : ∀ n : ℕ, n > 0 → a n = n * (a (n + 1) - a n)) : 
  ∀ n, a n = n := 
by 
  sorry

end sequence_general_formula_l392_39290


namespace calculation_is_one_l392_39229

noncomputable def calc_expression : ℝ :=
  (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (Real.pi / 3) - Real.sqrt 12

theorem calculation_is_one : calc_expression = 1 :=
by
  -- Each of the steps involved in calculating should match the problem's steps
  -- 1. (1/2)⁻¹ = 2
  -- 2. (2021 + π)^0 = 1
  -- 3. 4 * sin(π / 3) = 2√3 with sin(60°) = √3/2
  -- 4. sqrt(12) = 2√3
  -- Hence 2 - 1 + 2√3 - 2√3 = 1
  sorry

end calculation_is_one_l392_39229


namespace negation_proposition_l392_39255

theorem negation_proposition :
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proposition_l392_39255


namespace payal_book_length_l392_39249

theorem payal_book_length (P : ℕ) 
  (h1 : (2/3 : ℚ) * P = (1/3 : ℚ) * P + 20) : P = 60 :=
sorry

end payal_book_length_l392_39249


namespace rebecca_haircuts_l392_39266

-- Definitions based on the conditions
def charge_per_haircut : ℕ := 30
def charge_per_perm : ℕ := 40
def charge_per_dye_job : ℕ := 60
def dye_cost_per_job : ℕ := 10
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def tips : ℕ := 50
def total_amount : ℕ := 310

-- Define the unknown number of haircuts scheduled
variable (H : ℕ)

-- Statement of the proof problem
theorem rebecca_haircuts :
  charge_per_haircut * H + charge_per_perm * num_perms + charge_per_dye_job * num_dye_jobs
  - dye_cost_per_job * num_dye_jobs + tips = total_amount → H = 4 :=
by
  sorry

end rebecca_haircuts_l392_39266


namespace spelling_bee_initial_students_l392_39248

theorem spelling_bee_initial_students (x : ℕ) 
    (h1 : (2 / 3) * x = 2 / 3 * x)
    (h2 : (3 / 4) * ((1 / 3) * x) = 3 / 4 * (1 / 3 * x))
    (h3 : (1 / 3) * x * (1 / 4) = 30) : 
  x = 120 :=
sorry

end spelling_bee_initial_students_l392_39248


namespace four_consecutive_integers_divisible_by_12_l392_39299

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l392_39299


namespace minimize_expr_l392_39256

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b * c = 8)

-- Define the target expression and the proof goal
def expr := (3 * a + b) * (a + 3 * c) * (2 * b * c + 4)

-- Prove the main statement
theorem minimize_expr : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c = 8) ∧ expr a b c = 384 :=
sorry

end minimize_expr_l392_39256


namespace other_root_l392_39244

theorem other_root (m : ℤ) (h : (∀ x : ℤ, x^2 - x + m = 0 → (x = 2))) : (¬ ∃ y : ℤ, (y^2 - y + m = 0 ∧ y ≠ 2 ∧ y ≠ -1) ) := 
by {
  sorry
}

end other_root_l392_39244


namespace problem_1_problem_2_l392_39231

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l392_39231


namespace quadratic_roots_l392_39293

theorem quadratic_roots (p q r : ℝ) (h : p ≠ q) (k : ℝ) :
  (p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) →
  ((p * (q - r)) * k^2 + (q * (r - p)) * k + r * (p - q) = 0) →
  k = - (r * (p - q)) / (p * (q - r)) :=
by
  sorry

end quadratic_roots_l392_39293


namespace unique_m_for_prime_condition_l392_39208

theorem unique_m_for_prime_condition :
  ∃ (m : ℕ), m > 0 ∧ (∀ (p : ℕ), Prime p → (∀ (n : ℕ), ¬ p ∣ (n^m - m))) ↔ m = 1 :=
sorry

end unique_m_for_prime_condition_l392_39208


namespace calculation_result_l392_39254

theorem calculation_result :
  (-1) * (-4) + 2^2 / (7 - 5) = 6 :=
by
  sorry

end calculation_result_l392_39254


namespace quadratic_inequality_solution_l392_39287

theorem quadratic_inequality_solution (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c) * x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} :=
sorry

end quadratic_inequality_solution_l392_39287


namespace negation_of_proposition_true_l392_39257

theorem negation_of_proposition_true (a b : ℝ) : 
  (¬ ((a > b) → (∀ c : ℝ, c ^ 2 ≠ 0 → a * c ^ 2 > b * c ^ 2)) = true) :=
by
  sorry

end negation_of_proposition_true_l392_39257


namespace min_value_of_function_product_inequality_l392_39238

-- Part (1) Lean 4 statement
theorem min_value_of_function (x : ℝ) (hx : x > -1) : 
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 := 
by 
  sorry

-- Part (2) Lean 4 statement
theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) : 
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := 
by 
  sorry

end min_value_of_function_product_inequality_l392_39238


namespace spinner_probabilities_l392_39211

noncomputable def prob_A : ℚ := 1 / 3
noncomputable def prob_B : ℚ := 1 / 4
noncomputable def prob_C : ℚ := 5 / 18
noncomputable def prob_D : ℚ := 5 / 36

theorem spinner_probabilities :
  prob_A + prob_B + prob_C + prob_D = 1 ∧
  prob_C = 2 * prob_D :=
by {
  -- The statement of the theorem matches the given conditions and the correct answers.
  -- Proof will be provided later.
  sorry
}

end spinner_probabilities_l392_39211


namespace part_one_part_two_l392_39269

-- Defining the sequence {a_n} with the sum of the first n terms.
def S (n : ℕ) : ℕ := 3 * n ^ 2 + 10 * n

-- Defining a_n in terms of the sum S_n
def a (n : ℕ) : ℕ :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Defining the arithmetic sequence {b_n}
def b (n : ℕ) : ℕ := 3 * n + 2

-- Defining the sequence {c_n}
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n

-- Defining the sum of the first n terms of {c_n}
def T (n : ℕ) : ℕ :=
  (3 * n + 1) * 2^(n + 2) - 4

-- Theorem to prove general term formula for {b_n}
theorem part_one : ∀ n : ℕ, b n = 3 * n + 2 := 
by sorry

-- Theorem to prove the sum of the first n terms of {c_n}
theorem part_two (n : ℕ) : ∀ n : ℕ, T n = (3 * n + 1) * 2^(n + 2) - 4 :=
by sorry

end part_one_part_two_l392_39269


namespace number_of_white_balls_l392_39239

theorem number_of_white_balls (total_balls : ℕ) (red_prob black_prob : ℝ)
  (h_total : total_balls = 50)
  (h_red_prob : red_prob = 0.15)
  (h_black_prob : black_prob = 0.45) :
  ∃ (white_balls : ℕ), white_balls = 20 :=
by
  sorry

end number_of_white_balls_l392_39239


namespace dave_ice_cubes_total_l392_39258

theorem dave_ice_cubes_total : 
  let trayA_initial := 2
  let trayA_final := trayA_initial + 7
  let trayB := (1 / 3) * trayA_final
  let trayC := 2 * trayA_final
  trayA_final + trayB + trayC = 30 := by
  sorry

end dave_ice_cubes_total_l392_39258


namespace infinite_prime_set_exists_l392_39217

noncomputable def P : Set Nat := {p | Prime p ∧ ∃ m : Nat, p ∣ m^2 + 1}

theorem infinite_prime_set_exists :
  ∃ (P : Set Nat), (∀ p ∈ P, Prime p) ∧ (Set.Infinite P) ∧ 
  (∀ (p : Nat) (hp : p ∈ P) (k : ℕ),
    ∃ (m : Nat), p^k ∣ m^2 + 1 ∧ ¬(p^(k+1) ∣ m^2 + 1)) :=
sorry

end infinite_prime_set_exists_l392_39217


namespace exists_disjoint_A_B_l392_39298

def S (C : Finset ℕ) := C.sum id

theorem exists_disjoint_A_B : 
  ∃ (A B : Finset ℕ), 
  A ≠ ∅ ∧ B ≠ ∅ ∧ 
  A ∩ B = ∅ ∧ 
  A ∪ B = (Finset.range (2021 + 1)).erase 0 ∧ 
  ∃ k : ℕ, S A * S B = k^2 :=
by 
  sorry

end exists_disjoint_A_B_l392_39298


namespace linear_function_passes_through_point_l392_39283

theorem linear_function_passes_through_point :
  ∀ x y : ℝ, y = -2 * x - 6 → (x = -4 → y = 2) :=
by
  sorry

end linear_function_passes_through_point_l392_39283


namespace probability_two_red_or_blue_correct_l392_39246

noncomputable def probability_two_red_or_blue_sequential : ℚ := 1 / 5

theorem probability_two_red_or_blue_correct :
  let total_marbles := 15
  let red_blue_marbles := 7
  let first_draw_prob := (7 : ℚ) / 15
  let second_draw_prob := (6 : ℚ) / 14
  first_draw_prob * second_draw_prob = probability_two_red_or_blue_sequential :=
by
  sorry

end probability_two_red_or_blue_correct_l392_39246


namespace total_spent_l392_39268

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = 0.90 * B
def condition2 : Prop := B = D + 15

-- Question
theorem total_spent : condition1 B D ∧ condition2 B D → B + D = 285 := 
by
  intros h
  sorry

end total_spent_l392_39268


namespace tables_left_l392_39262

theorem tables_left (original_tables number_of_customers_per_table current_customers : ℝ) 
(h1 : original_tables = 44.0)
(h2 : number_of_customers_per_table = 8.0)
(h3 : current_customers = 256) : 
(original_tables - current_customers / number_of_customers_per_table) = 12.0 :=
by
  sorry

end tables_left_l392_39262


namespace change_is_correct_l392_39207

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def payment_given : ℕ := 500

-- Prices for different people in the family
def child_ticket_cost (age : ℕ) : ℕ :=
  if age < 12 then regular_ticket_cost - child_discount else regular_ticket_cost

def parent_ticket_cost : ℕ := regular_ticket_cost
def family_ticket_cost : ℕ :=
  (child_ticket_cost 6) + (child_ticket_cost 10) + parent_ticket_cost + parent_ticket_cost

def change_received : ℕ := payment_given - family_ticket_cost

-- Prove that the change received is 74
theorem change_is_correct : change_received = 74 :=
by sorry

end change_is_correct_l392_39207


namespace expand_and_simplify_l392_39281

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 4) + 6 = x^2 + x - 6 := by
  sorry

end expand_and_simplify_l392_39281


namespace intersection_of_lines_l392_39292

theorem intersection_of_lines :
  ∃ (x y : ℚ), (6 * x - 5 * y = 15) ∧ (8 * x + 3 * y = 1) ∧ x = 25 / 29 ∧ y = -57 / 29 :=
by
  sorry

end intersection_of_lines_l392_39292


namespace common_terms_sequence_l392_39234

-- Definitions of sequences
def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℤ := 2 ^ n
def c (n : ℕ) : ℤ := 2 ^ (2 * n - 1)

-- Theorem stating the conjecture
theorem common_terms_sequence :
  ∀ n : ℕ, ∃ m : ℕ, a m = b (2 * n - 1) :=
by
  sorry

end common_terms_sequence_l392_39234


namespace find_x_l392_39294

-- Define the angles as real numbers representing degrees.
variable (angle_SWR angle_WRU angle_x : ℝ)

-- Conditions given in the problem
def conditions (angle_SWR angle_WRU angle_x : ℝ) : Prop :=
  angle_SWR = 50 ∧ angle_WRU = 30 ∧ angle_SWR = angle_WRU + angle_x

-- Main theorem to prove that x = 20 given the conditions
theorem find_x (angle_SWR angle_WRU angle_x : ℝ) :
  conditions angle_SWR angle_WRU angle_x → angle_x = 20 := by
  sorry

end find_x_l392_39294


namespace hyperbola_eccentricity_l392_39270

theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h2 : ∀ c : ℝ, c - a^2 / c = 2 * a) :
  e = 1 + Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l392_39270


namespace rons_height_l392_39247

variable (R : ℝ)

theorem rons_height
  (depth_eq_16_ron_height : 16 * R = 208) :
  R = 13 :=
by {
  sorry
}

end rons_height_l392_39247


namespace union_eq_l392_39286

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l392_39286


namespace line_cannot_pass_through_third_quadrant_l392_39233

theorem line_cannot_pass_through_third_quadrant :
  ∀ (x y : ℝ), x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end line_cannot_pass_through_third_quadrant_l392_39233


namespace ratio_of_final_to_initial_l392_39230

theorem ratio_of_final_to_initial (P : ℝ) (R : ℝ) (T : ℝ) (hR : R = 0.02) (hT : T = 50) :
  let SI := P * R * T
  let A := P + SI
  A / P = 2 :=
by
  sorry

end ratio_of_final_to_initial_l392_39230


namespace how_many_people_in_group_l392_39228

-- Definition of the conditions
def ratio_likes_football : ℚ := 24 / 60
def ratio_plays_football_given_likes : ℚ := 1 / 2
def expected_to_play_football : ℕ := 50

-- Combining the ratios to get the fraction of total people playing football
def ratio_plays_football : ℚ := ratio_likes_football * ratio_plays_football_given_likes

-- Total number of people in the group
def total_people_in_group : ℕ := 250

-- Proof statement
theorem how_many_people_in_group (expected_to_play_football : ℕ) : 
  ratio_plays_football * total_people_in_group = expected_to_play_football :=
by {
  -- Directly using our definitions
  sorry
}

end how_many_people_in_group_l392_39228


namespace find_polynomial_R_l392_39260

-- Define the polynomials S(x), Q(x), and the remainder R(x)

noncomputable def S (x : ℝ) := 7 * x ^ 31 + 3 * x ^ 13 + 10 * x ^ 11 - 5 * x ^ 9 - 10 * x ^ 7 + 5 * x ^ 5 - 2
noncomputable def Q (x : ℝ) := x ^ 4 + x ^ 3 + x ^ 2 + x + 1
noncomputable def R (x : ℝ) := 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3

-- Statement of the proof
theorem find_polynomial_R :
  ∃ (P : ℝ → ℝ), ∀ x : ℝ, S x = P x * Q x + R x := sorry

end find_polynomial_R_l392_39260


namespace three_digit_number_is_473_l392_39205

theorem three_digit_number_is_473 (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) (h5 : 0 ≤ z) (h6 : z ≤ 9)
  (h7 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 99)
  (h8 : x + y + z = 14)
  (h9 : x + z = y) : 100 * x + 10 * y + z = 473 :=
by
  sorry

end three_digit_number_is_473_l392_39205


namespace min_value_quadratic_l392_39282

theorem min_value_quadratic : ∃ x : ℝ, ∀ y : ℝ, 3 * x ^ 2 - 18 * x + 2023 ≤ 3 * y ^ 2 - 18 * y + 2023 :=
sorry

end min_value_quadratic_l392_39282


namespace value_of_a3_minus_a2_l392_39224

theorem value_of_a3_minus_a2 : 
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S n = n^2) ∧ (S 3 - S 2 - (S 2 - S 1)) = 2) :=
sorry

end value_of_a3_minus_a2_l392_39224


namespace area_of_trapezoid_l392_39295

theorem area_of_trapezoid
  (r : ℝ)
  (AD BC : ℝ)
  (center_on_base : Bool)
  (height : ℝ)
  (area : ℝ)
  (inscribed_circle : r = 6)
  (base_AD : AD = 8)
  (base_BC : BC = 4)
  (K_height : height = 4 * Real.sqrt 2)
  (calc_area : area = (1 / 2) * (AD + BC) * height)
  : area = 32 * Real.sqrt 2 := by
  sorry

end area_of_trapezoid_l392_39295


namespace compare_solutions_l392_39240

theorem compare_solutions 
  (c d p q : ℝ) 
  (hc : c ≠ 0) 
  (hp : p ≠ 0) :
  (-d / c) < (-q / p) ↔ (q / p) < (d / c) :=
by
  sorry

end compare_solutions_l392_39240


namespace triangle_cos_C_correct_l392_39289

noncomputable def triangle_cos_C (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : ℝ :=
  Real.cos C -- This will be defined correctly in the proof phase.

theorem triangle_cos_C_correct (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : 
  triangle_cos_C A B C hABC hSinA hCosB = 16 / 65 :=
sorry

end triangle_cos_C_correct_l392_39289


namespace savings_fraction_l392_39245

variable (P : ℝ) -- worker's monthly take-home pay, assumed to be a real number
variable (f : ℝ) -- fraction of the take-home pay that she saves each month, assumed to be a real number

-- Condition: 12 times the fraction saved monthly should equal 8 times the amount not saved monthly.
axiom condition : 12 * f * P = 8 * (1 - f) * P

-- Prove: the fraction saved each month is 2/5
theorem savings_fraction : f = 2 / 5 := 
by
  sorry

end savings_fraction_l392_39245


namespace repeating_decimal_sum_l392_39253

theorem repeating_decimal_sum (x : ℚ) (hx : x = 45 / 99) (h_simplified : x = 5 / 11) : (5 + 11) = 16 :=
by {
  sorry
}

end repeating_decimal_sum_l392_39253


namespace no_real_b_for_inequality_l392_39227

theorem no_real_b_for_inequality (b : ℝ) :
  (∃ x : ℝ, |x^2 + 3*b*x + 4*b| ≤ 5 ∧ (∀ y : ℝ, |y^2 + 3*b*y + 4*b| ≤ 5 → y = x)) → false :=
by
  sorry

end no_real_b_for_inequality_l392_39227


namespace negate_universal_proposition_l392_39210

open Classical

def P (x : ℝ) : Prop := x^3 - 3*x > 0

theorem negate_universal_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end negate_universal_proposition_l392_39210


namespace smallest_X_divisible_by_60_l392_39250

/-
  Let \( T \) be a positive integer consisting solely of 0s and 1s.
  If \( X = \frac{T}{60} \) and \( X \) is an integer, prove that the smallest possible value of \( X \) is 185.
-/
theorem smallest_X_divisible_by_60 (T X : ℕ) 
  (hT_digit : ∀ d, d ∈ T.digits 10 → d = 0 ∨ d = 1) 
  (h1 : X = T / 60) 
  (h2 : T % 60 = 0) : 
  X = 185 :=
sorry

end smallest_X_divisible_by_60_l392_39250


namespace dogwood_trees_after_planting_l392_39215

-- Define the number of current dogwood trees and the number to be planted.
def current_dogwood_trees : ℕ := 34
def trees_to_be_planted : ℕ := 49

-- Problem statement to prove the total number of dogwood trees after planting.
theorem dogwood_trees_after_planting : current_dogwood_trees + trees_to_be_planted = 83 := by
  -- A placeholder for proof
  sorry

end dogwood_trees_after_planting_l392_39215


namespace range_of_m_l392_39297

noncomputable def G (x : ℝ) (m : ℝ) : ℝ := (8 * x ^ 2 + 24 * x + 5 * m) / 8

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ c : ℝ, G x m = (x + c) ^ 2 ∧ c ^ 2 = 3) → 4 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l392_39297


namespace expression_eval_l392_39203

theorem expression_eval :
  (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 :=
by
  sorry

end expression_eval_l392_39203


namespace other_root_of_quadratic_l392_39225

theorem other_root_of_quadratic (m : ℝ) (h : ∃ α : ℝ, α = 1 ∧ (3 * α^2 + m * α = 5)) :
  ∃ β : ℝ, β = -5 / 3 :=
by
  sorry

end other_root_of_quadratic_l392_39225


namespace trig_identity_cos_add_l392_39226

open Real

theorem trig_identity_cos_add (x : ℝ) (h1 : sin (π / 3 - x) = 3 / 5) (h2 : π / 2 < x ∧ x < π) :
  cos (x + π / 6) = 3 / 5 :=
by
  sorry

end trig_identity_cos_add_l392_39226


namespace abigail_score_l392_39265

theorem abigail_score (sum_20 : ℕ) (sum_21 : ℕ) (h1 : sum_20 = 1700) (h2 : sum_21 = 1806) : (sum_21 - sum_20) = 106 :=
by
  sorry

end abigail_score_l392_39265


namespace min_q_of_abs_poly_eq_three_l392_39209

theorem min_q_of_abs_poly_eq_three (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (|x1^2 + p * x1 + q| = 3) ∧ (|x2^2 + p * x2 + q| = 3) ∧ (|x3^2 + p * x3 + q| = 3)) →
  q = -3 :=
sorry

end min_q_of_abs_poly_eq_three_l392_39209


namespace groupD_can_form_triangle_l392_39223

def groupA := (5, 7, 12)
def groupB := (7, 7, 15)
def groupC := (6, 9, 16)
def groupD := (6, 8, 12)

def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem groupD_can_form_triangle : canFormTriangle 6 8 12 :=
by
  -- Proof of the above theorem will follow the example from the solution.
  sorry

end groupD_can_form_triangle_l392_39223


namespace earnings_difference_l392_39235

noncomputable def investment_ratio_a : ℕ := 3
noncomputable def investment_ratio_b : ℕ := 4
noncomputable def investment_ratio_c : ℕ := 5

noncomputable def return_ratio_a : ℕ := 6
noncomputable def return_ratio_b : ℕ := 5
noncomputable def return_ratio_c : ℕ := 4

noncomputable def total_earnings : ℕ := 2900

noncomputable def earnings_a (x y : ℕ) : ℚ := (investment_ratio_a * return_ratio_a * x * y) / 100
noncomputable def earnings_b (x y : ℕ) : ℚ := (investment_ratio_b * return_ratio_b * x * y) / 100

theorem earnings_difference (x y : ℕ) (h : (investment_ratio_a * return_ratio_a * x * y + investment_ratio_b * return_ratio_b * x * y + investment_ratio_c * return_ratio_c * x * y) / 100 = total_earnings) :
  earnings_b x y - earnings_a x y = 100 := by
  sorry

end earnings_difference_l392_39235


namespace mike_gave_4_marbles_l392_39271

noncomputable def marbles_given (original_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  original_marbles - remaining_marbles

theorem mike_gave_4_marbles (original_marbles remaining_marbles given_marbles : ℕ) 
  (h1 : original_marbles = 8) (h2 : remaining_marbles = 4) (h3 : given_marbles = marbles_given original_marbles remaining_marbles) : given_marbles = 4 :=
by
  sorry

end mike_gave_4_marbles_l392_39271


namespace Josephine_sold_10_liters_l392_39212

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l392_39212


namespace train_speed_l392_39275

/-
Problem Statement:
Prove that the speed of a train is 26.67 meters per second given:
  1. The length of the train is 320 meters.
  2. The time taken to cross the telegraph post is 12 seconds.
-/

theorem train_speed (distance time : ℝ) (h1 : distance = 320) (h2 : time = 12) :
  (distance / time) = 26.67 :=
by
  rw [h1, h2]
  norm_num
  sorry

end train_speed_l392_39275


namespace number_is_eight_l392_39264

theorem number_is_eight (x : ℤ) (h : x - 2 = 6) : x = 8 := 
sorry

end number_is_eight_l392_39264


namespace infinite_solutions_b_l392_39276

theorem infinite_solutions_b (x b : ℝ) : 
    (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) → b = -12 :=
by
  sorry

end infinite_solutions_b_l392_39276


namespace inscribed_circle_area_ratio_l392_39296

theorem inscribed_circle_area_ratio
  (R : ℝ) -- Radius of the original circle
  (r : ℝ) -- Radius of the inscribed circle
  (h : R = 3 * r) -- Relationship between the radii based on geometry problem
  :
  (π * R^2) / (π * r^2) = 9 :=
by sorry

end inscribed_circle_area_ratio_l392_39296


namespace mr_bird_exact_speed_l392_39261

-- Define the properties and calculating the exact speed
theorem mr_bird_exact_speed (d t : ℝ) (h1 : d = 50 * (t + 1 / 12)) (h2 : d = 70 * (t - 1 / 12)) :
  d / t = 58 :=
by 
  -- skipping the proof
  sorry

end mr_bird_exact_speed_l392_39261


namespace lift_ratio_l392_39202

theorem lift_ratio (total_weight first_lift second_lift : ℕ) (h1 : total_weight = 1500)
(h2 : first_lift = 600) (h3 : first_lift = 2 * (second_lift - 300)) : first_lift / second_lift = 1 := 
by
  sorry

end lift_ratio_l392_39202


namespace find_larger_number_l392_39243

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1000) 
  (h2 : L = 10 * S + 10) : 
  L = 1110 :=
sorry

end find_larger_number_l392_39243


namespace num_valid_k_values_l392_39252

theorem num_valid_k_values :
  ∃ (s : Finset ℕ), s = { 1, 2, 3, 6, 9, 18 } ∧ s.card = 6 :=
by
  sorry

end num_valid_k_values_l392_39252


namespace lucy_sales_is_43_l392_39218

def total_packs : Nat := 98
def robyn_packs : Nat := 55
def lucy_packs : Nat := total_packs - robyn_packs

theorem lucy_sales_is_43 : lucy_packs = 43 :=
by
  sorry

end lucy_sales_is_43_l392_39218


namespace shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l392_39219

noncomputable def k : ℝ := (1 / 20) * Real.log (1 / 4)
noncomputable def b : ℝ := Real.log 160
noncomputable def y (x : ℝ) : ℝ := Real.exp (k * x + b)

theorem shelf_life_at_30_degrees : y 30 = 20 := sorry

theorem temperature_condition_for_shelf_life (x : ℝ) : y x ≥ 80 → x ≤ 10 := sorry

end shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l392_39219


namespace train_speed_l392_39285

/-- A train that crosses a pole in a certain time of 7 seconds and is 210 meters long has a speed of 108 kilometers per hour. -/
theorem train_speed (time_to_cross: ℝ) (length_of_train: ℝ) (speed_kmh : ℝ) 
  (H_time: time_to_cross = 7) (H_length: length_of_train = 210) 
  (conversion_factor: ℝ := 3.6) : speed_kmh = 108 :=
by
  have speed_mps : ℝ := length_of_train / time_to_cross
  have speed_kmh_calc : ℝ := speed_mps * conversion_factor
  sorry

end train_speed_l392_39285


namespace probability_is_pi_over_12_l392_39200

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let radius := 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := 6 * 8
  circle_area / rectangle_area

theorem probability_is_pi_over_12 :
  probability_within_two_units_of_origin = Real.pi / 12 :=
by
  sorry

end probability_is_pi_over_12_l392_39200


namespace total_exercise_hours_l392_39214

-- Define the conditions
def Natasha_minutes_per_day : ℕ := 30
def Natasha_days : ℕ := 7
def Esteban_minutes_per_day : ℕ := 10
def Esteban_days : ℕ := 9
def Charlotte_monday_minutes : ℕ := 20
def Charlotte_wednesday_minutes : ℕ := 45
def Charlotte_thursday_minutes : ℕ := 30
def Charlotte_sunday_minutes : ℕ := 60

-- Sum up the minutes for each individual
def Natasha_total_minutes : ℕ := Natasha_minutes_per_day * Natasha_days
def Esteban_total_minutes : ℕ := Esteban_minutes_per_day * Esteban_days
def Charlotte_total_minutes : ℕ := Charlotte_monday_minutes + Charlotte_wednesday_minutes + Charlotte_thursday_minutes + Charlotte_sunday_minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Calculation of hours for each individual
noncomputable def Natasha_total_hours : ℚ := minutes_to_hours Natasha_total_minutes
noncomputable def Esteban_total_hours : ℚ := minutes_to_hours Esteban_total_minutes
noncomputable def Charlotte_total_hours : ℚ := minutes_to_hours Charlotte_total_minutes

-- Prove total hours of exercise for all three individuals
theorem total_exercise_hours : Natasha_total_hours + Esteban_total_hours + Charlotte_total_hours = 7.5833 := by
  sorry

end total_exercise_hours_l392_39214


namespace find_rate_l392_39237

-- Definitions of conditions
def Principal : ℝ := 2500
def Amount : ℝ := 3875
def Time : ℝ := 12

-- Main statement we are proving
theorem find_rate (P : ℝ) (A : ℝ) (T : ℝ) (R : ℝ) 
    (hP : P = Principal) 
    (hA : A = Amount) 
    (hT : T = Time) 
    (hR : R = (A - P) * 100 / (P * T)) : R = 55 / 12 := 
by 
  sorry

end find_rate_l392_39237


namespace product_of_all_possible_N_l392_39241

theorem product_of_all_possible_N (A B N : ℝ) 
  (h1 : A = B + N)
  (h2 : A - 4 = B + N - 4)
  (h3 : B + 5 = B + 5)
  (h4 : |((B + N - 4) - (B + 5))| = 1) :
  ∃ N₁ N₂ : ℝ, (|N₁ - 9| = 1 ∧ |N₂ - 9| = 1) ∧ N₁ * N₂ = 80 :=
by {
  -- We know the absolute value equation leads to two solutions
  -- hence we will consider N₁ and N₂ such that |N - 9| = 1
  -- which eventually yields N = 10 and N = 8, making their product 80.
  sorry
}

end product_of_all_possible_N_l392_39241


namespace ellipse_condition_necessary_not_sufficient_l392_39201

theorem ellipse_condition_necessary_not_sufficient {a b : ℝ} (h : a * b > 0):
  (∀ x y : ℝ, a * x^2 + b * y^2 = 1 → a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0) ∧ 
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) → a * b > 0) :=
sorry

end ellipse_condition_necessary_not_sufficient_l392_39201


namespace cost_of_chicken_l392_39232

theorem cost_of_chicken (cost_beef_per_pound : ℝ) (quantity_beef : ℝ) (cost_oil : ℝ) (total_grocery_cost : ℝ) (contribution_each : ℝ) :
  cost_beef_per_pound = 4 →
  quantity_beef = 3 →
  cost_oil = 1 →
  total_grocery_cost = 16 →
  contribution_each = 1 →
  ∃ (cost_chicken : ℝ), cost_chicken = 3 :=
by
  intros h1 h2 h3 h4 h5
  -- This line is required to help Lean handle any math operations
  have h6 := h1
  have h7 := h2
  have h8 := h3
  have h9 := h4
  have h10 := h5
  sorry

end cost_of_chicken_l392_39232


namespace least_number_remainder_seven_exists_l392_39274

theorem least_number_remainder_seven_exists :
  ∃ x : ℕ, x ≡ 7 [MOD 11] ∧ x ≡ 7 [MOD 17] ∧ x ≡ 7 [MOD 21] ∧ x ≡ 7 [MOD 29] ∧ x ≡ 7 [MOD 35] ∧ 
           x ≡ 1547 [MOD Nat.lcm 11 (Nat.lcm 17 (Nat.lcm 21 (Nat.lcm 29 35)))] :=
  sorry

end least_number_remainder_seven_exists_l392_39274


namespace sum_of_xyz_l392_39278

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem sum_of_xyz :
  ∃ x y z : ℝ,
  log_base 3 (log_base 4 (log_base 5 x)) = 0 ∧
  log_base 4 (log_base 5 (log_base 3 y)) = 0 ∧
  log_base 5 (log_base 3 (log_base 4 z)) = 0 ∧
  x + y + z = 932 :=
by
  sorry

end sum_of_xyz_l392_39278


namespace neg_of_if_pos_then_real_roots_l392_39222

variable (m : ℝ)

def has_real_roots (a b c : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + b * x + c = 0

theorem neg_of_if_pos_then_real_roots :
  (∀ m : ℝ, m > 0 → has_real_roots 1 1 (-m) )
  → ( ∀ m : ℝ, m ≤ 0 → ¬ has_real_roots 1 1 (-m) ) := 
sorry

end neg_of_if_pos_then_real_roots_l392_39222


namespace triangle_larger_segment_cutoff_l392_39280

open Real

theorem triangle_larger_segment_cutoff (a b c h s₁ s₂ : ℝ) (habc : a = 35) (hbc : b = 85) (hca : c = 90)
  (hh : h = 90)
  (eq₁ : a^2 = s₁^2 + h^2)
  (eq₂ : b^2 = s₂^2 + h^2)
  (h_sum : s₁ + s₂ = c) :
  max s₁ s₂ = 78.33 :=
by
  sorry

end triangle_larger_segment_cutoff_l392_39280


namespace total_students_l392_39277

-- Given definitions
def basketball_count : ℕ := 7
def cricket_count : ℕ := 5
def both_count : ℕ := 3

-- The goal to prove
theorem total_students : basketball_count + cricket_count - both_count = 9 :=
by
  sorry

end total_students_l392_39277


namespace cube_painting_probability_l392_39251

-- Define the conditions: a cube with six faces, each painted either green or yellow (independently, with probability 1/2)
structure Cube where
  faces : Fin 6 → Bool  -- Let's represent Bool with True for green, False for yellow

def is_valid_arrangement (c : Cube) : Prop :=
  ∃ (color : Bool), 
    (c.faces 0 = color ∧ c.faces 1 = color ∧ c.faces 2 = color ∧ c.faces 3 = color) ∧
    (∀ (i j : Fin 6), i = j ∨ ¬(c.faces i = color ∧ c.faces j = color))

def total_arrangements : ℕ := 2 ^ 6

def suitable_arrangements : ℕ := 20  -- As calculated previously: 2 + 12 + 6 = 20

-- We want to prove that the probability is 5/16
theorem cube_painting_probability :
  (suitable_arrangements : ℚ) / total_arrangements = 5 / 16 := 
by
  sorry

end cube_painting_probability_l392_39251


namespace diana_principal_charge_l392_39284

theorem diana_principal_charge :
  ∃ P : ℝ, P > 0 ∧ (P + P * 0.06 = 63.6) ∧ P = 60 :=
by
  use 60
  sorry

end diana_principal_charge_l392_39284


namespace chang_apple_problem_l392_39236

theorem chang_apple_problem 
  (A : ℝ)
  (h1 : 0.50 * A * 0.50 + 0.25 * A * 0.10 + 0.15 * A * 0.30 + 0.10 * A * 0.20 = 80)
  : A = 235 := 
sorry

end chang_apple_problem_l392_39236


namespace largest_n_divisible_by_n_plus_10_l392_39291

theorem largest_n_divisible_by_n_plus_10 :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧ ∀ m : ℕ, ((m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 := 
sorry

end largest_n_divisible_by_n_plus_10_l392_39291


namespace valid_digit_for_multiple_of_5_l392_39288

theorem valid_digit_for_multiple_of_5 (d : ℕ) (h : d < 10) : (45670 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 :=
by
  sorry

end valid_digit_for_multiple_of_5_l392_39288


namespace graph_empty_l392_39204

theorem graph_empty (x y : ℝ) : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 9 ≠ 0 :=
by
  -- Proof omitted
  sorry

end graph_empty_l392_39204


namespace quadratic_sums_l392_39273

variables {α : Type} [CommRing α] {a b c : α}

theorem quadratic_sums 
  (h₁ : ∀ (a b c : α), a + b ≠ 0 ∧ b + c ≠ 0 ∧ c + a ≠ 0)
  (h₂ : ∀ (r₁ r₂ : α), 
    (r₁^2 + a * r₁ + b = 0 ∧ r₂^2 + b * r₂ + c = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₃ : ∀ (r₁ r₂ : α), 
    (r₁^2 + b * r₁ + c = 0 ∧ r₂^2 + c * r₂ + a = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0)
  (h₄ : ∀ (r₁ r₂ : α), 
    (r₁^2 + c * r₁ + a = 0 ∧ r₂^2 + a * r₂ + b = 0) → r₁ + r₂ = 0 ∧ r₁ - r₂ ≠ 0) :
  a^2 + b^2 + c^2 = 18 ∧
  a^2 * b + b^2 * c + c^2 * a = 27 ∧
  a^3 * b^2 + b^3 * c^2 + c^3 * a^2 = -162 :=
sorry

end quadratic_sums_l392_39273


namespace altitude_from_A_to_BC_l392_39242

theorem altitude_from_A_to_BC (x y : ℝ) : 
  (3 * x + 4 * y + 12 = 0) ∧ 
  (4 * x - 3 * y + 16 = 0) ∧ 
  (2 * x + y - 2 = 0) → 
  (∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1 / 2) ∧ (b = 2 - 8)) :=
by 
  sorry

end altitude_from_A_to_BC_l392_39242


namespace smallest_solution_of_quadratic_eq_l392_39216

theorem smallest_solution_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ < x₂) ∧ (x₁^2 + 10 * x₁ - 40 = 0) ∧ (x₂^2 + 10 * x₂ - 40 = 0) ∧ x₁ = -8 :=
by {
  sorry
}

end smallest_solution_of_quadratic_eq_l392_39216


namespace range_of_a_l392_39279

-- Define the input conditions and requirements, and then state the theorem.
def is_acute_angle_cos_inequality (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2

theorem range_of_a (a : ℝ) :
  is_acute_angle_cos_inequality a 1 3 ∧ is_acute_angle_cos_inequality 1 3 a ∧
  is_acute_angle_cos_inequality 3 a 1 ↔ 2 * Real.sqrt 2 < a ∧ a < Real.sqrt 10 :=
by
  sorry

end range_of_a_l392_39279


namespace not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l392_39267

noncomputable def f (a x : ℝ) : ℝ :=
  1 + a * (1 / 2) ^ x + (1 / 4) ^ x

-- Problem (1)
theorem not_bounded_on_neg_infty_zero (a x : ℝ) (h : a = 1) : 
  ¬ ∃ M > 0, ∀ x < 0, |f a x| ≤ M :=
by sorry

-- Problem (2)
theorem range_of_a_bounded_on_zero_infty (a : ℝ) : 
  (∀ x ≥ 0, |f a x| ≤ 3) → -5 ≤ a ∧ a ≤ 1 :=
by sorry

end not_bounded_on_neg_infty_zero_range_of_a_bounded_on_zero_infty_l392_39267


namespace rabbit_parent_genotype_l392_39272

-- Define the types for alleles and genotypes
inductive Allele
| H : Allele -- Hairy allele, dominant
| h : Allele -- Hairy allele, recessive
| S : Allele -- Smooth allele, dominant
| s : Allele -- Smooth allele, recessive

structure RabbitGenotype where
  a1 : Allele
  a2 : Allele

-- Probability that the allele for hairy fur (H) occurs
def p_hairy_allele : ℝ := 0.1
-- Probability that the allele for smooth fur (S) occurs
def p_smooth_allele : ℝ := 1.0 - p_hairy_allele

-- Function to determine if a rabbit is hairy
def is_hairy (genotype : RabbitGenotype) : Prop :=
  (genotype.a1 = Allele.H) ∨ (genotype.a2 = Allele.H)

-- Mating resulted in all four offspring having hairy fur
def all_offspring_hairy (offspring : List RabbitGenotype) : Prop :=
  ∀ o ∈ offspring, is_hairy o

-- Statement of the proof problem
theorem rabbit_parent_genotype (offspring : List RabbitGenotype) (hf : offspring.length = 4) 
  (ha : all_offspring_hairy offspring) :
  ∃ (parent1 parent2 : RabbitGenotype), 
    (is_hairy parent1) ∧ 
    (¬ is_hairy parent2) ∧ 
    parent1 = { a1 := Allele.H, a2 := Allele.H } ∧ 
    parent2 = { a1 := Allele.S, a2 := Allele.h } :=
sorry

end rabbit_parent_genotype_l392_39272


namespace total_cost_in_dollars_l392_39259

theorem total_cost_in_dollars :
  (500 * 3 + 300 * 2) / 100 = 21 := 
by
  sorry

end total_cost_in_dollars_l392_39259


namespace discriminant_quadratic_eqn_l392_39263

def a := 1
def b := 1
def c := -2
def Δ : ℤ := b^2 - 4 * a * c

theorem discriminant_quadratic_eqn : Δ = 9 := by
  sorry

end discriminant_quadratic_eqn_l392_39263


namespace maggie_total_income_l392_39213

def total_income (h_tractor : ℕ) (r_office r_tractor : ℕ) :=
  let h_office := 2 * h_tractor
  (h_tractor * r_tractor) + (h_office * r_office)

theorem maggie_total_income :
  total_income 13 10 12 = 416 := 
  sorry

end maggie_total_income_l392_39213


namespace find_y_values_l392_39206

open Real

-- Problem statement as a Lean statement.
theorem find_y_values (x : ℝ) (hx : x^2 + 2 * (x / (x - 1)) ^ 2 = 20) :
  ∃ y : ℝ, (y = ((x - 1) ^ 3 * (x + 2)) / (2 * x - 1)) ∧ (y = 14 ∨ y = -56 / 3) := 
sorry

end find_y_values_l392_39206


namespace total_annual_gain_l392_39220

theorem total_annual_gain (x : ℝ) 
    (Lakshmi_share : ℝ) 
    (Lakshmi_share_eq: Lakshmi_share = 12000) : 
    (3 * Lakshmi_share = 36000) :=
by
  sorry

end total_annual_gain_l392_39220


namespace smallest_n_factorial_l392_39221

theorem smallest_n_factorial (a b c m n : ℕ) (h1 : a + b + c = 2020)
(h2 : c > a + 100)
(h3 : m * 10^n = a! * b! * c!)
(h4 : ¬ (10 ∣ m)) : 
  n = 499 :=
sorry

end smallest_n_factorial_l392_39221
