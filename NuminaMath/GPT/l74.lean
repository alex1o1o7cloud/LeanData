import Mathlib

namespace NUMINAMATH_GPT_Nicky_pace_5_mps_l74_7463

/-- Given the conditions:
  - Cristina runs at a pace of 5 meters per second.
  - Nicky runs for 30 seconds before Cristina catches up to him.
  Prove that Nicky’s pace is 5 meters per second. -/
theorem Nicky_pace_5_mps
  (Cristina_pace : ℝ)
  (time_Nicky : ℝ)
  (catchup : Cristina_pace * time_Nicky = 150)
  (def_Cristina_pace : Cristina_pace = 5)
  (def_time_Nicky : time_Nicky = 30) :
  (150 / 30) = 5 :=
by
  sorry

end NUMINAMATH_GPT_Nicky_pace_5_mps_l74_7463


namespace NUMINAMATH_GPT_real_values_x_l74_7467

theorem real_values_x (x y : ℝ) :
  (3 * y^2 + 5 * x * y + x + 7 = 0) →
  (5 * x + 6) * (5 * x - 14) ≥ 0 →
  x ≤ -6 / 5 ∨ x ≥ 14 / 5 :=
by
  sorry

end NUMINAMATH_GPT_real_values_x_l74_7467


namespace NUMINAMATH_GPT_total_savings_calculation_l74_7439

theorem total_savings_calculation
  (income : ℕ)
  (ratio_income_to_expenditure : ℕ)
  (ratio_expenditure_to_income : ℕ)
  (tax_rate : ℚ)
  (investment_rate : ℚ)
  (expenditure : ℕ)
  (taxes : ℚ)
  (investments : ℚ)
  (total_savings : ℚ)
  (h_income : income = 17000)
  (h_ratio : ratio_income_to_expenditure / ratio_expenditure_to_income = 5 / 4)
  (h_tax_rate : tax_rate = 0.15)
  (h_investment_rate : investment_rate = 0.1)
  (h_expenditure : expenditure = (income / 5) * 4)
  (h_taxes : taxes = 0.15 * income)
  (h_investments : investments = 0.1 * income)
  (h_total_savings : total_savings = income - (expenditure + taxes + investments)) :
  total_savings = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_calculation_l74_7439


namespace NUMINAMATH_GPT_zachary_pushups_l74_7499

variable (Zachary David John : ℕ)
variable (h1 : David = Zachary + 39)
variable (h2 : John = David - 13)
variable (h3 : David = 58)

theorem zachary_pushups : Zachary = 19 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_zachary_pushups_l74_7499


namespace NUMINAMATH_GPT_operation_result_l74_7424

theorem operation_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h_sum : a + b = 12) (h_prod : a * b = 32) 
: (1 / a : ℚ) + (1 / b) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_operation_result_l74_7424


namespace NUMINAMATH_GPT_trigonometric_values_l74_7475

variable (α : ℝ)

theorem trigonometric_values (h : Real.cos (3 * Real.pi + α) = 3 / 5) :
  Real.cos α = -3 / 5 ∧
  Real.cos (Real.pi + α) = 3 / 5 ∧
  Real.sin (3 * Real.pi / 2 - α) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_values_l74_7475


namespace NUMINAMATH_GPT_smaller_number_is_476_l74_7466

theorem smaller_number_is_476 (x y : ℕ) 
  (h1 : y - x = 2395) 
  (h2 : y = 6 * x + 15) : 
  x = 476 := 
by 
  sorry

end NUMINAMATH_GPT_smaller_number_is_476_l74_7466


namespace NUMINAMATH_GPT_transitiveSim_l74_7472

def isGreat (f : ℕ × ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1, n + 1) * f (m, n) - f (m + 1, n) * f (m, n + 1) = 1

def seqSim (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ × ℕ → ℤ, isGreat f ∧ (∀ n, f (n, 0) = A n) ∧ (∀ n, f (0, n) = B n)

theorem transitiveSim (A B C D : ℕ → ℤ)
  (h1 : seqSim A B)
  (h2 : seqSim B C)
  (h3 : seqSim C D) : seqSim D A :=
sorry

end NUMINAMATH_GPT_transitiveSim_l74_7472


namespace NUMINAMATH_GPT_knights_divisible_by_4_l74_7452

-- Define the conditions: Assume n is the total number of knights (n > 0).
-- Condition 1: Knights from two opposing clans A and B
-- Condition 2: Number of knights with an enemy to the right equals number of knights with a friend to the right.

open Nat

theorem knights_divisible_by_4 (n : ℕ) (h1 : 0 < n)
  (h2 : ∃k : ℕ, 2 * k = n ∧ ∀ (i : ℕ), (i < n → ((i % 2 = 0 → (i+1) % 2 = 1) ∧ (i % 2 = 1 → (i+1) % 2 = 0)))) :
  n % 4 = 0 :=
sorry

end NUMINAMATH_GPT_knights_divisible_by_4_l74_7452


namespace NUMINAMATH_GPT_shaded_region_volume_l74_7435

theorem shaded_region_volume :
  let r1 := 4   -- radius of the first cylinder
  let h1 := 2   -- height of the first cylinder
  let r2 := 1   -- radius of the second cylinder
  let h2 := 5   -- height of the second cylinder
  let V1 := π * r1^2 * h1 -- volume of the first cylinder
  let V2 := π * r2^2 * h2 -- volume of the second cylinder
  V1 + V2 = 37 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_volume_l74_7435


namespace NUMINAMATH_GPT_quadratic_complex_inequality_solution_l74_7483
noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  (x^2 / (x + 2) ≥ 3 / (x - 2) + 7/4) ↔ -2 < x ∧ x < 2 ∨ 3 ≤ x

theorem quadratic_complex_inequality_solution (x : ℝ) (hx : x ≠ -2 ∧ x ≠ 2):
  quadratic_inequality_solution x :=
  sorry

end NUMINAMATH_GPT_quadratic_complex_inequality_solution_l74_7483


namespace NUMINAMATH_GPT_rectangle_difference_l74_7459

theorem rectangle_difference (L B D : ℝ)
  (h1 : L - B = D)
  (h2 : 2 * (L + B) = 186)
  (h3 : L * B = 2030) :
  D = 23 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_difference_l74_7459


namespace NUMINAMATH_GPT_smallest_three_digit_number_satisfying_conditions_l74_7488

theorem smallest_three_digit_number_satisfying_conditions :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n + 6) % 9 = 0 ∧ (n - 4) % 6 = 0 ∧ n = 112 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_smallest_three_digit_number_satisfying_conditions_l74_7488


namespace NUMINAMATH_GPT_triangle_sides_l74_7411

noncomputable def sides (a b c : ℝ) : Prop :=
  (a = Real.sqrt (427 / 3)) ∧
  (b = Real.sqrt (427 / 3) + 3/2) ∧
  (c = Real.sqrt (427 / 3) - 3/2)

theorem triangle_sides (a b c : ℝ) (h1 : b - c = 3) (h2 : ∃ d : ℝ, d = 10)
  (h3 : ∃ BD CD : ℝ, CD - BD = 12 ∧ BD + CD = a ∧ 
    a = 2 * (BD + 12 / 2)) :
  sides a b c :=
  sorry

end NUMINAMATH_GPT_triangle_sides_l74_7411


namespace NUMINAMATH_GPT_expression_is_integer_l74_7462

theorem expression_is_integer (n : ℕ) : 
    ∃ k : ℤ, (n^5 : ℤ) / 5 + (n^3 : ℤ) / 3 + (7 * n : ℤ) / 15 = k :=
by
  sorry

end NUMINAMATH_GPT_expression_is_integer_l74_7462


namespace NUMINAMATH_GPT_rectangle_area_increase_l74_7481

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let l_new := 1.3 * l
  let w_new := 1.2 * w
  let A_new := l_new * w_new
  let A := l * w
  let increase := A_new - A
  let percent_increase := (increase / A) * 100
  percent_increase = 56 := sorry

end NUMINAMATH_GPT_rectangle_area_increase_l74_7481


namespace NUMINAMATH_GPT_sum_of_two_primes_l74_7433

theorem sum_of_two_primes (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 93) : p * q = 178 := 
sorry

end NUMINAMATH_GPT_sum_of_two_primes_l74_7433


namespace NUMINAMATH_GPT_wheel_distance_3_revolutions_l74_7436

theorem wheel_distance_3_revolutions (r : ℝ) (n : ℝ) (circumference : ℝ) (total_distance : ℝ) :
  r = 2 →
  n = 3 →
  circumference = 2 * Real.pi * r →
  total_distance = n * circumference →
  total_distance = 12 * Real.pi := by
  intros
  sorry

end NUMINAMATH_GPT_wheel_distance_3_revolutions_l74_7436


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l74_7469

theorem sum_of_reciprocals_of_roots : 
  ∀ {r1 r2 : ℝ}, (r1 + r2 = 14) → (r1 * r2 = 6) → (1 / r1 + 1 / r2 = 7 / 3) :=
by
  intros r1 r2 h_sum h_product
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l74_7469


namespace NUMINAMATH_GPT_marys_mother_bought_3_pounds_of_beef_l74_7437

-- Define the variables and constants
def total_paid : ℝ := 16
def cost_of_chicken : ℝ := 2 * 1  -- 2 pounds of chicken
def cost_per_pound_beef : ℝ := 4
def cost_of_oil : ℝ := 1
def shares : ℝ := 3  -- Mary and her two friends

theorem marys_mother_bought_3_pounds_of_beef:
  total_paid - (cost_of_chicken / shares) - cost_of_oil = 3 * cost_per_pound_beef :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_marys_mother_bought_3_pounds_of_beef_l74_7437


namespace NUMINAMATH_GPT_no_real_roots_of_equation_l74_7420

theorem no_real_roots_of_equation :
  (∃ x : ℝ, 2 * Real.cos (x / 2) = 10^x + 10^(-x) + 1) -> False :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_equation_l74_7420


namespace NUMINAMATH_GPT_find_fencing_cost_l74_7428

theorem find_fencing_cost
  (d : ℝ) (cost_per_meter : ℝ) (π : ℝ)
  (h1 : d = 22)
  (h2 : cost_per_meter = 2.50)
  (hπ : π = Real.pi) :
  (cost_per_meter * (π * d) = 172.80) :=
sorry

end NUMINAMATH_GPT_find_fencing_cost_l74_7428


namespace NUMINAMATH_GPT_train_length_correct_l74_7496

noncomputable def train_length (speed_kmph: ℝ) (time_sec: ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  speed_mps * time_sec

theorem train_length_correct : train_length 250 12 = 833.28 := by
  sorry

end NUMINAMATH_GPT_train_length_correct_l74_7496


namespace NUMINAMATH_GPT_caleb_spent_more_on_ice_cream_l74_7473

theorem caleb_spent_more_on_ice_cream :
  let num_ic_cream := 10
  let cost_ic_cream := 4
  let num_frozen_yog := 4
  let cost_frozen_yog := 1
  (num_ic_cream * cost_ic_cream - num_frozen_yog * cost_frozen_yog) = 36 := 
by
  sorry

end NUMINAMATH_GPT_caleb_spent_more_on_ice_cream_l74_7473


namespace NUMINAMATH_GPT_find_m_l74_7485

open Real

noncomputable def x_values : List ℝ := [1, 3, 4, 5, 7]
noncomputable def y_values (m : ℝ) : List ℝ := [1, m, 2 * m + 1, 2 * m + 3, 10]

noncomputable def mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem find_m (m : ℝ) :
  mean x_values = 4 →
  mean (y_values m) = m + 3 →
  (1.3 * 4 + 0.8 = m + 3) →
  m = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_m_l74_7485


namespace NUMINAMATH_GPT_tan_alpha_l74_7444

theorem tan_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 1 / 5) : Real.tan α = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_l74_7444


namespace NUMINAMATH_GPT_olivia_nigel_remaining_money_l74_7441

theorem olivia_nigel_remaining_money :
  let olivia_money := 112
  let nigel_money := 139
  let ticket_count := 6
  let ticket_price := 28
  let total_money := olivia_money + nigel_money
  let total_cost := ticket_count * ticket_price
  total_money - total_cost = 83 := 
by 
  sorry

end NUMINAMATH_GPT_olivia_nigel_remaining_money_l74_7441


namespace NUMINAMATH_GPT_quadrilateral_area_24_l74_7432

open Classical

noncomputable def quad_area (a b : ℤ) (h : a > b ∧ b > 0) : ℤ :=
let P := (a, b)
let Q := (2*b, a)
let R := (-a, -b)
let S := (-2*b, -a)
-- The proved area
24

theorem quadrilateral_area_24 (a b : ℤ) (h : a > b ∧ b > 0) :
  quad_area a b h = 24 :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_24_l74_7432


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_in_COD_l74_7450

theorem radius_of_inscribed_circle_in_COD
  (r1 : ℝ) (r2 : ℝ) (r3 : ℝ) (r4 : ℝ)
  (H1 : r1 = 6)
  (H2 : r2 = 2)
  (H3 : r3 = 1.5)
  (H4 : 1/r1 + 1/r3 = 1/r2 + 1/r4) :
  r4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_in_COD_l74_7450


namespace NUMINAMATH_GPT_relationship_y_values_l74_7400

theorem relationship_y_values (m y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-3 : ℝ)^2 - 12 * (-3 : ℝ) + m)
  (h2 : y2 = -3 * (-2 : ℝ)^2 - 12 * (-2 : ℝ) + m)
  (h3 : y3 = -3 * (1 : ℝ)^2 - 12 * (1 : ℝ) + m) :
  y2 > y1 ∧ y1 > y3 :=
sorry

end NUMINAMATH_GPT_relationship_y_values_l74_7400


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l74_7482

theorem geometric_sequence_common_ratio
  (a₁ a₂ a₃ : ℝ) (q : ℝ) 
  (h₀ : 0 < a₁) 
  (h₁ : a₂ = a₁ * q) 
  (h₂ : a₃ = a₁ * q^2) 
  (h₃ : 2 * a₁ + a₂ = 2 * (1 / 2 * a₃)) 
  : q = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l74_7482


namespace NUMINAMATH_GPT_line_intersects_parabola_exactly_one_point_l74_7423

theorem line_intersects_parabola_exactly_one_point (k : ℝ) :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 10 = k) ∧
  (∀ y z : ℝ, -3 * y^2 - 4 * y + 10 = k ∧ -3 * z^2 - 4 * z + 10 = k → y = z) 
  → k = 34 / 3 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_exactly_one_point_l74_7423


namespace NUMINAMATH_GPT_length_of_arc_l74_7402

theorem length_of_arc (C : ℝ) (θ : ℝ) (DE : ℝ) (c_circ : C = 100) (angle : θ = 120) :
  DE = 100 / 3 :=
by
  -- Place the actual proof here.
  sorry

end NUMINAMATH_GPT_length_of_arc_l74_7402


namespace NUMINAMATH_GPT_sin_double_angle_l74_7404

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l74_7404


namespace NUMINAMATH_GPT_total_weight_loss_l74_7478

theorem total_weight_loss (S J V : ℝ) 
  (hS : S = 17.5) 
  (hJ : J = 3 * S) 
  (hV : V = S + 1.5) : 
  S + J + V = 89 := 
by 
  sorry

end NUMINAMATH_GPT_total_weight_loss_l74_7478


namespace NUMINAMATH_GPT_force_for_18_inch_wrench_l74_7413

theorem force_for_18_inch_wrench (F : ℕ → ℕ → ℕ) : 
  (∀ L : ℕ, ∃ k : ℕ, F 300 12 = F (F L k) L) → 
  ((F 12 300) = 3600) → 
  (∀ k : ℕ, F (F 6 k) 6 = 3600) → 
  (∀ k : ℕ, F (F 18 k) 18 = 3600) → 
  (F 18 200 = 3600) :=
by
  sorry

end NUMINAMATH_GPT_force_for_18_inch_wrench_l74_7413


namespace NUMINAMATH_GPT_units_digit_difference_l74_7457

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_difference :
  units_digit (72^3) - units_digit (24^3) = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_difference_l74_7457


namespace NUMINAMATH_GPT_part1_part2_l74_7426

-- Define the complex number z in terms of m
def z (m : ℝ) : ℂ := (2 * m^2 - 3 * m - 2) + (m^2 - 3 * m + 2) * Complex.I

-- State the condition where z is a purely imaginary number
def purelyImaginary (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 = 0 ∧ m^2 - 3 * m + 2 ≠ 0

-- State the condition where z is in the second quadrant.
def inSecondQuadrant (m : ℝ) : Prop := 2 * m^2 - 3 * m - 2 < 0 ∧ m^2 - 3 * m + 2 > 0

-- Part 1: Prove that m = -1/2 given that z is purely imaginary.
theorem part1 : purelyImaginary m → m = -1/2 :=
sorry

-- Part 2: Prove the range of m for z in the second quadrant.
theorem part2 : inSecondQuadrant m → -1/2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l74_7426


namespace NUMINAMATH_GPT_largest_coefficient_term_in_expansion_l74_7498

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_coefficient_term_in_expansion :
  ∃ (T : ℕ × ℤ × ℕ), 
  (2 : ℤ) ^ (14 - 1) = 8192 ∧ 
  T = (binom 14 4, 2 ^ 10, 4) ∧ 
  ∀ (k : ℕ), 
    (binom 14 k * (2 ^ (14 - k))) ≤ (binom 14 4 * 2 ^ 10) :=
sorry

end NUMINAMATH_GPT_largest_coefficient_term_in_expansion_l74_7498


namespace NUMINAMATH_GPT_line_x_intercept_l74_7422

theorem line_x_intercept (P Q : ℝ × ℝ) (hP : P = (2, 3)) (hQ : Q = (6, 7)) :
  ∃ x, (x, 0) = (-1, 0) ∧ ∃ (m : ℝ), m = (Q.2 - P.2) / (Q.1 - P.1) ∧ ∀ (x y : ℝ), y = m * (x - P.1) + P.2 := 
  sorry

end NUMINAMATH_GPT_line_x_intercept_l74_7422


namespace NUMINAMATH_GPT_product_without_zero_digits_l74_7480

def no_zero_digits (n : ℕ) : Prop :=
  ¬ ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 0

theorem product_without_zero_digits :
  ∃ a b : ℕ, a * b = 1000000000 ∧ no_zero_digits a ∧ no_zero_digits b :=
by
  sorry

end NUMINAMATH_GPT_product_without_zero_digits_l74_7480


namespace NUMINAMATH_GPT_handshakes_at_convention_l74_7446

theorem handshakes_at_convention :
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  handshakes_among_gremlins + handshakes_between_imps_gremlins = 660 :=
by
  let gremlins := 30
  let imps := 15
  let handshakes_among_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_between_imps_gremlins := imps * (gremlins / 2)
  show handshakes_among_gremlins + handshakes_between_imps_gremlins = 660
  sorry

end NUMINAMATH_GPT_handshakes_at_convention_l74_7446


namespace NUMINAMATH_GPT_find_x_l74_7465

-- Let \( x \) be a real number such that 
-- \( x = 2 \left( \frac{1}{x} \cdot (-x) \right) - 5 \).
-- Prove \( x = -7 \).

theorem find_x (x : ℝ) (h : x = 2 * (1 / x * (-x)) - 5) : x = -7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l74_7465


namespace NUMINAMATH_GPT_division_quotient_proof_l74_7415

theorem division_quotient_proof (x : ℕ) (larger_number : ℕ) (h1 : larger_number - x = 1365)
    (h2 : larger_number = 1620) (h3 : larger_number % x = 15) : larger_number / x = 6 :=
by
  sorry

end NUMINAMATH_GPT_division_quotient_proof_l74_7415


namespace NUMINAMATH_GPT_complete_the_square_sum_l74_7454

theorem complete_the_square_sum :
  ∃ p q : ℝ, (∀ x : ℝ, 15 * x^2 - 30 * x - 60 = 0 → (x + p)^2 = q) ∧ p + q = 1 :=
by 
  sorry

end NUMINAMATH_GPT_complete_the_square_sum_l74_7454


namespace NUMINAMATH_GPT_Bill_composes_20_problems_l74_7447

theorem Bill_composes_20_problems :
  ∀ (B : ℕ), (∀ R : ℕ, R = 2 * B) →
    (∀ F : ℕ, F = 3 * R) →
    (∀ T : ℕ, T = 4) →
    (∀ P : ℕ, P = 30) →
    (∀ F : ℕ, F = T * P) →
    (∃ B : ℕ, B = 20) :=
by sorry

end NUMINAMATH_GPT_Bill_composes_20_problems_l74_7447


namespace NUMINAMATH_GPT_sin_cos_quad_ineq_l74_7494

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_quad_ineq_l74_7494


namespace NUMINAMATH_GPT_probability_xi_eq_1_l74_7492

-- Definitions based on conditions
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

-- Combinatorics function for choosing k items from n items
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Definition for probability P(ξ = 1)
def P_xi_eq_1 := 
  (C white_balls_bag_A 1 * C white_balls_bag_B 1 + C red_balls_bag_A 1 * C white_balls_bag_B 1) /
  (C (white_balls_bag_A + red_balls_bag_A) 1 * C (white_balls_bag_B + red_balls_bag_B) 1)

theorem probability_xi_eq_1 :
  P_xi_eq_1 = (C 8 1 * C 6 1 + C 4 1 * C 6 1) / (C 12 1 * C 12 1) :=
by
  sorry

end NUMINAMATH_GPT_probability_xi_eq_1_l74_7492


namespace NUMINAMATH_GPT_subtract_three_from_binary_l74_7484

theorem subtract_three_from_binary (M : ℕ) (M_binary: M = 0b10110000) : (M - 3) = 0b10101101 := by
  sorry

end NUMINAMATH_GPT_subtract_three_from_binary_l74_7484


namespace NUMINAMATH_GPT_land_remaining_is_correct_l74_7425

def lizzie_covered : ℕ := 250
def other_covered : ℕ := 265
def total_land : ℕ := 900
def land_remaining : ℕ := total_land - (lizzie_covered + other_covered)

theorem land_remaining_is_correct : land_remaining = 385 := 
by
  sorry

end NUMINAMATH_GPT_land_remaining_is_correct_l74_7425


namespace NUMINAMATH_GPT_original_number_l74_7455

theorem original_number (sum_orig : ℕ) (sum_new : ℕ) (changed_value : ℕ) (avg_orig : ℕ) (avg_new : ℕ) (n : ℕ) :
    sum_orig = n * avg_orig →
    sum_new = sum_orig - changed_value + 9 →
    avg_new = 8 →
    avg_orig = 7 →
    n = 7 →
    sum_new = n * avg_new →
    changed_value = 2 := 
by
  sorry

end NUMINAMATH_GPT_original_number_l74_7455


namespace NUMINAMATH_GPT_intersection_equiv_l74_7410

open Set

def A : Set ℝ := { x | 2 * x < 2 + x }
def B : Set ℝ := { x | 5 - x > 8 - 4 * x }

theorem intersection_equiv : A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_equiv_l74_7410


namespace NUMINAMATH_GPT_total_loss_is_correct_l74_7460

-- Definitions for each item's purchase conditions
def paintings_cost : ℕ := 18 * 75
def toys_cost : ℕ := 25 * 30
def hats_cost : ℕ := 12 * 20
def wallets_cost : ℕ := 10 * 50
def mugs_cost : ℕ := 35 * 10

def paintings_loss_percentage : ℝ := 0.22
def toys_loss_percentage : ℝ := 0.27
def hats_loss_percentage : ℝ := 0.15
def wallets_loss_percentage : ℝ := 0.05
def mugs_loss_percentage : ℝ := 0.12

-- Calculation of loss on each item
def paintings_loss : ℝ := paintings_cost * paintings_loss_percentage
def toys_loss : ℝ := toys_cost * toys_loss_percentage
def hats_loss : ℝ := hats_cost * hats_loss_percentage
def wallets_loss : ℝ := wallets_cost * wallets_loss_percentage
def mugs_loss : ℝ := mugs_cost * mugs_loss_percentage

-- Total loss calculation
def total_loss : ℝ := paintings_loss + toys_loss + hats_loss + wallets_loss + mugs_loss

-- Lean statement to verify the total loss
theorem total_loss_is_correct : total_loss = 602.50 := by
  sorry

end NUMINAMATH_GPT_total_loss_is_correct_l74_7460


namespace NUMINAMATH_GPT_find_number_l74_7458

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_find_number_l74_7458


namespace NUMINAMATH_GPT_class_average_l74_7453

theorem class_average (p1 p2 p3 avg1 avg2 avg3 overall_avg : ℕ) 
  (h1 : p1 = 45) 
  (h2 : p2 = 50) 
  (h3 : p3 = 100 - p1 - p2) 
  (havg1 : avg1 = 95) 
  (havg2 : avg2 = 78) 
  (havg3 : avg3 = 60) 
  (hoverall : overall_avg = (p1 * avg1 + p2 * avg2 + p3 * avg3) / 100) : 
  overall_avg = 85 :=
by
  sorry

end NUMINAMATH_GPT_class_average_l74_7453


namespace NUMINAMATH_GPT_bond_value_after_8_years_l74_7451

theorem bond_value_after_8_years (r t1 t2 : ℕ) (A1 A2 P : ℚ) :
  r = 4 / 100 ∧ t1 = 3 ∧ t2 = 8 ∧ A1 = 560 ∧ A1 = P * (1 + r * t1) 
  → A2 = P * (1 + r * t2) ∧ A2 = 660 :=
by
  intro h
  obtain ⟨hr, ht1, ht2, hA1, hA1eq⟩ := h
  -- Proof needs to be filled in here
  sorry

end NUMINAMATH_GPT_bond_value_after_8_years_l74_7451


namespace NUMINAMATH_GPT_min_value_expression_l74_7416

variable (a b m n : ℝ)

-- Conditions: a, b, m, n are positive, a + b = 1, mn = 2
def conditions (a b m n : ℝ) : Prop := 
  0 < a ∧ 0 < b ∧ 0 < m ∧ 0 < n ∧ a + b = 1 ∧ m * n = 2

-- Statement to prove: The minimum value of (am + bn) * (bm + an) is 2
theorem min_value_expression (a b m n : ℝ) (h : conditions a b m n) : 
  ∃ c : ℝ, c = 2 ∧ (∀ (x y z w : ℝ), conditions x y z w → (x * z + y * w) * (y * z + x * w) ≥ c) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l74_7416


namespace NUMINAMATH_GPT_train_length_l74_7456

noncomputable def length_of_each_train (L : ℝ) : Prop :=
  let v1 := 46 -- speed of faster train in km/hr
  let v2 := 36 -- speed of slower train in km/hr
  let relative_speed := (v1 - v2) * (5/18) -- converting relative speed to m/s
  let time := 72 -- time in seconds
  2 * L = relative_speed * time -- distance equation

theorem train_length : ∃ (L : ℝ), length_of_each_train L ∧ L = 100 :=
by
  use 100
  unfold length_of_each_train
  sorry

end NUMINAMATH_GPT_train_length_l74_7456


namespace NUMINAMATH_GPT_perimeter_of_square_is_160_cm_l74_7471

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_of_square (area_of_rectangle : ℝ) : ℝ := 5 * area_of_rectangle

noncomputable def side_length_of_square (area_of_square : ℝ) : ℝ := Real.sqrt area_of_square

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ := 4 * side_length

theorem perimeter_of_square_is_160_cm :
  perimeter_of_square (side_length_of_square (area_of_square (area_of_rectangle 32 10))) = 160 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_is_160_cm_l74_7471


namespace NUMINAMATH_GPT_carter_family_children_l74_7407

variable (f m x y : ℕ)

theorem carter_family_children 
  (avg_family : (3 * y + m + x * y) / (2 + x) = 25)
  (avg_mother_children : (m + x * y) / (1 + x) = 18)
  (father_age : f = 3 * y)
  (simplest_case : y = x) :
  x = 8 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_carter_family_children_l74_7407


namespace NUMINAMATH_GPT_card_probability_l74_7490

-- Define the total number of cards
def total_cards : ℕ := 52

-- Define the number of Kings in the deck
def kings_in_deck : ℕ := 4

-- Define the number of Aces in the deck
def aces_in_deck : ℕ := 4

-- Define the probability of the top card being a King
def prob_top_king : ℚ := kings_in_deck / total_cards

-- Define the probability of the second card being an Ace given the first card is a King
def prob_second_ace_given_king : ℚ := aces_in_deck / (total_cards - 1)

-- Define the combined probability of both events happening in sequence
def combined_probability : ℚ := prob_top_king * prob_second_ace_given_king

-- Theorem statement that the combined probability is equal to 4/663
theorem card_probability : combined_probability = 4 / 663 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_card_probability_l74_7490


namespace NUMINAMATH_GPT_common_divisors_count_l74_7491

def prime_exponents (n : Nat) : List (Nat × Nat) :=
  if n = 9240 then [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
  else if n = 10800 then [(2, 4), (3, 3), (5, 2)]
  else []

def gcd_prime_exponents (exps1 exps2 : List (Nat × Nat)) : List (Nat × Nat) :=
  exps1.filterMap (fun (p1, e1) =>
    match exps2.find? (fun (p2, _) => p1 = p2) with
    | some (p2, e2) => if e1 ≤ e2 then some (p1, e1) else some (p1, e2)
    | none => none
  )

def count_divisors (exps : List (Nat × Nat)) : Nat :=
  exps.foldl (fun acc (_, e) => acc * (e + 1)) 1

theorem common_divisors_count :
  count_divisors (gcd_prime_exponents (prime_exponents 9240) (prime_exponents 10800)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_common_divisors_count_l74_7491


namespace NUMINAMATH_GPT_appears_in_31st_equation_l74_7470

theorem appears_in_31st_equation : 
  ∃ n : ℕ, 2016 ∈ {x | 2*x^2 ≤ 2016 ∧ 2016 < 2*(x+1)^2} ∧ n = 31 :=
by
  sorry

end NUMINAMATH_GPT_appears_in_31st_equation_l74_7470


namespace NUMINAMATH_GPT_proof_triangle_tangent_l74_7461

open Real

def isCongruentAngles (ω : ℝ) := 
  let a := 15
  let b := 18
  let c := 21
  ∃ (x y z : ℝ), 
  (y^2 = x^2 + a^2 - 2 * a * x * cos ω) 
  ∧ (z^2 = y^2 + b^2 - 2 * b * y * cos ω)
  ∧ (x^2 = z^2 + c^2 - 2 * c * z * cos ω)

def isTriangleABCWithSides (AB BC CA : ℝ) (ω : ℝ) (tan_ω : ℝ) : Prop := 
  (AB = 15) ∧ (BC = 18) ∧ (CA = 21) ∧ isCongruentAngles ω 
  ∧ tan ω = tan_ω

theorem proof_triangle_tangent : isTriangleABCWithSides 15 18 21 ω (88/165) := 
by
  sorry

end NUMINAMATH_GPT_proof_triangle_tangent_l74_7461


namespace NUMINAMATH_GPT_range_of_a_l74_7412

def discriminant (a : ℝ) : ℝ := 4 * a^2 - 16
def P (a : ℝ) : Prop := discriminant a < 0
def Q (a : ℝ) : Prop := 5 - 2 * a > 1

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a ≤ -2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l74_7412


namespace NUMINAMATH_GPT_range_of_a_l74_7468

theorem range_of_a (m : ℝ) (a : ℝ) (hx : ∃ x : ℝ, mx^2 + x - m - a = 0) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l74_7468


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l74_7417

variables {f g : ℝ → ℝ}

theorem necessary_and_sufficient_condition (f g : ℝ → ℝ)
  (hdom : ∀ x : ℝ, true)
  (hst : ∀ y : ℝ, true) :
  (∀ x : ℝ, f x > g x) ↔ (∀ x : ℝ, ¬ (x ∈ {x : ℝ | f x ≤ g x})) :=
by sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l74_7417


namespace NUMINAMATH_GPT_find_multiple_of_t_l74_7408

variable (t : ℝ)
variable (x y : ℝ)

theorem find_multiple_of_t (h1 : x = 1 - 4 * t)
  (h2 : ∃ m : ℝ, y = m * t - 2)
  (h3 : t = 0.5)
  (h4 : x = y) : ∃ m : ℝ, (m = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_t_l74_7408


namespace NUMINAMATH_GPT_sin_cos_fraction_eq_two_l74_7449

theorem sin_cos_fraction_eq_two (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 :=
sorry

end NUMINAMATH_GPT_sin_cos_fraction_eq_two_l74_7449


namespace NUMINAMATH_GPT_equivalent_resistance_A_B_l74_7429

-- Parameters and conditions
def resistor_value : ℝ := 5 -- in MΩ
def num_resistors : ℕ := 4
def has_bridging_wire : Prop := true
def negligible_wire_resistance : Prop := true

-- Problem: Prove the equivalent resistance (R_eff) between points A and B is 5 MΩ.
theorem equivalent_resistance_A_B : 
  ∀ (R : ℝ) (n : ℕ) (bridge : Prop) (negligible_wire : Prop),
    R = 5 → n = 4 → bridge → negligible_wire → R = 5 :=
by sorry

end NUMINAMATH_GPT_equivalent_resistance_A_B_l74_7429


namespace NUMINAMATH_GPT_prove_values_l74_7477

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 1/x + b

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem prove_values (a b : ℝ) (h1 : a > 0) (h2 : is_integer b) :
  (f a b (Real.log a) = 6 ∧ f a b (Real.log (1 / a)) = 2) ∨
  (f a b (Real.log a) = -2 ∧ f a b (Real.log (1 / a)) = 2) :=
sorry

end NUMINAMATH_GPT_prove_values_l74_7477


namespace NUMINAMATH_GPT_pool_water_after_45_days_l74_7495

-- Defining the initial conditions and the problem statement in Lean
noncomputable def initial_amount : ℝ := 500
noncomputable def evaporation_rate : ℝ := 0.7
noncomputable def addition_rate : ℝ := 5
noncomputable def total_days : ℕ := 45

noncomputable def final_amount : ℝ :=
  initial_amount - (evaporation_rate * total_days) +
  (addition_rate * (total_days / 3))

theorem pool_water_after_45_days : final_amount = 543.5 :=
by
  -- Inserting the proof is not required here
  sorry

end NUMINAMATH_GPT_pool_water_after_45_days_l74_7495


namespace NUMINAMATH_GPT_shaded_area_of_circles_l74_7403

theorem shaded_area_of_circles :
  let R := 10
  let r1 := R / 2
  let r2 := R / 2
  (π * R^2 - (π * r1^2 + π * r1^2 + π * r2^2)) = 25 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_circles_l74_7403


namespace NUMINAMATH_GPT_option_D_is_linear_equation_with_two_variables_l74_7414

def is_linear_equation (eq : String) : Prop :=
  match eq with
  | "3x - 6 = x" => false
  | "x = 5 / y - 1" => false
  | "2x - 3y = x^2" => false
  | "3x = 2y" => true
  | _ => false

theorem option_D_is_linear_equation_with_two_variables :
  is_linear_equation "3x = 2y" = true := by
  sorry

end NUMINAMATH_GPT_option_D_is_linear_equation_with_two_variables_l74_7414


namespace NUMINAMATH_GPT_half_angle_quadrant_l74_7489

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end NUMINAMATH_GPT_half_angle_quadrant_l74_7489


namespace NUMINAMATH_GPT_power_sum_int_l74_7497

theorem power_sum_int {x : ℝ} (hx : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by
  sorry

end NUMINAMATH_GPT_power_sum_int_l74_7497


namespace NUMINAMATH_GPT_alice_has_ball_after_three_turns_l74_7440

def alice_keeps_ball (prob_Alice_to_Bob: ℚ) (prob_Bob_to_Alice: ℚ): ℚ := 
  let prob_Alice_keeps := 1 - prob_Alice_to_Bob
  let prob_Bob_keeps := 1 - prob_Bob_to_Alice
  let path1 := prob_Alice_to_Bob * prob_Bob_to_Alice * prob_Alice_keeps
  let path2 := prob_Alice_keeps * prob_Alice_keeps * prob_Alice_keeps
  path1 + path2

theorem alice_has_ball_after_three_turns:
  alice_keeps_ball (1/2) (1/3) = 5/24 := 
by
  sorry

end NUMINAMATH_GPT_alice_has_ball_after_three_turns_l74_7440


namespace NUMINAMATH_GPT_symmetric_point_exists_l74_7427

-- Define the point M
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point M
def M : Point3D := { x := 3, y := 3, z := 3 }

-- Define the parametric form of the line
def line (t : ℝ) : Point3D := { x := 1 - t, y := 1.5, z := 3 + t }

-- Define the point M' that we want to prove is symmetrical to M with respect to the line
def symmPoint : Point3D := { x := 1, y := 0, z := 1 }

-- The theorem that we need to prove, ensuring M' is symmetrical to M with respect to the given line
theorem symmetric_point_exists : ∃ t, line t = symmPoint ∧ 
  (∀ M_0 : Point3D, M_0.x = (M.x + symmPoint.x) / 2 ∧ M_0.y = (M.y + symmPoint.y) / 2 ∧ M_0.z = (M.z + symmPoint.z) / 2)
  → line t = M_0
  → M_0 = { x := 2, y := 1.5, z := 2 } := 
by
  sorry

end NUMINAMATH_GPT_symmetric_point_exists_l74_7427


namespace NUMINAMATH_GPT_polygon_sides_eq_seven_l74_7434

theorem polygon_sides_eq_seven (n d : ℕ) (h1 : d = (n * (n - 3)) / 2) (h2 : d = 2 * n) : n = 7 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_seven_l74_7434


namespace NUMINAMATH_GPT_sum_of_excluded_values_l74_7442

theorem sum_of_excluded_values (C D : ℝ) (h₁ : 2 * C^2 - 8 * C + 6 = 0)
    (h₂ : 2 * D^2 - 8 * D + 6 = 0) (h₃ : C ≠ D) :
    C + D = 4 :=
sorry

end NUMINAMATH_GPT_sum_of_excluded_values_l74_7442


namespace NUMINAMATH_GPT_time_for_c_to_finish_alone_l74_7486

variable (A B C : ℚ) -- A, B, and C are the work rates

theorem time_for_c_to_finish_alone :
  (A + B = 1/3) →
  (B + C = 1/4) →
  (C + A = 1/6) →
  1/C = 24 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_time_for_c_to_finish_alone_l74_7486


namespace NUMINAMATH_GPT_area_of_triangle_l74_7418

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (2, 6)

-- Define the function to calculate the area of the triangle formed by three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The theorem statement that the area of the triangle with given vertices is 14.5
theorem area_of_triangle : triangle_area A B C = 14.5 :=
by 
  -- Skipping the proof part
  sorry

end NUMINAMATH_GPT_area_of_triangle_l74_7418


namespace NUMINAMATH_GPT_cistern_fill_time_l74_7405

-- Define the filling rate and emptying rate as given conditions.
def R_fill : ℚ := 1 / 5
def R_empty : ℚ := 1 / 9

-- Define the net rate when both taps are opened simultaneously.
def R_net : ℚ := R_fill - R_empty

-- The total time to fill the cistern when both taps are opened.
def fill_time := 1 / R_net

-- Prove that the total time to fill the cistern is 11.25 hours.
theorem cistern_fill_time : fill_time = 11.25 := 
by 
    -- We include sorry to bypass the actual proof. This will allow the code to compile.
    sorry

end NUMINAMATH_GPT_cistern_fill_time_l74_7405


namespace NUMINAMATH_GPT_annual_population_increase_l74_7448

theorem annual_population_increase 
  (P : ℕ) (A : ℕ) (t : ℕ) (r : ℚ)
  (hP : P = 10000)
  (hA : A = 14400)
  (ht : t = 2)
  (h_eq : A = P * (1 + r)^t) :
  r = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_annual_population_increase_l74_7448


namespace NUMINAMATH_GPT_xy_product_solution_l74_7487

theorem xy_product_solution (x y : ℝ)
  (h1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (h2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -1 / Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_xy_product_solution_l74_7487


namespace NUMINAMATH_GPT_sally_found_more_balloons_l74_7443

def sally_original_balloons : ℝ := 9.0
def sally_new_balloons : ℝ := 11.0

theorem sally_found_more_balloons :
  sally_new_balloons - sally_original_balloons = 2.0 :=
by
  -- math proof goes here
  sorry

end NUMINAMATH_GPT_sally_found_more_balloons_l74_7443


namespace NUMINAMATH_GPT_quadratic_equation_divisible_by_x_minus_one_l74_7430

theorem quadratic_equation_divisible_by_x_minus_one (a b c : ℝ) (h1 : (x - 1) ∣ (a * x * x + b * x + c)) (h2 : c = 2) :
  (a = 1 ∧ b = -3 ∧ c = 2) → a * x * x + b * x + c = x^2 - 3 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_divisible_by_x_minus_one_l74_7430


namespace NUMINAMATH_GPT_calculate_expression_l74_7479

theorem calculate_expression :
  5 * 6 - 2 * 3 + 7 * 4 + 9 * 2 = 70 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l74_7479


namespace NUMINAMATH_GPT_molecular_weight_correct_l74_7431

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_N_in_N2O3 : ℕ := 2
def num_O_in_N2O3 : ℕ := 3

def molecular_weight_N2O3 : ℝ :=
  (num_N_in_N2O3 * atomic_weight_N) + (num_O_in_N2O3 * atomic_weight_O)

theorem molecular_weight_correct :
  molecular_weight_N2O3 = 76.02 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l74_7431


namespace NUMINAMATH_GPT_abcdeq_five_l74_7445

theorem abcdeq_five (a b c d : ℝ) 
    (h1 : a + b + c + d = 20) 
    (h2 : ab + ac + ad + bc + bd + cd = 150) : 
    a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := 
  by
  sorry

end NUMINAMATH_GPT_abcdeq_five_l74_7445


namespace NUMINAMATH_GPT_min_value_expression_l74_7476

theorem min_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 1) :
  9 ≤ (1 / (a^2 + 2 * b^2)) + (1 / (b^2 + 2 * c^2)) + (1 / (c^2 + 2 * a^2)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l74_7476


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l74_7406

theorem remainder_when_divided_by_6 (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 :=
by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l74_7406


namespace NUMINAMATH_GPT_crayons_selection_l74_7419

theorem crayons_selection : 
  ∃ (n : ℕ), n = Nat.choose 14 4 ∧ n = 1001 := by
  sorry

end NUMINAMATH_GPT_crayons_selection_l74_7419


namespace NUMINAMATH_GPT_tap_B_fills_remaining_pool_l74_7438

theorem tap_B_fills_remaining_pool :
  ∀ (flow_A flow_B : ℝ) (t_A t_B : ℕ),
  flow_A = 7.5 / 100 →  -- A fills 7.5% of the pool per hour
  flow_B = 5 / 100 →    -- B fills 5% of the pool per hour
  t_A = 2 →             -- A is open for 2 hours during the second phase
  t_A * flow_A = 15 / 100 →  -- A fills 15% of the pool in 2 hours
  4 * (flow_A + flow_B) = 50 / 100 →  -- A and B together fill 50% of the pool in 4 hours
  (100 / 100 - 50 / 100 - 15 / 100) / flow_B = t_B →  -- remaining pool filled only by B
  t_B = 7 := sorry    -- Prove that t_B is 7

end NUMINAMATH_GPT_tap_B_fills_remaining_pool_l74_7438


namespace NUMINAMATH_GPT_rectangle_length_eq_fifty_l74_7493

theorem rectangle_length_eq_fifty (x : ℝ) :
  (∃ w : ℝ, 6 * x * w = 6000 ∧ w = (2 / 5) * x) → x = 50 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_eq_fifty_l74_7493


namespace NUMINAMATH_GPT_initial_weight_of_beef_l74_7401

theorem initial_weight_of_beef (W : ℝ) 
  (stage1 : W' = 0.70 * W) 
  (stage2 : W'' = 0.80 * W') 
  (stage3 : W''' = 0.50 * W'') 
  (final_weight : W''' = 315) : 
  W = 1125 := by 
  sorry

end NUMINAMATH_GPT_initial_weight_of_beef_l74_7401


namespace NUMINAMATH_GPT_cookies_left_for_Monica_l74_7421

-- Definitions based on the conditions
def total_cookies : ℕ := 30
def father_cookies : ℕ := 10
def mother_cookies : ℕ := father_cookies / 2
def brother_cookies : ℕ := mother_cookies + 2

-- Statement for the theorem
theorem cookies_left_for_Monica : total_cookies - (father_cookies + mother_cookies + brother_cookies) = 8 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_cookies_left_for_Monica_l74_7421


namespace NUMINAMATH_GPT_sum_of_remainders_l74_7464

theorem sum_of_remainders (n : ℤ) (h : n % 15 = 7) : 
  (n % 3) + (n % 5) = 3 := 
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l74_7464


namespace NUMINAMATH_GPT_company_pays_240_per_month_l74_7474

-- Conditions as definitions
def box_length : ℕ := 15
def box_width : ℕ := 12
def box_height : ℕ := 10
def total_volume : ℕ := 1080000      -- 1.08 million cubic inches
def price_per_box_per_month : ℚ := 0.4

-- The volume of one box
def box_volume : ℕ := box_length * box_width * box_height

-- Calculate the number of boxes
def number_of_boxes : ℕ := total_volume / box_volume

-- Total amount paid per month for record storage
def total_amount_paid_per_month : ℚ := number_of_boxes * price_per_box_per_month

-- Theorem statement to prove
theorem company_pays_240_per_month : total_amount_paid_per_month = 240 := 
by 
  sorry

end NUMINAMATH_GPT_company_pays_240_per_month_l74_7474


namespace NUMINAMATH_GPT_valerie_initial_money_l74_7409

theorem valerie_initial_money (n m C_s C_l L I : ℕ) 
  (h1 : n = 3) (h2 : m = 1) (h3 : C_s = 8) (h4 : C_l = 12) (h5 : L = 24) :
  I = (n * C_s) + (m * C_l) + L :=
  sorry

end NUMINAMATH_GPT_valerie_initial_money_l74_7409
