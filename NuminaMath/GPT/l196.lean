import Mathlib

namespace pascal_30th_31st_numbers_l196_196636

-- Definitions based on conditions
def pascal_triangle_row_34 (k : ℕ) : ℕ := Nat.choose 34 k

-- Problem statement in Lean 4: proving the equations
theorem pascal_30th_31st_numbers :
  pascal_triangle_row_34 29 = 278256 ∧
  pascal_triangle_row_34 30 = 46376 :=
by
  sorry

end pascal_30th_31st_numbers_l196_196636


namespace age_of_b_l196_196444

theorem age_of_b (a b : ℕ) 
(h1 : a + 10 = 2 * (b - 10)) 
(h2 : a = b + 4) : 
b = 34 := 
sorry

end age_of_b_l196_196444


namespace similar_triangles_perimeter_ratio_l196_196279

theorem similar_triangles_perimeter_ratio
  (a₁ a₂ s₁ s₂ : ℝ)
  (h₁ : a₁ / a₂ = 1 / 4)
  (h₂ : s₁ / s₂ = 1 / 2) :
  (s₁ / s₂ = 1 / 2) :=
by {
  sorry
}

end similar_triangles_perimeter_ratio_l196_196279


namespace number_of_pumps_l196_196664

theorem number_of_pumps (P : ℕ) : 
  (P * 8 * 2 = 8 * 6) → P = 3 :=
by
  intro h
  sorry

end number_of_pumps_l196_196664


namespace crayons_eaten_correct_l196_196699

variable (initial_crayons final_crayons : ℕ)

def crayonsEaten (initial_crayons final_crayons : ℕ) : ℕ :=
  initial_crayons - final_crayons

theorem crayons_eaten_correct : crayonsEaten 87 80 = 7 :=
  by
  sorry

end crayons_eaten_correct_l196_196699


namespace percent_of_dollar_is_37_l196_196164

variable (coins_value_in_cents : ℕ)
variable (percent_of_one_dollar : ℕ)

def value_of_pennies : ℕ := 2 * 1
def value_of_nickels : ℕ := 3 * 5
def value_of_dimes : ℕ := 2 * 10

def total_coin_value : ℕ := value_of_pennies + value_of_nickels + value_of_dimes

theorem percent_of_dollar_is_37
  (h1 : total_coin_value = coins_value_in_cents)
  (h2 : percent_of_one_dollar = (coins_value_in_cents * 100) / 100) : 
  percent_of_one_dollar = 37 := 
by
  sorry

end percent_of_dollar_is_37_l196_196164


namespace probability_odd_divisor_15_factorial_l196_196566

theorem probability_odd_divisor_15_factorial :
  let number_of_divisors_15_fact : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let number_of_odd_divisors_15_fact : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (number_of_odd_divisors_15_fact : ℝ) / (number_of_divisors_15_fact : ℝ) = 1 / 12 :=
by
  sorry

end probability_odd_divisor_15_factorial_l196_196566


namespace distance_A_to_B_l196_196865

theorem distance_A_to_B (D_B D_C V_E V_F : ℝ) (h1 : D_B / 3 = V_E)
  (h2 : D_C / 4 = V_F) (h3 : V_E / V_F = 2.533333333333333)
  (h4 : D_B = 300 ∨ D_C = 300) : D_B = 570 :=
by
  -- Proof yet to be provided
  sorry

end distance_A_to_B_l196_196865


namespace hyperbola_asymptote_m_value_l196_196902

theorem hyperbola_asymptote_m_value
  (m : ℝ)
  (h1 : m > 0)
  (h2 : ∀ x y : ℝ, (5 * x - 2 * y = 0) → ((x^2 / 4) - (y^2 / m^2) = 1)) :
  m = 5 :=
sorry

end hyperbola_asymptote_m_value_l196_196902


namespace right_triangle_incenter_distance_l196_196438

noncomputable def triangle_right_incenter_distance : ℝ :=
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let area := (1 / 2) * AB * BC
  let s := (AB + BC + AC) / 2
  let r := area / s
  r

theorem right_triangle_incenter_distance :
  let AB := 4 * Real.sqrt 2
  let BC := 6
  let AC := 2 * Real.sqrt 17
  let area := 12 * Real.sqrt 2
  let s := 2 * Real.sqrt 2 + 3 + Real.sqrt 17
  let BI := area / s
  BI = triangle_right_incenter_distance := sorry

end right_triangle_incenter_distance_l196_196438


namespace mark_spends_47_l196_196200

def apple_price : ℕ := 2
def apple_quantity : ℕ := 4
def bread_price : ℕ := 3
def bread_quantity : ℕ := 5
def cheese_price : ℕ := 6
def cheese_quantity : ℕ := 3
def cereal_price : ℕ := 5
def cereal_quantity : ℕ := 4
def coupon : ℕ := 10

def calculate_total_cost (apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon : ℕ) : ℕ :=
  let apples_cost := apple_price * (apple_quantity / 2)  -- Apply buy-one-get-one-free
  let bread_cost := bread_price * bread_quantity
  let cheese_cost := cheese_price * cheese_quantity
  let cereal_cost := cereal_price * cereal_quantity
  let subtotal := apples_cost + bread_cost + cheese_cost + cereal_cost
  let total_cost := if subtotal > 50 then subtotal - coupon else subtotal
  total_cost

theorem mark_spends_47 : calculate_total_cost apple_price apple_quantity bread_price bread_quantity cheese_price cheese_quantity cereal_price cereal_quantity coupon = 47 :=
  sorry

end mark_spends_47_l196_196200


namespace average_of_first_12_results_l196_196575

theorem average_of_first_12_results
  (average_25_results : ℝ)
  (average_last_12_results : ℝ)
  (result_13th : ℝ)
  (total_results : ℕ)
  (num_first_12 : ℕ)
  (num_last_12 : ℕ)
  (total_sum : ℝ)
  (sum_first_12 : ℝ)
  (sum_last_12 : ℝ)
  (A : ℝ)
  (h1 : average_25_results = 24)
  (h2 : average_last_12_results = 17)
  (h3 : result_13th = 228)
  (h4 : total_results = 25)
  (h5 : num_first_12 = 12)
  (h6 : num_last_12 = 12)
  (h7 : total_sum = average_25_results * total_results)
  (h8 : sum_last_12 = average_last_12_results * num_last_12)
  (h9 : total_sum = sum_first_12 + result_13th + sum_last_12)
  (h10 : sum_first_12 = A * num_first_12) :
  A = 14 :=
by
  sorry

end average_of_first_12_results_l196_196575


namespace three_numbers_less_or_equal_than_3_l196_196669

theorem three_numbers_less_or_equal_than_3 : 
  let a := 0.8
  let b := 0.5
  let c := 0.9
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) → 
  3 = 3 :=
by
  intros h
  sorry

end three_numbers_less_or_equal_than_3_l196_196669


namespace problem_l196_196081

theorem problem
  (r s t : ℝ)
  (h₀ : r^3 - 15 * r^2 + 13 * r - 8 = 0)
  (h₁ : s^3 - 15 * s^2 + 13 * s - 8 = 0)
  (h₂ : t^3 - 15 * t^2 + 13 * t - 8 = 0) :
  (r / (1 / r + s * t) + s / (1 / s + t * r) + t / (1 / t + r * s) = 199 / 9) :=
sorry

end problem_l196_196081


namespace cheryl_initial_skitttles_l196_196126

-- Given conditions
def cheryl_ends_with (ends_with : ℕ) : Prop := ends_with = 97
def kathryn_gives (gives : ℕ) : Prop := gives = 89

-- To prove: cheryl_starts_with + kathryn_gives = cheryl_ends_with
theorem cheryl_initial_skitttles (cheryl_starts_with : ℕ) :
  (∃ ends_with gives, cheryl_ends_with ends_with ∧ kathryn_gives gives ∧ 
  cheryl_starts_with + gives = ends_with) →
  cheryl_starts_with = 8 :=
by
  sorry

end cheryl_initial_skitttles_l196_196126


namespace tax_free_value_is_500_l196_196504

-- Definitions of the given conditions
def total_value : ℝ := 730
def paid_tax : ℝ := 18.40
def tax_rate : ℝ := 0.08

-- Definition of the excess value
def excess_value (E : ℝ) := tax_rate * E = paid_tax

-- Definition of the tax-free threshold value
def tax_free_limit (V : ℝ) := total_value - (paid_tax / tax_rate) = V

-- The theorem to be proven
theorem tax_free_value_is_500 : 
  ∃ V : ℝ, (total_value - (paid_tax / tax_rate) = V) ∧ V = 500 :=
  by
    sorry -- Proof to be completed

end tax_free_value_is_500_l196_196504


namespace part1_part2_part3_l196_196585

theorem part1 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0 ↔ x < -3 ∨ x > -2) : k = -2/5 :=
sorry

theorem part2 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) : k < -Real.sqrt 6 / 6 :=
sorry

theorem part3 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, ¬ (k * x^2 - 2 * x + 6 * k < 0)) : k ≥ Real.sqrt 6 / 6 :=
sorry

end part1_part2_part3_l196_196585


namespace value_of_Y_l196_196066

/- Define the conditions given in the problem -/
def first_row_arithmetic_seq (a1 d1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d1
def fourth_row_arithmetic_seq (a4 d4 : ℕ) (n : ℕ) : ℕ := a4 + (n - 1) * d4

/- Constants given by the problem -/
def a1 : ℕ := 3
def fourth_term_first_row : ℕ := 27
def a4 : ℕ := 6
def fourth_term_fourth_row : ℕ := 66

/- Calculating common differences for first and fourth rows -/
def d1 : ℕ := (fourth_term_first_row - a1) / 3
def d4 : ℕ := (fourth_term_fourth_row - a4) / 3

/- Note that we are given that Y is at position (2, 2)
   Express Y in definition forms -/
def Y_row := first_row_arithmetic_seq (a1 + d1) d4 2
def Y_column := fourth_row_arithmetic_seq (a4 + d4) d1 2

/- Problem statement in Lean 4 -/
theorem value_of_Y : Y_row = 35 ∧ Y_column = 35 := by
  sorry

end value_of_Y_l196_196066


namespace gravel_weight_40_pounds_l196_196610

def weight_of_gravel_in_mixture (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) : ℝ :=
total_weight - (sand_fraction * total_weight + water_fraction * total_weight)

theorem gravel_weight_40_pounds
  (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) 
  (h1 : total_weight = 40) (h2 : sand_fraction = 1 / 4) (h3 : water_fraction = 2 / 5) :
  weight_of_gravel_in_mixture total_weight sand_fraction water_fraction = 14 :=
by
  -- Proof omitted
  sorry

end gravel_weight_40_pounds_l196_196610


namespace initial_temperature_is_20_l196_196321

-- Define the initial temperature, final temperature, rate of increase and time
def T_initial (T_final : ℕ) (rate_of_increase : ℕ) (time : ℕ) : ℕ :=
  T_final - rate_of_increase * time

-- Statement: The initial temperature is 20 degrees given the specified conditions.
theorem initial_temperature_is_20 :
  T_initial 100 5 16 = 20 :=
by
  sorry

end initial_temperature_is_20_l196_196321


namespace employees_original_number_l196_196608

noncomputable def original_employees_approx (employees_remaining : ℝ) (reduction_percent : ℝ) : ℝ :=
  employees_remaining / (1 - reduction_percent)

theorem employees_original_number (employees_remaining : ℝ) (reduction_percent : ℝ) (original : ℝ) :
  employees_remaining = 462 → reduction_percent = 0.276 →
  abs (original_employees_approx employees_remaining reduction_percent - original) < 1 →
  original = 638 :=
by
  intros h_remaining h_reduction h_approx
  sorry

end employees_original_number_l196_196608


namespace max_correct_answers_l196_196100

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
by {
  sorry
}

end max_correct_answers_l196_196100


namespace find_m_l196_196222

open Real

namespace VectorPerpendicular

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := (v₁.1 * v₂.1 + v₁.2 * v₂.2) = 0

theorem find_m (m : ℝ) (h : perpendicular a (b m)) : m = 1 / 2 :=
by
  sorry -- Proof is omitted

end VectorPerpendicular

end find_m_l196_196222


namespace calculate_expression_l196_196471

theorem calculate_expression (a : ℤ) (h : a = -2) : a^3 - a^2 = -12 := 
by
  sorry

end calculate_expression_l196_196471


namespace riza_son_age_l196_196481

theorem riza_son_age (R S : ℕ) (h1 : R = S + 25) (h2 : R + S = 105) : S = 40 :=
by
  sorry

end riza_son_age_l196_196481


namespace sum_of_coefficients_l196_196324

theorem sum_of_coefficients :
  ∃ (A B C D E F G H J K : ℤ),
  (∀ x y : ℤ, 125 * x ^ 8 - 2401 * y ^ 8 = (A * x + B * y) * (C * x ^ 4 + D * x * y + E * y ^ 4) * (F * x + G * y) * (H * x ^ 4 + J * x * y + K * y ^ 4))
  ∧ A + B + C + D + E + F + G + H + J + K = 102 := 
sorry

end sum_of_coefficients_l196_196324


namespace det_A_l196_196905

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![2, -4, 5],
  ![0, 6, -2],
  ![3, -1, 2]
]

theorem det_A : A.det = -46 := by
  sorry

end det_A_l196_196905


namespace range_g_a_values_l196_196010

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem range_g : ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 :=
sorry

theorem a_values (a : ℝ) : (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) :=
sorry

end range_g_a_values_l196_196010


namespace max_net_income_meeting_point_l196_196113

theorem max_net_income_meeting_point :
  let A := (9 : ℝ)
  let B := (6 : ℝ)
  let cost_per_mile := 1
  let payment_per_mile := 2
  ∃ x : ℝ, 
  let AP := Real.sqrt ((x - 9)^2 + 12^2)
  let PB := Real.sqrt ((x - 6)^2 + 3^2)
  let net_income := payment_per_mile * PB - (AP + PB)
  x = -12.5 := 
sorry

end max_net_income_meeting_point_l196_196113


namespace domain_f_l196_196648

open Real

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - 3

theorem domain_f :
  {x : ℝ | g x > 0} = {x : ℝ | x < 0 ∨ x > 3} :=
by 
  sorry

end domain_f_l196_196648


namespace remaining_units_correct_l196_196998

-- Definitions based on conditions
def total_units : ℕ := 2000
def fraction_built_in_first_half : ℚ := 3/5
def additional_units_by_october : ℕ := 300

-- Calculate units built in the first half of the year
def units_built_in_first_half : ℚ := fraction_built_in_first_half * total_units

-- Remaining units after the first half of the year
def remaining_units_after_first_half : ℚ := total_units - units_built_in_first_half

-- Remaining units after building additional units by October
def remaining_units_to_be_built : ℚ := remaining_units_after_first_half - additional_units_by_october

-- Theorem statement: Prove remaining units to be built is 500
theorem remaining_units_correct : remaining_units_to_be_built = 500 := by
  sorry

end remaining_units_correct_l196_196998


namespace units_digit_of_p_is_6_l196_196185

theorem units_digit_of_p_is_6 (p : ℕ) (h_even : Even p) (h_units_p_plus_1 : (p + 1) % 10 = 7) (h_units_p3_minus_p2 : ((p^3) % 10 - (p^2) % 10) % 10 = 0) : p % 10 = 6 := 
by 
  -- proof steps go here
  sorry

end units_digit_of_p_is_6_l196_196185


namespace solve_inequality_l196_196562

theorem solve_inequality (a x : ℝ) :
  ((x - a) * (x - 2 * a) < 0) ↔ 
  ((a < 0 ∧ 2 * a < x ∧ x < a) ∨ (a = 0 ∧ false) ∨ (a > 0 ∧ a < x ∧ x < 2 * a)) :=
by sorry

end solve_inequality_l196_196562


namespace number_of_dogs_l196_196758

-- Define variables for the number of cats (C) and dogs (D)
variables (C D : ℕ)

-- Define the conditions from the problem statement
def condition1 : Prop := C = D - 6
def condition2 : Prop := C * 3 = D * 2

-- State the theorem that D should be 18 given the conditions
theorem number_of_dogs (h1 : condition1 C D) (h2 : condition2 C D) : D = 18 :=
  sorry

end number_of_dogs_l196_196758


namespace smallest_pos_int_y_satisfies_congruence_l196_196379

theorem smallest_pos_int_y_satisfies_congruence :
  ∃ y : ℕ, (y > 0) ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, (z > 0) ∧ (26 * z + 8) % 16 = 4 → y ≤ z :=
sorry

end smallest_pos_int_y_satisfies_congruence_l196_196379


namespace geometric_sequence_third_term_l196_196316

theorem geometric_sequence_third_term (r : ℕ) (a : ℕ) (h1 : a = 6) (h2 : a * r^3 = 384) : a * r^2 = 96 :=
by
  sorry

end geometric_sequence_third_term_l196_196316


namespace john_pays_per_year_l196_196914

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l196_196914


namespace point_not_in_second_quadrant_l196_196340

theorem point_not_in_second_quadrant (m : ℝ) : ¬ (m^2 + m ≤ 0 ∧ m - 1 ≥ 0) :=
by
  sorry

end point_not_in_second_quadrant_l196_196340


namespace find_y_in_terms_of_abc_l196_196745

theorem find_y_in_terms_of_abc 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (h1 : xy / (x - y) = a)
  (h2 : xz / (x - z) = b)
  (h3 : yz / (y - z) = c) :
  y = bcx / ((b + c) * x - bc) := 
sorry

end find_y_in_terms_of_abc_l196_196745


namespace divisor_is_seven_l196_196939

theorem divisor_is_seven 
  (d x : ℤ)
  (h1 : x % d = 5)
  (h2 : 4 * x % d = 6) :
  d = 7 := 
sorry

end divisor_is_seven_l196_196939


namespace min_tan_expression_l196_196483

open Real

theorem min_tan_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
(h_eq : sin α * cos β - 2 * cos α * sin β = 0) :
  ∃ x, x = tan (2 * π + α) + tan (π / 2 - β) ∧ x = 2 * sqrt 2 :=
sorry

end min_tan_expression_l196_196483


namespace smallest_positive_period_max_value_in_interval_min_value_in_interval_l196_196808

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = 2 := by
  sorry

theorem min_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = -1 := by
  sorry

end smallest_positive_period_max_value_in_interval_min_value_in_interval_l196_196808


namespace unique_solution_to_functional_eq_l196_196013

theorem unique_solution_to_functional_eq :
  (∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2) :=
by
  sorry

end unique_solution_to_functional_eq_l196_196013


namespace extremum_of_function_l196_196740

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem extremum_of_function :
  (∀ x, f x ≥ -Real.exp 1) ∧ (f 1 = -Real.exp 1) ∧ (∀ M, ∃ x, f x > M) :=
by
  sorry

end extremum_of_function_l196_196740


namespace solve_first_sales_amount_l196_196238

noncomputable def first_sales_amount
  (S : ℝ) (R : ℝ) (next_sales_royalties : ℝ) (next_sales_amount : ℝ) : Prop :=
  (3 = R * S) ∧ (next_sales_royalties = 0.85 * R * next_sales_amount)

theorem solve_first_sales_amount (S R : ℝ) :
  first_sales_amount S R 9 108 → S = 30.6 :=
by
  intro h
  sorry

end solve_first_sales_amount_l196_196238


namespace minimum_n_is_835_l196_196704

def problem_statement : Prop :=
  ∀ (S : Finset ℕ), S.card = 835 → (∀ (T : Finset ℕ), T ⊆ S → T.card = 4 →
    ∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + 2 * b + 3 * c = d)

theorem minimum_n_is_835 : problem_statement :=
sorry

end minimum_n_is_835_l196_196704


namespace bridget_poster_board_side_length_l196_196319

theorem bridget_poster_board_side_length
  (num_cards : ℕ)
  (card_length : ℕ)
  (card_width : ℕ)
  (posterboard_area : ℕ)
  (posterboard_side_length_feet : ℕ)
  (posterboard_side_length_inches : ℕ)
  (cards_area : ℕ) :
  num_cards = 24 ∧
  card_length = 2 ∧
  card_width = 3 ∧
  posterboard_area = posterboard_side_length_inches ^ 2 ∧
  cards_area = num_cards * (card_length * card_width) ∧
  cards_area = posterboard_area ∧
  posterboard_side_length_inches = 12 ∧
  posterboard_side_length_feet = posterboard_side_length_inches / 12 →
  posterboard_side_length_feet = 1 :=
sorry

end bridget_poster_board_side_length_l196_196319


namespace simplest_form_expression_l196_196508

theorem simplest_form_expression (x y a : ℤ) : 
  (∃ (E : ℚ → Prop), (E (1/3) ∨ E (1/(x-2)) ∨ E ((x^2 * y) / (2*x)) ∨ E (2*a / 8)) → (E (1/(x-2)) ↔ E (1/(x-2)))) :=
by 
  sorry

end simplest_form_expression_l196_196508


namespace misha_is_lying_l196_196244

theorem misha_is_lying
  (truth_tellers_scores : Fin 9 → ℕ)
  (h_all_odd : ∀ i, truth_tellers_scores i % 2 = 1)
  (total_scores_truth_tellers : (Fin 9 → ℕ) → ℕ)
  (h_sum_scores : total_scores_truth_tellers truth_tellers_scores = 18) :
  ∀ (misha_score : ℕ), misha_score = 2 → misha_score % 2 = 1 → False :=
by
  intros misha_score hms hmo
  sorry

end misha_is_lying_l196_196244


namespace find_h_l196_196817

def infinite_sqrt_series (b : ℝ) : ℝ := sorry -- Placeholder for infinite series sqrt(b + sqrt(b + ...))

def diamond (a b : ℝ) : ℝ :=
  a^2 + infinite_sqrt_series b

theorem find_h (h : ℝ) : diamond 3 h = 12 → h = 6 :=
by
  intro h_condition
  -- Further steps will be used during proof
  sorry

end find_h_l196_196817


namespace probability_sum_less_than_16_l196_196606

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end probability_sum_less_than_16_l196_196606


namespace prove_f_neg_2_l196_196850

noncomputable def f (a b x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Main theorem statement
theorem prove_f_neg_2 (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := 
by
  sorry

end prove_f_neg_2_l196_196850


namespace average_age_l196_196360
open Nat

def age_to_months (years : ℕ) (months : ℕ) : ℕ := years * 12 + months

theorem average_age :
  let age1 := age_to_months 14 9
  let age2 := age_to_months 15 1
  let age3 := age_to_months 14 8
  let total_months := age1 + age2 + age3
  let avg_months := total_months / 3
  let avg_years := avg_months / 12
  let avg_remaining_months := avg_months % 12
  avg_years = 14 ∧ avg_remaining_months = 10 := by
  sorry

end average_age_l196_196360


namespace GreenValley_Absent_Percentage_l196_196952

theorem GreenValley_Absent_Percentage 
  (total_students boys girls absent_boys_frac absent_girls_frac : ℝ)
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : absent_boys_frac = 1 / 7)
  (h5 : absent_girls_frac = 1 / 5) :
  (absent_boys_frac * boys + absent_girls_frac * girls) / total_students * 100 = 16.67 := 
sorry

end GreenValley_Absent_Percentage_l196_196952


namespace time_to_cut_mans_hair_l196_196252

theorem time_to_cut_mans_hair :
  ∃ (x : ℕ),
    (3 * 50) + (2 * x) + (3 * 25) = 255 ∧ x = 15 :=
by {
  sorry
}

end time_to_cut_mans_hair_l196_196252


namespace correct_understanding_of_philosophy_l196_196487

-- Define the conditions based on the problem statement
def philosophy_from_life_and_practice : Prop :=
  -- Philosophy originates from people's lives and practice.
  sorry
  
def philosophy_affects_lives : Prop :=
  -- Philosophy consciously or unconsciously affects people's lives, learning, and work
  sorry

def philosophical_knowledge_requires_learning : Prop :=
  true

def philosophy_not_just_summary : Prop :=
  true

-- Given conditions 1, 2, 3 (as negation of 3 in original problem), and 4 (as negation of 4 in original problem),
-- We need to prove the correct understanding (which is combination ①②) is correct.
theorem correct_understanding_of_philosophy :
  philosophy_from_life_and_practice →
  philosophy_affects_lives →
  philosophical_knowledge_requires_learning →
  philosophy_not_just_summary →
  (philosophy_from_life_and_practice ∧ philosophy_affects_lives) :=
by
  intros
  apply And.intro
  · assumption
  · assumption

end correct_understanding_of_philosophy_l196_196487


namespace question1_question2_l196_196837

-- Question 1
theorem question1 (a : ℝ) (h : a = 1 / 2) :
  let A := {x | -1 / 2 < x ∧ x < 2}
  let B := {x | 0 < x ∧ x < 1}
  A ∩ B = {x | 0 < x ∧ x < 1} :=
by
  sorry

-- Question 2
theorem question2 (a : ℝ) :
  let A := {x | a - 1 < x ∧ x < 2 * a + 1}
  let B := {x | 0 < x ∧ x < 1}
  (A ∩ B = ∅) ↔ (a ≤ -1/2 ∨ a ≥ 2) :=
by
  sorry

end question1_question2_l196_196837


namespace parallelepiped_face_areas_l196_196653

theorem parallelepiped_face_areas
    (h₁ : ℝ := 2)  -- height corresponding to face area x
    (h₂ : ℝ := 3)  -- height corresponding to face area y
    (h₃ : ℝ := 4)  -- height corresponding to face area z
    (total_surface_area : ℝ := 36) : 
    ∃ (x y z : ℝ), 
    2 * x + 2 * y + 2 * z = total_surface_area ∧
    (∃ V : ℝ, V = h₁ * x ∧ V = h₂ * y ∧ V = h₃ * z) ∧
    x = 108 / 13 ∧ y = 72 / 13 ∧ z = 54 / 13 := 
by 
  sorry

end parallelepiped_face_areas_l196_196653


namespace probability_of_9_heads_in_12_flips_l196_196169

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| _, 0       => 1
| 0, _       => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 2^12

-- Define the number of ways to get exactly 9 heads in 12 flips
def favorable_outcomes : ℕ := binom 12 9

-- Define the probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 220/4096
theorem probability_of_9_heads_in_12_flips : probability = 220 / 4096 := by
  sorry

end probability_of_9_heads_in_12_flips_l196_196169


namespace quadratic_eq_roots_minus5_and_7_l196_196933

theorem quadratic_eq_roots_minus5_and_7 : ∀ x : ℝ, (x + 5) * (x - 7) = 0 ↔ x = -5 ∨ x = 7 := by
  sorry

end quadratic_eq_roots_minus5_and_7_l196_196933


namespace f_strictly_increasing_solve_inequality_l196_196795

variable (f : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 3

-- Prove monotonicity
theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Solve the inequality
theorem solve_inequality (m : ℝ) : -2/3 < m ∧ m < 2 ↔ f (3 * m^2 - m - 2) < 2 := by
  sorry

end f_strictly_increasing_solve_inequality_l196_196795


namespace total_opponents_runs_l196_196032

theorem total_opponents_runs (team_scores : List ℕ) (opponent_scores : List ℕ) :
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  ∃ lost_games won_games opponent_lost_scores opponent_won_scores,
    lost_games = [1, 3, 5, 7, 9, 11] ∧
    won_games = [2, 4, 6, 8, 10, 12] ∧
    (∀ (t : ℕ), t ∈ lost_games → ∃ o : ℕ, o = t + 1 ∧ o ∈ opponent_scores) ∧
    (∀ (t : ℕ), t ∈ won_games → ∃ o : ℕ, o = t / 2 ∧ o ∈ opponent_scores) ∧
    opponent_scores = opponent_lost_scores ++ opponent_won_scores ∧
    opponent_lost_scores = [2, 4, 6, 8, 10, 12] ∧
    opponent_won_scores = [1, 2, 3, 4, 5, 6] →
  opponent_scores.sum = 63 :=
by
  sorry

end total_opponents_runs_l196_196032


namespace total_students_l196_196243

theorem total_students (boys girls : ℕ) (h_boys : boys = 127) (h_girls : girls = boys + 212) : boys + girls = 466 :=
by
  sorry

end total_students_l196_196243


namespace part1_solution_part2_solution_l196_196160

-- Define the inequality
def inequality (m x : ℝ) : Prop := (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0

-- Part (1): Prove the solution set for m = 0 is (-2, 1)
theorem part1_solution :
  (∀ x : ℝ, inequality 0 x → (-2 : ℝ) < x ∧ x < 1) := 
by
  sorry

-- Part (2): Prove the range of values for m such that the solution set is R
theorem part2_solution (m : ℝ) :
  (∀ x : ℝ, inequality m x) ↔ (1 ≤ m ∧ m < 9) := 
by
  sorry

end part1_solution_part2_solution_l196_196160


namespace smallest_integer_in_set_l196_196676

def avg_seven_consecutive_integers (n : ℤ) : ℤ :=
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7

theorem smallest_integer_in_set : ∃ (n : ℤ), n = 0 ∧ (n + 6 < 3 * avg_seven_consecutive_integers n) :=
by
  sorry

end smallest_integer_in_set_l196_196676


namespace symmetry_about_origin_l196_196470

def Point : Type := ℝ × ℝ

def A : Point := (2, -1)
def B : Point := (-2, 1)

theorem symmetry_about_origin (A B : Point) : A = (2, -1) ∧ B = (-2, 1) → B = (-A.1, -A.2) :=
by
  sorry

end symmetry_about_origin_l196_196470


namespace part1_tangent_line_part2_monotonicity_l196_196771

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x ^ 2 - 2 * a * x) * Real.log x - x ^ 2 + 4 * a * x + 1

theorem part1_tangent_line (a : ℝ) (h : a = 0) :
  let e := Real.exp 1
  let f_x := f e 0
  let tangent_line := 4 * e - 3 * e ^ 2 + 1
  tangent_line = 4 * e * (x - e) + f_x :=
sorry

theorem part2_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → f (x) a > 0 ↔ a ≤ 0) ∧
  (∀ x : ℝ, 0 < x → x < a → f (x) a > 0 ↔ 0 < a ∧ a < 1) ∧
  (∀ x : ℝ, 1 < x → x < a → f (x) a < 0 ↔ a > 1) ∧
  (∀ x : ℝ, 0 < x → 1 < x → x < a → f (x) a < 0 ↔ (a > 1)) ∧
  (∀ x : ℝ, x > 1 → f (x) a > 0 ↔ (a < 1)) :=
sorry

end part1_tangent_line_part2_monotonicity_l196_196771


namespace find_ab_unique_l196_196528

theorem find_ab_unique (a b : ℕ) (h1 : a > 1) (h2 : b > a) (h3 : a ≤ 20) (h4 : b ≤ 20) (h5 : a * b = 52) (h6 : a + b = 17) : a = 4 ∧ b = 13 :=
by {
  -- Proof goes here
  sorry
}

end find_ab_unique_l196_196528


namespace correct_proposition_l196_196918

variable (a b : ℝ)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)
variable (a_gt_b : a > b)

theorem correct_proposition : 1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end correct_proposition_l196_196918


namespace product_telescope_identity_l196_196599

theorem product_telescope_identity :
  (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) = 8 :=
by
  sorry

end product_telescope_identity_l196_196599


namespace probability_first_green_then_blue_l196_196302

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l196_196302


namespace factorization_correct_l196_196114

theorem factorization_correct: 
  (a : ℝ) → a^2 - 9 = (a - 3) * (a + 3) :=
by
  intro a
  sorry

end factorization_correct_l196_196114


namespace consecutive_integers_sum_l196_196714

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l196_196714


namespace nelly_refrigerator_payment_l196_196303

theorem nelly_refrigerator_payment (T : ℝ) (p1 p2 p3 : ℝ) (p1_percent p2_percent p3_percent : ℝ)
  (h1 : p1 = 875) (h2 : p2 = 650) (h3 : p3 = 1200)
  (h4 : p1_percent = 0.25) (h5 : p2_percent = 0.15) (h6 : p3_percent = 0.35)
  (total_paid := p1 + p2 + p3)
  (percent_paid := p1_percent + p2_percent + p3_percent)
  (total_cost := total_paid / percent_paid)
  (remaining := total_cost - total_paid) :
  remaining = 908.33 := by
  sorry

end nelly_refrigerator_payment_l196_196303


namespace ratio_is_one_half_l196_196249

noncomputable def ratio_of_intercepts (b : ℝ) (hb : b ≠ 0) : ℝ :=
  let s := -b / 8
  let t := -b / 4
  s / t

theorem ratio_is_one_half (b : ℝ) (hb : b ≠ 0) :
  ratio_of_intercepts b hb = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l196_196249


namespace point_in_second_quadrant_l196_196122

structure Point where
  x : Int
  y : Int

-- Define point P
def P : Point := { x := -1, y := 2 }

-- Define the second quadrant condition
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

-- The mathematical statement to prove
theorem point_in_second_quadrant : second_quadrant P := by
  sorry

end point_in_second_quadrant_l196_196122


namespace age_difference_l196_196553

theorem age_difference (a b c : ℕ) (h₁ : b = 8) (h₂ : c = b / 2) (h₃ : a + b + c = 22) : a - b = 2 :=
by
  sorry

end age_difference_l196_196553


namespace initial_percentage_increase_l196_196241

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase_l196_196241


namespace evaluate_power_l196_196543

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end evaluate_power_l196_196543


namespace arccos_equivalence_l196_196283

open Real

theorem arccos_equivalence (α : ℝ) (h₀ : α ∈ Set.Icc 0 (2 * π)) (h₁ : cos α = 1 / 3) :
  α = arccos (1 / 3) ∨ α = 2 * π - arccos (1 / 3) := 
by 
  sorry

end arccos_equivalence_l196_196283


namespace stating_area_trapezoid_AMBQ_is_18_l196_196973

/-- Definition of the 20-sided polygon configuration with 2 unit sides and right-angle turns. -/
structure Polygon20 where
  sides : ℕ → ℝ
  units : ∀ i, sides i = 2
  right_angles : ∀ i, (i + 1) % 20 ≠ i -- Right angles between consecutive sides

/-- Intersection point of AJ and DP, named M, under the given polygon configuration. -/
def intersection_point (p : Polygon20) : ℝ × ℝ :=
  (5 * p.sides 0, 5 * p.sides 1)  -- Assuming relevant distances for simplicity

/-- Area of the trapezoid AMBQ formed given the defined Polygon20. -/
noncomputable def area_trapezoid_AMBQ (p : Polygon20) : ℝ :=
  let base1 := 10 * p.sides 0
  let base2 := 8 * p.sides 0
  let height := p.sides 0
  (base1 + base2) * height / 2

/-- 
  Theorem stating the area of the trapezoid AMBQ in the given configuration.
  We prove that the area is 18 units.
-/
theorem area_trapezoid_AMBQ_is_18 (p : Polygon20) :
  area_trapezoid_AMBQ p = 18 :=
sorry -- Proof to be done

end stating_area_trapezoid_AMBQ_is_18_l196_196973


namespace general_term_sequence_l196_196157

variable {a : ℕ → ℝ}
variable {n : ℕ}

def sequence_condition (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n ≥ 1 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0)

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n := by
  sorry

end general_term_sequence_l196_196157


namespace necessary_but_not_sufficient_condition_l196_196662

variable {a : ℕ → ℤ}

noncomputable def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ (m n k : ℕ), a m * a k = a n * a (m + k - n)

noncomputable def is_root_of_quadratic (x y : ℤ) : Prop :=
x^2 + 3*x + 1 = 0 ∧ y^2 + 3*y + 1 = 0

theorem necessary_but_not_sufficient_condition 
  (a : ℕ → ℤ)
  (hgeo : is_geometric_sequence a)
  (hroots : is_root_of_quadratic (a 4) (a 12)) :
  a 8 = -1 ↔ (∃ x y : ℤ, is_root_of_quadratic x y ∧ x + y = -3 ∧ x * y = 1) :=
sorry

end necessary_but_not_sufficient_condition_l196_196662


namespace quadratic_positivity_range_l196_196266

variable (a : ℝ)

def quadratic_function (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

theorem quadratic_positivity_range :
  (∀ x, 0 < x ∧ x < 3 → quadratic_function a x > 0)
  ↔ (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) := sorry

end quadratic_positivity_range_l196_196266


namespace remaining_dogs_eq_200_l196_196975

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end remaining_dogs_eq_200_l196_196975


namespace greatest_possible_sum_of_squares_l196_196936

theorem greatest_possible_sum_of_squares (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 :=
by sorry

end greatest_possible_sum_of_squares_l196_196936


namespace sector_area_l196_196258

def central_angle := 120 -- in degrees
def radius := 3 -- in units

theorem sector_area (n : ℕ) (R : ℕ) (h₁ : n = central_angle) (h₂ : R = radius) :
  (n * R^2 * Real.pi / 360) = 3 * Real.pi :=
by
  sorry

end sector_area_l196_196258


namespace one_add_i_cubed_eq_one_sub_i_l196_196929

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i :=
sorry

end one_add_i_cubed_eq_one_sub_i_l196_196929


namespace ellipse_eq_from_hyperbola_l196_196017

noncomputable def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = -1) →
  (x^2 / 4 + y^2 / 16 = 1)

theorem ellipse_eq_from_hyperbola :
  hyperbola_eq :=
by
  sorry

end ellipse_eq_from_hyperbola_l196_196017


namespace horner_rule_polynomial_polynomial_value_at_23_l196_196774

def polynomial (x : ℤ) : ℤ := 7 * x ^ 3 + 3 * x ^ 2 - 5 * x + 11

def horner_polynomial (x : ℤ) : ℤ := x * ((7 * x + 3) * x - 5) + 11

theorem horner_rule_polynomial (x : ℤ) : polynomial x = horner_polynomial x :=
by 
  -- The proof steps would go here,
  -- demonstrating that polynomial x = horner_polynomial x.
  sorry

-- Instantiation of the theorem for a specific value of x
theorem polynomial_value_at_23 : polynomial 23 = horner_polynomial 23 :=
by 
  -- Using the previously established theorem
  apply horner_rule_polynomial

end horner_rule_polynomial_polynomial_value_at_23_l196_196774


namespace is_exact_time_now_321_l196_196582

noncomputable def current_time_is_321 : Prop :=
  exists t : ℝ, 0 < t ∧ t < 60 ∧ |(6 * t + 48) - (90 + 0.5 * (t - 4))| = 180

theorem is_exact_time_now_321 : current_time_is_321 := 
  sorry

end is_exact_time_now_321_l196_196582


namespace sun_salutations_per_year_l196_196177

-- Definitions 
def sun_salutations_per_weekday : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_per_year : ℕ := 52

-- Problem statement to prove
theorem sun_salutations_per_year :
  sun_salutations_per_weekday * weekdays_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l196_196177


namespace boxes_needed_l196_196175

noncomputable def living_room_length : ℝ := 16
noncomputable def living_room_width : ℝ := 20
noncomputable def sq_ft_per_box : ℝ := 10
noncomputable def already_floored : ℝ := 250

theorem boxes_needed : 
  (living_room_length * living_room_width - already_floored) / sq_ft_per_box = 7 :=
by 
  sorry

end boxes_needed_l196_196175


namespace line_through_points_l196_196463

theorem line_through_points (a b : ℝ) (h1 : 3 = a * 2 + b) (h2 : 19 = a * 6 + b) :
  a - b = 9 :=
sorry

end line_through_points_l196_196463


namespace arithmetic_sequence_a9_l196_196869

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a 1)
  (h2 : a 2 + a 4 = 2)
  (h5 : a 5 = 3) :
  a 9 = 7 :=
by
  sorry

end arithmetic_sequence_a9_l196_196869


namespace union_complement_set_l196_196841

theorem union_complement_set (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 2, 3, 5}) (hB : B = {2, 4}) :
  (U \ A) ∪ B = {0, 2, 4} :=
by
  rw [Set.diff_eq, hU, hA, hB]
  simp
  sorry

end union_complement_set_l196_196841


namespace dayan_sequence_20th_term_l196_196785

theorem dayan_sequence_20th_term (a : ℕ → ℕ) (h1 : a 0 = 0)
    (h2 : a 1 = 2) (h3 : a 2 = 4) (h4 : a 3 = 8) (h5 : a 4 = 12)
    (h6 : a 5 = 18) (h7 : a 6 = 24) (h8 : a 7 = 32) (h9 : a 8 = 40) (h10 : a 9 = 50)
    (h_even : ∀ n : ℕ, a (2 * n) = 2 * n^2) :
  a 20 = 200 :=
  sorry

end dayan_sequence_20th_term_l196_196785


namespace three_x_y_z_l196_196331

variable (x y z : ℝ)

def equation1 : Prop := y + z = 17 - 2 * x
def equation2 : Prop := x + z = -11 - 2 * y
def equation3 : Prop := x + y = 9 - 2 * z

theorem three_x_y_z : equation1 x y z ∧ equation2 x y z ∧ equation3 x y z → 3 * x + 3 * y + 3 * z = 45 / 4 :=
by
  intros h
  sorry

end three_x_y_z_l196_196331


namespace ratio_Ryn_Nikki_l196_196779

def Joyce_movie_length (M : ℝ) : ℝ := M + 2
def Nikki_movie_length (M : ℝ) : ℝ := 3 * M
def Ryn_movie_fraction (F : ℝ) (Nikki_movie_length : ℝ) : ℝ := F * Nikki_movie_length

theorem ratio_Ryn_Nikki 
  (M : ℝ) 
  (Nikki_movie_is_30 : Nikki_movie_length M = 30) 
  (total_movie_hours_is_76 : M + Joyce_movie_length M + Nikki_movie_length M + Ryn_movie_fraction F (Nikki_movie_length M) = 76) 
  : F = 4 / 5 := 
by 
  sorry

end ratio_Ryn_Nikki_l196_196779


namespace solution_l196_196131

noncomputable def problem (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (∀ x : ℝ, x^2 - 14 * p * x - 15 * q = 0 → x = r ∨ x = s) ∧
  (∀ x : ℝ, x^2 - 14 * r * x - 15 * s = 0 → x = p ∨ x = q)

theorem solution (p q r s : ℝ) (h : problem p q r s) : p + q + r + s = 3150 :=
sorry

end solution_l196_196131


namespace total_problems_l196_196389

theorem total_problems (C W : ℕ) (h1 : 3 * C + 5 * W = 110) (h2 : C = 20) : C + W = 30 :=
by {
  sorry
}

end total_problems_l196_196389


namespace bad_carrots_eq_13_l196_196201

-- Define the number of carrots picked by Haley
def haley_picked : ℕ := 39

-- Define the number of carrots picked by her mom
def mom_picked : ℕ := 38

-- Define the number of good carrots
def good_carrots : ℕ := 64

-- Define the total number of carrots picked
def total_carrots : ℕ := haley_picked + mom_picked

-- State the theorem to prove the number of bad carrots
theorem bad_carrots_eq_13 : total_carrots - good_carrots = 13 := by
  sorry

end bad_carrots_eq_13_l196_196201


namespace unique_solution_nat_numbers_l196_196257

theorem unique_solution_nat_numbers (a b c : ℕ) (h : 2^a + 9^b = 2 * 5^c + 5) : 
  (a, b, c) = (1, 0, 0) :=
sorry

end unique_solution_nat_numbers_l196_196257


namespace opposite_of_neg_2023_l196_196511

theorem opposite_of_neg_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_neg_2023_l196_196511


namespace cakes_difference_l196_196721

theorem cakes_difference (cakes_made : ℕ) (cakes_sold : ℕ) (cakes_bought : ℕ) 
  (h1 : cakes_made = 648) (h2 : cakes_sold = 467) (h3 : cakes_bought = 193) :
  (cakes_sold - cakes_bought = 274) :=
by
  sorry

end cakes_difference_l196_196721


namespace sophia_pages_difference_l196_196112

theorem sophia_pages_difference (total_pages : ℕ) (f_fraction : ℚ) (l_fraction : ℚ) 
  (finished_pages : ℕ) (left_pages : ℕ) :
  f_fraction = 2/3 ∧ 
  l_fraction = 1/3 ∧
  total_pages = 270 ∧
  finished_pages = f_fraction * total_pages ∧
  left_pages = l_fraction * total_pages
  →
  finished_pages - left_pages = 90 :=
by
  intro h
  sorry

end sophia_pages_difference_l196_196112


namespace false_proposition_C_l196_196584

variable (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x^2 + b * x + c

theorem false_proposition_C 
  (ha : a > 0)
  (x0 : ℝ)
  (hx0 : x0 = -b / (2 * a)) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 :=
by
  sorry

end false_proposition_C_l196_196584


namespace find_y_value_l196_196991

theorem find_y_value : (12 ^ 3 * 6 ^ 4) / 432 = 5184 := by
  sorry

end find_y_value_l196_196991


namespace shipCargoCalculation_l196_196967

def initialCargo : Int := 5973
def cargoLoadedInBahamas : Int := 8723
def totalCargo (initial : Int) (loaded : Int) : Int := initial + loaded

theorem shipCargoCalculation : totalCargo initialCargo cargoLoadedInBahamas = 14696 := by
  sorry

end shipCargoCalculation_l196_196967


namespace prime_b_plus_1_l196_196436

def is_a_good (a b : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a * n ≥ b → (Nat.choose (a * n) b - 1) % (a * n + 1) = 0

theorem prime_b_plus_1 (a b : ℕ) (h1 : is_a_good a b) (h2 : ¬ is_a_good a (b + 2)) : Nat.Prime (b + 1) :=
by
  sorry

end prime_b_plus_1_l196_196436


namespace gravitational_field_height_depth_equality_l196_196671

theorem gravitational_field_height_depth_equality
  (R G ρ : ℝ) (hR : R > 0) :
  ∃ x : ℝ, x = R * ((-1 + Real.sqrt 5) / 2) ∧
  (G * ρ * ((4 / 3) * Real.pi * R^3) / (R + x)^2 = G * ρ * ((4 / 3) * Real.pi * (R - x)^3) / (R - x)^2) :=
by
  sorry

end gravitational_field_height_depth_equality_l196_196671


namespace swimming_pool_radius_l196_196768

theorem swimming_pool_radius 
  (r : ℝ)
  (h1 : ∀ (r : ℝ), r > 0)
  (h2 : π * (r + 4)^2 - π * r^2 = (11 / 25) * π * r^2) :
  r = 20 := 
sorry

end swimming_pool_radius_l196_196768


namespace james_total_points_l196_196415

def f : ℕ := 13
def s : ℕ := 20
def p_f : ℕ := 3
def p_s : ℕ := 2

def total_points : ℕ := (f * p_f) + (s * p_s)

theorem james_total_points : total_points = 79 := 
by
  -- Proof would go here.
  sorry

end james_total_points_l196_196415


namespace interior_and_exterior_angles_of_regular_dodecagon_l196_196361

-- Definition of a regular dodecagon
def regular_dodecagon_sides : ℕ := 12

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Measure of one interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Measure of one exterior angle of a regular polygon (180 degrees supplementary to interior angle)
def one_exterior_angle (n : ℕ) : ℕ := 180 - one_interior_angle n

-- The theorem to prove
theorem interior_and_exterior_angles_of_regular_dodecagon :
  one_interior_angle regular_dodecagon_sides = 150 ∧ one_exterior_angle regular_dodecagon_sides = 30 :=
by
  sorry

end interior_and_exterior_angles_of_regular_dodecagon_l196_196361


namespace find_x_l196_196390

def diamond (x y : ℤ) : ℤ := 3 * x - y^2

theorem find_x (x : ℤ) (h : diamond x 7 = 20) : x = 23 :=
sorry

end find_x_l196_196390


namespace product_586645_9999_l196_196291

theorem product_586645_9999 :
  586645 * 9999 = 5865885355 :=
by
  sorry

end product_586645_9999_l196_196291


namespace largest_multiple_of_15_less_than_500_l196_196409

theorem largest_multiple_of_15_less_than_500 : ∃ n : ℕ, (n > 0 ∧ 15 * n < 500) ∧ (∀ m : ℕ, m > n → 15 * m >= 500) ∧ 15 * n = 495 :=
by
  sorry

end largest_multiple_of_15_less_than_500_l196_196409


namespace initial_cargo_l196_196111

theorem initial_cargo (initial_cargo additional_cargo total_cargo : ℕ) 
  (h1 : additional_cargo = 8723) 
  (h2 : total_cargo = 14696) 
  (h3 : initial_cargo + additional_cargo = total_cargo) : 
  initial_cargo = 5973 := 
by 
  -- Start with the assumptions and directly obtain the calculation as required
  sorry

end initial_cargo_l196_196111


namespace geometric_sequence_condition_l196_196088

theorem geometric_sequence_condition (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → (a * d = b * c) ∧ 
  ¬ (∀ a b c d : ℝ, a * d = b * c → ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) := 
by
  sorry

end geometric_sequence_condition_l196_196088


namespace cube_pyramid_same_volume_height_l196_196245

theorem cube_pyramid_same_volume_height (h : ℝ) :
  let cube_edge : ℝ := 5
  let pyramid_base_edge : ℝ := 6
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume : ℝ := (1 / 3) * (pyramid_base_edge ^ 2) * h
  cube_volume = pyramid_volume → h = 125 / 12 :=
by
  intros
  sorry

end cube_pyramid_same_volume_height_l196_196245


namespace john_average_increase_l196_196073

theorem john_average_increase :
  let initial_scores := [92, 85, 91]
  let fourth_score := 95
  let initial_avg := (initial_scores.sum / initial_scores.length : ℚ)
  let new_avg := ((initial_scores.sum + fourth_score) / (initial_scores.length + 1) : ℚ)
  new_avg - initial_avg = 1.42 := 
by 
  sorry

end john_average_increase_l196_196073


namespace rival_awards_eq_24_l196_196161

-- Definitions
def Scott_awards : ℕ := 4
def Jessie_awards (scott: ℕ) : ℕ := 3 * scott
def Rival_awards (jessie: ℕ) : ℕ := 2 * jessie

-- Theorem to prove
theorem rival_awards_eq_24 : Rival_awards (Jessie_awards Scott_awards) = 24 := by
  sorry

end rival_awards_eq_24_l196_196161


namespace cat_food_sufficiency_l196_196462

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l196_196462


namespace solve_equation_l196_196101

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end solve_equation_l196_196101


namespace a_minus_b_greater_than_one_l196_196134

open Real

theorem a_minus_b_greater_than_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (f_has_three_roots : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ (Polynomial.aeval r1 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r2 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r3 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0)
  (g_no_real_roots : ∀ (x : ℝ), (2*x^2 + 2*b*x + a) ≠ 0) :
  a - b > 1 := by
  sorry

end a_minus_b_greater_than_one_l196_196134


namespace correct_result_l196_196318

theorem correct_result (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end correct_result_l196_196318


namespace triangle_side_length_l196_196657

theorem triangle_side_length (y z : ℝ) (cos_Y_minus_Z : ℝ) (h_y : y = 7) (h_z : z = 6) (h_cos : cos_Y_minus_Z = 17 / 18) : 
  ∃ x : ℝ, x = Real.sqrt 65 :=
by
  sorry

end triangle_side_length_l196_196657


namespace cos_seven_pi_over_six_l196_196172

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l196_196172


namespace cubic_equation_roots_l196_196589

theorem cubic_equation_roots :
  (∀ x : ℝ, (x^3 - 7*x^2 + 36 = 0) → (x = -2 ∨ x = 3 ∨ x = 6)) ∧
  ∃ (x1 x2 x3 : ℝ), (x1 * x2 = 18) ∧ (x1 * x2 * x3 = -36) :=
by
  sorry

end cubic_equation_roots_l196_196589


namespace polynomial_identity_l196_196076

theorem polynomial_identity
  (z1 z2 : ℂ)
  (h1 : z1 + z2 = -6)
  (h2 : z1 * z2 = 11)
  : (1 + z1^2 * z2) * (1 + z1 * z2^2) = 1266 := 
by 
  sorry

end polynomial_identity_l196_196076


namespace relationship_between_x_and_y_l196_196910

theorem relationship_between_x_and_y
  (x y : ℝ)
  (h1 : 2 * x - 3 * y > 6 * x)
  (h2 : 3 * x - 4 * y < 2 * y - x) :
  x < y ∧ x < 0 ∧ y < 0 :=
sorry

end relationship_between_x_and_y_l196_196910


namespace knight_count_l196_196218

theorem knight_count (K L : ℕ) (h1 : K + L = 15) 
  (h2 : ∀ k, k < K → (∃ l, l < L ∧ l = 6)) 
  (h3 : ∀ l, l < L → (K > 7)) : K = 9 :=
by 
  sorry

end knight_count_l196_196218


namespace average_mb_per_hour_l196_196491

theorem average_mb_per_hour
  (days : ℕ)
  (original_space  : ℕ)
  (compression_rate : ℝ)
  (total_hours : ℕ := days * 24)
  (effective_space : ℝ := original_space * (1 - compression_rate))
  (space_per_hour : ℝ := effective_space / total_hours) :
  days = 20 ∧ original_space = 25000 ∧ compression_rate = 0.10 → 
  (Int.floor (space_per_hour + 0.5)) = 47 := by
  intros
  sorry

end average_mb_per_hour_l196_196491


namespace union_sets_M_N_l196_196240

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement: the union of M and N should be x > -3
theorem union_sets_M_N : (M ∪ N) = {x | x > -3} :=
sorry

end union_sets_M_N_l196_196240


namespace num_solution_pairs_l196_196204

theorem num_solution_pairs : 
  ∃! (n : ℕ), 
    n = 2 ∧ 
    ∃ x y : ℕ, 
      x > 0 ∧ y >0 ∧ 
      4^x = y^2 + 15 := 
by 
  sorry

end num_solution_pairs_l196_196204


namespace elise_saving_correct_l196_196091

-- Definitions based on the conditions
def initial_money : ℤ := 8
def spent_comic_book : ℤ := 2
def spent_puzzle : ℤ := 18
def final_money : ℤ := 1

-- The theorem to prove the amount saved
theorem elise_saving_correct (x : ℤ) : 
  initial_money + x - spent_comic_book - spent_puzzle = final_money → x = 13 :=
by
  sorry

end elise_saving_correct_l196_196091


namespace function_increasing_on_R_l196_196603

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem function_increasing_on_R (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end function_increasing_on_R_l196_196603


namespace find_cost_price_l196_196567

noncomputable def cost_price (CP SP_loss SP_gain : ℝ) : Prop :=
SP_loss = 0.90 * CP ∧
SP_gain = 1.05 * CP ∧
(SP_gain - SP_loss = 225)

theorem find_cost_price (CP : ℝ) (h : cost_price CP (0.90 * CP) (1.05 * CP)) : CP = 1500 :=
by
  sorry

end find_cost_price_l196_196567


namespace janet_lunch_cost_l196_196809

theorem janet_lunch_cost 
  (num_children : ℕ) (num_chaperones : ℕ) (janet : ℕ) (extra_lunches : ℕ) (cost_per_lunch : ℕ) : 
  num_children = 35 → num_chaperones = 5 → janet = 1 → extra_lunches = 3 → cost_per_lunch = 7 → 
  cost_per_lunch * (num_children + num_chaperones + janet + extra_lunches) = 308 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_lunch_cost_l196_196809


namespace greening_investment_equation_l196_196983

theorem greening_investment_equation:
  ∃ (x : ℝ), 20 * (1 + x)^2 = 25 := 
sorry

end greening_investment_equation_l196_196983


namespace lisa_cleaning_time_l196_196308

theorem lisa_cleaning_time (L : ℝ) (h1 : (1 / L) + (1 / 12) = 1 / 4.8) : L = 8 :=
sorry

end lisa_cleaning_time_l196_196308


namespace tan2α_sin_β_l196_196193

open Real

variables {α β : ℝ}

axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α : sin α = 4 / 5
axiom cos_alpha_beta : cos (α + β) = 5 / 13

theorem tan2α : tan 2 * α = -24 / 7 :=
by sorry

theorem sin_β : sin β = 16 / 65 :=
by sorry

end tan2α_sin_β_l196_196193


namespace prices_and_subsidy_l196_196673

theorem prices_and_subsidy (total_cost : ℕ) (price_leather_jacket : ℕ) (price_sweater : ℕ) (subsidy_percentage : ℕ) 
  (leather_jacket_condition : price_leather_jacket = 5 * price_sweater + 600)
  (cost_condition : price_leather_jacket + price_sweater = total_cost)
  (total_sold : ℕ) (max_subsidy : ℕ) :
  (total_cost = 3000 ∧
   price_leather_jacket = 2600 ∧
   price_sweater = 400 ∧
   subsidy_percentage = 10) ∧ 
  ∃ a : ℕ, (2200 * a ≤ 50000 ∧ total_sold - a ≥ 128) :=
by
  sorry

end prices_and_subsidy_l196_196673


namespace breadth_of_boat_l196_196588

theorem breadth_of_boat
  (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ) (B : ℝ)
  (hL : L = 3)
  (hh : h = 0.01)
  (hm : m = 60)
  (hg : g = 9.81)
  (hρ : ρ = 1000) :
  B = 2 := by
  sorry

end breadth_of_boat_l196_196588


namespace range_k_l196_196857

noncomputable def point (α : Type*) := (α × α)

def M : point ℝ := (0, 2)
def N : point ℝ := (-2, 0)

def line (k : ℝ) (P : point ℝ) := k * P.1 - P.2 - 2 * k + 2 = 0
def angle_condition (M N P : point ℝ) := true -- placeholder for the condition that ∠MPN ≥ π/2

theorem range_k (k : ℝ) (P : point ℝ)
  (hP_on_line : line k P)
  (h_angle_cond : angle_condition M N P) :
  (1 / 7 : ℝ) ≤ k ∧ k ≤ 1 :=
sorry

end range_k_l196_196857


namespace books_read_l196_196881

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end books_read_l196_196881


namespace avg_daily_production_n_l196_196733

theorem avg_daily_production_n (n : ℕ) (h₁ : 50 * n + 110 = 55 * (n + 1)) : n = 11 :=
by
  -- Proof omitted
  sorry

end avg_daily_production_n_l196_196733


namespace predicted_holiday_shoppers_l196_196793

-- Conditions
def packages_per_bulk_box : Nat := 25
def every_third_shopper_buys_package : Nat := 3
def bulk_boxes_ordered : Nat := 5

-- Number of predicted holiday shoppers
theorem predicted_holiday_shoppers (pbb : packages_per_bulk_box = 25)
                                   (etsbp : every_third_shopper_buys_package = 3)
                                   (bbo : bulk_boxes_ordered = 5) :
  (bulk_boxes_ordered * packages_per_bulk_box * every_third_shopper_buys_package) = 375 :=
by 
  -- Proof steps can be added here
  sorry

end predicted_holiday_shoppers_l196_196793


namespace sqrt_of_expression_l196_196464

theorem sqrt_of_expression (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 :=
by
  rw [h]
  simp
  sorry

end sqrt_of_expression_l196_196464


namespace smaller_number_is_270_l196_196784

theorem smaller_number_is_270 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 :=
sorry

end smaller_number_is_270_l196_196784


namespace binomial_coefficient_8_5_l196_196651

theorem binomial_coefficient_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_coefficient_8_5_l196_196651


namespace hourly_wage_increase_l196_196509

variables (W W' H H' : ℝ)

theorem hourly_wage_increase :
  H' = (2/3) * H →
  W * H = W' * H' →
  W' = (3/2) * W :=
by
  intros h_eq income_eq
  rw [h_eq] at income_eq
  sorry

end hourly_wage_increase_l196_196509


namespace prime_between_30_40_with_remainder_l196_196611

theorem prime_between_30_40_with_remainder :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 4 ∧ n = 31 :=
by
  sorry

end prime_between_30_40_with_remainder_l196_196611


namespace hexagon_angle_in_arithmetic_progression_l196_196885

theorem hexagon_angle_in_arithmetic_progression :
  ∃ (a d : ℝ), (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) = 720) ∧ 
  (a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120 ∨ a + 3 * d = 120 ∨ a + 4 * d = 120 ∨ a + 5 * d = 120) := by
  sorry

end hexagon_angle_in_arithmetic_progression_l196_196885


namespace solve_diophantine_l196_196891

theorem solve_diophantine (x y : ℕ) (h1 : 1990 * x - 1989 * y = 1991) : x = 11936 ∧ y = 11941 := by
  have h_pos_x : 0 < x := by sorry
  have h_pos_y : 0 < y := by sorry
  have h_x : 1990 * 11936 = 1990 * x := by sorry
  have h_y : 1989 * 11941 = 1989 * y := by sorry
  sorry

end solve_diophantine_l196_196891


namespace range_of_a_l196_196919

theorem range_of_a (a : ℝ) : (∃ (x : ℤ), x > 1 ∧ x ≤ a) → ∃ (x : ℤ), (x = 2 ∨ x = 3 ∨ x = 4) ∧ 4 ≤ a ∧ a < 5 :=
by
  sorry

end range_of_a_l196_196919


namespace quadratic_to_vertex_form_l196_196670

theorem quadratic_to_vertex_form:
  ∀ (x : ℝ), (x^2 - 4 * x + 3 = (x - 2)^2 - 1) :=
by
  sorry

end quadratic_to_vertex_form_l196_196670


namespace maximum_value_of_function_l196_196899

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - 2 * Real.sin x - 2

theorem maximum_value_of_function :
  ∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1 → f y ≤ 1 :=
by
  sorry

end maximum_value_of_function_l196_196899


namespace blue_marbles_initial_count_l196_196232

variables (x y : ℕ)

theorem blue_marbles_initial_count (h1 : 5 * x = 8 * y) (h2 : 3 * (x - 12) = y + 21) : x = 24 :=
sorry

end blue_marbles_initial_count_l196_196232


namespace inequality_proof_l196_196972

-- Definitions for the conditions
variable (x y : ℝ)

-- Conditions
def conditions : Prop := 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Problem statement to be proven
theorem inequality_proof (h : conditions x y) : 
  x^3 + x * y^2 + 2 * x * y ≤ 2 * x^2 * y + x^2 + x + y := 
by 
  sorry

end inequality_proof_l196_196972


namespace example_function_not_power_function_l196_196472

-- Definition of a power function
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the function y = 2x^(1/2)
def example_function (x : ℝ) : ℝ :=
  2 * x ^ (1 / 2)

-- The statement we want to prove
theorem example_function_not_power_function : ¬ is_power_function example_function := by
  sorry

end example_function_not_power_function_l196_196472


namespace f_prime_at_1_l196_196477

def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

theorem f_prime_at_1 : (deriv f 1) = 11 :=
by
  sorry

end f_prime_at_1_l196_196477


namespace composite_for_infinitely_many_n_l196_196284

theorem composite_for_infinitely_many_n :
  ∃ᶠ n in at_top, (n > 0) ∧ (n % 6 = 4) → ∃ p, p ≠ 1 ∧ p ≠ n^n + (n+1)^(n+1) :=
sorry

end composite_for_infinitely_many_n_l196_196284


namespace abs_neg_one_tenth_l196_196078

theorem abs_neg_one_tenth : |(-1 : ℚ) / 10| = 1 / 10 :=
by
  sorry

end abs_neg_one_tenth_l196_196078


namespace good_walker_catches_up_l196_196844

-- Definitions based on the conditions in the problem
def steps_good_walker := 100
def steps_bad_walker := 60
def initial_lead := 100

-- Mathematical proof problem statement
theorem good_walker_catches_up :
  ∃ x : ℕ, x = initial_lead + (steps_bad_walker * x / steps_good_walker) :=
sorry

end good_walker_catches_up_l196_196844


namespace two_discounts_l196_196146

theorem two_discounts (p : ℝ) : (0.9 * 0.9 * p) = 0.81 * p :=
by
  sorry

end two_discounts_l196_196146


namespace multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l196_196259

variable (a b c : ℕ)

-- Define the conditions as hypotheses
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k, n = 3 * k
def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k
def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

-- Hypotheses
axiom ha : is_multiple_of_3 a
axiom hb : is_multiple_of_12 b
axiom hc : is_multiple_of_9 c

-- Statements to be proved
theorem multiple_of_3_b : is_multiple_of_3 b := sorry
theorem multiple_of_3_a_minus_b : is_multiple_of_3 (a - b) := sorry
theorem multiple_of_3_a_minus_c : is_multiple_of_3 (a - c) := sorry
theorem multiple_of_3_c_minus_b : is_multiple_of_3 (c - b) := sorry

end multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l196_196259


namespace value_of_nested_f_l196_196715

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_nested_f : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end value_of_nested_f_l196_196715


namespace range_of_n_l196_196315

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.exp 1) * Real.exp x + (1 / 2) * x^2 - x

theorem range_of_n :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) ↔ (n ≤ -1/2 ∨ 1 ≤ n) :=
sorry

end range_of_n_l196_196315


namespace binary_arith_proof_l196_196196

theorem binary_arith_proof :
  let a := 0b1101110  -- binary representation of 1101110_2
  let b := 0b101010   -- binary representation of 101010_2
  let c := 0b100      -- binary representation of 100_2
  (a * b / c) = 0b11001000010 :=  -- binary representation of the final result
by
  sorry

end binary_arith_proof_l196_196196


namespace S6_equals_63_l196_196717

variable {S : ℕ → ℕ}

-- Define conditions
axiom S_n_geometric_sequence (a : ℕ → ℕ) (n : ℕ) : n ≥ 1 → S n = (a 0) * ((a 1)^(n) -1) / (a 1 - 1)
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- State theorem
theorem S6_equals_63 : S 6 = 63 := by
  sorry

end S6_equals_63_l196_196717


namespace train_crosses_post_in_25_2_seconds_l196_196366

noncomputable def train_crossing_time (speed_kmph : ℝ) (length_m : ℝ) : ℝ :=
  length_m / (speed_kmph * 1000 / 3600)

theorem train_crosses_post_in_25_2_seconds :
  train_crossing_time 40 280.0224 = 25.2 :=
by 
  sorry

end train_crosses_post_in_25_2_seconds_l196_196366


namespace parabola_points_l196_196631

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_points :
  ∃ (a c m n : ℝ),
  a = 2 ∧ c = -2 ∧
  parabola a 1 c 2 = m ∧
  parabola a 1 c n = -2 ∧
  m = 8 ∧
  n = -1 / 2 :=
by
  use 2, -2, 8, -1/2
  simp [parabola]
  sorry

end parabola_points_l196_196631


namespace increase_in_output_with_assistant_l196_196544

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l196_196544


namespace original_painting_width_l196_196343

theorem original_painting_width {W : ℝ} 
  (orig_height : ℝ) (print_height : ℝ) (print_width : ℝ)
  (h1 : orig_height = 10) 
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  W = 15 :=
  sorry

end original_painting_width_l196_196343


namespace number_of_multiples_143_l196_196323

theorem number_of_multiples_143
  (h1 : 143 = 11 * 13)
  (h2 : ∀ i j : ℕ, 10^j - 10^i = 10^i * (10^(j-i) - 1))
  (h3 : ∀ i : ℕ, gcd (10^i) 143 = 1)
  (h4 : ∀ k : ℕ, 143 ∣ 10^k - 1 ↔ k % 6 = 0)
  (h5 : ∀ i j : ℕ, 0 ≤ i ∧ i < j ∧ j ≤ 99)
  : ∃ n : ℕ, n = 784 :=
by
  sorry

end number_of_multiples_143_l196_196323


namespace increasing_on_neg_reals_l196_196228

variable (f : ℝ → ℝ)

def even_function : Prop := ∀ x : ℝ, f (-x) = f x

def decreasing_on_pos_reals : Prop := ∀ x1 x2 : ℝ, (0 < x1 ∧ 0 < x2 ∧ x1 < x2) → f x1 > f x2

theorem increasing_on_neg_reals
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on_pos_reals f) :
  ∀ x1 x2 : ℝ, (x1 < 0 ∧ x2 < 0 ∧ x1 < x2) → f x1 < f x2 :=
by sorry

end increasing_on_neg_reals_l196_196228


namespace latoya_initial_payment_l196_196545

variable (cost_per_minute : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) 
variable (initial_credit : ℝ)

theorem latoya_initial_payment : 
  ∀ (cost_per_minute call_duration remaining_credit initial_credit : ℝ),
  cost_per_minute = 0.16 →
  call_duration = 22 →
  remaining_credit = 26.48 →
  initial_credit = (cost_per_minute * call_duration) + remaining_credit →
  initial_credit = 30 :=
by
  intros cost_per_minute call_duration remaining_credit initial_credit
  sorry

end latoya_initial_payment_l196_196545


namespace new_babysitter_rate_l196_196045

theorem new_babysitter_rate (x : ℝ) :
  (6 * 16) - 18 = 6 * x + 3 * 2 → x = 12 :=
by
  intros h
  sorry

end new_babysitter_rate_l196_196045


namespace max_not_divisible_by_3_l196_196858

theorem max_not_divisible_by_3 (s : Finset ℕ) (h₁ : s.card = 7) (h₂ : ∃ p ∈ s, p % 3 = 0) : 
  ∃t : Finset ℕ, t.card = 6 ∧ (∀ x ∈ t, x % 3 ≠ 0) ∧ (t ⊆ s) :=
sorry

end max_not_divisible_by_3_l196_196858


namespace solution_set_empty_l196_196889

-- Define the quadratic polynomial
def quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem that the solution set of the given inequality is empty
theorem solution_set_empty : ∀ x : ℝ, quadratic x < 0 → false :=
by
  intro x
  unfold quadratic
  sorry

end solution_set_empty_l196_196889


namespace find_angles_l196_196152

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

theorem find_angles (a : ℝ) (h1 : 0 < a) (h2 : a < 360)
  (h3 : is_arithmetic_sequence (Real.sin a) (Real.sin (2 * a)) (Real.sin (3 * a))) :
  a = 90 ∨ a = 270 := by
  sorry

end find_angles_l196_196152


namespace swimming_speed_in_still_water_l196_196001

-- Given conditions
def water_speed : ℝ := 4
def swim_time_against_current : ℝ := 2
def swim_distance_against_current : ℝ := 8

-- What we are trying to prove
theorem swimming_speed_in_still_water (v : ℝ) 
    (h1 : swim_distance_against_current = 8) 
    (h2 : swim_time_against_current = 2)
    (h3 : water_speed = 4) :
    v - water_speed = swim_distance_against_current / swim_time_against_current → v = 8 :=
by
  sorry

end swimming_speed_in_still_water_l196_196001


namespace katie_pink_marbles_l196_196761

-- Define variables for the problem
variables (P O R : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  O = P - 9 ∧
  R = 4 * (P - 9) ∧
  P + O + R = 33

-- Desired result
def result : Prop :=
  P = 13

-- Proof statement
theorem katie_pink_marbles : conditions P O R → result P :=
by
  intros h
  sorry

end katie_pink_marbles_l196_196761


namespace decrement_from_observation_l196_196369

theorem decrement_from_observation 
  (n : ℕ) (mean_original mean_updated : ℚ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 194)
  : (mean_original - mean_updated) = 6 :=
by
  sorry

end decrement_from_observation_l196_196369


namespace harry_water_per_mile_l196_196943

noncomputable def water_per_mile_during_first_3_miles (initial_water : ℝ) (remaining_water : ℝ) (leak_rate : ℝ) (hike_time : ℝ) (water_drunk_last_mile : ℝ) (first_3_miles : ℝ) : ℝ :=
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  water_drunk_first_3_miles / first_3_miles

theorem harry_water_per_mile :
  water_per_mile_during_first_3_miles 10 2 1 2 3 3 = 1 / 3 :=
by
  have initial_water := 10
  have remaining_water := 2
  have leak_rate := 1
  have hike_time := 2
  have water_drunk_last_mile := 3
  have first_3_miles := 3
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  let result := water_drunk_first_3_miles / first_3_miles
  exact sorry

end harry_water_per_mile_l196_196943


namespace solution_set_of_floor_eqn_l196_196860

theorem solution_set_of_floor_eqn:
  ∀ x y : ℝ, 
  (⌊x⌋ * ⌊x⌋ + ⌊y⌋ * ⌊y⌋ = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by
  sorry

end solution_set_of_floor_eqn_l196_196860


namespace popsicle_sticks_ratio_l196_196655

/-- Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sid has twice as many popsicle sticks as Steve. If Steve has 12 popsicle sticks and they can use 108 popsicle sticks for their Art class activity, prove that the ratio of the number of popsicle sticks Sam has to the number Sid has is 3:1. -/
theorem popsicle_sticks_ratio (Sid Sam Steve : ℕ) 
    (h1 : Sid = 2 * Steve) 
    (h2 : Steve = 12) 
    (h3 : Sam + Sid + Steve = 108) : 
    Sam / Sid = 3 :=
by 
    -- Proof steps go here
    sorry

end popsicle_sticks_ratio_l196_196655


namespace largest_fraction_l196_196090

theorem largest_fraction (d x : ℕ) 
  (h1: (2 * x / d) + (3 * x / d) + (4 * x / d) = 10 / 11)
  (h2: d = 11 * x) : (4 / 11 : ℚ) = (4 * x / d : ℚ) :=
by
  sorry

end largest_fraction_l196_196090


namespace radius_increase_is_0_31_l196_196293

noncomputable def increase_in_radius (initial_radius : ℝ) (odometer_summer : ℝ) (odometer_winter : ℝ) (miles_to_inches : ℝ) : ℝ :=
  let circumference_summer := 2 * Real.pi * initial_radius
  let distance_per_rotation_summer := circumference_summer / miles_to_inches
  let rotations_summer := odometer_summer / distance_per_rotation_summer
  let rotations_winter := odometer_winter / distance_per_rotation_summer
  let distance_winter := rotations_winter * distance_per_rotation_summer
  let new_radius := (distance_winter * miles_to_inches) / (2 * rotations_winter * Real.pi)
  new_radius - initial_radius

theorem radius_increase_is_0_31 : 
    increase_in_radius 16 530 520 63360 = 0.31 := 
by
    sorry

end radius_increase_is_0_31_l196_196293


namespace regular_hexagon_area_inscribed_in_circle_l196_196304

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l196_196304


namespace find_x_l196_196873

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end find_x_l196_196873


namespace right_drawing_num_triangles_l196_196836

-- Given the conditions:
-- 1. Nine distinct lines in the right drawing
-- 2. Any combination of 3 lines out of these 9 forms a triangle
-- 3. Count of intersections of these lines where exactly three lines intersect

def num_triangles : Nat := 84 -- Calculated via binomial coefficient
def num_intersections : Nat := 61 -- Given or calculated from the problem

-- The target theorem to prove that the number of triangles is equal to 23
theorem right_drawing_num_triangles :
  num_triangles - num_intersections = 23 :=
by
  -- Proof would go here, but we skip it as per the instructions
  sorry

end right_drawing_num_triangles_l196_196836


namespace div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l196_196442

theorem div_4800_by_125 : 4800 / 125 = 38.4 :=
by
  sorry

theorem expr_13_mul_74_add_27_mul_13_sub_13 : 13 * 74 + 27 * 13 - 13 = 1300 :=
by
  sorry

end div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l196_196442


namespace div_relation_l196_196677

variable {a b c : ℚ}

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 2/5) : c / a = 5/6 := by
  sorry

end div_relation_l196_196677


namespace common_difference_unique_l196_196313

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

theorem common_difference_unique {a : ℕ → ℝ}
  (h1 : a 2 = 5)
  (h2 : a 3 + a 5 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) ∧ d = -2 :=
sorry

end common_difference_unique_l196_196313


namespace real_inequality_l196_196979

theorem real_inequality
  (a1 a2 a3 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (S : ℝ)
  (hS : S = a1 + a2 + a3)
  (h4 : ∀ i ∈ [a1, a2, a3], (i^2 / (i - 1) > S)) :
  (1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1) := 
by
  sorry

end real_inequality_l196_196979


namespace tangent_lines_through_point_l196_196498

theorem tangent_lines_through_point (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) ∧ (x = -2 ∨ (15*x + 8*y - 10 = 0)) ↔ 
  (x = -2 ∨ (15*x + 8*y - 10 = 0)) :=
by
  sorry

end tangent_lines_through_point_l196_196498


namespace lowest_score_85_avg_l196_196812

theorem lowest_score_85_avg (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 = 79) (h2 : a2 = 88) (h3 : a3 = 94) 
  (h4 : a4 = 91) (h5 : 75 ≤ a5) (h6 : 75 ≤ a6) 
  (h7 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 85) : (a5 = 75 ∨ a6 = 75) ∧ (a5 = 75 ∨ a5 > 75) := 
by
  sorry

end lowest_score_85_avg_l196_196812


namespace first_number_lcm_14_20_l196_196937

theorem first_number_lcm_14_20 (x : ℕ) (h : Nat.lcm x (Nat.lcm 14 20) = 140) : x = 1 := sorry

end first_number_lcm_14_20_l196_196937


namespace alternating_sequence_probability_l196_196601

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l196_196601


namespace orange_profit_44_percent_l196_196890

theorem orange_profit_44_percent :
  (∀ CP SP : ℚ, 0.99 * CP = 1 ∧ SP = CP / 16 → 1 / 11 = CP * (1 + 44 / 100)) :=
by
  sorry

end orange_profit_44_percent_l196_196890


namespace largest_possible_product_l196_196127

theorem largest_possible_product : 
  ∃ S1 S2 : Finset ℕ, 
  (S1 ∪ S2 = {1, 3, 4, 6, 7, 8, 9} ∧ S1 ∩ S2 = ∅ ∧ S1.prod id = S2.prod id) ∧ 
  (S1.prod id = 504 ∧ S2.prod id = 504) :=
by
  sorry

end largest_possible_product_l196_196127


namespace remainder_difference_l196_196072

theorem remainder_difference :
  ∃ (d r: ℤ), (1 < d) ∧ (1250 % d = r) ∧ (1890 % d = r) ∧ (2500 % d = r) ∧ (d - r = 10) :=
sorry

end remainder_difference_l196_196072


namespace number_of_sequences_of_length_100_l196_196087

def sequence_count (n : ℕ) : ℕ :=
  3^n - 2^n

theorem number_of_sequences_of_length_100 :
  sequence_count 100 = 3^100 - 2^100 :=
by
  sorry

end number_of_sequences_of_length_100_l196_196087


namespace probability_white_second_given_red_first_l196_196883

theorem probability_white_second_given_red_first :
  let total_balls := 8
  let red_balls := 5
  let white_balls := 3
  let event_A := red_balls
  let event_B_given_A := white_balls

  (event_B_given_A * (total_balls - 1)) / (event_A * total_balls) = 3 / 7 :=
by
  sorry

end probability_white_second_given_red_first_l196_196883


namespace twelfth_equation_l196_196843

theorem twelfth_equation : (14 : ℤ)^2 - (12 : ℤ)^2 = 4 * 13 := by
  sorry

end twelfth_equation_l196_196843


namespace keith_gave_away_p_l196_196312

theorem keith_gave_away_p (k_init : Nat) (m_init : Nat) (final_pears : Nat) (k_gave_away : Nat) (total_init: Nat := k_init + m_init) :
  k_init = 47 →
  m_init = 12 →
  final_pears = 13 →
  k_gave_away = total_init - final_pears →
  k_gave_away = 46 :=
by
  -- Insert proof here (skip using sorry)
  sorry

end keith_gave_away_p_l196_196312


namespace fractional_equation_m_value_l196_196593

theorem fractional_equation_m_value {x m : ℝ} (hx : 0 < x) (h : 3 / (x - 4) = 1 - (x + m) / (4 - x))
: m = -1 := sorry

end fractional_equation_m_value_l196_196593


namespace range_of_m_l196_196396

variable {m x x1 x2 y1 y2 : ℝ}

noncomputable def linear_function (m x : ℝ) : ℝ := (m - 2) * x + (2 + m)

theorem range_of_m (h1 : x1 < x2) (h2 : y1 = linear_function m x1) (h3 : y2 = linear_function m x2) (h4 : y1 > y2) : m < 2 :=
by
  sorry

end range_of_m_l196_196396


namespace multiple_of_large_block_length_l196_196276

-- Define the dimensions and volumes
variables (w d l : ℝ) -- Normal block dimensions
variables (V_normal V_large : ℝ) -- Volumes
variables (m : ℝ) -- Multiple for the length of the large block

-- Volume conditions for normal and large blocks
def normal_volume_condition (w d l : ℝ) (V_normal : ℝ) : Prop :=
  V_normal = w * d * l

def large_volume_condition (w d l m V_large : ℝ) : Prop :=
  V_large = (2 * w) * (2 * d) * (m * l)

-- Given problem conditions
axiom V_normal_eq_3 : normal_volume_condition w d l 3
axiom V_large_eq_36 : large_volume_condition w d l m 36

-- Statement we want to prove
theorem multiple_of_large_block_length : m = 3 :=
by
  -- Proof steps would go here
  sorry

end multiple_of_large_block_length_l196_196276


namespace ellipse_properties_l196_196190

theorem ellipse_properties :
  (∃ a e : ℝ, (∃ b c : ℝ, a^2 = 25 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c = 4 ∧ e = c / a) ∧ a = 5 ∧ e = 4 / 5) :=
sorry

end ellipse_properties_l196_196190


namespace disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l196_196895

variable (p q : Prop)

theorem disjunction_false_implies_neg_p_true (hpq : ¬(p ∨ q)) : ¬p :=
by 
  sorry

theorem neg_p_true_does_not_imply_disjunction_false (hnp : ¬p) : ¬(¬(p ∨ q)) :=
by 
  sorry

end disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l196_196895


namespace solve_inequality_l196_196555

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 2) / (x - 1)

theorem solve_inequality : 
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 2 } :=
by
  sorry

end solve_inequality_l196_196555


namespace remainder_of_large_power_l196_196879

theorem remainder_of_large_power :
  (2^(2^(2^2))) % 500 = 36 :=
sorry

end remainder_of_large_power_l196_196879


namespace coffee_mix_price_l196_196000

theorem coffee_mix_price (
  weight1 price1 weight2 price2 total_weight : ℝ)
  (h1 : weight1 = 9)
  (h2 : price1 = 2.15)
  (h3 : weight2 = 9)
  (h4 : price2 = 2.45)
  (h5 : total_weight = 18)
  :
  (weight1 * price1 + weight2 * price2) / total_weight = 2.30 :=
by
  sorry

end coffee_mix_price_l196_196000


namespace right_triangle_condition_l196_196886

theorem right_triangle_condition (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : a^2 + b^2 = c^2 :=
by sorry

end right_triangle_condition_l196_196886


namespace prob_both_calligraphy_is_correct_prob_one_each_is_correct_l196_196947

section ProbabilityOfVolunteerSelection

variable (C P : ℕ) -- C = number of calligraphy competition winners, P = number of painting competition winners
variable (total_pairs : ℕ := 6 * (6 - 1) / 2) -- Number of ways to choose 2 out of 6 participants, binomial coefficient (6 choose 2)

-- Condition variables
def num_calligraphy_winners : ℕ := 4
def num_painting_winners : ℕ := 2
def num_total_winners : ℕ := num_calligraphy_winners + num_painting_winners

-- Number of pairs of both calligraphy winners
def pairs_both_calligraphy : ℕ := 4 * (4 - 1) / 2
-- Number of pairs of one calligraphy and one painting winner
def pairs_one_each : ℕ := 4 * 2

-- Probability calculations
def prob_both_calligraphy : ℚ := pairs_both_calligraphy / total_pairs
def prob_one_each : ℚ := pairs_one_each / total_pairs

-- Theorem statements to prove the probabilities of selected types of volunteers
theorem prob_both_calligraphy_is_correct : 
  prob_both_calligraphy = 2/5 := sorry

theorem prob_one_each_is_correct : 
  prob_one_each = 8/15 := sorry

end ProbabilityOfVolunteerSelection

end prob_both_calligraphy_is_correct_prob_one_each_is_correct_l196_196947


namespace log_expression_in_terms_of_a_l196_196668

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

variable (a : ℝ) (h : a = log3 2)

theorem log_expression_in_terms_of_a : log3 8 - 2 * log3 6 = a - 2 :=
by
  sorry

end log_expression_in_terms_of_a_l196_196668


namespace ratio_of_numbers_l196_196095

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l196_196095


namespace sum_of_children_ages_l196_196310

theorem sum_of_children_ages :
  ∃ E: ℕ, E = 12 ∧ 
  (∃ a b c d e : ℕ, a = E ∧ b = E - 2 ∧ c = E - 4 ∧ d = E - 6 ∧ e = E - 8 ∧ 
   a + b + c + d + e = 40) :=
sorry

end sum_of_children_ages_l196_196310


namespace nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l196_196957

theorem nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3
  (a b : ℤ)
  (h : 9 ∣ (a^2 + a * b + b^2)) :
  3 ∣ a ∧ 3 ∣ b :=
sorry

end nine_divides_a2_plus_ab_plus_b2_then_a_b_multiples_of_3_l196_196957


namespace jordan_final_weight_l196_196986

-- Defining the initial weight and the weight losses over the specified weeks
def initial_weight := 250
def loss_first_4_weeks := 4 * 3
def loss_next_8_weeks := 8 * 2
def total_loss := loss_first_4_weeks + loss_next_8_weeks

-- Theorem stating the final weight of Jordan
theorem jordan_final_weight : initial_weight - total_loss = 222 := by
  -- We skip the proof as requested
  sorry

end jordan_final_weight_l196_196986


namespace g_x_plus_three_l196_196411

variable (x : ℝ)

def g (x : ℝ) : ℝ := x^2 - x

theorem g_x_plus_three : g (x + 3) = x^2 + 5 * x + 6 := by
  sorry

end g_x_plus_three_l196_196411


namespace find_x_when_y_30_l196_196782

variable (x y k : ℝ)

noncomputable def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, x * y = k

theorem find_x_when_y_30
  (h_inv_prop : inversely_proportional x y) 
  (h_known_values : x = 5 ∧ y = 15) :
  ∃ x : ℝ, (∃ y : ℝ, y = 30) ∧ x = 5 / 2 := by
  sorry

end find_x_when_y_30_l196_196782


namespace range_of_a_l196_196384

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - a > 0 ∧ 3 * x - 4 < 5) -> False) ↔ (a ≥ 6) :=
by
  sorry

end range_of_a_l196_196384


namespace find_C_l196_196499

theorem find_C (C : ℤ) (h : 2 * C - 3 = 11) : C = 7 :=
sorry

end find_C_l196_196499


namespace hyperbola_eccentricity_range_l196_196079

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 < (Real.sqrt (a^2 + b^2)) / a) ∧ ((Real.sqrt (a^2 + b^2)) / a < (2 * Real.sqrt 3) / 3) :=
sorry

end hyperbola_eccentricity_range_l196_196079


namespace annual_interest_rate_l196_196709

theorem annual_interest_rate (initial_amount final_amount : ℝ) 
  (h_initial : initial_amount = 90) 
  (h_final : final_amount = 99) : 
  ((final_amount - initial_amount) / initial_amount) * 100 = 10 :=
by {
  sorry
}

end annual_interest_rate_l196_196709


namespace lines_parallel_coeff_l196_196578

theorem lines_parallel_coeff (a : ℝ) :
  (∀ x y: ℝ, a * x + 2 * y = 0 → 3 * x + (a + 1) * y + 1 = 0) ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end lines_parallel_coeff_l196_196578


namespace acute_triangle_angle_measure_acute_triangle_side_range_l196_196332

theorem acute_triangle_angle_measure (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) : B = π / 3 :=
by
  sorry

theorem acute_triangle_side_range (A B C a b c : ℝ) (h_acute : A + B + C = π) (h_acute_A : A < π / 2) (h_acute_B : B < π / 2) (h_acute_C : C < π / 2)
  (triangle_relation : (2 * a - c) / (Real.cos (A + B)) = b / (Real.cos (A + C))) (hB : B = π / 3) (hb : b = 3) :
  3 * Real.sqrt 3 < a + c ∧ a + c ≤ 6 :=
by
  sorry

end acute_triangle_angle_measure_acute_triangle_side_range_l196_196332


namespace jackson_entertainment_cost_l196_196485

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l196_196485


namespace one_third_pow_3_eq_3_pow_nineteen_l196_196734

theorem one_third_pow_3_eq_3_pow_nineteen (y : ℤ) (h : (1 / 3 : ℝ) * (3 ^ 20) = 3 ^ y) : y = 19 :=
by
  sorry

end one_third_pow_3_eq_3_pow_nineteen_l196_196734


namespace cevian_concurrency_l196_196124

-- Definitions for the acute triangle and the angles
structure AcuteTriangle (α β γ : ℝ) :=
  (A B C : ℝ)
  (acute_α : α > 0 ∧ α < π / 2)
  (acute_β : β > 0 ∧ β < π / 2)
  (acute_γ : γ > 0 ∧ γ < π / 2)
  (triangle_sum : α + β + γ = π)

-- Definition for the concurrency of cevians
def cevians_concurrent (α β γ : ℝ) (t : AcuteTriangle α β γ) :=
  ∀ (A₁ B₁ C₁ : ℝ), sorry -- placeholder

-- The main theorem with the proof of concurrency
theorem cevian_concurrency (α β γ : ℝ) (t : AcuteTriangle α β γ) :
  ∃ (A₁ B₁ C₁ : ℝ), cevians_concurrent α β γ t :=
  sorry -- proof to be provided


end cevian_concurrency_l196_196124


namespace max_f_of_polynomial_l196_196413

theorem max_f_of_polynomial (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (h_poly : ∃ p : Polynomial ℝ, ∀ x, f x = Polynomial.eval x p)
    (h1 : f 4 = 16)
    (h2 : f 16 = 512) :
    f 8 ≤ 64 :=
by
  sorry

end max_f_of_polynomial_l196_196413


namespace bijection_if_injective_or_surjective_l196_196162

variables {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y)

theorem bijection_if_injective_or_surjective (hX : Fintype.card X = Fintype.card Y)
  (hf : Function.Injective f ∨ Function.Surjective f) : Function.Bijective f :=
by
  sorry

end bijection_if_injective_or_surjective_l196_196162


namespace tenth_term_of_arithmetic_progression_l196_196474

variable (a d n T_n : ℕ)

def arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_progression :
  arithmetic_progression 8 2 10 = 26 :=
  by
  sorry

end tenth_term_of_arithmetic_progression_l196_196474


namespace sum_of_two_digit_factors_l196_196041

theorem sum_of_two_digit_factors (a b : ℕ) (h : a * b = 5681) (h1 : 10 ≤ a) (h2 : a < 100) (h3 : 10 ≤ b) (h4 : b < 100) : a + b = 154 :=
by
  sorry

end sum_of_two_digit_factors_l196_196041


namespace additional_time_required_l196_196990

-- Definitions based on conditions
def time_to_clean_three_sections : ℕ := 24
def total_sections : ℕ := 27

-- Rate of cleaning
def cleaning_rate_per_section (t : ℕ) (n : ℕ) : ℕ := t / n

-- Total time required to clean all sections
def total_cleaning_time (n : ℕ) (r : ℕ) : ℕ := n * r

-- Additional time required to clean the remaining sections
def additional_cleaning_time (t_total : ℕ) (t_spent : ℕ) : ℕ := t_total - t_spent

-- Theorem statement
theorem additional_time_required 
  (t3 : ℕ) (n : ℕ) (t_spent : ℕ) 
  (h₁ : t3 = time_to_clean_three_sections)
  (h₂ : n = total_sections)
  (h₃ : t_spent = time_to_clean_three_sections)
  : additional_cleaning_time (total_cleaning_time n (cleaning_rate_per_section t3 3)) t_spent = 192 :=
by
  sorry

end additional_time_required_l196_196990


namespace find_f_2011_l196_196950

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry  -- Placeholder, since f is only defined in (0, 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_2011 : f 2011 = -2 :=
by
  -- Use properties of f to reduce and eventually find f(2011)
  sorry

end find_f_2011_l196_196950


namespace maximum_value_P_l196_196267

open Classical

noncomputable def P (a b c d : ℝ) : ℝ := a * b + b * c + c * d + d * a

theorem maximum_value_P : ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 40 → P a b c d ≤ 800 :=
by
  sorry

end maximum_value_P_l196_196267


namespace german_russian_students_l196_196830

open Nat

theorem german_russian_students (G R : ℕ) (G_cap_R : ℕ) 
  (h_total : 1500 = G + R - G_cap_R)
  (hG_lb : 1125 ≤ G) (hG_ub : G ≤ 1275)
  (hR_lb : 375 ≤ R) (hR_ub : R ≤ 525) :
  300 = (max (G_cap_R) - min (G_cap_R)) :=
by
  -- Proof would go here
  sorry

end german_russian_students_l196_196830


namespace total_pens_bought_l196_196694

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 := 
sorry

end total_pens_bought_l196_196694


namespace non_egg_laying_chickens_count_l196_196500

noncomputable def num_chickens : ℕ := 80
noncomputable def roosters : ℕ := num_chickens / 4
noncomputable def hens : ℕ := num_chickens - roosters
noncomputable def egg_laying_hens : ℕ := (3 * hens) / 4
noncomputable def hens_on_vacation : ℕ := (2 * egg_laying_hens) / 10
noncomputable def remaining_hens_after_vacation : ℕ := egg_laying_hens - hens_on_vacation
noncomputable def ill_hens : ℕ := (1 * remaining_hens_after_vacation) / 10
noncomputable def non_egg_laying_chickens : ℕ := roosters + hens_on_vacation + ill_hens

theorem non_egg_laying_chickens_count : non_egg_laying_chickens = 33 := by
  sorry

end non_egg_laying_chickens_count_l196_196500


namespace eval_expression_l196_196632

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l196_196632


namespace remainder_53_pow_10_div_8_l196_196592

theorem remainder_53_pow_10_div_8 : (53^10) % 8 = 1 := 
by sorry

end remainder_53_pow_10_div_8_l196_196592


namespace regular_polygon_sides_l196_196269

theorem regular_polygon_sides (P s : ℕ) (hP : P = 150) (hs : s = 15) :
  P / s = 10 :=
by
  sorry

end regular_polygon_sides_l196_196269


namespace number_subtracted_l196_196059

theorem number_subtracted (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 :=
by
  sorry

end number_subtracted_l196_196059


namespace James_pays_6_dollars_l196_196246

-- Defining the conditions
def packs : ℕ := 4
def stickers_per_pack : ℕ := 30
def cost_per_sticker : ℚ := 0.10
def friend_share : ℚ := 0.5

-- Total number of stickers
def total_stickers : ℕ := packs * stickers_per_pack

-- Total cost calculation
def total_cost : ℚ := total_stickers * cost_per_sticker

-- James' payment calculation
def james_payment : ℚ := total_cost * friend_share

-- Theorem statement to be proven
theorem James_pays_6_dollars : james_payment = 6 := by
  sorry

end James_pays_6_dollars_l196_196246


namespace conditions_for_inequality_l196_196120

theorem conditions_for_inequality (a b : ℝ) :
  (∀ x : ℝ, abs ((x^2 + a * x + b) / (x^2 + 2 * x + 2)) < 1) → 
  (a = 2 ∧ 0 < b ∧ b < 2) :=
sorry

end conditions_for_inequality_l196_196120


namespace sum_first_n_abs_terms_arithmetic_seq_l196_196764

noncomputable def sum_abs_arithmetic_sequence (n : ℕ) (h : n ≥ 3) : ℚ :=
  if n = 1 ∨ n = 2 then (n * (4 + 7 - 3 * n)) / 2
  else (3 * n^2 - 11 * n + 20) / 2

theorem sum_first_n_abs_terms_arithmetic_seq (n : ℕ) (h : n ≥ 3) :
  sum_abs_arithmetic_sequence n h = (3 * n^2) / 2 - (11 * n) / 2 + 10 :=
sorry

end sum_first_n_abs_terms_arithmetic_seq_l196_196764


namespace f_monotonically_decreasing_range_of_a_tangent_intersection_l196_196629

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Part (I)
theorem f_monotonically_decreasing (a : ℝ) (x : ℝ) :
  (a > 0 → 0 < x ∧ x < (2 / 3) * a → f' x a < 0) ∧
  (a = 0 → ¬∃ x, f' x a < 0) ∧
  (a < 0 → (2 / 3) * a < x ∧ x < 0 → f' x a < 0) :=
sorry

-- Part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ abs x - 3 / 4) → (-1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part (III)
theorem tangent_intersection (a : ℝ) :
  (a = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ ∃ t : ℝ, (t - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t - x2^3 - 2 = 3 * x2^2 * (2 - x2)) ∧ 2 ≤ t ∧ t ≤ 10 ∧
  ∀ t', (t' - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t' - x2^3 - 2 = 3 * x2^2 * (2 - x2)) → t' ≤ 10) :=
sorry

end f_monotonically_decreasing_range_of_a_tangent_intersection_l196_196629


namespace positive_difference_of_complementary_ratio_5_1_l196_196261

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l196_196261


namespace small_branches_count_l196_196355

theorem small_branches_count (x : ℕ) (h : x^2 + x + 1 = 91) : x = 9 := 
  sorry

end small_branches_count_l196_196355


namespace bucket_capacities_l196_196888

theorem bucket_capacities (a b c : ℕ) 
  (h1 : a + b + c = 1440) 
  (h2 : a + b / 5 = c) 
  (h3 : b + a / 3 = c) : 
  a = 480 ∧ b = 400 ∧ c = 560 := 
by 
  sorry

end bucket_capacities_l196_196888


namespace lowest_possible_price_l196_196247

theorem lowest_possible_price
  (regular_discount_rate : ℚ)
  (sale_discount_rate : ℚ)
  (manufacturer_price : ℚ)
  (H1 : regular_discount_rate = 0.30)
  (H2 : sale_discount_rate = 0.20)
  (H3 : manufacturer_price = 35) :
  (manufacturer_price * (1 - regular_discount_rate) * (1 - sale_discount_rate)) = 19.60 := by
  sorry

end lowest_possible_price_l196_196247


namespace total_amount_spent_l196_196916

def cost_of_haley_paper : ℝ := 3.75 + (3.75 * 0.5)
def cost_of_sister_paper : ℝ := (4.50 * 2) + (4.50 * 0.5)
def cost_of_haley_pens : ℝ := (1.45 * 5) - ((1.45 * 5) * 0.25)
def cost_of_sister_pens : ℝ := (1.65 * 7) - ((1.65 * 7) * 0.25)

def total_cost_of_supplies : ℝ := cost_of_haley_paper + cost_of_sister_paper + cost_of_haley_pens + cost_of_sister_pens

theorem total_amount_spent : total_cost_of_supplies = 30.975 :=
by
  sorry

end total_amount_spent_l196_196916


namespace find_number_l196_196171

theorem find_number (p q N : ℝ) (h1 : N / p = 8) (h2 : N / q = 18) (h3 : p - q = 0.20833333333333334) : N = 3 :=
sorry

end find_number_l196_196171


namespace lcm_problem_l196_196730

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end lcm_problem_l196_196730


namespace coins_ratio_l196_196192

-- Conditions
def initial_coins : Nat := 125
def gift_coins : Nat := 35
def sold_coins : Nat := 80

-- Total coins after receiving the gift
def total_coins := initial_coins + gift_coins

-- Statement to prove the ratio simplifies to 1:2
theorem coins_ratio : (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end coins_ratio_l196_196192


namespace count_numbers_with_digit_2_from_200_to_499_l196_196892

def count_numbers_with_digit_2 (lower upper : ℕ) : ℕ :=
  let A := 100  -- Numbers of the form 2xx (from 200 to 299)
  let B := 30   -- Numbers of the form x2x (where first digit is 2, 3, or 4, last digit can be any)
  let C := 30   -- Numbers of the form xx2 (similar reasoning as B)
  let A_and_B := 10  -- Numbers of the form 22x
  let A_and_C := 10  -- Numbers of the form 2x2
  let B_and_C := 3   -- Numbers of the form x22
  let A_and_B_and_C := 1  -- The number 222
  A + B + C - A_and_B - A_and_C - B_and_C + A_and_B_and_C

theorem count_numbers_with_digit_2_from_200_to_499 : 
  count_numbers_with_digit_2 200 499 = 138 :=
by
  unfold count_numbers_with_digit_2
  exact rfl

end count_numbers_with_digit_2_from_200_to_499_l196_196892


namespace fraction_divisible_by_1963_l196_196433

theorem fraction_divisible_by_1963 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ,
    13 * 733^n + 1950 * 582^n = 1963 * k ∧
    ∃ m : ℤ,
      333^n - 733^n - 1068^n + 431^n = 1963 * m :=
by
  sorry

end fraction_divisible_by_1963_l196_196433


namespace combined_average_score_l196_196652

theorem combined_average_score (M A : ℝ) (m a : ℝ)
  (hM : M = 78) (hA : A = 85) (h_ratio : m = 2 * a / 3) :
  (78 * (2 * a / 3) + 85 * a) / ((2 * a / 3) + a) = 82 := by
  sorry

end combined_average_score_l196_196652


namespace parabola_vertex_trajectory_eq_l196_196580

noncomputable def parabola_vertex_trajectory : Prop :=
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = 2 * m) ∧ (x = -m^2) ∧ (y - 4 * x - 4 * m * y = 0)

theorem parabola_vertex_trajectory_eq :
  (∀ x y : ℝ, (∃ m : ℝ, y = 2 * m ∧ x = -m^2) → y^2 = -4 * x) :=
by
  sorry

end parabola_vertex_trajectory_eq_l196_196580


namespace seq_100_eq_11_div_12_l196_196742

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else if n ≥ 3 then (2 - seq (n - 1)) / (3 * seq (n - 2) + 1)
  else 0 -- This line handles the case n < 1, but shouldn't ever be used in practice.

theorem seq_100_eq_11_div_12 : seq 100 = 11 / 12 :=
  sorry

end seq_100_eq_11_div_12_l196_196742


namespace arrangement_of_students_l196_196069

theorem arrangement_of_students :
  let total_students := 5
  let total_communities := 2
  (2 ^ total_students - 2) = 30 :=
by
  let total_students := 5
  let total_communities := 2
  sorry

end arrangement_of_students_l196_196069


namespace smallest_n_l196_196063

theorem smallest_n :
  ∃ n : ℕ, n = 10 ∧ (n * (n + 1) > 100 ∧ ∀ m : ℕ, m < n → m * (m + 1) ≤ 100) := by
  sorry

end smallest_n_l196_196063


namespace find_third_number_l196_196856

-- Define the given conditions
def proportion_condition (x y : ℝ) : Prop :=
  (0.75 / x) = (y / 8)

-- The main statement to be proven
theorem find_third_number (x y : ℝ) (hx : x = 1.2) (h_proportion : proportion_condition x y) : y = 5 :=
by
  -- Using the assumptions and the definition provided.
  sorry

end find_third_number_l196_196856


namespace original_number_is_9_l196_196698

theorem original_number_is_9 (x : ℤ) (h : 10 * x = x + 81) : x = 9 :=
sorry

end original_number_is_9_l196_196698


namespace slope_of_line_l196_196295

theorem slope_of_line (x y : ℝ) (h : 6 * x + 7 * y - 3 = 0) : - (6 / 7) = -6 / 7 := 
by
  sorry

end slope_of_line_l196_196295


namespace maximize_a_n_l196_196250

-- Given sequence definition
noncomputable def a_n (n : ℕ) := (n + 2) * (7 / 8) ^ n

-- Prove that n = 5 or n = 6 maximizes the sequence
theorem maximize_a_n : ∃ n, (n = 5 ∨ n = 6) ∧ (∀ k, a_n k ≤ a_n n) :=
by
  sorry

end maximize_a_n_l196_196250


namespace find_sum_of_angles_l196_196398

-- Given conditions
def angleP := 34
def angleQ := 76
def angleR := 28

-- Proposition to prove
theorem find_sum_of_angles (x z : ℝ) (h1 : x + z = 138) : x + z = 138 :=
by
  have angleP := 34
  have angleQ := 76
  have angleR := 28
  exact h1

end find_sum_of_angles_l196_196398


namespace avg_of_first_5_numbers_equal_99_l196_196102

def avg_of_first_5 (S1 : ℕ) : ℕ := S1 / 5

theorem avg_of_first_5_numbers_equal_99
  (avg_9 : ℕ := 104) (avg_last_5 : ℕ := 100) (fifth_num : ℕ := 59)
  (sum_9 := 9 * avg_9) (sum_last_5 := 5 * avg_last_5) :
  avg_of_first_5 (sum_9 - sum_last_5 + fifth_num) = 99 :=
by
  sorry

end avg_of_first_5_numbers_equal_99_l196_196102


namespace problem_statement_l196_196955

variable (a b c d : ℝ)

theorem problem_statement :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ (9 / 16) * (a - b) * (b - c) * (c - d) * (d - a) :=
sorry

end problem_statement_l196_196955


namespace constant_ratio_arithmetic_progressions_l196_196199

theorem constant_ratio_arithmetic_progressions
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d p a1 b1 : ℝ)
  (h_a : ∀ k : ℕ, a (k + 1) = a1 + k * d)
  (h_b : ∀ k : ℕ, b (k + 1) = b1 + k * p)
  (h_pos : ∀ k : ℕ, a (k + 1) > 0 ∧ b (k + 1) > 0)
  (h_int : ∀ k : ℕ, ∃ n : ℤ, (a (k + 1) / b (k + 1)) = n) :
  ∃ r : ℝ, ∀ k : ℕ, (a (k + 1) / b (k + 1)) = r :=
by
  sorry

end constant_ratio_arithmetic_progressions_l196_196199


namespace number_of_x_intercepts_l196_196922

def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 3

theorem number_of_x_intercepts : (∃ y : ℝ, parabola y = 0) ∧ (∃! x : ℝ, parabola x = 0) :=
by
  sorry

end number_of_x_intercepts_l196_196922


namespace sin_max_value_l196_196334

open Real

theorem sin_max_value (a b : ℝ) (h : sin (a + b) = sin a + sin b) :
  sin a ≤ 1 :=
by
  sorry

end sin_max_value_l196_196334


namespace difference_in_roi_l196_196938

theorem difference_in_roi (E_investment : ℝ) (B_investment : ℝ) (E_rate : ℝ) (B_rate : ℝ) (years : ℕ) :
  E_investment = 300 → B_investment = 500 → E_rate = 0.15 → B_rate = 0.10 → years = 2 →
  (B_rate * B_investment * years) - (E_rate * E_investment * years) = 10 :=
by
  intros E_investment_eq B_investment_eq E_rate_eq B_rate_eq years_eq
  sorry

end difference_in_roi_l196_196938


namespace complement_union_l196_196821

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  U \ (A ∪ B) = {4} :=
by
  sorry

end complement_union_l196_196821


namespace initial_guinea_fowls_l196_196854

theorem initial_guinea_fowls (initial_chickens initial_turkeys : ℕ) 
  (initial_guinea_fowls : ℕ) (lost_chickens lost_turkeys lost_guinea_fowls : ℕ) 
  (total_birds_end : ℕ) (days : ℕ)
  (hc : initial_chickens = 300) (ht : initial_turkeys = 200) 
  (lc : lost_chickens = 20) (lt : lost_turkeys = 8) (lg : lost_guinea_fowls = 5) 
  (d : days = 7) (tb : total_birds_end = 349) :
  initial_guinea_fowls = 80 := 
by 
  sorry

end initial_guinea_fowls_l196_196854


namespace arithmetic_sequence_sum_l196_196554

-- Define the arithmetic sequence properties
def seq : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
def first := 81
def last := 99
def common_diff := 2
def n := 10

-- Main theorem statement proving the desired property
theorem arithmetic_sequence_sum :
  2 * (seq.sum) = 1800 := by
  sorry

end arithmetic_sequence_sum_l196_196554


namespace area_of_region_l196_196006

noncomputable def region_area : ℝ :=
  sorry

theorem area_of_region :
  region_area = sorry := 
sorry

end area_of_region_l196_196006


namespace problem1_problem2_l196_196798

-- Definitions of the sets A and B
def set_A (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4
def set_B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

-- Problem 1: If A ∩ B ≠ ∅, find the range of a
theorem problem1 (a : ℝ) : (∃ x : ℝ, set_A x ∧ set_B x a) → a ≤ -1 / 2 ∨ a = 2 :=
sorry

-- Problem 2: If A ∩ B = B, find the value of a
theorem problem2 (a : ℝ) : (∀ x : ℝ, set_B x a → set_A x) → a ≤ -1 / 2 ∨ a ≥ 2 :=
sorry

end problem1_problem2_l196_196798


namespace number_divisible_by_45_and_6_l196_196995

theorem number_divisible_by_45_and_6 (k : ℕ) (h1 : 1 ≤ k) (h2 : ∃ n : ℕ, 190 + 90 * (k - 1) ≤  n ∧ n < 190 + 90 * k) 
: 190 + 90 * 5 = 720 := by
  sorry

end number_divisible_by_45_and_6_l196_196995


namespace cheetah_catches_deer_in_10_minutes_l196_196406

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes_l196_196406


namespace share_of_each_person_l196_196403

theorem share_of_each_person (total_length : ℕ) (h1 : total_length = 12) (h2 : total_length % 2 = 0)
  : total_length / 2 = 6 :=
by
  sorry

end share_of_each_person_l196_196403


namespace ratio_of_increase_to_original_l196_196875

noncomputable def ratio_increase_avg_marks (T : ℝ) : ℝ :=
  let original_avg := T / 40
  let new_total := T + 20
  let new_avg := new_total / 40
  let increase_avg := new_avg - original_avg
  increase_avg / original_avg

theorem ratio_of_increase_to_original (T : ℝ) (hT : T > 0) :
  ratio_increase_avg_marks T = 20 / T :=
by
  unfold ratio_increase_avg_marks
  sorry

end ratio_of_increase_to_original_l196_196875


namespace problem1_problem2_l196_196538

-- define problem 1 as a theorem
theorem problem1: 
  ((-0.4) * (-0.8) * (-1.25) * 2.5 = -1) :=
  sorry

-- define problem 2 as a theorem
theorem problem2: 
  ((- (5:ℚ) / 8) * (3 / 14) * ((-16) / 5) * ((-7) / 6) = -1 / 2) :=
  sorry

end problem1_problem2_l196_196538


namespace ranking_emily_olivia_nicole_l196_196959

noncomputable def Emily_score : ℝ := sorry
noncomputable def Olivia_score : ℝ := sorry
noncomputable def Nicole_score : ℝ := sorry

theorem ranking_emily_olivia_nicole :
  (Emily_score > Olivia_score) ∧ (Emily_score > Nicole_score) → 
  (Emily_score > Olivia_score) ∧ (Olivia_score > Nicole_score) := 
by sorry

end ranking_emily_olivia_nicole_l196_196959


namespace marilyn_total_caps_l196_196183

def marilyn_initial_caps : ℝ := 51.0
def nancy_gives_caps : ℝ := 36.0
def total_caps (initial: ℝ) (given: ℝ) : ℝ := initial + given

theorem marilyn_total_caps : total_caps marilyn_initial_caps nancy_gives_caps = 87.0 :=
by
  sorry

end marilyn_total_caps_l196_196183


namespace loss_percentage_grinder_l196_196374

-- Conditions
def CP_grinder : ℝ := 15000
def CP_mobile : ℝ := 8000
def profit_mobile : ℝ := 0.10
def total_profit : ℝ := 200

-- Theorem to prove the loss percentage on the grinder
theorem loss_percentage_grinder : 
  ( (CP_grinder - (23200 - (CP_mobile * (1 + profit_mobile)))) / CP_grinder ) * 100 = 4 :=
by
  sorry

end loss_percentage_grinder_l196_196374


namespace min_p_plus_q_l196_196197

-- Define the conditions
variables {p q : ℕ}

-- Problem statement in Lean 4
theorem min_p_plus_q (h₁ : p > 0) (h₂ : q > 0) (h₃ : 108 * p = q^3) : p + q = 8 :=
sorry

end min_p_plus_q_l196_196197


namespace integer_root_of_quadratic_eq_l196_196621

theorem integer_root_of_quadratic_eq (m : ℤ) (hm : ∃ x : ℤ, m * x^2 + 2 * (m - 5) * x + (m - 4) = 0) : m = -4 ∨ m = 4 ∨ m = -16 :=
sorry

end integer_root_of_quadratic_eq_l196_196621


namespace train_length_l196_196656

theorem train_length (V L : ℝ) (h₁ : V = L / 18) (h₂ : V = (L + 200) / 30) : L = 300 :=
by
  sorry

end train_length_l196_196656


namespace value_of_m_l196_196833

def f (x : ℚ) : ℚ := 3 * x^3 - 1 / x + 2
def g (x : ℚ) (m : ℚ) : ℚ := 2 * x^3 - 3 * x + m
def h (x : ℚ) : ℚ := x^2

theorem value_of_m : f 3 - g 3 (122 / 3) + h 3 = 5 :=
by
  sorry

end value_of_m_l196_196833


namespace winnie_proof_l196_196339

def winnie_problem : Prop :=
  let initial_count := 2017
  let multiples_of_3 := initial_count / 3
  let multiples_of_6 := initial_count / 6
  let multiples_of_27 := initial_count / 27
  let multiples_to_erase_3 := multiples_of_3
  let multiples_to_reinstate_6 := multiples_of_6
  let multiples_to_erase_27 := multiples_of_27
  let final_count := initial_count - multiples_to_erase_3 + multiples_to_reinstate_6 - multiples_to_erase_27
  initial_count - final_count = 373

theorem winnie_proof : winnie_problem := by
  sorry

end winnie_proof_l196_196339


namespace sum_of_squares_of_real_solutions_l196_196423

theorem sum_of_squares_of_real_solutions (x : ℝ) (h : x ^ 64 = 16 ^ 16) : 
  (x = 2 ∨ x = -2) → (x ^ 2 + (-x) ^ 2) = 8 :=
by
  sorry

end sum_of_squares_of_real_solutions_l196_196423


namespace triangle_sides_external_tangent_l196_196047

theorem triangle_sides_external_tangent (R r : ℝ) (h : R > r) :
  ∃ (AB BC AC : ℝ),
    AB = 2 * Real.sqrt (R * r) ∧
    AC = 2 * r * Real.sqrt (R / (R + r)) ∧
    BC = 2 * R * Real.sqrt (r / (R + r)) :=
by
  sorry

end triangle_sides_external_tangent_l196_196047


namespace number_solution_l196_196109

theorem number_solution (x : ℝ) : (x / 5 + 4 = x / 4 - 4) → x = 160 := by
  intros h
  sorry

end number_solution_l196_196109


namespace tickets_required_l196_196054

theorem tickets_required (cost_ferris_wheel : ℝ) (cost_roller_coaster : ℝ) 
  (discount_multiple_rides : ℝ) (coupon_value : ℝ) 
  (total_cost_with_discounts : ℝ) : 
  cost_ferris_wheel = 2.0 ∧ 
  cost_roller_coaster = 7.0 ∧ 
  discount_multiple_rides = 1.0 ∧ 
  coupon_value = 1.0 → 
  total_cost_with_discounts = 7.0 :=
by
  sorry

end tickets_required_l196_196054


namespace litter_patrol_total_l196_196816

theorem litter_patrol_total (glass_bottles : Nat) (aluminum_cans : Nat) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 :=
by
  sorry

end litter_patrol_total_l196_196816


namespace maximize_profit_l196_196198

def total_orders := 100
def max_days := 160
def time_per_A := 5 / 4 -- days
def time_per_B := 5 / 3 -- days
def profit_per_A := 0.5 -- (10,000 RMB)
def profit_per_B := 0.8 -- (10,000 RMB)

theorem maximize_profit : 
  ∃ (x : ℝ) (y : ℝ), 
    (time_per_A * x + time_per_B * (total_orders - x) ≤ max_days) ∧ 
    (y = -0.3 * x + 80) ∧ 
    (x = 16) ∧ 
    (y = 75.2) :=
by 
  sorry

end maximize_profit_l196_196198


namespace jerry_money_left_after_shopping_l196_196452

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l196_196452


namespace senior_high_sample_count_l196_196864

theorem senior_high_sample_count 
  (total_students : ℕ)
  (junior_high_students : ℕ)
  (senior_high_students : ℕ)
  (total_sampled_students : ℕ)
  (H1 : total_students = 1800)
  (H2 : junior_high_students = 1200)
  (H3 : senior_high_students = 600)
  (H4 : total_sampled_students = 180) :
  (senior_high_students * total_sampled_students / total_students) = 60 := 
sorry

end senior_high_sample_count_l196_196864


namespace three_digit_cubes_divisible_by_eight_l196_196688

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l196_196688


namespace power_modulo_l196_196797

theorem power_modulo (k : ℕ) : 7^32 % 19 = 1 → 7^2050 % 19 = 11 :=
by {
  sorry
}

end power_modulo_l196_196797


namespace find_p_plus_q_l196_196654

noncomputable def f (k p : ℚ) : ℚ := 5 * k^2 - 2 * k + p
noncomputable def g (k q : ℚ) : ℚ := 4 * k^2 + q * k - 6

theorem find_p_plus_q (p q : ℚ) (h : ∀ k : ℚ, f k p * g k q = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) :
  p + q = -3 :=
sorry

end find_p_plus_q_l196_196654


namespace darryl_parts_cost_l196_196665

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end darryl_parts_cost_l196_196665


namespace simplify_expression_l196_196180

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (x-2) ^ 2 - x * (x-1) + (x^3 - 4 * x^2) / x^2 = -2 * x := 
by 
  sorry

end simplify_expression_l196_196180


namespace finishing_order_l196_196732

-- Definitions of conditions
def athletes := ["Grisha", "Sasha", "Lena"]

def overtakes : (String → ℕ) := 
  fun athlete =>
    if athlete = "Grisha" then 10
    else if athlete = "Sasha" then 4
    else if athlete = "Lena" then 6
    else 0

-- All three were never at the same point at the same time
def never_same_point_at_same_time : Prop := True -- Simplified for translation purpose

-- The main theorem stating the finishing order given the provided conditions
theorem finishing_order :
  never_same_point_at_same_time →
  (overtakes "Grisha" = 10) →
  (overtakes "Sasha" = 4) →
  (overtakes "Lena" = 6) →
  athletes = ["Grisha", "Sasha", "Lena"] :=
  by
    intro h1 h2 h3 h4
    exact sorry -- The proof is not required, just ensuring the statement is complete.


end finishing_order_l196_196732


namespace balloons_remaining_intact_l196_196720

def initial_balloons : ℕ := 200
def blown_up_after_half_hour (n : ℕ) : ℕ := n / 5
def remaining_balloons_after_half_hour (n : ℕ) : ℕ := n - blown_up_after_half_hour n

def percentage_of_remaining_balloons_blow_up (remaining : ℕ) : ℕ := remaining * 30 / 100
def remaining_balloons_after_one_hour (remaining : ℕ) : ℕ := remaining - percentage_of_remaining_balloons_blow_up remaining

def durable_balloons (remaining : ℕ) : ℕ := remaining * 10 / 100
def non_durable_balloons (remaining : ℕ) (durable : ℕ) : ℕ := remaining - durable

def twice_non_durable (non_durable : ℕ) : ℕ := non_durable * 2

theorem balloons_remaining_intact : 
  (remaining_balloons_after_half_hour initial_balloons) - 
  (percentage_of_remaining_balloons_blow_up 
    (remaining_balloons_after_half_hour initial_balloons)) - 
  (twice_non_durable 
    (non_durable_balloons 
      (remaining_balloons_after_one_hour 
        (remaining_balloons_after_half_hour initial_balloons)) 
      (durable_balloons 
        (remaining_balloons_after_one_hour 
          (remaining_balloons_after_half_hour initial_balloons))))) = 
  0 := 
by
  sorry

end balloons_remaining_intact_l196_196720


namespace middle_card_four_or_five_l196_196153

def three_cards (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c

theorem middle_card_four_or_five (a b c : ℕ) :
  three_cards a b c → (b = 4 ∨ b = 5) :=
by
  sorry

end middle_card_four_or_five_l196_196153


namespace letters_in_small_envelopes_l196_196590

theorem letters_in_small_envelopes (total_letters : ℕ) (large_envelopes : ℕ) (letters_per_large_envelope : ℕ) (letters_in_small_envelopes : ℕ) :
  total_letters = 80 →
  large_envelopes = 30 →
  letters_per_large_envelope = 2 →
  letters_in_small_envelopes = total_letters - (large_envelopes * letters_per_large_envelope) →
  letters_in_small_envelopes = 20 :=
by
  intros ht hl he hs
  rw [ht, hl, he] at hs
  exact hs

#check letters_in_small_envelopes

end letters_in_small_envelopes_l196_196590


namespace minimum_value_of_a2b_l196_196746

noncomputable def minimum_value (a b : ℝ) := a + 2 * b

theorem minimum_value_of_a2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / (2 * a + b) + 1 / (b + 1) = 1) :
  minimum_value a b = (2 * Real.sqrt 3 + 1) / 2 :=
sorry

end minimum_value_of_a2b_l196_196746


namespace numberOfSolutions_l196_196410

noncomputable def numberOfRealPositiveSolutions(x : ℝ) : Prop := 
  (x^6 + 1) * (x^4 + x^2 + 1) = 6 * x^5

theorem numberOfSolutions : ∃! x : ℝ, numberOfRealPositiveSolutions x := 
by
  sorry

end numberOfSolutions_l196_196410


namespace greatest_of_5_consec_even_numbers_l196_196184

-- Definitions based on the conditions
def avg_of_5_consec_even_numbers (N : ℤ) : ℤ := (N - 4 + N - 2 + N + N + 2 + N + 4) / 5

-- Proof statement
theorem greatest_of_5_consec_even_numbers (N : ℤ) (h : avg_of_5_consec_even_numbers N = 35) : N + 4 = 39 :=
by
  sorry -- proof is omitted

end greatest_of_5_consec_even_numbers_l196_196184


namespace sum_intercepts_of_line_l196_196083

theorem sum_intercepts_of_line (x y : ℝ) (h_eq : y - 6 = -2 * (x - 3)) :
  (∃ x_int : ℝ, (0 - 6 = -2 * (x_int - 3)) ∧ x_int = 6) ∧
  (∃ y_int : ℝ, (y_int - 6 = -2 * (0 - 3)) ∧ y_int = 12) →
  6 + 12 = 18 :=
by sorry

end sum_intercepts_of_line_l196_196083


namespace distance_incenters_ACD_BCD_l196_196042

noncomputable def distance_between_incenters (AC : ℝ) (angle_ABC : ℝ) (angle_BAC : ℝ) : ℝ :=
  -- Use the given conditions to derive the distance value
  -- Skipping the detailed calculations, denoted by "sorry"
  sorry

theorem distance_incenters_ACD_BCD :
  distance_between_incenters 1 (30 : ℝ) (60 : ℝ) = 0.5177 := sorry

end distance_incenters_ACD_BCD_l196_196042


namespace second_factor_of_lcm_l196_196884

theorem second_factor_of_lcm (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (lcm : ℕ) 
  (h1 : hcf = 20) 
  (h2 : A = 280)
  (h3 : factor1 = 13) 
  (h4 : lcm = hcf * factor1 * factor2) 
  (h5 : A = hcf * 14) : 
  factor2 = 14 :=
by 
  sorry

end second_factor_of_lcm_l196_196884


namespace verify_total_bill_l196_196970

def fixed_charge : ℝ := 20
def daytime_rate : ℝ := 0.10
def evening_rate : ℝ := 0.05
def free_evening_minutes : ℕ := 200

def daytime_minutes : ℕ := 200
def evening_minutes : ℕ := 300

noncomputable def total_bill : ℝ :=
  fixed_charge + (daytime_minutes * daytime_rate) +
  ((evening_minutes - free_evening_minutes) * evening_rate)

theorem verify_total_bill : total_bill = 45 := by
  sorry

end verify_total_bill_l196_196970


namespace find_f2_l196_196394

namespace ProofProblem

-- Define the polynomial function f
def f (x a b : ℤ) : ℤ := x^5 + a * x^3 + b * x - 8

-- Conditions given in the problem
axiom f_neg2 : ∃ a b : ℤ, f (-2) a b = 10

-- Define the theorem statement
theorem find_f2 : ∃ a b : ℤ, f 2 a b = -26 :=
by
  sorry

end ProofProblem

end find_f2_l196_196394


namespace fish_weight_l196_196762

theorem fish_weight (W : ℝ) (h : W = 2 + W / 3) : W = 3 :=
by
  sorry

end fish_weight_l196_196762


namespace spellbook_cost_in_gold_l196_196297

-- Define the constants
def num_spellbooks : ℕ := 5
def cost_potion_kit_in_silver : ℕ := 20
def num_potion_kits : ℕ := 3
def cost_owl_in_gold : ℕ := 28
def conversion_rate : ℕ := 9
def total_payment_in_silver : ℕ := 537

-- Define the problem to prove the cost of each spellbook in gold given the conditions
theorem spellbook_cost_in_gold : (total_payment_in_silver 
  - (cost_potion_kit_in_silver * num_potion_kits + cost_owl_in_gold * conversion_rate)) / num_spellbooks / conversion_rate = 5 := 
  by
  sorry

end spellbook_cost_in_gold_l196_196297


namespace tobee_points_l196_196741

theorem tobee_points (T J S : ℕ) (h1 : J = T + 6) (h2 : S = 2 * (T + 3) - 2) (h3 : T + J + S = 26) : T = 4 := 
by
  sorry

end tobee_points_l196_196741


namespace tan_pi_div_four_l196_196439

theorem tan_pi_div_four : Real.tan (π / 4) = 1 := by
  sorry

end tan_pi_div_four_l196_196439


namespace sequence_and_sum_l196_196170

-- Given conditions as definitions
def a₁ : ℕ := 1

def recurrence (a_n a_n1 : ℕ) (n : ℕ) : Prop := (a_n1 = 3 * a_n * (1 + (1 / n : ℝ)))

-- Stating the theorem
theorem sequence_and_sum (a : ℕ → ℕ) (S : ℕ → ℝ) :
  (a 1 = a₁) →
  (∀ n, recurrence (a n) (a (n + 1)) n) →
  (∀ n, a n = n * 3 ^ (n - 1)) ∧
  (∀ n, S n = (2 * n - 1) * 3 ^ n / 4 + 1 / 4) :=
by
  sorry

end sequence_and_sum_l196_196170


namespace mutually_exclusive_events_not_complementary_l196_196207

def event_a (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 1
def event_b (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 2

theorem mutually_exclusive_events_not_complementary :
  (∀ ball box, event_a ball box → ¬ event_b ball box) ∧ 
  (∃ box, ¬((event_a 1 box) ∨ (event_b 1 box))) :=
by
  sorry

end mutually_exclusive_events_not_complementary_l196_196207


namespace trebled_resultant_is_correct_l196_196667

-- Let's define the initial number and the transformations
def initial_number := 17
def doubled (n : ℕ) := n * 2
def added_five (n : ℕ) := n + 5
def trebled (n : ℕ) := n * 3

-- Finally, we state the problem to prove
theorem trebled_resultant_is_correct : 
  trebled (added_five (doubled initial_number)) = 117 :=
by
  -- Here we just print sorry which means the proof is expected but not provided yet.
  sorry

end trebled_resultant_is_correct_l196_196667


namespace line_passes_through_fixed_point_l196_196597

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3 / 2) + 3 * (1 / 6) + q = 0 :=
by 
  sorry

end line_passes_through_fixed_point_l196_196597


namespace binary_quadratic_lines_value_m_l196_196923

theorem binary_quadratic_lines_value_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + 2 * x * y + 8 * y^2 + 14 * y + m = 0) →
  m = 7 :=
sorry

end binary_quadratic_lines_value_m_l196_196923


namespace second_grade_survey_count_l196_196135

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end second_grade_survey_count_l196_196135


namespace solve_for_x_l196_196906

-- Definitions of δ and φ
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- The main proof statement
theorem solve_for_x :
  ∃ x : ℚ, delta (phi x) = 10 ∧ x = -31 / 36 :=
by
  sorry

end solve_for_x_l196_196906


namespace probability_team_A_champions_l196_196290

theorem probability_team_A_champions : 
  let p : ℚ := 1 / 2 
  let prob_team_A_win_next := p
  let prob_team_B_win_next_A_win_after := p * p
  prob_team_A_win_next + prob_team_B_win_next_A_win_after = 3 / 4 :=
by
  sorry

end probability_team_A_champions_l196_196290


namespace smallest_k_l196_196521

def v_seq (v : ℕ → ℝ) : Prop :=
  v 0 = 1/8 ∧ ∀ k, v (k + 1) = 3 * v k - 3 * (v k)^2

noncomputable def limit_M : ℝ := 0.5

theorem smallest_k 
  (v : ℕ → ℝ)
  (hv : v_seq v) :
  ∃ k : ℕ, |v k - limit_M| ≤ 1 / 2 ^ 500 ∧ ∀ n < k, ¬ (|v n - limit_M| ≤ 1 / 2 ^ 500) := 
sorry

end smallest_k_l196_196521


namespace sqrt_problem_l196_196559

theorem sqrt_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : a = (3 * m - 1) ^ 2) 
  (h2 : a = (-2 * m - 2) ^ 2) : 
  a = 64 ∨ a = 64 / 25 := 
sorry

end sqrt_problem_l196_196559


namespace Haley_has_25_necklaces_l196_196212

theorem Haley_has_25_necklaces (J H Q : ℕ) 
  (h1 : H = J + 5) 
  (h2 : Q = J / 2) 
  (h3 : H = Q + 15) : 
  H = 25 := 
sorry

end Haley_has_25_necklaces_l196_196212


namespace ratio_of_playground_area_to_total_landscape_area_l196_196616

theorem ratio_of_playground_area_to_total_landscape_area {B L : ℝ} 
    (h1 : L = 8 * B)
    (h2 : L = 240)
    (h3 : 1200 = (240 * B * L) / (240 * B)) :
    1200 / (240 * B) = 1 / 6 :=
sorry

end ratio_of_playground_area_to_total_landscape_area_l196_196616


namespace largest_element_sum_of_digits_in_E_l196_196806
open BigOperators
open Nat

def E : Set ℕ := { n | ∃ (r₉ r₁₀ r₁₁ : ℕ), 0 < r₉ ∧ r₉ ≤ 9 ∧ 0 < r₁₀ ∧ r₁₀ ≤ 10 ∧ 0 < r₁₁ ∧ r₁₁ ≤ 11 ∧
  r₉ = n % 9 ∧ r₁₀ = n % 10 ∧ r₁₁ = n % 11 ∧
  (r₉ > 1) ∧ (r₁₀ > 1) ∧ (r₁₁ > 1) ∧
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), r₉ = a ∧ r₁₀ = a * b ∧ r₁₁ = a * b * c ∧ b ≠ 1 ∧ c ≠ 1 }

noncomputable def N : ℕ := 
  max (max (74 % 990) (134 % 990)) (526 % 990)

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem largest_element_sum_of_digits_in_E :
  sum_of_digits N = 13 :=
sorry

end largest_element_sum_of_digits_in_E_l196_196806


namespace factorization_correct_l196_196712

noncomputable def factor_polynomial (x : ℝ) : ℝ := 4 * x^3 - 4 * x^2 + x

theorem factorization_correct (x : ℝ) : 
  factor_polynomial x = x * (2 * x - 1)^2 :=
by
  sorry

end factorization_correct_l196_196712


namespace inequality_amgm_l196_196913

theorem inequality_amgm (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) := 
by 
  sorry

end inequality_amgm_l196_196913


namespace range_of_m_l196_196326

theorem range_of_m (m : ℝ) (h1 : m + 3 > 0) (h2 : m - 1 < 0) : -3 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l196_196326


namespace polygon_with_given_angle_sum_l196_196834

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l196_196834


namespace inequality_property_l196_196978

theorem inequality_property (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) : (a / b) > (b / a) := 
sorry

end inequality_property_l196_196978


namespace find_multiplier_l196_196014

/-- Define the number -/
def number : ℝ := -10.0

/-- Define the multiplier m -/
def m : ℝ := 0.4

/-- Given conditions and prove the correct multiplier -/
theorem find_multiplier (number : ℝ) (m : ℝ) 
  (h1 : ∃ m : ℝ, m * number - 8 = -12) 
  (h2 : number = -10.0) : m = 0.4 :=
by
  -- We skip the actual steps and provide the answer using sorry
  sorry

end find_multiplier_l196_196014


namespace compare_powers_l196_196641

theorem compare_powers :
  100^100 > 50^50 * 150^50 := sorry

end compare_powers_l196_196641


namespace find_ab_l196_196565

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 100) : a * b = -3 :=
by
sorry

end find_ab_l196_196565


namespace amount_spent_on_shorts_l196_196707

def amount_spent_on_shirt := 12.14
def amount_spent_on_jacket := 7.43
def total_amount_spent_on_clothes := 33.56

theorem amount_spent_on_shorts : total_amount_spent_on_clothes - amount_spent_on_shirt - amount_spent_on_jacket = 13.99 :=
by
  sorry

end amount_spent_on_shorts_l196_196707


namespace coats_collected_elem_schools_correct_l196_196352

-- Conditions
def total_coats_collected : ℕ := 9437
def coats_collected_high_schools : ℕ := 6922

-- Definition to find coats collected from elementary schools
def coats_collected_elementary_schools : ℕ := total_coats_collected - coats_collected_high_schools

-- Theorem statement
theorem coats_collected_elem_schools_correct : 
  coats_collected_elementary_schools = 2515 := sorry

end coats_collected_elem_schools_correct_l196_196352


namespace sum_of_two_longest_altitudes_l196_196056

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l196_196056


namespace number_of_elements_in_union_l196_196086

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem number_of_elements_in_union : ncard (A ∪ B) = 4 :=
by
  sorry

end number_of_elements_in_union_l196_196086


namespace johns_starting_elevation_l196_196408

variable (horizontal_distance : ℝ) (final_elevation : ℝ) (initial_elevation : ℝ)
variable (vertical_ascent : ℝ)

-- Given conditions
axiom h1 : (vertical_ascent / horizontal_distance) = (1 / 2)
axiom h2 : final_elevation = 1450
axiom h3 : horizontal_distance = 2700

-- Prove that John's starting elevation is 100 feet
theorem johns_starting_elevation : initial_elevation = 100 := by
  sorry

end johns_starting_elevation_l196_196408


namespace holly_pills_per_week_l196_196048

theorem holly_pills_per_week 
  (insulin_pills_per_day : ℕ)
  (blood_pressure_pills_per_day : ℕ)
  (anticonvulsants_per_day : ℕ)
  (H1 : insulin_pills_per_day = 2)
  (H2 : blood_pressure_pills_per_day = 3)
  (H3 : anticonvulsants_per_day = 2 * blood_pressure_pills_per_day) :
  (insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsants_per_day) * 7 = 77 := 
by
  sorry

end holly_pills_per_week_l196_196048


namespace inversely_proportional_x_y_l196_196645

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l196_196645


namespace pigeon_percentage_l196_196674

-- Define the conditions
variables (total_birds : ℕ)
variables (geese swans herons ducks pigeons : ℕ)
variables (h1 : geese = total_birds * 20 / 100)
variables (h2 : swans = total_birds * 30 / 100)
variables (h3 : herons = total_birds * 15 / 100)
variables (h4 : ducks = total_birds * 25 / 100)
variables (h5 : pigeons = total_birds * 10 / 100)

-- Define the target problem
theorem pigeon_percentage (h_total : total_birds = 100) :
  (pigeons * 100 / (total_birds - swans)) = 14 :=
by sorry

end pigeon_percentage_l196_196674


namespace Mike_found_seashells_l196_196203

/-!
# Problem:
Mike found some seashells on the beach, he gave Tom 49 of his seashells.
He has thirteen seashells left. How many seashells did Mike find on the beach?

# Conditions:
1. Mike gave Tom 49 seashells.
2. Mike has 13 seashells left.

# Proof statement:
Prove that Mike found 62 seashells on the beach.
-/

/-- Define the variables and conditions -/
def seashells_given_to_Tom : ℕ := 49
def seashells_left_with_Mike : ℕ := 13

/-- Prove that Mike found 62 seashells on the beach -/
theorem Mike_found_seashells : 
  seashells_given_to_Tom + seashells_left_with_Mike = 62 := 
by
  -- This is where the proof would go
  sorry

end Mike_found_seashells_l196_196203


namespace measure_angle_F_correct_l196_196007

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end measure_angle_F_correct_l196_196007


namespace sin_B_of_arithmetic_sequence_angles_l196_196435

theorem sin_B_of_arithmetic_sequence_angles (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : A + B + C = Real.pi) :
  Real.sin B = Real.sqrt 3 / 2 :=
sorry

end sin_B_of_arithmetic_sequence_angles_l196_196435


namespace Cd_sum_l196_196253

theorem Cd_sum : ∀ (C D : ℝ), 
  (∀ x : ℝ, x ≠ 3 → (C / (x-3) + D * (x+2) = (-2 * x^2 + 8 * x + 28) / (x-3))) → 
  (C + D = 20) :=
by
  intros C D h
  sorry

end Cd_sum_l196_196253


namespace total_bags_l196_196614

theorem total_bags (people : ℕ) (bags_per_person : ℕ) (h_people : people = 4) (h_bags_per_person : bags_per_person = 8) : people * bags_per_person = 32 := by
  sorry

end total_bags_l196_196614


namespace sequence_general_term_l196_196118

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l196_196118


namespace percentage_sophia_ate_l196_196855

theorem percentage_sophia_ate : 
  ∀ (caden zoe noah sophia : ℝ),
    caden = 20 / 100 →
    zoe = caden + (0.5 * caden) →
    noah = zoe + (0.5 * zoe) →
    caden + zoe + noah + sophia = 1 →
    sophia = 5 / 100 :=
by
  intros
  sorry

end percentage_sophia_ate_l196_196855


namespace chess_tournament_participants_and_days_l196_196457

theorem chess_tournament_participants_and_days:
  ∃ n d : ℕ, 
    (n % 2 = 1) ∧
    (n * (n - 1) / 2 = 630) ∧
    (d = 34 / 2) ∧
    (n = 35) ∧
    (d = 17) :=
sorry

end chess_tournament_participants_and_days_l196_196457


namespace algebraic_expression_value_l196_196317

theorem algebraic_expression_value {x : ℝ} (h : x * (x + 2) = 2023) : 2 * (x + 3) * (x - 1) - 2018 = 2022 := 
by 
  sorry

end algebraic_expression_value_l196_196317


namespace simple_interest_rate_l196_196722

theorem simple_interest_rate (P A T : ℝ) (R : ℝ) (hP : P = 750) (hA : A = 900) (hT : T = 5) :
    (A - P) = (P * R * T) / 100 → R = 4 := by
  sorry

end simple_interest_rate_l196_196722


namespace profit_ratio_l196_196866

-- Definitions based on conditions
-- Let A_orig and B_orig represent the original profits of stores A and B
-- after increase and decrease respectively, they become equal

variable (A_orig B_orig : ℝ)
variable (h1 : (1.2 * A_orig) = (0.9 * B_orig))

-- Prove that the original profit of store A was 75% of the profit of store B
theorem profit_ratio (h1 : 1.2 * A_orig = 0.9 * B_orig) : A_orig = 0.75 * B_orig :=
by
  -- Insert proof here
  sorry

end profit_ratio_l196_196866


namespace penelope_saving_days_l196_196292

theorem penelope_saving_days :
  ∀ (daily_savings total_saved : ℕ),
  daily_savings = 24 ∧ total_saved = 8760 →
    total_saved / daily_savings = 365 :=
by
  rintro _ _ ⟨rfl, rfl⟩
  sorry

end penelope_saving_days_l196_196292


namespace find_first_number_l196_196956

theorem find_first_number (N : ℤ) (k m : ℤ) (h1 : N = 170 * k + 10) (h2 : 875 = 170 * m + 25) : N = 860 :=
by
  sorry

end find_first_number_l196_196956


namespace value_of_x_for_zero_expression_l196_196822

theorem value_of_x_for_zero_expression (x : ℝ) (h : (x-5 = 0)) (h2 : (6*x - 12 ≠ 0)) :
  x = 5 :=
by {
  sorry
}

end value_of_x_for_zero_expression_l196_196822


namespace probability_white_first_red_second_l196_196391

theorem probability_white_first_red_second :
  let total_marbles := 10
  let white_marbles := 6
  let red_marbles := 4
  let prob_white_first := white_marbles / total_marbles
  let prob_red_second_given_white_first := red_marbles / (total_marbles - 1)
  let prob_combined := prob_white_first * prob_red_second_given_white_first
  prob_combined = 4 / 15 :=
by
  sorry

end probability_white_first_red_second_l196_196391


namespace polynomial_factorization_l196_196223

noncomputable def polyExpression (a b c : ℕ) : ℕ := a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4

theorem polynomial_factorization (a b c : ℕ) :
  ∃ q : ℕ → ℕ → ℕ → ℕ, q a b c = (a + b + c)^3 - 3 * a * b * c ∧
  polyExpression a b c = (a - b) * (b - c) * (c - a) * q a b c := by
  -- The proof goes here
  sorry

end polynomial_factorization_l196_196223


namespace inequality_ab_ab2_a_l196_196027

theorem inequality_ab_ab2_a (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_ab_ab2_a_l196_196027


namespace second_number_is_180_l196_196928

theorem second_number_is_180 
  (x : ℝ) 
  (first : ℝ := 2 * x) 
  (third : ℝ := (1/3) * first)
  (h : first + x + third = 660) : 
  x = 180 :=
sorry

end second_number_is_180_l196_196928


namespace find_innings_l196_196783

noncomputable def calculate_innings (A : ℕ) (n : ℕ) : Prop :=
  (n * A + 140 = (n + 1) * (A + 8)) ∧ (A + 8 = 28)

theorem find_innings (n : ℕ) (A : ℕ) :
  calculate_innings A n → n = 14 :=
by
  intros h
  -- Here you would prove that h implies n = 14, but we use sorry to skip the proof steps.
  sorry

end find_innings_l196_196783


namespace solve_problem1_solve_problem2_l196_196437

noncomputable def problem1 (m n : ℝ) : Prop :=
  (m + n) ^ 2 - 10 * (m + n) + 25 = (m + n - 5) ^ 2

noncomputable def problem2 (x : ℝ) : Prop :=
  ((x ^ 2 - 6 * x + 8) * (x ^ 2 - 6 * x + 10) + 1) = (x - 3) ^ 4

-- Placeholder for proofs
theorem solve_problem1 (m n : ℝ) : problem1 m n :=
by
  sorry

theorem solve_problem2 (x : ℝ) : problem2 x :=
by
  sorry

end solve_problem1_solve_problem2_l196_196437


namespace range_of_a_l196_196473

theorem range_of_a (a : ℝ) : 
  (M = {x : ℝ | 2 * x + 1 < 3}) → 
  (N = {x : ℝ | x < a}) → 
  (M ∩ N = N) ↔ a ≤ 1 :=
by
  let M := {x : ℝ | 2 * x + 1 < 3}
  let N := {x : ℝ | x < a}
  simp [Set.subset_def]
  sorry

end range_of_a_l196_196473


namespace percentage_received_certificates_l196_196961

theorem percentage_received_certificates (boys girls : ℕ) (pct_boys pct_girls : ℝ) :
    boys = 30 ∧ girls = 20 ∧ pct_boys = 0.1 ∧ pct_girls = 0.2 →
    ((pct_boys * boys + pct_girls * girls) / (boys + girls) * 100) = 14 := by
  sorry

end percentage_received_certificates_l196_196961


namespace inequality_for_positive_real_numbers_l196_196586

theorem inequality_for_positive_real_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := 
by 
  sorry

end inequality_for_positive_real_numbers_l196_196586


namespace frank_money_made_l196_196506

theorem frank_money_made
  (spent_on_blades : ℕ)
  (number_of_games : ℕ)
  (cost_per_game : ℕ)
  (total_cost_games := number_of_games * cost_per_game)
  (total_money_made := spent_on_blades + total_cost_games)
  (H1 : spent_on_blades = 11)
  (H2 : number_of_games = 4)
  (H3 : cost_per_game = 2) :
  total_money_made = 19 :=
by
  sorry

end frank_money_made_l196_196506


namespace geom_seq_product_l196_196189

theorem geom_seq_product {a : ℕ → ℝ} (h_geom : ∀ n, a (n + 1) = a n * r)
 (h_a1 : a 1 = 1 / 2) (h_a5 : a 5 = 8) : a 2 * a 3 * a 4 = 8 := 
sorry

end geom_seq_product_l196_196189


namespace sum_of_x_and_y_l196_196930

-- Definitions of conditions
variables (x y : ℤ)
variable (h1 : x - y = 60)
variable (h2 : x = 37)

-- Statement of the problem to be proven
theorem sum_of_x_and_y : x + y = 14 :=
by
  sorry

end sum_of_x_and_y_l196_196930


namespace find_b_and_c_find_b_with_c_range_of_b_l196_196630

-- Part (Ⅰ)
theorem find_b_and_c (b c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_zeros : f (-1) = 0 ∧ f 1 = 0) : b = 0 ∧ c = -1 := sorry

-- Part (Ⅱ)
theorem find_b_with_c (b : ℝ) (f : ℝ → ℝ)
  (x1 x2 : ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + (b^2 + 2 * b + 3))
  (h_eq : (x1 + 1) * (x2 + 1) = 8) 
  (h_roots : f x1 = 0 ∧ f x2 = 0) : b = -2 := sorry

-- Part (Ⅲ)
theorem range_of_b (b : ℝ) (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f_def : ∀ x, f x = x^2 + 2 * b * x + (-1 - 2 * b))
  (h_f_1 : f 1 = 0)
  (h_g_def : ∀ x, g x = f x + x + b)
  (h_intervals : ∀ x, 
    ((-3 < x) ∧ (x < -2) → g x > 0) ∧
    ((-2 < x) ∧ (x < 0) → g x < 0) ∧
    ((0 < x) ∧ (x < 1) → g x < 0) ∧
    ((1 < x) → g x > 0)) : (1/5) < b ∧ b < (5/7) := sorry

end find_b_and_c_find_b_with_c_range_of_b_l196_196630


namespace g_at_6_l196_196330

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_at_3 : g 3 = 4

theorem g_at_6 : g 6 = 8 :=
by 
  sorry

end g_at_6_l196_196330


namespace one_cow_empties_pond_in_75_days_l196_196525

-- Define the necessary variables and their types
variable (c a b : ℝ) -- c represents daily water inflow from the spring
                      -- a represents the total volume of the pond
                      -- b represents the daily consumption per cow

-- Define the conditions
def condition1 : Prop := a + 3 * c = 3 * 17 * b
def condition2 : Prop := a + 30 * c = 30 * 2 * b

-- Target statement we want to prove
theorem one_cow_empties_pond_in_75_days (h1 : condition1 c a b) (h2 : condition2 c a b) :
  ∃ t : ℝ, t = 75 := 
sorry -- Proof to be provided


end one_cow_empties_pond_in_75_days_l196_196525


namespace alpha_beta_roots_eq_l196_196940

theorem alpha_beta_roots_eq {α β : ℝ} (hα : α^2 - α - 2006 = 0) (hβ : β^2 - β - 2006 = 0) (h_sum : α + β = 1) : 
  α + β^2 = 2007 :=
by
  sorry

end alpha_beta_roots_eq_l196_196940


namespace find_b_l196_196944

theorem find_b (b : ℝ) (x y : ℝ) (h1 : 2 * x^2 + b * x = 12) (h2 : y = x + 5.5) (h3 : y^2 * x + y * x^2 + y * (b * x) = 12) :
  b = -5 :=
sorry

end find_b_l196_196944


namespace city_mileage_per_tankful_l196_196917

theorem city_mileage_per_tankful :
  ∀ (T : ℝ), 
  ∃ (city_miles : ℝ),
    (462 = T * (32 + 12)) ∧
    (city_miles = 32 * T) ∧
    (city_miles = 336) :=
by
  sorry

end city_mileage_per_tankful_l196_196917


namespace vacation_days_l196_196278

theorem vacation_days (total_miles miles_per_day : ℕ) 
  (h1 : total_miles = 1250) (h2 : miles_per_day = 250) :
  total_miles / miles_per_day = 5 := by
  sorry

end vacation_days_l196_196278


namespace wire_length_is_180_l196_196486

def wire_problem (length1 length2 : ℕ) (h1 : length1 = 106) (h2 : length2 = 74) (h3 : length1 = length2 + 32) : Prop :=
  (length1 + length2 = 180)

-- Use the definition as an assumption to write the theorem.
theorem wire_length_is_180 (length1 length2 : ℕ) 
  (h1 : length1 = 106) 
  (h2 : length2 = 74) 
  (h3 : length1 = length2 + 32) : 
  length1 + length2 = 180 :=
by
  rw [h1, h2] at h3
  sorry

end wire_length_is_180_l196_196486


namespace batsman_average_after_17th_inning_l196_196904

theorem batsman_average_after_17th_inning (A : ℝ) :
  (16 * A + 87) / 17 = A + 3 → A + 3 = 39 :=
by
  intro h
  sorry

end batsman_average_after_17th_inning_l196_196904


namespace greatest_possible_integer_radius_l196_196176

theorem greatest_possible_integer_radius :
  ∃ r : ℤ, (50 < (r : ℝ)^2) ∧ ((r : ℝ)^2 < 75) ∧ 
  (∀ s : ℤ, (50 < (s : ℝ)^2) ∧ ((s : ℝ)^2 < 75) → s ≤ r) :=
sorry

end greatest_possible_integer_radius_l196_196176


namespace billy_videos_within_limit_l196_196195

def total_videos_watched_within_time_limit (time_limit : ℕ) (video_time : ℕ) (search_time : ℕ) (break_time : ℕ) (num_trials : ℕ) (videos_per_trial : ℕ) (categories : ℕ) (videos_per_category : ℕ) : ℕ :=
  let total_trial_time := videos_per_trial * video_time + search_time + break_time
  let total_category_time := videos_per_category * video_time
  let full_trial_time := num_trials * total_trial_time
  let full_category_time := categories * total_category_time
  let total_time := full_trial_time + full_category_time
  let non_watching_time := search_time * num_trials + break_time * (num_trials - 1)
  let available_time := time_limit - non_watching_time
  let max_videos := available_time / video_time
  max_videos

theorem billy_videos_within_limit : total_videos_watched_within_time_limit 90 4 3 5 5 15 2 10 = 13 := by
  sorry

end billy_videos_within_limit_l196_196195


namespace hot_dogs_served_today_l196_196208

theorem hot_dogs_served_today : 9 + 2 = 11 :=
by
  sorry

end hot_dogs_served_today_l196_196208


namespace value_by_which_number_is_multiplied_l196_196924

theorem value_by_which_number_is_multiplied (x : ℝ) : (5 / 6) * x = 10 ↔ x = 12 := by
  sorry

end value_by_which_number_is_multiplied_l196_196924


namespace fraction_decomposition_l196_196968

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ -8/3 → (7 * x - 19) / (3 * x^2 + 5 * x - 8) = A / (x - 1) + B / (3 * x + 8)) →
  A = -12 / 11 ∧ B = 113 / 11 :=
by
  sorry

end fraction_decomposition_l196_196968


namespace B_work_days_l196_196453

theorem B_work_days
  (A_work_rate : ℝ) (B_work_rate : ℝ) (A_days_worked : ℝ) (B_days_worked : ℝ)
  (total_work : ℝ) (remaining_work : ℝ) :
  A_work_rate = 1 / 15 →
  B_work_rate = total_work / 18 →
  A_days_worked = 5 →
  remaining_work = total_work - A_work_rate * A_days_worked →
  B_days_worked = 12 →
  remaining_work = B_work_rate * B_days_worked →
  total_work = 1 →
  B_days_worked = 12 →
  B_work_rate = total_work / 18 →
  B_days_alone = total_work / B_work_rate →
  B_days_alone = 18 := 
by
  intro hA_work_rate hB_work_rate hA_days_worked hremaining_work hB_days_worked hremaining_work_eq htotal_work hB_days_worked_again hsry_mul_inv hB_days_we_alone_eq
  sorry

end B_work_days_l196_196453


namespace blue_candies_count_l196_196755

theorem blue_candies_count (total_pieces red_pieces : Nat) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) : total_pieces - red_pieces = 3264 := 
by
  -- Proof will be provided here
  sorry

end blue_candies_count_l196_196755


namespace largest_prime_factor_of_set_l196_196239

def largest_prime_factor (n : ℕ) : ℕ :=
  -- pseudo-code for determining the largest prime factor of n
  sorry

lemma largest_prime_factor_45 : largest_prime_factor 45 = 5 := sorry
lemma largest_prime_factor_65 : largest_prime_factor 65 = 13 := sorry
lemma largest_prime_factor_85 : largest_prime_factor 85 = 17 := sorry
lemma largest_prime_factor_119 : largest_prime_factor 119 = 17 := sorry
lemma largest_prime_factor_143 : largest_prime_factor 143 = 13 := sorry

theorem largest_prime_factor_of_set :
  max (largest_prime_factor 45)
    (max (largest_prime_factor 65)
      (max (largest_prime_factor 85)
        (max (largest_prime_factor 119)
          (largest_prime_factor 143)))) = 17 :=
by
  rw [largest_prime_factor_45,
      largest_prime_factor_65,
      largest_prime_factor_85,
      largest_prime_factor_119,
      largest_prime_factor_143]
  sorry

end largest_prime_factor_of_set_l196_196239


namespace equivalent_eq_l196_196882

variable {x y : ℝ}

theorem equivalent_eq (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by
  sorry

end equivalent_eq_l196_196882


namespace min_employees_wednesday_l196_196583

noncomputable def minWednesdayBirthdays (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) :
  ℕ :=
  40

theorem min_employees_wednesday (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) 
  (h1 : total_employees = 61) 
  (h2 : ∃ lst, diff_birthdays lst ∧ max_birthdays 40 lst) :
  minWednesdayBirthdays total_employees diff_birthdays max_birthdays = 40 := 
sorry

end min_employees_wednesday_l196_196583


namespace part_a_part_b_l196_196921

noncomputable def tsunami_area_center_face (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  180000 * Real.pi + 270000 * Real.sqrt 3

noncomputable def tsunami_area_mid_edge (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7

theorem part_a (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_center_face l v t = 180000 * Real.pi + 270000 * Real.sqrt 3 :=
by
  sorry

theorem part_b (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_mid_edge l v t = 720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7 :=
by
  sorry

end part_a_part_b_l196_196921


namespace terminating_decimals_count_l196_196475

theorem terminating_decimals_count :
  (∀ m : ℤ, 1 ≤ m ∧ m ≤ 999 → ∃ k : ℕ, (m : ℝ) / 1000 = k / (2 ^ 3 * 5 ^ 3)) :=
by
  sorry

end terminating_decimals_count_l196_196475


namespace part1_part2_l196_196976

theorem part1 (a b h3 : ℝ) (C : ℝ) (h : 1 / h3 = 1 / a + 1 / b) : C ≤ 120 :=
sorry

theorem part2 (a b m3 : ℝ) (C : ℝ) (h : 1 / m3 = 1 / a + 1 / b) : C ≥ 120 :=
sorry

end part1_part2_l196_196976


namespace domain_log_function_l196_196868

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end domain_log_function_l196_196868


namespace batsman_average_l196_196931

theorem batsman_average (A : ℕ) (total_runs_before : ℕ) (new_score : ℕ) (increase : ℕ)
  (h1 : total_runs_before = 11 * A)
  (h2 : new_score = 70)
  (h3 : increase = 3)
  (h4 : 11 * A + new_score = 12 * (A + increase)) :
  (A + increase) = 37 :=
by
  -- skipping the proof with sorry
  sorry

end batsman_average_l196_196931


namespace range_of_a_l196_196163

theorem range_of_a (a : ℝ) (h₁ : ∀ x : ℝ, x > 0 → x + 4 / x ≥ a) (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) :
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l196_196163


namespace neg_five_power_zero_simplify_expression_l196_196305

-- Proof statement for the first question.
theorem neg_five_power_zero : (-5 : ℝ)^0 = 1 := 
by sorry

-- Proof statement for the second question.
theorem simplify_expression (a b : ℝ) : ((-2 * a^2)^2) * (3 * a * b^2) = 12 * a^5 * b^2 := 
by sorry

end neg_five_power_zero_simplify_expression_l196_196305


namespace incorrect_average_initially_l196_196613

theorem incorrect_average_initially (S : ℕ) :
  (S + 25) / 10 = 46 ↔ (S + 65) / 10 = 50 := by
  sorry

end incorrect_average_initially_l196_196613


namespace total_trip_time_l196_196426

theorem total_trip_time (driving_time : ℕ) (stuck_time : ℕ) (total_time : ℕ) :
  (stuck_time = 2 * driving_time) → (driving_time = 5) → (total_time = driving_time + stuck_time) → total_time = 15 :=
by
  intros h1 h2 h3
  sorry

end total_trip_time_l196_196426


namespace average_girls_score_l196_196144

open Function

variable (C c D d : ℕ)
variable (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)

-- Conditions
def CedarBoys := avgCedarBoys = 85
def CedarGirls := avgCedarGirls = 80
def CedarCombined := avgCedarCombined = 83
def DeltaBoys := avgDeltaBoys = 76
def DeltaGirls := avgDeltaGirls = 95
def DeltaCombined := avgDeltaCombined = 87
def CombinedBoys := avgCombinedBoys = 73

-- Correct answer
def CombinedGirls (avgCombinedGirls : ℤ) := avgCombinedGirls = 86

-- Final statement
theorem average_girls_score (C c D d : ℕ)
    (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)
    (H1 : CedarBoys avgCedarBoys)
    (H2 : CedarGirls avgCedarGirls)
    (H3 : CedarCombined avgCedarCombined)
    (H4 : DeltaBoys avgDeltaBoys)
    (H5 : DeltaGirls avgDeltaGirls)
    (H6 : DeltaCombined avgDeltaCombined)
    (H7 : CombinedBoys avgCombinedBoys) :
    ∃ avgCombinedGirls, CombinedGirls avgCombinedGirls :=
sorry

end average_girls_score_l196_196144


namespace system1_solution_system2_solution_l196_196182

theorem system1_solution (x y : ℝ) (h1 : 3 * x + y = 4) (h2 : 3 * x + 2 * y = 6) : x = 2 / 3 ∧ y = 2 :=
by
  sorry

theorem system2_solution (x y : ℝ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 5 * y = 11) : x = 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l196_196182


namespace part1_part2_l196_196417

-- Part 1
noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + a

theorem part1 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≤ a) → a ≥ 1 / Real.exp 1 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (x₀ : ℝ) : 
  (∀ x : ℝ, f x₀ a < f x a → x = x₀) → a < 1 / 2 → 2 * a - 1 < f x₀ a ∧ f x₀ a < 0 :=
by
  sorry

end part1_part2_l196_196417


namespace negation_of_proposition_exists_negation_of_proposition_l196_196380

theorem negation_of_proposition : 
  (∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) ↔ ¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0) :=
by
  sorry

theorem exists_negation_of_proposition : 
  (¬(∀ x : ℝ, 2^x - 2*x - 2 ≥ 0)) ↔ ∃ x : ℝ, 2^x - 2*x - 2 < 0 :=
by
  sorry

end negation_of_proposition_exists_negation_of_proposition_l196_196380


namespace weaving_output_first_day_l196_196539

theorem weaving_output_first_day (x : ℝ) :
  (x + 2*x + 4*x + 8*x + 16*x = 5) → x = 5 / 31 :=
by
  intros h
  sorry

end weaving_output_first_day_l196_196539


namespace number_of_people_in_team_l196_196026

def total_distance : ℕ := 150
def distance_per_member : ℕ := 30

theorem number_of_people_in_team :
  (total_distance / distance_per_member) = 5 := by
  sorry

end number_of_people_in_team_l196_196026


namespace preceding_integer_binary_l196_196518

--- The conditions as definitions in Lean 4

def M := 0b101100 -- M is defined as binary '101100' which is decimal 44
def preceding_binary (n : Nat) : Nat := n - 1 -- Define a function to get the preceding integer in binary

--- The proof problem statement in Lean 4
theorem preceding_integer_binary :
  preceding_binary M = 0b101011 :=
by
  sorry

end preceding_integer_binary_l196_196518


namespace irrational_pi_l196_196092

theorem irrational_pi :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (π = a / b)) :=
sorry

end irrational_pi_l196_196092


namespace manuscript_pages_l196_196756

theorem manuscript_pages (P : ℕ)
  (h1 : 30 = 30)
  (h2 : 20 = 20)
  (h3 : 50 = 30 + 20)
  (h4 : 710 = 5 * (P - 50) + 30 * 8 + 20 * 11) :
  P = 100 :=
by
  sorry

end manuscript_pages_l196_196756


namespace maximum_value_of_expression_l196_196031

variable (x y z : ℝ)

theorem maximum_value_of_expression (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) : 
  x + y^3 + z^4 ≤ 1 :=
sorry

end maximum_value_of_expression_l196_196031


namespace max_len_sequence_x_l196_196099

theorem max_len_sequence_x :
  ∃ x : ℕ, 3088 < x ∧ x < 3091 :=
sorry

end max_len_sequence_x_l196_196099


namespace product_cubed_roots_l196_196501

-- Given conditions
def cbrt (x : ℝ) : ℝ := x^(1/3)
def expr : ℝ := cbrt (1 + 27) * cbrt (1 + cbrt 27) * cbrt 9

-- Main statement to prove
theorem product_cubed_roots : expr = cbrt 1008 :=
by sorry

end product_cubed_roots_l196_196501


namespace black_stones_count_l196_196737

theorem black_stones_count (T W B : ℕ) (hT : T = 48) (hW1 : 4 * W = 37 * 2 + 26) (hB : B = T - W) : B = 23 :=
by
  sorry

end black_stones_count_l196_196737


namespace arithmetic_sequence_problem_l196_196535

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3 : a 3 = 5)
  (Sn : ∀ n, S n = n * (2 + (n - 1) * 2) / 2)
  (S_diff : ∀ k, S (k + 2) - S k = 36)
  : ∃ k : ℕ, k = 8 :=
by
  sorry

end arithmetic_sequence_problem_l196_196535


namespace blender_sales_inversely_proportional_l196_196236

theorem blender_sales_inversely_proportional (k : ℝ) (p : ℝ) (c : ℝ) 
  (h1 : p * c = k) (h2 : 10 * 300 = k) : (p * 600 = k) → p = 5 := 
by
  intros
  sorry

end blender_sales_inversely_proportional_l196_196236


namespace inequality_holds_l196_196107

-- Define parameters for the problem
variables (p q x y z : ℝ) (n : ℕ)

-- Define the conditions on x, y, and z
def condition1 : Prop := y = x^n + p*x + q
def condition2 : Prop := z = y^n + p*y + q
def condition3 : Prop := x = z^n + p*z + q

-- Define the statement of the inequality
theorem inequality_holds (h1 : condition1 p q x y n) (h2 : condition2 p q y z n) (h3 : condition3 p q x z n):
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y :=
sorry

end inequality_holds_l196_196107


namespace symmedian_length_l196_196620

theorem symmedian_length (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ AS : ℝ, AS = (b * c^2 / (b^2 + c^2)) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) :=
sorry

end symmedian_length_l196_196620


namespace democrats_and_republicans_seating_l196_196493

theorem democrats_and_republicans_seating : 
  let n := 6
  let factorial := Nat.factorial
  let arrangements := (factorial n) * (factorial n)
  let circular_table := 1
  arrangements * circular_table = 518400 :=
by 
  sorry

end democrats_and_republicans_seating_l196_196493


namespace intersection_complement_equivalence_l196_196703

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equivalence :
  ((U \ M) ∩ N) = {3} := by
  sorry

end intersection_complement_equivalence_l196_196703


namespace find_x_l196_196465

-- Given condition: 144 / x = 14.4 / 0.0144
theorem find_x (x : ℝ) (h : 144 / x = 14.4 / 0.0144) : x = 0.144 := by
  sorry

end find_x_l196_196465


namespace percentage_defective_meters_l196_196861

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (h1 : total_meters = 150) (h2 : defective_meters = 15) : 
  (defective_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
sorry

end percentage_defective_meters_l196_196861


namespace overlap_area_of_sectors_l196_196137

/--
Given two sectors of a circle with radius 10, with centers at points P and R respectively, 
one having a central angle of 45 degrees and the other having a central angle of 90 degrees, 
prove that the area of the shaded region where they overlap is 12.5π.
-/
theorem overlap_area_of_sectors 
  (r : ℝ) (θ₁ θ₂ : ℝ) (A₁ A₂ : ℝ)
  (h₀ : r = 10)
  (h₁ : θ₁ = 45)
  (h₂ : θ₂ = 90)
  (hA₁ : A₁ = (θ₁ / 360) * π * r ^ 2)
  (hA₂ : A₂ = (θ₂ / 360) * π * r ^ 2)
  : A₁ = 12.5 * π := 
sorry

end overlap_area_of_sectors_l196_196137


namespace unique_fraction_increased_by_20_percent_l196_196401

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem unique_fraction_increased_by_20_percent (x y : ℕ) (h1 : relatively_prime x y) (h2 : x > 0) (h3 : y > 0) :
  (∃! (x y : ℕ), relatively_prime x y ∧ (x > 0) ∧ (y > 0) ∧ (x + 2) * y = 6 * (y + 2) * x) :=
sorry

end unique_fraction_increased_by_20_percent_l196_196401


namespace value_of_X_when_S_reaches_15000_l196_196325

def X : Nat → Nat
| 0       => 5
| (n + 1) => X n + 3

def S : Nat → Nat
| 0       => 0
| (n + 1) => S n + X (n + 1)

theorem value_of_X_when_S_reaches_15000 :
  ∃ n, S n ≥ 15000 ∧ X n = 299 := by
  sorry

end value_of_X_when_S_reaches_15000_l196_196325


namespace line_equation_solution_l196_196849

noncomputable def line_equation (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (l P.fst = P.snd) ∧ (∀ (x : ℝ), l x = 4 * x - 2) ∨ (∀ (x : ℝ), x = 1)

theorem line_equation_solution : line_equation (1, 2) (2, 3) (0, -5) :=
sorry

end line_equation_solution_l196_196849


namespace vector_calculation_l196_196393

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)
def vec_result : ℝ × ℝ := (3 * vec_a.fst - 2 * vec_b.fst, 3 * vec_a.snd - 2 * vec_b.snd)
def target_vec : ℝ × ℝ := (1, 5)

theorem vector_calculation :
  vec_result = target_vec :=
sorry

end vector_calculation_l196_196393


namespace tangent_ellipse_hyperbola_l196_196132

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ∧ x^2 - n * (y - 1)^2 = 1) → n = 2 :=
by
  intro h
  sorry

end tangent_ellipse_hyperbola_l196_196132


namespace distance_from_two_eq_three_l196_196643

theorem distance_from_two_eq_three (x : ℝ) (h : |x - 2| = 3) : x = -1 ∨ x = 5 :=
sorry

end distance_from_two_eq_three_l196_196643


namespace select_twins_in_grid_l196_196896

theorem select_twins_in_grid (persons : Fin 8 × Fin 8 → Fin 2) :
  ∃ (selection : Fin 8 × Fin 8 → Bool), 
    (∀ i : Fin 8, ∃ j : Fin 8, selection (i, j) = true) ∧ 
    (∀ j : Fin 8, ∃ i : Fin 8, selection (i, j) = true) :=
sorry

end select_twins_in_grid_l196_196896


namespace percentage_of_passengers_in_first_class_l196_196987

theorem percentage_of_passengers_in_first_class (total_passengers : ℕ) (percentage_female : ℝ) (females_coach : ℕ) 
  (males_perc_first_class : ℝ) (Perc_first_class : ℝ) : 
  total_passengers = 120 → percentage_female = 0.45 → females_coach = 46 → males_perc_first_class = (1/3) → 
  Perc_first_class = 10 := by
  sorry

end percentage_of_passengers_in_first_class_l196_196987


namespace correct_propositions_l196_196581

theorem correct_propositions :
  let proposition1 := (∀ A B C : ℝ, C = (A + B) / 2 → C = (A + B) / 2)
  let proposition2 := (∀ a : ℝ, a - |a| = 0 → a ≥ 0)
  let proposition3 := false
  let proposition4 := (∀ a b : ℝ, |a| = |b| → a = -b)
  let proposition5 := (∀ a : ℝ, -a < 0)
  (cond1 : proposition1 = false) →
  (cond2 : proposition2 = false) →
  (cond3 : proposition3 = false) →
  (cond4 : proposition4 = true) →
  (cond5 : proposition5 = false) →
  1 = 1 :=
by
  intros
  sorry

end correct_propositions_l196_196581


namespace compound_interest_time_l196_196893

theorem compound_interest_time 
  (P : ℝ) (r : ℝ) (A₁ : ℝ) (A₂ : ℝ) (t₁ t₂ : ℕ)
  (h1 : r = 0.10)
  (h2 : A₁ = P * (1 + r) ^ t₁)
  (h3 : A₂ = P * (1 + r) ^ t₂)
  (h4 : A₁ = 2420)
  (h5 : A₂ = 2662)
  (h6 : t₂ = t₁ + 3) :
  t₁ = 3 := 
sorry

end compound_interest_time_l196_196893


namespace find_abc_l196_196368

theorem find_abc (a b c : ℝ) (h1 : a * (b + c) = 198) (h2 : b * (c + a) = 210) (h3 : c * (a + b) = 222) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
  a * b * c = 1069 :=
by
  sorry

end find_abc_l196_196368


namespace tan_105_eq_neg2_sub_sqrt3_l196_196862

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end tan_105_eq_neg2_sub_sqrt3_l196_196862


namespace calculate_expression_l196_196595

theorem calculate_expression :
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by sorry

end calculate_expression_l196_196595


namespace value_of_c_l196_196337

theorem value_of_c (a b c : ℕ) (hab : b = 1) (hd : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_pow : (10 * a + b)^2 = 100 * c + 10 * c + b) (h_gt : 100 * c + 10 * c + b > 300) : 
  c = 4 :=
sorry

end value_of_c_l196_196337


namespace local_minimum_of_function_l196_196272

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem local_minimum_of_function : 
  (∃ a, a = 1 ∧ ∀ ε > 0, f a ≤ f (a + ε) ∧ f a ≤ f (a - ε)) := sorry

end local_minimum_of_function_l196_196272


namespace girls_maple_grove_correct_l196_196799

variables (total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge : ℕ)
variables (girls_maple_grove : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 150 ∧ 
  boys = 82 ∧ 
  girls = 68 ∧ 
  pine_ridge_students = 70 ∧ 
  maple_grove_students = 80 ∧ 
  boys_pine_ridge = 36 ∧ 
  girls_maple_grove = girls - (pine_ridge_students - boys_pine_ridge)

-- Question and Answer translated to a proposition
def proof_problem : Prop :=
  conditions total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove → 
  girls_maple_grove = 34

-- Statement
theorem girls_maple_grove_correct : proof_problem total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove :=
by {
  sorry -- Proof omitted
}

end girls_maple_grove_correct_l196_196799


namespace statement_T_true_for_given_values_l196_196094

/-- Statement T: If the sum of the digits of a whole number m is divisible by 9, 
    then m is divisible by 9.
    The given values to check are 45, 54, 81, 63, and none of these. --/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem statement_T_true_for_given_values :
  ∀ (m : ℕ), (m = 45 ∨ m = 54 ∨ m = 81 ∨ m = 63) →
    (is_divisible_by_9 (sum_of_digits m) → is_divisible_by_9 m) :=
by
  intros m H
  cases H
  case inl H1 => sorry
  case inr H2 =>
    cases H2
    case inl H1 => sorry
    case inr H2 =>
      cases H2
      case inl H1 => sorry
      case inr H2 => sorry

end statement_T_true_for_given_values_l196_196094


namespace triangle_perimeter_l196_196932

-- Define the side lengths
def a : ℕ := 7
def b : ℕ := 10
def c : ℕ := 15

-- Define the perimeter
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Statement of the proof problem
theorem triangle_perimeter : perimeter 7 10 15 = 32 := by
  sorry

end triangle_perimeter_l196_196932


namespace find_large_number_l196_196739

theorem find_large_number (L S : ℤ)
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := 
sorry

end find_large_number_l196_196739


namespace bread_cooling_time_l196_196615

theorem bread_cooling_time 
  (dough_room_temp : ℕ := 60)   -- 1 hour in minutes
  (shape_dough : ℕ := 15)       -- 15 minutes
  (proof_dough : ℕ := 120)      -- 2 hours in minutes
  (bake_bread : ℕ := 30)        -- 30 minutes
  (start_time : ℕ := 2 * 60)    -- 2:00 am in minutes
  (end_time : ℕ := 6 * 60)      -- 6:00 am in minutes
  : (end_time - start_time) - (dough_room_temp + shape_dough + proof_dough + bake_bread) = 15 := 
  by
  sorry

end bread_cooling_time_l196_196615


namespace negation_of_proposition_l196_196181

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≥ 0) ↔ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by sorry

end negation_of_proposition_l196_196181


namespace student_correct_answers_l196_196810

noncomputable def correct_answers : ℕ := 58

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = correct_answers :=
by {
  -- placeholder for actual proof
  sorry
}

end student_correct_answers_l196_196810


namespace pencil_count_l196_196573

def total_pencils (drawer : Nat) (desk_0 : Nat) (add_dan : Nat) (remove_sarah : Nat) : Nat :=
  let desk_1 := desk_0 + add_dan
  let desk_2 := desk_1 - remove_sarah
  drawer + desk_2

theorem pencil_count :
  total_pencils 43 19 16 7 = 71 :=
by
  sorry

end pencil_count_l196_196573


namespace standard_equation_of_ellipse_locus_of_midpoint_M_l196_196039

-- Define the conditions of the ellipse
def isEllipse (a b c : ℝ) : Prop :=
  a = 2 ∧ c = Real.sqrt 3 ∧ b = Real.sqrt (a^2 - c^2)

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the locus of the midpoint M
def locus_midpoint (x y : ℝ) : Prop :=
  x^2 / 4 + 4 * y^2 = 1

theorem standard_equation_of_ellipse :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, ellipse_equation x y) :=
sorry

theorem locus_of_midpoint_M :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, locus_midpoint x y) :=
sorry

end standard_equation_of_ellipse_locus_of_midpoint_M_l196_196039


namespace product_of_bc_l196_196628

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l196_196628


namespace angle_B_is_arcsin_l196_196089

-- Define the triangle and its conditions
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    a = 8 ∧ b = Real.sqrt 3 ∧ 
    (2 * Real.cos (A - B) / 2 ^ 2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)

-- Prove that the measure of ∠B is arcsin(√3 / 10)
theorem angle_B_is_arcsin (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
sorry

end angle_B_is_arcsin_l196_196089


namespace find_perimeter_and_sin2A_of_triangle_l196_196829

theorem find_perimeter_and_sin2A_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3) (h_B : B = Real.pi / 3) (h_area : 6 * Real.sqrt 3 = 6 * Real.sqrt 3)
  (h_S : S_ABC = 6 * Real.sqrt 3) : 
  (a + b + c = 18) ∧ (Real.sin (2 * A) = (39 * Real.sqrt 3) / 98) := 
by 
  -- The proof will be placed here. Assuming a valid proof exists.
  sorry

end find_perimeter_and_sin2A_of_triangle_l196_196829


namespace tom_speed_RB_l196_196254

/-- Let d be the distance between B and C (in miles).
    Let 2d be the distance between R and B (in miles).
    Let v be Tom’s speed driving from R to B (in mph).
    Given conditions:
    1. Tom's speed from B to C = 20 mph.
    2. Total average speed of the whole journey = 36 mph.
    Prove that Tom's speed driving from R to B is 60 mph. -/
theorem tom_speed_RB
  (d : ℝ) (v : ℝ)
  (h1 : 20 ≠ 0)
  (h2 : 36 ≠ 0)
  (avg_speed : 3 * d / (2 * d / v + d / 20) = 36) :
  v = 60 := 
sorry

end tom_speed_RB_l196_196254


namespace find_a2_b2_l196_196867

theorem find_a2_b2 (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 :=
by
  sorry

end find_a2_b2_l196_196867


namespace triangle_bisector_length_l196_196934

theorem triangle_bisector_length (A B C : Type) [MetricSpace A] [MetricSpace B]
  [MetricSpace C] (angle_A angle_C : ℝ) (AC AB : ℝ) 
  (hAC : angle_A = 20) (hAngle_C : angle_C = 40) (hAC_minus_AB : AC - AB = 5) : ∃ BM : ℝ, BM = 5 :=
by
  sorry

end triangle_bisector_length_l196_196934


namespace sequence_value_a1_l196_196461

theorem sequence_value_a1 (a : ℕ → ℝ) 
  (h₁ : ∀ n, a (n + 1) = (1 / 2) * a n) 
  (h₂ : a 4 = 8) : a 1 = 64 :=
sorry

end sequence_value_a1_l196_196461


namespace total_interest_received_l196_196392

def principal_B := 5000
def principal_C := 3000
def rate := 9
def time_B := 2
def time_C := 4
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ := P * R * T / 100

theorem total_interest_received :
  let SI_B := simple_interest principal_B rate time_B
  let SI_C := simple_interest principal_C rate time_C
  SI_B + SI_C = 1980 := 
by
  sorry

end total_interest_received_l196_196392


namespace inequality_of_reals_l196_196769

theorem inequality_of_reals (a b c d : ℝ) : 
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := 
  sorry

end inequality_of_reals_l196_196769


namespace geometric_progression_sixth_term_proof_l196_196708

noncomputable def geometric_progression_sixth_term (b₁ b₅ : ℝ) (q : ℝ) := b₅ * q
noncomputable def find_q (b₁ b₅ : ℝ) := (b₅ / b₁)^(1/4)

theorem geometric_progression_sixth_term_proof (b₁ b₅ : ℝ) (h₁ : b₁ = Real.sqrt 3) (h₅ : b₅ = Real.sqrt 243) : 
  ∃ q : ℝ, (q = Real.sqrt 3 ∨ q = - Real.sqrt 3) ∧ geometric_progression_sixth_term b₁ b₅ q = 27 ∨ geometric_progression_sixth_term b₁ b₅ q = -27 :=
by
  sorry

end geometric_progression_sixth_term_proof_l196_196708


namespace fibonacci_arith_sequence_a_eq_665_l196_196030

theorem fibonacci_arith_sequence_a_eq_665 (F : ℕ → ℕ) (a b c : ℕ) :
  (F 1 = 1) →
  (F 2 = 1) →
  (∀ n, n ≥ 3 → F n = F (n - 1) + F (n - 2)) →
  (a + b + c = 2000) →
  (F a < F b ∧ F b < F c ∧ F b - F a = F c - F b) →
  a = 665 :=
by
  sorry

end fibonacci_arith_sequence_a_eq_665_l196_196030


namespace triangle_area_l196_196354

structure Point where
  x : ℝ
  y : ℝ

def area_triangle (A B C : Point) : ℝ := 
  0.5 * (B.x - A.x) * (C.y - A.y)

theorem triangle_area :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨8, 15⟩
  let C : Point := ⟨8, 0⟩
  area_triangle A B C = 60 :=
by
  sorry

end triangle_area_l196_196354


namespace distinct_remainders_mod_3n_l196_196422

open Nat

theorem distinct_remainders_mod_3n 
  (n : ℕ) 
  (hn_odd : Odd n)
  (ai : ℕ → ℕ)
  (bi : ℕ → ℕ)
  (ai_def : ∀ i, 1 ≤ i ∧ i ≤ n → ai i = 3*i - 2)
  (bi_def : ∀ i, 1 ≤ i ∧ i ≤ n → bi i = 3*i - 3)
  (k : ℕ) 
  (hk : 0 < k ∧ k < n)
  : ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → i ≠ j →
     ∀ ⦃ r s t u v : ℕ ⦄, 
       (r = (ai i + ai (i % n + 1)) % (3*n) ∧ 
        s = (ai i + bi i) % (3*n) ∧ 
        t = (bi i + bi ((i + k) % n + 1)) % (3*n)) →
       r ≠ s ∧ s ≠ t ∧ t ≠ r) := 
sorry

end distinct_remainders_mod_3n_l196_196422


namespace b_completes_work_alone_l196_196145

theorem b_completes_work_alone (A_twice_B : ∀ (B : ℕ), A = 2 * B)
  (together : ℕ := 7) : ∃ (B : ℕ), 21 = 3 * together :=
by
  sorry

end b_completes_work_alone_l196_196145


namespace required_amount_of_water_l196_196681

/-- 
Given:
- A solution of 12 ounces with 60% alcohol,
- A desired final concentration of 40% alcohol,

Prove:
- The required amount of water to add is 6 ounces.
-/
theorem required_amount_of_water 
    (original_volume : ℚ)
    (initial_concentration : ℚ)
    (desired_concentration : ℚ)
    (final_volume : ℚ)
    (amount_of_water : ℚ)
    (h1 : original_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : desired_concentration = 0.4)
    (h4 : final_volume = original_volume + amount_of_water)
    (h5 : amount_of_alcohol = original_volume * initial_concentration)
    (h6 : desired_amount_of_alcohol = final_volume * desired_concentration)
    (h7 : amount_of_alcohol = desired_amount_of_alcohol) : 
  amount_of_water = 6 := 
sorry

end required_amount_of_water_l196_196681


namespace minimum_square_side_length_l196_196353

theorem minimum_square_side_length (s : ℝ) (h1 : s^2 ≥ 625) (h2 : ∃ (t : ℝ), t = s / 2) : s = 25 :=
by
  sorry

end minimum_square_side_length_l196_196353


namespace percentage_support_of_surveyed_population_l196_196235

-- Definitions based on the conditions
def men_percentage_support : ℝ := 0.70
def women_percentage_support : ℝ := 0.75
def men_surveyed : ℕ := 200
def women_surveyed : ℕ := 800

-- Proof statement
theorem percentage_support_of_surveyed_population : 
  ((men_percentage_support * men_surveyed + women_percentage_support * women_surveyed) / 
   (men_surveyed + women_surveyed) * 100) = 74 := 
by
  sorry

end percentage_support_of_surveyed_population_l196_196235


namespace packs_of_string_cheese_l196_196050

theorem packs_of_string_cheese (cost_per_piece: ℕ) (pieces_per_pack: ℕ) (total_cost_dollars: ℕ) 
                                (h1: cost_per_piece = 10) 
                                (h2: pieces_per_pack = 20) 
                                (h3: total_cost_dollars = 6) : 
  (total_cost_dollars * 100) / (cost_per_piece * pieces_per_pack) = 3 := 
by
  -- Insert proof here
  sorry

end packs_of_string_cheese_l196_196050


namespace power_addition_l196_196311

variable {R : Type*} [CommRing R]

theorem power_addition (x : R) (m n : ℕ) (h1 : x^m = 6) (h2 : x^n = 2) : x^(m + n) = 12 :=
by
  sorry

end power_addition_l196_196311


namespace percentage_of_60_l196_196058

theorem percentage_of_60 (x : ℝ) : 
  (0.2 * 40) + (x / 100) * 60 = 23 → x = 25 :=
by
  sorry

end percentage_of_60_l196_196058


namespace find_a20_l196_196706

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 : ℤ}

def isArithmeticSeq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def formsGeomSeq (a1 a3 a4 : ℤ) : Prop :=
  (a3 - a1)^2 = a1 * (a4 - a1)

theorem find_a20 (h1 : isArithmeticSeq a (-2))
                 (h2 : formsGeomSeq a_1 (a_1 + 2*(-2)) (a_1 + 3*(-2)))
                 (ha1 : a_1 = 8) :
  a 20 = -30 :=
by
  sorry

end find_a20_l196_196706


namespace gasoline_price_increase_percent_l196_196037

theorem gasoline_price_increase_percent {P Q : ℝ}
  (h₁ : P > 0)
  (h₂: Q > 0)
  (x : ℝ)
  (condition : P * Q * 1.08 = P * (1 + x/100) * Q * 0.90) :
  x = 20 :=
by {
  sorry
}

end gasoline_price_increase_percent_l196_196037


namespace rectangle_side_lengths_l196_196336

theorem rectangle_side_lengths (x y : ℝ) (h1 : 2 * x + 4 = 10) (h2 : 8 * y - 2 = 10) : x + y = 4.5 := by
  sorry

end rectangle_side_lengths_l196_196336


namespace original_wattage_l196_196383

theorem original_wattage (W : ℝ) (h1 : 143 = 1.30 * W) : W = 110 := 
by
  sorry

end original_wattage_l196_196383


namespace general_formula_l196_196623

def sequence_a (n : ℕ) : ℕ :=
by sorry

def partial_sum (n : ℕ) : ℕ :=
by sorry

axiom base_case : partial_sum 1 = 5

axiom recurrence_relation (n : ℕ) (h : 2 ≤ n) : partial_sum (n - 1) = sequence_a n

theorem general_formula (n : ℕ) : partial_sum n = 5 * 2^(n-1) :=
by
-- Proof will be provided here
sorry

end general_formula_l196_196623


namespace largest_possible_m_value_l196_196234

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_possible_m_value :
  ∃ (m x y : ℕ), is_three_digit m ∧ is_prime x ∧ is_prime y ∧ x ≠ y ∧
  x < 10 ∧ y < 10 ∧ is_prime (10 * x - y) ∧ m = x * y * (10 * x - y) ∧ m = 705 := sorry

end largest_possible_m_value_l196_196234


namespace cube_weight_doubled_side_length_l196_196125

-- Theorem: Prove that the weight of a new cube with sides twice as long as the original cube is 40 pounds, given the conditions.
theorem cube_weight_doubled_side_length (s : ℝ) (h₁ : s > 0) (h₂ : (s^3 : ℝ) > 0) (w : ℝ) (h₃ : w = 5) : 
  8 * w = 40 :=
by
  sorry

end cube_weight_doubled_side_length_l196_196125


namespace c_less_than_a_l196_196174

variable (a b c : ℝ)

-- Conditions definitions
def are_negative : Prop := a < 0 ∧ b < 0 ∧ c < 0
def eq1 : Prop := c = 2 * (a + b)
def eq2 : Prop := c = 3 * (b - a)

-- Theorem statement
theorem c_less_than_a (h_neg : are_negative a b c) (h_eq1 : eq1 a b c) (h_eq2 : eq2 a b c) : c < a :=
  sorry

end c_less_than_a_l196_196174


namespace cos_sum_series_l196_196524

theorem cos_sum_series : 
  (∑' n : ℤ, if (n % 2 = 1 ∨ n % 2 = -1) then (1 : ℝ) / (n : ℝ)^2 else 0) = (π^2) / 8 := by
  sorry

end cos_sum_series_l196_196524


namespace solution_interval_l196_196548

theorem solution_interval (x : ℝ) : 
  (3/8 + |x - 1/4| < 7/8) ↔ (-1/4 < x ∧ x < 3/4) := 
sorry

end solution_interval_l196_196548


namespace larger_of_two_numbers_l196_196329

theorem larger_of_two_numbers (A B : ℕ) (hcf lcm : ℕ) (h1 : hcf = 23)
                              (h2 : lcm = hcf * 14 * 15) 
                              (h3 : lcm = A * B) (h4 : A = 23 * 14) 
                              (h5 : B = 23 * 15) : max A B = 345 :=
    sorry

end larger_of_two_numbers_l196_196329


namespace h_is_decreasing_intervals_l196_196348

noncomputable def f (x : ℝ) := if x >= 1 then x - 2 else 0
noncomputable def g (x : ℝ) := if x <= 2 then -2 * x + 3 else 0

noncomputable def h (x : ℝ) :=
  if x >= 1 ∧ x <= 2 then f x * g x
  else if x >= 1 then f x
  else if x <= 2 then g x
  else 0

theorem h_is_decreasing_intervals :
  (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → h x1 > h x2) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 7 / 4 → x2 ≤ 2 → h x1 ≥ h x2) :=
by
  sorry

end h_is_decreasing_intervals_l196_196348


namespace vicente_meat_purchase_l196_196726

theorem vicente_meat_purchase :
  ∃ (meat_lbs : ℕ),
  (∃ (rice_kgs cost_rice_per_kg cost_meat_per_lb total_spent : ℕ),
    rice_kgs = 5 ∧
    cost_rice_per_kg = 2 ∧
    cost_meat_per_lb = 5 ∧
    total_spent = 25 ∧
    total_spent - (rice_kgs * cost_rice_per_kg) = meat_lbs * cost_meat_per_lb) ∧
  meat_lbs = 3 :=
by {
  sorry
}

end vicente_meat_purchase_l196_196726


namespace geometric_series_sum_example_l196_196503

def geometric_series_sum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r ^ n - 1) / (r - 1))

theorem geometric_series_sum_example : geometric_series_sum 2 (-3) 8 = -3280 :=
by
  sorry

end geometric_series_sum_example_l196_196503


namespace cost_price_of_each_clock_l196_196051

theorem cost_price_of_each_clock
  (C : ℝ)
  (h1 : 40 * C * 1.1 + 50 * C * 1.2 - 90 * C * 1.15 = 40) :
  C = 80 :=
sorry

end cost_price_of_each_clock_l196_196051


namespace fraction_question_l196_196800

theorem fraction_question :
  ((3 / 8 + 5 / 6) / (5 / 12 + 1 / 4) = 29 / 16) :=
by
  -- This is where we will put the proof steps 
  sorry

end fraction_question_l196_196800


namespace quadractic_roots_value_l196_196450

theorem quadractic_roots_value (c d : ℝ) (h₁ : 3*c^2 + 9*c - 21 = 0) (h₂ : 3*d^2 + 9*d - 21 = 0) :
  (3*c - 4) * (6*d - 8) = -22 := by
  sorry

end quadractic_roots_value_l196_196450


namespace sqrt_meaningful_range_l196_196260

theorem sqrt_meaningful_range (x : ℝ) (h : x - 2 ≥ 0) : x ≥ 2 :=
by {
  sorry
}

end sqrt_meaningful_range_l196_196260


namespace fifteen_percent_eq_135_l196_196440

theorem fifteen_percent_eq_135 (x : ℝ) (h : (15 / 100) * x = 135) : x = 900 :=
sorry

end fifteen_percent_eq_135_l196_196440


namespace midpoint_trajectory_l196_196497

theorem midpoint_trajectory (x y : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (8, 0) ∧ (B.1, B.2) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ 
   ∃ P : ℝ × ℝ, P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ P = (x, y)) → (x - 4)^2 + y^2 = 1 :=
by sorry

end midpoint_trajectory_l196_196497


namespace total_savings_over_12_weeks_l196_196128

-- Define the weekly savings and durations for each period
def weekly_savings_period_1 : ℕ := 5
def duration_period_1 : ℕ := 4

def weekly_savings_period_2 : ℕ := 10
def duration_period_2 : ℕ := 4

def weekly_savings_period_3 : ℕ := 20
def duration_period_3 : ℕ := 4

-- Define the total savings calculation for each period
def total_savings_period_1 : ℕ := weekly_savings_period_1 * duration_period_1
def total_savings_period_2 : ℕ := weekly_savings_period_2 * duration_period_2
def total_savings_period_3 : ℕ := weekly_savings_period_3 * duration_period_3

-- Prove that the total savings over 12 weeks equals $140.00
theorem total_savings_over_12_weeks : total_savings_period_1 + total_savings_period_2 + total_savings_period_3 = 140 := 
by 
  sorry

end total_savings_over_12_weeks_l196_196128


namespace intersection_point_of_circle_and_line_l196_196736

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, 2 * Real.sin α)
noncomputable def line_polar (rho θ : ℝ) : Prop := rho * Real.sin θ = 2

theorem intersection_point_of_circle_and_line :
  ∃ (α : ℝ) (rho θ : ℝ), circle_parametric α = (1, 2) ∧ line_polar rho θ := sorry

end intersection_point_of_circle_and_line_l196_196736


namespace elisa_improvement_l196_196962

theorem elisa_improvement (cur_laps cur_minutes prev_laps prev_minutes : ℕ) 
  (h1 : cur_laps = 15) (h2 : cur_minutes = 30) 
  (h3 : prev_laps = 20) (h4 : prev_minutes = 50) : 
  ((prev_minutes / prev_laps : ℚ) - (cur_minutes / cur_laps : ℚ) = 0.5) :=
by
  sorry

end elisa_improvement_l196_196962


namespace vertical_angles_always_equal_l196_196534

theorem vertical_angles_always_equal (a b : ℝ) (h : a = b) : 
  (∀ θ1 θ2, θ1 + θ2 = 180 ∧ θ1 = a ∧ θ2 = b → θ1 = θ2) :=
by 
  intro θ1 θ2 
  intro h 
  sorry

end vertical_angles_always_equal_l196_196534


namespace area_bounded_region_l196_196364

theorem area_bounded_region (x y : ℝ) (h : y^2 + 2*x*y + 30*|x| = 300) : 
  ∃ A, A = 900 := 
sorry

end area_bounded_region_l196_196364


namespace solution_set_of_inequality_l196_196248

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2 * x > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l196_196248


namespace possible_sets_C_l196_196420

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def is_partition (A B C : Set ℕ) : Prop :=
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ A ∪ B ∪ C = M

def conditions (A B C : Set ℕ) : Prop :=
  is_partition A B C ∧ (∃ (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ), 
    A = {a1, a2, a3, a4} ∧
    B = {b1, b2, b3, b4} ∧
    C = {c1, c2, c3, c4} ∧
    c1 < c2 ∧ c2 < c3 ∧ c3 < c4 ∧
    a1 + b1 = c1 ∧ a2 + b2 = c2 ∧ a3 + b3 = c3 ∧ a4 + b4 = c4)

theorem possible_sets_C (A B C : Set ℕ) (h : conditions A B C) :
  C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
sorry

end possible_sets_C_l196_196420


namespace park_area_l196_196719

theorem park_area (l w : ℝ) (h1 : l + w = 40) (h2 : l = 3 * w) : l * w = 300 :=
by
  sorry

end park_area_l196_196719


namespace smallest_four_digit_divisible_by_53_l196_196790

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l196_196790


namespace sum_of_three_consecutive_even_numbers_l196_196561

theorem sum_of_three_consecutive_even_numbers (a : ℤ) (h : a * (a + 2) * (a + 4) = 960) : a + (a + 2) + (a + 4) = 30 := by
  sorry

end sum_of_three_consecutive_even_numbers_l196_196561


namespace union_complement_l196_196626

def universalSet : Set ℤ := { x | x^2 < 9 }

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

def complement_I_B : Set ℤ := universalSet \ B

theorem union_complement :
  A ∪ complement_I_B = {0, 1, 2} :=
by
  sorry

end union_complement_l196_196626


namespace intersection_l196_196416

def setA : Set ℝ := { x : ℝ | x^2 - 2*x - 3 < 0 }
def setB : Set ℝ := { x : ℝ | x > 1 }

theorem intersection (x : ℝ) : x ∈ setA ∧ x ∈ setB ↔ 1 < x ∧ x < 3 := by
  sorry

end intersection_l196_196416


namespace circumference_of_tire_l196_196421

theorem circumference_of_tire (rotations_per_minute : ℕ) (speed_kmh : ℕ) 
  (h1 : rotations_per_minute = 400) (h2 : speed_kmh = 72) :
  let speed_mpm := speed_kmh * 1000 / 60
  let circumference := speed_mpm / rotations_per_minute
  circumference = 3 :=
by
  sorry

end circumference_of_tire_l196_196421


namespace exchanges_count_l196_196557

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end exchanges_count_l196_196557


namespace first_cat_blue_eyed_kittens_l196_196378

variable (B : ℕ)
variable (C1 : 35 * (B + 17) = 100 * (B + 4))

theorem first_cat_blue_eyed_kittens : B = 3 :=
by
  -- proof
  sorry

end first_cat_blue_eyed_kittens_l196_196378


namespace num_ways_to_factor_2210_l196_196803

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_ways_to_factor_2210 : ∃! (a b : ℕ), a * b = 2210 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end num_ways_to_factor_2210_l196_196803


namespace problem1_problem2_l196_196680

noncomputable def arcSin (x : ℝ) : ℝ := Real.arcsin x

theorem problem1 :
  (S : ℝ) = 3 * Real.pi + 2 * Real.sqrt 2 - 6 * arcSin (Real.sqrt (2 / 3)) :=
by
  sorry

theorem problem2 :
  (S : ℝ) = 3 * arcSin (Real.sqrt (2 / 3)) - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l196_196680


namespace sum_of_cubes_of_consecutive_even_integers_l196_196989

theorem sum_of_cubes_of_consecutive_even_integers (x : ℤ) (h : x^2 + (x+2)^2 + (x+4)^2 = 2960) :
  x^3 + (x + 2)^3 + (x + 4)^3 = 90117 :=
sorry

end sum_of_cubes_of_consecutive_even_integers_l196_196989


namespace speed_of_first_car_l196_196108

variable (V1 V2 V3 : ℝ) -- Define the speeds of the three cars
variable (t x : ℝ) -- Time interval and distance from A to B

-- Conditions of the problem
axiom condition_1 : x / V1 = (x / V2) + t
axiom condition_2 : x / V2 = (x / V3) + t
axiom condition_3 : 120 / V1  = (120 / V2) + 1
axiom condition_4 : 40 / V1 = 80 / V3

-- Proof statement
theorem speed_of_first_car : V1 = 30 := by
  sorry

end speed_of_first_car_l196_196108


namespace hall_width_l196_196130

theorem hall_width (w : ℝ) (length height cost_per_m2 total_expenditure : ℝ)
  (h_length : length = 20)
  (h_height : height = 5)
  (h_cost : cost_per_m2 = 50)
  (h_expenditure : total_expenditure = 47500)
  (h_area : total_expenditure = cost_per_m2 * (2 * (length * w) + 2 * (length * height) + 2 * (w * height))) :
  w = 15 := 
sorry

end hall_width_l196_196130


namespace Tom_runs_60_miles_per_week_l196_196070

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l196_196070


namespace minimum_value_A_l196_196716

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l196_196716


namespace fraction_sum_ratio_l196_196399

theorem fraction_sum_ratio :
  let A := (Finset.range 1002).sum (λ k => 1 / ((2 * k + 1) * (2 * k + 2)))
  let B := (Finset.range 1002).sum (λ k => 1 / ((1003 + k) * (2004 - k)))
  (A / B) = (3007 / 2) :=
by
  sorry

end fraction_sum_ratio_l196_196399


namespace exponential_inequality_l196_196546

-- Define the conditions
variables {n : ℤ} {x : ℝ}

theorem exponential_inequality 
  (h1 : n ≥ 2) 
  (h2 : |x| < 1) 
  : 2^n > (1 - x)^n + (1 + x)^n :=
sorry

end exponential_inequality_l196_196546


namespace sitio_proof_l196_196607

theorem sitio_proof :
  (∃ t : ℝ, t = 4 + 7 + 12 ∧ 
    (∃ f : ℝ, 
      (∃ s : ℝ, s = 6 + 5 + 10 ∧ t = 23 ∧ f = 23 - s) ∧
      f = 2) ∧
    (∃ cost_per_hectare : ℝ, cost_per_hectare = 2420 / (4 + 12) ∧ 
      (∃ saci_spent : ℝ, saci_spent = 6 * cost_per_hectare ∧ saci_spent = 1320))) :=
by sorry

end sitio_proof_l196_196607


namespace number_of_valid_n_l196_196015

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m n : ℕ, n = 2^m * 5^n

def has_nonzero_thousandths_digit (n : ℕ) : Prop :=
  -- Placeholder for a formal definition to check the non-zero thousandths digit.
  sorry

theorem number_of_valid_n : 
  (∃ l : List ℕ, 
    l.length = 10 ∧ 
    ∀ n ∈ l, n <= 200 ∧ is_terminating_decimal n ∧ has_nonzero_thousandths_digit n) :=
sorry

end number_of_valid_n_l196_196015


namespace track_length_l196_196309

theorem track_length (h₁ : ∀ (x : ℕ), (exists y₁ y₂ : ℕ, y₁ = 120 ∧ y₂ = 180 ∧ y₁ + y₂ = x ∧ (y₂ - y₁ = 60) ∧ (y₂ = x - 120))) : 
  ∃ x : ℕ, x = 600 := by
  sorry

end track_length_l196_196309


namespace field_trip_people_per_bus_l196_196600

def number_of_people_on_each_bus (vans buses people_per_van total_people : ℕ) : ℕ :=
  (total_people - (vans * people_per_van)) / buses

theorem field_trip_people_per_bus :
  let vans := 9
  let buses := 10
  let people_per_van := 8
  let total_people := 342
  number_of_people_on_each_bus vans buses people_per_van total_people = 27 :=
by
  sorry

end field_trip_people_per_bus_l196_196600


namespace price_of_A_is_40_l196_196602

theorem price_of_A_is_40
  (p_a p_b : ℕ)
  (h1 : p_a = 2 * p_b)
  (h2 : 400 / p_a = 400 / p_b - 10) : p_a = 40 := 
by
  sorry

end price_of_A_is_40_l196_196602


namespace find_fraction_l196_196951

variable (x y z : ℝ)

theorem find_fraction (h : (x - y) / (z - y) = -10) : (x - z) / (y - z) = 11 := 
by
  sorry

end find_fraction_l196_196951


namespace sqrt_expression_l196_196878

theorem sqrt_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2 * x + 1)) :=
by
  sorry

end sqrt_expression_l196_196878


namespace find_pairs_of_nonneg_ints_l196_196373

theorem find_pairs_of_nonneg_ints (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n + 1) - 1) ↔ (m, n) = (9, 3) ∨ (m, n) = (6, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end find_pairs_of_nonneg_ints_l196_196373


namespace find_y_of_arithmetic_mean_l196_196828

theorem find_y_of_arithmetic_mean (y : ℝ) (h : (8 + 16 + 12 + 24 + 7 + y) / 6 = 12) : y = 5 :=
by
  sorry

end find_y_of_arithmetic_mean_l196_196828


namespace students_like_neither_l196_196216

theorem students_like_neither (N_Total N_Chinese N_Math N_Both N_Neither : ℕ)
  (h_total: N_Total = 62)
  (h_chinese: N_Chinese = 37)
  (h_math: N_Math = 49)
  (h_both: N_Both = 30)
  (h_neither: N_Neither = N_Total - (N_Chinese - N_Both) - (N_Math - N_Both) - N_Both) : 
  N_Neither = 6 :=
by 
  rw [h_total, h_chinese, h_math, h_both] at h_neither
  exact h_neither.trans (by norm_num)


end students_like_neither_l196_196216


namespace james_and_david_probability_l196_196211

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem james_and_david_probability :
  let total_workers := 22
  let chosen_workers := 4
  let j_and_d_chosen := 2
  (choose 20 2) / (choose 22 4) = (2 / 231) :=
by
  sorry

end james_and_david_probability_l196_196211


namespace chi_square_hypothesis_test_l196_196982

-- Definitions based on the conditions
def males_like_sports := "Males like to participate in sports activities"
def females_dislike_sports := "Females do not like to participate in sports activities"
def activities_related_to_gender := "Liking to participate in sports activities is related to gender"
def activities_not_related_to_gender := "Liking to participate in sports activities is not related to gender"

-- Statement to prove that D is the correct null hypothesis
theorem chi_square_hypothesis_test :
  activities_not_related_to_gender = "H₀: Liking to participate in sports activities is not related to gender" :=
sorry

end chi_square_hypothesis_test_l196_196982


namespace geom_seq_a12_value_l196_196966

-- Define the geometric sequence as a function from natural numbers to real numbers
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geom_seq_a12_value (a : ℕ → ℝ) 
  (H_geom : geom_seq a) 
  (H_7_9 : a 7 * a 9 = 4) 
  (H_4 : a 4 = 1) : 
  a 12 = 4 := 
by 
  sorry

end geom_seq_a12_value_l196_196966


namespace equilateral_triangle_l196_196780

theorem equilateral_triangle
  (A B C : Type)
  (angle_A : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : angle_A = 60)
  (h2 : side_BC = 1/3 * perimeter)
  (side_AB : ℝ)
  (side_AC : ℝ)
  (h3 : perimeter = side_BC + side_AB + side_AC) :
  (side_AB = side_BC) ∧ (side_AC = side_BC) :=
by
  sorry

end equilateral_triangle_l196_196780


namespace line_through_points_has_sum_m_b_3_l196_196347

-- Define the structure that two points are given
structure LineThroughPoints (P1 P2 : ℝ × ℝ) : Prop :=
  (slope_intercept_form : ∃ m b, (P1.snd = m * P1.fst + b) ∧ (P2.snd = b)) 

-- Define the particular points
def point1 : ℝ × ℝ := (-2, 0)
def point2 : ℝ × ℝ := (0, 2)

-- The theorem statement
theorem line_through_points_has_sum_m_b_3 
  (h : LineThroughPoints point1 point2) : 
  ∃ m b, (point1.snd = m * point1.fst + b) ∧ (point2.snd = b) ∧ (m + b = 3) :=
by
  sorry

end line_through_points_has_sum_m_b_3_l196_196347


namespace point_not_in_third_quadrant_l196_196718

theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) : ¬(x < 0 ∧ y < 0) :=
by
  sorry

end point_not_in_third_quadrant_l196_196718


namespace triangle_acute_angle_exists_l196_196820

theorem triangle_acute_angle_exists (a b c d e : ℝ)
  (h_abc : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_abd : a + b > d ∧ a + d > b ∧ b + d > a)
  (h_abe : a + b > e ∧ a + e > b ∧ b + e > a)
  (h_acd : a + c > d ∧ a + d > c ∧ c + d > a)
  (h_ace : a + c > e ∧ a + e > c ∧ c + e > a)
  (h_ade : a + d > e ∧ a + e > d ∧ d + e > a)
  (h_bcd : b + c > d ∧ b + d > c ∧ c + d > b)
  (h_bce : b + c > e ∧ b + e > c ∧ c + e > b)
  (h_bde : b + d > e ∧ b + e > d ∧ d + e > b)
  (h_cde : c + d > e ∧ c + e > d ∧ d + e > c) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
           x + y > z ∧ x + z > y ∧ y + z > x ∧
           (x * x + y * y > z * z ∧ y * y + z * z > x * x ∧ z * z + x * x > y * y) := 
sorry

end triangle_acute_angle_exists_l196_196820


namespace antisymmetric_function_multiplication_cauchy_solution_l196_196811

variable (f : ℤ → ℤ)
variable (h : ∀ x y : ℤ, f (x + y) = f x + f y)

theorem antisymmetric : ∀ x : ℤ, f (-x) = -f x := by
  sorry

theorem function_multiplication : ∀ x y : ℤ, f (x * y) = x * f y := by
  sorry

theorem cauchy_solution : ∃ c : ℤ, ∀ x : ℤ, f x = c * x := by
  sorry

end antisymmetric_function_multiplication_cauchy_solution_l196_196811


namespace find_result_l196_196587

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x - 3

theorem find_result : f (g 3) - g (f 3) = -6 := by
  sorry

end find_result_l196_196587


namespace simplified_expression_correct_l196_196029

noncomputable def simplified_expression : ℝ := 0.3 * 0.8 + 0.1 * 0.5

theorem simplified_expression_correct : simplified_expression = 0.29 := by 
  sorry

end simplified_expression_correct_l196_196029


namespace total_items_at_bakery_l196_196872

theorem total_items_at_bakery (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ) (h1 : bread_rolls = 49) (h2 : croissants = 19) (h3 : bagels = 22) : bread_rolls + croissants + bagels = 90 :=
by
  sorry

end total_items_at_bakery_l196_196872


namespace eval_composed_function_l196_196382

noncomputable def f (x : ℝ) := 3 * x^2 - 4
noncomputable def k (x : ℝ) := 5 * x^3 + 2

theorem eval_composed_function :
  f (k 2) = 5288 := 
by
  sorry

end eval_composed_function_l196_196382


namespace all_points_below_line_l196_196763

theorem all_points_below_line (a b : ℝ) (n : ℕ) (x y : ℕ → ℝ)
  (h1 : b > a)
  (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k = a + ((k : ℝ) * (b - a) / (n + 1)))
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k = a * (b / a) ^ (k / (n + 1))) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k < x k := 
sorry

end all_points_below_line_l196_196763


namespace porter_previous_painting_price_l196_196560

-- definitions from the conditions
def most_recent_sale : ℕ := 44000

-- definitions for the problem statement
def sale_equation (P : ℕ) : Prop :=
  most_recent_sale = 5 * P - 1000

theorem porter_previous_painting_price (P : ℕ) (h : sale_equation P) : P = 9000 :=
by {
  sorry
}

end porter_previous_painting_price_l196_196560


namespace apples_eaten_l196_196085

-- Define the number of apples eaten by Anna on Tuesday
def apples_eaten_on_Tuesday : ℝ := 4

theorem apples_eaten (A : ℝ) (h1 : A = apples_eaten_on_Tuesday) 
                      (h2 : 2 * A = 2 * apples_eaten_on_Tuesday) 
                      (h3 : A / 2 = apples_eaten_on_Tuesday / 2) 
                      (h4 : A + (2 * A) + (A / 2) = 14) : 
  A = 4 :=
by {
  sorry
}

end apples_eaten_l196_196085


namespace expand_binomials_l196_196666

variable (x y : ℝ)

theorem expand_binomials : 
  (3 * x - 2) * (2 * x + 4 * y + 1) = 6 * x^2 + 12 * x * y - x - 8 * y - 2 :=
by
  sorry

end expand_binomials_l196_196666


namespace solve_for_x_l196_196275

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 7 * x = 140) : x = 28 := by
  sorry

end solve_for_x_l196_196275


namespace shape_at_22_l196_196853

-- Define the pattern
def pattern : List String := ["triangle", "square", "diamond", "diamond", "circle"]

-- Function to get the nth shape in the repeated pattern sequence
def getShape (n : Nat) : String :=
  pattern.get! (n % pattern.length)

-- Statement to prove
theorem shape_at_22 : getShape 21 = "square" :=
by
  sorry

end shape_at_22_l196_196853


namespace polynomial_evaluation_l196_196591

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end polynomial_evaluation_l196_196591


namespace smallest_rectangle_area_l196_196541

-- Definitions based on conditions
def diameter : ℝ := 10
def length : ℝ := diameter
def width : ℝ := diameter + 2

-- Theorem statement
theorem smallest_rectangle_area : (length * width) = 120 :=
by
  -- The proof would go here, but we provide sorry for now
  sorry

end smallest_rectangle_area_l196_196541


namespace number_of_negative_x_l196_196650

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l196_196650


namespace total_budget_l196_196034

theorem total_budget (s_ticket : ℕ) (s_drinks_food : ℕ) (k_ticket : ℕ) (k_drinks : ℕ) (k_food : ℕ) 
  (h1 : s_ticket = 14) (h2 : s_drinks_food = 6) (h3 : k_ticket = 14) (h4 : k_drinks = 2) (h5 : k_food = 4) : 
  s_ticket + s_drinks_food + k_ticket + k_drinks + k_food = 40 := 
by
  sorry

end total_budget_l196_196034


namespace geom_seq_m_equals_11_l196_196505

theorem geom_seq_m_equals_11 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : |q| ≠ 1) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h4 : a m = (a 1) * (a 2) * (a 3) * (a 4) * (a 5)) : 
  m = 11 :=
sorry

end geom_seq_m_equals_11_l196_196505


namespace geometric_sequence_a6_l196_196925

variable {α : Type} [LinearOrderedSemiring α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ a₁ q : α, ∀ n, a n = a₁ * q ^ n

theorem geometric_sequence_a6 
  (a : ℕ → α) 
  (h_seq : is_geometric_sequence a) 
  (h1 : a 2 + a 4 = 20) 
  (h2 : a 3 + a 5 = 40) : 
  a 6 = 64 :=
by
  sorry

end geometric_sequence_a6_l196_196925


namespace derivative_at_two_l196_196256

noncomputable def f (a : ℝ) (g : ℝ) (x : ℝ) : ℝ := a * x^3 + g * x^2 + 3

theorem derivative_at_two (a f_prime_2 : ℝ) (h_deriv_at_1 : deriv (f a f_prime_2) 1 = -5) :
  deriv (f a f_prime_2) 2 = -5 := by
  sorry

end derivative_at_two_l196_196256


namespace prove_pqrstu_eq_416_l196_196530

-- Define the condition 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v
def condition (p q r s t u v : ℤ) (x : ℤ) : Prop :=
  1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v

-- State the theorem to prove p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416
theorem prove_pqrstu_eq_416 (p q r s t u v : ℤ) (h : ∀ x, condition p q r s t u v x) : 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 :=
sorry

end prove_pqrstu_eq_416_l196_196530


namespace min_bdf_proof_exists_l196_196106

noncomputable def minBDF (a b c d e f : ℕ) (A : ℕ) :=
  (A = 3 * a ∧ A = 4 * c ∧ A = 5 * e) →
  (a / b * c / d * e / f = A) →
  b * d * f = 60

theorem min_bdf_proof_exists :
  ∃ (a b c d e f A : ℕ), minBDF a b c d e f A :=
by
  sorry

end min_bdf_proof_exists_l196_196106


namespace original_price_of_sarees_l196_196116

theorem original_price_of_sarees
  (P : ℝ)
  (h_sale_price : 0.80 * P * 0.85 = 306) :
  P = 450 :=
sorry

end original_price_of_sarees_l196_196116


namespace find_smaller_number_l196_196011

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l196_196011


namespace base_10_uniqueness_l196_196675

theorem base_10_uniqueness : 
  (∀ a : ℕ, 12 = 3 * 4 ∧ 56 = 7 * 8 ↔ (a * b + (a + 1) = (a + 2) * (a + 3))) → b = 10 :=
sorry

end base_10_uniqueness_l196_196675


namespace income_distribution_after_tax_l196_196574

theorem income_distribution_after_tax (x : ℝ) (hx : 10 * x = 100) :
  let poor_income_initial := x
  let middle_income_initial := 4 * x
  let rich_income_initial := 5 * x
  let tax_rate := (x^2 / 4) + x
  let tax_collected := tax_rate * rich_income_initial / 100
  let poor_income_after := poor_income_initial + 3 / 4 * tax_collected
  let middle_income_after := middle_income_initial + 1 / 4 * tax_collected
  let rich_income_after := rich_income_initial - tax_collected
  poor_income_after = 0.23125 * 100 ∧
  middle_income_after = 0.44375 * 100 ∧
  rich_income_after = 0.325 * 100 :=
by {
  sorry
}

end income_distribution_after_tax_l196_196574


namespace rational_x_of_rational_x3_and_x2_add_x_l196_196097

variable {x : ℝ}

theorem rational_x_of_rational_x3_and_x2_add_x (hx3 : ∃ a : ℚ, x^3 = a)
  (hx2_add_x : ∃ b : ℚ, x^2 + x = b) : ∃ r : ℚ, x = r :=
sorry

end rational_x_of_rational_x3_and_x2_add_x_l196_196097


namespace perp_lines_l196_196480

noncomputable def line_1 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (5 - k) * y + 1
noncomputable def line_2 (k : ℝ) : ℝ → ℝ → ℝ := λ x y => 2 * (k - 3) * x - 2 * y + 3

theorem perp_lines (k : ℝ) : 
  let l1 := line_1 k
  let l2 := line_2 k
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k = 1 ∨ k = 4)) :=
by
    sorry

end perp_lines_l196_196480


namespace prime_factor_of_sum_of_four_consecutive_integers_l196_196835

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by 
  sorry

end prime_factor_of_sum_of_four_consecutive_integers_l196_196835


namespace number_of_digits_in_product_l196_196792

open Nat

noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log 10 n + 1

def compute_product : ℕ := 234567 * 123^3

theorem number_of_digits_in_product : num_digits compute_product = 13 := by 
  sorry

end number_of_digits_in_product_l196_196792


namespace manager_salary_3700_l196_196908

theorem manager_salary_3700
  (salary_20_employees_avg : ℕ)
  (salary_increase : ℕ)
  (total_employees : ℕ)
  (manager_salary : ℕ)
  (h_avg : salary_20_employees_avg = 1600)
  (h_increase : salary_increase = 100)
  (h_total_employees : total_employees = 20)
  (h_manager_salary : manager_salary = 21 * (salary_20_employees_avg + salary_increase) - 20 * salary_20_employees_avg) :
  manager_salary = 3700 :=
by
  sorry

end manager_salary_3700_l196_196908


namespace nate_search_time_l196_196052

theorem nate_search_time
  (rowsG : Nat) (cars_per_rowG : Nat)
  (rowsH : Nat) (cars_per_rowH : Nat)
  (rowsI : Nat) (cars_per_rowI : Nat)
  (walk_speed : Nat) : Nat :=
  let total_cars : Nat := rowsG * cars_per_rowG + rowsH * cars_per_rowH + rowsI * cars_per_rowI
  let total_minutes : Nat := total_cars / walk_speed
  if total_cars % walk_speed == 0 then total_minutes else total_minutes + 1

/-- Given:
- rows in Section G = 18, cars per row in Section G = 12
- rows in Section H = 25, cars per row in Section H = 10
- rows in Section I = 17, cars per row in Section I = 11
- Nate's walking speed is 8 cars per minute
Prove: Nate took 82 minutes to search the parking lot
-/
example : nate_search_time 18 12 25 10 17 11 8 = 82 := by
  sorry

end nate_search_time_l196_196052


namespace smallest_three_digit_number_l196_196159

theorem smallest_three_digit_number (digits : Finset ℕ) (h_digits : digits = {0, 3, 5, 6}) : 
  ∃ n, n = 305 ∧ ∀ m, (m ∈ digits) → (m ≠ 0) → (m < 305) → false :=
by
  sorry

end smallest_three_digit_number_l196_196159


namespace car_travels_more_l196_196766

theorem car_travels_more (train_speed : ℕ) (car_speed : ℕ) (time : ℕ)
  (h1 : train_speed = 60)
  (h2 : car_speed = 2 * train_speed)
  (h3 : time = 3) :
  car_speed * time - train_speed * time = 180 :=
by
  sorry

end car_travels_more_l196_196766


namespace willie_stickers_l196_196533

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (final_stickers : ℕ) 
  (h1 : initial_stickers = 124) 
  (h2 : given_stickers = 43) 
  (h3 : final_stickers = initial_stickers - given_stickers) :
  final_stickers = 81 :=
sorry

end willie_stickers_l196_196533


namespace problem_conditions_l196_196839

noncomputable def f (a b x : ℝ) : ℝ := abs (x + a) + abs (2 * x - b)

theorem problem_conditions (ha : 0 < a) (hb : 0 < b) 
  (hmin : ∃ x : ℝ, f a b x = 1) : 
  2 * a + b = 2 ∧ 
  ∀ (t : ℝ), (∀ a b : ℝ, 
    (0 < a) → (0 < b) → (a + 2 * b ≥ t * a * b)) → 
  t ≤ 9 / 2 :=
by
  sorry

end problem_conditions_l196_196839


namespace division_problem_l196_196659

theorem division_problem (n : ℕ) (h : n / 6 = 209) : n = 1254 := 
sorry

end division_problem_l196_196659


namespace value_of_a_if_perpendicular_l196_196569

theorem value_of_a_if_perpendicular (a l : ℝ) :
  (∀ x y : ℝ, (a + l) * x + 2 * y = 0 → x - a * y = 1 → false) → a = 1 :=
by
  -- Proof is omitted
  sorry

end value_of_a_if_perpendicular_l196_196569


namespace calculate_length_of_bridge_l196_196523

/-- Define the conditions based on given problem -/
def length_of_bridge (speed1 speed2 : ℕ) (length1 length2 : ℕ) (time : ℕ) : ℕ :=
    let distance_covered_train1 := speed1 * time
    let bridge_length_train1 := distance_covered_train1 - length1
    let distance_covered_train2 := speed2 * time
    let bridge_length_train2 := distance_covered_train2 - length2
    max bridge_length_train1 bridge_length_train2

/-- Given conditions -/
def speed_train1 := 15 -- in m/s
def length_train1 := 130 -- in meters
def speed_train2 := 20 -- in m/s
def length_train2 := 90 -- in meters
def crossing_time := 30 -- in seconds

theorem calculate_length_of_bridge : length_of_bridge speed_train1 speed_train2 length_train1 length_train2 crossing_time = 510 :=
by
  -- omitted proof
  sorry

end calculate_length_of_bridge_l196_196523


namespace part1_part2_l196_196963

def f (x : ℝ) : ℝ := abs (x - 1)
def g (x : ℝ) : ℝ := abs (2 * x - 1)

theorem part1 (x : ℝ) :
  abs (x + 4) ≤ x * abs (2 * x - 1) ↔ x ≥ 2 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, abs ((x + 2) - 1) + abs (x - 1) + a = 0 → False) ↔ a ≤ -2 :=
sorry

end part1_part2_l196_196963


namespace monotonic_increasing_k_l196_196262

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (3 * k - 2) * x - 5

theorem monotonic_increasing_k (k : ℝ) : (∀ x y : ℝ, 1 ≤ x → x ≤ y → f k x ≤ f k y) ↔ k ∈ Set.Ici (2 / 5) :=
by
  sorry

end monotonic_increasing_k_l196_196262


namespace shifted_parabola_eq_l196_196649

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -(x^2)

-- Define the transformation for shifting left 2 units
def shift_left (x : ℝ) : ℝ := x + 2

-- Define the transformation for shifting down 3 units
def shift_down (y : ℝ) : ℝ := y - 3

-- Define the new parabola equation after shifting
def new_parabola (x : ℝ) : ℝ := shift_down (original_parabola (shift_left x))

-- The theorem to be proven
theorem shifted_parabola_eq : new_parabola x = -(x + 2)^2 - 3 := by
  sorry

end shifted_parabola_eq_l196_196649


namespace parallel_lines_implies_a_eq_one_l196_196996

theorem parallel_lines_implies_a_eq_one 
(h_parallel: ∀ (a : ℝ), ∀ (x y : ℝ), (x + a * y = 2 * a + 2) → (a * x + y = a + 1) → -1/a = -a) :
  ∀ (a : ℝ), a = 1 := by
  sorry

end parallel_lines_implies_a_eq_one_l196_196996


namespace real_numbers_division_l196_196147

def is_non_neg (x : ℝ) : Prop := x ≥ 0

theorem real_numbers_division :
  ∀ x : ℝ, x < 0 ∨ is_non_neg x :=
by
  intro x
  by_cases h : x < 0
  · left
    exact h
  · right
    push_neg at h
    exact h

end real_numbers_division_l196_196147


namespace range_of_a_l196_196138

def proposition_p (a : ℝ) : Prop :=
  (a + 6) * (a - 7) < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4 * x + a < 0

def neg_q (a : ℝ) : Prop :=
  a ≥ 4

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ neg_q a) ↔ a ∈ Set.Ioo (-6 : ℝ) (7 : ℝ) ∪ Set.Ici (4 : ℝ) :=
sorry

end range_of_a_l196_196138


namespace find_prime_pairs_l196_196287

theorem find_prime_pairs :
  ∃ p q : ℕ, Prime p ∧ Prime q ∧
    ∃ a b : ℕ, a^2 = p - q ∧ b^2 = pq - q ∧ (p = 3 ∧ q = 2) :=
by
  sorry

end find_prime_pairs_l196_196287


namespace gum_total_l196_196057

theorem gum_total (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) : 
  58 + x + y = 58 + x + y :=
by sorry

end gum_total_l196_196057


namespace Gargamel_bought_tires_l196_196948

def original_price_per_tire := 84
def sale_price_per_tire := 75
def total_savings := 36
def discount_per_tire := original_price_per_tire - sale_price_per_tire
def num_tires (total_savings : ℕ) (discount_per_tire : ℕ) := total_savings / discount_per_tire

theorem Gargamel_bought_tires :
  num_tires total_savings discount_per_tire = 4 :=
by
  sorry

end Gargamel_bought_tires_l196_196948


namespace equation_of_circle_l196_196049

-- Definitions directly based on conditions 
noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)
noncomputable def directrix_of_parabola : ℝ × ℝ -> Prop
  | (x, _) => x = -1

-- The statement of the problem: equation of the circle with given conditions
theorem equation_of_circle : ∃ (r : ℝ), (∀ (x y : ℝ), (x - 1)^2 + y^2 = r^2) ∧ r = 2 :=
sorry

end equation_of_circle_l196_196049


namespace minimum_value_l196_196634

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), (∃ (x : ℝ), x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) ∧ (a^2 + b^2 = 4 / 5)

-- This line states that the minimum possible value of a^2 + b^2, given the condition, is 4/5.
theorem minimum_value (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
  sorry

end minimum_value_l196_196634


namespace exists_integers_a_b_for_m_l196_196103

theorem exists_integers_a_b_for_m (m : ℕ) (h : 0 < m) :
  ∃ a b : ℤ, |a| ≤ m ∧ |b| ≤ m ∧ 0 < a + b * Real.sqrt 2 ∧ a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2) :=
by
  sorry

end exists_integers_a_b_for_m_l196_196103


namespace fluffy_striped_or_spotted_cats_l196_196381

theorem fluffy_striped_or_spotted_cats (total_cats : ℕ) (striped_fraction : ℚ) (spotted_fraction : ℚ)
    (fluffy_striped_fraction : ℚ) (fluffy_spotted_fraction : ℚ) (striped_spotted_fraction : ℚ) :
    total_cats = 180 ∧ striped_fraction = 1/2 ∧ spotted_fraction = 1/3 ∧
    fluffy_striped_fraction = 1/8 ∧ fluffy_spotted_fraction = 3/7 →
    striped_spotted_fraction = 36 :=
by
    sorry

end fluffy_striped_or_spotted_cats_l196_196381


namespace loss_percentage_l196_196907

theorem loss_percentage (CP SP_gain L : ℝ) 
  (h1 : CP = 1500)
  (h2 : SP_gain = CP + 0.05 * CP)
  (h3 : SP_gain = CP - (L/100) * CP + 225) : 
  L = 10 :=
by
  sorry

end loss_percentage_l196_196907


namespace find_original_number_l196_196992

theorem find_original_number (x : ℕ) :
  (43 * x - 34 * x = 1251) → x = 139 :=
by
  sorry

end find_original_number_l196_196992


namespace term_transition_addition_l196_196301

theorem term_transition_addition (k : Nat) :
  (2:ℚ) / ((k + 1) * (k + 2)) = ((2:ℚ) / ((k * (k + 1))) - ((2:ℚ) / ((k + 1) * (k + 2)))) := 
sorry

end term_transition_addition_l196_196301


namespace find_x_in_isosceles_triangle_l196_196537

def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def triangle_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem find_x_in_isosceles_triangle (x : ℝ) :
  is_isosceles (x + 3) (2 * x + 1) 11 ∧ triangle_inequality (x + 3) (2 * x + 1) 11 →
  (x = 8) ∨ (x = 5) :=
sorry

end find_x_in_isosceles_triangle_l196_196537


namespace square_window_side_length_is_24_l196_196814

noncomputable def side_length_square_window
  (num_panes_per_row : ℕ) (pane_height_ratio : ℝ) (border_width : ℝ) (x : ℝ) : ℝ :=
  num_panes_per_row * x + (num_panes_per_row + 1) * border_width

theorem square_window_side_length_is_24
  (num_panes_per_row : ℕ)
  (pane_height_ratio : ℝ)
  (border_width : ℝ) 
  (pane_width : ℝ)
  (pane_height : ℝ)
  (window_side_length : ℝ) : 
  (num_panes_per_row = 3) →
  (pane_height_ratio = 3) →
  (border_width = 3) →
  (pane_height = pane_height_ratio * pane_width) →
  (window_side_length = side_length_square_window num_panes_per_row pane_height_ratio border_width pane_width) →
  (window_side_length = 24) :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end square_window_side_length_is_24_l196_196814


namespace simplify_sqrt_square_l196_196205

theorem simplify_sqrt_square (h : Real.sqrt 7 < 3) : Real.sqrt ((Real.sqrt 7 - 3)^2) = 3 - Real.sqrt 7 :=
by
  sorry

end simplify_sqrt_square_l196_196205


namespace arithmetic_result_l196_196838

theorem arithmetic_result :
  (3 * 13) + (3 * 14) + (3 * 17) + 11 = 143 :=
by
  sorry

end arithmetic_result_l196_196838


namespace eight_p_plus_one_is_composite_l196_196646

theorem eight_p_plus_one_is_composite (p : ℕ) (hp : Nat.Prime p) (h8p1 : Nat.Prime (8 * p - 1)) : ¬ Nat.Prime (8 * p + 1) :=
by
  sorry

end eight_p_plus_one_is_composite_l196_196646


namespace quadratic_ineq_solution_l196_196405

theorem quadratic_ineq_solution (a b : ℝ) 
  (h_solution_set : ∀ x, (ax^2 + bx - 1 > 0) ↔ (1 / 3 < x ∧ x < 1))
  (h_roots : (a / 3 + b = -1 / a) ∧ (a / 3 = -1 / a)) 
  (h_a_neg : a < 0) : a + b = 1 := 
sorry 

end quadratic_ineq_solution_l196_196405


namespace largest_interior_angle_l196_196825

theorem largest_interior_angle (x : ℝ) (h_ratio : (5*x + 4*x + 3*x = 360)) :
  let e1 := 3 * x
  let e2 := 4 * x
  let e3 := 5 * x
  let i1 := 180 - e1
  let i2 := 180 - e2
  let i3 := 180 - e3
  max i1 (max i2 i3) = 90 :=
sorry

end largest_interior_angle_l196_196825


namespace distinct_positive_integers_count_l196_196781

-- Define the digits' ranges
def digit (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 9
def nonzero_digit (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define the 4-digit numbers ABCD and DCBA
def ABCD (A B C D : ℤ) := 1000 * A + 100 * B + 10 * C + D
def DCBA (A B C D : ℤ) := 1000 * D + 100 * C + 10 * B + A

-- Define the difference
def difference (A B C D : ℤ) := ABCD A B C D - DCBA A B C D

-- The theorem to be proven
theorem distinct_positive_integers_count :
  ∃ n : ℤ, n = 161 ∧
  ∀ A B C D : ℤ,
  nonzero_digit A → nonzero_digit D → digit B → digit C → 
  0 < difference A B C D → (∃! x : ℤ, x = difference A B C D) :=
sorry

end distinct_positive_integers_count_l196_196781


namespace phoebe_dog_peanut_butter_l196_196455

-- Definitions based on the conditions
def servings_per_jar : ℕ := 15
def jars_needed : ℕ := 4
def days : ℕ := 30

-- Problem statement
theorem phoebe_dog_peanut_butter :
  (jars_needed * servings_per_jar) / days / 2 = 1 :=
by sorry

end phoebe_dog_peanut_butter_l196_196455


namespace range_of_a_l196_196043

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) :=
by
  sorry

end range_of_a_l196_196043


namespace count_positive_bases_for_log_1024_l196_196877

-- Define the conditions 
def is_positive_integer_log_base (b n : ℕ) : Prop := b^n = 1024 ∧ n > 0

-- State that there are exactly 4 positive integers b that satisfy the condition
theorem count_positive_bases_for_log_1024 :
  (∃ b1 b2 b3 b4 : ℕ, b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    (∀ b, is_positive_integer_log_base b 1 ∨ is_positive_integer_log_base b 2 ∨ is_positive_integer_log_base b 5 ∨ is_positive_integer_log_base b 10) ∧
    (is_positive_integer_log_base b1 1 ∨ is_positive_integer_log_base b1 2 ∨ is_positive_integer_log_base b1 5 ∨ is_positive_integer_log_base b1 10) ∧
    (is_positive_integer_log_base b2 1 ∨ is_positive_integer_log_base b2 2 ∨ is_positive_integer_log_base b2 5 ∨ is_positive_integer_log_base b2 10) ∧
    (is_positive_integer_log_base b3 1 ∨ is_positive_integer_log_base b3 2 ∨ is_positive_integer_log_base b3 5 ∨ is_positive_integer_log_base b3 10) ∧
    (is_positive_integer_log_base b4 1 ∨ is_positive_integer_log_base b4 2 ∨ is_positive_integer_log_base b4 5 ∨ is_positive_integer_log_base b4 10)) :=
sorry

end count_positive_bases_for_log_1024_l196_196877


namespace prove_fractions_sum_equal_11_l196_196213

variable (a b c : ℝ)

-- Given conditions
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -9
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 10

-- The proof problem statement
theorem prove_fractions_sum_equal_11 : (b / (a + b) + c / (b + c) + a / (c + a)) = 11 :=
by
  sorry

end prove_fractions_sum_equal_11_l196_196213


namespace drum_filled_capacity_l196_196298

theorem drum_filled_capacity (C : ℝ) (h1 : 0 < C) :
    (4 / 5) * C + (1 / 2) * C = (13 / 10) * C :=
by
  sorry

end drum_filled_capacity_l196_196298


namespace difference_of_squares_l196_196448

variable (a b : ℝ)

theorem difference_of_squares (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := 
by
  sorry

end difference_of_squares_l196_196448


namespace number_of_solutions_l196_196700

open Real

-- Define the main equation in terms of absolute values 
def equation (x : ℝ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that there are exactly 2 distinct solutions to the equation
theorem number_of_solutions : 
  ∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ :=
sorry

end number_of_solutions_l196_196700


namespace boy_reaches_early_l196_196596

theorem boy_reaches_early (usual_rate new_rate : ℝ) (Usual_Time New_Time : ℕ) 
  (Hrate : new_rate = 9/8 * usual_rate) (Htime : Usual_Time = 36) :
  New_Time = 32 → Usual_Time - New_Time = 4 :=
by
  intros
  subst_vars
  sorry

end boy_reaches_early_l196_196596


namespace total_potatoes_l196_196194

open Nat

theorem total_potatoes (P T R : ℕ) (h1 : P = 5) (h2 : T = 6) (h3 : R = 48) : P + (R / T) = 13 := by
  sorry

end total_potatoes_l196_196194


namespace coeff_sum_eq_twenty_l196_196612

theorem coeff_sum_eq_twenty 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h : ((2 * x - 3) ^ 5) = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 20 :=
by
  sorry

end coeff_sum_eq_twenty_l196_196612


namespace rosie_can_make_nine_pies_l196_196915

theorem rosie_can_make_nine_pies (apples pies : ℕ) (h : apples = 12 ∧ pies = 3) : 36 / (12 / 3) * pies = 9 :=
by
  sorry

end rosie_can_make_nine_pies_l196_196915


namespace line_tangent_to_parabola_l196_196098

theorem line_tangent_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → 16 - 16 * c = 0) → c = 1 :=
by
  intros h
  sorry

end line_tangent_to_parabola_l196_196098


namespace part_1_part_2_l196_196307

theorem part_1 (a : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) (n : ℕ) (hn_pos : 0 < n) : 
  a (n + 1) - 2 * a n = 0 :=
sorry

theorem part_2 (a : ℕ → ℚ) (b : ℕ → ℚ) (S : ℕ → ℚ) (h1 : ∀ n, S (n + 1) = 4 * a n - 2) (h2 : a 1 = 2) :
  (∀ n, b n = 1 / (a n * a (n + 1))) → ∀ n, S n = (1/6) * (1 - (1/4)^n) :=
sorry

end part_1_part_2_l196_196307


namespace inversely_proportional_value_l196_196358

theorem inversely_proportional_value (a b k : ℝ) (h1 : a * b = k) (h2 : a = 40) (h3 : b = 8) :
  ∃ a' : ℝ, a' * 10 = k ∧ a' = 32 :=
by {
  use 32,
  sorry
}

end inversely_proportional_value_l196_196358


namespace every_algorithm_must_have_sequential_structure_l196_196515

def is_sequential_structure (alg : Type) : Prop := sorry -- This defines what a sequential structure is

def must_have_sequential_structure (alg : Type) : Prop :=
∀ alg, is_sequential_structure alg

theorem every_algorithm_must_have_sequential_structure :
  must_have_sequential_structure nat := sorry

end every_algorithm_must_have_sequential_structure_l196_196515


namespace max_discount_l196_196637

variable (x : ℝ)

theorem max_discount (h1 : (1 + 0.8) * x = 360) : 360 - 1.2 * x = 120 := 
by
  sorry

end max_discount_l196_196637


namespace successive_percentage_reduction_l196_196876

theorem successive_percentage_reduction (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  a + b - (a * b) / 100 = 40 := by
  sorry

end successive_percentage_reduction_l196_196876


namespace triangle_perimeter_l196_196268

theorem triangle_perimeter :
  let a := 15
  let b := 10
  let c := 12
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a + b + c = 37) :=
by
  intros
  sorry

end triangle_perimeter_l196_196268


namespace combined_time_l196_196801

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l196_196801


namespace correct_statement_about_algorithms_l196_196074

-- Definitions based on conditions
def is_algorithm (A B C D : Prop) : Prop :=
  ¬A ∧ B ∧ ¬C ∧ ¬D

-- Ensure the correct statement using the conditions specified
theorem correct_statement_about_algorithms (A B C D : Prop) (h : is_algorithm A B C D) : B :=
by
  obtain ⟨hnA, hB, hnC, hnD⟩ := h
  exact hB

end correct_statement_about_algorithms_l196_196074


namespace store_hours_open_per_day_l196_196281

theorem store_hours_open_per_day
  (rent_per_week : ℝ)
  (utility_percentage : ℝ)
  (employees_per_shift : ℕ)
  (hourly_wage : ℝ)
  (days_per_week_open : ℕ)
  (weekly_expenses : ℝ)
  (H_rent : rent_per_week = 1200)
  (H_utility_percentage : utility_percentage = 0.20)
  (H_employees_per_shift : employees_per_shift = 2)
  (H_hourly_wage : hourly_wage = 12.50)
  (H_days_open : days_per_week_open = 5)
  (H_weekly_expenses : weekly_expenses = 3440) :
  (16 : ℝ) = weekly_expenses / ((rent_per_week * (1 + utility_percentage)) + (employees_per_shift * hourly_wage * days_per_week_open)) :=
by
  sorry

end store_hours_open_per_day_l196_196281


namespace solve_for_question_mark_l196_196402

/-- Prove that the number that should replace "?" in the equation 
    300 * 2 + (12 + ?) * (1 / 8) = 602 is equal to 4. -/
theorem solve_for_question_mark : 
  ∃ (x : ℕ), 300 * 2 + (12 + x) * (1 / 8) = 602 ∧ x = 4 := 
by
  sorry

end solve_for_question_mark_l196_196402


namespace min_value_fraction_l196_196644

theorem min_value_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b > 0) (h₃ : 2 * a + b = 1) : 
  ∃ x, x = 8 ∧ ∀ y, (y = (1 / a) + (2 / b)) → y ≥ x :=
sorry

end min_value_fraction_l196_196644


namespace min_value_expression_l196_196527

theorem min_value_expression (x : ℝ) (hx : x > 0) : x + 4/x ≥ 4 :=
sorry

end min_value_expression_l196_196527


namespace expression_of_fn_l196_196344

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then x else f (n - 1) x / (1 + n * x)

theorem expression_of_fn (n : ℕ) (x : ℝ) (hn : 1 ≤ n) : f n x = x / (1 + n * x) :=
sorry

end expression_of_fn_l196_196344


namespace angle_measure_l196_196322

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l196_196322


namespace increasing_or_decreasing_subseq_l196_196684

theorem increasing_or_decreasing_subseq (a : Fin (m * n + 1) → ℝ) :
  ∃ (s : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (s i) ≤ a (s j)) ∨
  ∃ (t : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (t i) ≥ a (t j)) :=
sorry

end increasing_or_decreasing_subseq_l196_196684


namespace square_roots_equal_implication_l196_196414

theorem square_roots_equal_implication (b : ℝ) (h : 5 * b = 3 + 2 * b) : -b = -1 := 
by sorry

end square_roots_equal_implication_l196_196414


namespace factor_expression_l196_196818

theorem factor_expression (b : ℤ) : 
  (8 * b ^ 3 + 120 * b ^ 2 - 14) - (9 * b ^ 3 - 2 * b ^ 2 + 14) 
  = -1 * (b ^ 3 - 122 * b ^ 2 + 28) := 
by {
  sorry
}

end factor_expression_l196_196818


namespace count_multiples_5_or_7_not_35_l196_196685

def count_multiples_5 (n : ℕ) : ℕ := n / 5
def count_multiples_7 (n : ℕ) : ℕ := n / 7
def count_multiples_35 (n : ℕ) : ℕ := n / 35
def inclusion_exclusion (a b c : ℕ) : ℕ := a + b - c

theorem count_multiples_5_or_7_not_35 : 
  inclusion_exclusion (count_multiples_5 3000) (count_multiples_7 3000) (count_multiples_35 3000) = 943 :=
by
  sorry

end count_multiples_5_or_7_not_35_l196_196685


namespace find_n_l196_196516

theorem find_n {
    n : ℤ
   } (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 99 * n ≡ 72 [ZMOD 103]) :
    n = 52 :=
sorry

end find_n_l196_196516


namespace match_sequences_count_l196_196274

-- Definitions based on the given conditions
def team_size : ℕ := 7
def total_matches : ℕ := 2 * team_size - 1

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement: number of possible match sequences
theorem match_sequences_count : 
  2 * binomial_coefficient total_matches team_size = 3432 :=
by
  sorry

end match_sequences_count_l196_196274


namespace find_slower_train_speed_l196_196053

theorem find_slower_train_speed (l : ℝ) (vf : ℝ) (t : ℝ) (v_s : ℝ) 
  (h1 : l = 37.5)   -- Length of each train
  (h2 : vf = 46)   -- Speed of the faster train in km/hr
  (h3 : t = 27)    -- Time in seconds to pass the slower train
  (h4 : (2 * l) = ((46 - v_s) * (5 / 18) * 27))   -- Distance covered at relative speed
  : v_s = 36 := 
sorry

end find_slower_train_speed_l196_196053


namespace tan_half_A_mul_tan_half_C_eq_third_l196_196119

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem tan_half_A_mul_tan_half_C_eq_third (h : a + c = 2 * b) :
  (Real.tan (A / 2)) * (Real.tan (C / 2)) = 1 / 3 :=
sorry

end tan_half_A_mul_tan_half_C_eq_third_l196_196119


namespace Ron_spends_15_dollars_l196_196520

theorem Ron_spends_15_dollars (cost_per_bar : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (s'mores_per_scout : ℕ) :
  cost_per_bar = 1.50 ∧ sections_per_bar = 3 ∧ num_scouts = 15 ∧ s'mores_per_scout = 2 →
  cost_per_bar * (num_scouts * s'mores_per_scout / sections_per_bar) = 15 :=
by
  sorry

end Ron_spends_15_dollars_l196_196520


namespace calculate_area_of_shaded_region_l196_196552

namespace Proof

noncomputable def AreaOfShadedRegion (width height : ℝ) (divisions : ℕ) : ℝ :=
  let small_width := width
  let small_height := height / divisions
  let area_of_small := small_width * small_height
  let shaded_in_small := area_of_small / 2
  let total_shaded := divisions * shaded_in_small
  total_shaded

theorem calculate_area_of_shaded_region :
  AreaOfShadedRegion 3 14 4 = 21 := by
  sorry

end Proof

end calculate_area_of_shaded_region_l196_196552


namespace not_right_triangle_angle_ratio_l196_196476

theorem not_right_triangle_angle_ratio (A B C : ℝ) (h₁ : A / B = 3 / 4) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end not_right_triangle_angle_ratio_l196_196476


namespace trigonometric_identity_l196_196492

theorem trigonometric_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) :
  Real.tan θ + 1 / Real.tan θ = 2 :=
by
  sorry

end trigonometric_identity_l196_196492


namespace gravel_weight_is_correct_l196_196286

def weight_of_gravel (total_weight : ℝ) (fraction_sand : ℝ) (fraction_water : ℝ) : ℝ :=
  total_weight - (fraction_sand * total_weight + fraction_water * total_weight)

theorem gravel_weight_is_correct :
  weight_of_gravel 23.999999999999996 (1 / 3) (1 / 4) = 10 :=
by
  sorry

end gravel_weight_is_correct_l196_196286


namespace find_a_l196_196679

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem find_a (a : ℝ) (h : f (f a) = f 9 + 1) : a = -1/4 := 
by 
  sorry

end find_a_l196_196679


namespace boat_avg_speed_ratio_l196_196110

/--
A boat moves at a speed of 20 mph in still water. When traveling in a river with a current of 3 mph, it travels 24 miles downstream and then returns upstream to the starting point. Prove that the ratio of the average speed for the entire round trip to the boat's speed in still water is 97765 / 100000.
-/
theorem boat_avg_speed_ratio :
  let boat_speed := 20 -- mph in still water
  let current_speed := 3 -- mph river current
  let distance := 24 -- miles downstream and upstream
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  let total_time := time_downstream + time_upstream
  let total_distance := distance * 2
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 97765 / 100000 :=
by
  sorry

end boat_avg_speed_ratio_l196_196110


namespace angles_sum_n_l196_196229

/-- Given that the sum of the measures in degrees of angles A, B, C, D, E, and F is 90 * n,
    we need to prove that n = 4. -/
theorem angles_sum_n (A B C D E F : ℝ) (n : ℕ) 
  (h : A + B + C + D + E + F = 90 * n) :
  n = 4 :=
sorry

end angles_sum_n_l196_196229


namespace D_is_quadratic_l196_196787

-- Define the equations
def eq_A (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq_B (x : ℝ) : Prop := 2 * x^2 - 3 * x = 2 * (x^2 - 2)
def eq_C (x : ℝ) : Prop := x^3 - 2 * x + 7 = 0
def eq_D (x : ℝ) : Prop := (x - 2)^2 - 4 = 0

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x^2 + b * x + c = 0)

theorem D_is_quadratic : is_quadratic eq_D :=
sorry

end D_is_quadratic_l196_196787


namespace opposite_numbers_pow_sum_zero_l196_196751

theorem opposite_numbers_pow_sum_zero (a b : ℝ) (h : a + b = 0) : a^5 + b^5 = 0 :=
by sorry

end opposite_numbers_pow_sum_zero_l196_196751


namespace infinite_series_sum_l196_196067

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * n - 1) / 3 ^ (n + 1)) = 2 :=
by
  sorry

end infinite_series_sum_l196_196067


namespace math_problem_proof_l196_196158

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l196_196158


namespace area_enclosed_by_curve_l196_196532

theorem area_enclosed_by_curve :
  let s : ℝ := 3
  let arc_length : ℝ := (3 * Real.pi) / 4
  let octagon_area : ℝ := (1 + Real.sqrt 2) * s^2
  let sector_area : ℝ := (3 / 8) * Real.pi
  let total_area : ℝ := 8 * sector_area + octagon_area
  total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi :=
by
  let s := 3
  let arc_length := (3 * Real.pi) / 4
  let r := arc_length / ((3 * Real.pi) / 4)
  have r_eq : r = 1 := by
    sorry
  let full_circle_area := Real.pi * r^2
  let sector_area := (3 / 8) * Real.pi
  have sector_area_eq : sector_area = (3 / 8) * Real.pi := by
    sorry
  let total_sector_area := 8 * sector_area
  have total_sector_area_eq : total_sector_area = 3 * Real.pi := by
    sorry
  let octagon_area := (1 + Real.sqrt 2) * s^2
  have octagon_area_eq : octagon_area = 9 * (1 + Real.sqrt 2) := by
    sorry
  let total_area := total_sector_area + octagon_area
  have total_area_eq : total_area = 9 + 9 * Real.sqrt 2 + 3 * Real.pi := by
    sorry
  exact total_area_eq

end area_enclosed_by_curve_l196_196532


namespace revenue_comparison_l196_196214

theorem revenue_comparison 
  (D N J F : ℚ) 
  (hN : N = (2 / 5) * D) 
  (hJ : J = (2 / 25) * D) 
  (hF : F = (3 / 4) * D) : 
  D / ((N + J + F) / 3) = 100 / 41 := 
by 
  sorry

end revenue_comparison_l196_196214


namespace joan_and_karl_sofas_l196_196359

variable (J K : ℝ)

theorem joan_and_karl_sofas (hJ : J = 230) (hSum : J + K = 600) :
  2 * J - K = 90 :=
by
  sorry

end joan_and_karl_sofas_l196_196359


namespace solvable_system_of_inequalities_l196_196036

theorem solvable_system_of_inequalities (n : ℕ) : 
  (∃ x : ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k < x ^ k ∧ x ^ k < k + 1)) ∧ (1 < x ∧ x < 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end solvable_system_of_inequalities_l196_196036


namespace total_oranges_and_weight_l196_196055

theorem total_oranges_and_weight 
  (oranges_per_child : ℕ) (num_children : ℕ) (average_weight_per_orange : ℝ)
  (h1 : oranges_per_child = 3)
  (h2 : num_children = 4)
  (h3 : average_weight_per_orange = 0.3) :
  oranges_per_child * num_children = 12 ∧ (oranges_per_child * num_children : ℝ) * average_weight_per_orange = 3.6 :=
by
  sorry

end total_oranges_and_weight_l196_196055


namespace product_is_zero_l196_196333

theorem product_is_zero (b : ℤ) (h : b = 3) :
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 :=
by {
  -- Substituting b = 3
  -- (3-5) * (3-4) * (3-3) * (3-2) * (3-1) * 3 * (3+1) * (3+2)
  -- = (-2) * (-1) * 0 * 1 * 2 * 3 * 4 * 5
  -- = 0
  sorry
}

end product_is_zero_l196_196333


namespace area_of_square_l196_196971

-- Define the conditions given in the problem
def radius_circle := 7 -- radius of each circle in inches

def diameter_circle := 2 * radius_circle -- diameter of each circle

def side_length_square := 2 * diameter_circle -- side length of the square

-- State the theorem we want to prove
theorem area_of_square : side_length_square ^ 2 = 784 := 
by
  sorry

end area_of_square_l196_196971


namespace x_add_one_greater_than_x_l196_196773

theorem x_add_one_greater_than_x (x : ℝ) : x + 1 > x :=
by
  sorry

end x_add_one_greater_than_x_l196_196773


namespace third_term_is_18_l196_196686

-- Define the first term and the common ratio
def a_1 : ℕ := 2
def q : ℕ := 3

-- Define the function for the nth term of an arithmetic-geometric sequence
def a_n (n : ℕ) : ℕ :=
  a_1 * q^(n-1)

-- Prove that the third term is 18
theorem third_term_is_18 : a_n 3 = 18 := by
  sorry

end third_term_is_18_l196_196686


namespace percentage_increase_is_20_percent_l196_196105

noncomputable def originalSalary : ℝ := 575 / 1.15
noncomputable def increasedSalary : ℝ := 600
noncomputable def percentageIncreaseTo600 : ℝ := (increasedSalary - originalSalary) / originalSalary * 100

theorem percentage_increase_is_20_percent :
  percentageIncreaseTo600 = 20 := 
by
  sorry -- The proof will go here

end percentage_increase_is_20_percent_l196_196105


namespace base_6_four_digit_odd_final_digit_l196_196388

-- Definition of the conditions
def four_digit_number (n b : ℕ) : Prop :=
  b^3 ≤ n ∧ n < b^4

def odd_digit (n b : ℕ) : Prop :=
  (n % b) % 2 = 1

-- Problem statement
theorem base_6_four_digit_odd_final_digit :
  four_digit_number 350 6 ∧ odd_digit 350 6 := by
  sorry

end base_6_four_digit_odd_final_digit_l196_196388


namespace books_selection_l196_196831

theorem books_selection 
  (num_mystery : ℕ)
  (num_fantasy : ℕ)
  (num_biographies : ℕ)
  (Hmystery : num_mystery = 5)
  (Hfantasy : num_fantasy = 4)
  (Hbiographies : num_biographies = 6) :
  (num_mystery * num_fantasy * num_biographies = 120) :=
by
  -- Proof goes here
  sorry

end books_selection_l196_196831


namespace range_of_S_on_ellipse_l196_196898

theorem range_of_S_on_ellipse :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 / 3 = 1) →
    -Real.sqrt 5 ≤ x + y ∧ x + y ≤ Real.sqrt 5 :=
by
  intro x y
  intro h
  sorry

end range_of_S_on_ellipse_l196_196898


namespace part_one_union_sets_l196_196540

theorem part_one_union_sets (a : ℝ) (A B : Set ℝ) :
  (a = 2) →
  A = {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0} →
  B = {x | -2 < x ∧ x < 2} →
  A ∪ B = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end part_one_union_sets_l196_196540


namespace cricketer_average_after_22nd_inning_l196_196738

theorem cricketer_average_after_22nd_inning (A : ℚ) 
  (h1 : 21 * A + 134 = (A + 3.5) * 22)
  (h2 : 57 = A) :
  A + 3.5 = 60.5 :=
by
  exact sorry

end cricketer_average_after_22nd_inning_l196_196738


namespace range_of_a_l196_196724

-- Define sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Mathematical statement to be proven
theorem range_of_a (a : ℝ) : (∃ x, x ∈ set_A ∧ x ∈ set_B a) → a ≥ -1 :=
by
  sorry

end range_of_a_l196_196724


namespace odot_computation_l196_196264

noncomputable def op (a b : ℚ) : ℚ := 
  (a + b) / (1 + a * b)

theorem odot_computation : op 2 (op 3 (op 4 5)) = 7 / 8 := 
  by 
  sorry

end odot_computation_l196_196264


namespace baskets_count_l196_196419

theorem baskets_count (total_apples apples_per_basket : ℕ) (h1 : total_apples = 629) (h2 : apples_per_basket = 17) : (total_apples / apples_per_basket) = 37 :=
by
  sorry

end baskets_count_l196_196419


namespace henrietta_paint_needed_l196_196351

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l196_196351


namespace minimum_x_plus_2y_exists_l196_196129

theorem minimum_x_plus_2y_exists (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) :
  ∃ z : ℝ, z = x + 2 * y ∧ z = -2 * Real.sqrt 2 - 1 :=
sorry

end minimum_x_plus_2y_exists_l196_196129


namespace calculate_M_minus_m_l196_196859

def total_students : ℕ := 2001
def students_studying_spanish (S : ℕ) : Prop := 1601 ≤ S ∧ S ≤ 1700
def students_studying_french (F : ℕ) : Prop := 601 ≤ F ∧ F ≤ 800
def studying_both_languages_lower_bound (S F m : ℕ) : Prop := S + F - m = total_students
def studying_both_languages_upper_bound (S F M : ℕ) : Prop := S + F - M = total_students

theorem calculate_M_minus_m :
  ∀ (S F m M : ℕ),
    students_studying_spanish S →
    students_studying_french F →
    studying_both_languages_lower_bound S F m →
    studying_both_languages_upper_bound S F M →
    S = 1601 ∨ S = 1700 →
    F = 601 ∨ F = 800 →
    M - m = 298 :=
by
  intros S F m M hs hf hl hb Hs Hf
  sorry

end calculate_M_minus_m_l196_196859


namespace g_range_l196_196945

noncomputable def g (x y z : ℝ) : ℝ :=
  (x ^ 2) / (x ^ 2 + y ^ 2) +
  (y ^ 2) / (y ^ 2 + z ^ 2) +
  (z ^ 2) / (z ^ 2 + x ^ 2)

theorem g_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 < g x y z ∧ g x y z < 2 :=
  sorry

end g_range_l196_196945


namespace find_other_number_l196_196570

theorem find_other_number (x y : ℕ) (h1 : x + y = 72) (h2 : y = x + 12) (h3 : y = 42) : x = 30 := by
  sorry

end find_other_number_l196_196570


namespace total_wet_surface_area_l196_196794

def length : ℝ := 8
def width : ℝ := 4
def depth : ℝ := 1.25

theorem total_wet_surface_area : length * width + 2 * (length * depth) + 2 * (width * depth) = 62 :=
by
  sorry

end total_wet_surface_area_l196_196794


namespace condition_for_a_pow_zero_eq_one_l196_196807

theorem condition_for_a_pow_zero_eq_one (a : Real) : a ≠ 0 ↔ a^0 = 1 :=
by
  sorry

end condition_for_a_pow_zero_eq_one_l196_196807


namespace tennis_handshakes_l196_196469

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l196_196469


namespace labor_budget_constraint_l196_196385

-- Define the conditions
def wage_per_carpenter : ℕ := 50
def wage_per_mason : ℕ := 40
def labor_budget : ℕ := 2000
def num_carpenters (x : ℕ) := x
def num_masons (y : ℕ) := y

-- The proof statement
theorem labor_budget_constraint (x y : ℕ) 
    (hx : wage_per_carpenter * num_carpenters x + wage_per_mason * num_masons y ≤ labor_budget) : 
    5 * x + 4 * y ≤ 200 := 
by sorry

end labor_budget_constraint_l196_196385


namespace angle_of_inclination_l196_196840

theorem angle_of_inclination (θ : ℝ) (h_range : 0 ≤ θ ∧ θ < 180)
  (h_line : ∀ x y : ℝ, x + y - 1 = 0 → x = -y + 1) :
  θ = 135 :=
by 
  sorry

end angle_of_inclination_l196_196840


namespace last_day_of_third_quarter_l196_196639

def is_common_year (year: Nat) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0) 

def days_in_month (year: Nat) (month: Nat) : Nat :=
  if month = 2 then 28
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else 31

def last_day_of_month (year: Nat) (month: Nat) : Nat :=
  days_in_month year month

theorem last_day_of_third_quarter (year: Nat) (h : is_common_year year) : last_day_of_month year 9 = 30 :=
by
  sorry

end last_day_of_third_quarter_l196_196639


namespace absolute_value_expression_evaluation_l196_196960

theorem absolute_value_expression_evaluation : abs (-2) * (abs (-Real.sqrt 25) - abs (Real.sin (5 * Real.pi / 2))) = 8 := by
  sorry

end absolute_value_expression_evaluation_l196_196960


namespace divisor_of_1025_l196_196168

theorem divisor_of_1025 : ∃ k : ℕ, 41 * k = 1025 :=
  sorry

end divisor_of_1025_l196_196168


namespace star_computation_l196_196842

def star (x y : ℝ) := x * y - 3 * x + y

theorem star_computation :
  (star 5 8) - (star 8 5) = 12 := by
  sorry

end star_computation_l196_196842


namespace find_x_squared_plus_y_squared_l196_196870

theorem find_x_squared_plus_y_squared (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y + x + y = 17) (h4 : x^2 * y + x * y^2 = 72) : x^2 + y^2 = 65 := 
  sorry

end find_x_squared_plus_y_squared_l196_196870


namespace meal_center_adults_l196_196280

theorem meal_center_adults (cans : ℕ) (children_served : ℕ) (adults_served : ℕ) (total_children : ℕ) 
  (initial_cans : cans = 10) 
  (children_per_can : children_served = 7) 
  (adults_per_can : adults_served = 4) 
  (children_to_feed : total_children = 21) : 
  (cans - (total_children / children_served)) * adults_served = 28 := by
  have h1: 3 = total_children / children_served := by
    sorry
  have h2: 7 = cans - 3 := by
    sorry
  have h3: 28 = 7 * adults_served := by
    sorry
  have h4: adults_served = 4 := by
    sorry
  sorry

end meal_center_adults_l196_196280


namespace Mrs_Hilt_remaining_money_l196_196002

theorem Mrs_Hilt_remaining_money :
  let initial_amount : ℝ := 3.75
  let pencil_cost : ℝ := 1.15
  let eraser_cost : ℝ := 0.85
  let notebook_cost : ℝ := 2.25
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -0.50 :=
by
  sorry

end Mrs_Hilt_remaining_money_l196_196002


namespace find_integer_solutions_l196_196173

noncomputable def integer_solutions (x y z w : ℤ) : Prop :=
  x * y * z / w + y * z * w / x + z * w * x / y + w * x * y / z = 4

theorem find_integer_solutions :
  { (x, y, z, w) : ℤ × ℤ × ℤ × ℤ |
    integer_solutions x y z w } =
  {(1,1,1,1), (-1,-1,-1,-1), (-1,-1,1,1), (-1,1,-1,1),
   (-1,1,1,-1), (1,-1,-1,1), (1,-1,1,-1), (1,1,-1,-1)} :=
by
  sorry

end find_integer_solutions_l196_196173


namespace no_integer_polynomial_exists_l196_196141

theorem no_integer_polynomial_exists 
    (a b c d : ℤ) (h : a ≠ 0) (P : ℤ → ℤ) 
    (h1 : ∀ x, P x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    (h2 : P 4 = 1) (h3 : P 7 = 2) : 
    false := 
by
    sorry

end no_integer_polynomial_exists_l196_196141


namespace calculateSurfaceArea_l196_196178

noncomputable def totalSurfaceArea (r : ℝ) : ℝ :=
  let hemisphereCurvedArea := 2 * Real.pi * r^2
  let cylinderLateralArea := 2 * Real.pi * r * r
  hemisphereCurvedArea + cylinderLateralArea

theorem calculateSurfaceArea :
  ∃ r : ℝ, (Real.pi * r^2 = 144 * Real.pi) ∧ totalSurfaceArea r = 576 * Real.pi :=
by
  exists 12
  constructor
  . sorry -- Proof that 144π = π*12^2 can be shown
  . sorry -- Proof that 576π = 288π + 288π can be shown

end calculateSurfaceArea_l196_196178


namespace giyoon_above_average_subjects_l196_196571

def points_korean : ℕ := 80
def points_mathematics : ℕ := 94
def points_social_studies : ℕ := 82
def points_english : ℕ := 76
def points_science : ℕ := 100
def number_of_subjects : ℕ := 5

def total_points : ℕ := points_korean + points_mathematics + points_social_studies + points_english + points_science
def average_points : ℚ := total_points / number_of_subjects

def count_above_average_points : ℕ := 
  (if points_korean > average_points then 1 else 0) + 
  (if points_mathematics > average_points then 1 else 0) +
  (if points_social_studies > average_points then 1 else 0) +
  (if points_english > average_points then 1 else 0) +
  (if points_science > average_points then 1 else 0)

theorem giyoon_above_average_subjects : count_above_average_points = 2 := by
  sorry

end giyoon_above_average_subjects_l196_196571


namespace determine_positive_intervals_l196_196341

noncomputable def positive_intervals (x : ℝ) : Prop :=
  (x+1) * (x-1) * (x-3) > 0

theorem determine_positive_intervals :
  ∀ x : ℝ, (positive_intervals x ↔ (x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∨ x ∈ Set.Ioi (3 : ℝ))) :=
by
  sorry

end determine_positive_intervals_l196_196341


namespace solve_for_m_l196_196713

theorem solve_for_m (x m : ℝ) (h1 : 2 * 1 - m = -3) : m = 5 :=
by
  sorry

end solve_for_m_l196_196713


namespace remainder_is_correct_l196_196980

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct_l196_196980


namespace finite_decimal_representation_nat_numbers_l196_196488

theorem finite_decimal_representation_nat_numbers (n : ℕ) : 
  (∀ k : ℕ, k < n → (∃ u v : ℕ, (k + 1 = 2^u ∨ k + 1 = 5^v) ∨ (k - 1 = 2^u ∨ k -1  = 5^v))) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 6) :=
by sorry

end finite_decimal_representation_nat_numbers_l196_196488


namespace numbers_with_digit_one_are_more_numerous_l196_196572

theorem numbers_with_digit_one_are_more_numerous :
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  total_numbers - numbers_without_one > numbers_without_one :=
by
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  show total_numbers - numbers_without_one > numbers_without_one
  sorry

end numbers_with_digit_one_are_more_numerous_l196_196572


namespace train_passes_tree_in_16_seconds_l196_196894

noncomputable def time_to_pass_tree (length_train : ℕ) (speed_train_kmh : ℕ) : ℕ :=
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  length_train / speed_train_ms

theorem train_passes_tree_in_16_seconds :
  time_to_pass_tree 280 63 = 16 :=
  by
    sorry

end train_passes_tree_in_16_seconds_l196_196894


namespace negation_of_p_l196_196255

variable {x : ℝ}

def proposition_p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

theorem negation_of_p :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
sorry

end negation_of_p_l196_196255


namespace circle_equation_l196_196154

theorem circle_equation 
  (x y : ℝ)
  (center : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (line1 : ℝ × ℝ → Prop)
  (line2 : ℝ × ℝ → Prop)
  (hx : line1 center)
  (hy : line2 tangent_point)
  (tangent_point_val : tangent_point = (2, -1))
  (line1_def : ∀ (p : ℝ × ℝ), line1 p ↔ 2 * p.1 + p.2 = 0)
  (line2_def : ∀ (p : ℝ × ℝ), line2 p ↔ p.1 + p.2 - 1 = 0) :
  (∃ (x0 y0 r : ℝ), center = (x0, y0) ∧ r > 0 ∧ (x - x0)^2 + (y - y0)^2 = r^2 ∧ 
                        (x - x0)^2 + (y - y0)^2 = (x - 1)^2 + (y + 2)^2 ∧ 
                        (x - 1)^2 + (y + 2)^2 = 2) :=
by {
  sorry
}

end circle_equation_l196_196154


namespace volume_of_one_slice_l196_196363

theorem volume_of_one_slice
  (circumference : ℝ)
  (c : circumference = 18 * Real.pi):
  ∃ V, V = 162 * Real.pi :=
by sorry

end volume_of_one_slice_l196_196363


namespace distance_between_A_and_B_l196_196451

variable (d : ℝ) -- Total distance between A and B

def car_speeds (vA vB t : ℝ) : Prop :=
vA = 80 ∧ vB = 100 ∧ t = 2

def total_covered_distance (vA vB t : ℝ) : ℝ :=
(vA + vB) * t

def percentage_distance (total_distance covered_distance : ℝ) : Prop :=
0.6 * total_distance = covered_distance

theorem distance_between_A_and_B (vA vB t : ℝ) (H1 : car_speeds vA vB t) 
  (H2 : percentage_distance d (total_covered_distance vA vB t)) : d = 600 := by
  sorry

end distance_between_A_and_B_l196_196451


namespace remainder_mul_three_division_l196_196166

theorem remainder_mul_three_division
    (N : ℤ) (k : ℤ)
    (h1 : N = 1927 * k + 131) :
    ((3 * N) % 43) = 6 :=
by
  sorry

end remainder_mul_three_division_l196_196166


namespace clock_ticks_6_times_at_6_oclock_l196_196887

theorem clock_ticks_6_times_at_6_oclock
  (h6 : 5 * t = 25)
  (h12 : 11 * t = 55) :
  t = 5 ∧ 6 = 6 :=
by
  sorry

end clock_ticks_6_times_at_6_oclock_l196_196887


namespace print_time_l196_196367

-- Conditions
def printer_pages_per_minute : ℕ := 25
def total_pages : ℕ := 350

-- Theorem
theorem print_time :
  (total_pages / printer_pages_per_minute : ℕ) = 14 :=
by sorry

end print_time_l196_196367


namespace problem_1_problem_2_l196_196434

def f (x : ℝ) : ℝ := |x - 3| - 2
def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem problem_1:
  { x : ℝ // 0 ≤ x ∧ x ≤ 6 } = { x : ℝ // f x ≤ 1 } :=
sorry

theorem problem_2:
  { m : ℝ // m ≤ -3 } = { m : ℝ // ∀ x : ℝ, f x - g x ≥ m + 1 } :=
sorry

end problem_1_problem_2_l196_196434


namespace gcd_360_504_is_72_l196_196942

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l196_196942


namespace geometric_sequence_S12_l196_196251

theorem geometric_sequence_S12 (S : ℕ → ℝ) (S_4_eq : S 4 = 20) (S_8_eq : S 8 = 30) :
  S 12 = 35 :=
by
  sorry

end geometric_sequence_S12_l196_196251


namespace largest_number_in_set_l196_196531

theorem largest_number_in_set (b : ℕ) (h₀ : 2 + 6 + b = 18) (h₁ : 2 ≤ 6 ∧ 6 ≤ b):
  b = 10 :=
sorry

end largest_number_in_set_l196_196531


namespace solutions_in_nat_solutions_in_non_neg_int_l196_196977

-- Definitions for Part A
def nat_sol_count (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem solutions_in_nat (x1 x2 x3 : ℕ) : 
  (x1 > 0) → (x2 > 0) → (x3 > 0) → (x1 + x2 + x3 = 1000) → 
  nat_sol_count 997 3 = Nat.choose 999 2 := sorry

-- Definitions for Part B
theorem solutions_in_non_neg_int (x1 x2 x3 : ℕ) : 
  (x1 + x2 + x3 = 1000) → 
  nat_sol_count 1000 3 = Nat.choose 1002 2 := sorry

end solutions_in_nat_solutions_in_non_neg_int_l196_196977


namespace return_kittens_due_to_rehoming_problems_l196_196529

def num_breeding_rabbits : Nat := 10
def kittens_first_spring : Nat := num_breeding_rabbits * num_breeding_rabbits
def kittens_adopted_first_spring : Nat := kittens_first_spring / 2
def kittens_second_spring : Nat := 60
def kittens_adopted_second_spring : Nat := 4
def total_rabbits : Nat := 121

def non_breeding_rabbits_from_first_spring : Nat :=
  total_rabbits - num_breeding_rabbits - kittens_second_spring

def kittens_returned_to_lola : Prop :=
  non_breeding_rabbits_from_first_spring - kittens_adopted_first_spring = 1

theorem return_kittens_due_to_rehoming_problems : kittens_returned_to_lola :=
sorry

end return_kittens_due_to_rehoming_problems_l196_196529


namespace ten_fact_minus_nine_fact_l196_196826

-- Definitions corresponding to the conditions
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Condition for 9!
def nine_factorial : ℕ := 362880

-- 10! can be expressed in terms of 9!
noncomputable def ten_factorial : ℕ := 10 * nine_factorial

-- Proof statement we need to show
theorem ten_fact_minus_nine_fact : ten_factorial - nine_factorial = 3265920 :=
by
  unfold ten_factorial
  unfold nine_factorial
  sorry

end ten_fact_minus_nine_fact_l196_196826


namespace find_first_term_l196_196927

theorem find_first_term
  (S : ℝ) (a r : ℝ)
  (h1 : S = 10)
  (h2 : a + a * r = 6)
  (h3 : a = 2 * r) :
  a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13 := by
  sorry

end find_first_term_l196_196927


namespace product_increases_exactly_13_times_by_subtracting_3_l196_196678

theorem product_increases_exactly_13_times_by_subtracting_3 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    13 * (n1 * n2 * n3 * n4 * n5 * n6 * n7) =
      ((n1 - 3) * (n2 - 3) * (n3 - 3) * (n4 - 3) * (n5 - 3) * (n6 - 3) * (n7 - 3)) :=
sorry

end product_increases_exactly_13_times_by_subtracting_3_l196_196678


namespace students_taking_both_chorus_and_band_l196_196743

theorem students_taking_both_chorus_and_band (total_students : ℕ) 
                                             (chorus_students : ℕ)
                                             (band_students : ℕ)
                                             (not_enrolled_students : ℕ) : 
                                             total_students = 50 ∧
                                             chorus_students = 18 ∧
                                             band_students = 26 ∧
                                             not_enrolled_students = 8 →
                                             ∃ (both_chorus_and_band : ℕ), both_chorus_and_band = 2 :=
by
  intros h
  sorry

end students_taking_both_chorus_and_band_l196_196743


namespace sum_of_two_consecutive_negative_integers_l196_196638

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 812) (h_neg : n < 0 ∧ (n + 1) < 0) : 
  n + (n + 1) = -57 :=
sorry

end sum_of_two_consecutive_negative_integers_l196_196638


namespace difference_of_squares_l196_196140

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x + y = 15
def condition2 : Prop := x - y = 10

-- Goal to prove
theorem difference_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 150 := 
by sorry

end difference_of_squares_l196_196140


namespace cooking_time_per_side_l196_196689

-- Defining the problem conditions
def total_guests : ℕ := 30
def guests_wanting_2_burgers : ℕ := total_guests / 2
def guests_wanting_1_burger : ℕ := total_guests / 2
def burgers_per_guest_2 : ℕ := 2
def burgers_per_guest_1 : ℕ := 1
def total_burgers : ℕ := guests_wanting_2_burgers * burgers_per_guest_2 + guests_wanting_1_burger * burgers_per_guest_1
def burgers_per_batch : ℕ := 5
def total_batches : ℕ := total_burgers / burgers_per_batch
def total_cooking_time : ℕ := 72
def time_per_batch : ℕ := total_cooking_time / total_batches
def sides_per_burger : ℕ := 2

-- the theorem to prove the desired cooking time per side
theorem cooking_time_per_side : (time_per_batch / sides_per_burger) = 4 := by {
    -- Here we would enter the proof steps, but this is omitted as per the instructions.
    sorry
}

end cooking_time_per_side_l196_196689


namespace females_with_advanced_degrees_l196_196777

noncomputable def total_employees := 200
noncomputable def total_females := 120
noncomputable def total_advanced_degrees := 100
noncomputable def males_college_degree_only := 40

theorem females_with_advanced_degrees :
  (total_employees - total_females) - males_college_degree_only = 
  total_employees - total_females - males_college_degree_only ∧ 
  total_females = 120 ∧ 
  total_advanced_degrees = 100 ∧ 
  total_employees = 200 ∧ 
  males_college_degree_only = 40 ∧
  total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60 :=
sorry

end females_with_advanced_degrees_l196_196777


namespace ratio_of_doctors_lawyers_engineers_l196_196377

variables (d l e : ℕ)

-- Conditions
def average_age_per_group (d l e : ℕ) : Prop :=
  (40 * d + 55 * l + 35 * e) = 45 * (d + l + e)

-- Theorem
theorem ratio_of_doctors_lawyers_engineers
  (h : average_age_per_group d l e) :
  l = d + 2 * e :=
by sorry

end ratio_of_doctors_lawyers_engineers_l196_196377


namespace village_population_l196_196033

theorem village_population (P : ℝ) (h : 0.9 * P = 45000) : P = 50000 :=
by
  sorry

end village_population_l196_196033


namespace find_p_l196_196815

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem find_p : ∃ p : ℝ, f (f (f (f p))) = -4 :=
by
  sorry

end find_p_l196_196815


namespace five_diff_numbers_difference_l196_196747

theorem five_diff_numbers_difference (S : Finset ℕ) (hS_size : S.card = 5) 
    (hS_range : ∀ x ∈ S, x ≤ 10) : 
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d ∧ a - b ≠ 0 :=
by
  sorry

end five_diff_numbers_difference_l196_196747


namespace sequence_sum_100_eq_200_l196_196594

theorem sequence_sum_100_eq_200
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (h4 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1)
  (h5 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)) :
  (Finset.range 100).sum (a ∘ Nat.succ) = 200 := by
  sorry

end sequence_sum_100_eq_200_l196_196594


namespace nonneg_real_inequality_l196_196428

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 := 
by
  sorry

end nonneg_real_inequality_l196_196428


namespace percent_of_pizza_not_crust_l196_196188

theorem percent_of_pizza_not_crust (total_weight crust_weight : ℝ) (h_total : total_weight = 800) (h_crust : crust_weight = 200) :
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end percent_of_pizza_not_crust_l196_196188


namespace jason_average_messages_l196_196349

theorem jason_average_messages : 
    let monday := 220
    let tuesday := monday / 2
    let wednesday := 50
    let thursday := 50
    let friday := 50
    let total_messages := monday + tuesday + wednesday + thursday + friday
    let average_messages := total_messages / 5
    average_messages = 96 :=
by
  let monday := 220
  let tuesday := monday / 2
  let wednesday := 50
  let thursday := 50
  let friday := 50
  let total_messages := monday + tuesday + wednesday + thursday + friday
  let average_messages := total_messages / 5
  have h : average_messages = 96 := sorry
  exact h

end jason_average_messages_l196_196349


namespace min_value_b1_b2_l196_196350

noncomputable def seq (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_value_b1_b2 (b : ℕ → ℕ)
  (h_pos : ∀ n, b n > 0)
  (h_seq : seq b) :
  b 1 + b 2 = 2018 := sorry

end min_value_b1_b2_l196_196350


namespace plot_area_is_nine_hectares_l196_196227

-- Definition of the dimensions of the plot
def length := 450
def width := 200

-- Definition of conversion factor from square meters to hectares
def sqMetersPerHectare := 10000

-- Calculated area in hectares
def area_hectares := (length * width) / sqMetersPerHectare

-- Theorem statement: prove that the area in hectares is 9
theorem plot_area_is_nine_hectares : area_hectares = 9 := 
by
  sorry

end plot_area_is_nine_hectares_l196_196227


namespace max_mogs_l196_196289

theorem max_mogs : ∃ x y z : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 3 * x + 4 * y + 8 * z = 100 ∧ z = 10 :=
by
  sorry

end max_mogs_l196_196289


namespace find_threedigit_number_l196_196206

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l196_196206


namespace sum_real_roots_eq_neg4_l196_196522

-- Define the equation condition
def equation_condition (x : ℝ) : Prop :=
  (2 * x / (x^2 + 5 * x + 3) + 3 * x / (x^2 + x + 3) = 1)

-- Define the statement that sums the real roots
theorem sum_real_roots_eq_neg4 : 
  ∃ S : ℝ, (∀ x : ℝ, equation_condition x → x = -1 ∨ x = -3) ∧ (S = -4) :=
sorry

end sum_real_roots_eq_neg4_l196_196522


namespace ball_bounce_height_l196_196139

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (hₖ : ℕ → ℝ) :
  h₀ = 500 ∧ r = 0.6 ∧ (∀ k, hₖ k = h₀ * r^k) → 
  ∃ k, hₖ k < 3 ∧ k ≥ 22 := 
by
  sorry

end ball_bounce_height_l196_196139


namespace problem_f_of_f_neg1_eq_neg1_l196_196215

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- State the proposition to be proved
theorem problem_f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := by
  sorry

end problem_f_of_f_neg1_eq_neg1_l196_196215


namespace unique_geometric_progression_12_a_b_ab_l196_196969

noncomputable def geometric_progression_12_a_b_ab : Prop :=
  ∃ (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3

theorem unique_geometric_progression_12_a_b_ab :
  ∃! (a b : ℝ), ∃ r : ℝ, a = 12 * r ∧ b = 12 * r^2 ∧ 12 * r * (12 * r^2) = 144 * r^3 :=
by
  sorry

end unique_geometric_progression_12_a_b_ab_l196_196969


namespace investment_total_correct_l196_196412

-- Define the initial investment, interest rate, and duration
def initial_investment : ℝ := 300
def monthly_interest_rate : ℝ := 0.10
def duration_in_months : ℝ := 2

-- Define the total amount after 2 months
noncomputable def total_after_two_months : ℝ := initial_investment * (1 + monthly_interest_rate) * (1 + monthly_interest_rate)

-- Define the correct answer
def correct_answer : ℝ := 363

-- The proof problem
theorem investment_total_correct :
  total_after_two_months = correct_answer :=
sorry

end investment_total_correct_l196_196412


namespace even_gt_one_square_gt_l196_196080

theorem even_gt_one_square_gt (m : ℕ) (h_even : ∃ k : ℕ, m = 2 * k) (h_gt_one : m > 1) : m < m * m :=
by
  sorry

end even_gt_one_square_gt_l196_196080


namespace last_digit_m_is_9_l196_196550

def x (n : ℕ) : ℕ := 2^(2^n) + 1

def m : ℕ := List.foldr Nat.lcm 1 (List.map x (List.range' 2 (1971 - 2 + 1)))

theorem last_digit_m_is_9 : m % 10 = 9 :=
  by
    sorry

end last_digit_m_is_9_l196_196550


namespace alex_wins_if_picks_two_l196_196004

theorem alex_wins_if_picks_two (matches_left : ℕ) (alex_picks bob_picks : ℕ) :
  matches_left = 30 →
  1 ≤ alex_picks ∧ alex_picks ≤ 6 →
  1 ≤ bob_picks ∧ bob_picks ≤ 6 →
  alex_picks = 2 →
  (∀ n, (n % 7 ≠ 0) → ¬ (∃ k, matches_left - k ≤ 0 ∧ (matches_left - k) % 7 = 0)) :=
by sorry

end alex_wins_if_picks_two_l196_196004


namespace calculate_expression_l196_196897

theorem calculate_expression :
  (Real.sqrt 2 - 3)^0 - Real.sqrt 9 + |(-2: ℝ)| + ((-1/3: ℝ)⁻¹)^2 = 9 :=
by
  sorry

end calculate_expression_l196_196897


namespace polygon_area_correct_l196_196328

-- Define the coordinates of the vertices
def vertex1 := (2, 1)
def vertex2 := (4, 3)
def vertex3 := (6, 1)
def vertex4 := (4, 6)

-- Define a function to calculate the area using the Shoelace Theorem
noncomputable def shoelace_area (vertices : List (ℕ × ℕ)) : ℚ :=
  let xys := vertices ++ [vertices.head!]
  let sum1 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => x1 * y2)
  let sum2 := (xys.zip (xys.tail!)).map (fun ((x1, y1), (x2, y2)) => y1 * x2)
  (sum1.sum - sum2.sum : ℚ) / 2

-- Instantiate the specific vertices
def polygon := [vertex1, vertex2, vertex3, vertex4]

-- The theorem statement
theorem polygon_area_correct : shoelace_area polygon = 6 := by
  sorry

end polygon_area_correct_l196_196328


namespace hannahs_vegetarian_restaurant_l196_196625

theorem hannahs_vegetarian_restaurant :
  let total_weight_of_peppers := 0.6666666666666666
  let weight_of_green_peppers := 0.3333333333333333
  total_weight_of_peppers - weight_of_green_peppers = 0.3333333333333333 :=
by
  sorry

end hannahs_vegetarian_restaurant_l196_196625


namespace construction_company_sand_weight_l196_196148

theorem construction_company_sand_weight :
  ∀ (total_weight gravel_weight : ℝ), total_weight = 14.02 → gravel_weight = 5.91 → 
  total_weight - gravel_weight = 8.11 :=
by 
  intros total_weight gravel_weight h_total h_gravel 
  sorry

end construction_company_sand_weight_l196_196148


namespace f_zero_f_odd_f_range_l196_196028

-- Condition 1: The function f is defined on ℝ
-- Condition 2: f(x + y) = f(x) + f(y)
-- Condition 3: f(1/3) = 1
-- Condition 4: f(x) < 0 when x > 0

variables (f : ℝ → ℝ)
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_third : f (1/3) = 1
axiom f_neg_positive : ∀ x : ℝ, 0 < x → f x < 0

-- Question 1: Find the value of f(0)
theorem f_zero : f 0 = 0 := sorry

-- Question 2: Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

-- Question 3: Find the range of x where f(x) + f(2 + x) < 2
theorem f_range : ∀ x : ℝ, f x + f (2 + x) < 2 → -2/3 < x := sorry

end f_zero_f_odd_f_range_l196_196028


namespace min_abc_value_l196_196285

noncomputable def minValue (a b c : ℝ) : ℝ := (a + b) / (a * b * c)

theorem min_abc_value (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
  (minValue a b c) ≥ 16 :=
by
  sorry

end min_abc_value_l196_196285


namespace trigonometric_identity_l196_196012

open Real

theorem trigonometric_identity
  (theta : ℝ)
  (h : cos (π / 6 - theta) = 2 * sqrt 2 / 3) : 
  cos (π / 3 + theta) = 1 / 3 ∨ cos (π / 3 + theta) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l196_196012


namespace exponents_subtraction_l196_196219

theorem exponents_subtraction (m n : ℕ) (hm : 3 ^ m = 8) (hn : 3 ^ n = 2) : 3 ^ (m - n) = 4 := 
by
  sorry

end exponents_subtraction_l196_196219


namespace arithmetic_sequence_general_formula_and_sum_max_l196_196958

theorem arithmetic_sequence_general_formula_and_sum_max :
  ∀ (a : ℕ → ℤ), 
  (a 7 = -8) → (a 17 = -28) → 
  (∀ n, a n = -2 * n + 6) ∧ 
  (∀ S : ℕ → ℤ, (∀ n, S n = -n^2 + 5 * n) → ∀ n, S n ≤ 6) :=
by
  sorry

end arithmetic_sequence_general_formula_and_sum_max_l196_196958


namespace midpoint_of_polar_line_segment_l196_196273

theorem midpoint_of_polar_line_segment
  (r θ : ℝ)
  (hr : r > 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (hA : ∃ A, A = (8, 5 * Real.pi / 12))
  (hB : ∃ B, B = (8, -3 * Real.pi / 12)) :
  (r, θ) = (4, Real.pi / 12) := 
sorry

end midpoint_of_polar_line_segment_l196_196273


namespace at_least_two_equal_l196_196023

noncomputable def positive_reals (x y z : ℝ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0

noncomputable def triangle_inequality_for_n (x y z : ℝ) (n : ℕ) : Prop :=
(x^n + y^n > z^n) ∧ (y^n + z^n > x^n) ∧ (z^n + x^n > y^n)

theorem at_least_two_equal (x y z : ℝ) 
  (pos : positive_reals x y z) 
  (triangle_ineq: ∀ n : ℕ, n > 0 → triangle_inequality_for_n x y z n) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l196_196023


namespace original_plan_trees_per_day_l196_196735

theorem original_plan_trees_per_day (x : ℕ) :
  (∃ x, (960 / x - 960 / (2 * x) = 4)) → x = 120 := 
sorry

end original_plan_trees_per_day_l196_196735


namespace general_term_formula_l196_196225

theorem general_term_formula (a S : ℕ → ℝ) (h : ∀ n, S n = (2 / 3) * a n + (1 / 3)) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = -2 * a (n - 1)) →
  ∀ n, a n = (-2)^(n - 1) :=
by
  sorry

end general_term_formula_l196_196225


namespace quadratic_inequality_ab_l196_196845

theorem quadratic_inequality_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + 1 > 0) ↔ -1 < x ∧ x < 1 / 3) :
  a * b = -6 :=
by
  -- Proof is omitted
  sorry

end quadratic_inequality_ab_l196_196845


namespace x_plus_y_eq_20_l196_196796

theorem x_plus_y_eq_20 (x y : ℝ) (hxy : x ≠ y) (hdet : (Matrix.det ![
  ![2, 3, 7],
  ![4, x, y],
  ![4, y, x]]) = 0) : x + y = 20 :=
by
  sorry

end x_plus_y_eq_20_l196_196796


namespace value_of_m_l196_196142

theorem value_of_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∃ (k : ℝ), (2 * m - 1) * x ^ (m ^ 2) = k * x ^ n) → m = 1 :=
by
  sorry

end value_of_m_l196_196142


namespace determinant_matrix_equivalence_l196_196852

variable {R : Type} [CommRing R]

theorem determinant_matrix_equivalence
  (x y z w : R)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end determinant_matrix_equivalence_l196_196852


namespace distance_ratio_l196_196187

-- Define the distances as given in the conditions
def distance_from_city_sky_falls := 8 -- Distance in miles
def distance_from_city_rocky_mist := 400 -- Distance in miles

theorem distance_ratio : distance_from_city_rocky_mist / distance_from_city_sky_falls = 50 := 
by
  -- Proof skipped
  sorry

end distance_ratio_l196_196187


namespace gcd_of_12547_23791_l196_196909

theorem gcd_of_12547_23791 : Nat.gcd 12547 23791 = 1 :=
by
  sorry

end gcd_of_12547_23791_l196_196909


namespace average_speed_of_journey_is_24_l196_196115

noncomputable def average_speed (D : ℝ) (speed_to_office speed_to_home : ℝ) : ℝ :=
  let time_to_office := D / speed_to_office
  let time_to_home := D / speed_to_home
  let total_distance := 2 * D
  let total_time := time_to_office + time_to_home
  total_distance / total_time

theorem average_speed_of_journey_is_24 (D : ℝ) : average_speed D 20 30 = 24 := by
  -- nonconstructive proof to fulfill theorem definition
  sorry

end average_speed_of_journey_is_24_l196_196115


namespace samantha_coins_worth_l196_196946

-- Define the conditions and the final question with an expected answer.
theorem samantha_coins_worth (n d : ℕ) (h1 : n + d = 30)
  (h2 : 10 * n + 5 * d = 5 * n + 10 * d + 120) :
  (5 * n + 10 * d) = 165 := 
sorry

end samantha_coins_worth_l196_196946


namespace roots_reciprocal_sum_eq_25_l196_196824

theorem roots_reciprocal_sum_eq_25 (p q r : ℝ) (hpq : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (hroot : ∀ x, x^3 - 9*x^2 + 8*x + 2 = 0 → (x = p ∨ x = q ∨ x = r)) :
  1/p^2 + 1/q^2 + 1/r^2 = 25 :=
by sorry

end roots_reciprocal_sum_eq_25_l196_196824


namespace final_combined_price_correct_l196_196151

theorem final_combined_price_correct :
  let i_p := 1000
  let d_1 := 0.10
  let d_2 := 0.20
  let t_1 := 0.08
  let t_2 := 0.06
  let s_p := 30
  let c_p := 50
  let t_a := 0.05
  let price_after_first_month := i_p * (1 - d_1) * (1 + t_1)
  let price_after_second_month := price_after_first_month * (1 - d_2) * (1 + t_2)
  let screen_protector_final := s_p * (1 + t_a)
  let case_final := c_p * (1 + t_a)
  price_after_second_month + screen_protector_final + case_final = 908.256 := by
  sorry  -- Proof not required

end final_combined_price_correct_l196_196151


namespace number_of_technicians_l196_196618

-- Define the problem statements
variables (T R : ℕ)

-- Conditions based on the problem description
def condition1 : Prop := T + R = 42
def condition2 : Prop := 3 * T + R = 56

-- The main goal to prove
theorem number_of_technicians (h1 : condition1 T R) (h2 : condition2 T R) : T = 7 :=
by
  sorry -- Proof is omitted as per instructions

end number_of_technicians_l196_196618


namespace odd_nat_numbers_eq_1_l196_196271

-- Definitions of conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem odd_nat_numbers_eq_1
  (a b c d : ℕ)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : is_odd a) (h5 : is_odd b) (h6 : is_odd c) (h7 : is_odd d)
  (h8 : a * d = b * c)
  (h9 : is_power_of_two (a + d))
  (h10 : is_power_of_two (b + c)) :
  a = 1 :=
sorry

end odd_nat_numbers_eq_1_l196_196271


namespace unfenced_side_length_l196_196299

-- Define the conditions
variables (L W : ℝ)
axiom area_condition : L * W = 480
axiom fence_condition : 2 * W + L = 64

-- Prove the unfenced side of the yard (L) is 40 feet
theorem unfenced_side_length : L = 40 :=
by
  -- Conditions, definitions, and properties go here.
  -- But we leave the proof as a placeholder since the statement is sufficient.
  sorry

end unfenced_side_length_l196_196299


namespace part1_l196_196096

variable {a b : ℝ}
variable {A B C : ℝ}
variable {S : ℝ}

-- Given Conditions
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (b * Real.cos C - c * Real.cos B = 2 * a) ∧ (c = a)

-- To prove
theorem part1 (h : is_triangle A B C a b a) : B = 2 * Real.pi / 3 := sorry

end part1_l196_196096


namespace carmen_more_miles_l196_196619

-- Definitions for the conditions
def carmen_distance : ℕ := 90
def daniel_distance : ℕ := 75

-- The theorem statement
theorem carmen_more_miles : carmen_distance - daniel_distance = 15 :=
by
  sorry

end carmen_more_miles_l196_196619


namespace three_gt_sqrt_seven_l196_196551

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_gt_sqrt_seven_l196_196551


namespace find_constant_b_l196_196226

variable (x : ℝ)
variable (b d e : ℝ)

theorem find_constant_b   
  (h1 : (7 * x ^ 2 - 2 * x + 4 / 3) * (d * x ^ 2 + b * x + e) = 28 * x ^ 4 - 10 * x ^ 3 + 18 * x ^ 2 - 8 * x + 5 / 3)
  (h2 : d = 4) : 
  b = -2 / 7 := 
sorry

end find_constant_b_l196_196226


namespace chef_potatoes_l196_196282

theorem chef_potatoes (total_potatoes cooked_potatoes time_per_potato rest_time: ℕ)
  (h1 : total_potatoes = 15)
  (h2 : time_per_potato = 9)
  (h3 : rest_time = 63)
  (h4 : time_per_potato * (total_potatoes - cooked_potatoes) = rest_time) :
  cooked_potatoes = 8 :=
by sorry

end chef_potatoes_l196_196282


namespace maximize_greenhouse_planting_area_l196_196536

theorem maximize_greenhouse_planting_area
    (a b : ℝ)
    (h : a * b = 800)
    (planting_area : ℝ := (a - 4) * (b - 2)) :
  (a = 40 ∧ b = 20) ↔ planting_area = 648 :=
by
  sorry

end maximize_greenhouse_planting_area_l196_196536


namespace remainder_of_7n_mod_4_l196_196025

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l196_196025


namespace sum_digits_18_to_21_l196_196997

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_18_to_21 :
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 :=
by
  sorry

end sum_digits_18_to_21_l196_196997


namespace distance_proof_l196_196296

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l196_196296


namespace production_in_three_minutes_l196_196609

noncomputable def production_rate_per_machine (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

noncomputable def production_per_minute (machines : ℕ) (rate_per_machine : ℕ) : ℕ :=
  machines * rate_per_machine

noncomputable def total_production (production_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  production_per_minute * minutes

theorem production_in_three_minutes :
  ∀ (total_bottles : ℕ) (num_machines : ℕ) (machines : ℕ) (minutes : ℕ),
  total_bottles = 16 → num_machines = 4 → machines = 8 → minutes = 3 →
  total_production (production_per_minute machines (production_rate_per_machine total_bottles num_machines)) minutes = 96 :=
by
  intros total_bottles num_machines machines minutes h_total_bottles h_num_machines h_machines h_minutes
  sorry

end production_in_three_minutes_l196_196609


namespace average_percentage_increase_is_correct_l196_196692

def initial_prices : List ℝ := [300, 450, 600]
def price_increases : List ℝ := [0.10, 0.15, 0.20]

noncomputable def total_original_price : ℝ :=
  initial_prices.sum

noncomputable def total_new_price : ℝ :=
  (List.zipWith (λ p i => p * (1 + i)) initial_prices price_increases).sum

noncomputable def total_price_increase : ℝ :=
  total_new_price - total_original_price

noncomputable def average_percentage_increase : ℝ :=
  (total_price_increase / total_original_price) * 100

theorem average_percentage_increase_is_correct :
  average_percentage_increase = 16.11 := by
  sorry

end average_percentage_increase_is_correct_l196_196692


namespace fraction_meaningful_range_l196_196819

theorem fraction_meaningful_range (x : ℝ) : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l196_196819


namespace problem1_correct_solution_problem2_correct_solution_l196_196335

noncomputable def g (x a : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

/-- 
    Prove that the set {x | -2/3 ≤ x ≤ 2} satisfies g(x) ≤ 4 when a = 3 
--/
theorem problem1_correct_solution (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

noncomputable def f (x a : ℝ) : ℝ := g (x - 2) a

/-- 
    Prove that the range of a such that f(x) ≥ 1 for all x ∈ ℝ 
    is a ≤ 1 or a ≥ 3
--/
theorem problem2_correct_solution (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end problem1_correct_solution_problem2_correct_solution_l196_196335


namespace circus_juggling_l196_196460

theorem circus_juggling (jugglers : ℕ) (balls_per_juggler : ℕ) (total_balls : ℕ)
  (h1 : jugglers = 5000)
  (h2 : balls_per_juggler = 12)
  (h3 : total_balls = jugglers * balls_per_juggler) :
  total_balls = 60000 :=
by
  rw [h1, h2] at h3
  exact h3

end circus_juggling_l196_196460


namespace melissa_total_score_l196_196863

theorem melissa_total_score (games : ℕ) (points_per_game : ℕ) 
  (h_games : games = 3) (h_points_per_game : points_per_game = 27) : 
  points_per_game * games = 81 := 
by 
  sorry

end melissa_total_score_l196_196863


namespace evaluate_g_ggg_neg1_l196_196093

def g (y : ℤ) : ℤ := y^3 - 3*y + 1

theorem evaluate_g_ggg_neg1 : g (g (g (-1))) = 6803 := 
by
  sorry

end evaluate_g_ggg_neg1_l196_196093


namespace solve_g_l196_196760

def g (a b : ℚ) : ℚ :=
if a + b ≤ 4 then (a * b - 2 * a + 3) / (3 * a)
else (a * b - 3 * b - 1) / (-3 * b)

theorem solve_g :
  g 3 1 + g 1 5 = 11 / 15 :=
by
  -- Here we just set up the theorem statement. Proof is not included.
  sorry

end solve_g_l196_196760


namespace geometric_sequence_a4_range_l196_196911

theorem geometric_sequence_a4_range
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : 0 < a 1 ∧ a 1 < 1)
  (h2 : 1 < a 1 * q ∧ a 1 * q < 2)
  (h3 : 2 < a 1 * q^2 ∧ a 1 * q^2 < 3) :
  ∃ a4 : ℝ, a4 = a 1 * q^3 ∧ 2 * Real.sqrt 2 < a4 ∧ a4 < 9 := 
sorry

end geometric_sequence_a4_range_l196_196911


namespace inequality_proof_l196_196725

theorem inequality_proof (a b c : ℝ) (h : a * c^2 > b * c^2) (hc2 : c^2 > 0) : a > b :=
sorry

end inequality_proof_l196_196725


namespace evaluate_fraction_l196_196209

theorem evaluate_fraction :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := 
by
  -- Skipping the proof part with 'sorry'
  sorry

end evaluate_fraction_l196_196209


namespace ellipse_fixed_point_l196_196060

theorem ellipse_fixed_point (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ) (h3 : c = 1) 
    (h4 : a = 2) (h5 : b = Real.sqrt 3) :
    (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        ∃ M : ℝ × ℝ, (M.1 = 4) ∧ 
        ∃ Q : ℝ × ℝ, (Q.1= (P.1) ∧ Q.2 = - (P.2)) ∧ 
            ∃ fixed_point : ℝ × ℝ, (fixed_point.1 = 5 / 2) ∧ (fixed_point.2 = 0) ∧ 
            ∃ k, (Q.2 - M.2) = k * (Q.1 - M.1) ∧ 
            ∃ l, fixed_point.2 = l * (fixed_point.1 - M.1)) :=
sorry

end ellipse_fixed_point_l196_196060


namespace rocket_soaring_time_l196_196851

theorem rocket_soaring_time 
  (avg_speed : ℝ)                      -- The average speed of the rocket
  (soar_speed : ℝ)                     -- Speed while soaring
  (plummet_distance : ℝ)               -- Distance covered during plummet
  (plummet_time : ℝ)                   -- Time of plummet
  (total_time : ℝ := plummet_time + t) -- Total time is the sum of soaring time and plummet time
  (total_distance : ℝ := soar_speed * t + plummet_distance) -- Total distance covered
  (h_avg_speed : avg_speed = total_distance / total_time)   -- Given condition for average speed
  :
  ∃ t : ℝ, t = 12 :=                   -- Prove that the soaring time is 12 seconds
by
  sorry

end rocket_soaring_time_l196_196851


namespace total_pens_l196_196357

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l196_196357


namespace size_of_each_bottle_l196_196123

-- Defining given conditions
def petals_per_ounce : ℕ := 320
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes : ℕ := 800
def bottles : ℕ := 20

-- Proving the size of each bottle in ounces
theorem size_of_each_bottle : (petals_per_rose * roses_per_bush * bushes / petals_per_ounce) / bottles = 12 := by
  sorry

end size_of_each_bottle_l196_196123


namespace answer_choices_l196_196467

theorem answer_choices (n : ℕ) (h : (n + 1) ^ 4 = 625) : n = 4 :=
by {
  sorry
}

end answer_choices_l196_196467


namespace total_pies_eq_l196_196265

-- Definitions for the number of pies made by each person
def pinky_pies : ℕ := 147
def helen_pies : ℕ := 56
def emily_pies : ℕ := 89
def jake_pies : ℕ := 122

-- The theorem stating the total number of pies
theorem total_pies_eq : pinky_pies + helen_pies + emily_pies + jake_pies = 414 :=
by sorry

end total_pies_eq_l196_196265


namespace total_students_at_competition_l196_196263

variable (K H N : ℕ)

theorem total_students_at_competition
  (H_eq : H = (3/5) * K)
  (N_eq : N = 2 * (K + H))
  (total_students : K + H + N = 240) :
  K + H + N = 240 :=
by
  sorry

end total_students_at_competition_l196_196263


namespace find_number_l196_196314

theorem find_number (x n : ℤ) (h1 : 5 * x + n = 10 * x - 17) (h2 : x = 4) : n = 3 := by
  sorry

end find_number_l196_196314


namespace average_temperature_l196_196502

theorem average_temperature :
  ∀ (T : ℝ) (Tt : ℝ),
  -- Conditions
  (43 + T + T + T) / 4 = 48 → 
  Tt = 35 →
  -- Proof
  (T + T + T + Tt) / 4 = 46 :=
by
  intros T Tt H1 H2
  sorry

end average_temperature_l196_196502


namespace investment_final_value_l196_196346

theorem investment_final_value 
  (original_investment : ℝ) 
  (increase_percentage : ℝ) 
  (original_investment_eq : original_investment = 12500)
  (increase_percentage_eq : increase_percentage = 2.15) : 
  original_investment * (1 + increase_percentage) = 39375 := 
by
  sorry

end investment_final_value_l196_196346


namespace multiple_of_a_power_l196_196723

theorem multiple_of_a_power (a n m : ℕ) (h : a^n ∣ m) : a^(n+1) ∣ (a+1)^m - 1 := 
sorry

end multiple_of_a_power_l196_196723


namespace bivalid_positions_count_l196_196008

/-- 
A position of the hands of a (12-hour, analog) clock is called valid if it occurs in the course of a day.
A position of the hands is called bivalid if it is valid and, in addition, the position formed by interchanging the hour and minute hands is valid.
-/
def is_valid (h m : ℕ) : Prop := 
  0 ≤ h ∧ h < 360 ∧ 
  0 ≤ m ∧ m < 360

def satisfies_conditions (h m : Int) (a b : Int) : Prop :=
  m = 12 * h - 360 * a ∧ h = 12 * m - 360 * b

def is_bivalid (h m : ℕ) : Prop := 
  ∃ (a b : Int), satisfies_conditions (h : Int) (m : Int) a b ∧ satisfies_conditions (m : Int) (h : Int) b a

theorem bivalid_positions_count : 
  ∃ (n : ℕ), n = 143 ∧ 
  ∀ (h m : ℕ), is_bivalid h m → n = 143 :=
sorry

end bivalid_positions_count_l196_196008


namespace parabola_vertex_point_l196_196458

theorem parabola_vertex_point (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c → 
  ∃ k : ℝ, ∃ h : ℝ, y = a * (x - h)^2 + k ∧ h = 2 ∧ k = -1 ∧ 
  (∃ y₀ : ℝ, 7 = a * (0 - h)^2 + k) ∧ y₀ = 7) 
  → (a = 2 ∧ b = -8 ∧ c = 7) := by
  sorry

end parabola_vertex_point_l196_196458


namespace apples_in_baskets_l196_196697

theorem apples_in_baskets (total_apples : ℕ) (first_basket : ℕ) (increase : ℕ) (baskets : ℕ) :
  total_apples = 495 ∧ first_basket = 25 ∧ increase = 2 ∧
  (total_apples = (baskets / 2) * (2 * first_basket + (baskets - 1) * increase)) -> baskets = 13 :=
by sorry

end apples_in_baskets_l196_196697


namespace Robin_needs_to_buy_more_bottles_l196_196974

/-- Robin wants to drink exactly nine bottles of water each day.
    She initially bought six hundred seventeen bottles.
    Prove that she will need to buy 4 more bottles on the last day
    to meet her goal of drinking exactly nine bottles each day. -/
theorem Robin_needs_to_buy_more_bottles :
  ∀ total_bottles bottles_per_day : ℕ, total_bottles = 617 → bottles_per_day = 9 → 
  ∃ extra_bottles : ℕ, (617 % 9) + extra_bottles = 9 ∧ extra_bottles = 4 :=
by
  sorry

end Robin_needs_to_buy_more_bottles_l196_196974


namespace max_discriminant_l196_196217

noncomputable def f (a b c x : ℤ) := a * x^2 + b * x + c

theorem max_discriminant (a b c u v w : ℤ)
  (h1 : u ≠ v) (h2 : v ≠ w) (h3 : u ≠ w)
  (hu : f a b c u = 0)
  (hv : f a b c v = 0)
  (hw : f a b c w = 2) :
  ∃ (a b c : ℤ), b^2 - 4 * a * c = 16 :=
sorry

end max_discriminant_l196_196217


namespace fraction_power_simplification_l196_196964

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l196_196964


namespace rectangle_width_l196_196705

theorem rectangle_width (L W : ℕ)
  (h1 : W = L + 3)
  (h2 : 2 * L + 2 * W = 54) :
  W = 15 :=
by
  sorry

end rectangle_width_l196_196705


namespace problem_a_l196_196019

variable {S : Type*}
variables (a b : S)
variables [Inhabited S] -- Ensures S has at least one element
variables (op : S → S → S) -- Defines the binary operation

-- Condition: binary operation a * (b * a) = b holds for all a, b in S
axiom binary_condition : ∀ a b : S, op a (op b a) = b

-- Theorem to prove: (a * b) * a ≠ a
theorem problem_a : (op (op a b) a) ≠ a :=
sorry

end problem_a_l196_196019


namespace repeated_root_and_m_value_l196_196482

theorem repeated_root_and_m_value :
  (∃ x m : ℝ, (x = 2 ∨ x = -2) ∧ 
              (m / (x ^ 2 - 4) + 2 / (x + 2) = 1 / (x - 2)) ∧ 
              (m = 4 ∨ m = 8)) :=
sorry

end repeated_root_and_m_value_l196_196482


namespace tammy_avg_speed_second_day_l196_196514

theorem tammy_avg_speed_second_day (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) :
  v + 0.5 = 4 :=
sorry

end tammy_avg_speed_second_day_l196_196514


namespace maximum_books_l196_196021

theorem maximum_books (dollars : ℝ) (price_per_book : ℝ) (n : ℕ) 
    (h1 : dollars = 12) (h2 : price_per_book = 1.25) : n ≤ 9 :=
    sorry

end maximum_books_l196_196021


namespace num_balls_total_l196_196365

theorem num_balls_total (m : ℕ) (h1 : 6 < m) (h2 : (6 : ℝ) / (m : ℝ) = 0.3) : m = 20 :=
by
  sorry

end num_balls_total_l196_196365


namespace cost_to_buy_450_candies_l196_196635

-- Define a structure representing the problem conditions
structure CandyStore where
  candies_per_box : Nat
  regular_price : Nat
  discounted_price : Nat
  discount_threshold : Nat

-- Define parameters for this specific problem
def store : CandyStore :=
  { candies_per_box := 15,
    regular_price := 5,
    discounted_price := 4,
    discount_threshold := 10 }

-- Define the cost function with the given conditions
def cost (store : CandyStore) (candies : Nat) : Nat :=
  let boxes := candies / store.candies_per_box
  if boxes >= store.discount_threshold then
    boxes * store.discounted_price
  else
    boxes * store.regular_price

-- State the theorem we want to prove
theorem cost_to_buy_450_candies (store : CandyStore) (candies := 450) :
  store.candies_per_box = 15 →
  store.discounted_price = 4 →
  store.discount_threshold = 10 →
  cost store candies = 120 := by
  sorry

end cost_to_buy_450_candies_l196_196635


namespace M_gt_N_l196_196167

variable (a b : ℝ)

def M := 10 * a^2 + 2 * b^2 - 7 * a + 6
def N := a^2 + 2 * b^2 + 5 * a + 1

theorem M_gt_N : M a b > N a b := by
  sorry

end M_gt_N_l196_196167


namespace felipe_building_time_l196_196372

theorem felipe_building_time
  (F E : ℕ)
  (combined_time_without_breaks : ℕ)
  (felipe_time_fraction : F = E / 2)
  (combined_time_condition : F + E = 90)
  (felipe_break : ℕ)
  (emilio_break : ℕ)
  (felipe_break_is_6_months : felipe_break = 6)
  (emilio_break_is_double_felipe : emilio_break = 2 * felipe_break) :
  F + felipe_break = 36 := by
  sorry

end felipe_building_time_l196_196372


namespace mark_last_shots_l196_196558

theorem mark_last_shots (h1 : 0.60 * 15 = 9) (h2 : 0.65 * 25 = 16.25) : 
  ∀ (successful_shots_first_15 successful_shots_total: ℤ),
  successful_shots_first_15 = 9 ∧ 
  successful_shots_total = 16 → 
  successful_shots_total - successful_shots_first_15 = 7 := by
  sorry

end mark_last_shots_l196_196558


namespace measure_of_angle_C_l196_196510

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 7 * D) : C = 157.5 := 
by 
  sorry

end measure_of_angle_C_l196_196510


namespace max_mean_weight_BC_l196_196993

theorem max_mean_weight_BC
  (A_n B_n C_n : ℕ)
  (w_A w_B : ℕ)
  (mean_A mean_B mean_AB mean_AC : ℤ)
  (hA : mean_A = 30)
  (hB : mean_B = 55)
  (hAB : mean_AB = 35)
  (hAC : mean_AC = 32)
  (h1 : mean_A * A_n + mean_B * B_n = mean_AB * (A_n + B_n))
  (h2 : mean_A * A_n + mean_AC * C_n = mean_AC * (A_n + C_n)) :
  ∃ n : ℕ, n ≤ 62 ∧ (mean_B * B_n + w_A * C_n) / (B_n + C_n) = n := 
sorry

end max_mean_weight_BC_l196_196993


namespace find_f_neg_one_l196_196832

theorem find_f_neg_one (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + x^2 = - (f x + x^2)) (h2 : f 1 = 1) : f (-1) = -3 := by
  sorry

end find_f_neg_one_l196_196832


namespace alpha_beta_squared_l196_196729

section
variables (α β : ℝ)
-- Given conditions
def is_root (a b : ℝ) : Prop :=
  a + b = 2 ∧ a * b = -1 ∧ (∀ x : ℝ, x^2 - 2 * x - 1 = 0 → x = a ∨ x = b)

-- The theorem to prove
theorem alpha_beta_squared (h: is_root α β) : α^2 + β^2 = 6 :=
sorry
end

end alpha_beta_squared_l196_196729


namespace range_of_first_term_in_geometric_sequence_l196_196143

theorem range_of_first_term_in_geometric_sequence (q a₁ : ℝ)
  (h_q : |q| < 1)
  (h_sum : a₁ / (1 - q) = q) :
  -2 < a₁ ∧ a₁ ≤ 0.25 ∧ a₁ ≠ 0 :=
by
  sorry

end range_of_first_term_in_geometric_sequence_l196_196143


namespace number_of_pencil_cartons_l196_196429

theorem number_of_pencil_cartons
  (P E : ℕ) 
  (h1 : P + E = 100)
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_pencil_cartons_l196_196429


namespace difference_of_squares_example_l196_196695

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 123) (h2 : b = 23) : a^2 - b^2 = 14600 :=
by
  rw [h1, h2]
  sorry

end difference_of_squares_example_l196_196695


namespace x_power6_y_power6_l196_196387

theorem x_power6_y_power6 (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6 * a^4 * b + 9 * a^2 * b^2 - 2 * b^3 :=
sorry

end x_power6_y_power6_l196_196387


namespace sum_arithmetic_sequence_l196_196767

theorem sum_arithmetic_sequence {a : ℕ → ℤ} (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 13 = S 2000 →
  S 2013 = 0 :=
by
  sorry

end sum_arithmetic_sequence_l196_196767


namespace hyperbola_eccentricity_l196_196179

def hyperbola : Prop :=
  ∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1

noncomputable def eccentricity : ℝ :=
  let a := 3
  let b := 4
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity : 
  (∀ (x y : ℝ), (x^2 / 9) - (y^2 / 16) = 1) → eccentricity = 5 / 3 :=
by
  intros h
  funext
  exact sorry

end hyperbola_eccentricity_l196_196179


namespace abs_eq_abs_of_unique_solution_l196_196568

variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
theorem abs_eq_abs_of_unique_solution
  (h : ∃ x : ℝ, ∀ y : ℝ, a * (y - a)^2 + b * (y - b)^2 = 0 ↔ y = x) :
  |a| = |b| :=
sorry

end abs_eq_abs_of_unique_solution_l196_196568


namespace squares_difference_l196_196965

theorem squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 := by
  sorry

end squares_difference_l196_196965


namespace right_handed_players_total_l196_196061

def total_players : ℕ := 64
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def total_right_handed : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_total : total_right_handed = 55 := by
  sorry

end right_handed_players_total_l196_196061


namespace order_of_abc_l196_196775

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l196_196775


namespace time_to_cover_escalator_l196_196624

def escalator_speed : ℝ := 12
def escalator_length : ℝ := 160
def person_speed : ℝ := 8

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 8 := by
  sorry

end time_to_cover_escalator_l196_196624


namespace octagon_side_length_l196_196288

theorem octagon_side_length 
  (num_sides : ℕ) 
  (perimeter : ℝ) 
  (h_sides : num_sides = 8) 
  (h_perimeter : perimeter = 23.6) :
  (perimeter / num_sides) = 2.95 :=
by
  have h_valid_sides : num_sides = 8 := h_sides
  have h_valid_perimeter : perimeter = 23.6 := h_perimeter
  sorry

end octagon_side_length_l196_196288


namespace arithmetic_seq_a7_l196_196003

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (d : ℕ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 8)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 7 = 6 := by
  sorry

end arithmetic_seq_a7_l196_196003


namespace curve_intersects_x_axis_at_4_over_5_l196_196999

-- Define the function for the curve
noncomputable def curve (x : ℝ) : ℝ :=
  (3 * x - 1) * (Real.sqrt (9 * x ^ 2 - 6 * x + 5) + 1) +
  (2 * x - 3) * (Real.sqrt (4 * x ^ 2 - 12 * x + 13) + 1)

-- Prove that curve(x) = 0 when x = 4 / 5
theorem curve_intersects_x_axis_at_4_over_5 :
  curve (4 / 5) = 0 :=
by
  sorry

end curve_intersects_x_axis_at_4_over_5_l196_196999


namespace boards_tested_l196_196663

-- Define the initial conditions and problem
def total_thumbtacks : ℕ := 450
def thumbtacks_remaining_each_can : ℕ := 30
def initial_thumbtacks_each_can := total_thumbtacks / 3
def thumbtacks_used_each_can := initial_thumbtacks_each_can - thumbtacks_remaining_each_can
def total_thumbtacks_used := thumbtacks_used_each_can * 3
def thumbtacks_per_board := 3

-- Define the proposition to prove 
theorem boards_tested (B : ℕ) : 
  (B = total_thumbtacks_used / thumbtacks_per_board) → B = 120 :=
by
  -- Proof skipped with sorry
  sorry

end boards_tested_l196_196663


namespace evaluate_expression_l196_196149

theorem evaluate_expression : (7 - 3) ^ 2 + (7 ^ 2 - 3 ^ 2) = 56 := by
  sorry

end evaluate_expression_l196_196149


namespace reduction_percentage_40_l196_196788

theorem reduction_percentage_40 (P : ℝ) : 
  1500 * 1.20 - (P / 100 * (1500 * 1.20)) = 1080 ↔ P = 40 :=
by
  sorry

end reduction_percentage_40_l196_196788


namespace tracy_sold_paintings_l196_196547

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end tracy_sold_paintings_l196_196547


namespace find_range_a_l196_196687

def setA (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def setB (x a : ℝ) : Prop := |x - a| < 5
def real_line (x : ℝ) : Prop := True

theorem find_range_a (a : ℝ) :
  (∀ x, setA x ∨ setB x a) ↔ (-3:ℝ) ≤ a ∧ a ≤ 1 := by
sorry

end find_range_a_l196_196687


namespace combined_total_value_of_items_l196_196772

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items_l196_196772


namespace x_minus_y_values_l196_196077

theorem x_minus_y_values (x y : ℝ) 
  (h1 : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) : x - y = -1 ∨ x - y = -7 := 
  sorry

end x_minus_y_values_l196_196077


namespace range_a_le_2_l196_196848
-- Import everything from Mathlib

-- Define the hypothesis and the conclusion in Lean 4
theorem range_a_le_2 (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) ↔ a ≤ 2 := 
sorry

end range_a_le_2_l196_196848


namespace problem1_range_problem2_range_l196_196770

theorem problem1_range (x y : ℝ) (h : y = 2*|x-1| - |x-4|) : -3 ≤ y := sorry

theorem problem2_range (x a : ℝ) (h : ∀ x, 2*|x-1| - |x-a| ≥ -1) : 0 ≤ a ∧ a ≤ 2 := sorry

end problem1_range_problem2_range_l196_196770


namespace ratio_of_volumes_l196_196549

theorem ratio_of_volumes (s : ℝ) (hs : s > 0) :
  let r_s := s / 2
  let r_c := s / 2
  let V_sphere := (4 / 3) * π * (r_s ^ 3)
  let V_cylinder := π * (r_c ^ 2) * s
  let V_total := V_sphere + V_cylinder
  let V_cube := s ^ 3
  V_total / V_cube = (5 * π) / 12 := by {
    -- Given the conditions and expressions
    sorry
  }

end ratio_of_volumes_l196_196549


namespace problem_1_problem_2_l196_196494

def A := {x : ℝ | 1 < 2 * x - 1 ∧ 2 * x - 1 < 7}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}

theorem problem_1 : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
sorry

theorem problem_2 : (A ∪ B)ᶜ = {x : ℝ | x ≤ -1 ∨ x ≥ 4} :=
sorry

end problem_1_problem_2_l196_196494


namespace sum_f_values_l196_196617

noncomputable def f (x : ℤ) : ℤ := (x - 1)^3 + 1

theorem sum_f_values :
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7) = 13 :=
by
  sorry

end sum_f_values_l196_196617


namespace sugar_merchant_profit_l196_196702

theorem sugar_merchant_profit 
    (total_sugar : ℕ)
    (sold_at_18 : ℕ)
    (remain_sugar : ℕ)
    (whole_profit : ℕ)
    (profit_18 : ℕ)
    (rem_profit_percent : ℕ) :
    total_sugar = 1000 →
    sold_at_18 = 600 →
    remain_sugar = total_sugar - sold_at_18 →
    whole_profit = 14 →
    profit_18 = 18 →
    (600 * profit_18 / 100) + (remain_sugar * rem_profit_percent / 100) = 
    (total_sugar * whole_profit / 100) →
    rem_profit_percent = 80 :=
by
    sorry

end sugar_merchant_profit_l196_196702


namespace sequence_difference_l196_196071

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sequence_difference (hS : ∀ n, S n = n^2 - 5 * n)
                            (hna : ∀ n, a n = S n - S (n - 1))
                            (hpq : p - q = 4) :
                            a p - a q = 8 := by
    sorry

end sequence_difference_l196_196071


namespace marble_leftovers_l196_196046

theorem marble_leftovers :
  ∃ r p : ℕ, (r % 8 = 5) ∧ (p % 8 = 7) ∧ ((r + p) % 10 = 0) → ((r + p) % 8 = 4) :=
by { sorry }

end marble_leftovers_l196_196046


namespace cristina_catches_up_l196_196512

theorem cristina_catches_up
  (t : ℝ)
  (cristina_speed : ℝ := 5)
  (nicky_speed : ℝ := 3)
  (nicky_head_start : ℝ := 54)
  (distance_cristina : ℝ := cristina_speed * t)
  (distance_nicky : ℝ := nicky_head_start + nicky_speed * t) :
  distance_cristina = distance_nicky → t = 27 :=
by
  intros h
  sorry

end cristina_catches_up_l196_196512


namespace xiaojuan_savings_l196_196432

-- Define the conditions
def spent_on_novel (savings : ℝ) : ℝ := 0.5 * savings
def mother_gave : ℝ := 5
def spent_on_dictionary (amount_given : ℝ) : ℝ := 0.5 * amount_given + 0.4
def remaining_amount : ℝ := 7.2

-- Define the theorem stating the equivalence
theorem xiaojuan_savings : ∃ (savings: ℝ), spent_on_novel savings + mother_gave - spent_on_dictionary mother_gave - remaining_amount = savings / 2 ∧ savings = 20.4 :=
by {
  sorry
}

end xiaojuan_savings_l196_196432


namespace intersection_A_B_l196_196082

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l196_196082


namespace simplify_sin_cos_expr_cos_pi_six_alpha_expr_l196_196016

open Real

-- Problem (1)
theorem simplify_sin_cos_expr (x : ℝ) :
  (sin x ^ 2 / (sin x - cos x)) - ((sin x + cos x) / (tan x ^ 2 - 1)) - sin x = cos x :=
sorry

-- Problem (2)
theorem cos_pi_six_alpha_expr (α : ℝ) (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + cos (4 * π / 3 + α) ^ 2 = (2 - sqrt 3) / 3 :=
sorry

end simplify_sin_cos_expr_cos_pi_six_alpha_expr_l196_196016


namespace ariana_total_owe_l196_196005

-- Definitions based on the conditions
def first_bill_principal : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_overdue_months : ℕ := 2

def second_bill_principal : ℕ := 130
def second_bill_late_fee : ℕ := 50
def second_bill_overdue_months : ℕ := 6

def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80

-- Theorem
theorem ariana_total_owe : 
  first_bill_principal + 
    (first_bill_principal : ℝ) * first_bill_interest_rate * (first_bill_overdue_months : ℝ) +
    second_bill_principal + 
    second_bill_late_fee * second_bill_overdue_months + 
    third_bill_first_month_fee + 
    third_bill_second_month_fee = 790 := 
by 
  sorry

end ariana_total_owe_l196_196005


namespace lateral_surface_area_of_cone_l196_196407

theorem lateral_surface_area_of_cone (diameter height : ℝ) (h_d : diameter = 2) (h_h : height = 2) :
  let radius := diameter / 2
  let slant_height := Real.sqrt (radius ^ 2 + height ^ 2)
  π * radius * slant_height = Real.sqrt 5 * π := 
  by
    sorry

end lateral_surface_area_of_cone_l196_196407


namespace expected_number_of_defective_products_l196_196327

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end expected_number_of_defective_products_l196_196327


namespace fraction_of_people_under_21_correct_l196_196064

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct_l196_196064


namespace units_digit_x4_invx4_l196_196605

theorem units_digit_x4_invx4 (x : ℝ) (h : x^2 - 12 * x + 1 = 0) : 
  (x^4 + (1 / x)^4) % 10 = 2 := 
by
  sorry

end units_digit_x4_invx4_l196_196605


namespace pages_copyable_l196_196233

-- Define the conditions
def cents_per_dollar : ℕ := 100
def dollars_available : ℕ := 25
def cost_per_page : ℕ := 3

-- Define the total cents available
def total_cents : ℕ := dollars_available * cents_per_dollar

-- Define the expected number of full pages
def expected_pages : ℕ := 833

theorem pages_copyable :
  (total_cents : ℕ) / cost_per_page = expected_pages := sorry

end pages_copyable_l196_196233


namespace simplify_and_evaluate_l196_196065

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = -1) (h2 : b = 1) :
  (4/5 * a * b - (2 * a * b^2 - 4 * (-1/5 * a * b + 3 * a^2 * b)) + 2 * a * b^2) = 12 :=
by
  have ha : a = -1 := h1
  have hb : b = 1 := h2
  sorry

end simplify_and_evaluate_l196_196065


namespace stripe_area_is_640pi_l196_196338

noncomputable def cylinder_stripe_area (diameter height stripe_width : ℝ) (revolutions : ℕ) : ℝ :=
  let circumference := Real.pi * diameter
  let length := circumference * (revolutions : ℝ)
  stripe_width * length

theorem stripe_area_is_640pi :
  cylinder_stripe_area 20 100 4 4 = 640 * Real.pi :=
by 
  sorry

end stripe_area_is_640pi_l196_196338


namespace find_base_b4_l196_196805

theorem find_base_b4 (b_4 : ℕ) : (b_4 - 1) * (b_4 - 2) * (b_4 - 3) = 168 → b_4 = 8 :=
by
  intro h
  -- proof goes here
  sorry

end find_base_b4_l196_196805


namespace total_students_in_class_l196_196449

def students_play_football : Nat := 26
def students_play_tennis : Nat := 20
def students_play_both : Nat := 17
def students_play_neither : Nat := 7

theorem total_students_in_class :
  (students_play_football + students_play_tennis - students_play_both + students_play_neither) = 36 :=
by
  sorry

end total_students_in_class_l196_196449


namespace solution_inequality_l196_196375

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function_at (f : ℝ → ℝ) (x : ℝ) : Prop := f (2 + x) = f (2 - x)
def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x < y → x ∈ s → y ∈ s → f x < f y

-- Main statement
theorem solution_inequality 
  (h1 : ∀ x, is_even_function_at f x)
  (h2 : is_increasing_on f {x : ℝ | x ≤ 2}) :
  (∀ a : ℝ, (a > -1) ∧ (a ≠ 0) ↔ f (a^2 + 3*a + 2) < f (a^2 - a + 2)) :=
by {
  sorry
}

end solution_inequality_l196_196375


namespace cistern_length_l196_196356

theorem cistern_length
  (L W D A : ℝ)
  (hW : W = 4)
  (hD : D = 1.25)
  (hA : A = 49)
  (hWetSurface : A = L * W + 2 * L * D) :
  L = 7.54 := by
  sorry

end cistern_length_l196_196356


namespace developer_break_even_price_l196_196778

theorem developer_break_even_price :
  let acres := 4
  let cost_per_acre := 1863
  let total_cost := acres * cost_per_acre
  let num_lots := 9
  let cost_per_lot := total_cost / num_lots
  cost_per_lot = 828 :=
by {
  sorry  -- This is where the proof would go.
} 

end developer_break_even_price_l196_196778


namespace min_value_expression_l196_196230

theorem min_value_expression : ∀ (x y : ℝ), ∃ z : ℝ, z ≥ 3*x^2 + 2*x*y + 3*y^2 + 5 ∧ z = 5 :=
by
  sorry

end min_value_expression_l196_196230


namespace diet_soda_bottles_l196_196804

-- Define the conditions and then state the problem
theorem diet_soda_bottles (R D : ℕ) (h1 : R = 67) (h2 : R = D + 58) : D = 9 :=
by
  -- The proof goes here
  sorry

end diet_soda_bottles_l196_196804


namespace arithmetic_sequence_a2_l196_196084

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a1_a3 : a 1 + a 3 = 2) : a 2 = 1 :=
sorry

end arithmetic_sequence_a2_l196_196084


namespace trailing_zeroes_500_fact_l196_196496

-- Define a function to count multiples of a given number in a range
def countMultiples (n m : Nat) : Nat :=
  m / n

-- Define a function to count trailing zeroes in the factorial
def trailingZeroesFactorial (n : Nat) : Nat :=
  countMultiples 5 n + countMultiples (5^2) n + countMultiples (5^3) n + countMultiples (5^4) n

theorem trailing_zeroes_500_fact : trailingZeroesFactorial 500 = 124 :=
by
  sorry

end trailing_zeroes_500_fact_l196_196496


namespace prove_cardinality_l196_196478

-- Definitions used in Lean 4 Statement adapted from conditions
variable (a b : ℕ)
variable (A B : Finset ℕ)

-- Hypotheses
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_disjoint : Disjoint A B)
variable (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B)

-- The statement to prove
theorem prove_cardinality (a b : ℕ) (A B : Finset ℕ)
  (ha : a > 0) (hb : b > 0) (h_disjoint : Disjoint A B)
  (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B) :
  a * A.card = b * B.card :=
by 
  sorry

end prove_cardinality_l196_196478


namespace wage_of_one_man_l196_196701

/-- Proof that the wage of one man is Rs. 24 given the conditions. -/
theorem wage_of_one_man (M W_w B_w : ℕ) (H1 : 120 = 5 * M + W_w * 5 + B_w * 8) 
  (H2 : 5 * M = W_w * 5) (H3 : W_w * 5 = B_w * 8) : M = 24 :=
by
  sorry

end wage_of_one_man_l196_196701


namespace infinite_rationals_sqrt_rational_l196_196038

theorem infinite_rationals_sqrt_rational : ∃ᶠ x : ℚ in Filter.atTop, ∃ y : ℚ, y = Real.sqrt (x^2 + x + 1) :=
sorry

end infinite_rationals_sqrt_rational_l196_196038


namespace minimum_value_inequality_l196_196871

variable {x y z : ℝ}
variable (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)

theorem minimum_value_inequality : (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 :=
sorry

end minimum_value_inequality_l196_196871


namespace factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l196_196901

-- Statements corresponding to the given problems

-- Theorem for 1)
theorem factorize_poly1 (a : ℤ) : 
  (a^7 + a^5 + 1) = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := 
by sorry

-- Theorem for 2)
theorem factorize_poly2 (a b : ℤ) : 
  (a^5 + a*b^4 + b^5) = (a + b) * (a^4 - a^3*b + a^2*b^2 - a*b^3 + b^4) := 
by sorry

-- Theorem for 3)
theorem factorize_poly3 (a : ℤ) : 
  (a^7 - 1) = (a - 1) * (a^6 + a^5 + a^4 + a^3 + a^2 + a + 1) := 
by sorry

-- Theorem for 4)
theorem factorize_poly4 (a x : ℤ) : 
  (2 * a^3 - a * x^2 - x^3) = (a - x) * (2 * a^2 + 2 * a * x + x^2) := 
by sorry

end factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l196_196901


namespace value_of_3y_l196_196988

theorem value_of_3y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h4 : z = 3) :
  3 * y = 12 :=
by
  sorry

end value_of_3y_l196_196988


namespace problem_l196_196424

variables {a b c : ℝ}

-- Given positive numbers a, b, c
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c

-- Given conditions
axiom h1 : a * b + a + b = 3
axiom h2 : b * c + b + c = 3
axiom h3 : a * c + a + c = 3

-- Goal statement
theorem problem : (a + 1) * (b + 1) * (c + 1) = 8 := 
by 
  sorry

end problem_l196_196424


namespace ratio_of_a_and_b_l196_196035

theorem ratio_of_a_and_b (x y a b : ℝ) (h1 : x / y = 3) (h2 : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end ratio_of_a_and_b_l196_196035


namespace mehki_age_l196_196920

variable (Mehki Jordyn Zrinka : ℕ)

axiom h1 : Mehki = Jordyn + 10
axiom h2 : Jordyn = 2 * Zrinka
axiom h3 : Zrinka = 6

theorem mehki_age : Mehki = 22 := by
  -- sorry to skip the proof
  sorry

end mehki_age_l196_196920


namespace star_3_5_l196_196633

def star (a b : ℕ) : ℕ := a^2 + 3 * a * b + b^2

theorem star_3_5 : star 3 5 = 79 := 
by
  sorry

end star_3_5_l196_196633


namespace max_value_of_x_plus_y_plus_z_l196_196231

theorem max_value_of_x_plus_y_plus_z : ∀ (x y z : ℤ), (∃ k : ℤ, x = 5 * k ∧ 6 = y * k ∧ z = 2 * k) → x + y + z ≤ 43 :=
by
  intros x y z h
  rcases h with ⟨k, hx, hy, hz⟩
  sorry

end max_value_of_x_plus_y_plus_z_l196_196231


namespace extreme_value_of_f_range_of_values_for_a_l196_196748

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_of_f :
  ∃ x_min : ℝ, f x_min = 1 :=
sorry

theorem range_of_values_for_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ (x^3) / 6 + a) → a ≤ 1 :=
sorry

end extreme_value_of_f_range_of_values_for_a_l196_196748


namespace weight_of_lighter_boxes_l196_196954

theorem weight_of_lighter_boxes :
  ∃ (x : ℝ),
  (∀ (w : ℝ), w = 20 ∨ w = x) ∧
  (20 * 18 = 360) ∧
  (∃ (n : ℕ), n = 15 → 15 * 20 = 300) ∧
  (∃ (m : ℕ), m = 5 → 5 * 12 = 60) ∧
  (360 - 300 = 60) ∧
  (∀ (l : ℝ), l = 60 / 5 → l = x) →
  x = 12 :=
by
  sorry

end weight_of_lighter_boxes_l196_196954


namespace graph_passes_through_point_l196_196009

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem graph_passes_through_point (a : ℝ) : f a 1 = 4 := by
  sorry

end graph_passes_through_point_l196_196009


namespace mulch_cost_l196_196186

-- Definitions based on conditions
def cost_per_cubic_foot : ℕ := 8
def cubic_yard_to_cubic_feet : ℕ := 27
def volume_in_cubic_yards : ℕ := 7

-- Target statement to prove
theorem mulch_cost :
    (volume_in_cubic_yards * cubic_yard_to_cubic_feet) * cost_per_cubic_foot = 1512 := by
  sorry

end mulch_cost_l196_196186


namespace evaluate_expression_l196_196400

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 5) :
  3 * x^4 + 2 * y^2 + 10 = 8 * 37 + 7 := 
by
  sorry

end evaluate_expression_l196_196400


namespace find_other_number_l196_196320

theorem find_other_number (x : ℕ) (h1 : 10 + x = 30) : x = 20 := by
  sorry

end find_other_number_l196_196320


namespace notebook_cost_l196_196362

theorem notebook_cost :
  let mean_expenditure := 500
  let daily_expenditures := [450, 600, 400, 500, 550, 300]
  let cost_earphone := 620
  let cost_pen := 30
  let total_days := 7
  let total_expenditure := mean_expenditure * total_days
  let sum_other_days := daily_expenditures.sum
  let expenditure_friday := total_expenditure - sum_other_days
  let cost_notebook := expenditure_friday - (cost_earphone + cost_pen)
  cost_notebook = 50 := by
  sorry

end notebook_cost_l196_196362


namespace g_diff_l196_196270

def g (n : ℤ) : ℤ := (1 / 4 : ℤ) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_diff (r : ℤ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_diff_l196_196270


namespace chuck_bicycle_trip_l196_196750

theorem chuck_bicycle_trip (D : ℝ) (h1 : D / 16 + D / 24 = 3) : D = 28.80 :=
by
  sorry

end chuck_bicycle_trip_l196_196750


namespace bacteria_colony_first_day_exceeds_100_l196_196953

theorem bacteria_colony_first_day_exceeds_100 :
  ∃ n : ℕ, 3 * 2^n > 100 ∧ (∀ m < n, 3 * 2^m ≤ 100) :=
sorry

end bacteria_colony_first_day_exceeds_100_l196_196953


namespace outer_perimeter_l196_196441

theorem outer_perimeter (F G H I J K L M N : ℕ) 
  (h_outer : F + G + H + I + J = 42) 
  (h_inner : K + L + M = 20) 
  (h_adjustment : N = 4) : 
  F + G + H + I + J - K - L - M + N = 26 := 
by 
  sorry

end outer_perimeter_l196_196441


namespace Maria_selling_price_l196_196202

-- Define the constants based on the given conditions
def brush_cost : ℕ := 20
def canvas_cost : ℕ := 3 * brush_cost
def paint_cost_per_liter : ℕ := 8
def paint_needed : ℕ := 5
def earnings : ℕ := 80

-- Calculate the total cost and the selling price
def total_cost : ℕ := brush_cost + canvas_cost + (paint_cost_per_liter * paint_needed)
def selling_price : ℕ := total_cost + earnings

-- Proof statement
theorem Maria_selling_price : selling_price = 200 := by
  sorry

end Maria_selling_price_l196_196202


namespace distance_center_of_ball_travels_l196_196237

noncomputable def radius_of_ball : ℝ := 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80

noncomputable def adjusted_R1 : ℝ := R1 - radius_of_ball
noncomputable def adjusted_R2 : ℝ := R2 + radius_of_ball
noncomputable def adjusted_R3 : ℝ := R3 - radius_of_ball

noncomputable def distance_travelled : ℝ :=
  (Real.pi * adjusted_R1) +
  (Real.pi * adjusted_R2) +
  (Real.pi * adjusted_R3)

theorem distance_center_of_ball_travels : distance_travelled = 238 * Real.pi :=
by
  sorry

end distance_center_of_ball_travels_l196_196237


namespace sixth_graders_forgot_homework_percentage_l196_196813

-- Definitions of the conditions
def num_students_A : ℕ := 20
def num_students_B : ℕ := 80
def percent_forgot_A : ℚ := 20 / 100
def percent_forgot_B : ℚ := 15 / 100

-- Statement to be proven
theorem sixth_graders_forgot_homework_percentage :
  (num_students_A * percent_forgot_A + num_students_B * percent_forgot_B) /
  (num_students_A + num_students_B) = 16 / 100 :=
by
  sorry

end sixth_graders_forgot_homework_percentage_l196_196813


namespace solve_cryptarithm_l196_196661

-- Declare non-computable constants for the letters
variables {A B C : ℕ}

-- Conditions from the problem
-- Different letters represent different digits
axiom diff_digits : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- A ≠ 0
axiom A_nonzero : A ≠ 0

-- Given cryptarithm equation
axiom cryptarithm_eq : 100 * C + 10 * B + A + 100 * A + 10 * A + A = 100 * B + A

-- The proof to show the correct values
theorem solve_cryptarithm : A = 5 ∧ B = 9 ∧ C = 3 :=
sorry

end solve_cryptarithm_l196_196661


namespace abs_neg_eq_five_l196_196306

theorem abs_neg_eq_five (a : ℝ) : abs (-a) = 5 ↔ (a = 5 ∨ a = -5) :=
by
  sorry

end abs_neg_eq_five_l196_196306


namespace original_price_of_trouser_l196_196427

theorem original_price_of_trouser (sale_price : ℝ) (percent_decrease : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 75) 
  (h2 : percent_decrease = 0.25) 
  (h3 : original_price - percent_decrease * original_price = sale_price) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l196_196427


namespace michael_passes_donovan_l196_196156

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ)
  (start_time : ℕ)
  (L : ℕ)
  (h1 : track_length = 500)
  (h2 : donovan_lap_time = 45)
  (h3 : michael_lap_time = 40)
  (h4 : start_time = 0)
  : L = 9 :=
by
  sorry

end michael_passes_donovan_l196_196156


namespace f_2019_value_l196_196683

noncomputable def B : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1}

noncomputable def g (x : ℚ) (h : x ∈ B) : ℚ :=
  1 - (2 / x)

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ :=
  sorry

theorem f_2019_value (h2019 : 2019 ∈ B) :
  f 2019 h2019 = Real.log ((2019 - 0.5) ^ 2 / 2018.5) :=
sorry

end f_2019_value_l196_196683


namespace largest_of_four_consecutive_even_numbers_l196_196754

-- Conditions
def sum_of_four_consecutive_even_numbers (x : ℤ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) = 92

-- Proof statement
theorem largest_of_four_consecutive_even_numbers (x : ℤ) 
  (h : sum_of_four_consecutive_even_numbers x) : x + 6 = 26 :=
by
  sorry

end largest_of_four_consecutive_even_numbers_l196_196754


namespace cos_540_eq_neg_1_l196_196935

theorem cos_540_eq_neg_1 : Real.cos (540 * Real.pi / 180) = -1 := by
  sorry

end cos_540_eq_neg_1_l196_196935


namespace statues_painted_l196_196786

-- Definitions based on the conditions provided in the problem
def paint_remaining : ℚ := 1/2
def paint_per_statue : ℚ := 1/4

-- The theorem that answers the question
theorem statues_painted (h : paint_remaining = 1/2 ∧ paint_per_statue = 1/4) : 
  (paint_remaining / paint_per_statue) = 2 := 
sorry

end statues_painted_l196_196786


namespace sin_double_angle_half_l196_196579

theorem sin_double_angle_half (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_half_l196_196579


namespace exists_x_for_every_n_l196_196445

theorem exists_x_for_every_n (n : ℕ) (hn : 0 < n) : ∃ x : ℤ, 2^n ∣ (x^2 - 17) :=
sorry

end exists_x_for_every_n_l196_196445


namespace wall_width_l196_196519

theorem wall_width (w h l : ℝ)
  (h_eq_6w : h = 6 * w)
  (l_eq_7h : l = 7 * h)
  (V_eq : w * h * l = 86436) :
  w = 7 :=
by
  sorry

end wall_width_l196_196519


namespace ratio_of_white_marbles_l196_196823

theorem ratio_of_white_marbles (total_marbles yellow_marbles red_marbles : ℕ)
    (h1 : total_marbles = 50)
    (h2 : yellow_marbles = 12)
    (h3 : red_marbles = 7)
    (green_marbles : ℕ)
    (h4 : green_marbles = yellow_marbles - yellow_marbles / 2) :
    (total_marbles - (yellow_marbles + green_marbles + red_marbles)) / total_marbles = 1 / 2 :=
by
  sorry

end ratio_of_white_marbles_l196_196823


namespace minimum_moves_l196_196507

theorem minimum_moves (n : ℕ) : 
  n > 0 → ∃ k l : ℕ, k + 2 * l ≥ ⌊ (n^2 : ℝ) / 2 ⌋₊ ∧ k + l ≥ ⌊ (n^2 : ℝ) / 3 ⌋₊ :=
by 
  intro hn
  sorry

end minimum_moves_l196_196507


namespace cost_of_building_fence_square_plot_l196_196062

-- Definition of conditions
def area_of_square_plot : ℕ := 289
def price_per_foot : ℕ := 60

-- Resulting theorem statement
theorem cost_of_building_fence_square_plot : 
  let side_length := Int.sqrt area_of_square_plot
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 4080 := 
by
  -- Placeholder for the actual proof
  sorry

end cost_of_building_fence_square_plot_l196_196062


namespace first_year_fee_correct_l196_196517

noncomputable def first_year_fee (n : ℕ) (annual_increase : ℕ) (sixth_year_fee : ℕ) : ℕ :=
  sixth_year_fee - (n - 1) * annual_increase

theorem first_year_fee_correct (n annual_increase sixth_year_fee value : ℕ) 
  (h_n : n = 6) (h_annual_increase : annual_increase = 10) 
  (h_sixth_year_fee : sixth_year_fee = 130) (h_value : value = 80) :
  first_year_fee n annual_increase sixth_year_fee = value :=
by {
  sorry
}

end first_year_fee_correct_l196_196517


namespace divisor_is_18_l196_196121

def dividend : ℕ := 165
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem divisor_is_18 (divisor : ℕ) : dividend = quotient * divisor + remainder → divisor = 18 :=
by sorry

end divisor_is_18_l196_196121


namespace point_lies_on_graph_l196_196604

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

theorem point_lies_on_graph (a : ℝ) : f (-a) = f (a) :=
by
  sorry

end point_lies_on_graph_l196_196604


namespace smallest_p_l196_196752

theorem smallest_p (p q : ℕ) (h1 : p + q = 2005) (h2 : (5:ℚ)/8 < p / q) (h3 : p / q < (7:ℚ)/8) : p = 772 :=
sorry

end smallest_p_l196_196752


namespace walls_per_room_is_8_l196_196759

-- Definitions and conditions
def total_rooms : Nat := 10
def green_rooms : Nat := 3 * total_rooms / 5
def purple_rooms : Nat := total_rooms - green_rooms
def purple_walls : Nat := 32
def walls_per_room : Nat := purple_walls / purple_rooms

-- Theorem to prove
theorem walls_per_room_is_8 : walls_per_room = 8 := by
  sorry

end walls_per_room_is_8_l196_196759


namespace mismatching_socks_count_l196_196495

-- Define the conditions given in the problem
def total_socks : ℕ := 65
def pairs_matching_ankle_socks : ℕ := 13
def pairs_matching_crew_socks : ℕ := 10

-- Define the calculated counts as per the conditions
def matching_ankle_socks : ℕ := pairs_matching_ankle_socks * 2
def matching_crew_socks : ℕ := pairs_matching_crew_socks * 2
def total_matching_socks : ℕ := matching_ankle_socks + matching_crew_socks

-- The statement to prove
theorem mismatching_socks_count : total_socks - total_matching_socks = 19 := by
  sorry

end mismatching_socks_count_l196_196495


namespace find_tuesday_temp_l196_196526

variable (temps : List ℝ) (avg : ℝ) (len : ℕ) 

theorem find_tuesday_temp (h1 : temps = [99.1, 98.2, 99.3, 99.8, 99, 98.9, tuesday_temp])
                         (h2 : avg = 99)
                         (h3 : len = 7)
                         (h4 : (temps.sum / len) = avg) :
                         tuesday_temp = 98.7 := 
sorry

end find_tuesday_temp_l196_196526


namespace royal_children_l196_196693

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end royal_children_l196_196693


namespace find_positive_integer_n_l196_196598

theorem find_positive_integer_n (n : ℕ) (h₁ : 200 % n = 5) (h₂ : 395 % n = 5) : n = 13 :=
sorry

end find_positive_integer_n_l196_196598


namespace apple_cost_price_l196_196696

theorem apple_cost_price (SP : ℝ) (loss_ratio : ℝ) (CP : ℝ) (h1 : SP = 18) (h2 : loss_ratio = 1/6) (h3 : SP = CP - loss_ratio * CP) : CP = 21.6 :=
by
  sorry

end apple_cost_price_l196_196696


namespace quadratic_complete_the_square_l196_196489

theorem quadratic_complete_the_square :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 + 1500 * x + 1500 = (x + b) ^ 2 + c)
      ∧ b = 750
      ∧ c = -748 * 750
      ∧ c / b = -748 := 
by {
  sorry
}

end quadratic_complete_the_square_l196_196489


namespace negate_one_even_l196_196564

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_one_even (a b c : ℕ) :
  (∃! x, x = a ∨ x = b ∨ x = c ∧ is_even x) ↔
  (∃ x y, x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧
    x ≠ y ∧ is_even x ∧ is_even y) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
by {
  sorry
}

end negate_one_even_l196_196564


namespace distance_between_andrey_and_valentin_l196_196221

-- Definitions based on conditions
def speeds_relation_andrey_boris (a b : ℝ) := b = 0.94 * a
def speeds_relation_boris_valentin (b c : ℝ) := c = 0.95 * b

theorem distance_between_andrey_and_valentin
  (a b c : ℝ)
  (h1 : speeds_relation_andrey_boris a b)
  (h2 : speeds_relation_boris_valentin b c)
  : 1000 - 1000 * c / a = 107 :=
by
  sorry

end distance_between_andrey_and_valentin_l196_196221


namespace only_positive_integer_a_squared_plus_2a_is_perfect_square_l196_196753

/-- Prove that the only positive integer \( a \) for which \( a^2 + 2a \) is a perfect square is \( a = 0 \). -/
theorem only_positive_integer_a_squared_plus_2a_is_perfect_square :
  ∀ (a : ℕ), (∃ (k : ℕ), a^2 + 2*a = k^2) → a = 0 :=
by
  intro a h
  sorry

end only_positive_integer_a_squared_plus_2a_is_perfect_square_l196_196753


namespace combined_age_in_ten_years_l196_196150

theorem combined_age_in_ten_years (B A: ℕ) (hA : A = 20) (h1: A + 10 = 2 * (B + 10)): 
  (A + 10) + (B + 10) = 45 := 
by
  sorry

end combined_age_in_ten_years_l196_196150


namespace num_values_satisfying_g_g_x_eq_4_l196_196104

def g (x : ℝ) : ℝ := sorry

theorem num_values_satisfying_g_g_x_eq_4 
  (h1 : g (-2) = 4)
  (h2 : g (2) = 4)
  (h3 : g (4) = 4)
  (h4 : ∀ x, g (x) ≠ -2)
  (h5 : ∃! x, g (x) = 2) 
  (h6 : ∃! x, g (x) = 4) 
  : ∃! x1 x2, g (g x1) = 4 ∧ g (g x2) = 4 ∧ x1 ≠ x2 :=
by
  sorry

end num_values_satisfying_g_g_x_eq_4_l196_196104


namespace cars_cleaned_per_day_l196_196802

theorem cars_cleaned_per_day
  (money_per_car : ℕ)
  (total_money : ℕ)
  (days : ℕ)
  (h1 : money_per_car = 5)
  (h2 : total_money = 2000)
  (h3 : days = 5) :
  (total_money / (money_per_car * days)) = 80 := by
  sorry

end cars_cleaned_per_day_l196_196802


namespace jackson_miles_l196_196220

theorem jackson_miles (beka_miles jackson_miles : ℕ) (h1 : beka_miles = 873) (h2 : beka_miles = jackson_miles + 310) : jackson_miles = 563 := by
  sorry

end jackson_miles_l196_196220


namespace exist_positive_m_l196_196731

theorem exist_positive_m {n p q : ℕ} (hn_pos : 0 < n) (hp_prime : Prime p) (hq_prime : Prime q) 
  (h1 : pq ∣ n ^ p + 2) (h2 : n + 2 ∣ n ^ p + q ^ p) : ∃ m : ℕ, q ∣ 4 ^ m * n + 2 := 
sorry

end exist_positive_m_l196_196731


namespace percentage_of_copper_is_correct_l196_196994

-- Defining the conditions
def total_weight := 100.0
def weight_20_percent_alloy := 30.0
def weight_27_percent_alloy := total_weight - weight_20_percent_alloy

def percentage_20 := 0.20
def percentage_27 := 0.27

def copper_20 := percentage_20 * weight_20_percent_alloy
def copper_27 := percentage_27 * weight_27_percent_alloy
def total_copper := copper_20 + copper_27

-- The statement to be proved
def percentage_copper := (total_copper / total_weight) * 100

-- The theorem to prove
theorem percentage_of_copper_is_correct : percentage_copper = 24.9 := by sorry

end percentage_of_copper_is_correct_l196_196994


namespace reciprocal_of_repeating_decimal_6_l196_196484

-- Define a repeating decimal .\overline{6}
noncomputable def repeating_decimal_6 : ℚ := 2 / 3

-- The theorem statement: reciprocal of .\overline{6} is 3/2
theorem reciprocal_of_repeating_decimal_6 :
  repeating_decimal_6⁻¹ = (3 / 2) :=
sorry

end reciprocal_of_repeating_decimal_6_l196_196484


namespace sum_of_areas_of_tangent_circles_l196_196342

theorem sum_of_areas_of_tangent_circles :
  ∃ r s t : ℝ, r > 0 ∧ s > 0 ∧ t > 0 ∧
    (r + s = 3) ∧
    (r + t = 4) ∧
    (s + t = 5) ∧
    π * (r^2 + s^2 + t^2) = 14 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l196_196342


namespace correct_conditions_for_cubic_eq_single_root_l196_196397

noncomputable def hasSingleRealRoot (a b : ℝ) : Prop :=
  let f := λ x : ℝ => x^3 - a * x + b
  let f' := λ x : ℝ => 3 * x^2 - a
  ∀ (x y : ℝ), f' x = 0 → f' y = 0 → x = y

theorem correct_conditions_for_cubic_eq_single_root :
  (hasSingleRealRoot 0 2) ∧ 
  (hasSingleRealRoot (-3) 2) ∧ 
  (hasSingleRealRoot 3 (-3)) :=
  by 
    sorry

end correct_conditions_for_cubic_eq_single_root_l196_196397


namespace employed_males_population_percentage_l196_196660

-- Define the conditions of the problem
variables (P : Type) (population : ℝ) (employed_population : ℝ) (employed_females : ℝ)

-- Assume total population is 100
def total_population : ℝ := 100

-- 70 percent of the population are employed
def employed_population_percentage : ℝ := total_population * 0.70

-- 70 percent of the employed people are females
def employed_females_percentage : ℝ := employed_population_percentage * 0.70

-- 21 percent of the population are employed males
def employed_males_percentage : ℝ := 21

-- Main statement to be proven
theorem employed_males_population_percentage :
  employed_males_percentage = ((employed_population_percentage - employed_females_percentage) / total_population) * 100 :=
sorry

end employed_males_population_percentage_l196_196660


namespace probability_green_marbles_correct_l196_196117

noncomputable def probability_of_two_green_marbles : ℚ :=
  let total_marbles := 12
  let green_marbles := 7
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green

theorem probability_green_marbles_correct :
  probability_of_two_green_marbles = 7 / 22 := by
    sorry

end probability_green_marbles_correct_l196_196117


namespace neg_pow_eq_pow_four_l196_196018

variable (a : ℝ)

theorem neg_pow_eq_pow_four (a : ℝ) : (-a)^4 = a^4 :=
sorry

end neg_pow_eq_pow_four_l196_196018


namespace marble_sharing_l196_196912

theorem marble_sharing 
  (total_marbles : ℕ) 
  (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 30) 
  (h2 : marbles_per_friend = 6) : 
  total_marbles / marbles_per_friend = 5 := 
by 
  sorry

end marble_sharing_l196_196912


namespace magician_ball_count_l196_196294

theorem magician_ball_count (k : ℕ) : ∃ k : ℕ, 6 * k + 7 = 1993 :=
by sorry

end magician_ball_count_l196_196294


namespace arithmetic_square_root_of_16_l196_196479

theorem arithmetic_square_root_of_16 : ∃! (x : ℝ), x^2 = 16 ∧ x ≥ 0 :=
by
  sorry

end arithmetic_square_root_of_16_l196_196479


namespace probability_both_tell_truth_l196_196024

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_tell_truth (hA : P_A = 0.75) (hB : P_B = 0.60) : P_A * P_B = 0.45 :=
by
  rw [hA, hB]
  norm_num

end probability_both_tell_truth_l196_196024


namespace find_y_l196_196395

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of the compound C6HyO7
def molecular_weight : ℝ := 192

-- Define the contribution of Carbon and Oxygen
def contribution_C : ℝ := 6 * atomic_weight_C
def contribution_O : ℝ := 7 * atomic_weight_O

-- The proof statement
theorem find_y (y : ℕ) :
  molecular_weight = contribution_C + y * atomic_weight_H + contribution_O → y = 8 :=
by
  sorry

end find_y_l196_196395


namespace problem1_problem2_l196_196900

-- Problem 1
theorem problem1 (x y : ℝ) :
  2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n :=
by sorry

end problem1_problem2_l196_196900


namespace minimum_detectors_required_l196_196376

/-- There is a cube with each face divided into 4 identical square cells, making a total of 24 cells.
Oleg wants to mark 8 cells with invisible ink such that no two marked cells share a side.
Rustem wants to place detectors in the cells so that all marked cells can be identified. -/
def minimum_detectors_to_identify_all_marked_cells (total_cells: ℕ) (marked_cells: ℕ) 
  (cells_per_face: ℕ) (faces: ℕ) : ℕ :=
  if total_cells = faces * cells_per_face ∧ marked_cells = 8 then 16 else 0

theorem minimum_detectors_required :
  minimum_detectors_to_identify_all_marked_cells 24 8 4 6 = 16 :=
by
  sorry

end minimum_detectors_required_l196_196376


namespace range_of_m_l196_196827

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ (x^3 - 3 * x + m = 0)) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end range_of_m_l196_196827


namespace free_cytosine_molecules_req_l196_196622

-- Definition of conditions
def DNA_base_pairs := 500
def AT_percentage := 34 / 100
def CG_percentage := 1 - AT_percentage

-- The total number of bases
def total_bases := 2 * DNA_base_pairs

-- The number of C or G bases
def CG_bases := total_bases * CG_percentage

-- Finally, the total number of free cytosine deoxyribonucleotide molecules 
def free_cytosine_molecules := 2 * CG_bases

-- Problem statement: Prove that the number of free cytosine deoxyribonucleotide molecules required is 1320
theorem free_cytosine_molecules_req : free_cytosine_molecules = 1320 :=
by
  -- conditions are defined, the proof is omitted
  sorry

end free_cytosine_molecules_req_l196_196622


namespace frenchwoman_present_l196_196710

theorem frenchwoman_present
    (M_F M_R W_R : ℝ)
    (condition_1 : M_F > M_R + W_R)
    (condition_2 : W_R > M_F + M_R) 
    : false :=
by
  -- We would assume the opposite of what we know to lead to a contradiction here.
  -- This is a placeholder to indicate the proof should lead to a contradiction.
  sorry

end frenchwoman_present_l196_196710


namespace consecutive_integers_sum_to_thirty_unique_sets_l196_196949

theorem consecutive_integers_sum_to_thirty_unique_sets :
  (∃ a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60) ↔ ∃! a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60 :=
by
  sorry

end consecutive_integers_sum_to_thirty_unique_sets_l196_196949


namespace child_b_share_l196_196446

def total_money : ℕ := 4320

def ratio_parts : List ℕ := [2, 3, 4, 5, 6]

def parts_sum (parts : List ℕ) : ℕ :=
  parts.foldl (· + ·) 0

def value_of_one_part (total : ℕ) (parts : ℕ) : ℕ :=
  total / parts

def b_share (value_per_part : ℕ) (b_parts : ℕ) : ℕ :=
  value_per_part * b_parts

theorem child_b_share :
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  b_share one_part_value b_parts = 648 := by
  let total_money := 4320
  let ratio_parts := [2, 3, 4, 5, 6]
  let total_parts := parts_sum ratio_parts
  let one_part_value := value_of_one_part total_money total_parts
  let b_parts := 3
  show b_share one_part_value b_parts = 648
  sorry

end child_b_share_l196_196446


namespace general_term_of_sequence_l196_196370

theorem general_term_of_sequence (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ n, a (n + 1) = a n + n + 1) :
  ∀ n, a n = (n^2 + n + 2) / 2 :=
by 
  sorry

end general_term_of_sequence_l196_196370


namespace highest_value_of_a_for_divisibility_l196_196542

/-- Given a number in the format of 365a2_, where 'a' is a digit (0 through 9),
prove that the highest value of 'a' that makes the number divisible by 8 is 9. -/
theorem highest_value_of_a_for_divisibility :
  ∃ (a : ℕ), a ≤ 9 ∧ (∃ (d : ℕ), d < 10 ∧ (365 * 100 + a * 10 + 20 + d) % 8 = 0 ∧ a = 9) :=
sorry

end highest_value_of_a_for_divisibility_l196_196542


namespace max_min_value_l196_196386

def f (x t : ℝ) : ℝ := x^2 - 2 * t * x + t

theorem max_min_value : 
  ∀ t : ℝ, (-1 ≤ t ∧ t ≤ 1) →
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t) →
  (∃ t : ℝ, (-1 ≤ t ∧ t ≤ 1) ∧ ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → f x t ≥ -t^2 + t ∧ -t^2 + t = 1/4) :=
sorry

end max_min_value_l196_196386


namespace coeffs_equal_implies_a_plus_b_eq_4_l196_196984

theorem coeffs_equal_implies_a_plus_b_eq_4 (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq_coeffs : (Nat.choose 2000 1998) * (a ^ 2) * (b ^ 1998) = (Nat.choose 2000 1997) * (a ^ 3) * (b ^ 1997)) :
  a + b = 4 := 
sorry

end coeffs_equal_implies_a_plus_b_eq_4_l196_196984


namespace emani_money_l196_196022

def emani_has_30_more (E H : ℝ) : Prop := E = H + 30
def equal_share (E H : ℝ) : Prop := (E + H) / 2 = 135

theorem emani_money (E H : ℝ) (h1: emani_has_30_more E H) (h2: equal_share E H) : E = 150 :=
by
  sorry

end emani_money_l196_196022


namespace cos_5theta_l196_196418

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (5 * θ) = 241/243 :=
by
  sorry

end cos_5theta_l196_196418


namespace rowing_distance_correct_l196_196640

variable (D : ℝ) -- distance to the place
variable (speed_in_still_water : ℝ := 10) -- rowing speed in still water
variable (current_speed : ℝ := 2) -- speed of the current
variable (total_time : ℝ := 30) -- total time for round trip
variable (effective_speed_with_current : ℝ := speed_in_still_water + current_speed) -- effective speed with current
variable (effective_speed_against_current : ℝ := speed_in_still_water - current_speed) -- effective speed against current

theorem rowing_distance_correct : 
  D / effective_speed_with_current + D / effective_speed_against_current = total_time → 
  D = 144 := 
by
  intros h
  sorry

end rowing_distance_correct_l196_196640


namespace parabola_directrix_l196_196075

theorem parabola_directrix (x y : ℝ) (h : x^2 + 12 * y = 0) : y = 3 :=
sorry

end parabola_directrix_l196_196075


namespace possible_values_of_a_l196_196658

-- Define the sets P and Q under the conditions given
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Prove that if Q ⊆ P, then a ∈ {0, 1/3, -1/2}
theorem possible_values_of_a (a : ℝ) (h : Q a ⊆ P) : a = 0 ∨ a = 1/3 ∨ a = -1/2 :=
sorry

end possible_values_of_a_l196_196658


namespace average_discount_rate_l196_196985

theorem average_discount_rate
  (bag_marked_price : ℝ) (bag_sold_price : ℝ)
  (shoes_marked_price : ℝ) (shoes_sold_price : ℝ)
  (jacket_marked_price : ℝ) (jacket_sold_price : ℝ)
  (h_bag : bag_marked_price = 80) (h_bag_sold : bag_sold_price = 68)
  (h_shoes : shoes_marked_price = 120) (h_shoes_sold : shoes_sold_price = 96)
  (h_jacket : jacket_marked_price = 150) (h_jacket_sold : jacket_sold_price = 135) : 
  (15 : ℝ) =
  (((bag_marked_price - bag_sold_price) / bag_marked_price * 100) + 
   ((shoes_marked_price - shoes_sold_price) / shoes_marked_price * 100) + 
   ((jacket_marked_price - jacket_sold_price) / jacket_marked_price * 100)) / 3 :=
by {
  sorry
}

end average_discount_rate_l196_196985


namespace jerusha_and_lottie_earnings_l196_196776

theorem jerusha_and_lottie_earnings :
  let J := 68
  let L := J / 4
  J + L = 85 := 
by
  sorry

end jerusha_and_lottie_earnings_l196_196776


namespace problem1_problem2_l196_196642

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end problem1_problem2_l196_196642


namespace polygon_sides_from_diagonals_l196_196165

theorem polygon_sides_from_diagonals (n : ℕ) (h : ↑((n * (n - 3)) / 2) = 14) : n = 7 :=
by
  sorry

end polygon_sides_from_diagonals_l196_196165


namespace arithmetic_sequence_sum_l196_196459

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : a 8 + a 10 = 2) : 
  (17 * (a 1 + a 17) / 2) = 17 := by
sorry

end arithmetic_sequence_sum_l196_196459


namespace number_of_solutions_l196_196155

open Real

-- Define main condition
def condition (θ : ℝ) : Prop := sin θ * tan θ = 2 * (cos θ)^2

-- Define the interval and exclusions
def valid_theta (θ : ℝ) : Prop := 
  0 ≤ θ ∧ θ ≤ 2 * π ∧ ¬ ( ∃ k : ℤ, (θ = k * (π/2)) )

-- Define the set of thetas that satisfy both the condition and the valid interval
def valid_solutions (θ : ℝ) : Prop := valid_theta θ ∧ condition θ

-- Formal statement of the problem
theorem number_of_solutions : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_solutions θ) ∧ (s.card = 4) := by
  sorry

end number_of_solutions_l196_196155


namespace compute_expression_l196_196744

theorem compute_expression :
  (75 * 1313 - 25 * 1313 + 50 * 1313 = 131300) :=
by
  sorry

end compute_expression_l196_196744


namespace sequence_limit_l196_196191

noncomputable def sequence_converges (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n > 1 ∧ a (n + 1) ^ 2 ≥ a n * a (n + 2)

theorem sequence_limit (a : ℕ → ℝ) (h : sequence_converges a) : 
  ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (Real.log (a (n + 1)) / Real.log (a n) - l) < ε := 
sorry

end sequence_limit_l196_196191


namespace range_of_a_l196_196647

theorem range_of_a (a : ℝ) (h : ∃ α β : ℝ, (α + β = -(a^2 - 1)) ∧ (α * β = a - 2) ∧ (1 < α ∧ β < 1) ∨ (α < 1 ∧ 1 < β)) :
  -2 < a ∧ a < 1 :=
sorry

end range_of_a_l196_196647


namespace paving_stone_width_l196_196757

theorem paving_stone_width
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_length : ℝ)
  (courtyard_area : ℝ) (paving_stone_area : ℝ)
  (width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16.5)
  (h3 : num_paving_stones = 99)
  (h4 : paving_stone_length = 2.5)
  (h5 : courtyard_area = courtyard_length * courtyard_width)
  (h6 : courtyard_area = 495)
  (h7 : paving_stone_area = courtyard_area / num_paving_stones)
  (h8 : paving_stone_area = 5)
  (h9 : paving_stone_area = paving_stone_length * width) :
  width = 2 := by
  sorry

end paving_stone_width_l196_196757


namespace scientific_notation_correct_l196_196044

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end scientific_notation_correct_l196_196044


namespace find_fx_for_neg_x_l196_196941

-- Let f be an odd function defined on ℝ 
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)

-- Given condition for x > 0
variable (h_pos : ∀ x, 0 < x → f x = x^2 + x - 1)

-- Problem: Prove that f(x) = -x^2 + x + 1 for x < 0
theorem find_fx_for_neg_x (x : ℝ) (h_neg : x < 0) : f x = -x^2 + x + 1 :=
sorry

end find_fx_for_neg_x_l196_196941


namespace locus_square_l196_196563

open Real

variables {x y c1 c2 d1 d2 : ℝ}

/-- The locus of points in a square -/
theorem locus_square (h_square: d1 < d2 ∧ c1 < c2) (h_x: d1 ≤ x ∧ x ≤ d2) (h_y: c1 ≤ y ∧ y ≤ c2) :
  |y - c1| + |y - c2| = |x - d1| + |x - d2| :=
by sorry

end locus_square_l196_196563


namespace probability_of_sine_inequality_l196_196490

open Set Real

noncomputable def probability_sine_inequality (x : ℝ) : Prop :=
  ∃ (μ : MeasureTheory.Measure ℝ), μ (Ioc (-3) 3) = 1 ∧
    μ {x | sin (π / 6 * x) ≥ 1 / 2} = 1 / 3

theorem probability_of_sine_inequality : probability_sine_inequality x :=
by
  sorry

end probability_of_sine_inequality_l196_196490


namespace product_of_geometric_sequence_l196_196690

theorem product_of_geometric_sequence (x y z : ℝ) 
  (h_seq : ∃ r, x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) : 
  1 * x * y * z * 4 = 32 :=
by
  sorry

end product_of_geometric_sequence_l196_196690


namespace tips_collected_l196_196345

-- Definitions based on conditions
def total_collected : ℕ := 240
def hourly_wage : ℕ := 10
def hours_worked : ℕ := 19

-- Correct answer translated into a proof problem
theorem tips_collected : total_collected - (hours_worked * hourly_wage) = 50 := by
  sorry

end tips_collected_l196_196345


namespace problem1_l196_196020

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end problem1_l196_196020


namespace diagonal_in_parallelogram_l196_196903

-- Define the conditions of the problem
variable (A B C D M : Point)
variable (parallelogram : Parallelogram A B C D)
variable (height_bisects_side : Midpoint M A D)
variable (height_length : Distance B M = 2)
variable (acute_angle_30 : Angle A B D = 30)

-- Define the theorem based on the conditions
theorem diagonal_in_parallelogram (h1 : parallelogram) (h2 : height_bisects_side)
  (h3 : height_length) (h4 : acute_angle_30) : 
  ∃ (BD_length : ℝ) (angle1 angle2 : ℝ), BD_length = 4 ∧ angle1 = 30 ∧ angle2 = 120 := 
sorry

end diagonal_in_parallelogram_l196_196903


namespace minimize_sum_of_squares_of_perpendiculars_l196_196277

open Real

variable {α β c : ℝ} -- angles and side length

theorem minimize_sum_of_squares_of_perpendiculars
    (habc : α + β = π)
    (P : ℝ)
    (AP BP : ℝ)
    (x : AP + BP = c)
    (u : ℝ)
    (v : ℝ)
    (hAP : AP = P)
    (hBP : BP = c - P)
    (hu : u = P * sin α)
    (hv : v = (c - P) * sin β)
    (f : ℝ)
    (hf : f = (P * sin α)^2 + ((c - P) * sin β)^2):
  (AP / BP = (sin β)^2 / (sin α)^2) := sorry

end minimize_sum_of_squares_of_perpendiculars_l196_196277


namespace minimum_value_of_expression_l196_196789

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value_expr a b = -2031948.5 :=
  sorry

end minimum_value_of_expression_l196_196789


namespace total_payment_correct_l196_196371

def rate_per_kg_grapes := 68
def quantity_grapes := 7
def rate_per_kg_mangoes := 48
def quantity_mangoes := 9

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes

def total_amount_paid := cost_grapes + cost_mangoes

theorem total_payment_correct :
  total_amount_paid = 908 := by
  sorry

end total_payment_correct_l196_196371


namespace fraction_squares_sum_l196_196242

theorem fraction_squares_sum (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := 
sorry

end fraction_squares_sum_l196_196242


namespace find_f_neg1_l196_196749

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -2^(-x) + 2*x + 1

theorem find_f_neg1 : f (-1) = -3 :=
by
  -- The proof is omitted.
  sorry

end find_f_neg1_l196_196749


namespace product_of_integers_around_sqrt_50_l196_196874

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end product_of_integers_around_sqrt_50_l196_196874


namespace number_of_different_towers_l196_196513

theorem number_of_different_towers
  (red blue yellow : ℕ)
  (total_height : ℕ)
  (total_cubes : ℕ)
  (discarded_cubes : ℕ)
  (ways_to_leave_out : ℕ)
  (multinomial_coefficient : ℕ) : 
  red = 3 → blue = 4 → yellow = 5 → total_height = 10 → total_cubes = 12 → discarded_cubes = 2 →
  ways_to_leave_out = 66 → multinomial_coefficient = 4200 →
  (ways_to_leave_out * multinomial_coefficient) = 277200 :=
by
  -- proof skipped
  sorry

end number_of_different_towers_l196_196513


namespace find_const_s_l196_196682

noncomputable def g (x : ℝ) (a b c d : ℝ) := (x + 2 * a) * (x + 2 * b) * (x + 2 * c) * (x + 2 * d)

theorem find_const_s (a b c d : ℝ) (p q r s : ℝ) (h1 : 1 + p + q + r + s = 4041)
  (h2 : g 1 a b c d = 1 + p + q + r + s) :
  s = 3584 := 
sorry

end find_const_s_l196_196682


namespace swimming_club_total_members_l196_196846

def valid_total_members (total : ℕ) : Prop :=
  ∃ (J S V : ℕ),
    3 * S = 2 * J ∧
    5 * V = 2 * S ∧
    total = J + S + V

theorem swimming_club_total_members :
  valid_total_members 58 := by
  sorry

end swimming_club_total_members_l196_196846


namespace area_of_quadrilateral_l196_196926

theorem area_of_quadrilateral (d h1 h2 : ℝ) (hd : d = 20) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  (1 / 2) * d * (h1 + h2) = 150 :=
by
  rw [hd, hh1, hh2]
  norm_num

end area_of_quadrilateral_l196_196926


namespace count_two_digit_powers_of_three_l196_196466

theorem count_two_digit_powers_of_three : 
  ∃ (n1 n2 : ℕ), 10 ≤ 3^n1 ∧ 3^n1 < 100 ∧ 10 ≤ 3^n2 ∧ 3^n2 < 100 ∧ n1 ≠ n2 ∧ ∀ n : ℕ, (10 ≤ 3^n ∧ 3^n < 100) → (n = n1 ∨ n = n2) ∧ n1 = 3 ∧ n2 = 4 := by
  sorry

end count_two_digit_powers_of_three_l196_196466


namespace system_solution_l196_196727

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 3) : x - y = 3 :=
by
  -- proof goes here
  sorry

end system_solution_l196_196727


namespace filling_rate_in_cubic_meters_per_hour_l196_196210

def barrels_per_minute_filling_rate : ℝ := 3
def liters_per_barrel : ℝ := 159
def liters_per_cubic_meter : ℝ := 1000
def minutes_per_hour : ℝ := 60

theorem filling_rate_in_cubic_meters_per_hour :
  (barrels_per_minute_filling_rate * liters_per_barrel / liters_per_cubic_meter * minutes_per_hour) = 28.62 :=
sorry

end filling_rate_in_cubic_meters_per_hour_l196_196210


namespace Tyler_has_200_puppies_l196_196576

-- Define the number of dogs
def numDogs : ℕ := 25

-- Define the number of puppies per dog
def puppiesPerDog : ℕ := 8

-- Define the total number of puppies
def totalPuppies : ℕ := numDogs * puppiesPerDog

-- State the theorem we want to prove
theorem Tyler_has_200_puppies : totalPuppies = 200 := by
  exact (by norm_num : 25 * 8 = 200)

end Tyler_has_200_puppies_l196_196576


namespace quadratic_coefficients_l196_196711

theorem quadratic_coefficients :
  ∀ (a b c : ℤ), (2 * a * a - b * a - 5 = 0) → (a = 2 ∧ b = -1) :=
by
  intros a b c H
  sorry

end quadratic_coefficients_l196_196711


namespace sin_A_over_1_minus_cos_A_l196_196404

variable {a b c : ℝ} -- Side lengths of the triangle
variable {A B C : ℝ} -- Angles opposite to the sides

theorem sin_A_over_1_minus_cos_A 
  (h_area : 0.5 * b * c * Real.sin A = a^2 - (b - c)^2) :
  Real.sin A / (1 - Real.cos A) = 3 :=
sorry

end sin_A_over_1_minus_cos_A_l196_196404


namespace win_sector_area_l196_196431

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 12) (h_p : p = 1 / 3) :
  ∃ A : ℝ, A = 48 * π :=
by {
  sorry
}

end win_sector_area_l196_196431


namespace find_a1_l196_196981

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 8 = 2 ∧ ∀ n, a (n + 1) = 1 / (1 - a n)

theorem find_a1 (a : ℕ → ℝ) (h : seq a) : a 1 = 1/2 := by
sorry

end find_a1_l196_196981


namespace fleas_after_treatment_l196_196728

theorem fleas_after_treatment
  (F : ℕ)  -- F is the number of fleas the dog has left after the treatments
  (half_fleas : ℕ → ℕ)  -- Function representing halving fleas
  (initial_fleas := F + 210)  -- Initial number of fleas before treatment
  (half_fleas_def : ∀ n, half_fleas n = n / 2)  -- Definition of half_fleas function
  (condition : F = (half_fleas (half_fleas (half_fleas (half_fleas initial_fleas)))))  -- Condition given in the problem
  :
  F = 14 := 
  sorry

end fleas_after_treatment_l196_196728


namespace grazing_months_for_b_l196_196068

/-
  We define the problem conditions and prove that b put his oxen for grazing for 5 months.
-/

theorem grazing_months_for_b (x : ℕ) :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let c_oxen := 15
  let c_months := 3
  let total_rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * x
  let c_ox_months := c_oxen * c_months
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  (c_share : ℚ) / total_rent = (c_ox_months : ℚ) / total_ox_months →
  x = 5 :=
by
  sorry

end grazing_months_for_b_l196_196068


namespace apples_in_first_group_l196_196791

variable (A O : ℝ) (X : ℕ)

-- Given conditions
axiom h1 : A = 0.21
axiom h2 : X * A + 3 * O = 1.77
axiom h3 : 2 * A + 5 * O = 1.27 

-- Goal: Prove that the number of apples in the first group is 6
theorem apples_in_first_group : X = 6 := 
by 
  sorry

end apples_in_first_group_l196_196791


namespace sector_properties_l196_196577

-- Definitions for the conditions
def central_angle (α : ℝ) : Prop := α = 2 * Real.pi / 3

def radius (r : ℝ) : Prop := r = 6

def sector_perimeter (l r : ℝ) : Prop := l + 2 * r = 20

-- The statement encapsulating the proof problem
theorem sector_properties :
  (central_angle (2 * Real.pi / 3) ∧ radius 6 →
    ∃ l S, l = 4 * Real.pi ∧ S = 12 * Real.pi) ∧
  (∃ l r, sector_perimeter l r ∧ 
    ∃ α S, α = 2 ∧ S = 25) := by
  sorry

end sector_properties_l196_196577


namespace total_animals_in_jacobs_flock_l196_196224

-- Define the conditions of the problem
def one_third_of_animals_are_goats (total goats : ℕ) : Prop := 
  3 * goats = total

def twelve_more_sheep_than_goats (goats sheep : ℕ) : Prop :=
  sheep = goats + 12

-- Define the main theorem to prove
theorem total_animals_in_jacobs_flock : 
  ∃ total goats sheep : ℕ, one_third_of_animals_are_goats total goats ∧ 
                           twelve_more_sheep_than_goats goats sheep ∧ 
                           total = 36 := 
by
  sorry

end total_animals_in_jacobs_flock_l196_196224


namespace tangent_x_axis_l196_196765

noncomputable def curve (k : ℝ) : ℝ → ℝ := λ x => Real.log x - k * x + 3

theorem tangent_x_axis (k : ℝ) : 
  ∃ t : ℝ, curve k t = 0 ∧ deriv (curve k) t = 0 → k = Real.exp 2 :=
by
  sorry

end tangent_x_axis_l196_196765


namespace complex_power_difference_l196_196425

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 18 - (1 - i) ^ 18 = 1024 * i :=
by
  sorry

end complex_power_difference_l196_196425


namespace contrapositive_iff_l196_196430

theorem contrapositive_iff (a b : ℝ) :
  (a^2 - b^2 = 0 → a = b) ↔ (a ≠ b → a^2 - b^2 ≠ 0) :=
by
  sorry

end contrapositive_iff_l196_196430


namespace sam_bought_9_cans_l196_196672

-- Definitions based on conditions
def spent_amount_dollars := 20 - 5.50
def spent_amount_cents := 1450 -- to avoid floating point precision issues we equate to given value in cents
def coupon_discount_cents := 5 * 25
def total_cost_no_discount := spent_amount_cents + coupon_discount_cents
def cost_per_can := 175

-- Main statement to prove
theorem sam_bought_9_cans : total_cost_no_discount / cost_per_can = 9 :=
by
  sorry -- Proof goes here

end sam_bought_9_cans_l196_196672


namespace find_the_number_l196_196040

theorem find_the_number (x : ℝ) (h : 150 - x = x + 68) : x = 41 :=
sorry

end find_the_number_l196_196040


namespace number_of_players_l196_196300

variable (total_socks : ℕ) (socks_per_player : ℕ)

theorem number_of_players (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 := by
  -- proof steps will go here
  sorry

end number_of_players_l196_196300


namespace correct_option_l196_196556

-- Conditions
def option_A (a : ℕ) : Prop := (a^5)^2 = a^7
def option_B (a : ℕ) : Prop := a + 2 * a = 3 * a^2
def option_C (a : ℕ) : Prop := (2 * a)^3 = 6 * a^3
def option_D (a : ℕ) : Prop := a^6 / a^2 = a^4

-- Theorem statement
theorem correct_option (a : ℕ) : ¬ option_A a ∧ ¬ option_B a ∧ ¬ option_C a ∧ option_D a := by
  sorry

end correct_option_l196_196556


namespace func_value_sum_l196_196847

noncomputable def f (x : ℝ) : ℝ :=
  -x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2 + 1

theorem func_value_sum : f (1/2) + f (-1/2) = 2 :=
by
  sorry

end func_value_sum_l196_196847


namespace no_integer_solution_l196_196133

theorem no_integer_solution :
  ∀ (x y : ℤ), ¬(x^4 + x + y^2 = 3 * y - 1) :=
by
  intros x y
  sorry

end no_integer_solution_l196_196133


namespace cylinder_unoccupied_volume_l196_196447

theorem cylinder_unoccupied_volume (r h_cylinder h_cone : ℝ) 
  (h : r = 10 ∧ h_cylinder = 30 ∧ h_cone = 15) :
  (π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π) :=
by
  rcases h with ⟨rfl, rfl, rfl⟩
  simp
  sorry

end cylinder_unoccupied_volume_l196_196447


namespace product_of_remainders_one_is_one_l196_196136

theorem product_of_remainders_one_is_one (a b : ℕ) (h1 : a % 3 = 1) (h2 : b % 3 = 1) : (a * b) % 3 = 1 :=
sorry

end product_of_remainders_one_is_one_l196_196136


namespace hyperbola_lattice_points_count_l196_196468

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - 2 * y^2 = 2000^2

def count_lattice_points (points : List (ℤ × ℤ)) : ℕ :=
  points.length

theorem hyperbola_lattice_points_count : count_lattice_points [(2000, 0), (-2000, 0)] = 2 :=
by
  sorry

end hyperbola_lattice_points_count_l196_196468


namespace find_sandwich_cost_l196_196454

theorem find_sandwich_cost (S : ℝ) :
  3 * S + 2 * 4 = 26 → S = 6 :=
by
  intro h
  sorry

end find_sandwich_cost_l196_196454


namespace intersection_of_M_and_N_l196_196691

theorem intersection_of_M_and_N (x : ℝ) :
  {x | x > 1} ∩ {x | x^2 - 2 * x < 0} = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l196_196691


namespace tablecloth_width_l196_196627

theorem tablecloth_width (length_tablecloth : ℕ) (napkins_count : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_material : ℕ) (width_tablecloth : ℕ) :
  length_tablecloth = 102 →
  napkins_count = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_material = 5844 →
  total_material = length_tablecloth * width_tablecloth + napkins_count * (napkin_length * napkin_width) →
  width_tablecloth = 54 :=
by
  intros h1 h2 h3 h4 h5 h_eq
  sorry

end tablecloth_width_l196_196627


namespace transformed_data_properties_l196_196456

-- Definitions of the initial mean and variance
def initial_mean : ℝ := 2.8
def initial_variance : ℝ := 3.6

-- Definitions of transformation constants
def multiplier : ℝ := 2
def increment : ℝ := 60

-- New mean after transformation
def new_mean : ℝ := multiplier * initial_mean + increment

-- New variance after transformation
def new_variance : ℝ := (multiplier ^ 2) * initial_variance

-- Theorem statement
theorem transformed_data_properties :
  new_mean = 65.6 ∧ new_variance = 14.4 :=
by
  sorry

end transformed_data_properties_l196_196456


namespace average_playtime_in_minutes_l196_196880

noncomputable def lena_playtime_hours : ℝ := 3.5
noncomputable def lena_playtime_minutes : ℝ := lena_playtime_hours * 60
noncomputable def brother_playtime_minutes : ℝ := 1.2 * lena_playtime_minutes + 17
noncomputable def sister_playtime_minutes : ℝ := 1.5 * brother_playtime_minutes

theorem average_playtime_in_minutes :
  (lena_playtime_minutes + brother_playtime_minutes + sister_playtime_minutes) / 3 = 294.17 :=
by
  sorry

end average_playtime_in_minutes_l196_196880


namespace cost_percentage_l196_196443

variable (t b : ℝ)

def C := t * b ^ 4
def R := t * (2 * b) ^ 4

theorem cost_percentage : R = 16 * C := by
  sorry

end cost_percentage_l196_196443
