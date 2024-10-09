import Mathlib

namespace rank_best_buy_LMS_l1241_124143

theorem rank_best_buy_LMS (c_S q_S : ℝ) :
  let c_M := 1.75 * c_S
  let q_M := 1.1 * q_S
  let c_L := 1.25 * c_M
  let q_L := 1.5 * q_M
  (c_S / q_S) > (c_M / q_M) ∧ (c_M / q_M) > (c_L / q_L) :=
by
  sorry

end rank_best_buy_LMS_l1241_124143


namespace rhombus_diagonal_length_l1241_124195

theorem rhombus_diagonal_length
  (d2 : ℝ)
  (h1 : d2 = 20)
  (area : ℝ)
  (h2 : area = 150) :
  ∃ d1 : ℝ, d1 = 15 ∧ (area = (d1 * d2) / 2) := by
  sorry

end rhombus_diagonal_length_l1241_124195


namespace find_a_and_b_solve_inequality_l1241_124144

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, f x a b > 0 ↔ x < 0 ∨ x > 2) : a = -2 ∧ b = 0 :=
by sorry

theorem solve_inequality (a b : ℝ) (m : ℝ) (h1 : a = -2) (h2 : b = 0) :
  (∀ x : ℝ, f x a b < m^2 - 1 ↔ 
    (m = 0 → ∀ x : ℝ, false) ∧
    (m > 0 → (1 - m < x ∧ x < 1 + m)) ∧
    (m < 0 → (1 + m < x ∧ x < 1 - m))) :=
by sorry

end find_a_and_b_solve_inequality_l1241_124144


namespace solution_set_range_l1241_124154

theorem solution_set_range (x : ℝ) : 
  (2 * |x - 10| + 3 * |x - 20| ≤ 35) ↔ (9 ≤ x ∧ x ≤ 23) :=
sorry

end solution_set_range_l1241_124154


namespace lizette_third_quiz_score_l1241_124113

theorem lizette_third_quiz_score :
  ∀ (x : ℕ),
  (2 * 95 + x) / 3 = 94 → x = 92 :=
by
  intro x h
  have h1 : 2 * 95 = 190 := by norm_num
  have h2 : 3 * 94 = 282 := by norm_num
  sorry

end lizette_third_quiz_score_l1241_124113


namespace cost_of_sandwiches_and_smoothies_l1241_124186

-- Define the cost of sandwiches and smoothies
def sandwich_cost := 4
def smoothie_cost := 3

-- Define the discount applicable
def sandwich_discount := 1
def total_sandwiches := 6
def total_smoothies := 7

-- Calculate the effective cost per sandwich considering discount
def effective_sandwich_cost := if total_sandwiches > 4 then sandwich_cost - sandwich_discount else sandwich_cost

-- Calculate the total cost for sandwiches
def sandwiches_cost := total_sandwiches * effective_sandwich_cost

-- Calculate the total cost for smoothies
def smoothies_cost := total_smoothies * smoothie_cost

-- Calculate the total cost
def total_cost := sandwiches_cost + smoothies_cost

-- The main statement to prove
theorem cost_of_sandwiches_and_smoothies : total_cost = 39 := by
  -- skip the proof
  sorry

end cost_of_sandwiches_and_smoothies_l1241_124186


namespace range_of_d_l1241_124171

noncomputable def sn (n a1 d : ℝ) := (n / 2) * (2 * a1 + (n - 1) * d)

theorem range_of_d (a1 d : ℝ) (h_eq : (sn 2 a1 d) * (sn 4 a1 d) / 2 + (sn 3 a1 d) ^ 2 / 9 + 2 = 0) :
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end range_of_d_l1241_124171


namespace sum_first_five_terms_l1241_124116

theorem sum_first_five_terms (a1 a2 a3 : ℝ) (S5 : ℝ) 
  (h1 : a1 * a3 = 8 * a2)
  (h2 : (a1 + a2) = 24) :
  S5 = 31 :=
sorry

end sum_first_five_terms_l1241_124116


namespace minimum_value_7a_4b_l1241_124149

noncomputable def original_cond (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4

theorem minimum_value_7a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  original_cond a b ha hb → 7 * a + 4 * b = 9 / 4 :=
by
  sorry

end minimum_value_7a_4b_l1241_124149


namespace scientific_notation_of_0_0000205_l1241_124190

noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_0_0000205 :
  scientific_notation 0.0000205 = (2.05, -5) :=
sorry

end scientific_notation_of_0_0000205_l1241_124190


namespace total_miles_traveled_l1241_124115

noncomputable def distance_to_first_museum : ℕ := 5
noncomputable def distance_to_second_museum : ℕ := 15
noncomputable def distance_to_cultural_center : ℕ := 10
noncomputable def extra_detour : ℕ := 3

theorem total_miles_traveled : 
  (2 * (distance_to_first_museum + extra_detour) + 2 * distance_to_second_museum + 2 * distance_to_cultural_center) = 66 :=
  by
  sorry

end total_miles_traveled_l1241_124115


namespace area_outside_circle_of_equilateral_triangle_l1241_124173

noncomputable def equilateral_triangle_area_outside_circle {a : ℝ} (h : a > 0) : ℝ :=
  let S1 := a^2 * Real.sqrt 3 / 4
  let S2 := Real.pi * (a / 3)^2
  let S3 := (Real.pi * (a / 3)^2 / 6) - (a^2 * Real.sqrt 3 / 36)
  S1 - S2 + 3 * S3

theorem area_outside_circle_of_equilateral_triangle
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_area_outside_circle h = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 :=
sorry

end area_outside_circle_of_equilateral_triangle_l1241_124173


namespace father_payment_l1241_124181

variable (x y : ℤ)

theorem father_payment :
  5 * x - 3 * y = 24 :=
sorry

end father_payment_l1241_124181


namespace all_push_ups_total_l1241_124126

-- Definitions derived from the problem's conditions
def ZacharyPushUps := 47
def DavidPushUps := ZacharyPushUps + 15
def EmilyPushUps := DavidPushUps * 2
def TotalPushUps := ZacharyPushUps + DavidPushUps + EmilyPushUps

-- The statement to be proved
theorem all_push_ups_total : TotalPushUps = 233 := by
  sorry

end all_push_ups_total_l1241_124126


namespace product_of_large_integers_l1241_124114

theorem product_of_large_integers :
  ∃ A B : ℤ, A > 10^2009 ∧ B > 10^2009 ∧ A * B = 3^(4^5) + 4^(5^6) :=
by
  sorry

end product_of_large_integers_l1241_124114


namespace min_value_of_m_l1241_124172

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) := (Real.exp (-x) - Real.exp x) / 2

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → m * g x + h x ≥ 0) → m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1) :=
by
  intro h
  have key_ineq : ∀ x, -1 ≤ x ∧ x ≤ 1 → m ≥ 1 - 2 / (Real.exp (2 * x) + 1) := sorry
  sorry

end min_value_of_m_l1241_124172


namespace uncle_age_when_seokjin_is_12_l1241_124138

-- Definitions for the conditions
def mother_age_when_seokjin_born : ℕ := 32
def uncle_is_younger_by : ℕ := 3
def seokjin_age : ℕ := 12

-- Definition for the main hypothesis
theorem uncle_age_when_seokjin_is_12 :
  let mother_age_when_seokjin_is_12 := mother_age_when_seokjin_born + seokjin_age
  let uncle_age_when_seokjin_is_12 := mother_age_when_seokjin_is_12 - uncle_is_younger_by
  uncle_age_when_seokjin_is_12 = 41 :=
by
  sorry

end uncle_age_when_seokjin_is_12_l1241_124138


namespace max_x_plus_y_l1241_124134

theorem max_x_plus_y (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : x + y ≤ 2 * Real.sqrt (3) / 3 :=
sorry

end max_x_plus_y_l1241_124134


namespace manufacturing_cost_before_decrease_l1241_124168

def original_manufacturing_cost (P : ℝ) (C_now : ℝ) (profit_rate_now : ℝ) : ℝ :=
  P - profit_rate_now * P

theorem manufacturing_cost_before_decrease
  (P : ℝ)
  (C_now : ℝ)
  (profit_rate_now : ℝ)
  (profit_rate_original : ℝ)
  (H1 : C_now = P - profit_rate_now * P)
  (H2 : profit_rate_now = 0.50)
  (H3 : profit_rate_original = 0.20)
  (H4 : C_now = 50) :
  original_manufacturing_cost P C_now profit_rate_now = 80 :=
sorry

end manufacturing_cost_before_decrease_l1241_124168


namespace area_of_triangle_l1241_124194

variables (yellow_area green_area blue_area : ℝ)
variables (is_equilateral_triangle : Prop)
variables (centered_at_vertices : Prop)
variables (radius_less_than_height : Prop)

theorem area_of_triangle (h_yellow : yellow_area = 1000)
                        (h_green : green_area = 100)
                        (h_blue : blue_area = 1)
                        (h_triangle : is_equilateral_triangle)
                        (h_centered : centered_at_vertices)
                        (h_radius : radius_less_than_height) :
  ∃ (area : ℝ), area = 150 :=
by
  sorry

end area_of_triangle_l1241_124194


namespace sequence_term_position_l1241_124105

theorem sequence_term_position :
  ∃ n : ℕ, ∀ k : ℕ, (k = 7 + 6 * (n - 1)) → k = 2005 → n = 334 :=
by
  sorry

end sequence_term_position_l1241_124105


namespace matt_twice_james_age_in_5_years_l1241_124198

theorem matt_twice_james_age_in_5_years :
  (∃ x : ℕ, (3 + 27 = 30) ∧ (Matt_current_age = 65) ∧ 
  (Matt_age_in_x_years = Matt_current_age + x) ∧ 
  (James_age_in_x_years = James_current_age + x) ∧ 
  (Matt_age_in_x_years = 2 * James_age_in_x_years) → x = 5) :=
sorry

end matt_twice_james_age_in_5_years_l1241_124198


namespace axis_of_symmetry_of_function_l1241_124136

theorem axis_of_symmetry_of_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * Real.cos x - Real.sqrt 3 * Real.sin x)
  : ∃ k : ℤ, x = k * Real.pi - Real.pi / 6 ∧ x = Real.pi - Real.pi / 6 :=
sorry

end axis_of_symmetry_of_function_l1241_124136


namespace find_a_l1241_124141

noncomputable def f (a x : ℝ) : ℝ := a * x * (x - 2)^2

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∃ x : ℝ, f a x = 32) :
  a = 27 :=
sorry

end find_a_l1241_124141


namespace pairs_of_mittens_correct_l1241_124185

variables (pairs_of_plugs_added pairs_of_plugs_original plugs_total pairs_of_plugs_current pairs_of_mittens : ℕ)

theorem pairs_of_mittens_correct :
  pairs_of_plugs_added = 30 →
  plugs_total = 400 →
  pairs_of_plugs_current = plugs_total / 2 →
  pairs_of_plugs_current = pairs_of_plugs_original + pairs_of_plugs_added →
  pairs_of_mittens = pairs_of_plugs_original - 20 →
  pairs_of_mittens = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pairs_of_mittens_correct_l1241_124185


namespace food_drive_total_cans_l1241_124164

def total_cans_brought (M J R : ℕ) : ℕ := M + J + R

theorem food_drive_total_cans (M J R : ℕ) 
  (h1 : M = 4 * J) 
  (h2 : J = 2 * R + 5) 
  (h3 : M = 100) : 
  total_cans_brought M J R = 135 :=
by sorry

end food_drive_total_cans_l1241_124164


namespace nina_age_l1241_124110

theorem nina_age : ∀ (M L A N : ℕ), 
  (M = L - 5) → 
  (L = A + 6) → 
  (N = A + 2) → 
  (M = 16) → 
  N = 17 :=
by
  intros M L A N h1 h2 h3 h4
  sorry

end nina_age_l1241_124110


namespace combinatorial_proof_l1241_124140

noncomputable def combinatorial_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) : ℕ :=
  let summation_term (i : ℕ) := Nat.choose k i * Nat.choose n (m - i)
  List.sum (List.map summation_term (List.range (k + 1)))

theorem combinatorial_proof (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) :
  combinatorial_identity n m k h1 h2 h3 = Nat.choose (n + k) m :=
sorry

end combinatorial_proof_l1241_124140


namespace fraction_product_l1241_124167

theorem fraction_product : (1 / 2) * (1 / 3) * (1 / 6) * 120 = 10 / 3 :=
by
  sorry

end fraction_product_l1241_124167


namespace k_value_l1241_124162

open Real

noncomputable def k_from_roots (α β : ℝ) : ℝ := - (α + β)

theorem k_value (k : ℝ) (α β : ℝ) (h1 : α + β = -k) (h2 : α * β = 8) (h3 : (α+3) + (β+3) = k) (h4 : (α+3) * (β+3) = 12) : k = 3 :=
by
  -- Here we skip the proof as instructed.
  sorry

end k_value_l1241_124162


namespace range_of_a_l1241_124128

open Set

-- Define proposition p
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0

-- Define proposition q
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define negation of p
def not_p (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define negation of q
def not_q (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 1 → -3 ≤ x ∧ x ≤ 1) → a ∈ Icc (-3 : ℝ) (0 : ℝ) :=
by
  intro h
  -- skipped detailed proof
  sorry

end range_of_a_l1241_124128


namespace triangle_inequality_l1241_124111

variable {x y z : ℝ}
variable {A B C : ℝ}

theorem triangle_inequality (hA: A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π):
  x^2 + y^2 + z^2 ≥ 2 * y * z * Real.sin A + 2 * z * x * Real.sin B - 2 * x * y * Real.cos C := by
  sorry

end triangle_inequality_l1241_124111


namespace solve_fraction_equation_l1241_124119

-- Defining the function f
def f (x : ℝ) : ℝ := x + 4

-- Statement of the problem
theorem solve_fraction_equation (x : ℝ) :
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ↔ x = 2 / 5 := by
  sorry

end solve_fraction_equation_l1241_124119


namespace triangle_angles_ratio_l1241_124188

theorem triangle_angles_ratio (A B C : ℕ) 
  (hA : A = 20)
  (hB : B = 3 * A)
  (hSum : A + B + C = 180) :
  (C / A) = 5 := 
by
  sorry

end triangle_angles_ratio_l1241_124188


namespace gondor_laptop_earning_l1241_124199

theorem gondor_laptop_earning :
  ∃ L : ℝ, (3 * 10 + 5 * 10 + 2 * L + 4 * L = 200) → L = 20 :=
by
  use 20
  sorry

end gondor_laptop_earning_l1241_124199


namespace oranges_to_apples_equiv_apples_for_36_oranges_l1241_124153

-- Conditions
def weight_equiv (oranges apples : ℕ) : Prop :=
  9 * oranges = 6 * apples

-- Question (Theorem to Prove)
theorem oranges_to_apples_equiv_apples_for_36_oranges:
  ∃ (apples : ℕ), apples = 24 ∧ weight_equiv 36 apples :=
by
  use 24
  sorry

end oranges_to_apples_equiv_apples_for_36_oranges_l1241_124153


namespace pens_sold_l1241_124189

theorem pens_sold (C : ℝ) (N : ℝ) (h_gain : 22 * C = 0.25 * N * C) : N = 88 :=
by {
  sorry
}

end pens_sold_l1241_124189


namespace total_bad_carrots_and_tomatoes_l1241_124170

theorem total_bad_carrots_and_tomatoes 
  (vanessa_carrots : ℕ := 17)
  (vanessa_tomatoes : ℕ := 12)
  (mother_carrots : ℕ := 14)
  (mother_tomatoes : ℕ := 22)
  (brother_carrots : ℕ := 6)
  (brother_tomatoes : ℕ := 8)
  (good_carrots : ℕ := 28)
  (good_tomatoes : ℕ := 35) :
  (vanessa_carrots + mother_carrots + brother_carrots - good_carrots) + 
  (vanessa_tomatoes + mother_tomatoes + brother_tomatoes - good_tomatoes) = 16 := 
by
  sorry

end total_bad_carrots_and_tomatoes_l1241_124170


namespace stimulus_check_total_l1241_124106

def find_stimulus_check (T : ℝ) : Prop :=
  let amount_after_wife := T * (3/5)
  let amount_after_first_son := amount_after_wife * (3/5)
  let amount_after_second_son := amount_after_first_son * (3/5)
  amount_after_second_son = 432

theorem stimulus_check_total (T : ℝ) : find_stimulus_check T → T = 2000 := by
  sorry

end stimulus_check_total_l1241_124106


namespace simplified_expression_value_l1241_124102

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end simplified_expression_value_l1241_124102


namespace tens_digit_of_13_pow_3007_l1241_124129

theorem tens_digit_of_13_pow_3007 : 
  (13 ^ 3007 / 10) % 10 = 1 :=
sorry

end tens_digit_of_13_pow_3007_l1241_124129


namespace symmetric_point_origin_l1241_124124

-- Define the point P
structure Point3D where
  x : Int
  y : Int
  z : Int

def P : Point3D := { x := 1, y := 3, z := -5 }

-- Define the symmetric function w.r.t. the origin
def symmetric_with_origin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Define the expected result
def Q : Point3D := { x := -1, y := -3, z := 5 }

-- The theorem to prove
theorem symmetric_point_origin : symmetric_with_origin P = Q := by
  sorry

end symmetric_point_origin_l1241_124124


namespace coordinate_fifth_point_l1241_124178

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end coordinate_fifth_point_l1241_124178


namespace add_base8_l1241_124123

theorem add_base8 : 
  let a := 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  let b := 5 * 8^2 + 7 * 8^1 + 3 * 8^0
  let c := 6 * 8^1 + 2 * 8^0
  let sum := a + b + c
  sum = 1 * 8^3 + 1 * 8^2 + 2 * 8^1 + 3 * 8^0 :=
by
  -- Proof skipped
  sorry

end add_base8_l1241_124123


namespace ramu_profit_percent_l1241_124112

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_percent :
  profit_percent 42000 13000 64500 = 17.27 :=
by
  -- Placeholder for the proof
  sorry

end ramu_profit_percent_l1241_124112


namespace solve_lambda_l1241_124104

variable (a b : ℝ × ℝ)
variable (lambda : ℝ)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

axiom a_def : a = (-3, 2)
axiom b_def : b = (-1, 0)
axiom perp_def : perpendicular (a.1 + lambda * b.1, a.2 + lambda * b.2) b

theorem solve_lambda : lambda = -3 :=
by
  sorry

end solve_lambda_l1241_124104


namespace min_value_sin_function_l1241_124148

theorem min_value_sin_function (α β : ℝ) (h : -5 * (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 3 * Real.sin α) :
  ∃ x : ℝ, x = Real.sin α ∧ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 0 :=
sorry

end min_value_sin_function_l1241_124148


namespace BB_digit_value_in_5BB3_l1241_124137

theorem BB_digit_value_in_5BB3 (B : ℕ) (h : 2 * B + 8 % 9 = 0) : B = 5 :=
sorry

end BB_digit_value_in_5BB3_l1241_124137


namespace no_rational_roots_of_prime_3_digit_l1241_124197

noncomputable def is_prime (n : ℕ) := Nat.Prime n

theorem no_rational_roots_of_prime_3_digit (a b c : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) 
(h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : 0 ≤ c ∧ c ≤ 9) 
(p := 100 * a + 10 * b + c) (hp : is_prime p) (h₃ : 100 ≤ p ∧ p ≤ 999) :
¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
sorry

end no_rational_roots_of_prime_3_digit_l1241_124197


namespace result_is_21_l1241_124182

theorem result_is_21 (n : ℕ) (h : n = 55) : (n / 5 + 10) = 21 :=
by
  sorry

end result_is_21_l1241_124182


namespace x_minus_y_div_x_eq_4_7_l1241_124125

-- Definitions based on the problem's conditions
axiom y_div_x_eq_3_7 (x y : ℝ) : y / x = 3 / 7

-- The main problem to prove
theorem x_minus_y_div_x_eq_4_7 (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end x_minus_y_div_x_eq_4_7_l1241_124125


namespace greatest_possible_sum_of_10_integers_l1241_124103

theorem greatest_possible_sum_of_10_integers (a b c d e f g h i j : ℕ) 
  (h_prod : a * b * c * d * e * f * g * h * i * j = 1024) : 
  a + b + c + d + e + f + g + h + i + j ≤ 1033 :=
sorry

end greatest_possible_sum_of_10_integers_l1241_124103


namespace inequality_abc_l1241_124145

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_abc_l1241_124145


namespace kevin_found_cards_l1241_124184

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end kevin_found_cards_l1241_124184


namespace martians_cannot_hold_hands_l1241_124166

-- Define the number of hands each Martian possesses
def hands_per_martian := 3

-- Define the number of Martians
def number_of_martians := 7

-- Define the total number of hands
def total_hands := hands_per_martian * number_of_martians

-- Prove that it is not possible for the seven Martians to hold hands with each other
theorem martians_cannot_hold_hands :
  ¬ ∃ (pairs : ℕ), 2 * pairs = total_hands :=
by
  sorry

end martians_cannot_hold_hands_l1241_124166


namespace initial_people_on_train_l1241_124187

theorem initial_people_on_train {x y z u v w : ℤ} 
  (h1 : y = 29) (h2 : z = 17) (h3 : u = 27) (h4 : v = 35) (h5 : w = 116) :
  x - (y - z) + (v - u) = w → x = 120 := 
by sorry

end initial_people_on_train_l1241_124187


namespace odometer_problem_l1241_124158

theorem odometer_problem :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a + b + c ≤ 10 ∧ (11 * c - 10 * a - b) % 6 = 0 ∧ a^2 + b^2 + c^2 = 54 :=
by
  sorry

end odometer_problem_l1241_124158


namespace roadster_paving_company_cement_usage_l1241_124151

theorem roadster_paving_company_cement_usage :
  let L := 10
  let T := 5.1
  L + T = 15.1 :=
by
  -- proof is omitted
  sorry

end roadster_paving_company_cement_usage_l1241_124151


namespace mr_smith_total_cost_l1241_124132

noncomputable def total_cost : ℝ :=
  let adult_price := 30
  let child_price := 15
  let teen_price := 25
  let senior_discount := 0.10
  let college_discount := 0.05
  let senior_price := adult_price * (1 - senior_discount)
  let college_price := adult_price * (1 - college_discount)
  let soda_price := 2
  let iced_tea_price := 3
  let coffee_price := 4
  let juice_price := 1.50
  let wine_price := 6
  let buffet_cost := 2 * adult_price + 2 * senior_price + 3 * child_price + teen_price + 2 * college_price
  let drinks_cost := 3 * soda_price + 2 * iced_tea_price + coffee_price + juice_price + 2 * wine_price
  buffet_cost + drinks_cost

theorem mr_smith_total_cost : total_cost = 270.50 :=
by
  sorry

end mr_smith_total_cost_l1241_124132


namespace ann_has_30_more_cards_than_anton_l1241_124157

theorem ann_has_30_more_cards_than_anton (heike_cards : ℕ) (anton_cards : ℕ) (ann_cards : ℕ) 
  (h1 : anton_cards = 3 * heike_cards)
  (h2 : ann_cards = 6 * heike_cards)
  (h3 : ann_cards = 60) : ann_cards - anton_cards = 30 :=
by
  sorry

end ann_has_30_more_cards_than_anton_l1241_124157


namespace equivalent_single_discount_l1241_124192

variable (x : ℝ)
variable (original_price : ℝ := x)
variable (discount1 : ℝ := 0.15)
variable (discount2 : ℝ := 0.10)
variable (discount3 : ℝ := 0.05)

theorem equivalent_single_discount :
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let equivalent_discount := (1 - final_price / original_price)
  equivalent_discount = 0.27 := 
by 
  sorry

end equivalent_single_discount_l1241_124192


namespace mrs_smith_strawberries_l1241_124127

theorem mrs_smith_strawberries (girls : ℕ) (strawberries_per_girl : ℕ) 
                                (h1 : girls = 8) (h2 : strawberries_per_girl = 6) :
    girls * strawberries_per_girl = 48 := by
  sorry

end mrs_smith_strawberries_l1241_124127


namespace parabola_standard_equation_l1241_124169

theorem parabola_standard_equation (x y : ℝ) : 
  (3 * x - 4 * y - 12 = 0) →
  (y = 0 → x = 4 ∨ y = -3 → x = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
by
  intros h_line h_intersect
  sorry

end parabola_standard_equation_l1241_124169


namespace problem_solution_l1241_124100

theorem problem_solution (k : ℕ) (hk : k ≥ 2) : 
  (∀ m n : ℕ, 1 ≤ m ∧ m ≤ k → 1 ≤ n ∧ n ≤ k → m ≠ n → ¬ k ∣ (n^(n-1) - m^(m-1))) ↔ (k = 2 ∨ k = 3) :=
by
  sorry

end problem_solution_l1241_124100


namespace total_footprints_l1241_124108

def pogo_footprints_per_meter : ℕ := 4
def grimzi_footprints_per_6_meters : ℕ := 3
def distance_traveled_meters : ℕ := 6000

theorem total_footprints : (pogo_footprints_per_meter * distance_traveled_meters) + (grimzi_footprints_per_6_meters * (distance_traveled_meters / 6)) = 27000 :=
by
  sorry

end total_footprints_l1241_124108


namespace brownie_pieces_count_l1241_124163

def area_of_pan (length width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

def number_of_pieces (pan_area piece_area : ℕ) : ℕ := pan_area / piece_area

theorem brownie_pieces_count :
  let pan_length := 24
  let pan_width := 15
  let piece_side := 3
  let pan_area := area_of_pan pan_length pan_width
  let piece_area := area_of_piece piece_side
  number_of_pieces pan_area piece_area = 40 :=
by
  sorry

end brownie_pieces_count_l1241_124163


namespace weekly_allowance_l1241_124147

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end weekly_allowance_l1241_124147


namespace lloyd_total_hours_worked_l1241_124120

noncomputable def total_hours_worked (daily_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier: ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_hours := 7.5
  let regular_pay := regular_hours * regular_rate
  if total_earnings <= regular_pay then daily_hours else
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / (regular_rate * overtime_multiplier)
  regular_hours + overtime_hours

theorem lloyd_total_hours_worked :
  total_hours_worked 7.5 5.50 1.5 66 = 10.5 :=
by
  sorry

end lloyd_total_hours_worked_l1241_124120


namespace smallest_gcd_for_system_l1241_124159

theorem smallest_gcd_for_system :
  ∃ n : ℕ, n > 0 ∧ 
    (∀ a b c : ℤ,
     gcd (gcd a b) c = n →
     ∃ x y z : ℤ, 
       (x + 2*y + 3*z = a) ∧ 
       (2*x + y - 2*z = b) ∧ 
       (3*x + y + 5*z = c)) ∧ 
  n = 28 :=
sorry

end smallest_gcd_for_system_l1241_124159


namespace average_speed_is_42_l1241_124122

theorem average_speed_is_42 (v t : ℝ) (h : t > 0)
  (h_eq : v * t = (v + 21) * (2/3) * t) : v = 42 :=
by
  sorry

end average_speed_is_42_l1241_124122


namespace calculate_fraction_l1241_124191

theorem calculate_fraction : (5 / (8 / 13) / (10 / 7) = 91 / 16) :=
by
  sorry

end calculate_fraction_l1241_124191


namespace pumpkins_total_weight_l1241_124150

-- Define the weights of the pumpkins as given in the conditions
def first_pumpkin_weight : ℝ := 4
def second_pumpkin_weight : ℝ := 8.7

-- Prove that the total weight of the two pumpkins is 12.7 pounds
theorem pumpkins_total_weight : first_pumpkin_weight + second_pumpkin_weight = 12.7 := by
  sorry

end pumpkins_total_weight_l1241_124150


namespace sequence_term_geometric_l1241_124133

theorem sequence_term_geometric :
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 →
    (∀ n, n ≥ 2 → (a n) / (a (n - 1)) = 2^(n-1)) →
    a 101 = 2^5050 :=
by
  sorry

end sequence_term_geometric_l1241_124133


namespace alien_run_time_l1241_124177

variable (v_r v_f : ℝ) -- velocities in km/h
variable (T_r T_f : ℝ) -- time in hours
variable (D_r D_f : ℝ) -- distances in kilometers

theorem alien_run_time :
  v_r = 15 ∧ v_f = 10 ∧ (T_f = T_r + 0.5) ∧ (D_r = D_f) ∧ (D_r = v_r * T_r) ∧ (D_f = v_f * T_f) → T_f = 1.5 :=
by
  intros h
  rcases h with ⟨_, _, _, _, _, _⟩
  -- proof goes here
  sorry

end alien_run_time_l1241_124177


namespace probability_same_color_is_117_200_l1241_124156

/-- There are eight green balls, five red balls, and seven blue balls in a bag. 
    A ball is taken from the bag, its color recorded, then placed back in the bag.
    A second ball is taken and its color recorded. -/
def probability_two_balls_same_color : ℚ :=
  let pGreen := (8 : ℚ) / 20
  let pRed := (5 : ℚ) / 20
  let pBlue := (7 : ℚ) / 20
  pGreen^2 + pRed^2 + pBlue^2

theorem probability_same_color_is_117_200 : probability_two_balls_same_color = 117 / 200 := by
  sorry

end probability_same_color_is_117_200_l1241_124156


namespace total_spent_l1241_124107

variable (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ)

/-- Conditions from the problem setup --/
def conditions :=
  T_L = 40 ∧
  J_L = 0.5 * T_L ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = 0.25 * T_L ∧
  J_C = 3 * J_L ∧
  C_C = 0.5 * C_L ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = 0.5 * J_C

/-- Total spent by Lisa --/
def total_Lisa := T_L + J_L + C_L + S_L

/-- Total spent by Carly --/
def total_Carly := T_C + J_C + C_C + S_C + D_C + A_C

/-- Combined total spent by Lisa and Carly --/
theorem total_spent :
  conditions T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_Lisa T_L J_L C_L S_L + total_Carly T_C J_C C_C S_C D_C A_C = 520 :=
by
  sorry

end total_spent_l1241_124107


namespace perfect_square_trinomial_m_l1241_124130

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a b : ℤ, (b^2 = 25) ∧ (a + b)^2 = x^2 - (m - 3) * x + 25) → (m = 13 ∨ m = -7) :=
by
  sorry

end perfect_square_trinomial_m_l1241_124130


namespace cone_lateral_surface_area_l1241_124101

theorem cone_lateral_surface_area (l d : ℝ) (h_l : l = 5) (h_d : d = 8) : 
  (π * (d / 2) * l) = 20 * π :=
by
  sorry

end cone_lateral_surface_area_l1241_124101


namespace algebraic_expression_value_l1241_124176

theorem algebraic_expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2 * x + 1) * (2 * x - 1) + x * (x - 1) = 1 :=
by
  sorry

end algebraic_expression_value_l1241_124176


namespace car_R_speed_l1241_124121

theorem car_R_speed (v : ℝ) (h1 : ∀ t_R t_P : ℝ, t_R * v = 800 ∧ t_P * (v + 10) = 800) (h2 : ∀ t_R t_P : ℝ, t_P + 2 = t_R) :
  v = 50 := by
  sorry

end car_R_speed_l1241_124121


namespace triangle_angle_contradiction_l1241_124109

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α > 60 ∧ β > 60 ∧ γ > 60 → False :=
by
  sorry

end triangle_angle_contradiction_l1241_124109


namespace votes_diff_eq_70_l1241_124139

noncomputable def T : ℝ := 350
def votes_against (T : ℝ) : ℝ := 0.40 * T
def votes_favor (T : ℝ) (X : ℝ) : ℝ := votes_against T + X

theorem votes_diff_eq_70 :
  ∃ X : ℝ, 350 = votes_against T + votes_favor T X → X = 70 :=
by
  sorry

end votes_diff_eq_70_l1241_124139


namespace ratio_garbage_zane_dewei_l1241_124135

-- Define the weights of garbage picked up by Daliah, Dewei, and Zane.
def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah_garbage - 2
def zane_garbage : ℝ := 62

-- The theorem that we need to prove
theorem ratio_garbage_zane_dewei : zane_garbage / dewei_garbage = 4 :=
by
  sorry

end ratio_garbage_zane_dewei_l1241_124135


namespace value_of_x_squared_plus_9y_squared_l1241_124131

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l1241_124131


namespace perimeter_of_garden_l1241_124174

def area (length width : ℕ) : ℕ := length * width

def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem perimeter_of_garden :
  ∀ (l w : ℕ), area l w = 28 ∧ l = 7 → perimeter l w = 22 := by
  sorry

end perimeter_of_garden_l1241_124174


namespace point_in_fourth_quadrant_l1241_124155

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end point_in_fourth_quadrant_l1241_124155


namespace inequality_solution_l1241_124161

theorem inequality_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1 / x + 4 / y) ≥ 9 / 4 := 
sorry

end inequality_solution_l1241_124161


namespace combined_salary_ABC_and_E_l1241_124160

def salary_D : ℕ := 7000
def avg_salary : ℕ := 9000
def num_individuals : ℕ := 5

theorem combined_salary_ABC_and_E :
  (avg_salary * num_individuals - salary_D) = 38000 :=
by
  -- proof goes here
  sorry

end combined_salary_ABC_and_E_l1241_124160


namespace cook_carrots_l1241_124179

theorem cook_carrots :
  ∀ (total_carrots : ℕ) (fraction_used_before_lunch : ℚ) (carrots_not_used_end_of_day : ℕ),
    total_carrots = 300 →
    fraction_used_before_lunch = 2 / 5 →
    carrots_not_used_end_of_day = 72 →
    let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
    let carrots_after_lunch := total_carrots - carrots_used_before_lunch
    let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
    (carrots_used_end_of_day / carrots_after_lunch) = 3 / 5 :=
by
  intros total_carrots fraction_used_before_lunch carrots_not_used_end_of_day
  intros h1 h2 h3
  let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
  let carrots_after_lunch := total_carrots - carrots_used_before_lunch
  let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
  have h : carrots_used_end_of_day / carrots_after_lunch = 3 / 5 := sorry
  exact h

end cook_carrots_l1241_124179


namespace time_comparison_l1241_124146

noncomputable def pedestrian_speed : Real := 6.5
noncomputable def cyclist_speed : Real := 20.0
noncomputable def distance_between_points_B_A : Real := 4 * Real.pi - 6.5
noncomputable def alley_distance : Real := 4 * Real.pi - 6.5
noncomputable def combined_speed_3 : Real := pedestrian_speed + cyclist_speed
noncomputable def combined_speed_2 : Real := 21.5
noncomputable def time_scenario_3 : Real := (4 * Real.pi - 6.5) / combined_speed_3
noncomputable def time_scenario_2 : Real := (10.5 - 2 * Real.pi) / combined_speed_2

theorem time_comparison : time_scenario_2 < time_scenario_3 :=
by
  sorry

end time_comparison_l1241_124146


namespace euro_operation_example_l1241_124175

def euro_operation (x y : ℕ) : ℕ := 3 * x * y - x - y

theorem euro_operation_example : euro_operation 6 (euro_operation 4 2) = 300 := by
  sorry

end euro_operation_example_l1241_124175


namespace sum_lent_is_1100_l1241_124183

variables (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)

-- Given conditions
def interest_formula := I = P * r * t
def interest_difference := I = P - 572

-- Values
def rate := r = 0.06
def time := t = 8

theorem sum_lent_is_1100 : P = 1100 :=
by
  -- Definitions and axioms
  sorry

end sum_lent_is_1100_l1241_124183


namespace find_p_l1241_124118

theorem find_p (A B C p q r s : ℝ) (h₀ : A ≠ 0)
  (h₁ : r + s = -B / A)
  (h₂ : r * s = C / A)
  (h₃ : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C + 2 * A^2 * C^2) / A^3 :=
sorry

end find_p_l1241_124118


namespace fill_pool_time_l1241_124142

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end fill_pool_time_l1241_124142


namespace find_sum_l1241_124117

noncomputable def sumPutAtSimpleInterest (R: ℚ) (P: ℚ) := 
  let I := P * R * 5 / 100
  I + 90 = P * (R + 6) * 5 / 100 → P = 300

theorem find_sum (R: ℚ) (P: ℚ) : sumPutAtSimpleInterest R P := by
  sorry

end find_sum_l1241_124117


namespace line_through_point_parallel_l1241_124193

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end line_through_point_parallel_l1241_124193


namespace domain_of_f_l1241_124180

def domain_valid (x : ℝ) :=
  1 - x ≥ 0 ∧ 1 - x ≠ 1

theorem domain_of_f :
  ∀ x : ℝ, domain_valid x ↔ (x ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end domain_of_f_l1241_124180


namespace original_amount_of_money_l1241_124196

variable (took : ℕ) (now : ℕ) (initial : ℕ)

-- conditions from the problem
def conditions := (took = 2) ∧ (now = 3)

-- the statement to prove
theorem original_amount_of_money {took now initial : ℕ} (h : conditions took now) :
  initial = now + took ↔ initial = 5 :=
by {
  sorry
}

end original_amount_of_money_l1241_124196


namespace correct_average_calculation_l1241_124152

-- Conditions as definitions
def incorrect_average := 5
def num_values := 10
def incorrect_num := 26
def correct_num := 36

-- Statement to prove
theorem correct_average_calculation : 
  (incorrect_average * num_values + (correct_num - incorrect_num)) / num_values = 6 :=
by
  -- Placeholder for the proof
  sorry

end correct_average_calculation_l1241_124152


namespace solve_system_of_equations_l1241_124165

theorem solve_system_of_equations :
  ∀ (x y z : ℚ), 
    (x * y = x + 2 * y ∧
     y * z = y + 3 * z ∧
     z * x = z + 4 * x) ↔
    (x = 0 ∧ y = 0 ∧ z = 0) ∨
    (x = 25 / 9 ∧ y = 25 / 7 ∧ z = 25 / 4) := by
  sorry

end solve_system_of_equations_l1241_124165
