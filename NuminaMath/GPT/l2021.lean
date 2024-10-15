import Mathlib

namespace NUMINAMATH_GPT_equal_sum_sequence_a18_l2021_202122

def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_sequence_a18 (a : ℕ → ℕ) (h : equal_sum_sequence a 5) (h1 : a 1 = 2) : a 18 = 3 :=
  sorry

end NUMINAMATH_GPT_equal_sum_sequence_a18_l2021_202122


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l2021_202106

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 3 * s = 8 * Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l2021_202106


namespace NUMINAMATH_GPT_max_students_l2021_202185

theorem max_students (pens pencils : ℕ) (h_pens : pens = 1340) (h_pencils : pencils = 1280) : Nat.gcd pens pencils = 20 := by
    sorry

end NUMINAMATH_GPT_max_students_l2021_202185


namespace NUMINAMATH_GPT_remaining_soup_feeds_adults_l2021_202169

theorem remaining_soup_feeds_adults (C A k c : ℕ) 
    (hC : C= 10) 
    (hA : A = 5) 
    (hk : k = 8) 
    (hc : c = 20) : k - c / C * 10 * A = 30 := sorry

end NUMINAMATH_GPT_remaining_soup_feeds_adults_l2021_202169


namespace NUMINAMATH_GPT_carson_air_per_pump_l2021_202162

-- Define the conditions
def total_air_needed : ℝ := 2 * 500 + 0.6 * 500 + 0.3 * 500

def total_pumps : ℕ := 29

-- Proof problem statement
theorem carson_air_per_pump : total_air_needed / total_pumps = 50 := by
  sorry

end NUMINAMATH_GPT_carson_air_per_pump_l2021_202162


namespace NUMINAMATH_GPT_exists_real_k_l2021_202151

theorem exists_real_k (c : Fin 1998 → ℕ)
  (h1 : 0 ≤ c 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → m + n < 1998 → c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1) :
  ∃ k : ℝ, ∀ n : Fin 1998, 1 ≤ n → c n = Int.floor (n * k) :=
by
  sorry

end NUMINAMATH_GPT_exists_real_k_l2021_202151


namespace NUMINAMATH_GPT_zan_guo_gets_one_deer_l2021_202132

noncomputable def a1 : ℚ := 5 / 3
noncomputable def sum_of_sequence (a1 : ℚ) (d : ℚ) : ℚ := 5 * a1 + (5 * 4 / 2) * d
noncomputable def d : ℚ := -1 / 3
noncomputable def a3 (a1 : ℚ) (d : ℚ) : ℚ := a1 + 2 * d

theorem zan_guo_gets_one_deer :
  a3 a1 d = 1 := by
  sorry

end NUMINAMATH_GPT_zan_guo_gets_one_deer_l2021_202132


namespace NUMINAMATH_GPT_investmentAmounts_l2021_202158

variable (totalInvestment : ℝ) (bonds stocks mutualFunds : ℝ)

-- Given conditions
def conditions := 
  totalInvestment = 210000 ∧
  stocks = 2 * bonds ∧
  mutualFunds = 4 * stocks ∧
  bonds + stocks + mutualFunds = totalInvestment

-- Prove the investments
theorem investmentAmounts (h : conditions totalInvestment bonds stocks mutualFunds) :
  bonds = 19090.91 ∧ stocks = 38181.82 ∧ mutualFunds = 152727.27 :=
sorry

end NUMINAMATH_GPT_investmentAmounts_l2021_202158


namespace NUMINAMATH_GPT_person_B_completion_time_l2021_202137

variables {A B : ℝ} (H : A + B = 1/6 ∧ (A + 10 * B = 1/6))

theorem person_B_completion_time :
    (1 / (1 - 2 * (A + B)) / B = 15) :=
by
  sorry

end NUMINAMATH_GPT_person_B_completion_time_l2021_202137


namespace NUMINAMATH_GPT_revolutions_same_distance_l2021_202191

theorem revolutions_same_distance (r R : ℝ) (revs_30 : ℝ) (dist_30 dist_10 : ℝ)
  (h_radius: r = 10) (H_radius: R = 30) (h_revs_30: revs_30 = 15) 
  (H_dist_30: dist_30 = 2 * Real.pi * R * revs_30) 
  (H_dist_10: dist_10 = 2 * Real.pi * r * 45) :
  dist_30 = dist_10 :=
by {
  sorry
}

end NUMINAMATH_GPT_revolutions_same_distance_l2021_202191


namespace NUMINAMATH_GPT_fraction_of_people_under_21_correct_l2021_202196

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

end NUMINAMATH_GPT_fraction_of_people_under_21_correct_l2021_202196


namespace NUMINAMATH_GPT_sum_of_other_endpoint_l2021_202129

theorem sum_of_other_endpoint (x y : ℝ) :
  (10, -6) = ((x + 12) / 2, (y + 4) / 2) → x + y = -8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_l2021_202129


namespace NUMINAMATH_GPT_factorization_pq_difference_l2021_202150

theorem factorization_pq_difference :
  ∃ (p q : ℤ), 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q) ∧ p - q = 36 := by
-- Given the conditions in the problem,
-- We assume ∃ integers p and q such that (5x + p)(5x + q) = 25x² - 135x - 150 and derive the difference p - q = 36.
  sorry

end NUMINAMATH_GPT_factorization_pq_difference_l2021_202150


namespace NUMINAMATH_GPT_inequal_min_value_l2021_202119

theorem inequal_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1/x + 4/y) ≥ 9/4 :=
sorry

end NUMINAMATH_GPT_inequal_min_value_l2021_202119


namespace NUMINAMATH_GPT_angle_RBC_10_degrees_l2021_202181

noncomputable def compute_angle_RBC (angle_BRA angle_BAC angle_ABC : ℝ) : ℝ :=
  let angle_RBA := 180 - angle_BRA - angle_BAC
  angle_RBA - angle_ABC

theorem angle_RBC_10_degrees :
  ∀ (angle_BRA angle_BAC angle_ABC : ℝ), 
    angle_BRA = 72 → angle_BAC = 43 → angle_ABC = 55 → 
    compute_angle_RBC angle_BRA angle_BAC angle_ABC = 10 :=
by
  intros
  unfold compute_angle_RBC
  sorry

end NUMINAMATH_GPT_angle_RBC_10_degrees_l2021_202181


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2021_202180

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2021_202180


namespace NUMINAMATH_GPT_problem_l2021_202184

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end NUMINAMATH_GPT_problem_l2021_202184


namespace NUMINAMATH_GPT_expand_polynomials_l2021_202126

variable (t : ℝ)

def poly1 := 3 * t^2 - 4 * t + 3
def poly2 := -2 * t^2 + 3 * t - 4
def expanded_poly := -6 * t^4 + 17 * t^3 - 30 * t^2 + 25 * t - 12

theorem expand_polynomials : (poly1 * poly2) = expanded_poly := 
by
  sorry

end NUMINAMATH_GPT_expand_polynomials_l2021_202126


namespace NUMINAMATH_GPT_complement_union_complement_intersection_l2021_202154

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_union (A B : Set ℝ) :
  (A ∪ B)ᶜ = { x : ℝ | x ≤ 2 ∨ x ≥ 10 } :=
by
  sorry

theorem complement_intersection (A B : Set ℝ) :
  (Aᶜ ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
by
  sorry

end NUMINAMATH_GPT_complement_union_complement_intersection_l2021_202154


namespace NUMINAMATH_GPT_find_a_and_b_l2021_202153

theorem find_a_and_b :
  ∃ a b : ℝ, 
    (∀ x : ℝ, (x^3 + 3*x^2 + 2*x > 0) ↔ (x > 0 ∨ -2 < x ∧ x < -1)) ∧
    (∀ x : ℝ, (x^2 + a*x + b ≤ 0) ↔ (-2 < x ∧ x ≤ 0 ∨ 0 < x ∧ x ≤ 2)) ∧ 
    a = -1 ∧ b = -2 := 
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2021_202153


namespace NUMINAMATH_GPT_geometric_progression_positions_l2021_202164

theorem geometric_progression_positions (u1 q : ℝ) (m n p : ℕ)
  (h27 : 27 = u1 * q ^ (m - 1))
  (h8 : 8 = u1 * q ^ (n - 1))
  (h12 : 12 = u1 * q ^ (p - 1)) :
  m = 3 * p - 2 * n :=
sorry

end NUMINAMATH_GPT_geometric_progression_positions_l2021_202164


namespace NUMINAMATH_GPT_right_triangle_with_integer_sides_l2021_202127

theorem right_triangle_with_integer_sides (k : ℤ) :
  ∃ (a b c : ℤ), a = 2*k+1 ∧ b = 2*k*(k+1) ∧ c = 2*k^2+2*k+1 ∧ (a^2 + b^2 = c^2) ∧ (c = a + 1) := by
  sorry

end NUMINAMATH_GPT_right_triangle_with_integer_sides_l2021_202127


namespace NUMINAMATH_GPT_min_max_values_l2021_202146

noncomputable def expression (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ( (x₁ ^ 2 / x₂) + (x₂ ^ 2 / x₃) + (x₃ ^ 2 / x₄) + (x₄ ^ 2 / x₁) ) /
  ( x₁ + x₂ + x₃ + x₄ )

theorem min_max_values
  (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (h₀ : 0 < a) (h₁ : a < b)
  (h₂ : a ≤ x₁) (h₃ : x₁ ≤ b)
  (h₄ : a ≤ x₂) (h₅ : x₂ ≤ b)
  (h₆ : a ≤ x₃) (h₇ : x₃ ≤ b)
  (h₈ : a ≤ x₄) (h₉ : x₄ ≤ b) :
  expression x₁ x₂ x₃ x₄ ≥ 1 / b ∧ expression x₁ x₂ x₃ x₄ ≤ 1 / a :=
  sorry

end NUMINAMATH_GPT_min_max_values_l2021_202146


namespace NUMINAMATH_GPT_cakes_remaining_l2021_202161

theorem cakes_remaining (initial_cakes sold_cakes remaining_cakes: ℕ) (h₀ : initial_cakes = 167) (h₁ : sold_cakes = 108) (h₂ : remaining_cakes = initial_cakes - sold_cakes) : remaining_cakes = 59 :=
by
  rw [h₀, h₁] at h₂
  exact h₂

end NUMINAMATH_GPT_cakes_remaining_l2021_202161


namespace NUMINAMATH_GPT_journey_time_ratio_l2021_202123

theorem journey_time_ratio (D : ℝ) (hD_pos : D > 0) :
  let T1 := D / 45
  let T2 := D / 30
  (T2 / T1) = (3 / 2) := 
by
  sorry

end NUMINAMATH_GPT_journey_time_ratio_l2021_202123


namespace NUMINAMATH_GPT_jackson_spends_on_school_supplies_l2021_202116

theorem jackson_spends_on_school_supplies :
  let num_students := 50
  let pens_per_student := 7
  let notebooks_per_student := 5
  let binders_per_student := 3
  let highlighters_per_student := 4
  let folders_per_student := 2
  let cost_pen := 0.70
  let cost_notebook := 1.60
  let cost_binder := 5.10
  let cost_highlighter := 0.90
  let cost_folder := 1.15
  let teacher_discount := 135
  let bulk_discount := 25
  let sales_tax_rate := 0.05
  let total_cost := 
    (num_students * pens_per_student * cost_pen) + 
    (num_students * notebooks_per_student * cost_notebook) + 
    (num_students * binders_per_student * cost_binder) + 
    (num_students * highlighters_per_student * cost_highlighter) + 
    (num_students * folders_per_student * cost_folder)
  let discounted_cost := total_cost - teacher_discount - bulk_discount
  let sales_tax := discounted_cost * sales_tax_rate
  let final_cost := discounted_cost + sales_tax
  final_cost = 1622.25 := by
  sorry

end NUMINAMATH_GPT_jackson_spends_on_school_supplies_l2021_202116


namespace NUMINAMATH_GPT_marina_more_fudge_l2021_202100

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end NUMINAMATH_GPT_marina_more_fudge_l2021_202100


namespace NUMINAMATH_GPT_B_completes_in_40_days_l2021_202118

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end NUMINAMATH_GPT_B_completes_in_40_days_l2021_202118


namespace NUMINAMATH_GPT_scientific_notation_of_22nm_l2021_202147

theorem scientific_notation_of_22nm (h : 22 * 10^(-9) = 0.000000022) : 0.000000022 = 2.2 * 10^(-8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_22nm_l2021_202147


namespace NUMINAMATH_GPT_apples_eaten_l2021_202198

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

end NUMINAMATH_GPT_apples_eaten_l2021_202198


namespace NUMINAMATH_GPT_initial_cargo_l2021_202195

theorem initial_cargo (initial_cargo additional_cargo total_cargo : ℕ) 
  (h1 : additional_cargo = 8723) 
  (h2 : total_cargo = 14696) 
  (h3 : initial_cargo + additional_cargo = total_cargo) : 
  initial_cargo = 5973 := 
by 
  -- Start with the assumptions and directly obtain the calculation as required
  sorry

end NUMINAMATH_GPT_initial_cargo_l2021_202195


namespace NUMINAMATH_GPT_cheaper_lens_price_l2021_202117

theorem cheaper_lens_price (original_price : ℝ) (discount_rate : ℝ) (savings : ℝ) 
  (h₁ : original_price = 300) 
  (h₂ : discount_rate = 0.20) 
  (h₃ : savings = 20) 
  (discounted_price : ℝ) 
  (cheaper_lens_price : ℝ)
  (discount_eq : discounted_price = original_price * (1 - discount_rate))
  (savings_eq : cheaper_lens_price = discounted_price - savings) :
  cheaper_lens_price = 220 := 
by sorry

end NUMINAMATH_GPT_cheaper_lens_price_l2021_202117


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_is_986_l2021_202189

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_is_986_l2021_202189


namespace NUMINAMATH_GPT_least_cans_required_l2021_202113

def maaza : ℕ := 20
def pepsi : ℕ := 144
def sprite : ℕ := 368

def GCD (a b : ℕ) : ℕ := Nat.gcd a b

def total_cans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_maaza_pepsi := GCD maaza pepsi
  let gcd_all := GCD gcd_maaza_pepsi sprite
  (maaza / gcd_all) + (pepsi / gcd_all) + (sprite / gcd_all)

theorem least_cans_required : total_cans maaza pepsi sprite = 133 := by
  sorry

end NUMINAMATH_GPT_least_cans_required_l2021_202113


namespace NUMINAMATH_GPT_f_sum_zero_l2021_202108

-- Define the function f with the given properties
noncomputable def f : ℝ → ℝ := sorry

-- Define hypotheses based on the problem's conditions
axiom f_cube (x : ℝ) : f (x ^ 3) = (f x) ^ 3
axiom f_inj (x1 x2 : ℝ) (h : x1 ≠ x2) : f x1 ≠ f x2

-- State the proof problem
theorem f_sum_zero : f 0 + f 1 + f (-1) = 0 :=
sorry

end NUMINAMATH_GPT_f_sum_zero_l2021_202108


namespace NUMINAMATH_GPT_range_of_a_l2021_202104

variable (a : ℝ)

def proposition_p := ∀ x : ℝ, a * x^2 - 2 * x + 1 > 0
def proposition_q := ∀ x : ℝ, x ∈ Set.Icc (1/2 : ℝ) (2 : ℝ) → x + (1 / x) > a

theorem range_of_a (hp : proposition_p a) (hq : proposition_q a) : 1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2021_202104


namespace NUMINAMATH_GPT_range_of_a_minus_b_l2021_202144

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 2) (h₃ : -2 < b) (h₄ : b < 1) :
  -2 < a - b ∧ a - b < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l2021_202144


namespace NUMINAMATH_GPT_total_candies_l2021_202155

def candies_in_boxes (num_boxes: Nat) (pieces_per_box: Nat) : Nat :=
  num_boxes * pieces_per_box

theorem total_candies :
  candies_in_boxes 3 6 + candies_in_boxes 5 8 + candies_in_boxes 4 10 = 98 := by
  sorry

end NUMINAMATH_GPT_total_candies_l2021_202155


namespace NUMINAMATH_GPT_max_pens_min_pens_l2021_202170

def pen_prices : List ℕ := [2, 3, 4]
def total_money : ℕ := 31

/-- Given the conditions of the problem, prove the maximum number of pens -/
theorem max_pens  (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 14 := by
  sorry

/-- Given the conditions of the problem, prove the minimum number of pens -/
theorem min_pens (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 9 := by
  sorry

end NUMINAMATH_GPT_max_pens_min_pens_l2021_202170


namespace NUMINAMATH_GPT_radius_triple_area_l2021_202110

variable (r n : ℝ)

theorem radius_triple_area (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n / 2) * (Real.sqrt 3 - 1) :=
sorry

end NUMINAMATH_GPT_radius_triple_area_l2021_202110


namespace NUMINAMATH_GPT_a_4_eq_28_l2021_202173

def Sn (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by
  sorry

end NUMINAMATH_GPT_a_4_eq_28_l2021_202173


namespace NUMINAMATH_GPT_probability_A_level_l2021_202149

theorem probability_A_level (p_B : ℝ) (p_C : ℝ) (h_B : p_B = 0.03) (h_C : p_C = 0.01) : 
  (1 - (p_B + p_C)) = 0.96 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_probability_A_level_l2021_202149


namespace NUMINAMATH_GPT_rational_x_of_rational_x3_and_x2_add_x_l2021_202193

variable {x : ℝ}

theorem rational_x_of_rational_x3_and_x2_add_x (hx3 : ∃ a : ℚ, x^3 = a)
  (hx2_add_x : ∃ b : ℚ, x^2 + x = b) : ∃ r : ℚ, x = r :=
sorry

end NUMINAMATH_GPT_rational_x_of_rational_x3_and_x2_add_x_l2021_202193


namespace NUMINAMATH_GPT_calculate_expression_l2021_202131

theorem calculate_expression : (Real.sqrt 8 + Real.sqrt (1 / 2)) * Real.sqrt 32 = 20 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2021_202131


namespace NUMINAMATH_GPT_simplify_expression_l2021_202174

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2021_202174


namespace NUMINAMATH_GPT_sum_of_three_smallest_positive_solutions_equals_ten_and_half_l2021_202103

noncomputable def sum_three_smallest_solutions : ℚ :=
    let x1 : ℚ := 2.75
    let x2 : ℚ := 3 + (4 / 9)
    let x3 : ℚ := 4 + (5 / 16)
    x1 + x2 + x3

theorem sum_of_three_smallest_positive_solutions_equals_ten_and_half :
  sum_three_smallest_solutions = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_smallest_positive_solutions_equals_ten_and_half_l2021_202103


namespace NUMINAMATH_GPT_fixed_point_always_on_line_l2021_202186

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_always_on_line_l2021_202186


namespace NUMINAMATH_GPT_intersection_S_T_l2021_202167

def S : Set ℝ := { x | (x - 2) * (x - 3) >= 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T :
  S ∩ T = { x | (0 < x ∧ x <= 2) ∨ (x >= 3) } := by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l2021_202167


namespace NUMINAMATH_GPT_total_hours_for_songs_l2021_202187

def total_hours_worked_per_day := 10
def total_days_per_song := 10
def number_of_songs := 3

theorem total_hours_for_songs :
  total_hours_worked_per_day * total_days_per_song * number_of_songs = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_hours_for_songs_l2021_202187


namespace NUMINAMATH_GPT_circle_radius_and_diameter_relations_l2021_202101

theorem circle_radius_and_diameter_relations
  (r_x r_y r_z A_x A_y A_z d_x d_z : ℝ)
  (hx_circumference : 2 * π * r_x = 18 * π)
  (hx_area : A_x = π * r_x^2)
  (hy_area_eq : A_y = A_x)
  (hz_area_eq : A_z = 4 * A_x)
  (hy_area : A_y = π * r_y^2)
  (hz_area : A_z = π * r_z^2)
  (dx_def : d_x = 2 * r_x)
  (dz_def : d_z = 2 * r_z)
  : r_y = r_z / 2 ∧ d_z = 2 * d_x := 
by 
  sorry

end NUMINAMATH_GPT_circle_radius_and_diameter_relations_l2021_202101


namespace NUMINAMATH_GPT_bruce_money_left_to_buy_more_clothes_l2021_202163

def calculate_remaining_money 
  (amount_given : ℝ) 
  (shirt_price : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ)
  (sock_price : ℝ) (num_socks : ℕ)
  (belt_original_price : ℝ) (belt_discount : ℝ)
  (total_discount : ℝ) : ℝ := 
let shirts_cost := shirt_price * num_shirts
let socks_cost := sock_price * num_socks
let belt_price := belt_original_price * (1 - belt_discount)
let total_cost := shirts_cost + pants_price + socks_cost + belt_price
let discount_cost := total_cost * total_discount
let final_cost := total_cost - discount_cost
amount_given - final_cost

theorem bruce_money_left_to_buy_more_clothes 
  : calculate_remaining_money 71 5 5 26 3 2 12 0.25 0.10 = 11.60 := 
by
  sorry

end NUMINAMATH_GPT_bruce_money_left_to_buy_more_clothes_l2021_202163


namespace NUMINAMATH_GPT_range_of_a_l2021_202159

-- Defining the function f(x) = x^2 + 2ax - 1
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - 1

-- Conditions: x1, x2 ∈ [1, +∞) and x1 < x2
variables (x1 x2 a : ℝ)
variables (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2)

-- Statement of the proof problem:
theorem range_of_a (hf_ineq : x2 * f x1 a - x1 * f x2 a < a * (x1 - x2)) : a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2021_202159


namespace NUMINAMATH_GPT_range_of_heights_l2021_202190

theorem range_of_heights (max_height min_height : ℝ) (h_max : max_height = 175) (h_min : min_height = 100) :
  (max_height - min_height) = 75 :=
by
  -- Defer proof
  sorry

end NUMINAMATH_GPT_range_of_heights_l2021_202190


namespace NUMINAMATH_GPT_number_of_yellow_balls_l2021_202156

-- Definitions based on conditions
def number_of_red_balls : ℕ := 10
def probability_red_ball := (1 : ℚ) / 3

-- Theorem stating the number of yellow balls
theorem number_of_yellow_balls :
  ∃ (y : ℕ), (number_of_red_balls : ℚ) / (number_of_red_balls + y) = probability_red_ball ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_yellow_balls_l2021_202156


namespace NUMINAMATH_GPT_trigonometric_product_identity_l2021_202139

theorem trigonometric_product_identity : 
  let cos_40 : Real := Real.cos (Real.pi * 40 / 180)
  let sin_40 : Real := Real.sin (Real.pi * 40 / 180)
  let cos_50 : Real := Real.cos (Real.pi * 50 / 180)
  let sin_50 : Real := Real.sin (Real.pi * 50 / 180)
  (sin_50 = cos_40) → (cos_50 = sin_40) →
  (1 - cos_40⁻¹) * (1 + sin_50⁻¹) * (1 - sin_40⁻¹) * (1 + cos_50⁻¹) = 1 := by
  sorry

end NUMINAMATH_GPT_trigonometric_product_identity_l2021_202139


namespace NUMINAMATH_GPT_sum_of_digits_divisible_by_9_l2021_202138

theorem sum_of_digits_divisible_by_9 (N : ℕ) (a b c : ℕ) (hN : N < 10^1962)
  (h1 : N % 9 = 0)
  (ha : a = (N.digits 10).sum)
  (hb : b = (a.digits 10).sum)
  (hc : c = (b.digits 10).sum) :
  c = 9 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_divisible_by_9_l2021_202138


namespace NUMINAMATH_GPT_solve_for_y_l2021_202152

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem solve_for_y (y : ℝ) : star 2 y = 10 → y = 0 := by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l2021_202152


namespace NUMINAMATH_GPT_ball_hits_ground_approx_time_l2021_202183

noncomputable def ball_hits_ground_time (t : ℝ) : ℝ :=
-6 * t^2 - 12 * t + 60

theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, |t - 2.32| < 0.01 ∧ ball_hits_ground_time t = 0 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_approx_time_l2021_202183


namespace NUMINAMATH_GPT_kaylin_age_32_l2021_202166

-- Defining the ages of the individuals as variables
variables (Kaylin Sarah Eli Freyja Alfred Olivia : ℝ)

-- Defining the given conditions
def conditions : Prop := 
  (Kaylin = Sarah - 5) ∧
  (Sarah = 2 * Eli) ∧
  (Eli = Freyja + 9) ∧
  (Freyja = 2.5 * Alfred) ∧
  (Alfred = (3/4) * Olivia) ∧
  (Freyja = 9.5)

-- Main statement to prove
theorem kaylin_age_32 (h : conditions Kaylin Sarah Eli Freyja Alfred Olivia) : Kaylin = 32 :=
by
  sorry

end NUMINAMATH_GPT_kaylin_age_32_l2021_202166


namespace NUMINAMATH_GPT_solve_quadratic_l2021_202124

theorem solve_quadratic (x : ℚ) (h_pos : x > 0) (h_eq : 3 * x^2 + 8 * x - 35 = 0) : 
    x = 7/3 :=
by
    sorry

end NUMINAMATH_GPT_solve_quadratic_l2021_202124


namespace NUMINAMATH_GPT_smallest_prime_linear_pair_l2021_202145

def is_prime (n : ℕ) : Prop := ¬(∃ k > 1, k < n ∧ k ∣ n)

theorem smallest_prime_linear_pair :
  ∃ a b : ℕ, is_prime a ∧ is_prime b ∧ a + b = 180 ∧ a > b ∧ b = 7 := 
by
  sorry

end NUMINAMATH_GPT_smallest_prime_linear_pair_l2021_202145


namespace NUMINAMATH_GPT_living_space_increase_l2021_202188

theorem living_space_increase (a b x : ℝ) (h₁ : a = 10) (h₂ : b = 12.1) : a * (1 + x) ^ 2 = b :=
sorry

end NUMINAMATH_GPT_living_space_increase_l2021_202188


namespace NUMINAMATH_GPT_ratio_boys_to_girls_l2021_202177

-- Define the given conditions
def G : ℕ := 300
def T : ℕ := 780

-- State the proposition to be proven
theorem ratio_boys_to_girls (B : ℕ) (h : B + G = T) : B / G = 8 / 5 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_ratio_boys_to_girls_l2021_202177


namespace NUMINAMATH_GPT_cos_half_angle_l2021_202171

open Real

theorem cos_half_angle (α : ℝ) (h_sin : sin α = (4 / 9) * sqrt 2) (h_obtuse : π / 2 < α ∧ α < π) :
  cos (α / 2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_half_angle_l2021_202171


namespace NUMINAMATH_GPT_total_points_scored_l2021_202160

theorem total_points_scored (layla_score nahima_score : ℕ)
  (h1 : layla_score = 70)
  (h2 : layla_score = nahima_score + 28) :
  layla_score + nahima_score = 112 :=
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l2021_202160


namespace NUMINAMATH_GPT_apples_distribution_l2021_202143

variable (p b t : ℕ)

theorem apples_distribution (p_eq : p = 40) (b_eq : b = p + 8) (t_eq : t = (3 * b) / 8) :
  t = 18 := by
  sorry

end NUMINAMATH_GPT_apples_distribution_l2021_202143


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_l2021_202197

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a1_a3 : a 1 + a 3 = 2) : a 2 = 1 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_l2021_202197


namespace NUMINAMATH_GPT_primes_in_arithmetic_sequence_l2021_202142

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_in_arithmetic_sequence (p : ℕ) :
  is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_primes_in_arithmetic_sequence_l2021_202142


namespace NUMINAMATH_GPT_books_sold_correct_l2021_202165

-- Definitions of the conditions
def initial_books : ℕ := 33
def remaining_books : ℕ := 7
def books_sold : ℕ := initial_books - remaining_books

-- The statement to be proven (with proof omitted)
theorem books_sold_correct : books_sold = 26 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_books_sold_correct_l2021_202165


namespace NUMINAMATH_GPT_inequality_proof_l2021_202140

theorem inequality_proof (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2021_202140


namespace NUMINAMATH_GPT_center_of_circle_l2021_202176

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center of the circle in polar coordinates
def center_polar (ρ θ : ℝ) : Prop := (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- The theorem states that the center of the given circle in polar coordinates is (1, π/2) or (1, 3π/2)
theorem center_of_circle : ∃ (ρ θ : ℝ), circle_polar ρ θ → center_polar ρ θ :=
by
  -- The center of the circle given the condition in polar coordinate system is (1, π/2) or (1, 3π/2)
  sorry

end NUMINAMATH_GPT_center_of_circle_l2021_202176


namespace NUMINAMATH_GPT_triangle_inequality_property_l2021_202179

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  (a * b * c) / (4 * Real.sqrt (A * B * C))

noncomputable def inradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  Real.sqrt (A * B * C) * perimeter a b c

theorem triangle_inequality_property (a b c A B C : ℝ)
  (h₁ : ∀ {x}, x > 0)
  (h₂ : A ≠ B)
  (h₃ : B ≠ C)
  (h₄ : C ≠ A) :
  ¬ (perimeter a b c ≤ circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c > circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c / 6 < circumradius a b c A B C + inradius a b c A B C ∨ 
  circumradius a b c A B C + inradius a b c A B C < 6 * perimeter a b c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_property_l2021_202179


namespace NUMINAMATH_GPT_remainder_zero_l2021_202135

theorem remainder_zero (x : ℂ) 
  (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) : 
  x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_zero_l2021_202135


namespace NUMINAMATH_GPT_find_f_minus_two_l2021_202105

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_minus_two : f (-2) = 2 :=
by sorry

end NUMINAMATH_GPT_find_f_minus_two_l2021_202105


namespace NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l2021_202182

variable (x y z : ℝ)

def area_xy : ℝ := x * y
def area_yz : ℝ := y * z
def area_zx : ℝ := z * x

theorem product_of_areas_eq_square_of_volume :
  (area_xy x y) * (area_yz y z) * (area_zx z x) = (x * y * z) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l2021_202182


namespace NUMINAMATH_GPT_complete_the_square_l2021_202148

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end NUMINAMATH_GPT_complete_the_square_l2021_202148


namespace NUMINAMATH_GPT_savings_are_equal_and_correct_l2021_202120

-- Definitions of the given conditions
variables (I1 I2 E1 E2 : ℝ)
variables (S1 S2 : ℝ)
variables (rI : ℝ := 5/4) -- ratio of incomes
variables (rE : ℝ := 3/2) -- ratio of expenditures
variables (I1_val : ℝ := 3000) -- P1's income

-- Given conditions
def given_conditions : Prop :=
  I1 = I1_val ∧
  I1 / I2 = rI ∧
  E1 / E2 = rE ∧
  S1 = S2

-- Required proof
theorem savings_are_equal_and_correct (I2_val : I2 = (I1_val * 4/5)) (x : ℝ) (h1 : E1 = 3 * x) (h2 : E2 = 2 * x) (h3 : S1 = 1200) :
  S1 = S2 ∧ S1 = 1200 := by
  sorry

end NUMINAMATH_GPT_savings_are_equal_and_correct_l2021_202120


namespace NUMINAMATH_GPT_arithmetic_seq_terms_greater_than_50_l2021_202125

theorem arithmetic_seq_terms_greater_than_50 :
  let a_n (n : ℕ) := 17 + (n-1) * 4
  let num_terms := (19 - 10) + 1
  ∀ (a_n : ℕ → ℕ), ((a_n 1 = 17) ∧ (∃ k, a_n k = 89) ∧ (∀ n, a_n (n + 1) = a_n n + 4)) →
  ∃ m, m = num_terms ∧ ∀ n, (10 ≤ n ∧ n ≤ 19) → a_n n > 50 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_terms_greater_than_50_l2021_202125


namespace NUMINAMATH_GPT_largest_fraction_l2021_202199

theorem largest_fraction (d x : ℕ) 
  (h1: (2 * x / d) + (3 * x / d) + (4 * x / d) = 10 / 11)
  (h2: d = 11 * x) : (4 / 11 : ℚ) = (4 * x / d : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_l2021_202199


namespace NUMINAMATH_GPT_choir_members_total_l2021_202114

theorem choir_members_total
  (first_group second_group third_group : ℕ)
  (h1 : first_group = 25)
  (h2 : second_group = 30)
  (h3 : third_group = 15) :
  first_group + second_group + third_group = 70 :=
by
  sorry

end NUMINAMATH_GPT_choir_members_total_l2021_202114


namespace NUMINAMATH_GPT_probability_first_or_second_l2021_202178

/-- Define the events and their probabilities --/
def prob_hit_first_sector : ℝ := 0.4
def prob_hit_second_sector : ℝ := 0.3
def prob_hit_first_or_second : ℝ := 0.7

/-- The proof that these probabilities add up as mutually exclusive events --/
theorem probability_first_or_second (P_A : ℝ) (P_B : ℝ) (P_A_or_B : ℝ) (hP_A : P_A = prob_hit_first_sector) (hP_B : P_B = prob_hit_second_sector) (hP_A_or_B : P_A_or_B = prob_hit_first_or_second) :
  P_A_or_B = P_A + P_B := 
  by
    rw [hP_A, hP_B, hP_A_or_B]
    sorry

end NUMINAMATH_GPT_probability_first_or_second_l2021_202178


namespace NUMINAMATH_GPT_roots_of_quadratic_l2021_202111

theorem roots_of_quadratic (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a + b + c = 0) (h₂ : a - b + c = 0) :
  (a * (1 : ℝ)^2 + b * (1 : ℝ) + c = 0) ∧ (a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l2021_202111


namespace NUMINAMATH_GPT_find_x_l2021_202112

def operation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) :
  operation 6 (operation 4 x) = 480 ↔ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2021_202112


namespace NUMINAMATH_GPT_polygon_sides_eq_six_l2021_202157

theorem polygon_sides_eq_six (n : ℕ) (h : 3 * n - (n * (n - 3)) / 2 = 6) : n = 6 := 
sorry

end NUMINAMATH_GPT_polygon_sides_eq_six_l2021_202157


namespace NUMINAMATH_GPT_common_divisors_9240_8820_l2021_202107

def prime_factors_9240 := [(2, 3), (3, 1), (5, 1), (7, 1), (11, 1)]
def prime_factors_8820 := [(2, 2), (3, 2), (5, 1), (7, 1), (11, 1)]

def gcd_prime_factors := [(2, 2), (3, 1), (5, 1), (7, 1), (11, 1)]

def num_divisors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc p => acc * (p.snd + 1)) 1

theorem common_divisors_9240_8820 :
  num_divisors gcd_prime_factors = 32 := by
  sorry

end NUMINAMATH_GPT_common_divisors_9240_8820_l2021_202107


namespace NUMINAMATH_GPT_missed_number_l2021_202102

/-
  A student finds the sum \(1 + 2 + 3 + \cdots\) as his patience runs out. 
  He found the sum as 575. When the teacher declared the result wrong, 
  the student realized that he missed a number.
  Prove that the number he missed is 20.
-/

theorem missed_number (n : ℕ) (S_incorrect S_correct S_missed : ℕ) 
  (h1 : S_incorrect = 575)
  (h2 : S_correct = n * (n + 1) / 2)
  (h3 : S_correct = 595)
  (h4 : S_missed = S_correct - S_incorrect) :
  S_missed = 20 :=
sorry

end NUMINAMATH_GPT_missed_number_l2021_202102


namespace NUMINAMATH_GPT_initial_mean_l2021_202136

theorem initial_mean (M : ℝ) (h1 : 50 * (36.5 : ℝ) - 23 = 50 * (36.04 : ℝ) + 23)
: M = 36.04 :=
by
  sorry

end NUMINAMATH_GPT_initial_mean_l2021_202136


namespace NUMINAMATH_GPT_find_third_number_l2021_202134

theorem find_third_number (x : ℕ) (h : (6 + 16 + x) / 3 = 13) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l2021_202134


namespace NUMINAMATH_GPT_simplify_product_l2021_202192

theorem simplify_product (x t : ℕ) : (x^2 * t^3) * (x^3 * t^4) = (x^5) * (t^7) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_product_l2021_202192


namespace NUMINAMATH_GPT_isosceles_triangle_of_condition_l2021_202172

theorem isosceles_triangle_of_condition (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 2 * b * Real.cos C)
  (h2 : A + B + C = Real.pi) :
  (B = C) ∨ (A = C) ∨ (A = B) := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_of_condition_l2021_202172


namespace NUMINAMATH_GPT_passed_candidates_count_l2021_202115

theorem passed_candidates_count
    (average_total : ℝ)
    (number_candidates : ℕ)
    (average_passed : ℝ)
    (average_failed : ℝ)
    (total_marks : ℝ) :
    average_total = 35 →
    number_candidates = 120 →
    average_passed = 39 →
    average_failed = 15 →
    total_marks = average_total * number_candidates →
    (∃ P F, P + F = number_candidates ∧ 39 * P + 15 * F = total_marks ∧ P = 100) :=
by
  sorry

end NUMINAMATH_GPT_passed_candidates_count_l2021_202115


namespace NUMINAMATH_GPT_sin_product_l2021_202175

theorem sin_product (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.sin (π / 2 - α) = 2 / 5 :=
by
  -- proof shorter placeholder
  sorry

end NUMINAMATH_GPT_sin_product_l2021_202175


namespace NUMINAMATH_GPT_boat_avg_speed_ratio_l2021_202194

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

end NUMINAMATH_GPT_boat_avg_speed_ratio_l2021_202194


namespace NUMINAMATH_GPT_angle_quadrant_l2021_202121

theorem angle_quadrant (θ : Real) (P : Real × Real) (h : P = (Real.sin θ * Real.cos θ, 2 * Real.cos θ) ∧ P.1 < 0 ∧ P.2 < 0) :
  π / 2 < θ ∧ θ < π :=
by
  sorry

end NUMINAMATH_GPT_angle_quadrant_l2021_202121


namespace NUMINAMATH_GPT_intersection_of_sets_l2021_202141

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 0 < x }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l2021_202141


namespace NUMINAMATH_GPT_find_a_l2021_202133

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then 2^x - a * x else -2^(-x) - a * x

-- Define the fact that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = -f (-x)

-- State the main theorem that needs to be proven
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) ∧ (f a 2 = 2) → a = -9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2021_202133


namespace NUMINAMATH_GPT_tomatoes_sold_to_mr_wilson_l2021_202130

theorem tomatoes_sold_to_mr_wilson :
  let T := 245.5
  let S_m := 125.5
  let N := 42
  let S_w := T - S_m - N
  S_w = 78 := 
by
  sorry

end NUMINAMATH_GPT_tomatoes_sold_to_mr_wilson_l2021_202130


namespace NUMINAMATH_GPT_find_dividend_l2021_202109

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 8) (h2 : quotient = 8) (h3 : dividend = k * quotient) : dividend = 64 := 
by 
  sorry

end NUMINAMATH_GPT_find_dividend_l2021_202109


namespace NUMINAMATH_GPT_total_cards_correct_l2021_202168

-- Define the number of dozens each person has
def dozens_per_person : Nat := 9

-- Define the number of cards per dozen
def cards_per_dozen : Nat := 12

-- Define the number of people
def num_people : Nat := 4

-- Define the total number of Pokemon cards in all
def total_cards : Nat := dozens_per_person * cards_per_dozen * num_people

-- The statement to be proved
theorem total_cards_correct : total_cards = 432 := 
by 
  -- Proof omitted as requested
  sorry

end NUMINAMATH_GPT_total_cards_correct_l2021_202168


namespace NUMINAMATH_GPT_expression_eq_one_l2021_202128

theorem expression_eq_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 1) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
   a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
   b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 := 
by
  sorry

end NUMINAMATH_GPT_expression_eq_one_l2021_202128
