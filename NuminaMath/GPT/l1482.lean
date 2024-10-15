import Mathlib

namespace NUMINAMATH_GPT_rational_root_of_polynomial_l1482_148236

-- Polynomial definition
def P (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

-- Theorem statement
theorem rational_root_of_polynomial : ∀ x : ℚ, P x = 0 ↔ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_rational_root_of_polynomial_l1482_148236


namespace NUMINAMATH_GPT_train_length_l1482_148207

theorem train_length (x : ℕ)
  (h1 : ∀ (x : ℕ), (790 + x) / 33 = (860 - x) / 22) : x = 200 := by
  sorry

end NUMINAMATH_GPT_train_length_l1482_148207


namespace NUMINAMATH_GPT_ratio_rate_down_to_up_l1482_148259

theorem ratio_rate_down_to_up 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down_eq_time_up : time_down = time_up) :
  (time_up = 2) → 
  (rate_up = 3) →
  (distance_down = 9) → 
  (time_down = time_up) →
  (distance_down / time_down / rate_up = 1.5) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ratio_rate_down_to_up_l1482_148259


namespace NUMINAMATH_GPT_fraction_product_cube_l1482_148283

theorem fraction_product_cube :
  ((5 : ℚ) / 8)^3 * ((4 : ℚ) / 9)^3 = (125 : ℚ) / 5832 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_cube_l1482_148283


namespace NUMINAMATH_GPT_legacy_total_earnings_l1482_148205

def floors := 4
def rooms_per_floor := 10
def hours_per_room := 6
def hourly_rate := 15
def total_rooms := floors * rooms_per_floor
def total_hours := total_rooms * hours_per_room
def total_earnings := total_hours * hourly_rate

theorem legacy_total_earnings :
  total_earnings = 3600 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_legacy_total_earnings_l1482_148205


namespace NUMINAMATH_GPT_binary_to_decimal_l1482_148267

theorem binary_to_decimal : (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l1482_148267


namespace NUMINAMATH_GPT_mark_total_spending_l1482_148209

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end NUMINAMATH_GPT_mark_total_spending_l1482_148209


namespace NUMINAMATH_GPT_expression_simplification_l1482_148266

theorem expression_simplification (a b : ℤ) : 
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1482_148266


namespace NUMINAMATH_GPT_first_motorcyclist_laps_per_hour_l1482_148291

noncomputable def motorcyclist_laps (x y z : ℝ) (P1 : 0 < x - y) (P2 : 0 < x - z) (P3 : 0 < y - z) : Prop :=
  (4.5 / (x - y) = 4.5) ∧ (4.5 / (x - z) = 4.5 - 0.5) ∧ (3 / (y - z) = 3) → x = 3

theorem first_motorcyclist_laps_per_hour (x y z : ℝ) (P1: 0 < x - y) (P2: 0 < x - z) (P3: 0 < y - z) :
  motorcyclist_laps x y z P1 P2 P3 →
  x = 3 :=
sorry

end NUMINAMATH_GPT_first_motorcyclist_laps_per_hour_l1482_148291


namespace NUMINAMATH_GPT_miley_discount_rate_l1482_148248

theorem miley_discount_rate :
  let cost_per_cellphone := 800
  let number_of_cellphones := 2
  let amount_paid := 1520
  let total_cost_without_discount := cost_per_cellphone * number_of_cellphones
  let discount_amount := total_cost_without_discount - amount_paid
  let discount_rate := (discount_amount / total_cost_without_discount) * 100
  discount_rate = 5 := by
    sorry

end NUMINAMATH_GPT_miley_discount_rate_l1482_148248


namespace NUMINAMATH_GPT_acute_angles_relation_l1482_148214

theorem acute_angles_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.sin α = (1 / 2) * Real.sin (α + β)) : α < β :=
sorry

end NUMINAMATH_GPT_acute_angles_relation_l1482_148214


namespace NUMINAMATH_GPT_inequality_holds_range_of_expression_l1482_148208

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem inequality_holds (x : ℝ) : f x < |x - 2| + 4 ↔ x ∈ Set.Ioo (-5 : ℝ) 3 := by
  sorry

theorem range_of_expression (m n : ℝ) (h : m + n = 2) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2) / m + (n^2 + 1) / n ∈ Set.Ici ((7 + 2 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_GPT_inequality_holds_range_of_expression_l1482_148208


namespace NUMINAMATH_GPT_find_number_l1482_148230

theorem find_number (number : ℝ) (h : 0.001 * number = 0.24) : number = 240 :=
sorry

end NUMINAMATH_GPT_find_number_l1482_148230


namespace NUMINAMATH_GPT_simplify_triangle_expression_l1482_148298

theorem simplify_triangle_expression (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c :=
by
  sorry

end NUMINAMATH_GPT_simplify_triangle_expression_l1482_148298


namespace NUMINAMATH_GPT_rectangle_in_triangle_area_l1482_148295

theorem rectangle_in_triangle_area (b h : ℕ) (hb : b = 12) (hh : h = 8)
  (x : ℕ) (hx : x = h / 2) : (b * x / 2) = 48 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_in_triangle_area_l1482_148295


namespace NUMINAMATH_GPT_chord_bisected_line_eq_l1482_148213

theorem chord_bisected_line_eq (x y : ℝ) (hx1 : x^2 + 4 * y^2 = 36) (hx2 : (4, 2) = ((x1 + x2) / 2, (y1 + y2) / 2)) :
  x + 2 * y - 8 = 0 :=
sorry

end NUMINAMATH_GPT_chord_bisected_line_eq_l1482_148213


namespace NUMINAMATH_GPT_find_quadratic_eq_l1482_148216

theorem find_quadratic_eq (x y : ℝ) 
  (h₁ : x + y = 10)
  (h₂ : |x - y| = 6) :
  x^2 - 10 * x + 16 = 0 :=
sorry

end NUMINAMATH_GPT_find_quadratic_eq_l1482_148216


namespace NUMINAMATH_GPT_ceil_sqrt_sum_l1482_148252

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end NUMINAMATH_GPT_ceil_sqrt_sum_l1482_148252


namespace NUMINAMATH_GPT_find_value_l1482_148279

theorem find_value (x : ℝ) (h : x^2 - 2 * x = 1) : 2023 + 6 * x - 3 * x^2 = 2020 := 
by 
sorry

end NUMINAMATH_GPT_find_value_l1482_148279


namespace NUMINAMATH_GPT_gcd_of_n13_minus_n_l1482_148255

theorem gcd_of_n13_minus_n : 
  ∀ n : ℤ, n ≠ 0 → 2730 ∣ (n ^ 13 - n) :=
by sorry

end NUMINAMATH_GPT_gcd_of_n13_minus_n_l1482_148255


namespace NUMINAMATH_GPT_find_three_digit_numbers_l1482_148235
open Nat

theorem find_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : ∀ (k : ℕ), n^k % 1000 = n % 1000) : n = 625 ∨ n = 376 :=
sorry

end NUMINAMATH_GPT_find_three_digit_numbers_l1482_148235


namespace NUMINAMATH_GPT_product_first_8_terms_l1482_148233

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a_2 : a 2 = 3 := sorry
def a_7 : a 7 = 1 := sorry

-- Proof statement
theorem product_first_8_terms (h_geom : is_geometric_sequence a q) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 1) : 
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 = 81) :=
sorry

end NUMINAMATH_GPT_product_first_8_terms_l1482_148233


namespace NUMINAMATH_GPT_inequality_proof_l1482_148263

theorem inequality_proof
  (a b x y z : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1482_148263


namespace NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l1482_148254

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Specify the given conditions
def a1 : ℚ := 10 / 11
def a15 : ℚ := 8 / 9

-- Prove that the eighth term is equal to 89 / 99
theorem arithmetic_sequence_eighth_term :
  ∃ d : ℚ, arithmetic_sequence a1 d 15 = a15 →
             arithmetic_sequence a1 d 8 = 89 / 99 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l1482_148254


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1482_148232

theorem parabola_focus_coordinates (x y : ℝ) (h : x = 2 * y^2) : (x, y) = (1/8, 0) :=
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1482_148232


namespace NUMINAMATH_GPT_boat_man_mass_l1482_148234

theorem boat_man_mass (L B h : ℝ) (rho g : ℝ): 
  L = 3 → B = 2 → h = 0.015 → rho = 1000 → g = 9.81 → (rho * L * B * h * g) / g = 9 :=
by
  intros
  simp_all
  sorry

end NUMINAMATH_GPT_boat_man_mass_l1482_148234


namespace NUMINAMATH_GPT_ratio_QP_l1482_148222

noncomputable def P : ℚ := 11 / 6
noncomputable def Q : ℚ := 5 / 2

theorem ratio_QP : Q / P = 15 / 11 := by 
  sorry

end NUMINAMATH_GPT_ratio_QP_l1482_148222


namespace NUMINAMATH_GPT_set_intersection_complement_l1482_148228

-- Definitions corresponding to conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 1}

-- Statement to prove
theorem set_intersection_complement : A ∩ (U \ B) = {x | -1 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1482_148228


namespace NUMINAMATH_GPT_stacy_savings_for_3_pairs_l1482_148226

-- Define the cost per pair of shorts
def cost_per_pair : ℕ := 10

-- Define the discount percentage as a decimal
def discount_percentage : ℝ := 0.1

-- Function to calculate the total cost without discount for n pairs
def total_cost_without_discount (n : ℕ) : ℕ := cost_per_pair * n

-- Function to calculate the total cost with discount for n pairs
noncomputable def total_cost_with_discount (n : ℕ) : ℝ :=
  if n >= 3 then
    let discount := discount_percentage * (cost_per_pair * n : ℝ)
    (cost_per_pair * n : ℝ) - discount
  else
    cost_per_pair * n

-- Function to calculate the savings for buying n pairs at once compared to individually
noncomputable def savings (n : ℕ) : ℝ :=
  (total_cost_without_discount n : ℝ) - total_cost_with_discount n

-- Proof statement
theorem stacy_savings_for_3_pairs : savings 3 = 3 := by
  sorry

end NUMINAMATH_GPT_stacy_savings_for_3_pairs_l1482_148226


namespace NUMINAMATH_GPT_correct_mark_l1482_148277

theorem correct_mark (x : ℕ) (S_Correct S_Wrong : ℕ) (n : ℕ) :
  n = 26 →
  S_Wrong = S_Correct + (83 - x) →
  (S_Wrong : ℚ) / n = (S_Correct : ℚ) / n + 1 / 2 →
  x = 70 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_correct_mark_l1482_148277


namespace NUMINAMATH_GPT_inheritance_amount_l1482_148264

theorem inheritance_amount (x : ℝ)
  (federal_tax_rate : ℝ := 0.25)
  (state_tax_rate : ℝ := 0.15)
  (total_taxes_paid : ℝ := 16000)
  (H : (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_taxes_paid) :
  x = 44138 := sorry

end NUMINAMATH_GPT_inheritance_amount_l1482_148264


namespace NUMINAMATH_GPT_trains_pass_each_other_l1482_148278

noncomputable def time_to_pass (speed1 speed2 distance : ℕ) : ℚ :=
  (distance : ℚ) / ((speed1 + speed2) : ℚ) * 60

theorem trains_pass_each_other :
  time_to_pass 60 80 100 = 42.86 := sorry

end NUMINAMATH_GPT_trains_pass_each_other_l1482_148278


namespace NUMINAMATH_GPT_add_to_fraction_l1482_148281

theorem add_to_fraction (x : ℕ) :
  (3 + x) / (11 + x) = 5 / 9 ↔ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_add_to_fraction_l1482_148281


namespace NUMINAMATH_GPT_expected_value_of_die_is_475_l1482_148270

-- Define the given probabilities
def prob_1 : ℚ := 1 / 12
def prob_2 : ℚ := 1 / 12
def prob_3 : ℚ := 1 / 6
def prob_4 : ℚ := 1 / 12
def prob_5 : ℚ := 1 / 12
def prob_6 : ℚ := 7 / 12

-- Define the expected value calculation
def expected_value := 
  prob_1 * 1 + prob_2 * 2 + prob_3 * 3 +
  prob_4 * 4 + prob_5 * 5 + prob_6 * 6

-- The problem statement to prove
theorem expected_value_of_die_is_475 : expected_value = 4.75 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_die_is_475_l1482_148270


namespace NUMINAMATH_GPT_gcd_lcm_product_eq_abc_l1482_148239

theorem gcd_lcm_product_eq_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  let D := Nat.gcd (Nat.gcd a b) c
  let m := Nat.lcm (Nat.lcm a b) c
  D * m = a * b * c :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_eq_abc_l1482_148239


namespace NUMINAMATH_GPT_find_nonzero_q_for_quadratic_l1482_148249

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_nonzero_q_for_quadratic_l1482_148249


namespace NUMINAMATH_GPT_probability_triangle_or_circle_l1482_148269

theorem probability_triangle_or_circle (total_figures triangles circles : ℕ) 
  (h1 : total_figures = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 3) : 
  (triangles + circles) / total_figures = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_triangle_or_circle_l1482_148269


namespace NUMINAMATH_GPT_monthly_income_l1482_148289

-- Define the conditions
variable (I : ℝ) -- Total monthly income
variable (remaining : ℝ) -- Remaining amount before donation
variable (remaining_after_donation : ℝ) -- Amount after donation

-- Conditions
def condition1 : Prop := remaining = I - 0.63 * I - 1500
def condition2 : Prop := remaining_after_donation = remaining - 0.05 * remaining
def condition3 : Prop := remaining_after_donation = 35000

-- Theorem to prove the total monthly income
theorem monthly_income (h1 : condition1 I remaining) (h2 : condition2 remaining remaining_after_donation) (h3 : condition3 remaining_after_donation) : I = 103600 := 
by sorry

end NUMINAMATH_GPT_monthly_income_l1482_148289


namespace NUMINAMATH_GPT_factor_1_factor_2_triangle_is_isosceles_l1482_148224

-- Factorization problems
theorem factor_1 (x y : ℝ) : 
  (x^2 - x * y + 4 * x - 4 * y) = ((x - y) * (x + 4)) :=
sorry

theorem factor_2 (x y : ℝ) : 
  (x^2 - y^2 + 4 * y - 4) = ((x + y - 2) * (x - y + 2)) :=
sorry

-- Triangle shape problem
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - a * c - b^2 + b * c = 0) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end NUMINAMATH_GPT_factor_1_factor_2_triangle_is_isosceles_l1482_148224


namespace NUMINAMATH_GPT_part1_part2_i_part2_ii_l1482_148231

def equation1 (x : ℝ) : Prop := 3 * x - 2 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 3 = 0
def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -7

def inequality1 (x : ℝ) : Prop := -x + 2 > x - 5
def inequality2 (x : ℝ) : Prop := 3 * x - 1 > -x + 2

def sys_ineq (x m : ℝ) : Prop := x + m < 2 * x ∧ x - 2 < m

def equation4 (x : ℝ) : Prop := (2 * x - 1) / 3 = -3

theorem part1 : 
  ∀ (x : ℝ), inequality1 x → inequality2 x → equation2 x → equation3 x :=
by sorry

theorem part2_i :
  ∀ (m : ℝ), (∃ (x : ℝ), equation4 x ∧ sys_ineq x m) → -6 < m ∧ m < -4 :=
by sorry

theorem part2_ii :
  ∀ (m : ℝ), ¬ (sys_ineq 1 m ∧ sys_ineq 2 m) → m ≥ 2 ∨ m ≤ -1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_i_part2_ii_l1482_148231


namespace NUMINAMATH_GPT_original_price_of_article_l1482_148219

theorem original_price_of_article (new_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) 
  (h_reduction : reduction_percentage = 56/100) (h_new_price : new_price = 4400) :
  original_price = 10000 :=
sorry

end NUMINAMATH_GPT_original_price_of_article_l1482_148219


namespace NUMINAMATH_GPT_common_ratio_geometric_sequence_l1482_148204

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

theorem common_ratio_geometric_sequence
  (a3_eq : a 3 = 2 * S 2 + 1)
  (a4_eq : a 4 = 2 * S 3 + 1)
  (geometric_seq : ∀ n, a (n+1) = a 1 * (q ^ n))
  (h₀ : a 1 ≠ 0)
  (h₁ : q ≠ 0) :
  q = 3 :=
sorry

end NUMINAMATH_GPT_common_ratio_geometric_sequence_l1482_148204


namespace NUMINAMATH_GPT_reading_time_equal_l1482_148261

/--
  Alice, Bob, and Chandra are reading a 760-page book. Alice reads a page in 20 seconds, 
  Bob reads a page in 45 seconds, and Chandra reads a page in 30 seconds. Prove that if 
  they divide the book into three sections such that each reads for the same length of 
  time, then each person will read for 7200 seconds.
-/
theorem reading_time_equal 
  (rate_A : ℝ := 1/20) 
  (rate_B : ℝ := 1/45) 
  (rate_C : ℝ := 1/30) 
  (total_pages : ℝ := 760) : 
  ∃ t : ℝ, t = 7200 ∧ 
    (t * rate_A + t * rate_B + t * rate_C = total_pages) := 
by
  sorry  -- proof to be provided

end NUMINAMATH_GPT_reading_time_equal_l1482_148261


namespace NUMINAMATH_GPT_result_after_subtraction_l1482_148250

-- Define the conditions
def x : ℕ := 40
def subtract_value : ℕ := 138

-- The expression we will evaluate
def result (x : ℕ) : ℕ := 6 * x - subtract_value

-- The theorem stating the evaluated result
theorem result_after_subtraction : result 40 = 102 :=
by
  unfold result
  rw [← Nat.mul_comm]
  simp
  sorry -- Proof placeholder

end NUMINAMATH_GPT_result_after_subtraction_l1482_148250


namespace NUMINAMATH_GPT_matrix_power_minus_l1482_148272

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![3, 4],
    ![0, 2]
  ]

theorem matrix_power_minus :
  B^15 - 3 • B^14 = ![
    ![0, 8192],
    ![0, -8192]
  ] :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_minus_l1482_148272


namespace NUMINAMATH_GPT_reciprocal_of_negative_one_sixth_l1482_148200

theorem reciprocal_of_negative_one_sixth : ∃ x : ℚ, - (1/6) * x = 1 ∧ x = -6 :=
by
  use -6
  constructor
  . sorry -- Need to prove - (1 / 6) * (-6) = 1
  . sorry -- Need to verify x = -6

end NUMINAMATH_GPT_reciprocal_of_negative_one_sixth_l1482_148200


namespace NUMINAMATH_GPT_find_range_of_m_l1482_148256

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3
def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15
def proposition_r (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def proposition_s (m : ℝ) : Prop := proposition_p m ∧ proposition_q m = False
def range_of_m (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

theorem find_range_of_m (m : ℝ) : proposition_r m ∧ proposition_s m → range_of_m m := by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l1482_148256


namespace NUMINAMATH_GPT_guilty_D_l1482_148247

def isGuilty (A B C D : Prop) : Prop :=
  ¬A ∧ (B → ∃! x, x ≠ A ∧ (x = C ∨ x = D)) ∧ (C → ∃! x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ A ∧ x₂ ≠ A ∧ ((x₁ = B ∨ x₁ = D) ∧ (x₂ = B ∨ x₂ = D))) ∧ (¬A ∨ B ∨ C ∨ D)

theorem guilty_D (A B C D : Prop) (h : isGuilty A B C D) : D :=
by
  sorry

end NUMINAMATH_GPT_guilty_D_l1482_148247


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1482_148223

theorem arithmetic_sequence_general_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9)
  : ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1482_148223


namespace NUMINAMATH_GPT_number_of_green_fish_and_carp_drawn_is_6_l1482_148220

-- Definitions/parameters from the problem
def total_fish := 80 + 20 + 40 + 40 + 20
def sample_size := 20
def number_of_green_fish := 20
def number_of_carp := 40
def probability_of_being_drawn := sample_size / total_fish

-- Theorem to prove the combined number of green fish and carp drawn is 6
theorem number_of_green_fish_and_carp_drawn_is_6 :
  (number_of_green_fish + number_of_carp) * probability_of_being_drawn = 6 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end NUMINAMATH_GPT_number_of_green_fish_and_carp_drawn_is_6_l1482_148220


namespace NUMINAMATH_GPT_Freddy_journey_time_l1482_148218

/-- Eddy and Freddy start simultaneously from city A. Eddy travels to city B, Freddy travels to city C.
    Eddy takes 3 hours from city A to city B, which is 900 km. The distance between city A and city C is
    300 km. The ratio of average speed of Eddy to Freddy is 4:1. Prove that Freddy takes 4 hours to travel. -/
theorem Freddy_journey_time (t_E : ℕ) (d_AB : ℕ) (d_AC : ℕ) (r : ℕ) (V_E V_F t_F : ℕ)
    (h1 : t_E = 3)
    (h2 : d_AB = 900)
    (h3 : d_AC = 300)
    (h4 : r = 4)
    (h5 : V_E = d_AB / t_E)
    (h6 : V_E = r * V_F)
    (h7 : t_F = d_AC / V_F)
  : t_F = 4 := 
  sorry

end NUMINAMATH_GPT_Freddy_journey_time_l1482_148218


namespace NUMINAMATH_GPT_best_player_total_hits_l1482_148215

theorem best_player_total_hits
  (team_avg_hits_per_game : ℕ)
  (games_played : ℕ)
  (total_players : ℕ)
  (other_players_avg_hits_next_6_games : ℕ)
  (correct_answer : ℕ)
  (h1 : team_avg_hits_per_game = 15)
  (h2 : games_played = 5)
  (h3 : total_players = 11)
  (h4 : other_players_avg_hits_next_6_games = 6)
  (h5 : correct_answer = 25) :
  ∃ total_hits_of_best_player : ℕ,
  total_hits_of_best_player = correct_answer := by
  sorry

end NUMINAMATH_GPT_best_player_total_hits_l1482_148215


namespace NUMINAMATH_GPT_find_a_l1482_148212

theorem find_a (x y z a : ℝ) (h1 : ∃ k : ℝ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) 
              (h2 : x + y + z = 70) 
              (h3 : y = 15 * a - 5) : 
  a = 5 / 3 := 
by sorry

end NUMINAMATH_GPT_find_a_l1482_148212


namespace NUMINAMATH_GPT_parallel_lines_l1482_148229

theorem parallel_lines :
  (∃ m: ℚ, (∀ x y: ℚ, (4 * y - 3 * x = 16 → y = m * x + (16 / 4)) ∧
                      (-3 * x - 4 * y = 15 → y = -m * x - (15 / 4)) ∧
                      (4 * y + 3 * x = 16 → y = -m * x + (16 / 4)) ∧
                      (3 * y + 4 * x = 15) → False)) :=
sorry

end NUMINAMATH_GPT_parallel_lines_l1482_148229


namespace NUMINAMATH_GPT_quadratic_distinct_zeros_range_l1482_148211

theorem quadratic_distinct_zeros_range (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - (k+1)*x1 + k + 4 = 0 ∧ x2^2 - (k+1)*x2 + k + 4 = 0)
  ↔ k ∈ (Set.Iio (-3) ∪ Set.Ioi 5) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_zeros_range_l1482_148211


namespace NUMINAMATH_GPT_kara_uses_28_cups_of_sugar_l1482_148276

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end NUMINAMATH_GPT_kara_uses_28_cups_of_sugar_l1482_148276


namespace NUMINAMATH_GPT_interval_representation_l1482_148253

def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem interval_representation : S = Set.Ioc (-1) 3 :=
sorry

end NUMINAMATH_GPT_interval_representation_l1482_148253


namespace NUMINAMATH_GPT_robot_material_handling_per_hour_min_num_type_A_robots_l1482_148285

-- Definitions and conditions for part 1
def material_handling_robot_B (x : ℕ) := x
def material_handling_robot_A (x : ℕ) := x + 30

def condition_time_handled (x : ℕ) :=
  1000 / material_handling_robot_A x = 800 / material_handling_robot_B x

-- Definitions for part 2
def total_robots := 20
def min_material_handling_per_hour := 2800

def material_handling_total (a b : ℕ) :=
  150 * a + 120 * b

-- Proof problems
theorem robot_material_handling_per_hour :
  ∃ (x : ℕ), material_handling_robot_B x = 120 ∧ material_handling_robot_A x = 150 ∧ condition_time_handled x :=
sorry

theorem min_num_type_A_robots :
  ∀ (a b : ℕ),
  a + b = total_robots →
  material_handling_total a b ≥ min_material_handling_per_hour →
  a ≥ 14 :=
sorry

end NUMINAMATH_GPT_robot_material_handling_per_hour_min_num_type_A_robots_l1482_148285


namespace NUMINAMATH_GPT_rationalize_denominator_l1482_148227

theorem rationalize_denominator 
  (cbrt32_eq_2cbrt4 : (32:ℝ)^(1/3) = 2 * (4:ℝ)^(1/3))
  (cbrt16_eq_2cbrt2 : (16:ℝ)^(1/3) = 2 * (2:ℝ)^(1/3))
  (cbrt64_eq_4 : (64:ℝ)^(1/3) = 4) :
  1 / ((4:ℝ)^(1/3) + (32:ℝ)^(1/3)) = ((2:ℝ)^(1/3)) / 6 :=
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1482_148227


namespace NUMINAMATH_GPT_consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l1482_148290

-- 6(a): Prove that the product of two consecutive integers is either divisible by 6 or gives a remainder of 2 when divided by 18.
theorem consecutive_integers_product (n : ℕ) : n * (n + 1) % 18 = 0 ∨ n * (n + 1) % 18 = 2 := 
sorry

-- 6(b): Prove that there does not exist an integer n such that the number 3n + 1 is the product of two consecutive integers.
theorem no_3n_plus_1_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, 3 * m + 1 = m * (m + 1) := 
sorry

-- 6(c): Prove that for no integer n, the number n^3 + 5n + 4 can be the product of two consecutive integers.
theorem no_n_cubed_plus_5n_plus_4_product_consecutive : ¬ ∃ n : ℕ, ∃ m : ℕ, n^3 + 5 * n + 4 = m * (m + 1) := 
sorry

-- 6(d): Prove that none of the numbers resulting from the rearrangement of the digits in 23456780 is the product of two consecutive integers.
def is_permutation (m : ℕ) (n : ℕ) : Prop := 
-- This function definition should check that m is a permutation of the digits of n
sorry

theorem no_permutation_23456780_product_consecutive : 
  ∀ m : ℕ, is_permutation m 23456780 → ¬ ∃ n : ℕ, m = n * (n + 1) := 
sorry

end NUMINAMATH_GPT_consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l1482_148290


namespace NUMINAMATH_GPT_extreme_points_inequality_l1482_148299

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log (1 - x)

theorem extreme_points_inequality (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1 / 4) 
  (h_sum : x1 + x2 = 1) (h_prod : x1 * x2 = a) (h_order : x1 < x2) :
  f x2 a - x1 > -(3 + Real.log 4) / 8 := 
by
  -- proof needed
  sorry

end NUMINAMATH_GPT_extreme_points_inequality_l1482_148299


namespace NUMINAMATH_GPT_Jason_spent_correct_amount_l1482_148286

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end NUMINAMATH_GPT_Jason_spent_correct_amount_l1482_148286


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l1482_148294

theorem minimum_value_of_fraction (a b : ℝ) (h1 : a > 2 * b) (h2 : 2 * b > 0) :
  (a^4 + 1) / (b * (a - 2 * b)) >= 16 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l1482_148294


namespace NUMINAMATH_GPT_seeds_in_big_garden_l1482_148262

-- Definitions based on conditions
def total_seeds : ℕ := 42
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 2
def seeds_planted_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden

-- Proof statement
theorem seeds_in_big_garden : total_seeds - seeds_planted_in_small_gardens = 36 :=
sorry

end NUMINAMATH_GPT_seeds_in_big_garden_l1482_148262


namespace NUMINAMATH_GPT_fraction_inequality_l1482_148242

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < (b + m) / (a + m) := 
sorry

end NUMINAMATH_GPT_fraction_inequality_l1482_148242


namespace NUMINAMATH_GPT_S_contains_finite_but_not_infinite_arith_progressions_l1482_148201

noncomputable def S : Set ℤ := {n | ∃ k : ℕ, n = Int.floor (k * Real.pi)}

theorem S_contains_finite_but_not_infinite_arith_progressions :
  (∀ (k : ℕ), ∃ (a d : ℤ), ∀ (i : ℕ) (h : i < k), (a + i * d) ∈ S) ∧
  ¬(∃ (a d : ℤ), ∀ (n : ℕ), (a + n * d) ∈ S) :=
by
  sorry

end NUMINAMATH_GPT_S_contains_finite_but_not_infinite_arith_progressions_l1482_148201


namespace NUMINAMATH_GPT_evaluate_expression_l1482_148243

theorem evaluate_expression :
  (π - 2023) ^ 0 + |(-9)| - 3 ^ 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1482_148243


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1482_148225

theorem ellipse_standard_equation :
  ∃ (a b c : ℝ),
    2 * a = 10 ∧
    c / a = 3 / 5 ∧
    b^2 = a^2 - c^2 ∧
    (∀ x y : ℝ, (x^2 / 16) + (y^2 / 25) = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1482_148225


namespace NUMINAMATH_GPT_meaningful_expression_condition_l1482_148258

theorem meaningful_expression_condition (x : ℝ) : (x > 1) ↔ (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_condition_l1482_148258


namespace NUMINAMATH_GPT_compute_expression_l1482_148284

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1482_148284


namespace NUMINAMATH_GPT_complex_multiplication_example_l1482_148237

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_multiplication_example (i : ℂ) (h : imaginary_unit i) :
  (3 + i) * (1 - 2 * i) = 5 - 5 * i := 
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_example_l1482_148237


namespace NUMINAMATH_GPT_gold_coins_percentage_l1482_148265

-- Definitions for conditions
def percent_beads : Float := 0.30
def percent_sculptures : Float := 0.10
def percent_silver_coins : Float := 0.30

-- Definitions derived from conditions
def percent_coins : Float := 1.0 - percent_beads - percent_sculptures
def percent_gold_coins_among_coins : Float := 1.0 - percent_silver_coins

-- Theorem statement
theorem gold_coins_percentage : percent_gold_coins_among_coins * percent_coins = 0.42 :=
by
sorry

end NUMINAMATH_GPT_gold_coins_percentage_l1482_148265


namespace NUMINAMATH_GPT_simplify_fraction_l1482_148246

theorem simplify_fraction :
  (45 * (14 / 25) * (1 / 18) * (5 / 11) : ℚ) = 7 / 11 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1482_148246


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l1482_148257

theorem probability_neither_red_nor_purple (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) : 
  total_balls = 60 →
  white_balls = 22 →
  green_balls = 18 →
  yellow_balls = 2 →
  red_balls = 15 →
  purple_balls = 3 →
  (total_balls - red_balls - purple_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l1482_148257


namespace NUMINAMATH_GPT_factorization_of_difference_of_squares_l1482_148273

theorem factorization_of_difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := 
by sorry

end NUMINAMATH_GPT_factorization_of_difference_of_squares_l1482_148273


namespace NUMINAMATH_GPT_remainder_div_101_l1482_148202

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_div_101_l1482_148202


namespace NUMINAMATH_GPT_range_of_m_l1482_148292

noncomputable def A (x : ℝ) : Prop := |x - 2| ≤ 4
noncomputable def B (x : ℝ) (m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) :
  (∀ x, (¬A x) → (¬B x m)) ∧ (∃ x, (¬B x m) ∧ ¬(¬A x)) → m ≥ 5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1482_148292


namespace NUMINAMATH_GPT_avg_A_lt_avg_B_combined_avg_eq_6_6_l1482_148244

-- Define the scores for A and B
def scores_A := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the average score function
def average (scores : List ℚ) : ℚ := (scores.sum : ℚ) / scores.length

-- Define the mean for the combined data
def combined_average : ℚ :=
  (average scores_A * scores_A.length + average scores_B * scores_B.length) / 
  (scores_A.length + scores_B.length)

-- Specify the variances given in the problem
def variance_A := 2.25
def variance_B := 4.41

-- Claim the average score of A is smaller than the average score of B
theorem avg_A_lt_avg_B : average scores_A < average scores_B := by sorry

-- Claim the average score of these 20 data points is 6.6
theorem combined_avg_eq_6_6 : combined_average = 6.6 := by sorry

end NUMINAMATH_GPT_avg_A_lt_avg_B_combined_avg_eq_6_6_l1482_148244


namespace NUMINAMATH_GPT_eval_poly_at_2_l1482_148238

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem eval_poly_at_2 :
  f 2 = 123 :=
by
  sorry

end NUMINAMATH_GPT_eval_poly_at_2_l1482_148238


namespace NUMINAMATH_GPT_hundred_squared_plus_two_hundred_one_is_composite_l1482_148293

theorem hundred_squared_plus_two_hundred_one_is_composite : 
    ¬ Prime (100^2 + 201) :=
by {
  sorry
}

end NUMINAMATH_GPT_hundred_squared_plus_two_hundred_one_is_composite_l1482_148293


namespace NUMINAMATH_GPT_total_amount_shared_l1482_148260

theorem total_amount_shared (jane mike nora total : ℝ) 
  (h1 : jane = 30) 
  (h2 : jane / 2 = mike / 3) 
  (h3 : mike / 3 = nora / 8) 
  (h4 : total = jane + mike + nora) : 
  total = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1482_148260


namespace NUMINAMATH_GPT_complement_of_angle_l1482_148275

variable (α : ℝ)

axiom given_angle : α = 63 + 21 / 60

theorem complement_of_angle :
  90 - α = 26 + 39 / 60 :=
by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l1482_148275


namespace NUMINAMATH_GPT_sum_cubed_identity_l1482_148245

theorem sum_cubed_identity
  (p q r : ℝ)
  (h1 : p + q + r = 5)
  (h2 : pq + pr + qr = 7)
  (h3 : pqr = -10) :
  p^3 + q^3 + r^3 = -10 := 
by
  sorry

end NUMINAMATH_GPT_sum_cubed_identity_l1482_148245


namespace NUMINAMATH_GPT_find_mistake_l1482_148241

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end NUMINAMATH_GPT_find_mistake_l1482_148241


namespace NUMINAMATH_GPT_find_x2_plus_y2_l1482_148268

-- Given conditions as definitions in Lean
variable {x y : ℝ}
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : x * y + x + y = 71)
variable (h4 : x^2 * y + x * y^2 = 880)

-- The statement to be proved
theorem find_x2_plus_y2 : x^2 + y^2 = 146 :=
by
  sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l1482_148268


namespace NUMINAMATH_GPT_find_rs_l1482_148297

-- Define a structure to hold the conditions
structure Conditions (r s : ℝ) : Prop :=
  (positive_r : 0 < r)
  (positive_s : 0 < s)
  (eq1 : r^3 + s^3 = 1)
  (eq2 : r^6 + s^6 = (15 / 16))

-- State the theorem
theorem find_rs (r s : ℝ) (h : Conditions r s) : rs = 1 / (48 : ℝ)^(1/3) :=
by
  sorry

end NUMINAMATH_GPT_find_rs_l1482_148297


namespace NUMINAMATH_GPT_multiply_square_expression_l1482_148296

theorem multiply_square_expression (x : ℝ) : ((-3 * x) ^ 2) * (2 * x) = 18 * x ^ 3 := by
  sorry

end NUMINAMATH_GPT_multiply_square_expression_l1482_148296


namespace NUMINAMATH_GPT_initial_milk_quantity_l1482_148203

theorem initial_milk_quantity (A B C D : ℝ) (hA : A > 0)
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (hTransferBC : B + 150 = C - 150 + 100)
  (hTransferDC : C - 50 = D - 100)
  (hEqual : B + 150 = D - 100) : 
  A = 1000 :=
by sorry

end NUMINAMATH_GPT_initial_milk_quantity_l1482_148203


namespace NUMINAMATH_GPT_ryan_weekly_commuting_time_l1482_148282

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end NUMINAMATH_GPT_ryan_weekly_commuting_time_l1482_148282


namespace NUMINAMATH_GPT_probability_two_female_one_male_l1482_148221

-- Define basic conditions
def total_contestants : Nat := 7
def female_contestants : Nat := 4
def male_contestants : Nat := 3
def choose_count : Nat := 3

-- Calculate combinations (binomial coefficients)
def comb (n k : Nat) : Nat := Nat.choose n k

-- Define the probability calculation steps in Lean
def total_ways := comb total_contestants choose_count
def favorable_ways_female := comb female_contestants 2
def favorable_ways_male := comb male_contestants 1
def favorable_ways := favorable_ways_female * favorable_ways_male

theorem probability_two_female_one_male :
  (favorable_ways : ℚ) / (total_ways : ℚ) = 18 / 35 := by
  sorry

end NUMINAMATH_GPT_probability_two_female_one_male_l1482_148221


namespace NUMINAMATH_GPT_determine_c_l1482_148240

theorem determine_c (c : ℝ) :
  let vertex_x := -(-10 / (2 * 1))
  let vertex_y := c - ((-10)^2 / (4 * 1))
  ((5 - 0)^2 + (vertex_y - 0)^2 = 10^2)
  → (c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_determine_c_l1482_148240


namespace NUMINAMATH_GPT_John_avg_speed_l1482_148271

theorem John_avg_speed :
  ∀ (initial final : ℕ) (time : ℕ),
    initial = 27372 →
    final = 27472 →
    time = 4 →
    ((final - initial) / time) = 25 :=
by
  intros initial final time h_initial h_final h_time
  sorry

end NUMINAMATH_GPT_John_avg_speed_l1482_148271


namespace NUMINAMATH_GPT_value_of_a_l1482_148217

def quadratic_vertex (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

def vertex_form (a h k x : ℤ) : ℤ :=
  a * (x - h)^2 + k

theorem value_of_a (a b c : ℤ) (h k x1 y1 x2 y2 : ℤ) (H_vert : h = 2) (H_vert_val : k = 3)
  (H_point : x1 = 1) (H_point_val : y1 = 5) (H_graph : ∀ x, quadratic_vertex a b c x = vertex_form a h k x) :
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1482_148217


namespace NUMINAMATH_GPT_roots_difference_is_one_l1482_148288

noncomputable def quadratic_eq (p : ℝ) :=
  ∃ (α β : ℝ), (α ≠ β) ∧ (α - β = 1) ∧ (α ^ 2 - p * α + (p ^ 2 - 1) / 4 = 0) ∧ (β ^ 2 - p * β + (p ^ 2 - 1) / 4 = 0)

theorem roots_difference_is_one (p : ℝ) : quadratic_eq p :=
  sorry

end NUMINAMATH_GPT_roots_difference_is_one_l1482_148288


namespace NUMINAMATH_GPT_remainder_3_pow_20_div_5_l1482_148206

theorem remainder_3_pow_20_div_5 : (3 ^ 20) % 5 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_3_pow_20_div_5_l1482_148206


namespace NUMINAMATH_GPT_concrete_volume_is_six_l1482_148287

def to_yards (feet : ℕ) (inches : ℕ) : ℚ :=
  feet * (1 / 3) + inches * (1 / 36)

def sidewalk_volume (width_feet : ℕ) (length_feet : ℕ) (thickness_inches : ℕ) : ℚ :=
  to_yards width_feet 0 * to_yards length_feet 0 * to_yards 0 thickness_inches

def border_volume (border_width_feet : ℕ) (border_thickness_inches : ℕ) (sidewalk_length_feet : ℕ) : ℚ :=
  to_yards (2 * border_width_feet) 0 * to_yards sidewalk_length_feet 0 * to_yards 0 border_thickness_inches

def total_concrete_volume (sidewalk_width_feet : ℕ) (sidewalk_length_feet : ℕ) (sidewalk_thickness_inches : ℕ)
  (border_width_feet : ℕ) (border_thickness_inches : ℕ) : ℚ :=
  sidewalk_volume sidewalk_width_feet sidewalk_length_feet sidewalk_thickness_inches +
  border_volume border_width_feet border_thickness_inches sidewalk_length_feet

def volume_in_cubic_yards (w1_feet : ℕ) (l1_feet : ℕ) (t1_inches : ℕ) (w2_feet : ℕ) (t2_inches : ℕ) : ℚ :=
  total_concrete_volume w1_feet l1_feet t1_inches w2_feet t2_inches

theorem concrete_volume_is_six :
  -- conditions
  volume_in_cubic_yards 4 80 4 1 2 = 6 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_concrete_volume_is_six_l1482_148287


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1482_148274

theorem hyperbola_eccentricity 
  (p1 p2 : ℝ × ℝ)
  (asymptote_passes_through_p1 : p1 = (1, 2))
  (hyperbola_passes_through_p2 : p2 = (2 * Real.sqrt 2, 4)) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1482_148274


namespace NUMINAMATH_GPT_minimum_value_of_function_l1482_148251

theorem minimum_value_of_function (x : ℝ) (hx : x > 4) : 
    (∃ y : ℝ, y = x + 9 / (x - 4) ∧ (∀ z : ℝ, (∃ w : ℝ, w > 4 ∧ z = w + 9 / (w - 4)) → z ≥ 10) ∧ y = 10) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1482_148251


namespace NUMINAMATH_GPT_min_disks_required_l1482_148280

-- Define the initial conditions
def num_files : ℕ := 40
def disk_capacity : ℕ := 2 -- capacity in MB
def num_files_1MB : ℕ := 5
def num_files_0_8MB : ℕ := 15
def num_files_0_5MB : ℕ := 20
def size_1MB : ℕ := 1
def size_0_8MB : ℕ := 8/10 -- 0.8 MB
def size_0_5MB : ℕ := 1/2 -- 0.5 MB

-- Define the mathematical problem
theorem min_disks_required :
  (num_files_1MB * size_1MB + num_files_0_8MB * size_0_8MB + num_files_0_5MB * size_0_5MB) / disk_capacity ≤ 15 := by
  sorry

end NUMINAMATH_GPT_min_disks_required_l1482_148280


namespace NUMINAMATH_GPT_ratio_sqrt5_over_5_l1482_148210

noncomputable def radius_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
a / b

theorem ratio_sqrt5_over_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) :
  radius_ratio a b h = 1 / Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_ratio_sqrt5_over_5_l1482_148210
