import Mathlib

namespace NUMINAMATH_GPT_checkered_rectangles_containing_one_gray_cell_l2190_219028

def total_number_of_rectangles_with_one_gray_cell :=
  let gray_cells := 40
  let blue_cells := 36
  let red_cells := 4
  
  let blue_rectangles_each := 4
  let red_rectangles_each := 8
  
  (blue_cells * blue_rectangles_each) + (red_cells * red_rectangles_each)

theorem checkered_rectangles_containing_one_gray_cell : total_number_of_rectangles_with_one_gray_cell = 176 :=
by 
  sorry

end NUMINAMATH_GPT_checkered_rectangles_containing_one_gray_cell_l2190_219028


namespace NUMINAMATH_GPT_Lorelai_jellybeans_correct_l2190_219023

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end NUMINAMATH_GPT_Lorelai_jellybeans_correct_l2190_219023


namespace NUMINAMATH_GPT_odometer_problem_l2190_219055

theorem odometer_problem
  (a b c : ℕ) -- a, b, c are natural numbers
  (h1 : 1 ≤ a) -- condition (a ≥ 1)
  (h2 : a + b + c ≤ 7) -- condition (a + b + c ≤ 7)
  (h3 : 99 * (c - a) % 55 = 0) -- 99(c - a) must be divisible by 55
  (h4 : 100 * a + 10 * b + c < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  (h5 : 100 * c + 10 * b + a < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  : a^2 + b^2 + c^2 = 37 := sorry

end NUMINAMATH_GPT_odometer_problem_l2190_219055


namespace NUMINAMATH_GPT_usual_time_28_l2190_219086

theorem usual_time_28 (R T : ℝ) (h1 : ∀ (d : ℝ), d = R * T)
  (h2 : ∀ (d : ℝ), d = (6/7) * R * (T - 4)) : T = 28 :=
by
  -- Variables:
  -- R : Usual rate of the boy
  -- T : Usual time to reach the school
  -- h1 : Expressing distance in terms of usual rate and time
  -- h2 : Expressing distance in terms of reduced rate and time minus 4
  sorry

end NUMINAMATH_GPT_usual_time_28_l2190_219086


namespace NUMINAMATH_GPT_resulting_expression_l2190_219083

def x : ℕ := 1000
def y : ℕ := 10

theorem resulting_expression : 
  (x + 2 * y) + x + 3 * y + x + 4 * y + x + y = 4 * x + 10 * y :=
by
  sorry

end NUMINAMATH_GPT_resulting_expression_l2190_219083


namespace NUMINAMATH_GPT_complex_number_solution_l2190_219056

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l2190_219056


namespace NUMINAMATH_GPT_odd_number_adjacent_product_diff_l2190_219014

variable (x : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem odd_number_adjacent_product_diff (h : is_odd x)
  (adjacent_diff : x * (x + 2) - x * (x - 2) = 44) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_odd_number_adjacent_product_diff_l2190_219014


namespace NUMINAMATH_GPT_sum_interior_ninth_row_l2190_219071

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_interior_ninth_row_l2190_219071


namespace NUMINAMATH_GPT_transportation_tax_correct_l2190_219068

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def total_months_in_year : ℕ := 12

def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / total_months_in_year

theorem transportation_tax_correct :
  adjusted_tax = 3125 := by
  sorry

end NUMINAMATH_GPT_transportation_tax_correct_l2190_219068


namespace NUMINAMATH_GPT_area_of_region_R_l2190_219073

open Real

noncomputable def area_of_strip (width : ℝ) (height : ℝ) : ℝ :=
  width * height

noncomputable def area_of_triangle (leg : ℝ) : ℝ :=
  1 / 2 * leg * leg

theorem area_of_region_R :
  let unit_square_area := 1
  let AE_BE := 1 / sqrt 2
  let area_triangle_ABE := area_of_triangle AE_BE
  let strip_width := 1 / 4
  let strip_height := 1
  let area_strip := area_of_strip strip_width strip_height
  let overlap_area := area_triangle_ABE / 2
  let area_R := area_strip - overlap_area
  area_R = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_R_l2190_219073


namespace NUMINAMATH_GPT_equation_descr_circle_l2190_219033

theorem equation_descr_circle : ∀ (x y : ℝ), (x - 0) ^ 2 + (y - 0) ^ 2 = 25 → ∃ (c : ℝ × ℝ) (r : ℝ), c = (0, 0) ∧ r = 5 ∧ ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_equation_descr_circle_l2190_219033


namespace NUMINAMATH_GPT_remaining_amount_eq_40_l2190_219058

-- Definitions and conditions
def initial_amount : ℕ := 100
def food_spending : ℕ := 20
def rides_spending : ℕ := 2 * food_spending
def total_spending : ℕ := food_spending + rides_spending

-- The proposition to be proved
theorem remaining_amount_eq_40 :
  initial_amount - total_spending = 40 :=
by
  sorry

end NUMINAMATH_GPT_remaining_amount_eq_40_l2190_219058


namespace NUMINAMATH_GPT_student_finished_6_problems_in_class_l2190_219099

theorem student_finished_6_problems_in_class (total_problems : ℕ) (x y : ℕ) (h1 : total_problems = 15) (h2 : 3 * y = 2 * x) (h3 : x + y = total_problems) : y = 6 :=
sorry

end NUMINAMATH_GPT_student_finished_6_problems_in_class_l2190_219099


namespace NUMINAMATH_GPT_Wilson_sledding_l2190_219039

variable (T S : ℕ)

theorem Wilson_sledding (h1 : S = T / 2) (h2 : (2 * T) + (3 * S) = 14) : T = 4 := by
  sorry

end NUMINAMATH_GPT_Wilson_sledding_l2190_219039


namespace NUMINAMATH_GPT_negation_of_real_root_proposition_l2190_219034

theorem negation_of_real_root_proposition :
  (¬ ∃ m : ℝ, ∃ (x : ℝ), x^2 + m * x + 1 = 0) ↔ (∀ m : ℝ, ∀ (x : ℝ), x^2 + m * x + 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_real_root_proposition_l2190_219034


namespace NUMINAMATH_GPT_triangle_area_is_correct_l2190_219092

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_correct : 
  area_of_triangle (1, 3) (5, -2) (8, 6) = 23.5 := 
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_correct_l2190_219092


namespace NUMINAMATH_GPT_equivalent_equation_l2190_219035

theorem equivalent_equation (x y : ℝ) 
  (x_ne_0 : x ≠ 0) (x_ne_3 : x ≠ 3) 
  (y_ne_0 : y ≠ 0) (y_ne_5 : y ≠ 5)
  (main_equation : (3 / x) + (4 / y) = 1 / 3) : 
  x = 9 * y / (y - 12) :=
sorry

end NUMINAMATH_GPT_equivalent_equation_l2190_219035


namespace NUMINAMATH_GPT_fever_above_threshold_l2190_219018

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end NUMINAMATH_GPT_fever_above_threshold_l2190_219018


namespace NUMINAMATH_GPT_rectangle_area_l2190_219053

structure Rectangle where
  length : ℕ    -- Length of the rectangle in cm
  width : ℕ     -- Width of the rectangle in cm
  perimeter : ℕ -- Perimeter of the rectangle in cm
  h : length = width + 4 -- Distance condition from the diagonal intersection

theorem rectangle_area (r : Rectangle) (h_perim : r.perimeter = 56) : r.length * r.width = 192 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2190_219053


namespace NUMINAMATH_GPT_find_principal_l2190_219075

theorem find_principal (CI SI : ℝ) (hCI : CI = 11730) (hSI : SI = 10200)
  (P R : ℝ)
  (hSI_form : SI = P * R * 2 / 100)
  (hCI_form : CI = P * (1 + R / 100)^2 - P) :
  P = 34000 := by
  sorry

end NUMINAMATH_GPT_find_principal_l2190_219075


namespace NUMINAMATH_GPT_no_real_solution_ratio_l2190_219031

theorem no_real_solution_ratio (x : ℝ) : (x + 3) / (2 * x + 5) = (5 * x + 4) / (8 * x + 5) → false :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_solution_ratio_l2190_219031


namespace NUMINAMATH_GPT_cost_to_fly_A_to_B_l2190_219082

noncomputable def flight_cost (distance : ℕ) : ℕ := (distance * 10 / 100) + 100

theorem cost_to_fly_A_to_B :
  flight_cost 3250 = 425 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_fly_A_to_B_l2190_219082


namespace NUMINAMATH_GPT_slope_ratio_l2190_219005

theorem slope_ratio (s t k b : ℝ) 
  (h1: b = -12 * s)
  (h2: b = k - 7) 
  (ht: t = (7 - k) / 7) 
  (hs: s = (7 - k) / 12): 
  s / t = 7 / 12 := 
  sorry

end NUMINAMATH_GPT_slope_ratio_l2190_219005


namespace NUMINAMATH_GPT_sufficient_condition_l2190_219036

theorem sufficient_condition (a b : ℝ) (h : |a + b| > 1) : |a| + |b| > 1 := 
by sorry

end NUMINAMATH_GPT_sufficient_condition_l2190_219036


namespace NUMINAMATH_GPT_total_boys_fraction_of_girls_l2190_219089

theorem total_boys_fraction_of_girls
  (n : ℕ)
  (b1 g1 b2 g2 : ℕ)
  (h_equal_students : b1 + g1 = b2 + g2)
  (h_ratio_class1 : b1 / g1 = 2 / 3)
  (h_ratio_class2: b2 / g2 = 4 / 5) :
  ((b1 + b2) / (g1 + g2) = 19 / 26) :=
by sorry

end NUMINAMATH_GPT_total_boys_fraction_of_girls_l2190_219089


namespace NUMINAMATH_GPT_oranges_sold_l2190_219087

def bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : (bags * oranges_per_bag) - rotten_oranges - oranges_for_juice = 220 := by
  sorry

end NUMINAMATH_GPT_oranges_sold_l2190_219087


namespace NUMINAMATH_GPT_swimming_pool_length_l2190_219004

theorem swimming_pool_length :
  ∀ (w d1 d2 V : ℝ), w = 9 → d1 = 1 → d2 = 4 → V = 270 → 
  (((V = (1 / 2) * (d1 + d2) * w * l) → l = 12)) :=
by
  intros w d1 d2 V hw hd1 hd2 hV hv
  simp only [hw, hd1, hd2, hV] at hv
  sorry

end NUMINAMATH_GPT_swimming_pool_length_l2190_219004


namespace NUMINAMATH_GPT_problem1_problem2_l2190_219096

-- Define the conditions: f is an odd and decreasing function on [-1, 1]
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_decreasing : ∀ x y, x ≤ y → f y ≤ f x)

-- The domain of interest is [-1, 1]
variable (x1 x2 : ℝ)
variable (h_x1 : x1 ∈ Set.Icc (-1 : ℝ) 1)
variable (h_x2 : x2 ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 1
theorem problem1 : (f x1 + f x2) * (x1 + x2) ≤ 0 := by
  sorry

-- Assume condition for Problem 2
variable (a : ℝ)
variable (h_ineq : f (1 - a) + f (1 - a ^ 2) < 0)
variable (h_dom : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → x ∈ Set.Icc (-1 : ℝ) 1)

-- Proof Problem 2
theorem problem2 : 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2190_219096


namespace NUMINAMATH_GPT_multiplication_verification_l2190_219090

-- Define the variables
variables (P Q R S T U : ℕ)

-- Define the known digits in the numbers
def multiplicand := 60000 + 1000 * P + 100 * Q + 10 * R
def multiplier := 5000000 + 10000 * S + 1000 * T + 100 * U + 5

-- Define the proof statement
theorem multiplication_verification : 
  (multiplicand P Q R) * (multiplier S T U) = 20213 * 732575 :=
  sorry

end NUMINAMATH_GPT_multiplication_verification_l2190_219090


namespace NUMINAMATH_GPT_least_possible_value_of_d_l2190_219088

theorem least_possible_value_of_d
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  (z - x) = 9 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_d_l2190_219088


namespace NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_5_l2190_219095

theorem remainder_of_3_pow_2023_mod_5 : (3^2023) % 5 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_5_l2190_219095


namespace NUMINAMATH_GPT_var_of_or_l2190_219048

theorem var_of_or (p q : Prop) (h : ¬ (p ∧ q)) : (p ∨ q = true) ∨ (p ∨ q = false) :=
by
  sorry

end NUMINAMATH_GPT_var_of_or_l2190_219048


namespace NUMINAMATH_GPT_evaluate_expression_l2190_219038

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2190_219038


namespace NUMINAMATH_GPT_find_a_minus_inv_a_l2190_219010

variable (a : ℝ)
variable (h : a + 1 / a = Real.sqrt 13)

theorem find_a_minus_inv_a : a - 1 / a = 3 ∨ a - 1 / a = -3 := by
  sorry

end NUMINAMATH_GPT_find_a_minus_inv_a_l2190_219010


namespace NUMINAMATH_GPT_a_number_M_middle_digit_zero_l2190_219064

theorem a_number_M_middle_digit_zero (d e f M : ℕ) (h1 : M = 36 * d + 6 * e + f)
  (h2 : M = 64 * f + 8 * e + d) (hd : d < 6) (he : e < 6) (hf : f < 6) : e = 0 :=
by sorry

end NUMINAMATH_GPT_a_number_M_middle_digit_zero_l2190_219064


namespace NUMINAMATH_GPT_scientific_notation_826M_l2190_219009

theorem scientific_notation_826M : 826000000 = 8.26 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_826M_l2190_219009


namespace NUMINAMATH_GPT_sum_of_squares_is_289_l2190_219020

theorem sum_of_squares_is_289 (x y : ℤ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_is_289_l2190_219020


namespace NUMINAMATH_GPT_radius_correct_l2190_219025

noncomputable def radius_of_circle (chord_length tang_secant_segment : ℝ) : ℝ :=
  let r := 6.25
  r

theorem radius_correct
  (chord_length : ℝ)
  (tangent_secant_segment : ℝ)
  (parallel_secant_internal_segment : ℝ)
  : chord_length = 10 ∧ parallel_secant_internal_segment = 12 → radius_of_circle chord_length parallel_secant_internal_segment = 6.25 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_radius_correct_l2190_219025


namespace NUMINAMATH_GPT_quadratic_solution_l2190_219078

theorem quadratic_solution (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 → (x = 1 / 2 ∨ x = 1) :=
by sorry

end NUMINAMATH_GPT_quadratic_solution_l2190_219078


namespace NUMINAMATH_GPT_basketball_cards_price_l2190_219061

theorem basketball_cards_price :
  let toys_cost := 3 * 10
  let shirts_cost := 5 * 6
  let total_cost := 70
  let basketball_cards_cost := total_cost - (toys_cost + shirts_cost)
  let packs_of_cards := 2
  (basketball_cards_cost / packs_of_cards) = 5 :=
by
  sorry

end NUMINAMATH_GPT_basketball_cards_price_l2190_219061


namespace NUMINAMATH_GPT_train_length_l2190_219076

theorem train_length
  (S : ℝ)
  (L : ℝ)
  (h1 : L + 140 = S * 15)
  (h2 : L + 250 = S * 20) :
  L = 190 :=
by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_train_length_l2190_219076


namespace NUMINAMATH_GPT_find_f_sqrt_5753_l2190_219047

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_sqrt_5753 (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)
  (h2 : ∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993)) :
  f (Real.sqrt 5753) = 0 :=
sorry

end NUMINAMATH_GPT_find_f_sqrt_5753_l2190_219047


namespace NUMINAMATH_GPT_focus_of_parabola_l2190_219063

theorem focus_of_parabola (m : ℝ) (m_nonzero : m ≠ 0) :
    ∃ (focus_x focus_y : ℝ), (focus_x, focus_y) = (m, 0) ∧
        ∀ (y : ℝ), (x = 1/(4*m) * y^2) := 
sorry

end NUMINAMATH_GPT_focus_of_parabola_l2190_219063


namespace NUMINAMATH_GPT_geometric_sequence_logarithm_identity_l2190_219024

variable {a : ℕ+ → ℝ}

-- Assumptions
def common_ratio (a : ℕ+ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_logarithm_identity
  (r : ℝ)
  (hr : r = -Real.sqrt 2)
  (h : common_ratio a r) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_logarithm_identity_l2190_219024


namespace NUMINAMATH_GPT_gcd_polynomial_l2190_219067

theorem gcd_polynomial (b : ℤ) (h : 570 ∣ b) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 :=
sorry

end NUMINAMATH_GPT_gcd_polynomial_l2190_219067


namespace NUMINAMATH_GPT_geometric_seq_seventh_term_l2190_219006

theorem geometric_seq_seventh_term (a r : ℕ) (r_pos : r > 0) (first_term : a = 3)
    (fifth_term : a * r^4 = 243) : a * r^6 = 2187 := by
  sorry

end NUMINAMATH_GPT_geometric_seq_seventh_term_l2190_219006


namespace NUMINAMATH_GPT_div_by_17_l2190_219040

theorem div_by_17 (n : ℕ) (h : ¬ 17 ∣ n) : 17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := 
by sorry

end NUMINAMATH_GPT_div_by_17_l2190_219040


namespace NUMINAMATH_GPT_not_necessarily_periodic_l2190_219080

-- Define the conditions of the problem
noncomputable def a : ℕ → ℕ := sorry
noncomputable def t : ℕ → ℕ := sorry
axiom h_t : ∀ k : ℕ, ∃ t_k : ℕ, ∀ n : ℕ, a (k + n * t_k) = a k

-- The theorem stating that the sequence is not necessarily periodic
theorem not_necessarily_periodic : ¬ ∃ T : ℕ, ∀ k : ℕ, a (k + T) = a k := sorry

end NUMINAMATH_GPT_not_necessarily_periodic_l2190_219080


namespace NUMINAMATH_GPT_number_of_articles_l2190_219017

-- Define main conditions
variable (N : ℕ) -- Number of articles
variable (CP SP : ℝ) -- Cost price and Selling price per article

-- Condition 1: Cost price of N articles equals the selling price of 15 articles
def condition1 : Prop := N * CP = 15 * SP

-- Condition 2: Selling price includes a 33.33% profit on cost price
def condition2 : Prop := SP = CP * 1.3333

-- Prove that the number of articles N equals 20
theorem number_of_articles (h1 : condition1 N CP SP) (h2 : condition2 CP SP) : N = 20 :=
by sorry

end NUMINAMATH_GPT_number_of_articles_l2190_219017


namespace NUMINAMATH_GPT_center_circle_is_correct_l2190_219081

noncomputable def find_center_of_circle : ℝ × ℝ :=
  let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
  let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
  let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
  let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
  (-18, -11)

theorem center_circle_is_correct (x y : ℝ) :
  (let line1 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = 20
   let line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -40
   let center_line : ℝ → ℝ → Prop := λ x y => x - 3 * y = 15
   let mid_line : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y = -10
   (x, y) = find_center_of_circle) :=
  sorry

end NUMINAMATH_GPT_center_circle_is_correct_l2190_219081


namespace NUMINAMATH_GPT_find_amount_l2190_219042

-- Definitions based on the conditions provided
def gain : ℝ := 0.70
def gain_percent : ℝ := 1.0

-- The theorem statement
theorem find_amount (h : gain_percent = 1) : ∀ (amount : ℝ), amount = gain / (gain_percent / 100) → amount = 70 :=
by
  intros amount h_calc
  sorry

end NUMINAMATH_GPT_find_amount_l2190_219042


namespace NUMINAMATH_GPT_marbles_with_at_least_one_blue_l2190_219013

theorem marbles_with_at_least_one_blue :
  (Nat.choose 10 4) - (Nat.choose 8 4) = 140 :=
by
  sorry

end NUMINAMATH_GPT_marbles_with_at_least_one_blue_l2190_219013


namespace NUMINAMATH_GPT_algebra_inequality_l2190_219070

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_algebra_inequality_l2190_219070


namespace NUMINAMATH_GPT_intersection_A_B_l2190_219045

def A : Set ℝ := { x | 1 < x - 1 ∧ x - 1 ≤ 3 }
def B : Set ℝ := { 2, 3, 4 }

theorem intersection_A_B : A ∩ B = {3, 4} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2190_219045


namespace NUMINAMATH_GPT_machine_P_additional_hours_unknown_l2190_219029

noncomputable def machine_A_rate : ℝ := 1.0000000000000013

noncomputable def machine_Q_rate : ℝ := machine_A_rate + 0.10 * machine_A_rate

noncomputable def total_sprockets : ℝ := 110

noncomputable def machine_Q_hours : ℝ := total_sprockets / machine_Q_rate

variable (x : ℝ) -- additional hours taken by Machine P

theorem machine_P_additional_hours_unknown :
  ∃ x, total_sprockets / machine_Q_rate + x = total_sprockets / ((total_sprockets + total_sprockets / machine_Q_rate * x) / total_sprockets) :=
sorry

end NUMINAMATH_GPT_machine_P_additional_hours_unknown_l2190_219029


namespace NUMINAMATH_GPT_salt_solution_concentration_l2190_219051

theorem salt_solution_concentration (m x : ℝ) (h1 : m > 30) (h2 : (m * m / 100) = ((m - 20) / 100) * (m + 2 * x)) :
  x = 10 * m / (m + 20) :=
sorry

end NUMINAMATH_GPT_salt_solution_concentration_l2190_219051


namespace NUMINAMATH_GPT_ceil_sqrt_196_eq_14_l2190_219094

theorem ceil_sqrt_196_eq_14 : ⌈Real.sqrt 196⌉ = 14 := 
by 
  sorry

end NUMINAMATH_GPT_ceil_sqrt_196_eq_14_l2190_219094


namespace NUMINAMATH_GPT_trillion_in_scientific_notation_l2190_219015

theorem trillion_in_scientific_notation :
  (10^4) * (10^4) * (10^4) = 10^(12) := 
by sorry

end NUMINAMATH_GPT_trillion_in_scientific_notation_l2190_219015


namespace NUMINAMATH_GPT_find_d_plus_f_l2190_219021

noncomputable def a : ℂ := sorry
noncomputable def c : ℂ := sorry
noncomputable def e : ℂ := -2 * a - c
noncomputable def d : ℝ := sorry
noncomputable def f : ℝ := sorry

theorem find_d_plus_f (a c e : ℂ) (d f : ℝ) (h₁ : e = -2 * a - c) (h₂ : a.im + d + f = 4) (h₃ : a.re + c.re + e.re = 0) (h₄ : 2 + d + f = 4) : d + f = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_d_plus_f_l2190_219021


namespace NUMINAMATH_GPT_student_range_exact_student_count_l2190_219060

-- Definitions for the conditions
def retail_price (x : ℕ) : ℕ := 240
def wholesale_price (x : ℕ) : ℕ := 260 / (x + 60)

def student_conditions (x : ℕ) : Prop := (x < 250) ∧ (x + 60 ≥ 250)
def wholesale_retail_equation (a : ℕ) : Prop := (240^2 / a) * 240 = (260 / (a+60)) * 288

-- Proofs of the required statements
theorem student_range (x : ℕ) (hc : student_conditions x) : 190 ≤ x ∧ x < 250 :=
by {
  sorry
}

theorem exact_student_count (a : ℕ) (heq : wholesale_retail_equation a) : a = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_student_range_exact_student_count_l2190_219060


namespace NUMINAMATH_GPT_sum_four_least_tau_equals_eight_l2190_219049

def tau (n : ℕ) : ℕ := n.divisors.card

theorem sum_four_least_tau_equals_eight :
  ∃ n1 n2 n3 n4 : ℕ, 
    tau n1 + tau (n1 + 1) = 8 ∧ 
    tau n2 + tau (n2 + 1) = 8 ∧
    tau n3 + tau (n3 + 1) = 8 ∧
    tau n4 + tau (n4 + 1) = 8 ∧
    n1 + n2 + n3 + n4 = 80 := 
sorry

end NUMINAMATH_GPT_sum_four_least_tau_equals_eight_l2190_219049


namespace NUMINAMATH_GPT_cherries_per_quart_of_syrup_l2190_219057

-- Definitions based on conditions
def time_to_pick_cherries : ℚ := 2
def cherries_picked_in_time : ℚ := 300
def time_to_make_syrup : ℚ := 3
def total_time_for_all_syrup : ℚ := 33
def total_quarts : ℚ := 9

-- Derivation of how many cherries are needed per quart
theorem cherries_per_quart_of_syrup : 
  (cherries_picked_in_time / time_to_pick_cherries) * (total_time_for_all_syrup - total_quarts * time_to_make_syrup) / total_quarts = 100 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_cherries_per_quart_of_syrup_l2190_219057


namespace NUMINAMATH_GPT_eccentricity_range_l2190_219026

-- We start with the given problem and conditions
variables {a c b : ℝ}
def C1 := ∀ x y, x^2 + 2 * c * x + y^2 = 0
def C2 := ∀ x y, x^2 - 2 * c * x + y^2 = 0
def ellipse := ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

-- Ellipse semi-latus rectum condition and circles inside the ellipse
axiom h1 : c = b^2 / a
axiom h2 : a > 2 * c

-- Proving the range of the eccentricity
theorem eccentricity_range : 0 < c / a ∧ c / a < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_range_l2190_219026


namespace NUMINAMATH_GPT_difference_cubed_divisible_by_27_l2190_219044

theorem difference_cubed_divisible_by_27 (a b : ℤ) :
    ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3) % 27 = 0 := 
by
  sorry

end NUMINAMATH_GPT_difference_cubed_divisible_by_27_l2190_219044


namespace NUMINAMATH_GPT_probability_all_six_draws_white_l2190_219041

theorem probability_all_six_draws_white :
  let total_balls := 14
  let white_balls := 7
  let single_draw_white_probability := (white_balls : ℚ) / total_balls
  (single_draw_white_probability ^ 6 = (1 : ℚ) / 64) :=
by
  sorry

end NUMINAMATH_GPT_probability_all_six_draws_white_l2190_219041


namespace NUMINAMATH_GPT_two_sin_cos_75_eq_half_l2190_219043

noncomputable def two_sin_cos_of_75_deg : ℝ :=
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)

theorem two_sin_cos_75_eq_half : two_sin_cos_of_75_deg = 1 / 2 :=
by
  -- The steps to prove this theorem are omitted deliberately
  sorry

end NUMINAMATH_GPT_two_sin_cos_75_eq_half_l2190_219043


namespace NUMINAMATH_GPT_average_reading_days_l2190_219030

theorem average_reading_days :
  let days_participated := [2, 3, 4, 5, 6]
  let students := [5, 4, 7, 3, 6]
  let total_days := List.zipWith (· * ·) days_participated students |>.sum
  let total_students := students.sum
  let average := total_days / total_students
  average = 4.04 := sorry

end NUMINAMATH_GPT_average_reading_days_l2190_219030


namespace NUMINAMATH_GPT_relationship_of_abc_l2190_219002

theorem relationship_of_abc (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 4) : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_relationship_of_abc_l2190_219002


namespace NUMINAMATH_GPT_ze_age_conditions_l2190_219054

theorem ze_age_conditions 
  (z g t : ℕ)
  (h1 : z = 2 * g + 3 * t)
  (h2 : 2 * (z + 15) = 2 * (g + 15) + 3 * (t + 15))
  (h3 : 2 * (g + 15) = 3 * (t + 15)) :
  z = 45 ∧ t = 5 :=
by
  sorry

end NUMINAMATH_GPT_ze_age_conditions_l2190_219054


namespace NUMINAMATH_GPT_number_of_pieces_of_paper_l2190_219052

def three_digit_number_with_unique_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n / 100 ≠ (n / 10) % 10 ∧ n / 100 ≠ n % 10 ∧ (n / 10) % 10 ≠ n % 10

theorem number_of_pieces_of_paper (n : ℕ) (k : ℕ) (h1 : three_digit_number_with_unique_digits n) (h2 : 2331 = k * n) : k = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pieces_of_paper_l2190_219052


namespace NUMINAMATH_GPT_speed_of_stream_l2190_219046

theorem speed_of_stream (x : ℝ) (boat_speed : ℝ) (distance_one_way : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : distance_one_way = 7560) 
  (h3 : total_time = 960) 
  (h4 : (distance_one_way / (boat_speed + x)) + (distance_one_way / (boat_speed - x)) = total_time) 
  : x = 2 := 
  sorry

end NUMINAMATH_GPT_speed_of_stream_l2190_219046


namespace NUMINAMATH_GPT_total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l2190_219093

theorem total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = 1001 ∧ 2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l2190_219093


namespace NUMINAMATH_GPT_car_highway_miles_per_tankful_l2190_219069

-- Condition definitions
def city_miles_per_tankful : ℕ := 336
def miles_per_gallon_city : ℕ := 24
def city_to_highway_diff : ℕ := 9

-- Calculation from conditions
def miles_per_gallon_highway : ℕ := miles_per_gallon_city + city_to_highway_diff
def tank_size : ℤ := city_miles_per_tankful / miles_per_gallon_city

-- Desired result
def highway_miles_per_tankful : ℤ := miles_per_gallon_highway * tank_size

-- Proof statement
theorem car_highway_miles_per_tankful :
  highway_miles_per_tankful = 462 := by
  unfold highway_miles_per_tankful
  unfold miles_per_gallon_highway
  unfold tank_size
  -- Sorry here to skip the detailed proof steps
  sorry

end NUMINAMATH_GPT_car_highway_miles_per_tankful_l2190_219069


namespace NUMINAMATH_GPT_problem_statement_l2190_219097

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 5) : 
  -a - m * c * d - b = -5 ∨ -a - m * c * d - b = 5 := 
  sorry

end NUMINAMATH_GPT_problem_statement_l2190_219097


namespace NUMINAMATH_GPT_fraction_female_attendees_on_time_l2190_219074

theorem fraction_female_attendees_on_time (A : ℝ) (h1 : A > 0) :
  let males_fraction := 3/5
  let males_on_time := 7/8
  let not_on_time := 0.155
  let total_on_time_fraction := 1 - not_on_time
  let males := males_fraction * A
  let males_arrived_on_time := males_on_time * males
  let females := (1 - males_fraction) * A
  let females_arrived_on_time_fraction := (total_on_time_fraction * A - males_arrived_on_time) / females
  females_arrived_on_time_fraction = 4/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_female_attendees_on_time_l2190_219074


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_inequality_l2190_219003

theorem problem1_solution_set (x : ℝ) : (-1 < x) ∧ (x < 9) ↔ (|x| + |x - 3| < x + 6) :=
by sorry

theorem problem2_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hn : 9 * x + y = 1) : x + y ≥ 16 * x * y :=
by sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_inequality_l2190_219003


namespace NUMINAMATH_GPT_product_of_digits_l2190_219032

-- Define the conditions and state the theorem
theorem product_of_digits (A B : ℕ) (h1 : (10 * A + B) % 12 = 0) (h2 : A + B = 12) : A * B = 32 :=
  sorry

end NUMINAMATH_GPT_product_of_digits_l2190_219032


namespace NUMINAMATH_GPT_value_of_pq_s_l2190_219022

-- Definitions for the problem
def polynomial_divisible (p q s : ℚ) : Prop :=
  ∀ x : ℚ, (x^3 + 4 * x^2 + 16 * x + 8) ∣ (x^4 + 6 * x^3 + 8 * p * x^2 + 6 * q * x + s)

-- The main theorem statement to prove
theorem value_of_pq_s (p q s : ℚ) (h : polynomial_divisible p q s) : (p + q) * s = 332 / 3 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_value_of_pq_s_l2190_219022


namespace NUMINAMATH_GPT_tan_of_negative_7pi_over_4_l2190_219001

theorem tan_of_negative_7pi_over_4 : Real.tan (-7 * Real.pi / 4) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_of_negative_7pi_over_4_l2190_219001


namespace NUMINAMATH_GPT_neither_odd_nor_even_and_min_value_at_one_l2190_219098

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem neither_odd_nor_even_and_min_value_at_one :
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ - f x) ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x :=
by
  sorry

end NUMINAMATH_GPT_neither_odd_nor_even_and_min_value_at_one_l2190_219098


namespace NUMINAMATH_GPT_michael_birth_year_l2190_219050

theorem michael_birth_year (first_AMC8_year : ℕ) (tenth_AMC8_year : ℕ) (age_during_tenth_AMC8 : ℕ) 
  (h1 : first_AMC8_year = 1985) (h2 : tenth_AMC8_year = (first_AMC8_year + 9)) (h3 : age_during_tenth_AMC8 = 15) :
  (tenth_AMC8_year - age_during_tenth_AMC8) = 1979 :=
by
  sorry

end NUMINAMATH_GPT_michael_birth_year_l2190_219050


namespace NUMINAMATH_GPT_total_food_pounds_l2190_219059

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end NUMINAMATH_GPT_total_food_pounds_l2190_219059


namespace NUMINAMATH_GPT_volume_of_cube_with_surface_area_l2190_219019

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end NUMINAMATH_GPT_volume_of_cube_with_surface_area_l2190_219019


namespace NUMINAMATH_GPT_thomas_total_blocks_l2190_219072

def stack1 := 7
def stack2 := stack1 + 3
def stack3 := stack2 - 6
def stack4 := stack3 + 10
def stack5 := stack2 * 2

theorem thomas_total_blocks : stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end NUMINAMATH_GPT_thomas_total_blocks_l2190_219072


namespace NUMINAMATH_GPT_problem_statement_l2190_219091

theorem problem_statement (x : ℤ) (y : ℝ) (h : y = 0.5) : 
  (⌈x + y⌉ - ⌊x + y⌋ = 1) ∧ (⌈x + y⌉ - (x + y) = 0.5) := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2190_219091


namespace NUMINAMATH_GPT_complement_intersection_l2190_219066

-- Definitions for the sets
def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2, 3}

-- Statement to be proved
theorem complement_intersection (hU : U = {0, 1, 2, 3}) (hA : A = {0, 1}) (hB : B = {1, 2, 3}) :
  ((U \ A) ∩ B) = {2, 3} :=
by
  -- Greek delta: skip proof details
  sorry

end NUMINAMATH_GPT_complement_intersection_l2190_219066


namespace NUMINAMATH_GPT_pair_B_equal_l2190_219037

theorem pair_B_equal : (∀ x : ℝ, 4 * x^4 = |x|) :=
by sorry

end NUMINAMATH_GPT_pair_B_equal_l2190_219037


namespace NUMINAMATH_GPT_min_det_is_neg_six_l2190_219008

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end NUMINAMATH_GPT_min_det_is_neg_six_l2190_219008


namespace NUMINAMATH_GPT_yoongi_caught_frogs_l2190_219012

theorem yoongi_caught_frogs (initial_frogs caught_later : ℕ) (h1 : initial_frogs = 5) (h2 : caught_later = 2) : (initial_frogs + caught_later = 7) :=
by
  sorry

end NUMINAMATH_GPT_yoongi_caught_frogs_l2190_219012


namespace NUMINAMATH_GPT_compare_abc_l2190_219085

noncomputable def a : ℝ := Real.exp (Real.sqrt Real.pi)
noncomputable def b : ℝ := Real.sqrt Real.pi + 1
noncomputable def c : ℝ := (Real.log Real.pi) / Real.exp 1 + 2

theorem compare_abc : c < b ∧ b < a := by
  sorry

end NUMINAMATH_GPT_compare_abc_l2190_219085


namespace NUMINAMATH_GPT_wall_height_l2190_219079

noncomputable def brick_volume : ℝ := 25 * 11.25 * 6

noncomputable def total_brick_volume : ℝ := brick_volume * 6400

noncomputable def wall_length : ℝ := 800

noncomputable def wall_width : ℝ := 600

theorem wall_height :
  ∀ (wall_volume : ℝ), 
  wall_volume = total_brick_volume → 
  wall_volume = wall_length * wall_width * 22.48 :=
by
  sorry

end NUMINAMATH_GPT_wall_height_l2190_219079


namespace NUMINAMATH_GPT_orange_juice_production_l2190_219084

theorem orange_juice_production :
  let total_oranges := 8 -- in million tons
  let exported_oranges := total_oranges * 0.25
  let remaining_oranges := total_oranges - exported_oranges
  let juice_oranges_ratio := 0.60
  let juice_oranges := remaining_oranges * juice_oranges_ratio
  juice_oranges = 3.6  :=
by
  sorry

end NUMINAMATH_GPT_orange_juice_production_l2190_219084


namespace NUMINAMATH_GPT_unit_fraction_decomposition_l2190_219011

theorem unit_fraction_decomposition (n : ℕ) (hn : 0 < n): 
  (1 : ℚ) / n = (1 : ℚ) / (2 * n) + (1 : ℚ) / (3 * n) + (1 : ℚ) / (6 * n) :=
by
  sorry

end NUMINAMATH_GPT_unit_fraction_decomposition_l2190_219011


namespace NUMINAMATH_GPT_solution_exists_l2190_219077

def divide_sum_of_squares_and_quotient_eq_seventy_two (x : ℝ) : Prop :=
  (10 - x)^2 + x^2 + (10 - x) / x = 72

theorem solution_exists (x : ℝ) : divide_sum_of_squares_and_quotient_eq_seventy_two x → x = 2 := sorry

end NUMINAMATH_GPT_solution_exists_l2190_219077


namespace NUMINAMATH_GPT_roger_bike_rides_total_l2190_219062

theorem roger_bike_rides_total 
  (r1 : ℕ) (h1 : r1 = 2) 
  (r2 : ℕ) (h2 : r2 = 5 * r1) 
  (r : ℕ) (h : r = r1 + r2) : 
  r = 12 := 
by
  sorry

end NUMINAMATH_GPT_roger_bike_rides_total_l2190_219062


namespace NUMINAMATH_GPT_find_triples_l2190_219007

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end NUMINAMATH_GPT_find_triples_l2190_219007


namespace NUMINAMATH_GPT_fenced_area_l2190_219000

theorem fenced_area (length_large : ℕ) (width_large : ℕ) 
                    (length_cutout : ℕ) (width_cutout : ℕ) 
                    (h_large : length_large = 20 ∧ width_large = 15)
                    (h_cutout : length_cutout = 4 ∧ width_cutout = 2) : 
                    ((length_large * width_large) - (length_cutout * width_cutout) = 292) := 
by
  sorry

end NUMINAMATH_GPT_fenced_area_l2190_219000


namespace NUMINAMATH_GPT_sum_two_numbers_l2190_219027

theorem sum_two_numbers (x y : ℝ) (h₁ : x * y = 16) (h₂ : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 :=
by
  -- Proof follows the steps outlined in the solution, but this is where the proof ends for now.
  sorry

end NUMINAMATH_GPT_sum_two_numbers_l2190_219027


namespace NUMINAMATH_GPT_smallest_possible_third_term_l2190_219016

theorem smallest_possible_third_term :
  ∃ (d : ℝ), (d = -3 + Real.sqrt 134 ∨ d = -3 - Real.sqrt 134) ∧ 
  (7, 7 + d + 3, 7 + 2 * d + 18) = (7, 10 + d, 25 + 2 * d) ∧ 
  min (25 + 2 * (-3 + Real.sqrt 134)) (25 + 2 * (-3 - Real.sqrt 134)) = 19 + 2 * Real.sqrt 134 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_third_term_l2190_219016


namespace NUMINAMATH_GPT_geun_bae_fourth_day_jumps_l2190_219065

-- Define a function for number of jump ropes Geun-bae does on each day
def jump_ropes (n : ℕ) : ℕ :=
  match n with
  | 0     => 15
  | n + 1 => 2 * jump_ropes n

-- Theorem stating the number of jump ropes Geun-bae does on the fourth day
theorem geun_bae_fourth_day_jumps : jump_ropes 3 = 120 := 
by {
  sorry
}

end NUMINAMATH_GPT_geun_bae_fourth_day_jumps_l2190_219065
