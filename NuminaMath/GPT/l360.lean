import Mathlib

namespace NUMINAMATH_GPT_range_of_m_l360_36065

theorem range_of_m (m : ℝ) :
  (∃ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 0 ∧
    (∃ ρ₀ θ₀ : ℝ, ∀ ρ θ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 
      m * ρ₀ * (Real.cos θ₀)^2 + 3 * ρ₀ * (Real.sin θ₀)^2 - 6 * (Real.cos θ₀))) →
  m > 0 ∧ m ≠ 3 := sorry

end NUMINAMATH_GPT_range_of_m_l360_36065


namespace NUMINAMATH_GPT_qualified_weight_example_l360_36029

-- Define the range of qualified weights
def is_qualified_weight (w : ℝ) : Prop :=
  9.9 ≤ w ∧ w ≤ 10.1

-- State the problem: show that 10 kg is within the qualified range
theorem qualified_weight_example : is_qualified_weight 10 :=
  by
    sorry

end NUMINAMATH_GPT_qualified_weight_example_l360_36029


namespace NUMINAMATH_GPT_find_total_income_l360_36089

theorem find_total_income (I : ℝ) (H : (0.27 * I = 35000)) : I = 129629.63 :=
by
  sorry

end NUMINAMATH_GPT_find_total_income_l360_36089


namespace NUMINAMATH_GPT_area_of_circle_l360_36064

noncomputable def point : Type := ℝ × ℝ

def A : point := (8, 15)
def B : point := (14, 9)

def is_on_circle (P : point) (r : ℝ) (C : point) : Prop :=
  (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = r ^ 2

def tangent_intersects_x_axis (tangent_point : point) (circle_center : point) : Prop :=
  ∃ x : ℝ, ∃ C : point, C.2 = 0 ∧ tangent_point = C ∧ circle_center = (x, 0)

theorem area_of_circle :
  ∃ C : point, ∃ r : ℝ,
    is_on_circle A r C ∧ 
    is_on_circle B r C ∧ 
    tangent_intersects_x_axis A C ∧ 
    tangent_intersects_x_axis B C ∧ 
    (↑(π * r ^ 2) = (117 * π) / 8) :=
sorry

end NUMINAMATH_GPT_area_of_circle_l360_36064


namespace NUMINAMATH_GPT_grades_calculation_l360_36010

-- Defining the conditions
def total_students : ℕ := 22800
def students_per_grade : ℕ := 75

-- Stating the theorem to be proved
theorem grades_calculation : total_students / students_per_grade = 304 := sorry

end NUMINAMATH_GPT_grades_calculation_l360_36010


namespace NUMINAMATH_GPT_person_a_age_l360_36019

theorem person_a_age (A B : ℕ) (h1 : A + B = 43) (h2 : A + 4 = B + 7) : A = 23 :=
by sorry

end NUMINAMATH_GPT_person_a_age_l360_36019


namespace NUMINAMATH_GPT_minimum_value_is_8_l360_36037

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_is_8_l360_36037


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l360_36061

theorem relationship_between_y1_y2 (y1 y2 : ℝ) :
    (y1 = -3 * 2 + 4 ∧ y2 = -3 * (-1) + 4) → y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_l360_36061


namespace NUMINAMATH_GPT_remainder_of_product_mod_7_l360_36085

theorem remainder_of_product_mod_7
  (a b c : ℕ)
  (ha : a ≡ 2 [MOD 7])
  (hb : b ≡ 3 [MOD 7])
  (hc : c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_7_l360_36085


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_q_implies_range_of_a_l360_36091

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_q_implies_range_of_a_l360_36091


namespace NUMINAMATH_GPT_maria_total_cost_l360_36092

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end NUMINAMATH_GPT_maria_total_cost_l360_36092


namespace NUMINAMATH_GPT_circle_intersection_l360_36098

theorem circle_intersection (a : ℝ) :
  ((-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2)) ↔
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_GPT_circle_intersection_l360_36098


namespace NUMINAMATH_GPT_range_of_a_add_b_l360_36007

-- Define the problem and assumptions
variables (a b : ℝ)
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom ab_eq_a_add_b_add_3 : a * b = a + b + 3

-- Define the theorem to prove
theorem range_of_a_add_b : a + b ≥ 6 :=
sorry

end NUMINAMATH_GPT_range_of_a_add_b_l360_36007


namespace NUMINAMATH_GPT_intersection_eq_singleton_zero_l360_36015

-- Definition of the sets M and N
def M : Set ℤ := {0, 1}
def N : Set ℤ := { x | ∃ n : ℤ, x = 2 * n }

-- The theorem stating that the intersection of M and N is {0}
theorem intersection_eq_singleton_zero : M ∩ N = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_singleton_zero_l360_36015


namespace NUMINAMATH_GPT_fraction_power_equality_l360_36060

theorem fraction_power_equality :
  (72000 ^ 4) / (24000 ^ 4) = 81 := 
by
  sorry

end NUMINAMATH_GPT_fraction_power_equality_l360_36060


namespace NUMINAMATH_GPT_tax_rate_calculation_l360_36073

theorem tax_rate_calculation (price_before_tax total_price : ℝ) 
  (h_price_before_tax : price_before_tax = 92) 
  (h_total_price : total_price = 98.90) : 
  (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := 
by 
  -- Proof will be provided here.
  sorry

end NUMINAMATH_GPT_tax_rate_calculation_l360_36073


namespace NUMINAMATH_GPT_intersection_points_form_line_slope_l360_36046

theorem intersection_points_form_line_slope (s : ℝ) :
  ∃ (m : ℝ), m = 1/18 ∧ ∀ (x y : ℝ),
    (3 * x + y = 5 * s + 6) ∧ (2 * x - 3 * y = 3 * s - 5) →
    ∃ k : ℝ, (y = m * x + k) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_form_line_slope_l360_36046


namespace NUMINAMATH_GPT_molecular_weight_bleach_l360_36079

theorem molecular_weight_bleach :
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  molecular_weight = 74.44
:=
by
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  sorry

end NUMINAMATH_GPT_molecular_weight_bleach_l360_36079


namespace NUMINAMATH_GPT_max_blue_points_l360_36035

theorem max_blue_points (n : ℕ) (r b : ℕ)
  (h1 : n = 2009)
  (h2 : b + r = n)
  (h3 : ∀(k : ℕ), b ≤ k * (k - 1) / 2 → r ≥ k) :
  b = 1964 :=
by
  sorry

end NUMINAMATH_GPT_max_blue_points_l360_36035


namespace NUMINAMATH_GPT_mila_total_distance_l360_36011

/-- Mila's car consumes a gallon of gas every 40 miles, her full gas tank holds 16 gallons, starting with a full tank, she drove 400 miles, then refueled with 10 gallons, 
and upon arriving at her destination her gas tank was a third full.
Prove that the total distance Mila drove that day is 826 miles. -/
theorem mila_total_distance (consumption_per_mile : ℝ) (tank_capacity : ℝ) (initial_drive : ℝ) (refuel_amount : ℝ) (final_fraction : ℝ)
  (consumption_per_mile_def : consumption_per_mile = 1 / 40)
  (tank_capacity_def : tank_capacity = 16)
  (initial_drive_def : initial_drive = 400)
  (refuel_amount_def : refuel_amount = 10)
  (final_fraction_def : final_fraction = 1 / 3) :
  ∃ total_distance : ℝ, total_distance = 826 :=
by
  sorry

end NUMINAMATH_GPT_mila_total_distance_l360_36011


namespace NUMINAMATH_GPT_order_of_6_with_respect_to_f_is_undefined_l360_36095

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end NUMINAMATH_GPT_order_of_6_with_respect_to_f_is_undefined_l360_36095


namespace NUMINAMATH_GPT_pastries_more_than_cakes_l360_36005

def cakes_made : ℕ := 19
def pastries_made : ℕ := 131

theorem pastries_more_than_cakes : pastries_made - cakes_made = 112 :=
by {
  -- Proof will be inserted here
  sorry
}

end NUMINAMATH_GPT_pastries_more_than_cakes_l360_36005


namespace NUMINAMATH_GPT_total_money_l360_36034

variable (A B C: ℕ)
variable (h1: A + C = 200) 
variable (h2: B + C = 350)
variable (h3: C = 200)

theorem total_money : A + B + C = 350 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l360_36034


namespace NUMINAMATH_GPT_percentage_increase_of_numerator_l360_36051

theorem percentage_increase_of_numerator (N D : ℝ) (P : ℝ) (h1 : N / D = 0.75)
  (h2 : (N + (P / 100) * N) / (D - (8 / 100) * D) = 15 / 16) :
  P = 15 :=
sorry

end NUMINAMATH_GPT_percentage_increase_of_numerator_l360_36051


namespace NUMINAMATH_GPT_value_of_v_3_l360_36043

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end NUMINAMATH_GPT_value_of_v_3_l360_36043


namespace NUMINAMATH_GPT_find_prime_b_l360_36075

-- Define the polynomial function f
def f (n a : ℕ) : ℕ := n^3 - 4 * a * n^2 - 12 * n + 144

-- Define b as a prime number
def b (n : ℕ) (a : ℕ) : ℕ := f n a

-- Theorem statement
theorem find_prime_b (n : ℕ) (a : ℕ) (h : n = 7) (ha : a = 2) (hb : ∃ p : ℕ, Nat.Prime p ∧ p = b n a) :
  b n a = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_prime_b_l360_36075


namespace NUMINAMATH_GPT_q_r_share_difference_l360_36049

theorem q_r_share_difference
  (T : ℝ) -- Total amount of money
  (x : ℝ) -- Common multiple of shares
  (p_share q_share r_share s_share : ℝ) -- Shares before tax
  (p_tax q_tax r_tax s_tax : ℝ) -- Tax percentages
  (h_ratio : p_share = 3 * x ∧ q_share = 7 * x ∧ r_share = 12 * x ∧ s_share = 5 * x) -- Ratio condition
  (h_tax : p_tax = 0.10 ∧ q_tax = 0.15 ∧ r_tax = 0.20 ∧ s_tax = 0.25) -- Tax condition
  (h_difference_pq : q_share * (1 - q_tax) - p_share * (1 - p_tax) = 2400) -- Difference between p and q after tax
  : (r_share * (1 - r_tax) - q_share * (1 - q_tax)) = 2695.38 := sorry

end NUMINAMATH_GPT_q_r_share_difference_l360_36049


namespace NUMINAMATH_GPT_ratio_cost_price_selling_price_l360_36027

theorem ratio_cost_price_selling_price (CP SP : ℝ) (h : SP = 1.5 * CP) : CP / SP = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_cost_price_selling_price_l360_36027


namespace NUMINAMATH_GPT_james_total_pay_l360_36050

def original_prices : List ℝ := [15, 20, 25, 18, 22, 30]
def discounts : List ℝ := [0.30, 0.50, 0.40, 0.20, 0.45, 0.25]

def discounted_price (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price_after_discount (prices discounts : List ℝ) : ℝ :=
  (List.zipWith discounted_price prices discounts).sum

theorem james_total_pay :
  total_price_after_discount original_prices discounts = 84.50 :=
  by sorry

end NUMINAMATH_GPT_james_total_pay_l360_36050


namespace NUMINAMATH_GPT_angle_D_is_90_l360_36033

theorem angle_D_is_90 (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : B = 130) (h5 : C + D = 180) :
  D = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_D_is_90_l360_36033


namespace NUMINAMATH_GPT_division_result_l360_36083

theorem division_result : (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3)) = 124 / 509 := 
by
  sorry

end NUMINAMATH_GPT_division_result_l360_36083


namespace NUMINAMATH_GPT_solve_for_a_l360_36009

variable (a u : ℝ)

def eq1 := (3 / a) + (1 / u) = 7 / 2
def eq2 := (2 / a) - (3 / u) = 6

theorem solve_for_a (h1 : eq1 a u) (h2 : eq2 a u) : a = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l360_36009


namespace NUMINAMATH_GPT_inequality_ab_bc_ca_l360_36074

open Real

theorem inequality_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a) / (2 * (a + b + c))) := by
sorry

end NUMINAMATH_GPT_inequality_ab_bc_ca_l360_36074


namespace NUMINAMATH_GPT_number_of_students_not_enrolled_in_biology_l360_36093

noncomputable def total_students : ℕ := 880

noncomputable def biology_enrollment_percent : ℕ := 40

noncomputable def students_not_enrolled_in_biology : ℕ :=
  (100 - biology_enrollment_percent) * total_students / 100

theorem number_of_students_not_enrolled_in_biology :
  students_not_enrolled_in_biology = 528 :=
by
  -- Proof goes here.
  -- Use sorry to skip the proof for this placeholder:
  sorry

end NUMINAMATH_GPT_number_of_students_not_enrolled_in_biology_l360_36093


namespace NUMINAMATH_GPT_probability_closer_to_6_than_0_is_0_6_l360_36096

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end NUMINAMATH_GPT_probability_closer_to_6_than_0_is_0_6_l360_36096


namespace NUMINAMATH_GPT_correct_rounded_result_l360_36068

-- Definition of rounding to the nearest hundred
def rounded_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 < 50 then n / 100 * 100 else (n / 100 + 1) * 100

-- Given conditions
def sum : ℕ := 68 + 57

-- The theorem to prove
theorem correct_rounded_result : rounded_to_nearest_hundred sum = 100 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_correct_rounded_result_l360_36068


namespace NUMINAMATH_GPT_find_m_l360_36048

-- Define the condition for m to be within the specified range
def valid_range (m : ℤ) : Prop := -180 < m ∧ m < 180

-- Define the relationship with the trigonometric equation to be proven
def tan_eq (m : ℤ) : Prop := Real.tan (m * Real.pi / 180) = Real.tan (1500 * Real.pi / 180)

-- State the main theorem to be proved
theorem find_m (m : ℤ) (h1 : valid_range m) (h2 : tan_eq m) : m = 60 :=
sorry

end NUMINAMATH_GPT_find_m_l360_36048


namespace NUMINAMATH_GPT_maura_classroom_students_l360_36016

theorem maura_classroom_students (T : ℝ) (h1 : Tina_students = T) (h2 : Maura_students = T) (h3 : Zack_students = T / 2) (h4 : Tina_students + Maura_students + Zack_students = 69) : T = 138 / 5 := by
  sorry

end NUMINAMATH_GPT_maura_classroom_students_l360_36016


namespace NUMINAMATH_GPT_extreme_values_l360_36062

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem extreme_values (x : ℝ) (hx : x ≠ 0) :
  (x = -2 → f x = -4 ∧ ∀ y, y > -2 → f y > -4) ∧
  (x = 2 → f x = 4 ∧ ∀ y, y < 2 → f y > 4) :=
sorry

end NUMINAMATH_GPT_extreme_values_l360_36062


namespace NUMINAMATH_GPT_num_triangles_from_decagon_l360_36040

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end NUMINAMATH_GPT_num_triangles_from_decagon_l360_36040


namespace NUMINAMATH_GPT_determine_abcd_l360_36038

-- Define a 4-digit natural number abcd in terms of its digits a, b, c, d
def four_digit_number (abcd a b c d : ℕ) :=
  abcd = 1000 * a + 100 * b + 10 * c + d

-- Define the condition given in the problem
def satisfies_condition (abcd a b c d : ℕ) :=
  abcd - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

-- Define the main theorem statement proving the number is 2243
theorem determine_abcd : ∃ (a b c d abcd : ℕ), four_digit_number abcd a b c d ∧ satisfies_condition abcd a b c d ∧ abcd = 2243 :=
by
  sorry

end NUMINAMATH_GPT_determine_abcd_l360_36038


namespace NUMINAMATH_GPT_manny_problem_l360_36053

noncomputable def num_slices_left (num_pies : Nat) (slices_per_pie : Nat) (num_classmates : Nat) (num_teachers : Nat) (num_slices_per_person : Nat) : Nat :=
  let total_slices := num_pies * slices_per_pie
  let total_people := 1 + num_classmates + num_teachers
  let slices_taken := total_people * num_slices_per_person
  total_slices - slices_taken

theorem manny_problem : num_slices_left 3 10 24 1 1 = 4 := by
  sorry

end NUMINAMATH_GPT_manny_problem_l360_36053


namespace NUMINAMATH_GPT_cost_of_bananas_l360_36023

theorem cost_of_bananas
  (apple_cost : ℕ)
  (orange_cost : ℕ)
  (banana_cost : ℕ)
  (num_apples : ℕ)
  (num_oranges : ℕ)
  (num_bananas : ℕ)
  (total_paid : ℕ) 
  (discount_threshold : ℕ)
  (discount_amount : ℕ)
  (total_fruits : ℕ)
  (total_without_discount : ℕ) :
  apple_cost = 1 → 
  orange_cost = 2 → 
  num_apples = 5 → 
  num_oranges = 3 → 
  num_bananas = 2 → 
  total_paid = 15 → 
  discount_threshold = 5 → 
  discount_amount = 1 → 
  total_fruits = num_apples + num_oranges + num_bananas →
  total_without_discount = (num_apples * apple_cost) + (num_oranges * orange_cost) + (num_bananas * banana_cost) →
  (total_without_discount - (discount_amount * (total_fruits / discount_threshold))) = total_paid →
  banana_cost = 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_cost_of_bananas_l360_36023


namespace NUMINAMATH_GPT_farm_problem_l360_36042

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end NUMINAMATH_GPT_farm_problem_l360_36042


namespace NUMINAMATH_GPT_trapezoid_diagonals_l360_36077

theorem trapezoid_diagonals {BC AD AB CD AC BD : ℝ} (h b1 b2 : ℝ) 
  (hBC : BC = b1) (hAD : AD = b2) (hAB : AB = h) (hCD : CD = h) 
  (hAC : AC^2 = AB^2 + BC^2) (hBD : BD^2 = CD^2 + AD^2) :
  BD^2 - AC^2 = b2^2 - b1^2 := 
by 
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_trapezoid_diagonals_l360_36077


namespace NUMINAMATH_GPT_evaluate_expression_l360_36045

theorem evaluate_expression :
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l360_36045


namespace NUMINAMATH_GPT_candy_cost_l360_36076

theorem candy_cost (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) 
  (total_tickets : ℕ) (candies : ℕ) (cost_per_candy : ℕ) 
  (h1 : tickets_whack_a_mole = 8) (h2 : tickets_skee_ball = 7)
  (h3 : total_tickets = tickets_whack_a_mole + tickets_skee_ball)
  (h4 : candies = 3) (h5 : total_tickets = candies * cost_per_candy) :
  cost_per_candy = 5 :=
by
  sorry

end NUMINAMATH_GPT_candy_cost_l360_36076


namespace NUMINAMATH_GPT_cranberries_left_in_bog_l360_36008

theorem cranberries_left_in_bog
  (total_cranberries : ℕ)
  (percentage_harvested_by_humans : ℕ)
  (cranberries_eaten_by_elk : ℕ)
  (initial_count : total_cranberries = 60000)
  (harvested_percentage : percentage_harvested_by_humans = 40)
  (eaten_by_elk : cranberries_eaten_by_elk = 20000) :
  total_cranberries - (total_cranberries * percentage_harvested_by_humans / 100) - cranberries_eaten_by_elk = 16000 :=
by
  sorry

end NUMINAMATH_GPT_cranberries_left_in_bog_l360_36008


namespace NUMINAMATH_GPT_expected_number_of_2s_when_three_dice_rolled_l360_36021

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_expected_number_of_2s_when_three_dice_rolled_l360_36021


namespace NUMINAMATH_GPT_age_difference_l360_36057

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_difference_l360_36057


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_is_7_l360_36059

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_k_squared_plus_2_k_is_7 : (k^2 + 2^k) % 10 = 7 :=
by sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_is_7_l360_36059


namespace NUMINAMATH_GPT_solution_set_real_implies_conditions_l360_36004

variable {a b c : ℝ}

theorem solution_set_real_implies_conditions (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c < 0) : a < 0 ∧ (b^2 - 4 * a * c) < 0 := 
sorry

end NUMINAMATH_GPT_solution_set_real_implies_conditions_l360_36004


namespace NUMINAMATH_GPT_percentage_value_l360_36070

theorem percentage_value (M : ℝ) (h : (25 / 100) * M = (55 / 100) * 1500) : M = 3300 :=
by
  sorry

end NUMINAMATH_GPT_percentage_value_l360_36070


namespace NUMINAMATH_GPT_f_divisible_by_13_l360_36097

def f : ℕ → ℤ := sorry

theorem f_divisible_by_13 :
  (f 0 = 0) ∧ (f 1 = 0) ∧
  (∀ n, f (n + 2) = 4 ^ (n + 2) * f (n + 1) - 16 ^ (n + 1) * f n + n * 2 ^ (n ^ 2)) →
  (f 1989 % 13 = 0) ∧ (f 1990 % 13 = 0) ∧ (f 1991 % 13 = 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_f_divisible_by_13_l360_36097


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_eq_4_l360_36006

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: a_n is an arithmetic sequence, so a_(n+1) = a_n + d
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: a_2 = 2
def a_2_eq_2 (a : ℕ → ℝ) : Prop :=
  a 2 = 2

-- Condition: S_4 = 9, where S_n is the sum of first n terms of the sequence
def sum_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

def S_4_eq_9 (S : ℕ → ℝ) : Prop :=
  S 4 = 9

-- Proof: a_6 = 4
theorem arithmetic_sequence_a6_eq_4 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a_2_eq_2 a)
  (h3 : sum_S_n a S) 
  (h4 : S_4_eq_9 S) :
  a 6 = 4 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_eq_4_l360_36006


namespace NUMINAMATH_GPT_DanielCandies_l360_36081

noncomputable def initialCandies (x : ℝ) : Prop :=
  (3 / 8) * x - (3 / 2) - 16 = 10

theorem DanielCandies : ∃ x : ℝ, initialCandies x ∧ x = 93 :=
by
  use 93
  simp [initialCandies]
  norm_num
  sorry

end NUMINAMATH_GPT_DanielCandies_l360_36081


namespace NUMINAMATH_GPT_servant_cash_received_l360_36071

theorem servant_cash_received (annual_cash : ℕ) (turban_price : ℕ) (served_months : ℕ) (total_months : ℕ) (cash_received : ℕ) :
  annual_cash = 90 → turban_price = 50 → served_months = 9 → total_months = 12 → 
  cash_received = (annual_cash + turban_price) * served_months / total_months - turban_price → 
  cash_received = 55 :=
by {
  intros;
  sorry
}

end NUMINAMATH_GPT_servant_cash_received_l360_36071


namespace NUMINAMATH_GPT_line_through_intersection_points_of_circles_l360_36039

theorem line_through_intersection_points_of_circles :
  (∀ x y : ℝ, x^2 + y^2 = 9 ∧ (x + 4)^2 + (y + 3)^2 = 8 → 4 * x + 3 * y + 13 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_intersection_points_of_circles_l360_36039


namespace NUMINAMATH_GPT_inverse_function_l360_36063

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

noncomputable def f_inv (y : ℝ) : ℝ := -2 - Real.sqrt ((1 + y) / 2)

theorem inverse_function :
  ∀ (x : ℝ), x < -2 → f_inv (f x) = x ∧ ∀ (y : ℝ), y > -1 → f (f_inv y) = y :=
by
  sorry

end NUMINAMATH_GPT_inverse_function_l360_36063


namespace NUMINAMATH_GPT_find_x_l360_36018

theorem find_x : 
  (∃ x : ℝ, 
    2.5 * ((3.6 * 0.48 * 2.5) / (0.12 * x * 0.5)) = 2000.0000000000002) → 
  x = 0.225 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l360_36018


namespace NUMINAMATH_GPT_sum_of_fourth_powers_of_solutions_l360_36084

theorem sum_of_fourth_powers_of_solutions (x y : ℝ)
  (h : |x^2 - 2 * x + 1/1004| = 1/1004 ∨ |y^2 - 2 * y + 1/1004| = 1/1004) :
  x^4 + y^4 = 20160427280144 / 12600263001 :=
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_of_solutions_l360_36084


namespace NUMINAMATH_GPT_bags_bought_l360_36054

theorem bags_bought (initial_bags : ℕ) (bags_given : ℕ) (final_bags : ℕ) (bags_bought : ℕ) :
  initial_bags = 20 → 
  bags_given = 4 → 
  final_bags = 22 → 
  bags_bought = final_bags - (initial_bags - bags_given) → 
  bags_bought = 6 := 
by
  intros h_initial h_given h_final h_buy
  rw [h_initial, h_given, h_final] at h_buy
  exact h_buy

#check bags_bought

end NUMINAMATH_GPT_bags_bought_l360_36054


namespace NUMINAMATH_GPT_negation_universal_proposition_l360_36058

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) → ∃ x : ℝ, x^2 - 2 * x + 1 < 0 :=
by sorry

end NUMINAMATH_GPT_negation_universal_proposition_l360_36058


namespace NUMINAMATH_GPT_power_mod_l360_36082

theorem power_mod (n : ℕ) : (3 ^ 2017) % 17 = 3 := 
by
  sorry

end NUMINAMATH_GPT_power_mod_l360_36082


namespace NUMINAMATH_GPT_smallest_positive_integer_l360_36031

theorem smallest_positive_integer (n : ℕ) : 13 * n ≡ 567 [MOD 5] ↔ n = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l360_36031


namespace NUMINAMATH_GPT_weights_system_l360_36099

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end NUMINAMATH_GPT_weights_system_l360_36099


namespace NUMINAMATH_GPT_correct_statement_C_l360_36055

theorem correct_statement_C
  (a : ℚ) : a < 0 → |a| = -a := 
by
  sorry

end NUMINAMATH_GPT_correct_statement_C_l360_36055


namespace NUMINAMATH_GPT_largest_n_crates_same_orange_count_l360_36066

theorem largest_n_crates_same_orange_count :
  ∀ (num_crates : ℕ) (min_oranges max_oranges : ℕ),
    num_crates = 200 →
    min_oranges = 100 →
    max_oranges = 130 →
    (∃ (n : ℕ), n = 7 ∧ (∃ (distribution : ℕ → ℕ), 
      (∀ x, min_oranges ≤ x ∧ x ≤ max_oranges) ∧ 
      (∀ x, distribution x ≤ num_crates ∧ 
          ∃ y, distribution y ≥ n))) := sorry

end NUMINAMATH_GPT_largest_n_crates_same_orange_count_l360_36066


namespace NUMINAMATH_GPT_sixth_bar_placement_l360_36025

theorem sixth_bar_placement (f : ℕ → ℕ) (h1 : f 1 = 1) (h2 : f 2 = 121) :
  (∃ n, f 6 = n ∧ (n = 16 ∨ n = 46 ∨ n = 76 ∨ n = 106)) :=
sorry

end NUMINAMATH_GPT_sixth_bar_placement_l360_36025


namespace NUMINAMATH_GPT_race_speeds_l360_36012

theorem race_speeds (x y : ℕ) 
  (h1 : 5 * x + 10 = 5 * y) 
  (h2 : 6 * x = 4 * y) :
  x = 4 ∧ y = 6 :=
by {
  -- Proof will go here, but for now we skip it.
  sorry
}

end NUMINAMATH_GPT_race_speeds_l360_36012


namespace NUMINAMATH_GPT_cone_from_sector_l360_36036

theorem cone_from_sector
  (r : ℝ) (slant_height : ℝ)
  (radius_circle : ℝ := 10)
  (angle_sector : ℝ := 252) :
  (r = 7 ∧ slant_height = 10) :=
by
  sorry

end NUMINAMATH_GPT_cone_from_sector_l360_36036


namespace NUMINAMATH_GPT_part1_part2_l360_36041

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 {a : ℝ} :
  (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

theorem part2 :
  ∀ x > 0, Real.log x > (1 / Real.exp x) - (2 / (Real.exp 1) * x) :=
sorry

end NUMINAMATH_GPT_part1_part2_l360_36041


namespace NUMINAMATH_GPT_part1_part2_l360_36052

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^4 - 4 * x^3 + (3 + m) * x^2 - 12 * x + 12

theorem part1 (m : ℤ) : 
  (∀ x : ℝ, f x m - f (1 - x) m + 4 * x^3 = 0) ↔ (m = 8 ∨ m = 12) := 
sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x m ≥ 0) ↔ (4 ≤ m) := 
sorry

end NUMINAMATH_GPT_part1_part2_l360_36052


namespace NUMINAMATH_GPT_rationalize_denominator_sum_equals_49_l360_36028

open Real

noncomputable def A : ℚ := -1
noncomputable def B : ℚ := -3
noncomputable def C : ℚ := 1
noncomputable def D : ℚ := 2
noncomputable def E : ℚ := 33
noncomputable def F : ℚ := 17

theorem rationalize_denominator_sum_equals_49 :
  let expr := (A * sqrt 3 + B * sqrt 5 + C * sqrt 11 + D * sqrt E) / F
  49 = A + B + C + D + E + F :=
by {
  -- The proof will go here.
  exact sorry
}

end NUMINAMATH_GPT_rationalize_denominator_sum_equals_49_l360_36028


namespace NUMINAMATH_GPT_quadratic_roots_identity_l360_36017

noncomputable def a := - (2 / 5 : ℝ)
noncomputable def b := (1 / 5 : ℝ)
noncomputable def quadraticRoots := (a, b)

theorem quadratic_roots_identity :
  a + b ^ 2 = - (9 / 25 : ℝ) := 
by 
  rw [a, b]
  sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l360_36017


namespace NUMINAMATH_GPT_jars_needed_l360_36072

def hives : ℕ := 5
def honey_per_hive : ℕ := 20
def jar_capacity : ℝ := 0.5
def friend_ratio : ℝ := 0.5

theorem jars_needed : (hives * honey_per_hive) / 2 / jar_capacity = 100 := 
by sorry

end NUMINAMATH_GPT_jars_needed_l360_36072


namespace NUMINAMATH_GPT_c_share_of_profit_l360_36032

theorem c_share_of_profit 
  (x : ℝ) -- The amount invested by B
  (total_profit : ℝ := 11000) -- Total profit
  (A_invest : ℝ := 3 * x) -- A's investment
  (C_invest : ℝ := (3/2) * A_invest) -- C's investment
  (total_invest : ℝ := A_invest + x + C_invest) -- Total investment
  (C_share : ℝ := C_invest / total_invest * total_profit) -- C's share of the profit
  : C_share = 99000 / 17 := 
  by sorry

end NUMINAMATH_GPT_c_share_of_profit_l360_36032


namespace NUMINAMATH_GPT_trip_duration_60_mph_l360_36013

noncomputable def time_at_new_speed (initial_time : ℚ) (initial_speed : ℚ) (new_speed : ℚ) : ℚ :=
  initial_time * (initial_speed / new_speed)

theorem trip_duration_60_mph :
  time_at_new_speed (9 / 2) 70 60 = 5.25 := 
by
  sorry

end NUMINAMATH_GPT_trip_duration_60_mph_l360_36013


namespace NUMINAMATH_GPT_joans_remaining_kittens_l360_36090

theorem joans_remaining_kittens (initial_kittens given_away : ℕ) (h1 : initial_kittens = 15) (h2 : given_away = 7) : initial_kittens - given_away = 8 := sorry

end NUMINAMATH_GPT_joans_remaining_kittens_l360_36090


namespace NUMINAMATH_GPT_A_beats_B_by_14_meters_l360_36078

theorem A_beats_B_by_14_meters :
  let distance := 70
  let time_A := 20
  let time_B := 25
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let distance_B_in_A_time := speed_B * time_A
  (distance - distance_B_in_A_time) = 14 :=
by
  sorry

end NUMINAMATH_GPT_A_beats_B_by_14_meters_l360_36078


namespace NUMINAMATH_GPT_rectangle_area_l360_36024

noncomputable def area_of_rectangle (radius : ℝ) (ab ad : ℝ) : ℝ :=
  ab * ad

theorem rectangle_area (radius : ℝ) (ad : ℝ) (ab : ℝ) 
  (h_radius : radius = Real.sqrt 5)
  (h_ab_ad_relation : ab = 4 * ad) : 
  area_of_rectangle radius ab ad = 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l360_36024


namespace NUMINAMATH_GPT_matrix_count_l360_36002

-- A definition for the type of 3x3 matrices with 1's on the diagonal and * can be 0 or 1
def valid_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 = 1 ∧ 
  m 1 1 = 1 ∧ 
  m 2 2 = 1 ∧ 
  (m 0 1 = 0 ∨ m 0 1 = 1) ∧
  (m 0 2 = 0 ∨ m 0 2 = 1) ∧
  (m 1 0 = 0 ∨ m 1 0 = 1) ∧
  (m 1 2 = 0 ∨ m 1 2 = 1) ∧
  (m 2 0 = 0 ∨ m 2 0 = 1) ∧
  (m 2 1 = 0 ∨ m 2 1 = 1)

-- A definition to check that rows are distinct
def distinct_rows (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 ≠ m 1 ∧ m 1 ≠ m 2 ∧ m 0 ≠ m 2

-- Complete proof problem statement
theorem matrix_count : ∃ (n : ℕ), 
  (∀ m : Matrix (Fin 3) (Fin 3) ℕ, valid_matrix m → distinct_rows m) ∧ 
  n = 45 :=
by
  sorry

end NUMINAMATH_GPT_matrix_count_l360_36002


namespace NUMINAMATH_GPT_minimum_value_l360_36047

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l360_36047


namespace NUMINAMATH_GPT_carolyn_marbles_l360_36094

theorem carolyn_marbles (initial_marbles : ℕ) (shared_items : ℕ) (end_marbles: ℕ) : 
  initial_marbles = 47 → shared_items = 42 → end_marbles = initial_marbles - shared_items → end_marbles = 5 :=
by
  intros h₀ h₁ h₂
  rw [h₀, h₁] at h₂
  exact h₂

end NUMINAMATH_GPT_carolyn_marbles_l360_36094


namespace NUMINAMATH_GPT_probability_of_drawing_two_black_two_white_l360_36069

noncomputable def probability_two_black_two_white : ℚ :=
  let total_ways := (Nat.choose 18 4)
  let ways_black := (Nat.choose 10 2)
  let ways_white := (Nat.choose 8 2)
  let favorable_ways := ways_black * ways_white
  favorable_ways / total_ways

theorem probability_of_drawing_two_black_two_white :
  probability_two_black_two_white = 7 / 17 := sorry

end NUMINAMATH_GPT_probability_of_drawing_two_black_two_white_l360_36069


namespace NUMINAMATH_GPT_man_speed_is_correct_l360_36080

noncomputable def train_length : ℝ := 165
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def time_seconds : ℝ := 9

-- Function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph: ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Function to convert speed from m/s to kmph
noncomputable def mps_to_kmph (speed_mps: ℝ) : ℝ :=
  speed_mps * 3600 / 1000

-- The speed of the train in m/s
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- The relative speed of the train with respect to the man in m/s
noncomputable def relative_speed_mps : ℝ := train_length / time_seconds

-- The speed of the man in m/s
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps

-- The speed of the man in kmph
noncomputable def man_speed_kmph : ℝ := mps_to_kmph man_speed_mps

-- The statement to be proved
theorem man_speed_is_correct : man_speed_kmph = 5.976 := 
sorry

end NUMINAMATH_GPT_man_speed_is_correct_l360_36080


namespace NUMINAMATH_GPT_general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l360_36087

-- Part 1: Finding the general term of the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℤ) (h1 : a 1 = 25) (h4 : a 4 = 16) :
  ∃ d : ℤ, a n = 28 - 3 * n := 
sorry

-- Part 2: Finding the value of n that maximizes the sum of the first n terms
theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 1 = 25)
  (h4 : a 4 = 16) 
  (ha : ∀ n, a n = 28 - 3 * n) -- Using the result from part 1
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n : ℕ, S n < S (n + 1)) →
  9 = 9 :=
sorry

end NUMINAMATH_GPT_general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l360_36087


namespace NUMINAMATH_GPT_number_of_rabbits_l360_36020

-- Given conditions
variable (r c : ℕ)
variable (cond1 : r + c = 51)
variable (cond2 : 4 * r = 3 * (2 * c) + 4)

-- To prove
theorem number_of_rabbits : r = 31 :=
sorry

end NUMINAMATH_GPT_number_of_rabbits_l360_36020


namespace NUMINAMATH_GPT_industrial_lubricants_percentage_l360_36088

theorem industrial_lubricants_percentage :
  let a := 12   -- percentage for microphotonics
  let b := 24   -- percentage for home electronics
  let c := 15   -- percentage for food additives
  let d := 29   -- percentage for genetically modified microorganisms
  let angle_basic_astrophysics := 43.2 -- degrees for basic astrophysics
  let total_angle := 360              -- total degrees in a circle
  let total_budget := 100             -- total budget in percentage
  let e := (angle_basic_astrophysics / total_angle) * total_budget -- percentage for basic astrophysics
  a + b + c + d + e = 92 → total_budget - (a + b + c + d + e) = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_industrial_lubricants_percentage_l360_36088


namespace NUMINAMATH_GPT_ratio_flow_chart_to_total_time_l360_36014

noncomputable def T := 48
noncomputable def D := 18
noncomputable def C := (3 / 8) * T
noncomputable def F := T - C - D

theorem ratio_flow_chart_to_total_time : (F / T) = (1 / 4) := by
  sorry

end NUMINAMATH_GPT_ratio_flow_chart_to_total_time_l360_36014


namespace NUMINAMATH_GPT_car_cost_l360_36003

theorem car_cost (days_in_week : ℕ) (sue_days : ℕ) (sister_days : ℕ) 
  (sue_payment : ℕ) (car_cost : ℕ) 
  (h1 : days_in_week = 7)
  (h2 : sue_days = days_in_week - sister_days)
  (h3 : sister_days = 4)
  (h4 : sue_payment = 900)
  (h5 : sue_payment * days_in_week = sue_days * car_cost) :
  car_cost = 2100 := 
by {
  sorry
}

end NUMINAMATH_GPT_car_cost_l360_36003


namespace NUMINAMATH_GPT_number_of_elements_in_S_l360_36056

def S : Set ℕ := { n : ℕ | ∃ k : ℕ, n > 1 ∧ (10^10 - 1) % n = 0 }

theorem number_of_elements_in_S (h1 : Nat.Prime 9091) :
  ∃ T : Finset ℕ, T.card = 127 ∧ ∀ n, n ∈ T ↔ n ∈ S :=
sorry

end NUMINAMATH_GPT_number_of_elements_in_S_l360_36056


namespace NUMINAMATH_GPT_sum_of_first_nine_terms_l360_36067

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 0 + a (n - 1))

theorem sum_of_first_nine_terms 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms a S)
  (h_sum_terms : a 2 + a 3 + a 4 + a 5 + a 6 = 20) :
  S 9 = 36 :=
sorry

end NUMINAMATH_GPT_sum_of_first_nine_terms_l360_36067


namespace NUMINAMATH_GPT_find_original_price_of_dish_l360_36026

noncomputable def original_price_of_dish (P : ℝ) : Prop :=
  let john_paid := (0.9 * P) + (0.15 * P)
  let jane_paid := (0.9 * P) + (0.135 * P)
  john_paid = jane_paid + 0.60 → P = 40

theorem find_original_price_of_dish (P : ℝ) (h : original_price_of_dish P) : P = 40 := by
  sorry

end NUMINAMATH_GPT_find_original_price_of_dish_l360_36026


namespace NUMINAMATH_GPT_job_completion_time_l360_36030

theorem job_completion_time (h1 : ∀ {a d : ℝ}, 4 * (1/a + 1/d) = 1)
                             (h2 : ∀ d : ℝ, d = 11.999999999999998) :
                             (∀ a : ℝ, a = 6) :=
by
  sorry

end NUMINAMATH_GPT_job_completion_time_l360_36030


namespace NUMINAMATH_GPT_range_g_l360_36001

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

noncomputable def g (x : ℝ) : ℝ := 
  let h (x : ℝ) := (Real.sin (2 * x + Real.pi))
  h (x - (5 * Real.pi / 12))

theorem range_g :
  (Set.image g (Set.Icc (-Real.pi/12) (Real.pi/3))) = Set.Icc (-1) (1/2) :=
  sorry

end NUMINAMATH_GPT_range_g_l360_36001


namespace NUMINAMATH_GPT_exists_n_l360_36044

def F_n (a n : ℕ) : ℕ :=
  let q := a ^ (1 / n)
  let r := a % n
  q + r

noncomputable def largest_A : ℕ :=
  53590

theorem exists_n (a : ℕ) (h : a ≤ largest_A) :
  ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    F_n (F_n (F_n (F_n (F_n (F_n a n1) n2) n3) n4) n5) n6 = 1 := 
sorry

end NUMINAMATH_GPT_exists_n_l360_36044


namespace NUMINAMATH_GPT_age_ratio_l360_36022

theorem age_ratio (darcie_age : ℕ) (father_age : ℕ) (mother_ratio : ℚ) (mother_fraction : ℚ)
  (h1 : darcie_age = 4)
  (h2 : father_age = 30)
  (h3 : mother_ratio = 4/5)
  (h4 : mother_fraction = mother_ratio * father_age)
  (h5 : mother_fraction = 24) :
  (darcie_age : ℚ) / mother_fraction = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l360_36022


namespace NUMINAMATH_GPT_compare_doubling_l360_36086

theorem compare_doubling (a b : ℝ) (h : a > b) : 2 * a > 2 * b :=
  sorry

end NUMINAMATH_GPT_compare_doubling_l360_36086


namespace NUMINAMATH_GPT_p_as_percentage_of_x_l360_36000

-- Given conditions
variables (x y z w t u p : ℝ)
variables (h1 : 0.37 * z = 0.84 * y)
variables (h2 : y = 0.62 * x)
variables (h3 : 0.47 * w = 0.73 * z)
variables (h4 : w = t - u)
variables (h5 : u = 0.25 * t)
variables (h6 : p = z + t + u)

-- Prove that p is 505.675% of x
theorem p_as_percentage_of_x : p = 5.05675 * x := by
  sorry

end NUMINAMATH_GPT_p_as_percentage_of_x_l360_36000
