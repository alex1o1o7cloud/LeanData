import Mathlib

namespace NUMINAMATH_GPT_min_rectangles_to_cover_minimum_number_of_rectangles_required_l2030_203008

-- Definitions based on the conditions
def corners_type1 : Nat := 12
def corners_type2 : Nat := 12

theorem min_rectangles_to_cover (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) : Nat :=
12

theorem minimum_number_of_rectangles_required (type1_corners type2_corners : Nat) (h1 : type1_corners = corners_type1) (h2 : type2_corners = corners_type2) :
  min_rectangles_to_cover type1_corners type2_corners h1 h2 = 12 := by
  sorry

end NUMINAMATH_GPT_min_rectangles_to_cover_minimum_number_of_rectangles_required_l2030_203008


namespace NUMINAMATH_GPT_work_completion_in_6_days_l2030_203037

-- Definitions for the work rates of a, b, and c.
def work_rate_a_b : ℚ := 1 / 8
def work_rate_a : ℚ := 1 / 16
def work_rate_c : ℚ := 1 / 24

-- The theorem to prove that a, b, and c together can complete the work in 6 days.
theorem work_completion_in_6_days : 
  (1 / (work_rate_a_b - work_rate_a)) + work_rate_c = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_in_6_days_l2030_203037


namespace NUMINAMATH_GPT_handshake_count_l2030_203030

theorem handshake_count (couples : ℕ) (people : ℕ) (total_handshakes : ℕ) :
  couples = 6 →
  people = 2 * couples →
  total_handshakes = (people * (people - 1)) / 2 - couples →
  total_handshakes = 60 :=
by
  intros h_couples h_people h_handshakes
  sorry

end NUMINAMATH_GPT_handshake_count_l2030_203030


namespace NUMINAMATH_GPT_part_a_part_b_l2030_203005

theorem part_a (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℤ), a < m * α - n ∧ m * α - n < b :=
sorry

theorem part_b (α : ℝ) (h_irr : Irrational α) (a b : ℝ) (h_lt : a < b) :
  ∃ (m n : ℕ), a < m * α - n ∧ m * α - n < b :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2030_203005


namespace NUMINAMATH_GPT_average_calculation_l2030_203045

def average (a b c : ℚ) : ℚ := (a + b + c) / 3
def pairAverage (a b : ℚ) : ℚ := (a + b) / 2

theorem average_calculation :
  average (average (pairAverage 2 2) 3 1) (pairAverage 1 2) 1 = 3 / 2 := sorry

end NUMINAMATH_GPT_average_calculation_l2030_203045


namespace NUMINAMATH_GPT_abs_value_solutions_l2030_203015

theorem abs_value_solutions (y : ℝ) :
  |4 * y - 5| = 39 ↔ (y = 11 ∨ y = -8.5) :=
by
  sorry

end NUMINAMATH_GPT_abs_value_solutions_l2030_203015


namespace NUMINAMATH_GPT_cubic_sum_l2030_203095

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l2030_203095


namespace NUMINAMATH_GPT_bales_stacked_correct_l2030_203097

-- Given conditions
def initial_bales : ℕ := 28
def final_bales : ℕ := 82

-- Define the stacking function
def bales_stacked (initial final : ℕ) : ℕ := final - initial

-- Theorem statement we need to prove
theorem bales_stacked_correct : bales_stacked initial_bales final_bales = 54 := by
  sorry

end NUMINAMATH_GPT_bales_stacked_correct_l2030_203097


namespace NUMINAMATH_GPT_solve_equation_l2030_203017

theorem solve_equation (x : ℚ) :
  (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) → x = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2030_203017


namespace NUMINAMATH_GPT_triangle_perimeter_l2030_203047

noncomputable def smallest_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_perimeter (a b c : ℕ) (A B C : ℝ) (h1 : A = 2 * B) 
  (h2 : C > π / 2) (h3 : a^2 = b * (b + c)) (h4 : ∃ m n : ℕ, b = m^2 ∧ b + c = n^2 ∧ a = m * n) :
  smallest_perimeter 28 16 33 = 77 :=
by sorry

end NUMINAMATH_GPT_triangle_perimeter_l2030_203047


namespace NUMINAMATH_GPT_arithmetic_identity_l2030_203036

theorem arithmetic_identity : 72 * 989 - 12 * 989 = 59340 := by
  sorry

end NUMINAMATH_GPT_arithmetic_identity_l2030_203036


namespace NUMINAMATH_GPT_trajectory_of_P_l2030_203086

-- Definitions for points and distance
structure Point where
  x : ℝ
  y : ℝ

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Fixed points F1 and F2
variable (F1 F2 : Point)
-- Distance condition
axiom dist_F1F2 : dist F1 F2 = 8

-- Moving point P satisfying the condition
variable (P : Point)
axiom dist_PF1_PF2 : dist P F1 + dist P F2 = 8

-- Proof goal: P lies on the line segment F1F2
theorem trajectory_of_P : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = ⟨(1 - t) * F1.x + t * F2.x, (1 - t) * F1.y + t * F2.y⟩ :=
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l2030_203086


namespace NUMINAMATH_GPT_number_of_performance_orders_l2030_203059

-- Define the options for the programs
def programs : List String := ["A", "B", "C", "D", "E", "F", "G", "H"]

-- Define a function to count valid performance orders under given conditions
def countPerformanceOrders (progs : List String) : ℕ :=
  sorry  -- This is where the logic to count performance orders goes

-- The theorem to assert the total number of performance orders
theorem number_of_performance_orders : countPerformanceOrders programs = 2860 :=
by
  sorry  -- Proof of the theorem

end NUMINAMATH_GPT_number_of_performance_orders_l2030_203059


namespace NUMINAMATH_GPT_integer_solutions_count_l2030_203055

theorem integer_solutions_count :
  ∃ (s : Finset ℤ), s.card = 6 ∧ ∀ x ∈ s, 4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 6 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l2030_203055


namespace NUMINAMATH_GPT_student_correct_answers_l2030_203032

theorem student_correct_answers (C I : ℕ) 
  (h1 : C + I = 100) 
  (h2 : C - 2 * I = 61) : 
  C = 87 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l2030_203032


namespace NUMINAMATH_GPT_balloons_problem_l2030_203062

theorem balloons_problem :
  ∃ (b y : ℕ), y = 3414 ∧ b + y = 8590 ∧ b - y = 1762 := 
by
  sorry

end NUMINAMATH_GPT_balloons_problem_l2030_203062


namespace NUMINAMATH_GPT_second_class_students_count_l2030_203022

theorem second_class_students_count 
    (x : ℕ)
    (h1 : 12 * 40 = 480)
    (h2 : ∀ x, x * 60 = 60 * x)
    (h3 : (12 + x) * 54 = 480 + 60 * x) : 
    x = 28 :=
by
  sorry

end NUMINAMATH_GPT_second_class_students_count_l2030_203022


namespace NUMINAMATH_GPT_fraction_sum_eq_decimal_l2030_203040

theorem fraction_sum_eq_decimal : (2 / 5) + (2 / 50) + (2 / 500) = 0.444 := by
  sorry

end NUMINAMATH_GPT_fraction_sum_eq_decimal_l2030_203040


namespace NUMINAMATH_GPT_sufficient_not_necessary_p_q_l2030_203039

theorem sufficient_not_necessary_p_q {m : ℝ} 
  (hp : ∀ x, (x^2 - 8*x - 20 ≤ 0) → (-2 ≤ x ∧ x ≤ 10))
  (hq : ∀ x, ((x - 1 - m) * (x - 1 + m) ≤ 0) → (1 - m ≤ x ∧ x ≤ 1 + m))
  (m_pos : 0 < m)  :
  (∀ x, (x - 1 - m) * (x - 1 + m) ≤ 0 → x^2 - 8*x - 20 ≤ 0) ∧ ¬ (∀ x, x^2 - 8*x - 20 ≤ 0 → (x - 1 - m) * (x - 1 + m) ≤ 0) →
  m ≤ 3 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_p_q_l2030_203039


namespace NUMINAMATH_GPT_transformed_roots_l2030_203050

theorem transformed_roots (b c : ℝ) (h₁ : (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C b * Polynomial.X + Polynomial.C c).roots = {2, -3}) :
  (Polynomial.C 1 * (Polynomial.X - Polynomial.C 4)^2 + Polynomial.C b * (Polynomial.X - Polynomial.C 4) + Polynomial.C c).roots = {1, 6} :=
by
  sorry

end NUMINAMATH_GPT_transformed_roots_l2030_203050


namespace NUMINAMATH_GPT_value_of_e_l2030_203065

theorem value_of_e
  (a b c d e : ℤ)
  (h1 : b = a + 2)
  (h2 : c = a + 4)
  (h3 : d = a + 6)
  (h4 : e = a + 8)
  (h5 : a + c = 146) :
  e = 79 :=
  by sorry

end NUMINAMATH_GPT_value_of_e_l2030_203065


namespace NUMINAMATH_GPT_part1_part2_l2030_203026

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |5 - x|

theorem part1 : ∃ m, m = 9 / 2 ∧ ∀ x, f x ≥ m :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 3) : 
  (1 / (a + 1) + 1 / (b + 2)) ≥ 2 / 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2030_203026


namespace NUMINAMATH_GPT_total_length_of_ropes_l2030_203069

theorem total_length_of_ropes (L : ℝ) 
  (h1 : (L - 12 = 4 * (L - 42))) : 
  2 * L = 104 := 
by
  sorry

end NUMINAMATH_GPT_total_length_of_ropes_l2030_203069


namespace NUMINAMATH_GPT_solve_for_x_l2030_203000

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 * x + 4 * x = 150) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2030_203000


namespace NUMINAMATH_GPT_simplify_proof_l2030_203078

noncomputable def simplify_expression (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : ℝ :=
  (1 - 1/x) / ((1 - x^2) / x)

theorem simplify_proof (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : 
  simplify_expression x hx hx1 hx_1 = -1 / (1 + x) := by 
  sorry

end NUMINAMATH_GPT_simplify_proof_l2030_203078


namespace NUMINAMATH_GPT_find_fraction_result_l2030_203090

open Complex

theorem find_fraction_result (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
    (h1 : x + y + z = 30)
    (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 33 := 
    sorry

end NUMINAMATH_GPT_find_fraction_result_l2030_203090


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2030_203001

variable (a b : ℕ) 

theorem isosceles_triangle_perimeter (h1 : a = 3) (h2 : b = 6) : 
  ∃ P, (a = 3 ∧ b = 6 ∧ P = 15 ∨ b = 3 ∧ a = 6 ∧ P = 15) := by
  use 15
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2030_203001


namespace NUMINAMATH_GPT_vicki_donated_fraction_l2030_203003

/-- Given Jeff had 300 pencils and donated 30% of them, and Vicki had twice as many pencils as Jeff originally 
    had, and there are 360 pencils remaining altogether after both donations,
    prove that Vicki donated 3/4 of her pencils. -/
theorem vicki_donated_fraction : 
  let jeff_pencils := 300
  let jeff_donated := jeff_pencils * 0.30
  let jeff_remaining := jeff_pencils - jeff_donated
  let vicki_pencils := 2 * jeff_pencils
  let total_remaining := 360
  let vicki_remaining := total_remaining - jeff_remaining
  let vicki_donated := vicki_pencils - vicki_remaining
  vicki_donated / vicki_pencils = 3 / 4 :=
by
  -- Proof needs to be inserted here
  sorry

end NUMINAMATH_GPT_vicki_donated_fraction_l2030_203003


namespace NUMINAMATH_GPT_tan_product_l2030_203067

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end NUMINAMATH_GPT_tan_product_l2030_203067


namespace NUMINAMATH_GPT_original_savings_l2030_203024

-- Given conditions:
def total_savings (s : ℝ) : Prop :=
  1 / 4 * s = 230

-- Theorem statement: 
theorem original_savings (s : ℝ) (h : total_savings s) : s = 920 :=
sorry

end NUMINAMATH_GPT_original_savings_l2030_203024


namespace NUMINAMATH_GPT_find_x_l2030_203098

theorem find_x (x : ℝ) : (x / 18) * (36 / 72) = 1 → x = 36 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l2030_203098


namespace NUMINAMATH_GPT_no_pos_integers_exist_l2030_203085

theorem no_pos_integers_exist (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬ (3 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
sorry

end NUMINAMATH_GPT_no_pos_integers_exist_l2030_203085


namespace NUMINAMATH_GPT_smallest_N_l2030_203079

-- Definitions for the problem conditions
def is_rectangular_block (a b c : ℕ) (N : ℕ) : Prop :=
  N = a * b * c ∧ 143 = (a - 1) * (b - 1) * (c - 1)

-- Theorem to prove the smallest possible value of N
theorem smallest_N : ∃ a b c : ℕ, is_rectangular_block a b c 336 :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_l2030_203079


namespace NUMINAMATH_GPT_cost_price_of_radio_l2030_203077

-- Definitions for conditions
def selling_price := 1245
def loss_percentage := 17

-- Prove that the cost price is Rs. 1500 given the conditions
theorem cost_price_of_radio : 
  ∃ C, (C - 1245) * 100 / C = 17 ∧ C = 1500 := 
sorry

end NUMINAMATH_GPT_cost_price_of_radio_l2030_203077


namespace NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l2030_203052

open Classical

-- Definitions
def isSecondQuadrant (α : ℝ) : Prop := 90 < α ∧ α < 180
def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ 180 < α ∧ α < 270

-- The theorem statement
theorem second_quadrant_necessary_not_sufficient (α : ℝ) :
  (isSecondQuadrant α → isObtuseAngle α) ∧ ¬(isSecondQuadrant α ↔ isObtuseAngle α) :=
by
  sorry

end NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l2030_203052


namespace NUMINAMATH_GPT_divisor_is_seventeen_l2030_203068

theorem divisor_is_seventeen (D x : ℕ) (h1 : D = 7 * x) (h2 : D + x = 136) : x = 17 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_seventeen_l2030_203068


namespace NUMINAMATH_GPT_trig_sum_identity_l2030_203082

theorem trig_sum_identity :
  Real.sin (20 * Real.pi / 180) + Real.sin (40 * Real.pi / 180) +
  Real.sin (60 * Real.pi / 180) - Real.sin (80 * Real.pi / 180) = Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_trig_sum_identity_l2030_203082


namespace NUMINAMATH_GPT_interest_payment_frequency_l2030_203048

theorem interest_payment_frequency (i : ℝ) (EAR : ℝ) (n : ℕ)
  (h1 : i = 0.10) (h2 : EAR = 0.1025) :
  (1 + i / n)^n = 1 + EAR → n = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_interest_payment_frequency_l2030_203048


namespace NUMINAMATH_GPT_alices_number_l2030_203013

theorem alices_number :
  ∃ (m : ℕ), (180 ∣ m) ∧ (45 ∣ m) ∧ (1000 ≤ m) ∧ (m ≤ 3000) ∧
    (m = 1260 ∨ m = 1440 ∨ m = 1620 ∨ m = 1800 ∨ m = 1980 ∨
     m = 2160 ∨ m = 2340 ∨ m = 2520 ∨ m = 2700 ∨ m = 2880) :=
by
  sorry

end NUMINAMATH_GPT_alices_number_l2030_203013


namespace NUMINAMATH_GPT_sqrt_product_simplification_l2030_203074

variable (p : ℝ)

theorem sqrt_product_simplification (hp : 0 ≤ p) :
  (Real.sqrt (42 * p) * Real.sqrt (7 * p) * Real.sqrt (14 * p)) = 42 * p * Real.sqrt (7 * p) :=
sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l2030_203074


namespace NUMINAMATH_GPT_cost_of_shoes_l2030_203076

   theorem cost_of_shoes (initial_budget remaining_budget : ℝ) (H_initial : initial_budget = 999) (H_remaining : remaining_budget = 834) : 
   initial_budget - remaining_budget = 165 := by
     sorry
   
end NUMINAMATH_GPT_cost_of_shoes_l2030_203076


namespace NUMINAMATH_GPT_find_b_l2030_203071

theorem find_b (b : ℝ) : (∃ c : ℝ, (16 : ℝ) * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2030_203071


namespace NUMINAMATH_GPT_apples_total_l2030_203063

theorem apples_total (Benny_picked Dan_picked : ℕ) (hB : Benny_picked = 2) (hD : Dan_picked = 9) : Benny_picked + Dan_picked = 11 :=
by
  -- Definitions
  sorry

end NUMINAMATH_GPT_apples_total_l2030_203063


namespace NUMINAMATH_GPT_value_of_a_minus_c_l2030_203027

theorem value_of_a_minus_c
  (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) :
  a - c = -200 := sorry

end NUMINAMATH_GPT_value_of_a_minus_c_l2030_203027


namespace NUMINAMATH_GPT_average_increase_by_3_l2030_203096

def initial_average_before_inning_17 (A : ℝ) : Prop :=
  16 * A + 85 = 17 * 37

theorem average_increase_by_3 (A : ℝ) (h : initial_average_before_inning_17 A) :
  37 - A = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_increase_by_3_l2030_203096


namespace NUMINAMATH_GPT_solution_of_system_of_inequalities_l2030_203007

theorem solution_of_system_of_inequalities (p : ℝ) :
  19 * p < 10 ∧ p > 0.5 ↔ (1 / 2) < p ∧ p < (10 / 19) :=
by
  sorry

end NUMINAMATH_GPT_solution_of_system_of_inequalities_l2030_203007


namespace NUMINAMATH_GPT_total_pixels_correct_l2030_203088

-- Define the monitor's dimensions and pixel density as given conditions
def width_inches : ℕ := 21
def height_inches : ℕ := 12
def pixels_per_inch : ℕ := 100

-- Define the width and height in pixels based on the given conditions
def width_pixels : ℕ := width_inches * pixels_per_inch
def height_pixels : ℕ := height_inches * pixels_per_inch

-- State the objective: proving the total number of pixels on the monitor
theorem total_pixels_correct : width_pixels * height_pixels = 2520000 := by
  sorry

end NUMINAMATH_GPT_total_pixels_correct_l2030_203088


namespace NUMINAMATH_GPT_percentage_difference_l2030_203072

theorem percentage_difference : (70 / 100 : ℝ) * 100 - (60 / 100 : ℝ) * 80 = 22 := by
  sorry

end NUMINAMATH_GPT_percentage_difference_l2030_203072


namespace NUMINAMATH_GPT_gcd_lcm_product_l2030_203087

theorem gcd_lcm_product (a b : ℕ) (ha : a = 24) (hb : b = 36) : 
  Nat.gcd a b * Nat.lcm a b = 864 :=
by
  rw [ha, hb]
  -- This theorem proves that the product of the GCD and LCM of 24 and 36 equals 864.

  sorry -- Proof will go here

end NUMINAMATH_GPT_gcd_lcm_product_l2030_203087


namespace NUMINAMATH_GPT_part1_part2_l2030_203081

-- Part (I)
theorem part1 (a : ℝ) :
  (∀ x : ℝ, 3 * x - abs (-2 * x + 1) ≥ a ↔ 2 ≤ x) → a = 3 :=
by
  sorry

-- Part (II)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (x - abs (x - a) ≤ 1)) → (a ≤ 1 ∨ 3 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2030_203081


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2030_203043

noncomputable def A : Set ℝ := {-2, -1, 0, 1}
noncomputable def B : Set ℝ := {x | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2030_203043


namespace NUMINAMATH_GPT_translate_upwards_l2030_203070

theorem translate_upwards (x : ℝ) : (2 * x^2) + 2 = 2 * x^2 + 2 := by
  sorry

end NUMINAMATH_GPT_translate_upwards_l2030_203070


namespace NUMINAMATH_GPT_probability_slope_le_one_l2030_203061

noncomputable def point := (ℝ × ℝ)

def Q_in_unit_square (Q : point) : Prop :=
  0 ≤ Q.1 ∧ Q.1 ≤ 1 ∧ 0 ≤ Q.2 ∧ Q.2 ≤ 1

def slope_le_one (Q : point) : Prop :=
  (Q.2 - (1/4)) / (Q.1 - (3/4)) ≤ 1

theorem probability_slope_le_one :
  ∃ p q : ℕ, Q_in_unit_square Q → slope_le_one Q →
  p.gcd q = 1 ∧ (p + q = 11) :=
sorry

end NUMINAMATH_GPT_probability_slope_le_one_l2030_203061


namespace NUMINAMATH_GPT_maddie_watched_138_on_monday_l2030_203012

-- Define the constants and variables from the problem statement
def total_episodes : ℕ := 8
def minutes_per_episode : ℕ := 44
def watched_thursday : ℕ := 21
def watched_friday_episodes : ℕ := 2
def watched_weekend : ℕ := 105

-- Calculate the total minutes watched from all episodes
def total_minutes : ℕ := total_episodes * minutes_per_episode

-- Calculate the minutes watched on Friday
def watched_friday : ℕ := watched_friday_episodes * minutes_per_episode

-- Calculate the total minutes watched on weekdays excluding Monday
def watched_other_days : ℕ := watched_thursday + watched_friday + watched_weekend

-- Statement to prove that Maddie watched 138 minutes on Monday
def minutes_watched_on_monday : ℕ := total_minutes - watched_other_days

-- The final statement for proof in Lean 4
theorem maddie_watched_138_on_monday : minutes_watched_on_monday = 138 := by
  -- This theorem should be proved using the above definitions and calculations, proof skipped with sorry
  sorry

end NUMINAMATH_GPT_maddie_watched_138_on_monday_l2030_203012


namespace NUMINAMATH_GPT_chelsea_cupcakes_time_l2030_203010

theorem chelsea_cupcakes_time
  (batches : ℕ)
  (bake_time_per_batch : ℕ)
  (ice_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : batches = 4)
  (h2 : bake_time_per_batch = 20)
  (h3 : ice_time_per_batch = 30)
  (h4 : total_time = (bake_time_per_batch + ice_time_per_batch) * batches) :
  total_time = 200 :=
  by
  -- The proof statement here
  -- The proof would go here, but we skip it for now
  sorry

end NUMINAMATH_GPT_chelsea_cupcakes_time_l2030_203010


namespace NUMINAMATH_GPT_ellipse_equation_l2030_203042

noncomputable def point := (ℝ × ℝ)

theorem ellipse_equation (a b : ℝ) (P Q : point) (h1 : a > b) (h2: b > 0) (e : ℝ) (h3 : e = 1/2)
  (h4 : P = (2, 3)) (h5 : Q = (2, -3))
  (h6 : (P.1^2)/(a^2) + (P.2^2)/(b^2) = 1) (h7 : (Q.1^2)/(a^2) + (Q.2^2)/(b^2) = 1) :
  (∀ x y: ℝ, (x^2/16 + y^2/12 = 1) ↔ (x^2/a^2 + y^2/b^2 = 1)) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l2030_203042


namespace NUMINAMATH_GPT_delores_remaining_money_l2030_203089

variable (delores_money : ℕ := 450)
variable (computer_price : ℕ := 1000)
variable (computer_discount : ℝ := 0.30)
variable (printer_price : ℕ := 100)
variable (printer_tax_rate : ℝ := 0.15)
variable (table_price_euros : ℕ := 200)
variable (exchange_rate : ℝ := 1.2)

def computer_sale_price : ℝ := computer_price * (1 - computer_discount)
def printer_total_cost : ℝ := printer_price * (1 + printer_tax_rate)
def table_cost_dollars : ℝ := table_price_euros * exchange_rate
def total_cost : ℝ := computer_sale_price + printer_total_cost + table_cost_dollars
def remaining_money : ℝ := delores_money - total_cost

theorem delores_remaining_money : remaining_money = -605 := by
  sorry

end NUMINAMATH_GPT_delores_remaining_money_l2030_203089


namespace NUMINAMATH_GPT_piesEatenWithForksPercentage_l2030_203094

def totalPies : ℕ := 2000
def notEatenWithForks : ℕ := 640
def eatenWithForks : ℕ := totalPies - notEatenWithForks

def percentageEatenWithForks := (eatenWithForks : ℚ) / totalPies * 100

theorem piesEatenWithForksPercentage : percentageEatenWithForks = 68 := by
  sorry

end NUMINAMATH_GPT_piesEatenWithForksPercentage_l2030_203094


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2030_203092

variable (a b c d : ℝ)

-- The given conditions:
def cond1 : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def cond2 : Prop := a^2 + b^2 = 4
def cond3 : Prop := c * d = 1

-- The minimum value:
def expression_value : ℝ := (a^2 * c^2 + b^2 * d^2) * (b^2 * c^2 + a^2 * d^2)

theorem minimum_value_of_expression :
  cond1 a b c d → cond2 a b → cond3 c d → expression_value a b c d ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2030_203092


namespace NUMINAMATH_GPT_equal_roots_B_value_l2030_203014

theorem equal_roots_B_value (B : ℝ) :
  (∀ k : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (k = 1 → (B^2 - 4 * (2 * 1) * 2 = 0))) → B = 4 ∨ B = -4 :=
by
  sorry

end NUMINAMATH_GPT_equal_roots_B_value_l2030_203014


namespace NUMINAMATH_GPT_problem_l2030_203019

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end NUMINAMATH_GPT_problem_l2030_203019


namespace NUMINAMATH_GPT_unique_real_solution_bound_l2030_203093

theorem unique_real_solution_bound (b : ℝ) :
  (∀ x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0 → ∃! y : ℝ, y = x) → b < 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_real_solution_bound_l2030_203093


namespace NUMINAMATH_GPT_odd_function_increasing_ln_x_condition_l2030_203083

theorem odd_function_increasing_ln_x_condition 
  {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) 
  {x : ℝ} 
  (h_f_ln_x : f (Real.log x) < 0) : 
  0 < x ∧ x < 1 := 
sorry

end NUMINAMATH_GPT_odd_function_increasing_ln_x_condition_l2030_203083


namespace NUMINAMATH_GPT_dot_product_eq_neg20_l2030_203004

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (-5, 5)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_eq_neg20 :
  dot_product a b = -20 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_eq_neg20_l2030_203004


namespace NUMINAMATH_GPT_min_value_of_expression_l2030_203018

noncomputable def smallest_value (a b c : ℕ) : ℤ :=
  3 * a - 2 * a * b + a * c

theorem min_value_of_expression : ∃ (a b c : ℕ), 0 < a ∧ a < 7 ∧ 0 < b ∧ b ≤ 3 ∧ 0 < c ∧ c ≤ 4 ∧ smallest_value a b c = -12 := by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2030_203018


namespace NUMINAMATH_GPT_mark_total_flowers_l2030_203064

theorem mark_total_flowers (yellow purple green total : ℕ) 
  (hyellow : yellow = 10)
  (hpurple : purple = yellow + (yellow * 80 / 100))
  (hgreen : green = (yellow + purple) * 25 / 100)
  (htotal : total = yellow + purple + green) : 
  total = 35 :=
by
  sorry

end NUMINAMATH_GPT_mark_total_flowers_l2030_203064


namespace NUMINAMATH_GPT_odd_positive_int_divides_3pow_n_plus_1_l2030_203053

theorem odd_positive_int_divides_3pow_n_plus_1 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : n > 0) : 
  n ∣ (3^n + 1) ↔ n = 1 := 
by
  sorry

end NUMINAMATH_GPT_odd_positive_int_divides_3pow_n_plus_1_l2030_203053


namespace NUMINAMATH_GPT_tangent_length_to_circle_l2030_203023

-- Definitions capturing the conditions
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0
def line_l (x y a : ℝ) : Prop := x + a * y - 1 = 0
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Main theorem statement proving the question against the answer
theorem tangent_length_to_circle (a : ℝ) (x y : ℝ) (hC : circle_C x y) (hl : line_l 2 1 a) :
  (a = -1) -> (point_A a = (-4, -1)) -> ∃ b : ℝ, b = 6 := 
sorry

end NUMINAMATH_GPT_tangent_length_to_circle_l2030_203023


namespace NUMINAMATH_GPT_real_root_quadratic_l2030_203028

theorem real_root_quadratic (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 9 = 0) ↔ b ≤ -6 ∨ b ≥ 6 := 
sorry

end NUMINAMATH_GPT_real_root_quadratic_l2030_203028


namespace NUMINAMATH_GPT_bricks_in_wall_l2030_203035

theorem bricks_in_wall (h : ℕ) 
  (brenda_rate : ℕ := h / 8)
  (brandon_rate : ℕ := h / 12)
  (combined_rate : ℕ := (5 * h) / 24)
  (decreased_combined_rate : ℕ := combined_rate - 15)
  (work_time : ℕ := 6) :
  work_time * decreased_combined_rate = h → h = 360 := by
  intros h_eq
  sorry

end NUMINAMATH_GPT_bricks_in_wall_l2030_203035


namespace NUMINAMATH_GPT_jerry_reaches_five_probability_l2030_203025

noncomputable def probability_move_reaches_five_at_some_point : ℚ :=
  let num_heads_needed := 7
  let num_tails_needed := 3
  let total_tosses := 10
  let num_ways_to_choose_heads := Nat.choose total_tosses num_heads_needed
  let total_possible_outcomes : ℚ := 2^total_tosses
  let prob_reach_4 := num_ways_to_choose_heads / total_possible_outcomes
  let prob_reach_5_at_some_point := 2 * prob_reach_4
  prob_reach_5_at_some_point

theorem jerry_reaches_five_probability :
  probability_move_reaches_five_at_some_point = 15 / 64 := by
  sorry

end NUMINAMATH_GPT_jerry_reaches_five_probability_l2030_203025


namespace NUMINAMATH_GPT_percent_relation_l2030_203073

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end NUMINAMATH_GPT_percent_relation_l2030_203073


namespace NUMINAMATH_GPT_min_value_of_expression_l2030_203029

/-- Given the area of △ ABC is 2, and the sides opposite to angles A, B, C are a, b, c respectively,
    prove that the minimum value of a^2 + 2b^2 + 3c^2 is 8 * sqrt(11). -/
theorem min_value_of_expression
  (a b c : ℝ)
  (h₁ : 1/2 * b * c * Real.sin A = 2) :
  a^2 + 2 * b^2 + 3 * c^2 ≥ 8 * Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2030_203029


namespace NUMINAMATH_GPT_right_building_shorter_l2030_203046

-- Define the conditions as hypotheses
def middle_building_height : ℕ := 100
def left_building_height : ℕ := (80 * middle_building_height) / 100
def combined_height_left_middle : ℕ := left_building_height + middle_building_height
def total_height : ℕ := 340
def right_building_height : ℕ := total_height - combined_height_left_middle

-- Define the statement we need to prove
theorem right_building_shorter :
  combined_height_left_middle - right_building_height = 20 :=
by sorry

end NUMINAMATH_GPT_right_building_shorter_l2030_203046


namespace NUMINAMATH_GPT_number_of_ordered_pairs_lcm_232848_l2030_203020

theorem number_of_ordered_pairs_lcm_232848 :
  let count_pairs :=
    let pairs_1 := 9
    let pairs_2 := 7
    let pairs_3 := 5
    let pairs_4 := 3
    pairs_1 * pairs_2 * pairs_3 * pairs_4
  count_pairs = 945 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_lcm_232848_l2030_203020


namespace NUMINAMATH_GPT_garden_perimeter_is_64_l2030_203084

-- Define the playground dimensions and its area 
def playground_length := 16
def playground_width := 12
def playground_area := playground_length * playground_width

-- Define the garden width and its area being the same as the playground's area
def garden_width := 8
def garden_area := playground_area

-- Calculate the garden's length
def garden_length := garden_area / garden_width

-- Calculate the perimeter of the garden
def garden_perimeter := 2 * (garden_length + garden_width)

theorem garden_perimeter_is_64 :
  garden_perimeter = 64 := 
sorry

end NUMINAMATH_GPT_garden_perimeter_is_64_l2030_203084


namespace NUMINAMATH_GPT_evaluate_expression_l2030_203080

theorem evaluate_expression : ((3 ^ 2) ^ 3) - ((2 ^ 3) ^ 2) = 665 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2030_203080


namespace NUMINAMATH_GPT_book_distribution_ways_l2030_203060

theorem book_distribution_ways : 
  ∃ n : ℕ, n = 7 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 7 →
  ∃ l : ℕ, l + (8 - l) = 8 ∧ 1 ≤ l ∧ 1 ≤ 8 - l :=
by
  -- We will provide a proof here.
  sorry

end NUMINAMATH_GPT_book_distribution_ways_l2030_203060


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l2030_203054

theorem arithmetic_geometric_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (d : ℚ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : a 3 = a 1 + 2 * d) (h3 : a 5 = a 1 + 4 * d) (h4 : (a 3) ^ 2 = a 1 * a 5)
  (h5 : d ≠ 0) : S n = (n^2 + 7 * n) / 4 := sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l2030_203054


namespace NUMINAMATH_GPT_price_on_friday_is_correct_l2030_203051

-- Define initial price on Tuesday
def price_on_tuesday : ℝ := 50

-- Define the percentage increase on Wednesday (20%)
def percentage_increase : ℝ := 0.20

-- Define the percentage discount on Friday (15%)
def percentage_discount : ℝ := 0.15

-- Define the price on Wednesday after the increase
def price_on_wednesday : ℝ := price_on_tuesday * (1 + percentage_increase)

-- Define the price on Friday after the discount
def price_on_friday : ℝ := price_on_wednesday * (1 - percentage_discount)

-- Theorem statement to prove that the price on Friday is 51 dollars
theorem price_on_friday_is_correct : price_on_friday = 51 :=
by
  sorry

end NUMINAMATH_GPT_price_on_friday_is_correct_l2030_203051


namespace NUMINAMATH_GPT_number_of_people_in_first_group_l2030_203057

-- Define variables representing the work done by one person in one day (W) and the number of people in the first group (P)
variable (W : ℕ) (P : ℕ)

-- Conditions from the problem
-- Some people can do 3 times a particular work in 3 days
def condition1 : Prop := P * 3 * W = 3 * W

-- It takes 6 people 3 days to do 6 times of that particular work
def condition2 : Prop := 6 * 3 * W = 6 * W

-- The statement to prove
theorem number_of_people_in_first_group 
  (h1 : condition1 W P) 
  (h2 : condition2 W) : P = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_first_group_l2030_203057


namespace NUMINAMATH_GPT_none_of_these_l2030_203091

theorem none_of_these (s x y : ℝ) (hs : s > 1) (hx2y_ne_zero : x^2 * y ≠ 0) (hineq : x * s^2 > y * s^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < y / x) :=
by
  sorry

end NUMINAMATH_GPT_none_of_these_l2030_203091


namespace NUMINAMATH_GPT_sequence_periodic_l2030_203041

noncomputable def exists_N (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → a (n+2) = abs (a (n+1)) - a n

theorem sequence_periodic (a : ℕ → ℝ) (h : exists_N a) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a (n+9) = a n :=
sorry

end NUMINAMATH_GPT_sequence_periodic_l2030_203041


namespace NUMINAMATH_GPT_jordan_rectangle_width_l2030_203049

theorem jordan_rectangle_width :
  ∀ (areaC areaJ : ℕ) (lengthC widthC lengthJ widthJ : ℕ), 
    (areaC = lengthC * widthC) →
    (areaJ = lengthJ * widthJ) →
    (areaC = areaJ) →
    (lengthC = 5) →
    (widthC = 24) →
    (lengthJ = 3) →
    widthJ = 40 :=
by
  intros areaC areaJ lengthC widthC lengthJ widthJ
  intro hAreaC
  intro hAreaJ
  intro hEqualArea
  intro hLengthC
  intro hWidthC
  intro hLengthJ
  sorry

end NUMINAMATH_GPT_jordan_rectangle_width_l2030_203049


namespace NUMINAMATH_GPT_percentage_increase_after_lawnmower_l2030_203031

-- Definitions from conditions
def initial_daily_yards := 8
def weekly_yards_after_lawnmower := 84
def days_in_week := 7

-- Problem statement
theorem percentage_increase_after_lawnmower : 
  ((weekly_yards_after_lawnmower / days_in_week - initial_daily_yards) / initial_daily_yards) * 100 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_after_lawnmower_l2030_203031


namespace NUMINAMATH_GPT_smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l2030_203099

theorem smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum :
  ∃ (a : ℤ), (∃ (l : List ℤ), l.length = 50 ∧ List.prod l = 0 ∧ 0 < List.sum l ∧ List.sum l = 25) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_50_consecutive_integers_with_product_zero_and_positive_sum_l2030_203099


namespace NUMINAMATH_GPT_greatest_natural_number_exists_l2030_203006

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
    n * (n + 1) * (2 * n + 1) / 6

noncomputable def squared_sum_from_to (a b : ℕ) : ℕ :=
    sum_of_squares b - sum_of_squares (a - 1)

noncomputable def is_perfect_square (n : ℕ) : Prop :=
    ∃ k, k * k = n

theorem greatest_natural_number_exists :
    ∃ n : ℕ, n = 1921 ∧ n ≤ 2008 ∧ 
    is_perfect_square ((sum_of_squares n) * (squared_sum_from_to (n + 1) (2 * n))) :=
by
  sorry

end NUMINAMATH_GPT_greatest_natural_number_exists_l2030_203006


namespace NUMINAMATH_GPT_last_two_digits_7_pow_2011_l2030_203034

noncomputable def pow_mod_last_two_digits (n : ℕ) : ℕ :=
  (7^n) % 100

theorem last_two_digits_7_pow_2011 : pow_mod_last_two_digits 2011 = 43 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_7_pow_2011_l2030_203034


namespace NUMINAMATH_GPT_distance_points_lt_2_over_3_r_l2030_203044

theorem distance_points_lt_2_over_3_r (r : ℝ) (h_pos_r : 0 < r) (points : Fin 17 → ℝ × ℝ)
  (h_points_in_circle : ∀ i, (points i).1 ^ 2 + (points i).2 ^ 2 < r ^ 2) :
  ∃ i j : Fin 17, i ≠ j ∧ (dist (points i) (points j) < 2 * r / 3) :=
by
  sorry

end NUMINAMATH_GPT_distance_points_lt_2_over_3_r_l2030_203044


namespace NUMINAMATH_GPT_fitness_center_cost_effectiveness_l2030_203009

noncomputable def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 30 then 90 
  else 2 * x + 30

def cost_comparison (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : Prop :=
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x)

theorem fitness_center_cost_effectiveness (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : cost_comparison x h1 h2 :=
by
  sorry

end NUMINAMATH_GPT_fitness_center_cost_effectiveness_l2030_203009


namespace NUMINAMATH_GPT_dollar_neg3_4_eq_neg27_l2030_203033

-- Define the operation $$
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem stating the value of (-3) $$ 4
theorem dollar_neg3_4_eq_neg27 : dollar (-3) 4 = -27 := 
by
  sorry

end NUMINAMATH_GPT_dollar_neg3_4_eq_neg27_l2030_203033


namespace NUMINAMATH_GPT_work_together_days_l2030_203058

theorem work_together_days (ravi_days prakash_days : ℕ) (hr : ravi_days = 50) (hp : prakash_days = 75) : 
  (ravi_days * prakash_days) / (ravi_days + prakash_days) = 30 :=
sorry

end NUMINAMATH_GPT_work_together_days_l2030_203058


namespace NUMINAMATH_GPT_repeating_decimals_sum_l2030_203066

def repeating_decimal1 : ℚ := (2 : ℚ) / 9  -- 0.\overline{2}
def repeating_decimal2 : ℚ := (3 : ℚ) / 99 -- 0.\overline{03}

theorem repeating_decimals_sum : repeating_decimal1 + repeating_decimal2 = (25 : ℚ) / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_sum_l2030_203066


namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l2030_203002

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n = 23 ∧ ∀ m : ℤ, (m ≤ 23 → m^2 ≤ 500) → (m < 23 ∧ m > 0 → (m + 1)^2 > 500) :=
by
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_500_l2030_203002


namespace NUMINAMATH_GPT_desired_interest_rate_l2030_203011

theorem desired_interest_rate 
  (F : ℝ) -- Face value of each share
  (D : ℝ) -- Dividend rate
  (M : ℝ) -- Market value of each share
  (annual_dividend : ℝ := (D / 100) * F) -- Annual dividend per share
  (desired_interest_rate : ℝ := (annual_dividend / M) * 100) -- Desired interest rate
  (F_eq : F = 44) -- Given Face value
  (D_eq : D = 9) -- Given Dividend rate
  (M_eq : M = 33) -- Given Market value
  : desired_interest_rate = 12 := 
by
  sorry

end NUMINAMATH_GPT_desired_interest_rate_l2030_203011


namespace NUMINAMATH_GPT_g_neg_six_eq_neg_twenty_l2030_203016

theorem g_neg_six_eq_neg_twenty (g : ℤ → ℤ)
    (h1 : g 1 - 1 > 0)
    (h2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x)
    (h3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3) :
  g (-6) = -20 := 
sorry

end NUMINAMATH_GPT_g_neg_six_eq_neg_twenty_l2030_203016


namespace NUMINAMATH_GPT_restaurant_bill_l2030_203038

theorem restaurant_bill 
  (salisbury_steak : ℝ := 16.00)
  (chicken_fried_steak : ℝ := 18.00)
  (mozzarella_sticks : ℝ := 8.00)
  (caesar_salad : ℝ := 6.00)
  (bowl_chili : ℝ := 7.00)
  (chocolate_lava_cake : ℝ := 7.50)
  (cheesecake : ℝ := 6.50)
  (iced_tea : ℝ := 3.00)
  (soda : ℝ := 3.50)
  (half_off_meal : ℝ := 0.5)
  (dessert_discount : ℝ := 0.1)
  (tip_percent : ℝ := 0.2)
  (sales_tax : ℝ := 0.085) :
  let total : ℝ :=
    (salisbury_steak * half_off_meal) +
    (chicken_fried_steak * half_off_meal) +
    mozzarella_sticks +
    caesar_salad +
    bowl_chili +
    (chocolate_lava_cake * (1 - dessert_discount)) +
    (cheesecake * (1 - dessert_discount)) +
    iced_tea +
    soda
  let total_with_tax : ℝ := total * (1 + sales_tax)
  let final_total : ℝ := total_with_tax * (1 + tip_percent)
  final_total = 73.04 :=
by
  sorry

end NUMINAMATH_GPT_restaurant_bill_l2030_203038


namespace NUMINAMATH_GPT_remainder_when_xyz_divided_by_9_is_0_l2030_203021

theorem remainder_when_xyz_divided_by_9_is_0
  (x y z : ℕ)
  (hx : x < 9)
  (hy : y < 9)
  (hz : z < 9)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 9])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 9])
  (h3 : 2 * x + y + 3 * z ≡ 5 [MOD 9]) :
  (x * y * z) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_when_xyz_divided_by_9_is_0_l2030_203021


namespace NUMINAMATH_GPT_apples_to_cucumbers_l2030_203056

theorem apples_to_cucumbers (a b c : ℕ) 
    (h₁ : 10 * a = 5 * b) 
    (h₂ : 3 * b = 4 * c) : 
    (24 * a) = 16 * c := 
by
  sorry

end NUMINAMATH_GPT_apples_to_cucumbers_l2030_203056


namespace NUMINAMATH_GPT_line_divides_circle_l2030_203075

theorem line_divides_circle (k m : ℝ) :
  (∀ x y : ℝ, y = x - 1 → x^2 + y^2 + k*x + m*y - 4 = 0 → m - k = 2) :=
sorry

end NUMINAMATH_GPT_line_divides_circle_l2030_203075
