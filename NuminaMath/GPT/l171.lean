import Mathlib

namespace NUMINAMATH_GPT_find_crayons_in_pack_l171_17188

variables (crayons_in_locker : ℕ) (crayons_given_by_bobby : ℕ) (crayons_given_to_mary : ℕ) (crayons_final_count : ℕ) (crayons_in_pack : ℕ)

-- Definitions from the conditions
def initial_crayons := 36
def bobby_gave := initial_crayons / 2
def mary_crayons := 25
def final_crayons := initial_crayons + bobby_gave - mary_crayons

-- The theorem to prove
theorem find_crayons_in_pack : initial_crayons = 36 ∧ bobby_gave = 18 ∧ mary_crayons = 25 ∧ final_crayons = 29 → crayons_in_pack = 29 :=
by
  sorry

end NUMINAMATH_GPT_find_crayons_in_pack_l171_17188


namespace NUMINAMATH_GPT_parabola_vertex_in_other_l171_17141

theorem parabola_vertex_in_other (p q a : ℝ) (h₁ : a ≠ 0) 
  (h₂ : ∀ (x : ℝ),  x = a → pa^2 = p * x^2) 
  (h₃ : ∀ (x : ℝ),  x = 0 → 0 = q * (x - a)^2 + pa^2) : 
  p + q = 0 := 
sorry

end NUMINAMATH_GPT_parabola_vertex_in_other_l171_17141


namespace NUMINAMATH_GPT_jiahao_estimate_larger_l171_17194

variable (x y : ℝ)
variable (hxy : x > y)
variable (hy0 : y > 0)

theorem jiahao_estimate_larger (x y : ℝ) (hxy : x > y) (hy0 : y > 0) :
  (x + 2) - (y - 1) > x - y :=
by
  sorry

end NUMINAMATH_GPT_jiahao_estimate_larger_l171_17194


namespace NUMINAMATH_GPT_library_visitors_on_sunday_l171_17102

def avg_visitors_sundays (S : ℕ) : Prop :=
  let total_days := 30
  let avg_other_days := 240
  let avg_total := 285
  let sundays := 5
  let other_days := total_days - sundays
  (S * sundays) + (avg_other_days * other_days) = avg_total * total_days

theorem library_visitors_on_sunday (S : ℕ) (h : avg_visitors_sundays S) : S = 510 :=
by
  sorry

end NUMINAMATH_GPT_library_visitors_on_sunday_l171_17102


namespace NUMINAMATH_GPT_spent_more_on_candy_bar_l171_17170

-- Definitions of conditions
def money_Dan_has : ℕ := 2
def candy_bar_cost : ℕ := 6
def chocolate_cost : ℕ := 3

-- Statement of the proof problem
theorem spent_more_on_candy_bar : candy_bar_cost - chocolate_cost = 3 := by
  sorry

end NUMINAMATH_GPT_spent_more_on_candy_bar_l171_17170


namespace NUMINAMATH_GPT_count_perfect_square_factors_of_360_l171_17171

def is_prime_fact_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_perfect_square (d : ℕ) : Prop :=
  ∃ a b c : ℕ, d = 2^(2*a) * 3^(2*b) * 5^(2*c)

def prime_factorization_360 : Prop :=
  ∀ d : ℕ, d ∣ 360 → is_perfect_square d

theorem count_perfect_square_factors_of_360 : ∃ count : ℕ, count = 4 :=
  sorry

end NUMINAMATH_GPT_count_perfect_square_factors_of_360_l171_17171


namespace NUMINAMATH_GPT_positive_difference_of_perimeters_l171_17105

theorem positive_difference_of_perimeters :
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  (perimeter1 - perimeter2) = 4 :=
by
  let length1 := 6
  let width1 := 1
  let length2 := 3
  let width2 := 2
  let perimeter1 := 2 * (length1 + width1)
  let perimeter2 := 2 * (length2 + width2)
  show (perimeter1 - perimeter2) = 4
  sorry

end NUMINAMATH_GPT_positive_difference_of_perimeters_l171_17105


namespace NUMINAMATH_GPT_total_people_on_boats_l171_17161

-- Definitions based on the given conditions
def boats : Nat := 5
def people_per_boat : Nat := 3

-- Theorem statement to prove the total number of people on boats in the lake
theorem total_people_on_boats : boats * people_per_boat = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_people_on_boats_l171_17161


namespace NUMINAMATH_GPT_solution_of_equation_l171_17147

theorem solution_of_equation (x : ℤ) : 7 * x - 5 = 6 * x → x = 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_solution_of_equation_l171_17147


namespace NUMINAMATH_GPT_total_pupils_correct_l171_17127

-- Definitions of the conditions
def number_of_girls : ℕ := 308
def number_of_boys : ℕ := 318

-- Definition of the number of pupils
def total_number_of_pupils : ℕ := number_of_girls + number_of_boys

-- The theorem to be proven
theorem total_pupils_correct : total_number_of_pupils = 626 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_total_pupils_correct_l171_17127


namespace NUMINAMATH_GPT_original_cost_l171_17100

theorem original_cost (SP : ℝ) (C : ℝ) (h1 : SP = 540) (h2 : SP = C + 0.35 * C) : C = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_cost_l171_17100


namespace NUMINAMATH_GPT_meet_time_l171_17197

theorem meet_time 
  (circumference : ℝ) 
  (deepak_speed_kmph : ℝ) 
  (wife_speed_kmph : ℝ) 
  (deepak_speed_mpm : ℝ := deepak_speed_kmph * 1000 / 60) 
  (wife_speed_mpm : ℝ := wife_speed_kmph * 1000 / 60) 
  (relative_speed : ℝ := deepak_speed_mpm + wife_speed_mpm)
  (time_to_meet : ℝ := circumference / relative_speed) :
  circumference = 660 → 
  deepak_speed_kmph = 4.5 → 
  wife_speed_kmph = 3.75 → 
  time_to_meet = 4.8 :=
by 
  intros h1 h2 h3 
  sorry

end NUMINAMATH_GPT_meet_time_l171_17197


namespace NUMINAMATH_GPT_value_of_x_minus_y_l171_17149

theorem value_of_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l171_17149


namespace NUMINAMATH_GPT_min_value_expression_l171_17144

theorem min_value_expression (a b c : ℝ) (h : 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 4) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (4 / c - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l171_17144


namespace NUMINAMATH_GPT_compute_expression_l171_17168

theorem compute_expression : 7^3 - 5 * (6^2) + 2^4 = 179 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l171_17168


namespace NUMINAMATH_GPT_m_range_circle_l171_17137

noncomputable def circle_equation (m : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 2 * x + 4 * y + m = 0

theorem m_range_circle (m : ℝ) : circle_equation m → m < 5 := by
  sorry

end NUMINAMATH_GPT_m_range_circle_l171_17137


namespace NUMINAMATH_GPT_find_length_of_FC_l171_17136

theorem find_length_of_FC (DC CB AD AB ED FC : ℝ) (h1 : DC = 9) (h2 : CB = 10) (h3 : AB = (1 / 3) * AD) (h4 : ED = (2 / 3) * AD) : 
  FC = 13 := by
  sorry

end NUMINAMATH_GPT_find_length_of_FC_l171_17136


namespace NUMINAMATH_GPT_find_original_shirt_price_l171_17190

noncomputable def original_shirt_price (S pants_orig_price jacket_orig_price total_paid : ℝ) :=
  let discounted_shirt := S * 0.5625
  let discounted_pants := pants_orig_price * 0.70
  let discounted_jacket := jacket_orig_price * 0.64
  let total_before_loyalty := discounted_shirt + discounted_pants + discounted_jacket
  let total_after_loyalty := total_before_loyalty * 0.90
  let total_after_tax := total_after_loyalty * 1.15
  total_after_tax = total_paid

theorem find_original_shirt_price : 
  original_shirt_price S 50 75 150 → S = 110.07 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_original_shirt_price_l171_17190


namespace NUMINAMATH_GPT_quadrilateral_probability_l171_17163

def total_shapes : ℕ := 6
def quadrilateral_shapes : ℕ := 3

theorem quadrilateral_probability : (quadrilateral_shapes : ℚ) / (total_shapes : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_probability_l171_17163


namespace NUMINAMATH_GPT_geometric_sequence_y_l171_17173

theorem geometric_sequence_y (x y z : ℝ) (h1 : 1 ≠ 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : z ≠ 0) (h5 : 9 ≠ 0)
  (h_seq : ∀ a b c d e : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ a * e = b * d ∧ b * d = c^2) →
           (a, b, c, d, e) = (1, x, y, z, 9)) :
  y = 3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_y_l171_17173


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l171_17176

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l171_17176


namespace NUMINAMATH_GPT_solve_abs_system_eq_l171_17160

theorem solve_abs_system_eq (x y : ℝ) :
  (|x + y| + |1 - x| = 6) ∧ (|x + y + 1| + |1 - y| = 4) ↔ x = -2 ∧ y = -1 :=
by sorry

end NUMINAMATH_GPT_solve_abs_system_eq_l171_17160


namespace NUMINAMATH_GPT_kevin_correct_answer_l171_17116

theorem kevin_correct_answer (k : ℝ) (h : (20 + 1) * (6 + k) = 126 + 21 * k) :
  (20 + 1 * 6 + k) = 21 := by
sorry

end NUMINAMATH_GPT_kevin_correct_answer_l171_17116


namespace NUMINAMATH_GPT_percentage_increase_in_pay_rate_l171_17109

-- Given conditions
def regular_rate : ℝ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def total_earnings : ℝ := 605

-- We need to demonstrate that the percentage increase in the pay rate for surveys involving the use of her cellphone is 30%
theorem percentage_increase_in_pay_rate :
  let earnings_at_regular_rate := regular_rate * total_surveys
  let earnings_from_cellphone_surveys := total_earnings - earnings_at_regular_rate
  let rate_per_cellphone_survey := earnings_from_cellphone_surveys / cellphone_surveys
  let increase_in_rate := rate_per_cellphone_survey - regular_rate
  let percentage_increase := (increase_in_rate / regular_rate) * 100
  percentage_increase = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_pay_rate_l171_17109


namespace NUMINAMATH_GPT_find_r_l171_17131

variable (k r : ℝ)

theorem find_r (h1 : 5 = k * 2^r) (h2 : 45 = k * 8^r) : r = (1/2) * Real.log 9 / Real.log 2 :=
sorry

end NUMINAMATH_GPT_find_r_l171_17131


namespace NUMINAMATH_GPT_male_students_plant_trees_l171_17120

theorem male_students_plant_trees (total_avg : ℕ) (female_trees : ℕ) (male_trees : ℕ) 
  (h1 : total_avg = 6) 
  (h2 : female_trees = 15)
  (h3 : 1 / (male_trees : ℝ) + 1 / (female_trees : ℝ) = 1 / (total_avg : ℝ)) : 
  male_trees = 10 := 
sorry

end NUMINAMATH_GPT_male_students_plant_trees_l171_17120


namespace NUMINAMATH_GPT_find_y_six_l171_17140

theorem find_y_six (y : ℝ) (h : y > 0) (h_eq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
    y^6 = 116 / 27 :=
by
  sorry

end NUMINAMATH_GPT_find_y_six_l171_17140


namespace NUMINAMATH_GPT_selling_price_per_unit_profit_per_unit_after_discount_l171_17152

-- Define the initial cost per unit
variable (a : ℝ)

-- Problem statement for part 1: Selling price per unit is 1.22a yuan
theorem selling_price_per_unit (a : ℝ) : 1.22 * a = a + 0.22 * a :=
by
  sorry

-- Problem statement for part 2: Profit per unit after 15% discount is still 0.037a yuan
theorem profit_per_unit_after_discount (a : ℝ) : 
  (1.22 * a * 0.85) - a = 0.037 * a :=
by
  sorry

end NUMINAMATH_GPT_selling_price_per_unit_profit_per_unit_after_discount_l171_17152


namespace NUMINAMATH_GPT_digit_sum_26_l171_17103

theorem digit_sum_26 
  (A B C D E : ℕ)
  (h1 : 1 ≤ A ∧ A ≤ 9)
  (h2 : 0 ≤ B ∧ B ≤ 9)
  (h3 : 0 ≤ C ∧ C ≤ 9)
  (h4 : 0 ≤ D ∧ D ≤ 9)
  (h5 : 0 ≤ E ∧ E ≤ 9)
  (h6 : 100000 + 10000 * A + 1000 * B + 100 * C + 10 * D + E * 3 = 100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + 1):
  A + B + C + D + E = 26 
  := 
  by
    sorry

end NUMINAMATH_GPT_digit_sum_26_l171_17103


namespace NUMINAMATH_GPT_balance_proof_l171_17115

variable (a b c : ℕ)

theorem balance_proof (h1 : 5 * a + 2 * b = 15 * c) (h2 : 2 * a = b + 3 * c) : 4 * b = 7 * c :=
sorry

end NUMINAMATH_GPT_balance_proof_l171_17115


namespace NUMINAMATH_GPT_sphere_tangent_plane_normal_line_l171_17110

variable {F : ℝ → ℝ → ℝ → ℝ}
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

def tangent_plane (x y z : ℝ) : Prop := 2*x + y + 2*z - 15 = 0

def normal_line (x y z : ℝ) : Prop := (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_plane_normal_line :
  sphere 3 (-1) 5 →
  tangent_plane 3 (-1) 5 ∧ normal_line 3 (-1) 5 :=
by
  intros h
  constructor
  sorry
  sorry

end NUMINAMATH_GPT_sphere_tangent_plane_normal_line_l171_17110


namespace NUMINAMATH_GPT_complex_number_purely_imaginary_l171_17175

theorem complex_number_purely_imaginary (a : ℝ) (i : ℂ) (h₁ : (a^2 - a - 2 : ℝ) = 0) (h₂ : (a + 1 ≠ 0)) : a = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_complex_number_purely_imaginary_l171_17175


namespace NUMINAMATH_GPT_union_sets_eq_real_l171_17114

def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

theorem union_sets_eq_real : A ∪ B = Set.univ :=
by
  sorry

end NUMINAMATH_GPT_union_sets_eq_real_l171_17114


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l171_17151

theorem minimum_value_of_quadratic :
  ∃ x : ℝ, (x = 6) ∧ (∀ y : ℝ, (y^2 - 12 * y + 32) ≥ -4) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_l171_17151


namespace NUMINAMATH_GPT_polygon_interior_angle_increase_l171_17129

theorem polygon_interior_angle_increase (n : ℕ) (h : 3 ≤ n) :
  ((n + 1 - 2) * 180 - (n - 2) * 180 = 180) :=
by sorry

end NUMINAMATH_GPT_polygon_interior_angle_increase_l171_17129


namespace NUMINAMATH_GPT_percentage_female_officers_on_duty_correct_l171_17172

-- Define the conditions
def total_officers_on_duty : ℕ := 144
def total_female_officers : ℕ := 400
def female_officers_on_duty : ℕ := total_officers_on_duty / 2

-- Define the percentage calculation
def percentage_female_officers_on_duty : ℕ :=
  (female_officers_on_duty * 100) / total_female_officers

-- The theorem that what we need to prove
theorem percentage_female_officers_on_duty_correct :
  percentage_female_officers_on_duty = 18 :=
by
  sorry

end NUMINAMATH_GPT_percentage_female_officers_on_duty_correct_l171_17172


namespace NUMINAMATH_GPT_find_smaller_number_l171_17106

theorem find_smaller_number (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 124) : x = 31 := 
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l171_17106


namespace NUMINAMATH_GPT_train_time_to_cross_tree_l171_17123

-- Definitions based on conditions
def length_of_train := 1200 -- in meters
def time_to_pass_platform := 150 -- in seconds
def length_of_platform := 300 -- in meters
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_pass_platform
def time_to_cross_tree := length_of_train / speed_of_train

-- Theorem stating the main question
theorem train_time_to_cross_tree : time_to_cross_tree = 120 := by
  sorry

end NUMINAMATH_GPT_train_time_to_cross_tree_l171_17123


namespace NUMINAMATH_GPT_problem1_l171_17126

theorem problem1 (a b : ℝ) (i : ℝ) (h : (a-2*i)*i = b-i) : a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_GPT_problem1_l171_17126


namespace NUMINAMATH_GPT_felicia_flour_amount_l171_17113

-- Define the conditions as constants
def white_sugar := 1 -- cups
def brown_sugar := 1 / 4 -- cups
def oil := 1 / 2 -- cups
def scoop := 1 / 4 -- cups
def total_scoops := 15 -- number of scoops

-- Define the proof statement
theorem felicia_flour_amount : 
  (total_scoops * scoop - (white_sugar + brown_sugar / scoop + oil / scoop)) * scoop = 2 :=
by
  sorry

end NUMINAMATH_GPT_felicia_flour_amount_l171_17113


namespace NUMINAMATH_GPT_minimum_value_f_l171_17146

open Real

noncomputable def f (x : ℝ) : ℝ :=
  x + (3 * x) / (x^2 + 3) + (x * (x + 3)) / (x^2 + 1) + (3 * (x + 1)) / (x * (x^2 + 1))

theorem minimum_value_f (x : ℝ) (hx : x > 0) : f x ≥ 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_minimum_value_f_l171_17146


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l171_17145

theorem problem_a : 37.3 / (1 / 2) = 74.6 := by
  sorry

theorem problem_b : 0.45 - (1 / 20) = 0.4 := by
  sorry

theorem problem_c : (33 / 40) * (10 / 11) = 0.75 := by
  sorry

theorem problem_d : 0.375 + (1 / 40) = 0.4 := by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l171_17145


namespace NUMINAMATH_GPT_even_decreasing_function_l171_17128

theorem even_decreasing_function (f : ℝ → ℝ) (x1 x2 : ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (hx1_neg : x1 < 0)
  (hx1x2_pos : x1 + x2 > 0) :
  f x1 < f x2 :=
sorry

end NUMINAMATH_GPT_even_decreasing_function_l171_17128


namespace NUMINAMATH_GPT_calculate_f_f_2_l171_17192

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 * x ^ 2 - 4
else if x = 0 then 2
else -1

theorem calculate_f_f_2 : f (f 2) = 188 :=
by
  sorry

end NUMINAMATH_GPT_calculate_f_f_2_l171_17192


namespace NUMINAMATH_GPT_probability_between_C_and_E_l171_17150

theorem probability_between_C_and_E
  (AB AD BC BE : ℝ)
  (h₁ : AB = 4 * AD)
  (h₂ : AB = 8 * BC)
  (h₃ : AB = 2 * BE) : 
  (AB / 2 - AB / 8) / AB = 3 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_probability_between_C_and_E_l171_17150


namespace NUMINAMATH_GPT_difference_between_max_and_min_l171_17134

noncomputable def maxThree (a b c : ℝ) : ℝ :=
  max a (max b c)

noncomputable def minThree (a b c : ℝ) : ℝ :=
  min a (min b c)

theorem difference_between_max_and_min :
  maxThree 0.12 0.23 0.22 - minThree 0.12 0.23 0.22 = 0.11 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_max_and_min_l171_17134


namespace NUMINAMATH_GPT_cone_surface_area_ratio_l171_17119

theorem cone_surface_area_ratio (l : ℝ) (h_l_pos : 0 < l) :
  let θ := (120 * Real.pi) / 180 -- converting 120 degrees to radians
  let side_area := (1/2) * l^2 * θ
  let r := l / 3
  let base_area := Real.pi * r^2
  let surface_area := side_area + base_area
  side_area ≠ 0 → 
  surface_area / side_area = 4 / 3 := 
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_cone_surface_area_ratio_l171_17119


namespace NUMINAMATH_GPT_graph_squares_count_l171_17155

theorem graph_squares_count :
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  non_diagonal_squares / 2 = 88 :=
by
  let x_intercept := 45
  let y_intercept := 5
  let total_squares := x_intercept * y_intercept
  let diagonal_squares := x_intercept + y_intercept - 1
  let non_diagonal_squares := total_squares - diagonal_squares
  have h : (non_diagonal_squares / 2 = 88) := sorry
  exact h

end NUMINAMATH_GPT_graph_squares_count_l171_17155


namespace NUMINAMATH_GPT_probability_ball_sports_l171_17156

theorem probability_ball_sports (clubs : Finset String)
  (ball_clubs : Finset String)
  (count_clubs : clubs.card = 5)
  (count_ball_clubs : ball_clubs.card = 3)
  (h1 : "basketball" ∈ clubs)
  (h2 : "soccer" ∈ clubs)
  (h3 : "volleyball" ∈ clubs)
  (h4 : "swimming" ∈ clubs)
  (h5 : "gymnastics" ∈ clubs)
  (h6 : "basketball" ∈ ball_clubs)
  (h7 : "soccer" ∈ ball_clubs)
  (h8 : "volleyball" ∈ ball_clubs) :
  (2 / ((5 : ℝ) * (4 : ℝ)) * ((3 : ℝ) * (2 : ℝ)) = (3 / 10)) :=
by
  sorry

end NUMINAMATH_GPT_probability_ball_sports_l171_17156


namespace NUMINAMATH_GPT_Pyarelal_loss_is_1800_l171_17159

noncomputable def Ashok_and_Pyarelal_loss (P L : ℝ) : Prop :=
  let Ashok_cap := (1 / 9) * P
  let total_cap := P + Ashok_cap
  let Pyarelal_ratio := P / total_cap
  let total_loss := 2000
  let Pyarelal_loss := Pyarelal_ratio * total_loss
  Pyarelal_loss = 1800

theorem Pyarelal_loss_is_1800 (P : ℝ) (h1 : P > 0) (h2 : L = 2000) :
  Ashok_and_Pyarelal_loss P L := sorry

end NUMINAMATH_GPT_Pyarelal_loss_is_1800_l171_17159


namespace NUMINAMATH_GPT_part_a_part_b_l171_17199

-- Part (a)
theorem part_a (n : ℕ) (a b : ℝ) : 
  a^(n+1) + b^(n+1) = (a + b) * (a^n + b^n) - a * b * (a^(n - 1) + b^(n - 1)) :=
by sorry

-- Part (b)
theorem part_b {a b : ℝ} (h1 : a + b = 1) (h2: a * b = -1) : 
  a^10 + b^10 = 123 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l171_17199


namespace NUMINAMATH_GPT_AC_plus_third_BA_l171_17142

def point := (ℝ × ℝ)

def A : point := (2, 4)
def B : point := (-1, -5)
def C : point := (3, -2)

noncomputable def vec (p₁ p₂ : point) : point :=
  (p₂.1 - p₁.1, p₂.2 - p₁.2)

noncomputable def scal_mult (scalar : ℝ) (v : point) : point :=
  (scalar * v.1, scalar * v.2)

noncomputable def vec_add (v₁ v₂ : point) : point :=
  (v₁.1 + v₂.1, v₁.2 + v₂.2)

theorem AC_plus_third_BA : 
  vec_add (vec A C) (scal_mult (1 / 3) (vec B A)) = (2, -3) :=
by
  sorry

end NUMINAMATH_GPT_AC_plus_third_BA_l171_17142


namespace NUMINAMATH_GPT_martin_improved_lap_time_l171_17138

def initial_laps := 15
def initial_time := 45 -- in minutes
def final_laps := 18
def final_time := 42 -- in minutes

noncomputable def initial_lap_time := initial_time / initial_laps
noncomputable def final_lap_time := final_time / final_laps
noncomputable def improvement := initial_lap_time - final_lap_time

theorem martin_improved_lap_time : improvement = 2 / 3 := by 
  sorry

end NUMINAMATH_GPT_martin_improved_lap_time_l171_17138


namespace NUMINAMATH_GPT_lucy_groceries_total_l171_17143

theorem lucy_groceries_total (cookies noodles : ℕ) (h1 : cookies = 12) (h2 : noodles = 16) : cookies + noodles = 28 :=
by
  sorry

end NUMINAMATH_GPT_lucy_groceries_total_l171_17143


namespace NUMINAMATH_GPT_compare_fractions_difference_l171_17196

theorem compare_fractions_difference :
  let a := (1 : ℝ) / 2
  let b := (1 : ℝ) / 3
  a - b = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_compare_fractions_difference_l171_17196


namespace NUMINAMATH_GPT_factor_x4_plus_81_l171_17162

theorem factor_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) :=
by 
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_factor_x4_plus_81_l171_17162


namespace NUMINAMATH_GPT_sum_of_underlined_numbers_non_negative_l171_17112

-- Definitions used in the problem
def is_positive (n : Int) : Prop := n > 0
def underlined (nums : List Int) : List Int := sorry -- Define underlining based on conditions

def sum_of_underlined_numbers (nums : List Int) : Int :=
  (underlined nums).sum

-- The proof problem statement
theorem sum_of_underlined_numbers_non_negative
  (nums : List Int)
  (h_len : nums.length = 100) :
  0 < sum_of_underlined_numbers nums := sorry

end NUMINAMATH_GPT_sum_of_underlined_numbers_non_negative_l171_17112


namespace NUMINAMATH_GPT_sum_x_y_eq_2_l171_17178

open Real

theorem sum_x_y_eq_2 (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_x_y_eq_2_l171_17178


namespace NUMINAMATH_GPT_roads_with_five_possible_roads_with_four_not_possible_l171_17182

-- Problem (a)
theorem roads_with_five_possible :
  ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 5) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

-- Problem (b)
theorem roads_with_four_not_possible :
  ¬ ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 4) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

end NUMINAMATH_GPT_roads_with_five_possible_roads_with_four_not_possible_l171_17182


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l171_17153

theorem sum_of_squares_of_roots : 
  (∃ (a b c d : ℝ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 15 * x^2 + 56 = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
    (a^2 + b^2 + c^2 + d^2 = 30)) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l171_17153


namespace NUMINAMATH_GPT_arccos_of_sqrt3_div_2_l171_17148

theorem arccos_of_sqrt3_div_2 : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_arccos_of_sqrt3_div_2_l171_17148


namespace NUMINAMATH_GPT_Mrs_Hilt_walks_to_fountain_l171_17133

theorem Mrs_Hilt_walks_to_fountain :
  ∀ (distance trips : ℕ), distance = 30 → trips = 4 → distance * trips = 120 :=
by
  intros distance trips h_distance h_trips
  sorry

end NUMINAMATH_GPT_Mrs_Hilt_walks_to_fountain_l171_17133


namespace NUMINAMATH_GPT_inscribed_square_area_l171_17191

noncomputable def area_inscribed_square (AB CD : ℕ) (BCFE : ℕ) : Prop :=
  AB = 36 ∧ CD = 64 ∧ BCFE = (AB * CD)

theorem inscribed_square_area :
  ∀ (AB CD : ℕ),
  area_inscribed_square AB CD 2304 :=
by
  intros
  sorry

end NUMINAMATH_GPT_inscribed_square_area_l171_17191


namespace NUMINAMATH_GPT_purple_chip_value_l171_17104

theorem purple_chip_value 
  (x : ℕ)
  (blue_chip_value : 1 = 1)
  (green_chip_value : 5 = 5)
  (red_chip_value : 11 = 11)
  (purple_chip_condition1 : x > 5)
  (purple_chip_condition2 : x < 11)
  (product_of_points : ∃ b g p r, (b = 1 ∨ b = 1) ∧ (g = 5 ∨ g = 5) ∧ (p = x ∨ p = x) ∧ (r = 11 ∨ r = 11) ∧ b * g * p * r = 28160) : 
  x = 7 :=
sorry

end NUMINAMATH_GPT_purple_chip_value_l171_17104


namespace NUMINAMATH_GPT_tangent_line_length_l171_17118

noncomputable def curve_C (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

def problem_conditions : Prop :=
  curve_C 0 = 4 ∧ cartesian 4 0 = (4, 0)

theorem tangent_line_length :
  problem_conditions → 
  ∃ l : ℝ, l = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_length_l171_17118


namespace NUMINAMATH_GPT_square_diagonal_l171_17183

theorem square_diagonal (s d : ℝ) (h : 4 * s = 40) : d = s * Real.sqrt 2 → d = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_square_diagonal_l171_17183


namespace NUMINAMATH_GPT_b_profit_share_l171_17198

theorem b_profit_share (total_capital : ℝ) (profit : ℝ) (A_invest : ℝ) (B_invest : ℝ) (C_invest : ℝ) (D_invest : ℝ)
 (A_time : ℝ) (B_time : ℝ) (C_time : ℝ) (D_time : ℝ) :
  total_capital = 100000 ∧
  A_invest = B_invest + 10000 ∧
  B_invest = C_invest + 5000 ∧
  D_invest = A_invest + 8000 ∧
  A_time = 12 ∧
  B_time = 10 ∧
  C_time = 8 ∧
  D_time = 6 ∧
  profit = 50000 →
  (B_invest * B_time / (A_invest * A_time + B_invest * B_time + C_invest * C_time + D_invest * D_time)) * profit = 10925 :=
by
  sorry

end NUMINAMATH_GPT_b_profit_share_l171_17198


namespace NUMINAMATH_GPT_irreducible_fraction_l171_17111

-- Definition of gcd
def my_gcd (m n : Int) : Int :=
  gcd m n

-- Statement of the problem
theorem irreducible_fraction (a : Int) : my_gcd (a^3 + 2 * a) (a^4 + 3 * a^2 + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fraction_l171_17111


namespace NUMINAMATH_GPT_abs_inequality_solution_l171_17154

theorem abs_inequality_solution (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) := 
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l171_17154


namespace NUMINAMATH_GPT_certain_number_l171_17122

theorem certain_number (n : ℕ) : 
  (55 * 57) % n = 6 ∧ n = 1043 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_l171_17122


namespace NUMINAMATH_GPT_simplify_polynomial_l171_17184

theorem simplify_polynomial (x : ℤ) :
  (3 * x - 2) * (6 * x^12 + 3 * x^11 + 5 * x^9 + x^8 + 7 * x^7) =
  18 * x^13 - 3 * x^12 + 15 * x^10 - 7 * x^9 + 19 * x^8 - 14 * x^7 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l171_17184


namespace NUMINAMATH_GPT_parabola_vertex_and_point_l171_17177

theorem parabola_vertex_and_point (a b c : ℝ) : 
  (∀ x, y = a * x^2 + b * x + c) ∧ 
  ∃ x y, (y = a * (x - 4)^2 + 3) → 
  (a * 2^2 + b * 2 + c = 5) → 
  (a = 1/2 ∧ b = -4 ∧ c = 11) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_and_point_l171_17177


namespace NUMINAMATH_GPT_trees_died_l171_17124

theorem trees_died 
  (original_trees : ℕ) 
  (cut_trees : ℕ) 
  (remaining_trees : ℕ) 
  (died_trees : ℕ)
  (h1 : original_trees = 86)
  (h2 : cut_trees = 23)
  (h3 : remaining_trees = 48)
  (h4 : original_trees - died_trees - cut_trees = remaining_trees) : 
  died_trees = 15 :=
by
  sorry

end NUMINAMATH_GPT_trees_died_l171_17124


namespace NUMINAMATH_GPT_train_stoppage_time_l171_17107

-- Definitions of the conditions
def speed_excluding_stoppages : ℝ := 48 -- in kmph
def speed_including_stoppages : ℝ := 32 -- in kmph
def time_per_hour : ℝ := 60 -- 60 minutes in an hour

-- The problem statement
theorem train_stoppage_time :
  (speed_excluding_stoppages - speed_including_stoppages) * time_per_hour / speed_excluding_stoppages = 20 :=
by
  -- Initial statement
  sorry

end NUMINAMATH_GPT_train_stoppage_time_l171_17107


namespace NUMINAMATH_GPT_point_B_not_on_curve_C_l171_17169

theorem point_B_not_on_curve_C {a : ℝ} : 
  ¬ ((2 * a) ^ 2 + (4 * a) ^ 2 + 6 * a * (2 * a) - 8 * a * (4 * a) = 0) :=
by 
  sorry

end NUMINAMATH_GPT_point_B_not_on_curve_C_l171_17169


namespace NUMINAMATH_GPT_estimate_fitness_population_l171_17174

theorem estimate_fitness_population :
  ∀ (sample_size total_population : ℕ) (sample_met_standards : Nat) (percentage_met_standards estimated_met_standards : ℝ),
  sample_size = 1000 →
  total_population = 1200000 →
  sample_met_standards = 950 →
  percentage_met_standards = (sample_met_standards : ℝ) / (sample_size : ℝ) →
  estimated_met_standards = percentage_met_standards * (total_population : ℝ) →
  estimated_met_standards = 1140000 := by sorry

end NUMINAMATH_GPT_estimate_fitness_population_l171_17174


namespace NUMINAMATH_GPT_cheesecake_factory_savings_l171_17187

noncomputable def combined_savings : ℕ := 3000

theorem cheesecake_factory_savings :
  let hourly_wage := 10
  let daily_hours := 10
  let working_days := 5
  let weekly_hours := daily_hours * working_days
  let weekly_salary := weekly_hours * hourly_wage
  let robby_savings := (2/5 : ℚ) * weekly_salary
  let jaylen_savings := (3/5 : ℚ) * weekly_salary
  let miranda_savings := (1/2 : ℚ) * weekly_salary
  let combined_weekly_savings := robby_savings + jaylen_savings + miranda_savings
  4 * combined_weekly_savings = combined_savings :=
by
  sorry

end NUMINAMATH_GPT_cheesecake_factory_savings_l171_17187


namespace NUMINAMATH_GPT_find_n_l171_17165

theorem find_n (N : ℕ) (hN : 10 ≤ N ∧ N < 100)
  (h : ∃ a b : ℕ, N = 10 * a + b ∧ 4 * a + 2 * b = N / 2) : 
  N = 32 ∨ N = 64 ∨ N = 96 :=
sorry

end NUMINAMATH_GPT_find_n_l171_17165


namespace NUMINAMATH_GPT_polynomials_common_zero_k_l171_17164

theorem polynomials_common_zero_k
  (k : ℝ) :
  (∃ x : ℝ, (1988 * x^2 + k * x + 8891 = 0) ∧ (8891 * x^2 + k * x + 1988 = 0)) ↔ (k = 10879 ∨ k = -10879) :=
sorry

end NUMINAMATH_GPT_polynomials_common_zero_k_l171_17164


namespace NUMINAMATH_GPT_flower_bee_relationship_l171_17130

def numberOfBees (flowers : ℕ) (fewer_bees : ℕ) : ℕ :=
  flowers - fewer_bees

theorem flower_bee_relationship :
  numberOfBees 5 2 = 3 := by
  sorry

end NUMINAMATH_GPT_flower_bee_relationship_l171_17130


namespace NUMINAMATH_GPT_quadratic_inequality_always_holds_l171_17193

theorem quadratic_inequality_always_holds (k : ℝ) (h : ∀ x : ℝ, (x^2 - k*x + 1) > 0) : -2 < k ∧ k < 2 :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_always_holds_l171_17193


namespace NUMINAMATH_GPT_at_least_one_not_less_than_six_l171_17117

-- Definitions for the conditions.
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The proof statement.
theorem at_least_one_not_less_than_six :
  (a + 4 / b) < 6 ∧ (b + 9 / c) < 6 ∧ (c + 16 / a) < 6 → false :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_six_l171_17117


namespace NUMINAMATH_GPT_base_conversion_to_zero_l171_17121

theorem base_conversion_to_zero (A B : ℕ) (hA : 0 ≤ A ∧ A < 12) (hB : 0 ≤ B ∧ B < 5) 
    (h1 : 12 * A + B = 5 * B + A) : 12 * A + B = 0 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_to_zero_l171_17121


namespace NUMINAMATH_GPT_compute_expr_l171_17195

theorem compute_expr : 5^2 - 3 * 4 + 3^2 = 22 := by
  sorry

end NUMINAMATH_GPT_compute_expr_l171_17195


namespace NUMINAMATH_GPT_width_of_rectangular_plot_l171_17179

theorem width_of_rectangular_plot 
  (length : ℝ) 
  (poles : ℕ) 
  (distance_between_poles : ℝ) 
  (num_poles : ℕ) 
  (total_wire_length : ℝ) 
  (perimeter : ℝ) 
  (width : ℝ) :
  length = 90 ∧ 
  distance_between_poles = 5 ∧ 
  num_poles = 56 ∧ 
  total_wire_length = (num_poles - 1) * distance_between_poles ∧ 
  total_wire_length = 275 ∧ 
  perimeter = 2 * (length + width) 
  → width = 47.5 :=
by
  sorry

end NUMINAMATH_GPT_width_of_rectangular_plot_l171_17179


namespace NUMINAMATH_GPT_total_floor_area_covered_l171_17135

theorem total_floor_area_covered (A B C : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : B = 24) 
  (h3 : C = 19) : 
  A - (B - C) - 2 * C = 138 := 
by sorry

end NUMINAMATH_GPT_total_floor_area_covered_l171_17135


namespace NUMINAMATH_GPT_parabola_focus_distance_l171_17101

theorem parabola_focus_distance (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 100) : x = 9 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l171_17101


namespace NUMINAMATH_GPT_volume_formula_l171_17125

noncomputable def volume_of_parallelepiped
  (a b : ℝ) (h : ℝ) (θ : ℝ) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ)
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2)) : ℝ :=
  a * b * h 

theorem volume_formula 
  (a b : ℝ) (h : ℝ) (θ : ℝ)
  (area_base : ℝ) 
  (area_of_base_eq : area_base = a * b) 
  (θ_eq : θ = Real.pi / 3) 
  (base_diagonal : ℝ) 
  (base_diagonal_eq : base_diagonal = Real.sqrt (a ^ 2 + b ^ 2))
  (height_eq : h = (base_diagonal / 2) * (Real.sqrt 3)): 
  volume_of_parallelepiped a b h θ θ_eq base_diagonal base_diagonal_eq 
  = (144 * Real.sqrt 3) / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_formula_l171_17125


namespace NUMINAMATH_GPT_mabel_tomatoes_l171_17180

theorem mabel_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 := by
  sorry

end NUMINAMATH_GPT_mabel_tomatoes_l171_17180


namespace NUMINAMATH_GPT_probability_heart_or_king_l171_17167

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end NUMINAMATH_GPT_probability_heart_or_king_l171_17167


namespace NUMINAMATH_GPT_carla_sharpening_time_l171_17139

theorem carla_sharpening_time (x : ℕ) (h : x + 3 * x = 40) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_carla_sharpening_time_l171_17139


namespace NUMINAMATH_GPT_work_speed_ratio_l171_17181

open Real

theorem work_speed_ratio (A B : Type) 
  (A_work_speed B_work_speed : ℝ) 
  (combined_work_time : ℝ) 
  (B_work_time : ℝ)
  (h_combined : combined_work_time = 12)
  (h_B : B_work_time = 36)
  (combined_speed : A_work_speed + B_work_speed = 1 / combined_work_time)
  (B_speed : B_work_speed = 1 / B_work_time) :
  A_work_speed / B_work_speed = 2 :=
by sorry

end NUMINAMATH_GPT_work_speed_ratio_l171_17181


namespace NUMINAMATH_GPT_marco_older_than_twice_marie_l171_17158

variable (M m x : ℕ)

def marie_age : ℕ := 12
def sum_of_ages : ℕ := 37

theorem marco_older_than_twice_marie :
  m = marie_age → (M = 2 * m + x) → (M + m = sum_of_ages) → x = 1 :=
by
  intros h1 h2 h3
  rw [h1] at h2 h3
  sorry

end NUMINAMATH_GPT_marco_older_than_twice_marie_l171_17158


namespace NUMINAMATH_GPT_initial_paint_l171_17166

variable (total_needed : ℕ) (paint_bought : ℕ) (still_needed : ℕ)

theorem initial_paint (h_total_needed : total_needed = 70)
                      (h_paint_bought : paint_bought = 23)
                      (h_still_needed : still_needed = 11) : 
                      ∃ x : ℕ, x = 36 :=
by
  sorry

end NUMINAMATH_GPT_initial_paint_l171_17166


namespace NUMINAMATH_GPT_find_stadium_width_l171_17132

-- Conditions
def stadium_length : ℝ := 24
def stadium_height : ℝ := 16
def longest_pole : ℝ := 34

-- Width to be solved
def stadium_width : ℝ := 18

-- Theorem stating that given the conditions, the width must be 18
theorem find_stadium_width :
  stadium_length^2 + stadium_width^2 + stadium_height^2 = longest_pole^2 :=
by
  sorry

end NUMINAMATH_GPT_find_stadium_width_l171_17132


namespace NUMINAMATH_GPT_radius_of_circle_l171_17186

theorem radius_of_circle (r x y : ℝ): 
  x = π * r^2 → 
  y = 2 * π * r → 
  x - y = 72 * π → 
  r = 12 := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_l171_17186


namespace NUMINAMATH_GPT_Amanda_notebooks_l171_17157

theorem Amanda_notebooks (initial ordered lost final : ℕ) 
  (h_initial: initial = 65) 
  (h_ordered: ordered = 23) 
  (h_lost: lost = 14) : 
  final = 74 := 
by 
  -- calculation and proof will go here
  sorry 

end NUMINAMATH_GPT_Amanda_notebooks_l171_17157


namespace NUMINAMATH_GPT_second_discount_percentage_l171_17185

-- Defining the variables
variables (P S : ℝ) (d1 d2 : ℝ)

-- Given conditions
def original_price : P = 200 := by sorry
def sale_price_after_initial_discount : S = 171 := by sorry
def first_discount_rate : d1 = 0.10 := by sorry

-- Required to prove
theorem second_discount_percentage :
  ∃ d2, (d2 = 0.05) :=
sorry

end NUMINAMATH_GPT_second_discount_percentage_l171_17185


namespace NUMINAMATH_GPT_trip_time_total_l171_17189

noncomputable def wrong_direction_time : ℝ := 75 / 60
noncomputable def return_time : ℝ := 75 / 45
noncomputable def normal_trip_time : ℝ := 250 / 45

theorem trip_time_total :
  wrong_direction_time + return_time + normal_trip_time = 8.48 := by
  sorry

end NUMINAMATH_GPT_trip_time_total_l171_17189


namespace NUMINAMATH_GPT_cost_price_l171_17108

-- Given conditions
variable (x : ℝ)
def profit (x : ℝ) : ℝ := 54 - x
def loss (x : ℝ) : ℝ := x - 40

-- Claim
theorem cost_price (h : profit x = loss x) : x = 47 :=
by {
  -- This is where the proof would go
  sorry
}

end NUMINAMATH_GPT_cost_price_l171_17108
