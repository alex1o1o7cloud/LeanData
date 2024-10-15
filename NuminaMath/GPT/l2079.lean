import Mathlib

namespace NUMINAMATH_GPT_complement_intersection_example_l2079_207937

open Set

variable (U A B : Set ℕ)

def C_U (A : Set ℕ) (U : Set ℕ) : Set ℕ := U \ A

theorem complement_intersection_example 
  (hU : U = {0, 1, 2, 3})
  (hA : A = {0, 1})
  (hB : B = {1, 2, 3}) :
  (C_U A U) ∩ B = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_example_l2079_207937


namespace NUMINAMATH_GPT_bea_glasses_sold_is_10_l2079_207909

variable (B : ℕ)
variable (earnings_bea earnings_dawn : ℕ)

def bea_price_per_glass := 25
def dawn_price_per_glass := 28
def dawn_glasses_sold := 8
def earnings_diff := 26

def bea_earnings := bea_price_per_glass * B
def dawn_earnings := dawn_price_per_glass * dawn_glasses_sold

def bea_earnings_greater := bea_earnings = dawn_earnings + earnings_diff

theorem bea_glasses_sold_is_10 (h : bea_earnings_greater) : B = 10 :=
by sorry

end NUMINAMATH_GPT_bea_glasses_sold_is_10_l2079_207909


namespace NUMINAMATH_GPT_eval_expr_x_eq_3_y_eq_4_l2079_207918

theorem eval_expr_x_eq_3_y_eq_4 : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 5 * x^y + 6 * y^x + x * y = 801 := 
by 
  intros x y hx hy 
  rw [hx, hy]
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_eval_expr_x_eq_3_y_eq_4_l2079_207918


namespace NUMINAMATH_GPT_distinct_ordered_pairs_count_l2079_207985

theorem distinct_ordered_pairs_count : 
  ∃ (n : ℕ), (∀ (a b : ℕ), a + b = 50 → 0 ≤ a ∧ 0 ≤ b) ∧ n = 51 :=
by
  sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_count_l2079_207985


namespace NUMINAMATH_GPT_sin_cos_identity_proof_l2079_207939

noncomputable def solution : ℝ := Real.sin (Real.pi / 6) * Real.cos (Real.pi / 12) + Real.cos (Real.pi / 6) * Real.sin (Real.pi / 12)

theorem sin_cos_identity_proof : solution = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_proof_l2079_207939


namespace NUMINAMATH_GPT_red_jellybeans_count_l2079_207946

theorem red_jellybeans_count (total_jellybeans : ℕ)
  (blue_jellybeans : ℕ)
  (purple_jellybeans : ℕ)
  (orange_jellybeans : ℕ)
  (H1 : total_jellybeans = 200)
  (H2 : blue_jellybeans = 14)
  (H3 : purple_jellybeans = 26)
  (H4 : orange_jellybeans = 40) :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 :=
by sorry

end NUMINAMATH_GPT_red_jellybeans_count_l2079_207946


namespace NUMINAMATH_GPT_perpendicular_lines_b_value_l2079_207912

theorem perpendicular_lines_b_value 
  (b : ℝ) 
  (line1 : ∀ x y : ℝ, x + 3 * y + 5 = 0 → True) 
  (line2 : ∀ x y : ℝ, b * x + 3 * y + 5 = 0 → True)
  (perpendicular_condition : (-1 / 3) * (-b / 3) = -1) : 
  b = -9 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_b_value_l2079_207912


namespace NUMINAMATH_GPT_intersection_complement_l2079_207943

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement :
  A ∩ (compl B) = {x | 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2079_207943


namespace NUMINAMATH_GPT_new_salary_l2079_207949

theorem new_salary (increase : ℝ) (percent_increase : ℝ) (S_new : ℝ) :
  increase = 25000 → percent_increase = 38.46153846153846 → S_new = 90000 :=
by
  sorry

end NUMINAMATH_GPT_new_salary_l2079_207949


namespace NUMINAMATH_GPT_value_of_expression_l2079_207930

open Polynomial

theorem value_of_expression (a b : ℚ) (h1 : (3 : ℚ) * a ^ 2 + 9 * a - 21 = 0) (h2 : (3 : ℚ) * b ^ 2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (2 * b - 2) = -4 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l2079_207930


namespace NUMINAMATH_GPT_range_of_values_abs_range_of_values_l2079_207956

noncomputable def problem (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1

theorem range_of_values (x y : ℝ) (h : problem x y) :
  2 ≤ (2 * x + y - 1) / x ∧ (2 * x + y - 1) / x ≤ 10 / 3 :=
sorry

theorem abs_range_of_values (x y : ℝ) (h : problem x y) :
  5 - Real.sqrt 2 ≤ abs (x + y + 1) ∧ abs (x + y + 1) ≤ 5 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_values_abs_range_of_values_l2079_207956


namespace NUMINAMATH_GPT_length_of_living_room_l2079_207969

theorem length_of_living_room (L : ℝ) (width : ℝ) (border_width : ℝ) (border_area : ℝ) 
  (h1 : width = 10)
  (h2 : border_width = 2)
  (h3 : border_area = 72) :
  L = 12 :=
by
  sorry

end NUMINAMATH_GPT_length_of_living_room_l2079_207969


namespace NUMINAMATH_GPT_fraction_to_decimal_subtraction_l2079_207981

theorem fraction_to_decimal_subtraction 
    (h : (3 : ℚ) / 40 = 0.075) : 
    0.075 - 0.005 = 0.070 := 
by 
    sorry

end NUMINAMATH_GPT_fraction_to_decimal_subtraction_l2079_207981


namespace NUMINAMATH_GPT_find_fraction_l2079_207965

theorem find_fraction : 
  ∀ (x : ℚ), (120 - x * 125 = 45) → x = 3 / 5 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_find_fraction_l2079_207965


namespace NUMINAMATH_GPT_maximum_possible_value_of_x_l2079_207990

-- Define the conditions and the question
def ten_teams_playing_each_other_once (number_of_teams : ℕ) : Prop :=
  number_of_teams = 10

def points_system (win_points draw_points loss_points : ℕ) : Prop :=
  win_points = 3 ∧ draw_points = 1 ∧ loss_points = 0

def max_points_per_team (x : ℕ) : Prop :=
  x = 13

-- The theorem to be proved: maximum possible value of x given the conditions
theorem maximum_possible_value_of_x :
  ∀ (number_of_teams win_points draw_points loss_points x : ℕ),
    ten_teams_playing_each_other_once number_of_teams →
    points_system win_points draw_points loss_points →
    max_points_per_team x :=
  sorry

end NUMINAMATH_GPT_maximum_possible_value_of_x_l2079_207990


namespace NUMINAMATH_GPT_total_stoppage_time_per_hour_l2079_207933

variables (speed_ex_stoppages_1 speed_in_stoppages_1 : ℕ)
variables (speed_ex_stoppages_2 speed_in_stoppages_2 : ℕ)
variables (speed_ex_stoppages_3 speed_in_stoppages_3 : ℕ)

-- Definitions of the speeds given in the problem's conditions.
def speed_bus_1_ex_stoppages := 54
def speed_bus_1_in_stoppages := 36
def speed_bus_2_ex_stoppages := 60
def speed_bus_2_in_stoppages := 40
def speed_bus_3_ex_stoppages := 72
def speed_bus_3_in_stoppages := 48

-- The main theorem to be proved.
theorem total_stoppage_time_per_hour :
  ((1 - speed_bus_1_in_stoppages / speed_bus_1_ex_stoppages : ℚ)
   + (1 - speed_bus_2_in_stoppages / speed_bus_2_ex_stoppages : ℚ)
   + (1 - speed_bus_3_in_stoppages / speed_bus_3_ex_stoppages : ℚ)) = 1 := by
  sorry

end NUMINAMATH_GPT_total_stoppage_time_per_hour_l2079_207933


namespace NUMINAMATH_GPT_height_of_spheres_l2079_207926

theorem height_of_spheres (R r : ℝ) (h : ℝ) :
  0 < r ∧ r < R → h = R - Real.sqrt ((3 * R^2 - 6 * R * r - r^2) / 3) :=
by
  intros h0
  sorry

end NUMINAMATH_GPT_height_of_spheres_l2079_207926


namespace NUMINAMATH_GPT_eighth_grade_girls_l2079_207974

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end NUMINAMATH_GPT_eighth_grade_girls_l2079_207974


namespace NUMINAMATH_GPT_pie_difference_l2079_207923

theorem pie_difference (s1 s3 : ℚ) (h1 : s1 = 7/8) (h3 : s3 = 3/4) :
  s1 - s3 = 1/8 :=
by
  sorry

end NUMINAMATH_GPT_pie_difference_l2079_207923


namespace NUMINAMATH_GPT_combined_savings_after_5_years_l2079_207921

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + (r / n)) ^ (n * t)

theorem combined_savings_after_5_years :
  let P1 := 600
  let r1 := 0.10
  let n1 := 12
  let t := 5
  let P2 := 400
  let r2 := 0.08
  let n2 := 4
  compound_interest P1 r1 n1 t + compound_interest P2 r2 n2 t = 1554.998 :=
by
  sorry

end NUMINAMATH_GPT_combined_savings_after_5_years_l2079_207921


namespace NUMINAMATH_GPT_annual_decrease_rate_l2079_207945

theorem annual_decrease_rate (r : ℝ) 
  (h1 : 15000 * (1 - r / 100)^2 = 9600) : 
  r = 20 := 
sorry

end NUMINAMATH_GPT_annual_decrease_rate_l2079_207945


namespace NUMINAMATH_GPT_lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l2079_207995

-- Define a polynomial and conditions for divisibility by 7
def poly_deg_6 (a b c d e f g x : ℤ) : ℤ :=
  a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g

-- Theorem for divisibility by 7
theorem lowest_degree_for_divisibility_by_7 : 
  (∀ x : ℤ, poly_deg_6 a b c d e f g x % 7 = 0) → false :=
sorry

-- Define a polynomial and conditions for divisibility by 12
def poly_deg_3 (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

-- Theorem for divisibility by 12
theorem lowest_degree_for_divisibility_by_12 : 
  (∀ x : ℤ, poly_deg_3 a b c d x % 12 = 0) → false :=
sorry

end NUMINAMATH_GPT_lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l2079_207995


namespace NUMINAMATH_GPT_toy_cost_price_l2079_207935

theorem toy_cost_price (x : ℝ) (h : 1.5 * x * 0.8 - x = 20) : x = 100 := 
sorry

end NUMINAMATH_GPT_toy_cost_price_l2079_207935


namespace NUMINAMATH_GPT_event_distance_l2079_207905

noncomputable def distance_to_event (cost_per_mile : ℝ) (days : ℕ) (rides_per_day : ℕ) (total_cost : ℝ) : ℝ :=
  total_cost / (days * rides_per_day * cost_per_mile)

theorem event_distance 
  (cost_per_mile : ℝ)
  (days : ℕ)
  (rides_per_day : ℕ)
  (total_cost : ℝ)
  (h1 : cost_per_mile = 2.5)
  (h2 : days = 7)
  (h3 : rides_per_day = 2)
  (h4 : total_cost = 7000) : 
  distance_to_event cost_per_mile days rides_per_day total_cost = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_event_distance_l2079_207905


namespace NUMINAMATH_GPT_second_container_mass_l2079_207957

-- Given conditions
def height1 := 4 -- height of first container in cm
def width1 := 2 -- width of first container in cm
def length1 := 8 -- length of first container in cm
def mass1 := 64 -- mass of material the first container can hold in grams

def height2 := 3 * height1 -- height of second container in cm
def width2 := 2 * width1 -- width of second container in cm
def length2 := length1 -- length of second container in cm

def volume (height width length : ℤ) : ℤ := height * width * length

-- The proof statement
theorem second_container_mass : volume height2 width2 length2 = 6 * volume height1 width1 length1 → 6 * mass1 = 384 :=
by
  sorry

end NUMINAMATH_GPT_second_container_mass_l2079_207957


namespace NUMINAMATH_GPT_find_common_difference_l2079_207936

variable {α : Type*} [LinearOrderedField α]

-- Define the properties of the arithmetic sequence
def arithmetic_sum (a1 d : α) (n : ℕ) : α := n * a1 + (n * (n - 1) * d) / 2

variables (a1 d : α) -- First term and common difference of the arithmetic sequence (to be found)
variable (S : ℕ → α) -- Sum of the first n terms of the arithmetic sequence

-- Conditions given in the problem
axiom sum_3_eq_6 : S 3 = 6
axiom term_3_eq_4 : a1 + 2 * d = 4

-- The question translated into a theorem statement that the common difference is 2
theorem find_common_difference : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l2079_207936


namespace NUMINAMATH_GPT_remainder_when_divided_by_20_l2079_207907

theorem remainder_when_divided_by_20 (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_20_l2079_207907


namespace NUMINAMATH_GPT_equation_one_solutions_equation_two_solutions_l2079_207972

theorem equation_one_solutions (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := 
by {
  sorry
}

theorem equation_two_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_equation_one_solutions_equation_two_solutions_l2079_207972


namespace NUMINAMATH_GPT_green_and_yellow_peaches_total_is_correct_l2079_207976

-- Define the number of red, yellow, and green peaches
def red_peaches : ℕ := 5
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6

-- Definition of the total number of green and yellow peaches
def total_green_and_yellow_peaches : ℕ := green_peaches + yellow_peaches

-- Theorem stating that the total number of green and yellow peaches is 20
theorem green_and_yellow_peaches_total_is_correct : total_green_and_yellow_peaches = 20 :=
by 
  sorry

end NUMINAMATH_GPT_green_and_yellow_peaches_total_is_correct_l2079_207976


namespace NUMINAMATH_GPT_part1_part2_l2079_207962

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1): Given m = 4, prove A ∪ B = {x | -2 ≤ x ∧ x ≤ 7}
theorem part1 : A ∪ B 4 = {x | -2 ≤ x ∧ x ≤ 7} :=
by
  sorry

-- Part (2): Given B ⊆ A, prove m ∈ (-∞, 3]
theorem part2 {m : ℝ} (h : B m ⊆ A) : m ∈ Set.Iic 3 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2079_207962


namespace NUMINAMATH_GPT_cube_modulo_9_l2079_207960

theorem cube_modulo_9 (N : ℤ) (h : N % 9 = 2 ∨ N % 9 = 5 ∨ N % 9 = 8) : 
  (N^3) % 9 = 8 :=
by sorry

end NUMINAMATH_GPT_cube_modulo_9_l2079_207960


namespace NUMINAMATH_GPT_church_distance_l2079_207970

def distance_to_church (speed : ℕ) (hourly_rate : ℕ) (flat_fee : ℕ) (total_paid : ℕ) : ℕ :=
  let hours := (total_paid - flat_fee) / hourly_rate
  hours * speed

theorem church_distance :
  distance_to_church 10 30 20 80 = 20 :=
by
  sorry

end NUMINAMATH_GPT_church_distance_l2079_207970


namespace NUMINAMATH_GPT_total_cost_div_selling_price_eq_23_div_13_l2079_207964

-- Conditions from part (a)
def pencil_count := 140
def pen_count := 90
def eraser_count := 60

def loss_pencils := 70
def loss_pens := 30
def loss_erasers := 20

def pen_cost (P : ℝ) := P
def pencil_cost (P : ℝ) := 2 * P
def eraser_cost (P : ℝ) := 1.5 * P

def total_cost (P : ℝ) :=
  pencil_count * pencil_cost P +
  pen_count * pen_cost P +
  eraser_count * eraser_cost P

def loss (P : ℝ) :=
  loss_pencils * pencil_cost P +
  loss_pens * pen_cost P +
  loss_erasers * eraser_cost P

def selling_price (P : ℝ) :=
  total_cost P - loss P

-- Statement to be proved: the total cost is 23/13 times the selling price.
theorem total_cost_div_selling_price_eq_23_div_13 (P : ℝ) :
  total_cost P / selling_price P = 23 / 13 := by
  sorry

end NUMINAMATH_GPT_total_cost_div_selling_price_eq_23_div_13_l2079_207964


namespace NUMINAMATH_GPT_product_is_square_of_24975_l2079_207998

theorem product_is_square_of_24975 : (500 * 49.95 * 4.995 * 5000 : ℝ) = (24975 : ℝ) ^ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_is_square_of_24975_l2079_207998


namespace NUMINAMATH_GPT_volume_rectangular_box_l2079_207996

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end NUMINAMATH_GPT_volume_rectangular_box_l2079_207996


namespace NUMINAMATH_GPT_prize_expectation_l2079_207954

theorem prize_expectation :
  let total_people := 100
  let envelope_percentage := 0.4
  let grand_prize_prob := 0.1
  let second_prize_prob := 0.2
  let consolation_prize_prob := 0.3
  let people_with_envelopes := total_people * envelope_percentage
  let grand_prize_winners := people_with_envelopes * grand_prize_prob
  let second_prize_winners := people_with_envelopes * second_prize_prob
  let consolation_prize_winners := people_with_envelopes * consolation_prize_prob
  let empty_envelopes := people_with_envelopes - (grand_prize_winners + second_prize_winners + consolation_prize_winners)
  grand_prize_winners = 4 ∧
  second_prize_winners = 8 ∧
  consolation_prize_winners = 12 ∧
  empty_envelopes = 16 := by
  sorry

end NUMINAMATH_GPT_prize_expectation_l2079_207954


namespace NUMINAMATH_GPT_quadratic_root_reciprocal_l2079_207917

theorem quadratic_root_reciprocal (p q r s : ℝ) 
    (h1 : ∃ a : ℝ, a^2 + p * a + q = 0 ∧ (1 / a)^2 + r * (1 / a) + s = 0) :
    (p * s - r) * (q * r - p) = (q * s - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_reciprocal_l2079_207917


namespace NUMINAMATH_GPT_cubic_of_m_eq_4_l2079_207961

theorem cubic_of_m_eq_4 (m : ℕ) (h : 3 ^ m = 81) : m ^ 3 = 64 := 
by
  sorry

end NUMINAMATH_GPT_cubic_of_m_eq_4_l2079_207961


namespace NUMINAMATH_GPT_M_subset_N_l2079_207977

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end NUMINAMATH_GPT_M_subset_N_l2079_207977


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_D_l2079_207991

theorem statement_A (x : ℝ) (hx : x > 1) : 
  ∃(y : ℝ), y = 3 * x + 1 / (x - 1) ∧ y = 2 * Real.sqrt 3 + 3 := 
  sorry

theorem statement_B (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃(z : ℝ), z = 1 / (x + 1) + 2 / y ∧ z = 9 / 2 := 
  sorry

theorem statement_D (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃(k : ℝ), k = (x^2 + y^2 + z^2) / (3 * x * y + 4 * y * z) ∧ k = 2 / 5 := 
  sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_D_l2079_207991


namespace NUMINAMATH_GPT_arc_length_condition_l2079_207997

open Real

noncomputable def hyperbola_eq (a b x y: ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem arc_length_condition (a b r: ℝ) (h1: hyperbola_eq a b 2 1) (h2: r > 0)
  (h3: ∃ x y, x^2 + y^2 = r^2 ∧ hyperbola_eq a b x y) :
  r > 2 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_arc_length_condition_l2079_207997


namespace NUMINAMATH_GPT_value_of_s_for_g_neg_1_eq_0_l2079_207941

def g (x s : ℝ) := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

theorem value_of_s_for_g_neg_1_eq_0 (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_s_for_g_neg_1_eq_0_l2079_207941


namespace NUMINAMATH_GPT_total_scoops_needed_l2079_207992

def cups_of_flour : ℕ := 4
def cups_of_sugar : ℕ := 3
def cups_of_milk : ℕ := 2

def flour_scoop_size : ℚ := 1 / 4
def sugar_scoop_size : ℚ := 1 / 3
def milk_scoop_size : ℚ := 1 / 2

theorem total_scoops_needed : 
  (cups_of_flour / flour_scoop_size) + (cups_of_sugar / sugar_scoop_size) + (cups_of_milk / milk_scoop_size) = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_scoops_needed_l2079_207992


namespace NUMINAMATH_GPT_even_function_a_value_l2079_207915

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_value_l2079_207915


namespace NUMINAMATH_GPT_each_sibling_gets_13_pencils_l2079_207932

theorem each_sibling_gets_13_pencils (colored_pencils black_pencils kept_pencils siblings : ℕ) 
  (h1 : colored_pencils = 14)
  (h2 : black_pencils = 35)
  (h3 : kept_pencils = 10)
  (h4 : siblings = 3) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by
  sorry

end NUMINAMATH_GPT_each_sibling_gets_13_pencils_l2079_207932


namespace NUMINAMATH_GPT_number_of_homework_situations_l2079_207958

theorem number_of_homework_situations (teachers students : ℕ) (homework_options : students = 4 ∧ teachers = 3) :
  teachers ^ students = 81 :=
by
  sorry

end NUMINAMATH_GPT_number_of_homework_situations_l2079_207958


namespace NUMINAMATH_GPT_polynomial_min_value_P_l2079_207950

theorem polynomial_min_value_P (a b : ℝ) (h_root_pos : ∀ x, a * x^3 - x^2 + b * x - 1 = 0 → 0 < x) :
    (∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) →
    ∃ P : ℝ, P = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_polynomial_min_value_P_l2079_207950


namespace NUMINAMATH_GPT_zander_stickers_l2079_207952

/-- Zander starts with 100 stickers, Andrew receives 1/5 of Zander's total, 
    and Bill receives 3/10 of the remaining stickers. Prove that the total 
    number of stickers given to Andrew and Bill is 44. -/
theorem zander_stickers :
  let total_stickers := 100
  let andrew_fraction := 1 / 5
  let remaining_stickers := total_stickers - (total_stickers * andrew_fraction)
  let bill_fraction := 3 / 10
  (total_stickers * andrew_fraction) + (remaining_stickers * bill_fraction) = 44 := 
by
  sorry

end NUMINAMATH_GPT_zander_stickers_l2079_207952


namespace NUMINAMATH_GPT_probability_sum_greater_than_five_l2079_207955

theorem probability_sum_greater_than_five (dice_outcomes : List (ℕ × ℕ)) (h: dice_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,2), (4,1), (5,1), (2,4)] ++ 
                              [(1,5), (2,6), (3,3), (3,4), (3,5), (3,6), (4,2), (4,3), (4,4), (4,5), (4,6), 
                               (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]) :
  p_greater_5 = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_probability_sum_greater_than_five_l2079_207955


namespace NUMINAMATH_GPT_two_digit_num_square_ends_in_self_l2079_207978

theorem two_digit_num_square_ends_in_self {x : ℕ} (hx : 10 ≤ x ∧ x < 100) (hx0 : x % 10 ≠ 0) : 
  (x * x % 100 = x) ↔ (x = 25 ∨ x = 76) :=
sorry

end NUMINAMATH_GPT_two_digit_num_square_ends_in_self_l2079_207978


namespace NUMINAMATH_GPT_cookies_baked_total_l2079_207914

   -- Definitions based on the problem conditions
   def cookies_yesterday : ℕ := 435
   def cookies_this_morning : ℕ := 139

   -- The theorem we want to prove
   theorem cookies_baked_total : cookies_yesterday + cookies_this_morning = 574 :=
   by sorry
   
end NUMINAMATH_GPT_cookies_baked_total_l2079_207914


namespace NUMINAMATH_GPT_largest_sphere_radius_l2079_207984

-- Define the conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 7
def circle_center_x := 5
def circle_center_z := 2
def circle_radius := 2

-- Define the question into a statement
noncomputable def radius_of_largest_sphere : ℝ :=
  (29 : ℝ) / 4

-- Prove the required radius given the conditions
theorem largest_sphere_radius:
  ∀ (r : ℝ),
  r = radius_of_largest_sphere → r * r = inner_radius * inner_radius + (circle_center_x * circle_center_x + (r - circle_center_z) * (r - circle_center_z))
:=
by
  sorry

end NUMINAMATH_GPT_largest_sphere_radius_l2079_207984


namespace NUMINAMATH_GPT_average_rate_of_change_is_4_l2079_207999

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_rate_of_change_is_4_l2079_207999


namespace NUMINAMATH_GPT_floral_arrangement_carnations_percentage_l2079_207903

theorem floral_arrangement_carnations_percentage :
  ∀ (F : ℕ),
  (1 / 4) * (7 / 10) * F + (2 / 3) * (3 / 10) * F = (29 / 40) * F :=
by
  sorry

end NUMINAMATH_GPT_floral_arrangement_carnations_percentage_l2079_207903


namespace NUMINAMATH_GPT_matrix_power_B150_l2079_207988

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_B150_l2079_207988


namespace NUMINAMATH_GPT_joe_initial_tests_count_l2079_207900

theorem joe_initial_tests_count (n S : ℕ) (h1 : S = 45 * n) (h2 : S - 30 = 50 * (n - 1)) : n = 4 := by
  sorry

end NUMINAMATH_GPT_joe_initial_tests_count_l2079_207900


namespace NUMINAMATH_GPT_part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l2079_207934

variable {x a : ℝ}

theorem part1_solution (h1 : a > 1 / 3) (h2 : (a * x - 1) / (x ^ 2 - 1) = 0) : x = 3 := by
  sorry

theorem part2_solution_1 (h1 : -1 < a) (h2 : a < 0) : {x | x < (1 / a) ∨ (-1 < x ∧ x < 1)} := by
  sorry

theorem part2_solution_2 (h1 : a = -1) : {x | x < 1 ∧ x ≠ -1} := by
  sorry

theorem part2_solution_3 (h1 : a < -1) : {x | x < -1 ∨ (1 / a < x ∧ x < 1)} := by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l2079_207934


namespace NUMINAMATH_GPT_polynomial_value_l2079_207911

noncomputable def p (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) + 24 * x

theorem polynomial_value :
  (p 1 = 24) ∧ (p 2 = 48) ∧ (p 3 = 72) ∧ (p 4 = 96) →
  p 0 + p 5 = 168 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l2079_207911


namespace NUMINAMATH_GPT_integer_triangle_answer_l2079_207986

def integer_triangle_condition :=
∀ a r : ℕ, (1 ≤ a ∧ a ≤ 19) → 
(a = 12) → (r = 3) → 
(r = 96 / (20 + a))

theorem integer_triangle_answer : 
  integer_triangle_condition := 
by
  sorry

end NUMINAMATH_GPT_integer_triangle_answer_l2079_207986


namespace NUMINAMATH_GPT_product_of_two_numbers_l2079_207944

theorem product_of_two_numbers
  (x y : ℝ)
  (h_diff : x - y ≠ 0)
  (h1 : x + y = 5 * (x - y))
  (h2 : x * y = 15 * (x - y)) :
  x * y = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l2079_207944


namespace NUMINAMATH_GPT_retail_store_paid_40_percent_more_l2079_207979

variables (C R : ℝ)

-- Condition: The customer price is 96% more than manufacturing cost
def customer_price_from_manufacturing (C : ℝ) : ℝ := 1.96 * C

-- Condition: The customer price is 40% more than the retailer price
def customer_price_from_retail (R : ℝ) : ℝ := 1.40 * R

-- Theorem to be proved
theorem retail_store_paid_40_percent_more (C R : ℝ) 
  (h_customer_price : customer_price_from_manufacturing C = customer_price_from_retail R) :
  (R - C) / C = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_retail_store_paid_40_percent_more_l2079_207979


namespace NUMINAMATH_GPT_solution_quad_ineq_l2079_207942

noncomputable def quadratic_inequality_solution_set :=
  {x : ℝ | (x > -1) ∧ (x < 3) ∧ (x ≠ 2)}

theorem solution_quad_ineq (x : ℝ) :
  ((x^2 - 2*x - 3)*(x^2 - 4*x + 4) < 0) ↔ x ∈ quadratic_inequality_solution_set :=
by sorry

end NUMINAMATH_GPT_solution_quad_ineq_l2079_207942


namespace NUMINAMATH_GPT_necessary_condition_for_inequality_l2079_207980

theorem necessary_condition_for_inequality (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3 * x + 2 < 0) → (x > m)) ∧ (∃ x : ℝ, (x > m) ∧ ¬(x^2 - 3 * x + 2 < 0)) → m ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_necessary_condition_for_inequality_l2079_207980


namespace NUMINAMATH_GPT_cost_of_pencil_l2079_207994

theorem cost_of_pencil (s n c : ℕ) (h_majority : s > 15) (h_pencils : n > 1) (h_cost : c > n)
  (h_total_cost : s * c * n = 1771) : c = 11 :=
sorry

end NUMINAMATH_GPT_cost_of_pencil_l2079_207994


namespace NUMINAMATH_GPT_range_of_a2_l2079_207967

theorem range_of_a2 (a : ℕ → ℝ) (S : ℕ → ℝ) (a2 : ℝ) (a3 a6 : ℝ) (h1: 3 * a3 = a6 + 4) (h2 : S 5 < 10) :
  a2 < 2 := 
sorry

end NUMINAMATH_GPT_range_of_a2_l2079_207967


namespace NUMINAMATH_GPT_problem_l2079_207908

theorem problem (n : ℕ) (p : ℕ) (a b c : ℤ)
  (hn : 0 < n)
  (hp : Nat.Prime p)
  (h_eq : a^n + p * b = b^n + p * c)
  (h_eq2 : b^n + p * c = c^n + p * a) :
  a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_problem_l2079_207908


namespace NUMINAMATH_GPT_total_amount_lent_l2079_207919

theorem total_amount_lent (A T : ℝ) (hA : A = 15008) (hInterest : 0.08 * A + 0.10 * (T - A) = 850) : 
  T = 11501.6 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_lent_l2079_207919


namespace NUMINAMATH_GPT_find_g_neg1_l2079_207931

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_eq : ∀ x : ℝ, f x = g x + x^2)
variable (h_g1 : g 1 = 1)

-- The statement to prove
theorem find_g_neg1 : g (-1) = -3 :=
sorry

end NUMINAMATH_GPT_find_g_neg1_l2079_207931


namespace NUMINAMATH_GPT_original_average_of_15_numbers_l2079_207901

theorem original_average_of_15_numbers (A : ℝ) (h1 : 15 * A + 15 * 12 = 52 * 15) :
  A = 40 :=
sorry

end NUMINAMATH_GPT_original_average_of_15_numbers_l2079_207901


namespace NUMINAMATH_GPT_min_combined_horses_and_ponies_l2079_207947

theorem min_combined_horses_and_ponies : 
  ∀ (P : ℕ), 
  (∃ (P' : ℕ), 
    (P = P' ∧ (∃ (x : ℕ), x = 3 * P' / 10 ∧ x = 3 * P' / 16) ∧
     (∃ (y : ℕ), y = 5 * x / 8) ∧ 
      ∀ (H : ℕ), (H = 3 + P')) → 
  P + (3 + P) = 35) := 
sorry

end NUMINAMATH_GPT_min_combined_horses_and_ponies_l2079_207947


namespace NUMINAMATH_GPT_arun_borrowed_amount_l2079_207989

theorem arun_borrowed_amount :
  ∃ P : ℝ, 
    (P * 0.08 * 4 + P * 0.10 * 6 + P * 0.12 * 5 = 12160) → P = 8000 :=
sorry

end NUMINAMATH_GPT_arun_borrowed_amount_l2079_207989


namespace NUMINAMATH_GPT_total_flowers_l2079_207973

theorem total_flowers (pots: ℕ) (flowers_per_pot: ℕ) (h_pots: pots = 2150) (h_flowers_per_pot: flowers_per_pot = 128) :
    pots * flowers_per_pot = 275200 :=
by 
    sorry

end NUMINAMATH_GPT_total_flowers_l2079_207973


namespace NUMINAMATH_GPT_correct_judgment_is_C_l2079_207948

-- Definitions based on conditions
def three_points_determine_a_plane (p1 p2 p3 : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by three points
  sorry

def line_and_point_determine_a_plane (l : Line) (p : Point) : Prop :=
  -- This would use some axiom or definition of a plane determined by a line and a point not on the line
  sorry

def two_parallel_lines_and_intersecting_line_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Axiom 3 and its corollary stating that two parallel lines intersected by the same line are in the same plane
  sorry

def three_lines_intersect_pairwise_same_plane (l1 l2 l3 : Line) : Prop :=
  -- Definition stating that three lines intersecting pairwise might be co-planar or not
  sorry

-- Statement of the problem in Lean
theorem correct_judgment_is_C :
    ¬ (three_points_determine_a_plane p1 p2 p3)
  ∧ ¬ (line_and_point_determine_a_plane l p)
  ∧ (two_parallel_lines_and_intersecting_line_same_plane l1 l2 l3)
  ∧ ¬ (three_lines_intersect_pairwise_same_plane l1 l2 l3) :=
  sorry

end NUMINAMATH_GPT_correct_judgment_is_C_l2079_207948


namespace NUMINAMATH_GPT_number_of_smallest_squares_l2079_207916

-- Conditions
def length_cm : ℝ := 28
def width_cm : ℝ := 48
def total_lines_cm : ℝ := 6493.6

-- The main question is about the number of smallest squares
theorem number_of_smallest_squares (d : ℝ) (h_d : d = 0.4) :
  ∃ n : ℕ, n = (length_cm / d - 2) * (width_cm / d - 2) ∧ n = 8024 :=
by
  sorry

end NUMINAMATH_GPT_number_of_smallest_squares_l2079_207916


namespace NUMINAMATH_GPT_total_minutes_to_finish_album_l2079_207929

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end NUMINAMATH_GPT_total_minutes_to_finish_album_l2079_207929


namespace NUMINAMATH_GPT_gcd_expression_l2079_207924

theorem gcd_expression (a : ℤ) (k : ℤ) (h1 : a = k * 1171) (h2 : k % 2 = 1) (prime_1171 : Prime 1171) : 
  Int.gcd (3 * a^2 + 35 * a + 77) (a + 15) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_expression_l2079_207924


namespace NUMINAMATH_GPT_center_and_radius_of_circle_l2079_207987

theorem center_and_radius_of_circle (x y : ℝ) : 
  (x + 1)^2 + (y - 2)^2 = 4 → (x = -1 ∧ y = 2 ∧ ∃ r, r = 2) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_center_and_radius_of_circle_l2079_207987


namespace NUMINAMATH_GPT_pen_cost_l2079_207971

def pencil_cost : ℝ := 1.60
def elizabeth_money : ℝ := 20.00
def num_pencils : ℕ := 5
def num_pens : ℕ := 6

theorem pen_cost (pen_cost : ℝ) : 
  elizabeth_money - (num_pencils * pencil_cost) = num_pens * pen_cost → 
  pen_cost = 2 :=
by 
  sorry

end NUMINAMATH_GPT_pen_cost_l2079_207971


namespace NUMINAMATH_GPT_correct_operation_l2079_207983

theorem correct_operation (x : ℝ) : (-x^3)^2 = x^6 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l2079_207983


namespace NUMINAMATH_GPT_first_person_amount_l2079_207925

theorem first_person_amount (A B C : ℕ) (h1 : A = 28) (h2 : B = 72) (h3 : C = 98) (h4 : A + B + C = 198) (h5 : 99 ≤ max (A + B) (B + C) / 2) : 
  A = 28 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_first_person_amount_l2079_207925


namespace NUMINAMATH_GPT_percentage_of_green_ducks_l2079_207963

theorem percentage_of_green_ducks (ducks_small_pond ducks_large_pond : ℕ) 
  (green_fraction_small_pond green_fraction_large_pond : ℚ) 
  (h1 : ducks_small_pond = 20) 
  (h2 : ducks_large_pond = 80) 
  (h3 : green_fraction_small_pond = 0.20) 
  (h4 : green_fraction_large_pond = 0.15) :
  let total_ducks := ducks_small_pond + ducks_large_pond
  let green_ducks := (green_fraction_small_pond * ducks_small_pond) + 
                     (green_fraction_large_pond * ducks_large_pond)
  (green_ducks / total_ducks) * 100 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_green_ducks_l2079_207963


namespace NUMINAMATH_GPT_max_mn_value_min_4m_square_n_square_l2079_207940

variable {m n : ℝ}
variable (h_cond1 : m > 0)
variable (h_cond2 : n > 0)
variable (h_eq : 2 * m + n = 1)

theorem max_mn_value : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ m * n = 1/8) := 
  sorry

theorem min_4m_square_n_square : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ 4 * m^2 + n^2 = 1/2) := 
  sorry

end NUMINAMATH_GPT_max_mn_value_min_4m_square_n_square_l2079_207940


namespace NUMINAMATH_GPT_tablecloth_radius_l2079_207910

theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 :=
by {
  -- Outline the proof structure to ensure the statement is correct
  sorry
}

end NUMINAMATH_GPT_tablecloth_radius_l2079_207910


namespace NUMINAMATH_GPT_find_m_l2079_207993

-- Definitions for the conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_of_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = a 1 * (1 - (a n / a 1)) / (1 - (a 2 / a 1))

def arithmetic_sequence (S3 S9 S6 : ℝ) :=
  2 * S9 = S3 + S6

def condition_3 (a : ℕ → ℝ) (m : ℕ) :=
  a 2 + a 5 = 2 * a m

-- Lean 4 statement that requires proof
theorem find_m 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ) 
  (geom_seq : geometric_sequence a a1 q)
  (sum_geom_seq : sum_of_geometric_sequence S a)
  (arith_seq : arithmetic_sequence (S 3) (S 9) (S 6))
  (cond3 : condition_3 a 8) : 
  8 = 8 := 
sorry

end NUMINAMATH_GPT_find_m_l2079_207993


namespace NUMINAMATH_GPT_contrapositive_l2079_207906

variable (k : ℝ)

theorem contrapositive (h : ¬∃ x : ℝ, x^2 - x - k = 0) : k ≤ 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_l2079_207906


namespace NUMINAMATH_GPT_find_new_songs_l2079_207928

-- Definitions for the conditions
def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

-- The number of new songs added
def new_songs_added : ℕ := 20

-- Statement of the proof problem
theorem find_new_songs (n d f x : ℕ) (h1 : n = initial_songs) (h2 : d = deleted_songs) (h3 : f = final_songs) : f = n - d + x → x = new_songs_added :=
by
  intros h4
  sorry

end NUMINAMATH_GPT_find_new_songs_l2079_207928


namespace NUMINAMATH_GPT_solve_inequality_l2079_207953

def inequality_solution :=
  {x : ℝ // x < -3 ∨ x > -6/5}

theorem solve_inequality (x : ℝ) : 
  |2*x - 4| - |3*x + 9| < 1 → x < -3 ∨ x > -6/5 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2079_207953


namespace NUMINAMATH_GPT_milk_production_l2079_207982

theorem milk_production 
  (initial_cows : ℕ)
  (initial_milk : ℕ)
  (initial_days : ℕ)
  (max_milk_per_cow_per_day : ℕ)
  (available_cows : ℕ)
  (days : ℕ)
  (H_initial : initial_cows = 10)
  (H_initial_milk : initial_milk = 40)
  (H_initial_days : initial_days = 5)
  (H_max_milk : max_milk_per_cow_per_day = 2)
  (H_available_cows : available_cows = 15)
  (H_days : days = 8) :
  available_cows * initial_milk / (initial_cows * initial_days) * days = 96 := 
by 
  sorry

end NUMINAMATH_GPT_milk_production_l2079_207982


namespace NUMINAMATH_GPT_number_of_tiles_l2079_207920

theorem number_of_tiles (w l : ℕ) (h1 : 2 * w + 2 * l - 4 = (w * l - (2 * w + 2 * l - 4)))
  (h2 : w > 0) (h3 : l > 0) : w * l = 48 ∨ w * l = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tiles_l2079_207920


namespace NUMINAMATH_GPT_boxes_left_for_Sonny_l2079_207927

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end NUMINAMATH_GPT_boxes_left_for_Sonny_l2079_207927


namespace NUMINAMATH_GPT_min_cost_proof_l2079_207959

-- Define the costs and servings for each ingredient
def pasta_cost : ℝ := 1.12
def pasta_servings_per_box : ℕ := 5

def meatballs_cost : ℝ := 5.24
def meatballs_servings_per_pack : ℕ := 4

def tomato_sauce_cost : ℝ := 2.31
def tomato_sauce_servings_per_jar : ℕ := 5

def tomatoes_cost : ℝ := 1.47
def tomatoes_servings_per_pack : ℕ := 4

def lettuce_cost : ℝ := 0.97
def lettuce_servings_per_head : ℕ := 6

def olives_cost : ℝ := 2.10
def olives_servings_per_jar : ℕ := 8

def cheese_cost : ℝ := 2.70
def cheese_servings_per_block : ℕ := 7

-- Define the number of people to serve
def number_of_people : ℕ := 8

-- The total cost calculated
def total_cost : ℝ := 
  (2 * pasta_cost) +
  (2 * meatballs_cost) +
  (2 * tomato_sauce_cost) +
  (2 * tomatoes_cost) +
  (2 * lettuce_cost) +
  (1 * olives_cost) +
  (2 * cheese_cost)

-- The minimum total cost
def min_total_cost : ℝ := 29.72

theorem min_cost_proof : total_cost = min_total_cost :=
by sorry

end NUMINAMATH_GPT_min_cost_proof_l2079_207959


namespace NUMINAMATH_GPT_group_size_l2079_207913

def total_blocks : ℕ := 820
def num_groups : ℕ := 82

theorem group_size :
  total_blocks / num_groups = 10 := 
by 
  sorry

end NUMINAMATH_GPT_group_size_l2079_207913


namespace NUMINAMATH_GPT_not_or_implies_both_false_l2079_207951

-- The statement of the problem in Lean
theorem not_or_implies_both_false {p q : Prop} (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
sorry

end NUMINAMATH_GPT_not_or_implies_both_false_l2079_207951


namespace NUMINAMATH_GPT_inequality_holds_l2079_207902

theorem inequality_holds (x : ℝ) : x + 2 < x + 3 := 
by {
    sorry
}

end NUMINAMATH_GPT_inequality_holds_l2079_207902


namespace NUMINAMATH_GPT_expression_not_defined_l2079_207968

theorem expression_not_defined (x : ℝ) : 
  (x^2 - 21 * x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by
sorry

end NUMINAMATH_GPT_expression_not_defined_l2079_207968


namespace NUMINAMATH_GPT_solve_10_arithmetic_in_1_minute_l2079_207966

-- Define the times required for each task
def time_math_class : Nat := 40 -- in minutes
def time_walk_kilometer : Nat := 20 -- in minutes
def time_solve_arithmetic : Nat := 1 -- in minutes

-- The question: Which task can be completed in 1 minute?
def task_completed_in_1_minute : Nat := 1

theorem solve_10_arithmetic_in_1_minute :
  time_solve_arithmetic = task_completed_in_1_minute :=
by
  sorry

end NUMINAMATH_GPT_solve_10_arithmetic_in_1_minute_l2079_207966


namespace NUMINAMATH_GPT_value_of_b_l2079_207975

theorem value_of_b (b x : ℝ) (h1 : 2 * x + 7 = 3) (h2 : b * x - 10 = -2) : b = -4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2079_207975


namespace NUMINAMATH_GPT_perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l2079_207904

theorem perfect_squares_multiple_of_72 (N : ℕ) : 
  (N^2 < 1000000) ∧ (N^2 % 72 = 0) ↔ N ≤ 996 :=
sorry

theorem number_of_perfect_squares_multiple_of_72 : 
  ∃ upper_bound : ℕ, upper_bound = 83 ∧ ∀ n : ℕ, (n < 1000000) → (n % 144 = 0) → n ≤ (12 * upper_bound) :=
sorry

end NUMINAMATH_GPT_perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l2079_207904


namespace NUMINAMATH_GPT_no_three_digit_numbers_with_sum_27_are_even_l2079_207922

-- We define a 3-digit number and its conditions based on digit-sum and even properties
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem no_three_digit_numbers_with_sum_27_are_even :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ is_even n :=
by sorry

end NUMINAMATH_GPT_no_three_digit_numbers_with_sum_27_are_even_l2079_207922


namespace NUMINAMATH_GPT_cube_dimension_l2079_207938

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end NUMINAMATH_GPT_cube_dimension_l2079_207938
