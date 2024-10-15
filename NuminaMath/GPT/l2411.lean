import Mathlib

namespace NUMINAMATH_GPT_outer_term_in_proportion_l2411_241198

theorem outer_term_in_proportion (a b x : ℝ) (h_ab : a * b = 1) (h_x : x = 0.2) : b = 5 :=
by
  sorry

end NUMINAMATH_GPT_outer_term_in_proportion_l2411_241198


namespace NUMINAMATH_GPT_solve_inequality_l2411_241111

theorem solve_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2411_241111


namespace NUMINAMATH_GPT_problem_l2411_241128

theorem problem (x y : ℝ) : 
  2 * x + y = 11 → x + 2 * y = 13 → 10 * x^2 - 6 * x * y + y^2 = 530 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2411_241128


namespace NUMINAMATH_GPT_g_minus3_is_correct_l2411_241103

theorem g_minus3_is_correct (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 3 * x^2) : 
  g (-3) = 247 / 39 :=
by
  sorry

end NUMINAMATH_GPT_g_minus3_is_correct_l2411_241103


namespace NUMINAMATH_GPT_vertex_hyperbola_l2411_241138

theorem vertex_hyperbola (a b : ℝ) (h_cond : 8 * a^2 + 4 * a * b = b^3) :
    let xv := -b / (2 * a)
    let yv := (4 * a - b^2) / (4 * a)
    (xv * yv = 1) :=
  by
  sorry

end NUMINAMATH_GPT_vertex_hyperbola_l2411_241138


namespace NUMINAMATH_GPT_discount_is_28_l2411_241190

-- Definitions
def price_notebook : ℕ := 15
def price_planner : ℕ := 10
def num_notebooks : ℕ := 4
def num_planners : ℕ := 8
def total_cost_with_discount : ℕ := 112

-- The original cost without discount
def original_cost : ℕ := num_notebooks * price_notebook + num_planners * price_planner

-- The discount amount
def discount_amount : ℕ := original_cost - total_cost_with_discount

-- Proof statement
theorem discount_is_28 : discount_amount = 28 := by
  sorry

end NUMINAMATH_GPT_discount_is_28_l2411_241190


namespace NUMINAMATH_GPT_solve_fraction_equation_l2411_241141

theorem solve_fraction_equation (x : ℚ) (h : (x + 7) / (x - 4) = (x - 5) / (x + 3)) : x = -1 / 19 := 
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l2411_241141


namespace NUMINAMATH_GPT_greatest_possible_employees_take_subway_l2411_241134

variable (P F : ℕ)

def part_time_employees_take_subway : ℕ := P / 3
def full_time_employees_take_subway : ℕ := F / 4

theorem greatest_possible_employees_take_subway 
  (h1 : P + F = 48) : part_time_employees_take_subway P + full_time_employees_take_subway F ≤ 15 := 
sorry

end NUMINAMATH_GPT_greatest_possible_employees_take_subway_l2411_241134


namespace NUMINAMATH_GPT_abs_eq_self_iff_nonneg_l2411_241143

variable (a : ℝ)

theorem abs_eq_self_iff_nonneg (h : |a| = a) : a ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_self_iff_nonneg_l2411_241143


namespace NUMINAMATH_GPT_rotation_transform_l2411_241171

theorem rotation_transform (x y α : ℝ) :
    let x' := x * Real.cos α - y * Real.sin α
    let y' := x * Real.sin α + y * Real.cos α
    (x', y') = (x * Real.cos α - y * Real.sin α, x * Real.sin α + y * Real.cos α) := by
  sorry

end NUMINAMATH_GPT_rotation_transform_l2411_241171


namespace NUMINAMATH_GPT_infinite_product_equals_nine_l2411_241104

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, ite (n = 0) 1 (3^(n * (1 / 2^n)))

theorem infinite_product_equals_nine : infinite_product = 9 := sorry

end NUMINAMATH_GPT_infinite_product_equals_nine_l2411_241104


namespace NUMINAMATH_GPT_find_number_l2411_241176

theorem find_number (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2411_241176


namespace NUMINAMATH_GPT_max_a2_plus_b2_l2411_241161

theorem max_a2_plus_b2 (a b : ℝ) (h1 : b = 1) (h2 : 1 ≤ -a + 7) (h3 : 1 ≥ a - 3) : a^2 + b^2 = 37 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_a2_plus_b2_l2411_241161


namespace NUMINAMATH_GPT_volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l2411_241150

noncomputable def volume_tetrahedron (V A B C : Point) : ℝ := sorry

def is_interior_point (M V A B C : Point) : Prop := sorry -- Definition of an interior point

def is_barycenter (M V A B C : Point) : Prop := sorry -- Definition of a barycenter

def intersects_lines_planes (M V A B C A1 B1 C1 : Point) : Prop := sorry -- Definition of intersection points

def intersects_lines_sides (V A1 B1 C1 A B C A2 B2 C2 : Point) : Prop := sorry -- Definition of intersection points with sides

theorem volume_le_one_fourth_of_original (V A B C: Point) 
  (M : Point) (A1 B1 C1 A2 B2 C2 : Point) 
  (h_interior : is_interior_point M V A B C) 
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1) 
  (h_intersects_sides : intersects_lines_sides V A1 B1 C1 A B C A2 B2 C2) :
  volume_tetrahedron V A2 B2 C2 ≤ (1/4) * volume_tetrahedron V A B C :=
sorry

theorem volume_of_sub_tetrahedron (V A B C: Point) 
  (M V1 : Point) (A1 B1 C1 : Point)
  (h_barycenter : is_barycenter M V A B C)
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1)
  (h_point_V1 : intersects_something_to_find_V1) : 
  volume_tetrahedron V1 A1 B1 C1 = (1/4) * volume_tetrahedron V A B C :=
sorry

end NUMINAMATH_GPT_volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l2411_241150


namespace NUMINAMATH_GPT_yard_length_l2411_241181

theorem yard_length
  (trees : ℕ) (gaps : ℕ) (distance_between_trees : ℕ) :
  trees = 26 → 
  gaps = trees - 1 → 
  distance_between_trees = 14 → 
  length_of_yard = gaps * distance_between_trees → 
  length_of_yard = 350 :=
by
  intros h_trees h_gaps h_distance h_length
  sorry

end NUMINAMATH_GPT_yard_length_l2411_241181


namespace NUMINAMATH_GPT_probability_of_selecting_one_of_each_color_l2411_241196

noncomputable def number_of_ways_to_select_4_marbles_from_10 := Nat.choose 10 4
noncomputable def ways_to_select_1_red := Nat.choose 3 1
noncomputable def ways_to_select_1_blue := Nat.choose 3 1
noncomputable def ways_to_select_1_green := Nat.choose 2 1
noncomputable def ways_to_select_1_yellow := Nat.choose 2 1

theorem probability_of_selecting_one_of_each_color :
  (ways_to_select_1_red * ways_to_select_1_blue * ways_to_select_1_green * ways_to_select_1_yellow) / number_of_ways_to_select_4_marbles_from_10 = 6 / 35 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_of_each_color_l2411_241196


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l2411_241136

theorem megatek_manufacturing_percentage 
  (total_degrees : ℝ := 360)
  (manufacturing_degrees : ℝ := 18)
  (is_proportional : (manufacturing_degrees / total_degrees) * 100 = 5) :
  (manufacturing_degrees / total_degrees) * 100 = 5 := 
  by
  exact is_proportional

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l2411_241136


namespace NUMINAMATH_GPT_n_must_be_even_l2411_241192

open Nat

-- Define the system of equations:
def equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (∀ i, 2 ≤ i ∧ i ≤ n - 1 → (-x (i-1) + 2 * x i - x (i+1) = 1)) ∧
  (2 * x 1 - x 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0)

-- Define the last equation separately due to its unique form:
def last_equation (n : ℕ) (x : ℕ → ℤ) : Prop :=
  (n ≥ 1 → -x (n-1) + 2 * x n = 1)

-- The theorem to prove that n must be even:
theorem n_must_be_even (n : ℕ) (x : ℕ → ℤ) : 
  equation n x → last_equation n x → Even n :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_n_must_be_even_l2411_241192


namespace NUMINAMATH_GPT_ticket_cost_difference_l2411_241148

noncomputable def total_cost_adults (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_cost_children (tickets : ℕ) (price : ℝ) : ℝ := tickets * price
noncomputable def total_tickets (adults : ℕ) (children : ℕ) : ℕ := adults + children
noncomputable def discount (threshold : ℕ) (discount_rate : ℝ) (cost : ℝ) (tickets : ℕ) : ℝ :=
  if tickets > threshold then cost * discount_rate else 0
noncomputable def final_cost (initial_cost : ℝ) (discount : ℝ) : ℝ := initial_cost - discount
noncomputable def proportional_discount (partial_cost : ℝ) (total_cost : ℝ) (total_discount : ℝ) : ℝ :=
  (partial_cost / total_cost) * total_discount
noncomputable def difference (cost1 : ℝ) (cost2 : ℝ) : ℝ := cost1 - cost2

theorem ticket_cost_difference :
  let adult_tickets := 9
  let children_tickets := 7
  let adult_price := 11
  let children_price := 7
  let discount_rate := 0.15
  let discount_threshold := 10
  let total_adult_cost := total_cost_adults adult_tickets adult_price
  let total_children_cost := total_cost_children children_tickets children_price
  let all_tickets := total_tickets adult_tickets children_tickets
  let initial_total_cost := total_adult_cost + total_children_cost
  let total_discount := discount discount_threshold discount_rate initial_total_cost all_tickets
  let final_total_cost := final_cost initial_total_cost total_discount
  let adult_discount := proportional_discount total_adult_cost initial_total_cost total_discount
  let children_discount := proportional_discount total_children_cost initial_total_cost total_discount
  let final_adult_cost := final_cost total_adult_cost adult_discount
  let final_children_cost := final_cost total_children_cost children_discount
  difference final_adult_cost final_children_cost = 42.52 := by
  sorry

end NUMINAMATH_GPT_ticket_cost_difference_l2411_241148


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l2411_241100

theorem no_positive_integer_solutions (m : ℕ) (h_pos : m > 0) :
  ¬ ∃ x : ℚ, m * x^2 + 40 * x + m = 0 :=
by {
  -- the proof goes here
  sorry
}

end NUMINAMATH_GPT_no_positive_integer_solutions_l2411_241100


namespace NUMINAMATH_GPT_abs_f_at_1_eq_20_l2411_241125

noncomputable def fourth_degree_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ p : Polynomial ℝ, p.degree = 4 ∧ ∀ x, f x = p.eval x

theorem abs_f_at_1_eq_20 
  (f : ℝ → ℝ)
  (h_f_poly : fourth_degree_polynomial f)
  (h_f_neg2 : |f (-2)| = 10)
  (h_f_0 : |f 0| = 10)
  (h_f_3 : |f 3| = 10)
  (h_f_7 : |f 7| = 10) :
  |f 1| = 20 := 
sorry

end NUMINAMATH_GPT_abs_f_at_1_eq_20_l2411_241125


namespace NUMINAMATH_GPT_principal_sum_l2411_241126

/-!
# Problem Statement
Given:
1. The difference between compound interest (CI) and simple interest (SI) on a sum at 10% per annum for 2 years is 65.
2. The rate of interest \( R \) is 10%.
3. The time \( T \) is 2 years.

We need to prove that the principal sum \( P \) is 6500.
-/

theorem principal_sum (P : ℝ) (R : ℝ) (T : ℕ) (H : (P * (1 + R / 100)^T - P) - (P * R * T / 100) = 65) 
                      (HR : R = 10) (HT : T = 2) : P = 6500 := 
by 
  sorry

end NUMINAMATH_GPT_principal_sum_l2411_241126


namespace NUMINAMATH_GPT_area_of_smallest_square_that_encloses_circle_l2411_241116

def radius : ℕ := 5

def diameter (r : ℕ) : ℕ := 2 * r

def side_length (d : ℕ) : ℕ := d

def area_of_square (s : ℕ) : ℕ := s * s

theorem area_of_smallest_square_that_encloses_circle :
  area_of_square (side_length (diameter radius)) = 100 := by
  sorry

end NUMINAMATH_GPT_area_of_smallest_square_that_encloses_circle_l2411_241116


namespace NUMINAMATH_GPT_min_coins_for_any_amount_below_dollar_l2411_241106

-- Definitions of coin values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- Statement: The minimum number of coins required to pay any amount less than a dollar
theorem min_coins_for_any_amount_below_dollar :
  ∃ (n : ℕ), n = 11 ∧
  (∀ (amount : ℕ), 1 ≤ amount ∧ amount < 100 →
   ∃ (a b c d : ℕ), amount = a * penny + b * nickel + c * dime + d * half_dollar ∧ 
   a + b + c + d ≤ n) :=
sorry

end NUMINAMATH_GPT_min_coins_for_any_amount_below_dollar_l2411_241106


namespace NUMINAMATH_GPT_mean_identity_example_l2411_241133

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end NUMINAMATH_GPT_mean_identity_example_l2411_241133


namespace NUMINAMATH_GPT_prime_square_mod_180_l2411_241124

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : 5 < p) :
  ∃ (s : Finset ℕ), s.card = 2 ∧ ∀ r ∈ s, ∃ n : ℕ, p^2 % 180 = r :=
sorry

end NUMINAMATH_GPT_prime_square_mod_180_l2411_241124


namespace NUMINAMATH_GPT_focal_length_of_hyperbola_l2411_241168

theorem focal_length_of_hyperbola (a b p: ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (p_pos : 0 < p) :
  (∃ (F V : ℝ × ℝ), 4 = dist F V ∧ F = (2, 0) ∧ V = (-2, 0)) ∧
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧ (∃ (d : ℝ), d = d / 2 ∧ P = (d, 0))) →
  2 * (Real.sqrt (a^2 + b^2)) = 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_focal_length_of_hyperbola_l2411_241168


namespace NUMINAMATH_GPT_extreme_points_l2411_241112

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem extreme_points (P : ℝ × ℝ) :
  (P = (2, f 2) ∨ P = (-2, f (-2))) ↔ 
  ∃ x : ℝ, x ≠ 0 ∧ (P = (x, f x)) ∧ 
    (∀ ε > 0, f (x - ε) < f x ∧ f x > f (x + ε) ∨ f (x - ε) > f x ∧ f x < f (x + ε)) := 
sorry

end NUMINAMATH_GPT_extreme_points_l2411_241112


namespace NUMINAMATH_GPT_hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l2411_241122

namespace CatchUpProblem

-- Part (a)
theorem hieu_catches_up_beatrice_in_5_minutes :
  ∀ (d_b_walked : ℕ) (relative_speed : ℕ) (catch_up_time : ℕ),
  d_b_walked = 5 / 6 ∧ relative_speed = 10 ∧ catch_up_time = 5 :=
sorry

-- Part (b)(i)
theorem probability_beatrice_hieu_same_place :
  ∀ (total_pairs : ℕ) (valid_pairs : ℕ) (probability : Rat),
  total_pairs = 3600 ∧ valid_pairs = 884 ∧ probability = 221 / 900 :=
sorry

-- Part (b)(ii)
theorem range_of_x_for_meeting_probability :
  ∀ (probability : Rat) (valid_pairs : ℕ) (total_pairs : ℕ) (lower_bound : ℕ) (upper_bound : ℕ),
  probability = 13 / 200 ∧ valid_pairs = 234 ∧ total_pairs = 3600 ∧ 
  lower_bound = 10 ∧ upper_bound = 120 / 11 :=
sorry

end CatchUpProblem

end NUMINAMATH_GPT_hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l2411_241122


namespace NUMINAMATH_GPT_Avianna_red_candles_l2411_241154

theorem Avianna_red_candles (R : ℕ) : 
  (R / 27 = 5 / 3) → R = 45 := 
by
  sorry

end NUMINAMATH_GPT_Avianna_red_candles_l2411_241154


namespace NUMINAMATH_GPT_problem_statement_l2411_241166

def f (x : ℝ) : ℝ := 3 * x^2 - 2
def k (x : ℝ) : ℝ := -2 * x^3 + 2

theorem problem_statement : f (k 2) = 586 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2411_241166


namespace NUMINAMATH_GPT_triangle_internal_region_l2411_241155

-- Define the three lines forming the triangle
def line1 (x y : ℝ) : Prop := x + 2 * y = 2
def line2 (x y : ℝ) : Prop := 2 * x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the inequalities representing the internal region of the triangle
def region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2 * y < 2 ∧ 2 * x + y > 2

-- State that the internal region excluding the boundary is given by the inequalities
theorem triangle_internal_region (x y : ℝ) :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 x y) → region x y :=
  sorry

end NUMINAMATH_GPT_triangle_internal_region_l2411_241155


namespace NUMINAMATH_GPT_time_after_9876_seconds_l2411_241159

noncomputable def currentTime : Nat := 2 * 3600 + 45 * 60 + 0
noncomputable def futureDuration : Nat := 9876
noncomputable def resultingTime : Nat := 5 * 3600 + 29 * 60 + 36

theorem time_after_9876_seconds : 
  (currentTime + futureDuration) % (24 * 3600) = resultingTime := 
by 
  sorry

end NUMINAMATH_GPT_time_after_9876_seconds_l2411_241159


namespace NUMINAMATH_GPT_common_tangents_count_l2411_241132

def circleC1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 15 = 0
def circleC2 : Prop := ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y + 4 = 0

theorem common_tangents_count (C1 : circleC1) (C2 : circleC2) : 
  ∃ (n : ℕ), n = 2 := 
sorry

end NUMINAMATH_GPT_common_tangents_count_l2411_241132


namespace NUMINAMATH_GPT_product_of_first_two_numbers_l2411_241167

theorem product_of_first_two_numbers (A B C : ℕ) (h_coprime: Nat.gcd A B = 1 ∧ Nat.gcd B C = 1 ∧ Nat.gcd A C = 1)
  (h_product: B * C = 1073) (h_sum: A + B + C = 85) : A * B = 703 :=
sorry

end NUMINAMATH_GPT_product_of_first_two_numbers_l2411_241167


namespace NUMINAMATH_GPT_veranda_width_l2411_241115

def room_length : ℕ := 17
def room_width : ℕ := 12
def veranda_area : ℤ := 132

theorem veranda_width :
  ∃ (w : ℝ), (17 + 2 * w) * (12 + 2 * w) - 17 * 12 = 132 ∧ w = 2 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_veranda_width_l2411_241115


namespace NUMINAMATH_GPT_range_of_a_l2411_241129

def p (x : ℝ) : Prop := x ≤ 1/2 ∨ x ≥ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

def not_q (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, not_q x a → p x) ∧ (∃ x : ℝ, ¬ (p x → not_q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2411_241129


namespace NUMINAMATH_GPT_distance_3D_l2411_241110

theorem distance_3D : 
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  d = Real.sqrt 145 :=
by
  let A := (2, -2, 0)
  let B := (8, 8, 3)
  let d := Real.sqrt ((8 - 2) ^ 2 + (8 - (-2)) ^ 2 + (3 - 0) ^ 2)
  dsimp [A, B, d]
  sorry

end NUMINAMATH_GPT_distance_3D_l2411_241110


namespace NUMINAMATH_GPT_Alice_min_speed_l2411_241163

theorem Alice_min_speed
  (distance : Real := 120)
  (bob_speed : Real := 40)
  (alice_delay : Real := 0.5)
  (alice_min_speed : Real := distance / (distance / bob_speed - alice_delay)) :
  alice_min_speed = 48 := 
by
  sorry

end NUMINAMATH_GPT_Alice_min_speed_l2411_241163


namespace NUMINAMATH_GPT_pony_average_speed_l2411_241149

theorem pony_average_speed
  (time_head_start : ℝ)
  (time_catch : ℝ)
  (horse_speed : ℝ)
  (distance_covered_by_horse : ℝ)
  (distance_covered_by_pony : ℝ)
  (pony's_head_start : ℝ)
  : (time_head_start = 3) → (time_catch = 4) → (horse_speed = 35) → 
    (distance_covered_by_horse = horse_speed * time_catch) → 
    (pony's_head_start = time_head_start * v) → 
    (distance_covered_by_pony = pony's_head_start + (v * time_catch)) → 
    (distance_covered_by_horse = distance_covered_by_pony) → v = 20 :=
  by 
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end NUMINAMATH_GPT_pony_average_speed_l2411_241149


namespace NUMINAMATH_GPT_age_difference_l2411_241162

theorem age_difference (A B n : ℕ) (h1 : A = B + n) (h2 : A - 1 = 3 * (B - 1)) (h3 : A = B^2) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l2411_241162


namespace NUMINAMATH_GPT_parabola_with_given_focus_l2411_241105

-- Defining the given condition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Defining the focus coordinates
def focus_coords : ℝ × ℝ := (-3, 0)

-- Proving that the standard equation of the parabola with the left focus of the hyperbola as its focus is y^2 = -12x
theorem parabola_with_given_focus :
  ∃ p : ℝ, (∃ focus : ℝ × ℝ, focus = focus_coords) → 
  ∀ y x : ℝ, y^2 = 4 * p * x → y^2 = -12 * x :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_parabola_with_given_focus_l2411_241105


namespace NUMINAMATH_GPT_quadratic_rewrite_ab_l2411_241102

theorem quadratic_rewrite_ab : 
  ∃ (a b c : ℤ), (16*(x:ℝ)^2 - 40*x + 24 = (a*x + b)^2 + c) ∧ (a * b = -20) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_rewrite_ab_l2411_241102


namespace NUMINAMATH_GPT_geometric_sequence_a6_l2411_241130

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 5 / 2) (h2 : a 2 + a 4 = 5 / 4) 
  (h3 : ∀ n, a (n + 1) = a n * q) : a 6 = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l2411_241130


namespace NUMINAMATH_GPT_arjun_becca_3_different_colors_l2411_241178

open Classical

noncomputable def arjun_becca_probability : ℚ := 
  let arjun_initial := [2, 1, 1, 1] -- 2 red, 1 green, 1 yellow, 1 violet
  let becca_initial := [2, 1] -- 2 black, 1 orange
  
  -- possible cases represented as a list of probabilities
  let cases := [
    (2/5) * (1/4) * (3/5),    -- Case 1: Arjun does move a red ball to Becca, and then processes accordingly
    (3/5) * (1/2) * (1/5),    -- Case 2a: Arjun moves a non-red ball, followed by Becca moving a black ball, concluding in the defined manner
    (3/5) * (1/2) * (3/5)     -- Case 2b: Arjun moves a non-red ball, followed by Becca moving a non-black ball, again concluding appropriately
  ]
  
  -- sum of cases representing the total probability
  let total_probability := List.sum cases
  
  total_probability

theorem arjun_becca_3_different_colors : arjun_becca_probability = 3/10 := 
  by
    simp [arjun_becca_probability]
    sorry

end NUMINAMATH_GPT_arjun_becca_3_different_colors_l2411_241178


namespace NUMINAMATH_GPT_largest_three_digit_n_l2411_241187

theorem largest_three_digit_n (n : ℕ) : 
  (70 * n ≡ 210 [MOD 350]) ∧ (n < 1000) → 
  n = 998 := by
  sorry

end NUMINAMATH_GPT_largest_three_digit_n_l2411_241187


namespace NUMINAMATH_GPT_infinitely_many_k_numbers_unique_k_4_l2411_241140

theorem infinitely_many_k_numbers_unique_k_4 :
  ∀ k : ℕ, (∃ n : ℕ, (∃ r : ℕ, n = r * (r + k)) ∧ (∃ m : ℕ, n = m^2 - k)
          ∧ ∀ N : ℕ, ∃ r : ℕ, ∃ m : ℕ, N < r ∧ (r * (r + k) = m^2 - k)) ↔ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_k_numbers_unique_k_4_l2411_241140


namespace NUMINAMATH_GPT_most_marbles_l2411_241152

def total_marbles := 24
def red_marble_fraction := 1 / 4
def red_marbles := red_marble_fraction * total_marbles
def blue_marbles := red_marbles + 6
def yellow_marbles := total_marbles - red_marbles - blue_marbles

theorem most_marbles : blue_marbles > red_marbles ∧ blue_marbles > yellow_marbles :=
by
  sorry

end NUMINAMATH_GPT_most_marbles_l2411_241152


namespace NUMINAMATH_GPT_no_maximal_radius_of_inscribed_cylinder_l2411_241191

theorem no_maximal_radius_of_inscribed_cylinder
  (base_radius_cone : ℝ) (height_cone : ℝ)
  (h_base_radius : base_radius_cone = 5) (h_height : height_cone = 10) :
  ¬ ∃ r : ℝ, 0 < r ∧ r < 5 ∧
    ∀ t : ℝ, 0 < t ∧ t < 5 → 2 * Real.pi * (10 * r - r ^ 2) ≥ 2 * Real.pi * (10 * t - t ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_no_maximal_radius_of_inscribed_cylinder_l2411_241191


namespace NUMINAMATH_GPT_f_zero_f_odd_f_range_l2411_241108

noncomputable def f : ℝ → ℝ := sorry

-- Add the hypothesis for the conditions
axiom f_domain : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_value_one_third : f (1 / 3) = 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 0

-- (1) Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- (2) Prove that f(x) is odd
theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

-- (3) Given f(x) + f(2 + x) < 2, find the range of x
theorem f_range (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 := sorry

end NUMINAMATH_GPT_f_zero_f_odd_f_range_l2411_241108


namespace NUMINAMATH_GPT_percentage_import_tax_l2411_241121

theorem percentage_import_tax (total_value import_paid excess_amount taxable_amount : ℝ) 
  (h1 : total_value = 2570) 
  (h2 : import_paid = 109.90) 
  (h3 : excess_amount = 1000) 
  (h4 : taxable_amount = total_value - excess_amount) : 
  taxable_amount = 1570 →
  (import_paid / taxable_amount) * 100 = 7 := 
by
  intros h_taxable_amount
  simp [h1, h2, h3, h4, h_taxable_amount]
  sorry -- Proof goes here

end NUMINAMATH_GPT_percentage_import_tax_l2411_241121


namespace NUMINAMATH_GPT_geric_bills_l2411_241175

variable (G K J : ℕ)

theorem geric_bills (h1 : G = 2 * K) 
                    (h2 : K = J - 2) 
                    (h3 : J = 7 + 3) : 
    G = 16 := by
  sorry

end NUMINAMATH_GPT_geric_bills_l2411_241175


namespace NUMINAMATH_GPT_range_of_a_l2411_241139

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ x^2 + (a - 1) * x + 1 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2411_241139


namespace NUMINAMATH_GPT_amount_spent_on_milk_is_1500_l2411_241137

def total_salary (saved : ℕ) (saving_percent : ℕ) : ℕ := 
  saved / (saving_percent / 100)

def total_spent_excluding_milk (rent groceries education petrol misc : ℕ) : ℕ := 
  rent + groceries + education + petrol + misc

def amount_spent_on_milk (total_salary total_spent savings : ℕ) : ℕ := 
  total_salary - total_spent - savings

theorem amount_spent_on_milk_is_1500 :
  let rent := 5000
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 2500
  let savings := 2000
  let saving_percent := 10
  let salary := total_salary savings saving_percent
  let spent_excluding_milk := total_spent_excluding_milk rent groceries education petrol misc
  amount_spent_on_milk salary spent_excluding_milk savings = 1500 :=
by {
  sorry
}

end NUMINAMATH_GPT_amount_spent_on_milk_is_1500_l2411_241137


namespace NUMINAMATH_GPT_billy_initial_lemon_heads_l2411_241183

theorem billy_initial_lemon_heads (n f : ℕ) (h_friends : f = 6) (h_eat : n = 12) :
  f * n = 72 := 
by
  -- Proceed by proving the statement using Lean
  sorry

end NUMINAMATH_GPT_billy_initial_lemon_heads_l2411_241183


namespace NUMINAMATH_GPT_gcd_98_63_l2411_241188

-- The statement of the problem in Lean 4
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_98_63_l2411_241188


namespace NUMINAMATH_GPT_functional_equation_solution_l2411_241123

theorem functional_equation_solution (f : ℚ → ℚ) (H : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := 
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2411_241123


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2411_241165

theorem sum_of_three_numbers (A B C : ℕ) 
  (h1 : B = 30)
  (h2 : A * 3 = 2 * B)
  (h3 : C * 5 = 8 * B) : 
  A + B + C = 98 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2411_241165


namespace NUMINAMATH_GPT_find_c_l2411_241158

theorem find_c (c : ℝ) (h : ∃ (f : ℝ → ℝ), (f = λ x => c * x^3 + 23 * x^2 - 5 * c * x + 55) ∧ f (-5) = 0) : c = 6.3 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l2411_241158


namespace NUMINAMATH_GPT_dormouse_is_thief_l2411_241160

-- Definitions of the suspects
inductive Suspect
| MarchHare
| Hatter
| Dormouse

open Suspect

-- Definitions of the statement conditions
def statement (s : Suspect) : Suspect :=
match s with
| MarchHare => Hatter
| Hatter => sorry -- Sonya and Hatter's testimonies are not recorded
| Dormouse => sorry -- Sonya and Hatter's testimonies are not recorded

-- Condition that only the thief tells the truth
def tells_truth (thief : Suspect) (s : Suspect) : Prop :=
s = thief

-- Conditions of the problem
axiom condition1 : statement MarchHare = Hatter
axiom condition2 : ∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse

-- Proposition that Dormouse (Sonya) is the thief
theorem dormouse_is_thief : (∃ t, tells_truth t MarchHare ∧ ¬ tells_truth t Hatter ∧ ¬ tells_truth t Dormouse) → t = Dormouse :=
sorry

end NUMINAMATH_GPT_dormouse_is_thief_l2411_241160


namespace NUMINAMATH_GPT_rich_walks_ratio_is_2_l2411_241177

-- Define the conditions in the problem
def house_to_sidewalk : ℕ := 20
def sidewalk_to_end : ℕ := 200
def total_distance_walked : ℕ := 1980
def ratio_after_left_to_so_far (x : ℕ) : ℕ := (house_to_sidewalk + sidewalk_to_end) * x / (house_to_sidewalk + sidewalk_to_end)

-- Main theorem to prove the ratio is 2:1
theorem rich_walks_ratio_is_2 (x : ℕ) (h : 2 * ((house_to_sidewalk + sidewalk_to_end) * 2 + house_to_sidewalk + sidewalk_to_end / 2 * 3 ) = total_distance_walked) :
  ratio_after_left_to_so_far x = 2 :=
by
  sorry

end NUMINAMATH_GPT_rich_walks_ratio_is_2_l2411_241177


namespace NUMINAMATH_GPT_find_principal_l2411_241113

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℝ) (hSI : SI = 4025.25) (hR : R = 9) (hT : T = 5) :
    let P := SI / (R * T / 100)
    P = 8950 :=
by
  -- we will put proof steps here
  sorry

end NUMINAMATH_GPT_find_principal_l2411_241113


namespace NUMINAMATH_GPT_girl_scouts_short_amount_l2411_241189

-- Definitions based on conditions
def amount_earned : ℝ := 30
def pool_entry_cost_per_person : ℝ := 2.50
def num_people : ℕ := 10
def transportation_fee_per_person : ℝ := 1.25
def snack_cost_per_person : ℝ := 3.00

-- Calculate individual costs
def total_pool_entry_cost : ℝ := pool_entry_cost_per_person * num_people
def total_transportation_fee : ℝ := transportation_fee_per_person * num_people
def total_snack_cost : ℝ := snack_cost_per_person * num_people

-- Calculate total expenses
def total_expenses : ℝ := total_pool_entry_cost + total_transportation_fee + total_snack_cost

-- The amount left after expenses
def amount_left : ℝ := amount_earned - total_expenses

-- Proof problem statement
theorem girl_scouts_short_amount : amount_left = -37.50 := by
  sorry

end NUMINAMATH_GPT_girl_scouts_short_amount_l2411_241189


namespace NUMINAMATH_GPT_calculate_AE_l2411_241172

variable {k : ℝ} (A B C D E : Type*)

namespace Geometry

def shared_angle (A B C : Type*) : Prop := sorry -- assumes triangles share angle A

def prop_constant_proportion (AB AC AD AE : ℝ) (k : ℝ) : Prop :=
  AB * AC = k * AD * AE

theorem calculate_AE
  (A B C D E : Type*) 
  (AB AC AD AE : ℝ)
  (h_shared : shared_angle A B C)
  (h_AB : AB = 5)
  (h_AC : AC = 7)
  (h_AD : AD = 2)
  (h_proportion : prop_constant_proportion AB AC AD AE k)
  (h_k : k = 1) :
  AE = 17.5 := 
sorry

end Geometry

end NUMINAMATH_GPT_calculate_AE_l2411_241172


namespace NUMINAMATH_GPT_quadratic_expression_value_l2411_241182

theorem quadratic_expression_value (a : ℝ) (h : a^2 - 2 * a - 3 = 0) : a^2 - 2 * a + 1 = 4 :=
by 
  -- Proof omitted for clarity in this part
  sorry 

end NUMINAMATH_GPT_quadratic_expression_value_l2411_241182


namespace NUMINAMATH_GPT_average_math_test_score_l2411_241180

theorem average_math_test_score :
    let june_score := 97
    let patty_score := 85
    let josh_score := 100
    let henry_score := 94
    let num_children := 4
    let total_score := june_score + patty_score + josh_score + henry_score
    total_score / num_children = 94 := by
  sorry

end NUMINAMATH_GPT_average_math_test_score_l2411_241180


namespace NUMINAMATH_GPT_evaluate_expression_at_zero_l2411_241142

theorem evaluate_expression_at_zero :
  (0^2 + 5 * 0 - 10) = -10 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_zero_l2411_241142


namespace NUMINAMATH_GPT_total_profit_calculation_l2411_241153

-- Define the parameters of the problem
def rajan_investment : ℕ := 20000
def rakesh_investment : ℕ := 25000
def mukesh_investment : ℕ := 15000
def rajan_investment_time : ℕ := 12 -- in months
def rakesh_investment_time : ℕ := 4 -- in months
def mukesh_investment_time : ℕ := 8 -- in months
def rajan_final_share : ℕ := 2400

-- Calculation for total profit
def total_profit (rajan_investment rakesh_investment mukesh_investment
                  rajan_investment_time rakesh_investment_time mukesh_investment_time
                  rajan_final_share : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_investment_time
  let rakesh_share := rakesh_investment * rakesh_investment_time
  let mukesh_share := mukesh_investment * mukesh_investment_time
  let total_investment := rajan_share + rakesh_share + mukesh_share
  (rajan_final_share * total_investment) / rajan_share

-- Proof problem statement
theorem total_profit_calculation :
  total_profit rajan_investment rakesh_investment mukesh_investment
               rajan_investment_time rakesh_investment_time mukesh_investment_time
               rajan_final_share = 4600 :=
by sorry

end NUMINAMATH_GPT_total_profit_calculation_l2411_241153


namespace NUMINAMATH_GPT_units_digit_diff_is_seven_l2411_241109

noncomputable def units_digit_resulting_difference (a b c : ℕ) (h1 : a = c - 3) :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let difference := original - reversed
  difference % 10

theorem units_digit_diff_is_seven (a b c : ℕ) (h1 : a = c - 3) :
  units_digit_resulting_difference a b c h1 = 7 :=
by sorry

end NUMINAMATH_GPT_units_digit_diff_is_seven_l2411_241109


namespace NUMINAMATH_GPT_tan_alpha_minus_beta_alpha_plus_beta_l2411_241144

variable (α β : ℝ)

-- Conditions as hypotheses
axiom tan_alpha : Real.tan α = 2
axiom tan_beta : Real.tan β = -1 / 3
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : Real.pi / 2 < β ∧ β < Real.pi

-- Proof statements
theorem tan_alpha_minus_beta : Real.tan (α - β) = 7 := by
  sorry

theorem alpha_plus_beta : α + β = 5 * Real.pi / 4 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_beta_alpha_plus_beta_l2411_241144


namespace NUMINAMATH_GPT_tan_70_sin_80_eq_neg1_l2411_241186

theorem tan_70_sin_80_eq_neg1 :
  (Real.tan 70 * Real.sin 80 * (Real.sqrt 3 * Real.tan 20 - 1) = -1) :=
sorry

end NUMINAMATH_GPT_tan_70_sin_80_eq_neg1_l2411_241186


namespace NUMINAMATH_GPT_find_y_l2411_241164

theorem find_y (y : ℝ) : (∃ y : ℝ, (4, y) ≠ (2, -3) ∧ ((-3 - y) / (2 - 4) = 1)) → y = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2411_241164


namespace NUMINAMATH_GPT_scientific_notation_correct_l2411_241101

def number_in_scientific_notation : ℝ := 1600000
def expected_scientific_notation : ℝ := 1.6 * 10^6

theorem scientific_notation_correct :
  number_in_scientific_notation = expected_scientific_notation := by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2411_241101


namespace NUMINAMATH_GPT_logarithm_function_decreasing_l2411_241151

theorem logarithm_function_decreasing (a : ℝ) : 
  (∀ x ∈ Set.Ici (-1), (3 * x^2 - a * x + 5) ≤ (3 * x^2 - a * (x + 1) + 5)) ↔ (-8 < a ∧ a ≤ -6) :=
by
  sorry

end NUMINAMATH_GPT_logarithm_function_decreasing_l2411_241151


namespace NUMINAMATH_GPT_highest_place_value_quotient_and_remainder_l2411_241173

-- Conditions
def dividend := 438
def divisor := 4

-- Theorem stating that the highest place value of the quotient is the hundreds place, and the remainder is 2
theorem highest_place_value_quotient_and_remainder : 
  (dividend = divisor * (dividend / divisor) + (dividend % divisor)) ∧ 
  ((dividend / divisor) >= 100) ∧ 
  ((dividend % divisor) = 2) :=
by
  sorry

end NUMINAMATH_GPT_highest_place_value_quotient_and_remainder_l2411_241173


namespace NUMINAMATH_GPT_part1_part2_l2411_241120

namespace Proof

def A (a b : ℝ) : ℝ := 3 * a ^ 2 - 4 * a * b
def B (a b : ℝ) : ℝ := a ^ 2 + 2 * a * b

theorem part1 (a b : ℝ) : 2 * A a b - 3 * B a b = 3 * a ^ 2 - 14 * a * b := by
  sorry
  
theorem part2 (a b : ℝ) (h : |3 * a + 1| + (2 - 3 * b) ^ 2 = 0) : A a b - 2 * B a b = 5 / 3 := by
  have ha : a = -1 / 3 := by
    sorry
  have hb : b = 2 / 3 := by
    sorry
  rw [ha, hb]
  sorry

end Proof

end NUMINAMATH_GPT_part1_part2_l2411_241120


namespace NUMINAMATH_GPT_batsman_average_runs_l2411_241174

theorem batsman_average_runs
  (average_20_matches : ℕ → ℕ)
  (average_10_matches : ℕ → ℕ)
  (h1 : average_20_matches = 20 * 40)
  (h2 : average_10_matches = 10 * 13) :
  (average_20_matches + average_10_matches) / 30 = 31 := 
by 
  sorry

end NUMINAMATH_GPT_batsman_average_runs_l2411_241174


namespace NUMINAMATH_GPT_problem_l2411_241193

noncomputable def cubeRoot (x : ℝ) : ℝ :=
  x ^ (1 / 3)

theorem problem (t : ℝ) (h : t = 1 / (1 - cubeRoot 2)) :
  t = (1 + cubeRoot 2) * (1 + cubeRoot 4) :=
by
  sorry

end NUMINAMATH_GPT_problem_l2411_241193


namespace NUMINAMATH_GPT_fibonacci_contains_21_l2411_241157

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Theorem statement: Proving that 21 is in the Fibonacci sequence
theorem fibonacci_contains_21 : ∃ n, fibonacci n = 21 :=
by
  sorry

end NUMINAMATH_GPT_fibonacci_contains_21_l2411_241157


namespace NUMINAMATH_GPT_reflection_matrix_correct_l2411_241131

-- Definitions based on the conditions given in the problem
def reflect_over_line_y_eq_x_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![1, 0]]

-- The main theorem to state the equivalence
theorem reflection_matrix_correct :
  reflect_over_line_y_eq_x_matrix = ![![0, 1], ![1, 0]] :=
by
  sorry

end NUMINAMATH_GPT_reflection_matrix_correct_l2411_241131


namespace NUMINAMATH_GPT_circle_center_l2411_241195

theorem circle_center (x y : ℝ) : (x^2 - 6 * x + y^2 + 2 * y = 20) → (x,y) = (3,-1) :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_center_l2411_241195


namespace NUMINAMATH_GPT_find_quantities_l2411_241107

variables {a b x y : ℝ}

-- Original total expenditure condition
axiom h1 : a * x + b * y = 1500

-- New prices and quantities for the first scenario
axiom h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529

-- New prices and quantities for the second scenario
axiom h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5

-- Inequality constraint
axiom h4 : 205 < 2 * x + y ∧ 2 * x + y < 210

-- Range for 'a'
axiom h5 : 17.5 < a ∧ a < 18.5

-- Proving x and y are specific values.
theorem find_quantities :
  x = 76 ∧ y = 55 :=
sorry

end NUMINAMATH_GPT_find_quantities_l2411_241107


namespace NUMINAMATH_GPT_cole_drive_time_correct_l2411_241169

noncomputable def cole_drive_time : ℕ :=
  let distance_to_work := 45 -- derived from the given problem   
  let speed_to_work := 30
  let time_to_work := distance_to_work / speed_to_work -- in hours
  (time_to_work * 60 : ℕ) -- converting hours to minutes

theorem cole_drive_time_correct
  (speed_to_work speed_return: ℕ)
  (total_time: ℕ)
  (H1: speed_to_work = 30)
  (H2: speed_return = 90)
  (H3: total_time = 2):
  cole_drive_time = 90 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cole_drive_time_correct_l2411_241169


namespace NUMINAMATH_GPT_part1_part2_l2411_241194

section

variable (a x : ℝ)

def A : Set ℝ := { x | x ≤ -1 } ∪ { x | x ≥ 5 }
def B (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 2 }

-- Part 1
theorem part1 (h : a = -1) :
  B a = { x | -2 ≤ x ∧ x ≤ 1 } ∧
  (A ∩ B a) = { x | -2 ≤ x ∧ x ≤ -1 } ∧
  (A ∪ B a) = { x | x ≤ 1 ∨ x ≥ 5 } := 
sorry

-- Part 2
theorem part2 (h : A ∩ B a = B a) :
  a ≤ -3 ∨ a > 2 := 
sorry

end

end NUMINAMATH_GPT_part1_part2_l2411_241194


namespace NUMINAMATH_GPT_tina_sells_more_than_katya_l2411_241114

noncomputable def katya_rev : ℝ := 8 * 1.5
noncomputable def ricky_rev : ℝ := 9 * 2.0
noncomputable def combined_rev : ℝ := katya_rev + ricky_rev
noncomputable def tina_target : ℝ := 2 * combined_rev
noncomputable def tina_glasses : ℝ := tina_target / 3.0
noncomputable def difference_glasses : ℝ := tina_glasses - 8

theorem tina_sells_more_than_katya :
  difference_glasses = 12 := by
  sorry

end NUMINAMATH_GPT_tina_sells_more_than_katya_l2411_241114


namespace NUMINAMATH_GPT_mr_castiel_sausages_l2411_241156

theorem mr_castiel_sausages (S : ℕ) :
  S * (3 / 5) * (1 / 2) * (1 / 4) * (3 / 4) = 45 → S = 600 :=
by
  sorry

end NUMINAMATH_GPT_mr_castiel_sausages_l2411_241156


namespace NUMINAMATH_GPT_robert_ate_more_chocolates_l2411_241184

-- Define the number of chocolates eaten by Robert and Nickel
def robert_chocolates : ℕ := 12
def nickel_chocolates : ℕ := 3

-- State the problem as a theorem to prove
theorem robert_ate_more_chocolates :
  robert_chocolates - nickel_chocolates = 9 :=
by
  sorry

end NUMINAMATH_GPT_robert_ate_more_chocolates_l2411_241184


namespace NUMINAMATH_GPT_find_b_l2411_241135

variable (x : ℝ)

theorem find_b (a b: ℝ) (h1 : x + 1/x = a) (h2 : x^3 + 1/x^3 = b) (ha : a = 3): b = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2411_241135


namespace NUMINAMATH_GPT_simplify_expression_l2411_241179

variable (x : ℝ)

theorem simplify_expression (h : x ≠ 1) : (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2411_241179


namespace NUMINAMATH_GPT_melanie_total_dimes_l2411_241145

/-- Melanie had 7 dimes in her bank. Her dad gave her 8 dimes. Her mother gave her 4 dimes. -/
def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

/-- How many dimes does Melanie have now? -/
theorem melanie_total_dimes : initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end NUMINAMATH_GPT_melanie_total_dimes_l2411_241145


namespace NUMINAMATH_GPT_tangent_line_through_P_is_correct_l2411_241185

-- Define the point P
def P : ℝ × ℝ := (2, 4)

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent line equation to prove
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- Problem statement in Lean 4
theorem tangent_line_through_P_is_correct :
  C P.1 P.2 → tangent_line P.1 P.2 :=
by
  intros hC
  sorry

end NUMINAMATH_GPT_tangent_line_through_P_is_correct_l2411_241185


namespace NUMINAMATH_GPT_x_divisible_by_5_l2411_241118

theorem x_divisible_by_5 (x y : ℕ) (hx : x > 1) (h : 2 * x^2 - 1 = y^15) : 5 ∣ x := 
sorry

end NUMINAMATH_GPT_x_divisible_by_5_l2411_241118


namespace NUMINAMATH_GPT_zoo_visitors_sunday_l2411_241199

-- Definitions based on conditions
def friday_visitors : ℕ := 1250
def saturday_multiplier : ℚ := 3
def sunday_decrease_percent : ℚ := 0.15

-- Assert the equivalence
theorem zoo_visitors_sunday : 
  let saturday_visitors := friday_visitors * saturday_multiplier
  let sunday_visitors := saturday_visitors * (1 - sunday_decrease_percent)
  round (sunday_visitors : ℚ) = 3188 :=
by
  sorry

end NUMINAMATH_GPT_zoo_visitors_sunday_l2411_241199


namespace NUMINAMATH_GPT_inv_of_15_mod_1003_l2411_241147

theorem inv_of_15_mod_1003 : ∃ x : ℕ, x ≤ 1002 ∧ 15 * x ≡ 1 [MOD 1003] ∧ x = 937 :=
by sorry

end NUMINAMATH_GPT_inv_of_15_mod_1003_l2411_241147


namespace NUMINAMATH_GPT_binom_odd_n_eq_2_pow_m_minus_1_l2411_241117

open Nat

/-- For which n will binom n k be odd for every 0 ≤ k ≤ n?
    Prove that n = 2^m - 1 for some m ≥ 1. -/
theorem binom_odd_n_eq_2_pow_m_minus_1 (n : ℕ) :
  (∀ k : ℕ, k ≤ n → Nat.choose n k % 2 = 1) ↔ (∃ m : ℕ, m ≥ 1 ∧ n = 2^m - 1) :=
by
  sorry

end NUMINAMATH_GPT_binom_odd_n_eq_2_pow_m_minus_1_l2411_241117


namespace NUMINAMATH_GPT_simplify_fraction_l2411_241127

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l2411_241127


namespace NUMINAMATH_GPT_find_a_l2411_241170

-- Defining the curve y in terms of x and a
def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Defining the derivative of the curve
def derivative (x : ℝ) (a : ℝ) : ℝ := 4*x^3 + 2*a*x

-- The proof statement asserting the value of a
theorem find_a (a : ℝ) (h1 : derivative (-1) a = 8): a = -6 :=
by
  -- we assume here the necessary calculations and logical steps to prove the theorem
  sorry

end NUMINAMATH_GPT_find_a_l2411_241170


namespace NUMINAMATH_GPT_total_arrangements_l2411_241197

-- Question: 
-- Given 6 teachers and 4 schools with specific constraints, 
-- prove that the number of different ways to arrange the teachers is 240.

def teachers : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def schools : List Nat := [1, 2, 3, 4]

def B_and_D_in_same_school (assignment: Char → Nat) : Prop :=
  assignment 'B' = assignment 'D'

def each_school_has_at_least_one_teacher (assignment: Char → Nat) : Prop :=
  ∀ s ∈ schools, ∃ t ∈ teachers, assignment t = s

noncomputable def num_arrangements : Nat := sorry -- This would actually involve complex combinatorial calculations

theorem total_arrangements : num_arrangements = 240 :=
  sorry

end NUMINAMATH_GPT_total_arrangements_l2411_241197


namespace NUMINAMATH_GPT_find_a4_plus_b4_l2411_241119

theorem find_a4_plus_b4 (a b : ℝ)
  (h1 : (a^2 - b^2)^2 = 100)
  (h2 : a^3 * b^3 = 512) :
  a^4 + b^4 = 228 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_plus_b4_l2411_241119


namespace NUMINAMATH_GPT_meaning_of_sum_of_squares_l2411_241146

theorem meaning_of_sum_of_squares (a b : ℝ) : a ^ 2 + b ^ 2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_meaning_of_sum_of_squares_l2411_241146
