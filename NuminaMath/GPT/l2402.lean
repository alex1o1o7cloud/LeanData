import Mathlib

namespace NUMINAMATH_GPT_line_perpendicular_value_of_a_l2402_240231

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end NUMINAMATH_GPT_line_perpendicular_value_of_a_l2402_240231


namespace NUMINAMATH_GPT_perimeter_right_triangle_l2402_240219

-- Given conditions
def area : ℝ := 200
def b : ℝ := 20

-- Mathematical problem
theorem perimeter_right_triangle :
  ∀ (x c : ℝ), 
  (1 / 2) * b * x = area →
  c^2 = x^2 + b^2 →
  x + b + c = 40 + 20 * Real.sqrt 2 := 
  by
  sorry

end NUMINAMATH_GPT_perimeter_right_triangle_l2402_240219


namespace NUMINAMATH_GPT_part_I_part_II_l2402_240262

section problem_1

def f (x : ℝ) (a : ℝ) := |x - 3| - |x + a|

theorem part_I (x : ℝ) (hx : f x 2 < 1) : 0 < x :=
by
  sorry

theorem part_II (a : ℝ) (h : ∀ (x : ℝ), f x a ≤ 2 * a) : 3 ≤ a :=
by
  sorry

end problem_1

end NUMINAMATH_GPT_part_I_part_II_l2402_240262


namespace NUMINAMATH_GPT_unit_vector_perpendicular_to_a_l2402_240285

-- Definitions of a vector and the properties of unit and perpendicular vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def is_unit_vector (v : Vector2D) : Prop :=
  v.x ^ 2 + v.y ^ 2 = 1

def is_perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

-- Given vector a
def a : Vector2D := ⟨3, 4⟩

-- Coordinates of the unit vector that is perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ (b : Vector2D), is_unit_vector b ∧ is_perpendicular a b ∧
  (b = ⟨-4 / 5, 3 / 5⟩ ∨ b = ⟨4 / 5, -3 / 5⟩) :=
sorry

end NUMINAMATH_GPT_unit_vector_perpendicular_to_a_l2402_240285


namespace NUMINAMATH_GPT_sum_of_cubes_three_consecutive_divisible_by_three_l2402_240261

theorem sum_of_cubes_three_consecutive_divisible_by_three (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 3 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_three_consecutive_divisible_by_three_l2402_240261


namespace NUMINAMATH_GPT_smaller_circle_radius_l2402_240279

theorem smaller_circle_radius
  (radius_largest : ℝ)
  (h1 : radius_largest = 10)
  (aligned_circles : ℝ)
  (h2 : 4 * aligned_circles = 2 * radius_largest) :
  aligned_circles / 2 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l2402_240279


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2402_240276

theorem geometric_sequence_common_ratio
  (q : ℝ) (a_n : ℕ → ℝ)
  (h_inc : ∀ n, a_n (n + 1) = q * a_n n ∧ q > 1)
  (h_a2 : a_n 2 = 2)
  (h_a4_a3 : a_n 4 - a_n 3 = 4) : 
  q = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2402_240276


namespace NUMINAMATH_GPT_find_a_b_sum_l2402_240223

-- Conditions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f_prime (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem find_a_b_sum (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a + b = 7 := 
sorry

end NUMINAMATH_GPT_find_a_b_sum_l2402_240223


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l2402_240242

noncomputable def is_power_of_prime (n : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ n = p ^ k

theorem number_of_ordered_pairs :
  (∃ (n : ℕ), n = 29 ∧
    ∀ (x y : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ x ≤ 2020 ∧ y ≤ 2020 →
    is_power_of_prime (3 * x^2 + 10 * x * y + 3 * y^2) → n = 29) :=
by
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l2402_240242


namespace NUMINAMATH_GPT_prepaid_card_cost_correct_l2402_240222

noncomputable def prepaid_phone_card_cost
    (cost_per_minute : ℝ) (call_minutes : ℝ) (remaining_credit : ℝ) : ℝ :=
  remaining_credit + (call_minutes * cost_per_minute)

theorem prepaid_card_cost_correct :
  let cost_per_minute := 0.16
  let call_minutes := 22
  let remaining_credit := 26.48
  prepaid_phone_card_cost cost_per_minute call_minutes remaining_credit = 30.00 := by
  sorry

end NUMINAMATH_GPT_prepaid_card_cost_correct_l2402_240222


namespace NUMINAMATH_GPT_number_of_special_divisors_l2402_240275

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end NUMINAMATH_GPT_number_of_special_divisors_l2402_240275


namespace NUMINAMATH_GPT_rationalize_sqrt_5_div_18_l2402_240250

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end NUMINAMATH_GPT_rationalize_sqrt_5_div_18_l2402_240250


namespace NUMINAMATH_GPT_land_for_crop_production_l2402_240204

-- Conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def cattle_rearing : ℕ := 40

-- Proof statement defining the goal
theorem land_for_crop_production : 
  total_land - (house_and_machinery + future_expansion + cattle_rearing) = 70 := 
by
  sorry

end NUMINAMATH_GPT_land_for_crop_production_l2402_240204


namespace NUMINAMATH_GPT_smallest_natural_number_l2402_240290

theorem smallest_natural_number :
  ∃ N : ℕ, ∃ f : ℕ → ℕ → ℕ, 
  f (f (f 9 8 - f 7 6) 5 + 4 - f 3 2) 1 = N ∧
  N = 1 := 
by sorry

end NUMINAMATH_GPT_smallest_natural_number_l2402_240290


namespace NUMINAMATH_GPT_right_triangle_third_side_l2402_240256

theorem right_triangle_third_side (a b : ℕ) (c : ℝ) (h₁: a = 3) (h₂: b = 4) (h₃: ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2)) ∨ (c^2 + b^2 = a^2)):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l2402_240256


namespace NUMINAMATH_GPT_spadesuit_problem_l2402_240245

def spadesuit (x y : ℝ) := (x + y) * (x - y)

theorem spadesuit_problem : spadesuit 5 (spadesuit 3 2) = 0 := by
  sorry

end NUMINAMATH_GPT_spadesuit_problem_l2402_240245


namespace NUMINAMATH_GPT_percentage_return_on_investment_l2402_240265

theorem percentage_return_on_investment (dividend_rate : ℝ) (face_value : ℝ) (purchase_price : ℝ) (return_percentage : ℝ) :
  dividend_rate = 0.125 → face_value = 40 → purchase_price = 20 → return_percentage = 25 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_return_on_investment_l2402_240265


namespace NUMINAMATH_GPT_lines_parallel_l2402_240280

theorem lines_parallel (m : ℝ) : 
  (m = 2 ↔ ∀ x y : ℝ, (2 * x - m * y - 1 = 0) ∧ ((m - 1) * x - y + 1 = 0) → 
  (∃ k : ℝ, (2 * x - m * y - 1 = k * ((m - 1) * x - y + 1)))) :=
by sorry

end NUMINAMATH_GPT_lines_parallel_l2402_240280


namespace NUMINAMATH_GPT_jason_total_spending_l2402_240283

def cost_of_shorts : ℝ := 14.28
def cost_of_jacket : ℝ := 4.74
def total_spent : ℝ := 19.02

theorem jason_total_spending : cost_of_shorts + cost_of_jacket = total_spent :=
by
  sorry

end NUMINAMATH_GPT_jason_total_spending_l2402_240283


namespace NUMINAMATH_GPT_complement_intersection_l2402_240228

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 3 ≤ x}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A ∩ B) ↔ (0 ≤ x ∧ x < 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_intersection_l2402_240228


namespace NUMINAMATH_GPT_total_cost_all_children_l2402_240211

-- Defining the constants and conditions
def regular_tuition : ℕ := 45
def early_bird_discount : ℕ := 15
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10
def weekend_class_extra_cost : ℕ := 20
def multi_instrument_discount : ℕ := 10

def Ali_cost : ℕ := regular_tuition - early_bird_discount
def Matt_cost : ℕ := regular_tuition - first_sibling_discount
def Jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount
def Sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount

-- Proof statement
theorem total_cost_all_children : Ali_cost + Matt_cost + Jane_cost + Sarah_cost = 150 := by
  sorry

end NUMINAMATH_GPT_total_cost_all_children_l2402_240211


namespace NUMINAMATH_GPT_triangle_area_example_l2402_240216

-- Define the right triangle DEF with angle at D being 45 degrees and DE = 8 units
noncomputable def area_of_45_45_90_triangle (DE : ℝ) (angle_d : ℝ) (h_angle : angle_d = 45) (h_DE : DE = 8) : ℝ :=
  1 / 2 * DE * DE

-- State the theorem to prove the area
theorem triangle_area_example {DE : ℝ} {angle_d : ℝ} (h_angle : angle_d = 45) (h_DE : DE = 8) :
  area_of_45_45_90_triangle DE angle_d h_angle h_DE = 32 := 
sorry

end NUMINAMATH_GPT_triangle_area_example_l2402_240216


namespace NUMINAMATH_GPT_height_of_first_building_l2402_240296

theorem height_of_first_building (h : ℕ) (h_condition : h + 2 * h + 9 * h = 7200) : h = 600 :=
by
  sorry

end NUMINAMATH_GPT_height_of_first_building_l2402_240296


namespace NUMINAMATH_GPT_find_set_of_x_l2402_240210

noncomputable def exponential_inequality_solution (x : ℝ) : Prop :=
  1 < Real.exp x ∧ Real.exp x < 2

theorem find_set_of_x (x : ℝ) :
  exponential_inequality_solution x ↔ 0 < x ∧ x < Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_find_set_of_x_l2402_240210


namespace NUMINAMATH_GPT_population_hypothetical_town_l2402_240213

theorem population_hypothetical_town :
  ∃ (a b c : ℕ), a^2 + 150 = b^2 + 1 ∧ b^2 + 1 + 150 = c^2 ∧ a^2 = 5476 :=
by {
  sorry
}

end NUMINAMATH_GPT_population_hypothetical_town_l2402_240213


namespace NUMINAMATH_GPT_inscribed_circle_radius_eq_four_l2402_240212

theorem inscribed_circle_radius_eq_four
  (A p s r : ℝ)
  (hA : A = 2 * p)
  (hp : p = 2 * s)
  (hArea : A = r * s) :
  r = 4 :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_eq_four_l2402_240212


namespace NUMINAMATH_GPT_geometric_sequence_a9_value_l2402_240243

theorem geometric_sequence_a9_value {a : ℕ → ℝ} (q a1 : ℝ) 
  (h_geom : ∀ n, a n = a1 * q ^ n)
  (h_a3 : a 3 = 2)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (h_sum : S 12 = 4 * S 6) : a 9 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_a9_value_l2402_240243


namespace NUMINAMATH_GPT_liar_and_truth_tellers_l2402_240264

-- Define the characters and their nature (truth-teller or liar)
inductive Character : Type
| Kikimora
| Leshy
| Vodyanoy

def always_truthful (c : Character) : Prop := sorry
def always_lying (c : Character) : Prop := sorry

axiom kikimora_statement : always_lying Character.Kikimora
axiom leshy_statement : ∃ l₁ l₂ : Character, l₁ ≠ l₂ ∧ always_lying l₁ ∧ always_lying l₂
axiom vodyanoy_statement : true -- Vodyanoy's silence

-- Proof that Kikimora and Vodyanoy are liars and Leshy is truthful
theorem liar_and_truth_tellers :
  always_lying Character.Kikimora ∧
  always_lying Character.Vodyanoy ∧
  always_truthful Character.Leshy := sorry

end NUMINAMATH_GPT_liar_and_truth_tellers_l2402_240264


namespace NUMINAMATH_GPT_smaller_number_l2402_240258

theorem smaller_number (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end NUMINAMATH_GPT_smaller_number_l2402_240258


namespace NUMINAMATH_GPT_polygon_diagonals_l2402_240244

theorem polygon_diagonals (n : ℕ) (h : n - 3 ≤ 6) : n = 9 :=
by sorry

end NUMINAMATH_GPT_polygon_diagonals_l2402_240244


namespace NUMINAMATH_GPT_unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l2402_240221

theorem unique_solution_x_ln3_plus_x_ln4_eq_x_ln5 :
  ∃! x : ℝ, 0 < x ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) := sorry

end NUMINAMATH_GPT_unique_solution_x_ln3_plus_x_ln4_eq_x_ln5_l2402_240221


namespace NUMINAMATH_GPT_avg_xy_36_l2402_240225

-- Given condition: The average of the numbers 2, 6, 10, x, and y is 18
def avg_condition (x y : ℝ) : Prop :=
  (2 + 6 + 10 + x + y) / 5 = 18

-- Goal: To prove that the average of x and y is 36
theorem avg_xy_36 (x y : ℝ) (h : avg_condition x y) : (x + y) / 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_avg_xy_36_l2402_240225


namespace NUMINAMATH_GPT_lawn_unmowed_fraction_l2402_240267

noncomputable def rate_mary : ℚ := 1 / 6
noncomputable def rate_tom : ℚ := 1 / 3

theorem lawn_unmowed_fraction :
  (1 : ℚ) - ((1 * rate_tom) + (2 * (rate_mary + rate_tom))) = 1 / 6 :=
by
  -- This part will be the actual proof which we are skipping
  sorry

end NUMINAMATH_GPT_lawn_unmowed_fraction_l2402_240267


namespace NUMINAMATH_GPT_solve_inequality_system_l2402_240229

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l2402_240229


namespace NUMINAMATH_GPT_average_breadth_of_plot_l2402_240294

theorem average_breadth_of_plot :
  ∃ B L : ℝ, (L - B = 10) ∧ (23 * B = (1/2) * (L + B) * B) ∧ (B = 18) :=
by
  sorry

end NUMINAMATH_GPT_average_breadth_of_plot_l2402_240294


namespace NUMINAMATH_GPT_problem_solution_l2402_240201

theorem problem_solution (m : ℝ) (h : (m - 2023)^2 + (2024 - m)^2 = 2025) :
  (m - 2023) * (2024 - m) = -1012 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2402_240201


namespace NUMINAMATH_GPT_Jina_mascots_total_l2402_240205

theorem Jina_mascots_total :
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  teddies + bunnies + koala + additional_teddies = 51 :=
by
  let teddies := 5
  let bunnies := 3 * teddies
  let koala := 1
  let additional_teddies := 2 * bunnies
  show teddies + bunnies + koala + additional_teddies = 51
  sorry

end NUMINAMATH_GPT_Jina_mascots_total_l2402_240205


namespace NUMINAMATH_GPT_investment_time_l2402_240292

theorem investment_time
  (p_investment_ratio : ℚ) (q_investment_ratio : ℚ)
  (profit_ratio_p : ℚ) (profit_ratio_q : ℚ)
  (q_investment_time : ℕ)
  (h1 : p_investment_ratio / q_investment_ratio = 7 / 5)
  (h2 : profit_ratio_p / profit_ratio_q = 7 / 10)
  (h3 : q_investment_time = 40) :
  ∃ t : ℚ, t = 28 :=
by
  sorry

end NUMINAMATH_GPT_investment_time_l2402_240292


namespace NUMINAMATH_GPT_PersonX_job_completed_time_l2402_240287

-- Definitions for conditions
def Dan_job_time := 15 -- hours
def PersonX_job_time (x : ℝ) := x -- hours
def Dan_work_time := 3 -- hours
def PersonX_remaining_work_time := 8 -- hours

-- Given Dan's and Person X's work time, prove Person X's job completion time
theorem PersonX_job_completed_time (x : ℝ) (h1 : Dan_job_time > 0)
    (h2 : PersonX_job_time x > 0)
    (h3 : Dan_work_time > 0)
    (h4 : PersonX_remaining_work_time * (1 - Dan_work_time / Dan_job_time) = 1 / x * 8) :
    x = 10 :=
  sorry

end NUMINAMATH_GPT_PersonX_job_completed_time_l2402_240287


namespace NUMINAMATH_GPT_mom_prepared_pieces_l2402_240224

-- Define the conditions
def jane_pieces : ℕ := 4
def total_eaters : ℕ := 3

-- Define the hypothesis that each of the eaters ate an equal number of pieces
def each_ate_equal (pieces : ℕ) : Prop := pieces = jane_pieces

-- The number of pieces Jane's mom prepared
theorem mom_prepared_pieces : total_eaters * jane_pieces = 12 :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_mom_prepared_pieces_l2402_240224


namespace NUMINAMATH_GPT_count_3_digit_integers_with_product_36_l2402_240278

theorem count_3_digit_integers_with_product_36 : 
  ∃ n, n = 21 ∧ 
         (∀ d1 d2 d3 : ℕ, 
           1 ≤ d1 ∧ d1 ≤ 9 ∧ 
           1 ≤ d2 ∧ d2 ≤ 9 ∧ 
           1 ≤ d3 ∧ d3 ≤ 9 ∧
           d1 * d2 * d3 = 36 → 
           (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0)) := sorry

end NUMINAMATH_GPT_count_3_digit_integers_with_product_36_l2402_240278


namespace NUMINAMATH_GPT_edward_money_left_l2402_240234

noncomputable def toy_cost : ℝ := 0.95

noncomputable def toy_quantity : ℕ := 4

noncomputable def toy_discount : ℝ := 0.15

noncomputable def race_track_cost : ℝ := 6.00

noncomputable def race_track_tax : ℝ := 0.08

noncomputable def initial_amount : ℝ := 17.80

noncomputable def total_toy_cost_before_discount : ℝ := toy_quantity * toy_cost

noncomputable def discount_amount : ℝ := toy_discount * total_toy_cost_before_discount

noncomputable def total_toy_cost_after_discount : ℝ := total_toy_cost_before_discount - discount_amount

noncomputable def race_track_tax_amount : ℝ := race_track_tax * race_track_cost

noncomputable def total_race_track_cost_after_tax : ℝ := race_track_cost + race_track_tax_amount

noncomputable def total_amount_spent : ℝ := total_toy_cost_after_discount + total_race_track_cost_after_tax

noncomputable def money_left : ℝ := initial_amount - total_amount_spent

theorem edward_money_left : money_left = 8.09 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_edward_money_left_l2402_240234


namespace NUMINAMATH_GPT_sin_neg_pi_over_three_l2402_240226

theorem sin_neg_pi_over_three : Real.sin (-Real.pi / 3) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_pi_over_three_l2402_240226


namespace NUMINAMATH_GPT_enrico_earnings_l2402_240235

def roosterPrice (weight: ℕ) : ℝ :=
  if weight < 20 then weight * 0.80
  else if weight ≤ 35 then weight * 0.65
  else weight * 0.50

theorem enrico_earnings :
  roosterPrice 15 + roosterPrice 30 + roosterPrice 40 + roosterPrice 50 = 76.50 := 
by
  sorry

end NUMINAMATH_GPT_enrico_earnings_l2402_240235


namespace NUMINAMATH_GPT_floor_sqrt_50_l2402_240297

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_50_l2402_240297


namespace NUMINAMATH_GPT_angle_ABC_is_83_l2402_240266

-- Definitions of angles and the quadrilateral
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angleBAC angleCAD angleACD : ℝ)
variables (AB AD AC : ℝ)

-- Conditions as hypotheses
axiom h1 : angleBAC = 60
axiom h2 : angleCAD = 60
axiom h3 : AB + AD = AC
axiom h4 : angleACD = 23

-- The theorem to prove
theorem angle_ABC_is_83 (h1 : angleBAC = 60) (h2 : angleCAD = 60) (h3 : AB + AD = AC) (h4 : angleACD = 23) : 
  ∃ angleABC : ℝ, angleABC = 83 :=
sorry

end NUMINAMATH_GPT_angle_ABC_is_83_l2402_240266


namespace NUMINAMATH_GPT_zebras_total_games_l2402_240232

theorem zebras_total_games 
  (x y : ℝ)
  (h1 : x = 0.40 * y)
  (h2 : (x + 8) / (y + 11) = 0.55) 
  : y + 11 = 24 :=
sorry

end NUMINAMATH_GPT_zebras_total_games_l2402_240232


namespace NUMINAMATH_GPT_compare_expr_l2402_240255

theorem compare_expr (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) :=
sorry

end NUMINAMATH_GPT_compare_expr_l2402_240255


namespace NUMINAMATH_GPT_quadratic_conversion_l2402_240251

theorem quadratic_conversion (x : ℝ) :
  (2*x - 1)^2 = (x + 1)*(3*x + 4) →
  ∃ a b c : ℝ, a = 1 ∧ b = -11 ∧ c = -3 ∧ a*x^2 + b*x + c = 0 :=
by simp [pow_two, mul_add, add_mul, mul_comm]; sorry

end NUMINAMATH_GPT_quadratic_conversion_l2402_240251


namespace NUMINAMATH_GPT_annual_population_change_l2402_240298

theorem annual_population_change (initial_population : Int) (moved_in : Int) (moved_out : Int) (final_population : Int) (years : Int) : 
  initial_population = 780 → 
  moved_in = 100 →
  moved_out = 400 →
  final_population = 60 →
  years = 4 →
  (initial_population + moved_in - moved_out - final_population) / years = 105 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_annual_population_change_l2402_240298


namespace NUMINAMATH_GPT_claire_earning_l2402_240272

noncomputable def flowers := 400
noncomputable def tulips := 120
noncomputable def total_roses := flowers - tulips
noncomputable def white_roses := 80
noncomputable def red_roses := total_roses - white_roses
noncomputable def red_rose_value : ℝ := 0.75
noncomputable def roses_to_sell := red_roses / 2

theorem claire_earning : (red_rose_value * roses_to_sell) = 75 := 
by 
  sorry

end NUMINAMATH_GPT_claire_earning_l2402_240272


namespace NUMINAMATH_GPT_least_possible_value_f_1998_l2402_240286

theorem least_possible_value_f_1998 
  (f : ℕ → ℕ)
  (h : ∀ m n, f (n^2 * f m) = m * (f n)^2) : 
  f 1998 = 120 :=
sorry

end NUMINAMATH_GPT_least_possible_value_f_1998_l2402_240286


namespace NUMINAMATH_GPT_first_worker_time_l2402_240233

def productivity (x y z : ℝ) : Prop :=
  x + y + z = 20 ∧
  (20 / x) > 3 ∧
  (20 / x) + (60 / (y + z)) = 8

theorem first_worker_time (x y z : ℝ) (h : productivity x y z) : 
  (80 / x) = 16 :=
  sorry

end NUMINAMATH_GPT_first_worker_time_l2402_240233


namespace NUMINAMATH_GPT_twelve_sided_figure_area_is_13_cm2_l2402_240209

def twelve_sided_figure_area_cm2 : ℝ :=
  let unit_square := 1
  let full_squares := 9
  let triangle_pairs := 4
  full_squares * unit_square + triangle_pairs * unit_square

theorem twelve_sided_figure_area_is_13_cm2 :
  twelve_sided_figure_area_cm2 = 13 := 
by
  sorry

end NUMINAMATH_GPT_twelve_sided_figure_area_is_13_cm2_l2402_240209


namespace NUMINAMATH_GPT_pipe_fill_time_l2402_240269

theorem pipe_fill_time (T : ℝ) 
  (h1 : ∃ T : ℝ, 0 < T) 
  (h2 : T + (1/2) > 0) 
  (h3 : ∃ leak_rate : ℝ, leak_rate = 1/10) 
  (h4 : ∃ pipe_rate : ℝ, pipe_rate = 1/T) 
  (h5 : ∃ effective_rate : ℝ, effective_rate = pipe_rate - leak_rate) 
  (h6 : effective_rate = 1 / (T + 1/2))  : 
  T = Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_pipe_fill_time_l2402_240269


namespace NUMINAMATH_GPT_find_multiple_l2402_240293

theorem find_multiple (x y : ℕ) (h1 : x = 11) (h2 : x + y = 55) (h3 : ∃ k m : ℕ, y = k * x + m) :
  ∃ k : ℕ, y = k * x ∧ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l2402_240293


namespace NUMINAMATH_GPT_trapezoid_ratio_l2402_240214

-- Define the isosceles trapezoid properties and the point inside it
noncomputable def isosceles_trapezoid (r s : ℝ) (hr : r > s) (triangle_areas : List ℝ) : Prop :=
  triangle_areas = [2, 3, 4, 5]

-- Define the problem statement
theorem trapezoid_ratio (r s : ℝ) (hr : r > s) (areas : List ℝ) (hareas : isosceles_trapezoid r s hr areas) :
  r / s = 2 + Real.sqrt 2 := sorry

end NUMINAMATH_GPT_trapezoid_ratio_l2402_240214


namespace NUMINAMATH_GPT_worker_efficiency_l2402_240277

theorem worker_efficiency (Wq : ℝ) (x : ℝ) : 
  (1.4 * (1 / x) = 1 / (1.4 * x)) → 
  (14 * (1 / x + 1 / (1.4 * x)) = 1) → 
  x = 24 :=
by
  sorry

end NUMINAMATH_GPT_worker_efficiency_l2402_240277


namespace NUMINAMATH_GPT_real_solutions_x4_plus_3_minus_x4_eq_82_l2402_240259

theorem real_solutions_x4_plus_3_minus_x4_eq_82 :
  ∀ x : ℝ, x = 2.6726 ∨ x = 0.3274 → x^4 + (3 - x)^4 = 82 := by
  sorry

end NUMINAMATH_GPT_real_solutions_x4_plus_3_minus_x4_eq_82_l2402_240259


namespace NUMINAMATH_GPT_paint_needed_270_statues_l2402_240270

theorem paint_needed_270_statues:
  let height_large := 12
  let paint_large := 2
  let height_small := 3
  let num_statues := 270
  let ratio_height := (height_small : ℝ) / (height_large : ℝ)
  let ratio_area := ratio_height ^ 2
  let paint_small := paint_large * ratio_area
  let total_paint := num_statues * paint_small
  total_paint = 33.75 := by
  sorry

end NUMINAMATH_GPT_paint_needed_270_statues_l2402_240270


namespace NUMINAMATH_GPT_circle_condition_l2402_240200

theorem circle_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + 5 * k = 0) ↔ k < 1 := 
sorry

end NUMINAMATH_GPT_circle_condition_l2402_240200


namespace NUMINAMATH_GPT_tangent_line_through_P_l2402_240268

theorem tangent_line_through_P (x y : ℝ) :
  (∃ l : ℝ, l = 3*x - 4*y + 5) ∨ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_P_l2402_240268


namespace NUMINAMATH_GPT_P_at_2007_l2402_240299

noncomputable def P (x : ℝ) : ℝ :=
x^15 - 2008 * x^14 + 2008 * x^13 - 2008 * x^12 + 2008 * x^11
- 2008 * x^10 + 2008 * x^9 - 2008 * x^8 + 2008 * x^7
- 2008 * x^6 + 2008 * x^5 - 2008 * x^4 + 2008 * x^3
- 2008 * x^2 + 2008 * x

-- Statement to show that P(2007) = 2007
theorem P_at_2007 : P 2007 = 2007 :=
  sorry

end NUMINAMATH_GPT_P_at_2007_l2402_240299


namespace NUMINAMATH_GPT_range_of_a_l2402_240218

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 
  a * x + 1 - 4 * a 
else 
  x ^ 2 - 3 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → 
  a ∈ (Set.Ioi (2/3) ∪ Set.Iic 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2402_240218


namespace NUMINAMATH_GPT_ratio_problem_l2402_240227

/-
  Given the ratio A : B : C = 3 : 2 : 5, we need to prove that 
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19.
-/

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 :=
by sorry

end NUMINAMATH_GPT_ratio_problem_l2402_240227


namespace NUMINAMATH_GPT_cards_received_at_home_l2402_240220

-- Definitions based on the conditions
def initial_cards := 403
def total_cards := 690

-- The theorem to prove the number of cards received at home
theorem cards_received_at_home : total_cards - initial_cards = 287 :=
by
  -- Proof goes here, but we use sorry as a placeholder.
  sorry

end NUMINAMATH_GPT_cards_received_at_home_l2402_240220


namespace NUMINAMATH_GPT_cube_strictly_increasing_l2402_240288

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_strictly_increasing_l2402_240288


namespace NUMINAMATH_GPT_avg_marks_l2402_240239

theorem avg_marks (P C M B E H G : ℝ) 
  (h1 : C = P + 75)
  (h2 : M = P + 105)
  (h3 : B = P - 15)
  (h4 : E = P - 25)
  (h5 : H = P - 25)
  (h6 : G = P - 25)
  (h7 : P + C + M + B + E + H + G = P + 520) :
  (M + B + H + G) / 4 = 82 :=
by 
  sorry

end NUMINAMATH_GPT_avg_marks_l2402_240239


namespace NUMINAMATH_GPT_domain_of_function_l2402_240260

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x * (x - 1) ≥ 0) ↔ (x = 0 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_GPT_domain_of_function_l2402_240260


namespace NUMINAMATH_GPT_scenery_photos_correct_l2402_240291

-- Define the problem conditions
def animal_photos := 10
def flower_photos := 3 * animal_photos
def photos_total := 45
def scenery_photos := flower_photos - 10

-- State the theorem
theorem scenery_photos_correct : scenery_photos = 20 ∧ animal_photos + flower_photos + scenery_photos = photos_total := by
  sorry

end NUMINAMATH_GPT_scenery_photos_correct_l2402_240291


namespace NUMINAMATH_GPT_total_drink_volume_l2402_240289

theorem total_drink_volume (coke_parts sprite_parts mtndew_parts : ℕ) (coke_volume : ℕ) :
  coke_parts = 2 → sprite_parts = 1 → mtndew_parts = 3 → coke_volume = 6 →
  (coke_volume / coke_parts) * (coke_parts + sprite_parts + mtndew_parts) = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_drink_volume_l2402_240289


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l2402_240253

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) : (x + 1) * (x - 2) > 0 → x > 2 :=
by sorry

theorem converse_not_true 
  (x : ℝ) : x > 2 → (x + 1) * (x - 2) > 0 :=
by sorry

theorem cond_x_gt_2_iff_sufficient_not_necessary 
  (x : ℝ) : (x > 2 → (x + 1) * (x - 2) > 0) ∧ 
            ((x + 1) * (x - 2) > 0 → x > 2) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l2402_240253


namespace NUMINAMATH_GPT_quadratic_has_exactly_one_solution_l2402_240271

theorem quadratic_has_exactly_one_solution (k : ℚ) :
  (3 * x^2 - 8 * x + k = 0) → ((-8)^2 - 4 * 3 * k = 0) → k = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_exactly_one_solution_l2402_240271


namespace NUMINAMATH_GPT_fraction_of_students_getting_A_l2402_240236

theorem fraction_of_students_getting_A
    (frac_B : ℚ := 1/2)
    (frac_C : ℚ := 1/8)
    (frac_D : ℚ := 1/12)
    (frac_F : ℚ := 1/24)
    (passing_grade_frac: ℚ := 0.875) :
    (1 - (frac_B + frac_C + frac_D + frac_F) = 1/8) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_getting_A_l2402_240236


namespace NUMINAMATH_GPT_square_area_increase_l2402_240237

theorem square_area_increase (s : ℕ) (h : (s = 5) ∨ (s = 10) ∨ (s = 15)) :
  (1.35^2 - 1) * 100 = 82.25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_increase_l2402_240237


namespace NUMINAMATH_GPT_mike_travel_distance_l2402_240282

theorem mike_travel_distance
  (mike_start : ℝ := 2.50)
  (mike_per_mile : ℝ := 0.25)
  (annie_start : ℝ := 2.50)
  (annie_toll : ℝ := 5.00)
  (annie_per_mile : ℝ := 0.25)
  (annie_miles : ℝ := 14)
  (mike_cost : ℝ)
  (annie_cost : ℝ) :
  mike_cost = annie_cost → mike_cost = mike_start + mike_per_mile * 34 := by
  sorry

end NUMINAMATH_GPT_mike_travel_distance_l2402_240282


namespace NUMINAMATH_GPT_conversion_problems_l2402_240254

def decimal_to_binary (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 2 + 10 * decimal_to_binary (n / 2)

def largest_two_digit_octal : ℕ := 77

theorem conversion_problems :
  decimal_to_binary 111 = 1101111 ∧ (7 * 8 + 7) = 63 :=
by
  sorry

end NUMINAMATH_GPT_conversion_problems_l2402_240254


namespace NUMINAMATH_GPT_y_intercept_of_line_is_minus_one_l2402_240215

theorem y_intercept_of_line_is_minus_one : 
  (∀ x y : ℝ, y = 2 * x - 1 → y = -1) :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_is_minus_one_l2402_240215


namespace NUMINAMATH_GPT_oliver_shirts_problem_l2402_240208

-- Defining the quantities of short sleeve shirts, long sleeve shirts, and washed shirts.
def shortSleeveShirts := 39
def longSleeveShirts  := 47
def shirtsWashed := 20

-- Stating the problem formally.
theorem oliver_shirts_problem :
  shortSleeveShirts + longSleeveShirts - shirtsWashed = 66 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_oliver_shirts_problem_l2402_240208


namespace NUMINAMATH_GPT_pencils_per_person_l2402_240263

theorem pencils_per_person (x : ℕ) (h : 3 * x = 24) : x = 8 :=
by
  -- sorry we are skipping the actual proof
  sorry

end NUMINAMATH_GPT_pencils_per_person_l2402_240263


namespace NUMINAMATH_GPT_problem1_expr_eval_l2402_240217

theorem problem1_expr_eval : 
  (1:ℤ) - (1:ℤ)^(2022:ℕ) - (3 * (2/3:ℚ)^2 - (8/3:ℚ) / ((-2)^3:ℤ)) = -8/3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_expr_eval_l2402_240217


namespace NUMINAMATH_GPT_determine_a_l2402_240202

lemma even_exponent (a : ℤ) : (a^2 - 4*a) % 2 = 0 :=
sorry

lemma decreasing_function (a : ℤ) : a^2 - 4*a < 0 :=
sorry

theorem determine_a (a : ℤ) (h1 : (a^2 - 4*a) % 2 = 0) (h2 : a^2 - 4*a < 0) : a = 2 :=
sorry

end NUMINAMATH_GPT_determine_a_l2402_240202


namespace NUMINAMATH_GPT_cyclist_speed_ratio_l2402_240295

theorem cyclist_speed_ratio (v_1 v_2 : ℝ)
  (h1 : v_1 = 2 * v_2)
  (h2 : v_1 + v_2 = 6)
  (h3 : v_1 - v_2 = 2) :
  v_1 / v_2 = 2 := 
sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_l2402_240295


namespace NUMINAMATH_GPT_r_fourth_power_sum_l2402_240230

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end NUMINAMATH_GPT_r_fourth_power_sum_l2402_240230


namespace NUMINAMATH_GPT_p_scale_measurement_l2402_240249

theorem p_scale_measurement (a b P S : ℝ) (h1 : 30 = 6 * a + b) (h2 : 60 = 24 * a + b) (h3 : 100 = a * P + b) : P = 48 :=
by
  sorry

end NUMINAMATH_GPT_p_scale_measurement_l2402_240249


namespace NUMINAMATH_GPT_sum_of_four_numbers_eq_zero_l2402_240281

theorem sum_of_four_numbers_eq_zero
  (x y s t : ℝ)
  (h₀ : x ≠ y)
  (h₁ : x ≠ s)
  (h₂ : x ≠ t)
  (h₃ : y ≠ s)
  (h₄ : y ≠ t)
  (h₅ : s ≠ t)
  (h_eq : (x + s) / (x + t) = (y + t) / (y + s)) :
  x + y + s + t = 0 := by
sorry

end NUMINAMATH_GPT_sum_of_four_numbers_eq_zero_l2402_240281


namespace NUMINAMATH_GPT_sandy_net_amount_spent_l2402_240252

def amount_spent_shorts : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def amount_received_return : ℝ := 7.43

theorem sandy_net_amount_spent :
  amount_spent_shorts + amount_spent_shirt - amount_received_return = 18.70 :=
by
  sorry

end NUMINAMATH_GPT_sandy_net_amount_spent_l2402_240252


namespace NUMINAMATH_GPT_frood_game_least_n_l2402_240246

theorem frood_game_least_n (n : ℕ) (h : n > 0) (drop_score : ℕ := n * (n + 1) / 2) (eat_score : ℕ := 15 * n) 
  : drop_score > eat_score ↔ n ≥ 30 :=
by
  sorry

end NUMINAMATH_GPT_frood_game_least_n_l2402_240246


namespace NUMINAMATH_GPT_find_larger_number_l2402_240257

theorem find_larger_number
  (x y : ℝ)
  (h1 : y = 2 * x + 3)
  (h2 : x + y = 27)
  : y = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l2402_240257


namespace NUMINAMATH_GPT_maximize_binom_term_l2402_240203

theorem maximize_binom_term :
  ∃ k, k ∈ Finset.range (207) ∧
  (∀ m ∈ Finset.range (207), (Nat.choose 206 k * (Real.sqrt 5)^k) ≥ (Nat.choose 206 m * (Real.sqrt 5)^m)) ∧ k = 143 :=
sorry

end NUMINAMATH_GPT_maximize_binom_term_l2402_240203


namespace NUMINAMATH_GPT_smallest_integer_in_set_of_seven_l2402_240284

theorem smallest_integer_in_set_of_seven (n : ℤ) (h : n + 6 < 3 * (n + 3)) : n = -1 :=
sorry

end NUMINAMATH_GPT_smallest_integer_in_set_of_seven_l2402_240284


namespace NUMINAMATH_GPT_yulia_max_candies_l2402_240248

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end NUMINAMATH_GPT_yulia_max_candies_l2402_240248


namespace NUMINAMATH_GPT_goldfish_remaining_to_catch_l2402_240240

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_goldfish_remaining_to_catch_l2402_240240


namespace NUMINAMATH_GPT_find_x_l2402_240206

theorem find_x (x : ℚ) (h : 2 / 5 = (4 / 3) / x) : x = 10 / 3 :=
by
sorry

end NUMINAMATH_GPT_find_x_l2402_240206


namespace NUMINAMATH_GPT_geometric_sequence_general_term_and_sum_l2402_240207

theorem geometric_sequence_general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₁ : ∀ n, a n = 2 ^ n)
  (h₂ : ∀ n, b n = 2 * n - 1)
  : (∀ n, T n = 6 + (2 * n - 3) * 2 ^ (n + 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_general_term_and_sum_l2402_240207


namespace NUMINAMATH_GPT_calculate_final_price_l2402_240238

noncomputable def final_price (j_init p_init : ℝ) (j_inc p_inc : ℝ) (tax discount : ℝ) (j_quantity p_quantity : ℕ) : ℝ :=
  let j_new := j_init + j_inc
  let p_new := p_init * (1 + p_inc)
  let total_price := (j_new * j_quantity) + (p_new * p_quantity)
  let tax_amount := total_price * tax
  let price_with_tax := total_price + tax_amount
  let final_price := if j_quantity > 1 ∧ p_quantity >= 3 then price_with_tax * (1 - discount) else price_with_tax
  final_price

theorem calculate_final_price :
  final_price 30 100 10 (0.20) (0.07) (0.10) 2 5 = 654.84 :=
by
  sorry

end NUMINAMATH_GPT_calculate_final_price_l2402_240238


namespace NUMINAMATH_GPT_correct_statement_a_l2402_240247

theorem correct_statement_a (x y : ℝ) (h : x + y < 0) : x^2 - y > x :=
sorry

end NUMINAMATH_GPT_correct_statement_a_l2402_240247


namespace NUMINAMATH_GPT_two_x_plus_y_eq_12_l2402_240274

-- Variables representing the prime numbers x and y
variables {x y : ℕ}

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Prime n
def lcm_eq (a b c : ℕ) : Prop := Nat.lcm a b = c

-- The theorem statement
theorem two_x_plus_y_eq_12 (h1 : lcm_eq x y 10) (h2 : is_prime x) (h3 : is_prime y) (h4 : x > y) :
    2 * x + y = 12 :=
sorry

end NUMINAMATH_GPT_two_x_plus_y_eq_12_l2402_240274


namespace NUMINAMATH_GPT_factorize_expression_l2402_240241

theorem factorize_expression (a b x y : ℝ) : 
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = ab * (x - y)^2 * (a * x - a * y - b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2402_240241


namespace NUMINAMATH_GPT_correct_sum_104th_parenthesis_l2402_240273

noncomputable def sum_104th_parenthesis : ℕ := sorry

theorem correct_sum_104th_parenthesis :
  sum_104th_parenthesis = 2072 := 
by 
  sorry

end NUMINAMATH_GPT_correct_sum_104th_parenthesis_l2402_240273
