import Mathlib

namespace joel_garden_size_l1992_199223

-- Definitions based on the conditions
variable (G : ℕ) -- G is the size of Joel's garden.

-- Condition 1: Half of the garden is for fruits.
def half_garden_fruits (G : ℕ) := G / 2

-- Condition 2: Half of the garden is for vegetables.
def half_garden_vegetables (G : ℕ) := G / 2

-- Condition 3: A quarter of the fruit section is used for strawberries.
def quarter_fruit_section (G : ℕ) := (half_garden_fruits G) / 4

-- Condition 4: The quarter for strawberries takes up 8 square feet.
axiom strawberry_section : quarter_fruit_section G = 8

-- Hypothesis: The size of Joel's garden is 64 square feet.
theorem joel_garden_size : G = 64 :=
by
  -- Insert the logical progression of the proof here.
  sorry

end joel_garden_size_l1992_199223


namespace man_arrived_earlier_l1992_199245

-- Definitions of conditions as Lean variables
variables
  (usual_arrival_time_home : ℕ)  -- The usual arrival time at home
  (usual_drive_time : ℕ) -- The usual drive time for the wife to reach the station
  (early_arrival_difference : ℕ := 16) -- They arrived home 16 minutes earlier
  (man_walk_time : ℕ := 52) -- The man walked for 52 minutes

-- The proof statement
theorem man_arrived_earlier
  (usual_arrival_time_home : ℕ)
  (usual_drive_time : ℕ)
  (H : usual_arrival_time_home - man_walk_time <= usual_drive_time - early_arrival_difference)
  : man_walk_time = 52 :=
sorry

end man_arrived_earlier_l1992_199245


namespace valid_third_side_l1992_199273

theorem valid_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < c) (h₄ : c < 11) : c = 8 := 
by 
  sorry

end valid_third_side_l1992_199273


namespace cubic_expression_value_l1992_199264

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l1992_199264


namespace volume_ratio_of_sphere_surface_area_l1992_199209

theorem volume_ratio_of_sphere_surface_area 
  {V1 V2 V3 : ℝ} 
  (h : V1/V3 = 1/27 ∧ V2/V3 = 8/27) 
  : V1 + V2 = (1/3) * V3 := 
sorry

end volume_ratio_of_sphere_surface_area_l1992_199209


namespace greatest_possible_perimeter_l1992_199253

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_possible_perimeter :
  ∃ x : ℕ, 6 ≤ x ∧ x < 17 ∧ is_triangle x (2 * x) 17 ∧ (x + 2 * x + 17 = 65) := by
  sorry

end greatest_possible_perimeter_l1992_199253


namespace values_of_t_l1992_199283

theorem values_of_t (x y z t : ℝ) 
  (h1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (h2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (h3 : x^2 - x * y + y^2 = t) : 
  t ≤ 10 :=
sorry

end values_of_t_l1992_199283


namespace capacity_of_new_bathtub_is_400_liters_l1992_199201

-- Definitions based on conditions
def possible_capacities : Set ℕ := {4, 40, 400, 4000}  -- The possible capacities

-- Proof statement
theorem capacity_of_new_bathtub_is_400_liters (c : ℕ) 
  (h : c ∈ possible_capacities) : 
  c = 400 := 
sorry

end capacity_of_new_bathtub_is_400_liters_l1992_199201


namespace complete_set_contains_all_rationals_l1992_199290

theorem complete_set_contains_all_rationals (T : Set ℚ) (hT : ∀ (p q : ℚ), p / q ∈ T → p / (p + q) ∈ T ∧ q / (p + q) ∈ T) (r : ℚ) : 
  (r = 1 ∨ r = 1 / 2) → (∀ x : ℚ, 0 < x ∧ x < 1 → x ∈ T) :=
by
  sorry

end complete_set_contains_all_rationals_l1992_199290


namespace magnitude_of_error_l1992_199212

theorem magnitude_of_error (x : ℝ) (hx : 0 < x) :
  abs ((4 * x) - (x / 4)) / (4 * x) * 100 = 94 := 
sorry

end magnitude_of_error_l1992_199212


namespace max_integer_values_correct_l1992_199272

noncomputable def max_integer_values (a b c : ℝ) : ℕ :=
  if a > 100 then 2 else 0

theorem max_integer_values_correct (a b c : ℝ) (h : a > 100) :
  max_integer_values a b c = 2 :=
by sorry

end max_integer_values_correct_l1992_199272


namespace number_of_sets_A_l1992_199281

/-- Given conditions about intersections and unions of set A, we want to find the number of 
  possible sets A that satisfy the given conditions. Specifically, prove the following:
  - A ∩ {-1, 0, 1} = {0, 1}
  - A ∪ {-2, 0, 2} = {-2, 0, 1, 2}
  Total number of such sets A is 4.
-/
theorem number_of_sets_A : ∃ (As : Finset (Finset ℤ)), 
  (∀ A ∈ As, A ∩ {-1, 0, 1} = {0, 1} ∧ A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) ∧
  As.card = 4 := 
sorry

end number_of_sets_A_l1992_199281


namespace white_triangle_pairs_condition_l1992_199230

def number_of_white_pairs (total_triangles : Nat) 
                          (red_pairs : Nat) 
                          (blue_pairs : Nat)
                          (mixed_pairs : Nat) : Nat :=
  let red_involved := red_pairs * 2
  let blue_involved := blue_pairs * 2
  let remaining_red := total_triangles / 2 * 5 - red_involved - mixed_pairs
  let remaining_blue := total_triangles / 2 * 4 - blue_involved - mixed_pairs
  (total_triangles / 2 * 7) - (remaining_red + remaining_blue)/2

theorem white_triangle_pairs_condition : number_of_white_pairs 32 3 2 1 = 6 := by
  sorry

end white_triangle_pairs_condition_l1992_199230


namespace complete_the_square_l1992_199234

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 11 ∧ a = -4) ↔ (x ^ 2 - 8 * x + 5 = 0) :=
by
  sorry

end complete_the_square_l1992_199234


namespace least_clock_equivalent_hour_l1992_199241

theorem least_clock_equivalent_hour (h : ℕ) (h_gt_9 : h > 9) (clock_equiv : (h^2 - h) % 12 = 0) : h = 13 :=
sorry

end least_clock_equivalent_hour_l1992_199241


namespace system1_solution_system2_solution_l1992_199296

theorem system1_solution (x y : ℝ) (h1 : 3 * x + y = 4) (h2 : 3 * x + 2 * y = 6) : x = 2 / 3 ∧ y = 2 :=
by
  sorry

theorem system2_solution (x y : ℝ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 5 * y = 11) : x = 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l1992_199296


namespace sequence_limit_l1992_199298

noncomputable def sequence_converges (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n > 1 ∧ a (n + 1) ^ 2 ≥ a n * a (n + 2)

theorem sequence_limit (a : ℕ → ℝ) (h : sequence_converges a) : 
  ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (Real.log (a (n + 1)) / Real.log (a n) - l) < ε := 
sorry

end sequence_limit_l1992_199298


namespace max_integer_value_of_f_l1992_199267

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 21) / (4 * x^2 + 8 * x + 5)

theorem max_integer_value_of_f :
  ∃ n : ℤ, n = 17 ∧ ∀ x : ℝ, f x ≤ (n : ℝ) :=
by
  sorry

end max_integer_value_of_f_l1992_199267


namespace molecular_weight_of_3_moles_of_Fe2_SO4_3_l1992_199293

noncomputable def mol_weight_fe : ℝ := 55.845
noncomputable def mol_weight_s : ℝ := 32.065
noncomputable def mol_weight_o : ℝ := 15.999

noncomputable def mol_weight_fe2_so4_3 : ℝ :=
  (2 * mol_weight_fe) + (3 * (mol_weight_s + (4 * mol_weight_o)))

theorem molecular_weight_of_3_moles_of_Fe2_SO4_3 :
  3 * mol_weight_fe2_so4_3 = 1199.619 := by
  sorry

end molecular_weight_of_3_moles_of_Fe2_SO4_3_l1992_199293


namespace arithmetic_sequence_value_l1992_199202

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℤ), 
  a 1 = 1 → 
  a 3 = -5 → 
  (a 1 - a 2 - a 3 - a 4 = 16) :=
by
  intros a h1 h3
  sorry

end arithmetic_sequence_value_l1992_199202


namespace probability_ge_sqrt2_l1992_199226

noncomputable def probability_length_chord_ge_sqrt2
  (a : ℝ)
  (h : a ≠ 0)
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  : ℝ :=
  if -1 ≤ a ∧ a ≤ 1 then (1 / Real.sqrt (1^2 + 1^2)) else 0

theorem probability_ge_sqrt2 
  (a : ℝ) 
  (h : a ≠ 0) 
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  (length_cond : (Real.sqrt (4 - 2*a^2) ≥ Real.sqrt 2)) : 
  probability_length_chord_ge_sqrt2 a h intersect_cond = (Real.sqrt 2 / 2) :=
by
  sorry

end probability_ge_sqrt2_l1992_199226


namespace tangent_line_parabola_l1992_199270

theorem tangent_line_parabola (a : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 3 * y + a = 0) → a = 18 :=
by
  sorry

end tangent_line_parabola_l1992_199270


namespace undecided_voters_percentage_l1992_199225

theorem undecided_voters_percentage
  (biff_percent : ℝ)
  (total_people : ℤ)
  (marty_votes : ℤ)
  (undecided_percent : ℝ) :
  biff_percent = 0.45 →
  total_people = 200 →
  marty_votes = 94 →
  undecided_percent = ((total_people - (marty_votes + (biff_percent * total_people))) / total_people) * 100 →
  undecided_percent = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end undecided_voters_percentage_l1992_199225


namespace max_value_of_a_plus_b_l1992_199277

theorem max_value_of_a_plus_b (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : a ≤ 3) (h₃ : b ≥ 3) :
  a + b ≤ 7 :=
sorry

end max_value_of_a_plus_b_l1992_199277


namespace a_when_a_minus_1_no_reciprocal_l1992_199208

theorem a_when_a_minus_1_no_reciprocal (a : ℝ) (h : ¬ ∃ b : ℝ, (a - 1) * b = 1) : a = 1 := 
by
  sorry

end a_when_a_minus_1_no_reciprocal_l1992_199208


namespace rubies_in_chest_l1992_199288

theorem rubies_in_chest (R : ℕ) (h₁ : 421 = R + 44) : R = 377 :=
by 
  sorry

end rubies_in_chest_l1992_199288


namespace smallest_value_l1992_199252

theorem smallest_value (y : ℝ) (hy : 0 < y ∧ y < 1) :
  y^3 < y^2 ∧ y^3 < 3*y ∧ y^3 < (y)^(1/3:ℝ) ∧ y^3 < (1/y) :=
sorry

end smallest_value_l1992_199252


namespace scientific_notation_of_2270000_l1992_199266

theorem scientific_notation_of_2270000 : 
  (2270000 : ℝ) = 2.27 * 10^6 :=
sorry

end scientific_notation_of_2270000_l1992_199266


namespace total_value_of_item_l1992_199224

theorem total_value_of_item
  (import_tax : ℝ)
  (V : ℝ)
  (h₀ : import_tax = 110.60)
  (h₁ : import_tax = 0.07 * (V - 1000)) :
  V = 2579.43 := 
sorry

end total_value_of_item_l1992_199224


namespace min_value_proof_l1992_199231

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  9 / a + 16 / b + 25 / (c ^ 2)

theorem min_value_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 5) :
  minimum_value a b c ≥ 50 :=
sorry

end min_value_proof_l1992_199231


namespace tangent_line_eq_l1992_199200

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x

noncomputable def f' (x : ℝ) : ℝ := (x : ℝ) * Real.exp x

theorem tangent_line_eq (x : ℝ) (h : x = 0) : 
  ∃ (c : ℝ), (1 : ℝ) = 1 ∧ f x = c ∧ f' x = 0 ∧ (∀ y, y = c) :=
by
  sorry

end tangent_line_eq_l1992_199200


namespace least_positive_three_digit_multiple_of_8_l1992_199215

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l1992_199215


namespace order_of_logs_l1992_199276

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 16 / Real.log 8

theorem order_of_logs : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l1992_199276


namespace certain_number_value_l1992_199249

theorem certain_number_value
  (x : ℝ)
  (y : ℝ)
  (h1 : (28 + x + 42 + 78 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  y = 104 :=
by
  -- Proof goes here
  sorry

end certain_number_value_l1992_199249


namespace quadratic_func_inequality_l1992_199282

theorem quadratic_func_inequality (c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 4 * x + c)
  (h_increasing : ∀ x y, x ≤ y → -2 ≤ x → f x ≤ f y) :
  f 1 > f 0 ∧ f 0 > f (-2) :=
by
  sorry

end quadratic_func_inequality_l1992_199282


namespace find_sum_l1992_199262

variable {x y z w : ℤ}

-- Conditions: Consecutive integers and their sum condition
def consecutive_integers (x y z : ℤ) : Prop := y = x + 1 ∧ z = x + 2
def sum_is_150 (x y z : ℤ) : Prop := x + y + z = 150
def w_definition (w z x : ℤ) : Prop := w = 2 * z - x

-- Theorem statement
theorem find_sum (h1 : consecutive_integers x y z) (h2 : sum_is_150 x y z) (h3 : w_definition w z x) :
  x + y + z + w = 203 :=
sorry

end find_sum_l1992_199262


namespace line_tangent_parabola_unique_d_l1992_199205

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l1992_199205


namespace alpha_div_3_range_l1992_199278

theorem alpha_div_3_range (α : ℝ) (k : ℤ) 
  (h1 : Real.sin α > 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi / 3) ∨ 
            (2 * k * Real.pi + 5 * Real.pi / 6 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi) :=
sorry

end alpha_div_3_range_l1992_199278


namespace cylinder_volume_l1992_199237

theorem cylinder_volume (r : ℝ) (h : ℝ) (A : ℝ) (V : ℝ) 
  (sphere_surface_area : A = 256 * Real.pi)
  (cylinder_height : h = 2 * r) 
  (sphere_surface_formula : A = 4 * Real.pi * r^2) 
  (cylinder_volume_formula : V = Real.pi * r^2 * h) : V = 1024 * Real.pi := 
by
  -- Definitions provided as conditions
  sorry

end cylinder_volume_l1992_199237


namespace exists_l_l1992_199216

theorem exists_l (n : ℕ) (h : n ≥ 4011^2) : ∃ l : ℤ, n < l^2 ∧ l^2 < (1 + 1/2005) * n := 
sorry

end exists_l_l1992_199216


namespace wage_constraint_l1992_199206

/-- Wage constraints for hiring carpenters and tilers given a budget -/
theorem wage_constraint (x y : ℕ) (h_carpenter_wage : 50 * x + 40 * y = 2000) : 5 * x + 4 * y = 200 := by
  sorry

end wage_constraint_l1992_199206


namespace nuts_per_student_l1992_199263

theorem nuts_per_student (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) (total_nuts : ℕ) (nuts_per_student : ℕ)
    (h1 : bags = 65)
    (h2 : nuts_per_bag = 15)
    (h3 : students = 13)
    (h4 : total_nuts = bags * nuts_per_bag)
    (h5 : nuts_per_student = total_nuts / students)
    : nuts_per_student = 75 :=
by
  sorry

end nuts_per_student_l1992_199263


namespace valentine_cards_l1992_199213

theorem valentine_cards (x y : ℕ) (h : x * y = x + y + 18) : x * y = 40 :=
by
  sorry

end valentine_cards_l1992_199213


namespace seconds_in_9_point_4_minutes_l1992_199227

def seconds_in_minute : ℕ := 60
def minutes : ℝ := 9.4
def expected_seconds : ℝ := 564

theorem seconds_in_9_point_4_minutes : minutes * seconds_in_minute = expected_seconds :=
by 
  sorry

end seconds_in_9_point_4_minutes_l1992_199227


namespace Yuna_drank_most_l1992_199244

noncomputable def Jimin_juice : ℝ := 0.7
noncomputable def Eunji_juice : ℝ := Jimin_juice - 1/10
noncomputable def Yoongi_juice : ℝ := 4/5
noncomputable def Yuna_juice : ℝ := Jimin_juice + 0.2

theorem Yuna_drank_most :
  Yuna_juice = max (max Jimin_juice Eunji_juice) (max Yoongi_juice Yuna_juice) :=
by
  sorry

end Yuna_drank_most_l1992_199244


namespace calculate_full_recipes_needed_l1992_199217

def initial_attendance : ℕ := 125
def attendance_drop_percentage : ℝ := 0.40
def cookies_per_student : ℕ := 2
def cookies_per_recipe : ℕ := 18

theorem calculate_full_recipes_needed :
  let final_attendance := initial_attendance * (1 - attendance_drop_percentage : ℝ)
  let total_cookies_needed := (final_attendance * (cookies_per_student : ℕ))
  let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℕ)
  ⌈recipes_needed⌉ = 9 :=
  by
  sorry

end calculate_full_recipes_needed_l1992_199217


namespace ratio_M_N_l1992_199274

variable (M Q P N : ℝ)

-- Conditions
axiom h1 : M = 0.40 * Q
axiom h2 : Q = 0.25 * P
axiom h3 : N = 0.60 * P

theorem ratio_M_N : M / N = 1 / 6 :=
by
  sorry

end ratio_M_N_l1992_199274


namespace min_p_plus_q_l1992_199297

-- Define the conditions
variables {p q : ℕ}

-- Problem statement in Lean 4
theorem min_p_plus_q (h₁ : p > 0) (h₂ : q > 0) (h₃ : 108 * p = q^3) : p + q = 8 :=
sorry

end min_p_plus_q_l1992_199297


namespace relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l1992_199229

variable (x y : ℝ)
variable (h1 : 2 * (x + y) = 18)
variable (h2 : x * y = 18)
variable (h3 : x > 0) (h4 : y > 0) (h5 : x > y)
variable (h6 : x * y = 21)

theorem relationship_and_range : (y = 9 - x ∧ 0 < x ∧ x < 9) :=
by sorry

theorem dimensions_when_area_18 :
  (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) :=
by sorry

theorem impossibility_of_area_21 :
  ¬(∃ x y, x * y = 21 ∧ 2 * (x + y) = 18 ∧ x > y) :=
by sorry

end relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l1992_199229


namespace length_ab_is_constant_l1992_199265

noncomputable def length_AB_constant (p : ℝ) (hp : p > 0) : Prop :=
  let parabola := { P : ℝ × ℝ | P.1 ^ 2 = 2 * p * P.2 }
  let line := { P : ℝ × ℝ | P.2 = P.1 + p / 2 }
  (∃ A B : ℝ × ℝ, A ∈ parabola ∧ B ∈ parabola ∧ A ∈ line ∧ B ∈ line ∧ 
    dist A B = 4 * p)

theorem length_ab_is_constant (p : ℝ) (hp : p > 0) : length_AB_constant p hp :=
by {
  sorry
}

end length_ab_is_constant_l1992_199265


namespace number_of_balls_sold_l1992_199255

-- Definitions from conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 120
def loss : ℕ := 5 * cost_price_per_ball

-- Mathematically equivalent proof statement
theorem number_of_balls_sold (n : ℕ) (h : n * cost_price_per_ball - selling_price = loss) : n = 11 :=
  sorry

end number_of_balls_sold_l1992_199255


namespace value_of_c_l1992_199232

theorem value_of_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end value_of_c_l1992_199232


namespace relationship_y1_y2_y3_l1992_199239

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l1992_199239


namespace binary_to_decimal_eq_l1992_199291

theorem binary_to_decimal_eq :
  (1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 205 :=
by
  sorry

end binary_to_decimal_eq_l1992_199291


namespace xy_value_l1992_199258

noncomputable def x (y : ℝ) : ℝ := 36 * y

theorem xy_value (y : ℝ) (h1 : y = 0.16666666666666666) : x y * y = 1 :=
by
  rw [h1, x]
  sorry

end xy_value_l1992_199258


namespace solve_for_n_l1992_199207

theorem solve_for_n (n : ℝ) : 
  (0.05 * n + 0.06 * (30 + n)^2 = 45) ↔ 
  (n = -2.5833333333333335 ∨ n = -58.25) :=
sorry

end solve_for_n_l1992_199207


namespace anthony_total_pencils_l1992_199275

theorem anthony_total_pencils :
  let original_pencils := 9
  let given_pencils := 56
  original_pencils + given_pencils = 65 := by
  sorry

end anthony_total_pencils_l1992_199275


namespace lychee_production_increase_l1992_199240

variable (x : ℕ) -- percentage increase as a natural number

def lychee_increase_2006 (x : ℕ) : ℕ :=
  (1 + x)*(1 + x)

theorem lychee_production_increase (x : ℕ) :
  lychee_increase_2006 x = (1 + x) * (1 + x) :=
by
  sorry

end lychee_production_increase_l1992_199240


namespace common_chord_length_common_chord_diameter_eq_circle_l1992_199251

/-
Given two circles C1: x^2 + y^2 - 2x + 10y - 24 = 0 and C2: x^2 + y^2 + 2x + 2y - 8 = 0,
prove that 
1. The length of the common chord is 2 * sqrt(5).
2. The equation of the circle that has the common chord as its diameter is (x + 8/5)^2 + (y - 6/5)^2 = 36/5.
-/

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 10 * y - 24 = 0

-- Define the second circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Prove the length of the common chord
theorem common_chord_length : ∃ d : ℝ, d = 2 * Real.sqrt 5 :=
sorry

-- Prove the equation of the circle that has the common chord as its diameter
theorem common_chord_diameter_eq_circle : ∃ (x y : ℝ → ℝ), (x + 8/5)^2 + (y - 6/5)^2 = 36/5 :=
sorry

end common_chord_length_common_chord_diameter_eq_circle_l1992_199251


namespace not_p_is_sufficient_but_not_necessary_for_not_q_l1992_199214

variable (x : ℝ)

def proposition_p : Prop := |x| < 2
def proposition_q : Prop := x^2 - x - 2 < 0

theorem not_p_is_sufficient_but_not_necessary_for_not_q :
  (¬ proposition_p x) → (¬ proposition_q x) ∧ (¬ proposition_q x) → (¬ proposition_p x) → False := by
  sorry

end not_p_is_sufficient_but_not_necessary_for_not_q_l1992_199214


namespace isosceles_triangle_perimeter_l1992_199246

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : a = c ∨ b = c) :
  a + b + c = 22 :=
by
  -- This part of the proof is simplified using the conditions
  sorry

end isosceles_triangle_perimeter_l1992_199246


namespace cookies_in_jar_l1992_199243

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l1992_199243


namespace leak_empties_cistern_in_24_hours_l1992_199287

theorem leak_empties_cistern_in_24_hours (F L : ℝ) (h1: F = 1 / 8) (h2: F - L = 1 / 12) :
  1 / L = 24 := 
by {
  sorry
}

end leak_empties_cistern_in_24_hours_l1992_199287


namespace simultaneous_equations_in_quadrant_I_l1992_199256

theorem simultaneous_equations_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 4 / 3) :=
  sorry

end simultaneous_equations_in_quadrant_I_l1992_199256


namespace cards_per_pack_l1992_199236

-- Definitions from the problem conditions
def packs := 60
def cards_per_page := 10
def pages_needed := 42

-- Theorem statement for the mathematically equivalent proof problem
theorem cards_per_pack : (pages_needed * cards_per_page) / packs = 7 :=
by sorry

end cards_per_pack_l1992_199236


namespace rate_per_sqm_l1992_199219

theorem rate_per_sqm (length width : ℝ) (cost : ℝ) (Area : ℝ := length * width) (rate : ℝ := cost / Area) 
  (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 8250) : 
  rate = 400 :=
sorry

end rate_per_sqm_l1992_199219


namespace revenue_fell_by_percentage_l1992_199235

theorem revenue_fell_by_percentage :
  let old_revenue : ℝ := 69.0
  let new_revenue : ℝ := 52.0
  let percentage_decrease : ℝ := ((old_revenue - new_revenue) / old_revenue) * 100
  abs (percentage_decrease - 24.64) < 1e-2 :=
by
  sorry

end revenue_fell_by_percentage_l1992_199235


namespace inversely_proportional_solve_y_l1992_199203

theorem inversely_proportional_solve_y (k : ℝ) (x y : ℝ)
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = -10) :
  y = -67.5 :=
by
  sorry

end inversely_proportional_solve_y_l1992_199203


namespace find_triangle_sides_l1992_199257

variable (a b c : ℕ)
variable (P : ℕ)
variable (R : ℚ := 65 / 8)
variable (r : ℕ := 4)

theorem find_triangle_sides (h1 : R = 65 / 8) (h2 : r = 4) (h3 : P = a + b + c) : 
  a = 13 ∧ b = 14 ∧ c = 15 :=
  sorry

end find_triangle_sides_l1992_199257


namespace smallest_consecutive_integers_product_l1992_199210

theorem smallest_consecutive_integers_product (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 5040) : 
  n = 7 :=
sorry

end smallest_consecutive_integers_product_l1992_199210


namespace vasya_max_consecutive_liked_numbers_l1992_199248

def is_liked_by_vasya (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0 → n % d = 0

theorem vasya_max_consecutive_liked_numbers : 
  ∃ (seq : ℕ → ℕ), 
    (∀ n, seq n = n ∧ is_liked_by_vasya (seq n)) ∧
    (∀ m, seq m + 1 < seq (m + 1)) ∧ seq 12 - seq 0 + 1 = 13 :=
sorry

end vasya_max_consecutive_liked_numbers_l1992_199248


namespace distance_between_vertices_of_hyperbola_l1992_199292

def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), c₁ = 4 ∧ c₂ = -4 ∧
    (c₁ * x^2 + 24 * x + c₂ * y^2 + 8 * y + 44 = 0)

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_equation x y) → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end distance_between_vertices_of_hyperbola_l1992_199292


namespace gcd_90_250_l1992_199268

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end gcd_90_250_l1992_199268


namespace find_minimum_a_l1992_199220

theorem find_minimum_a (a x : ℤ) : 
  (x - a < 0) → 
  (x > -3 / 2) → 
  (∃ n : ℤ, ∀ y : ℤ, y ∈ {k | -1 ≤ k ∧ k ≤ n} ∧ y < a) → 
  a = 3 := sorry

end find_minimum_a_l1992_199220


namespace ten_percent_eq_l1992_199204

variable (s t : ℝ)

def ten_percent_of (x : ℝ) : ℝ := 0.1 * x

theorem ten_percent_eq (h : ten_percent_of s = t) : s = 10 * t :=
by sorry

end ten_percent_eq_l1992_199204


namespace abs_inequality_l1992_199211

variables (a b c : ℝ)

theorem abs_inequality (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l1992_199211


namespace sufficient_but_not_necessary_l1992_199280

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem sufficient_but_not_necessary (m n : ℝ) :
  vectors_parallel (m, 1) (n, 1) ↔ (m = n) := sorry

end sufficient_but_not_necessary_l1992_199280


namespace constants_inequality_value_l1992_199238

theorem constants_inequality_value
  (a b c d : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∀ x, (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26) ∨ x < -4 ↔ (x - a) * (x - b) * (x - c) / (x - d) ≤ 0) :
  a + 3 * b + 3 * c + 4 * d = 72 :=
sorry

end constants_inequality_value_l1992_199238


namespace general_term_formula_l1992_199294

theorem general_term_formula (a S : ℕ → ℝ) (h : ∀ n, S n = (2 / 3) * a n + (1 / 3)) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = -2 * a (n - 1)) →
  ∀ n, a n = (-2)^(n - 1) :=
by
  sorry

end general_term_formula_l1992_199294


namespace sum_of_solutions_l1992_199254

theorem sum_of_solutions (x y : ℝ) (h₁ : y = 8) (h₂ : x^2 + y^2 = 144) : 
  ∃ x1 x2 : ℝ, (x1 = 4 * Real.sqrt 5 ∧ x2 = -4 * Real.sqrt 5) ∧ (x1 + x2 = 0) :=
by
  sorry

end sum_of_solutions_l1992_199254


namespace janet_miles_per_day_l1992_199284

def total_miles : ℕ := 72
def days : ℕ := 9
def miles_per_day : ℕ := 8

theorem janet_miles_per_day : total_miles / days = miles_per_day :=
by {
  sorry
}

end janet_miles_per_day_l1992_199284


namespace intersect_single_point_l1992_199260

theorem intersect_single_point (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 4 * x + 2 = 0) ∧ ∀ x₁ x₂ : ℝ, 
  (m - 3) * x₁^2 - 4 * x₁ + 2 = 0 → (m - 3) * x₂^2 - 4 * x₂ + 2 = 0 → x₁ = x₂ ↔ m = 3 ∨ m = 5 := 
sorry

end intersect_single_point_l1992_199260


namespace Fran_speed_l1992_199250

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end Fran_speed_l1992_199250


namespace at_least_one_gt_one_l1992_199222

theorem at_least_one_gt_one (x y : ℝ) (h : x + y > 2) : ¬(x > 1 ∨ y > 1) → (x ≤ 1 ∧ y ≤ 1) := 
by
  sorry

end at_least_one_gt_one_l1992_199222


namespace root_of_equation_l1992_199218

theorem root_of_equation (a : ℝ) (h : a^2 * (-1)^2 + 2011 * a * (-1) - 2012 = 0) : 
  a = 2012 ∨ a = -1 :=
by sorry

end root_of_equation_l1992_199218


namespace range_of_a_l1992_199285

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_period : ∀ x, f (x + 3) = f x)
  (h1 : f 1 > 1) 
  (h2018 : f 2018 = (a : ℝ) ^ 2 - 5) : 
  -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l1992_199285


namespace contest_sum_l1992_199259

theorem contest_sum 
(A B C D E : ℕ) 
(h_sum : A + B + C + D + E = 35)
(h_right_E : B + C + D + E = 13)
(h_right_D : C + D + E = 31)
(h_right_A : B + C + D + E = 21)
(h_right_C : C + D + E = 7)
: D + B = 11 :=
sorry

end contest_sum_l1992_199259


namespace price_per_exercise_book_is_correct_l1992_199233

-- Define variables and conditions from the problem statement
variables (xM xH booksM booksH pricePerBook : ℝ)
variables (xH_gives_xM : ℝ)

-- Conditions set up from the problem statement
axiom pooled_money : xM = xH
axiom books_ming : booksM = 8
axiom books_hong : booksH = 12
axiom amount_given : xH_gives_xM = 1.1

-- Problem statement to prove
theorem price_per_exercise_book_is_correct :
  (8 + 12) * pricePerBook / 2 = 1.1 → pricePerBook = 0.55 := by
  sorry

end price_per_exercise_book_is_correct_l1992_199233


namespace value_of_7x_minus_3y_l1992_199289

theorem value_of_7x_minus_3y (x y : ℚ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := 
sorry

end value_of_7x_minus_3y_l1992_199289


namespace find_a_b_l1992_199221

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_b 
  (h_max : ∀ x, f a b x ≤ 3)
  (h_min : ∀ x, f a b x ≥ 2)
  : (a = 0.5 ∨ a = -0.5) ∧ b = 2.5 :=
by
  sorry

end find_a_b_l1992_199221


namespace ellipse_properties_l1992_199295

theorem ellipse_properties :
  (∃ a e : ℝ, (∃ b c : ℝ, a^2 = 25 ∧ b^2 = 9 ∧ c^2 = a^2 - b^2 ∧ c = 4 ∧ e = c / a) ∧ a = 5 ∧ e = 4 / 5) :=
sorry

end ellipse_properties_l1992_199295


namespace cost_of_first_10_kgs_of_apples_l1992_199271

theorem cost_of_first_10_kgs_of_apples 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 663) 
  (h2 : 30 * l + 6 * q = 726) : 
  10 * l = 200 :=
by
  -- Proof would follow here
  sorry

end cost_of_first_10_kgs_of_apples_l1992_199271


namespace coins_ratio_l1992_199299

-- Conditions
def initial_coins : Nat := 125
def gift_coins : Nat := 35
def sold_coins : Nat := 80

-- Total coins after receiving the gift
def total_coins := initial_coins + gift_coins

-- Statement to prove the ratio simplifies to 1:2
theorem coins_ratio : (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end coins_ratio_l1992_199299


namespace calculation_l1992_199279

theorem calculation : (3 * 4 * 5) * ((1 / 3 : ℚ) + (1 / 4 : ℚ) - (1 / 5 : ℚ)) = 23 := by
  sorry

end calculation_l1992_199279


namespace find_P_coordinates_l1992_199247

-- Given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- The area of triangle PAB is 5
def areaPAB (P : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))

-- Point P lies on the x-axis
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem find_P_coordinates (P : ℝ × ℝ) :
  on_x_axis P → areaPAB P = 5 → (P = (-4, 0) ∨ P = (6, 0)) :=
by
  sorry

end find_P_coordinates_l1992_199247


namespace factorize_a_squared_plus_2a_l1992_199261

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2 * a = a * (a + 2) :=
  sorry

end factorize_a_squared_plus_2a_l1992_199261


namespace cheapest_lamp_cost_l1992_199286

/--
Frank wants to buy a new lamp for his bedroom. The cost of the cheapest lamp is some amount, and the most expensive in the store is 3 times more expensive. Frank has $90, and if he buys the most expensive lamp available, he would have $30 remaining. Prove that the cost of the cheapest lamp is $20.
-/
theorem cheapest_lamp_cost (c most_expensive : ℝ) (h_cheapest_lamp : most_expensive = 3 * c) 
(h_frank_money : 90 - most_expensive = 30) : c = 20 := 
sorry

end cheapest_lamp_cost_l1992_199286


namespace range_of_m_l1992_199269

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (m x : ℝ) : ℝ := m * x + 1
noncomputable def h (x : ℝ) : ℝ := (1 / x) - (2 * Real.log x / x)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 2)) ∧ (g m x = 2 - 2 * f x)) ↔
  (-2 * Real.exp (-3/2) ≤ m ∧ m ≤ 3 * Real.exp 1) :=
sorry

end range_of_m_l1992_199269


namespace triangle_inequality_proof_l1992_199242

theorem triangle_inequality_proof (a b c : ℝ) (h : a + b > c) : a^3 + b^3 + 3 * a * b * c > c^3 :=
by sorry

end triangle_inequality_proof_l1992_199242


namespace smallest_circle_area_l1992_199228

/-- The smallest possible area of a circle passing through two given points in the coordinate plane. -/
theorem smallest_circle_area (P Q : ℝ × ℝ) (hP : P = (-3, -2)) (hQ : Q = (2, 4)) : 
  ∃ (A : ℝ), A = (61 * Real.pi) / 4 :=
by
  sorry

end smallest_circle_area_l1992_199228
