import Mathlib

namespace NUMINAMATH_GPT_sine_sum_zero_l1262_126263

open Real 

theorem sine_sum_zero (α β γ : ℝ) :
  (sin α / (sin (α - β) * sin (α - γ))
  + sin β / (sin (β - α) * sin (β - γ))
  + sin γ / (sin (γ - α) * sin (γ - β)) = 0) :=
sorry

end NUMINAMATH_GPT_sine_sum_zero_l1262_126263


namespace NUMINAMATH_GPT_factor_2210_two_digit_l1262_126209

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end NUMINAMATH_GPT_factor_2210_two_digit_l1262_126209


namespace NUMINAMATH_GPT_Tony_change_l1262_126241

structure Conditions where
  bucket_capacity : ℕ := 2
  sandbox_depth : ℕ := 2
  sandbox_width : ℕ := 4
  sandbox_length : ℕ := 5
  sand_weight_per_cubic_foot : ℕ := 3
  water_per_4_trips : ℕ := 3
  water_bottle_cost : ℕ := 2
  water_bottle_volume: ℕ := 15
  initial_money : ℕ := 10

theorem Tony_change (conds : Conditions) : 
  let volume_sandbox := conds.sandbox_depth * conds.sandbox_width * conds.sandbox_length
  let total_weight_sand := volume_sandbox * conds.sand_weight_per_cubic_foot
  let trips := total_weight_sand / conds.bucket_capacity
  let drinks := trips / 4
  let total_water := drinks * conds.water_per_4_trips
  let bottles_needed := total_water / conds.water_bottle_volume
  let total_cost := bottles_needed * conds.water_bottle_cost
  let change := conds.initial_money - total_cost
  change = 4 := by
  /- calculations corresponding to steps that are translated from the solution to show that the 
     change indeed is $4 -/
  sorry

end NUMINAMATH_GPT_Tony_change_l1262_126241


namespace NUMINAMATH_GPT_fourth_watercraft_is_submarine_l1262_126260

-- Define the conditions as Lean definitions
def same_direction_speed (w1 w2 w3 w4 : Type) : Prop :=
  -- All watercraft are moving in the same direction at the same speed
  true

def separation (w1 w2 w3 w4 : Type) (d : ℝ) : Prop :=
  -- Each pair of watercraft is separated by distance d
  true

def cargo_ship (w : Type) : Prop := true
def fishing_boat (w : Type) : Prop := true
def passenger_vessel (w : Type) : Prop := true

-- Define that the fourth watercraft is unique
def unique_watercraft (w : Type) : Prop := true

-- Proof statement that the fourth watercraft is a submarine
theorem fourth_watercraft_is_submarine 
  (w1 w2 w3 w4 : Type)
  (h1 : same_direction_speed w1 w2 w3 w4)
  (h2 : separation w1 w2 w3 w4 100)
  (h3 : cargo_ship w1)
  (h4 : fishing_boat w2)
  (h5 : passenger_vessel w3) :
  unique_watercraft w4 := 
sorry

end NUMINAMATH_GPT_fourth_watercraft_is_submarine_l1262_126260


namespace NUMINAMATH_GPT_salary_increase_l1262_126250

theorem salary_increase (x : ℕ) (hB_C_sum : 2*x + 3*x = 6000) : 
  ((3 * x - 1 * x) / (1 * x) ) * 100 = 200 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_salary_increase_l1262_126250


namespace NUMINAMATH_GPT_degree_f_x2_mul_g_x4_l1262_126210

open Polynomial

theorem degree_f_x2_mul_g_x4 {f g : Polynomial ℝ} (hf : degree f = 4) (hg : degree g = 5) :
  degree (f.comp (X ^ 2) * g.comp (X ^ 4)) = 28 :=
sorry

end NUMINAMATH_GPT_degree_f_x2_mul_g_x4_l1262_126210


namespace NUMINAMATH_GPT_jake_pure_alcohol_l1262_126261

-- Definitions based on the conditions
def shots : ℕ := 8
def ounces_per_shot : ℝ := 1.5
def vodka_purity : ℝ := 0.5
def friends : ℕ := 2

-- Statement to prove the amount of pure alcohol Jake drank
theorem jake_pure_alcohol : (shots * ounces_per_shot * vodka_purity) / friends = 3 := by
  sorry

end NUMINAMATH_GPT_jake_pure_alcohol_l1262_126261


namespace NUMINAMATH_GPT_bug_total_distance_l1262_126243

/-!
# Problem Statement
A bug starts crawling on a number line from position -3. It first moves to -7, then turns around and stops briefly at 0 before continuing on to 8. Prove that the total distance the bug crawls is 19 units.
-/

def bug_initial_position : ℤ := -3
def bug_position_1 : ℤ := -7
def bug_position_2 : ℤ := 0
def bug_final_position : ℤ := 8

theorem bug_total_distance : 
  |bug_position_1 - bug_initial_position| + 
  |bug_position_2 - bug_position_1| + 
  |bug_final_position - bug_position_2| = 19 :=
by 
  sorry

end NUMINAMATH_GPT_bug_total_distance_l1262_126243


namespace NUMINAMATH_GPT_max_value_trig_expr_exists_angle_for_max_value_l1262_126272

theorem max_value_trig_expr : ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5 :=
sorry

theorem exists_angle_for_max_value : ∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5 :=
sorry

end NUMINAMATH_GPT_max_value_trig_expr_exists_angle_for_max_value_l1262_126272


namespace NUMINAMATH_GPT_largest_two_numbers_l1262_126277

def a : Real := 2^(1/2)
def b : Real := 3^(1/3)
def c : Real := 8^(1/8)
def d : Real := 9^(1/9)

theorem largest_two_numbers : 
  (max (max (max a b) c) d = b) ∧ 
  (max (max a c) d = a) := 
sorry

end NUMINAMATH_GPT_largest_two_numbers_l1262_126277


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l1262_126296

theorem solve_system1 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : x = 2 * y + 1) : x = 3 ∧ y = 1 := 
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x - y = 6) (h2 : 3 * x + 2 * y = 2) : x = 2 ∧ y = -2 := 
by sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l1262_126296


namespace NUMINAMATH_GPT_red_cars_count_l1262_126287

variable (R B : ℕ)
variable (h1 : R * 8 = 3 * B)
variable (h2 : B = 90)

theorem red_cars_count : R = 33 :=
by
  -- here we would provide the proof
  sorry

end NUMINAMATH_GPT_red_cars_count_l1262_126287


namespace NUMINAMATH_GPT_inequality_of_function_l1262_126275

theorem inequality_of_function (x : ℝ) : 
  (1 / 2 : ℝ) ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_inequality_of_function_l1262_126275


namespace NUMINAMATH_GPT_g_does_not_pass_second_quadrant_l1262_126208

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+5) + 4

def M : ℝ × ℝ := (-5, 5)

noncomputable def g (x : ℝ) : ℝ := -5 + (5 : ℝ)^(x)

theorem g_does_not_pass_second_quadrant (a : ℝ) (x : ℝ) 
  (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hM : f a (-5) = 5) : 
  ∀ x < 0, g x < 0 :=
by
  sorry

end NUMINAMATH_GPT_g_does_not_pass_second_quadrant_l1262_126208


namespace NUMINAMATH_GPT_perimeter_non_shaded_region_l1262_126229

def shaded_area : ℤ := 78
def large_rect_area : ℤ := 80
def small_rect_area : ℤ := 8
def total_area : ℤ := large_rect_area + small_rect_area
def non_shaded_area : ℤ := total_area - shaded_area
def non_shaded_width : ℤ := 2
def non_shaded_length : ℤ := non_shaded_area / non_shaded_width
def non_shaded_perimeter : ℤ := 2 * (non_shaded_length + non_shaded_width)

theorem perimeter_non_shaded_region : non_shaded_perimeter = 14 := 
by
  exact rfl

end NUMINAMATH_GPT_perimeter_non_shaded_region_l1262_126229


namespace NUMINAMATH_GPT_sequence_either_increases_or_decreases_l1262_126289

theorem sequence_either_increases_or_decreases {x : ℕ → ℝ} (x1_pos : 0 < x 1) (x1_ne_one : x 1 ≠ 1) 
    (recurrence : ∀ n : ℕ, x (n + 1) = x n * (x n ^ 2 + 3) / (3 * x n ^ 2 + 1)) :
    (∀ n : ℕ, x n < x (n + 1)) ∨ (∀ n : ℕ, x n > x (n + 1)) :=
sorry

end NUMINAMATH_GPT_sequence_either_increases_or_decreases_l1262_126289


namespace NUMINAMATH_GPT_gas_station_total_boxes_l1262_126281

theorem gas_station_total_boxes
  (chocolate_boxes : ℕ)
  (sugar_boxes : ℕ)
  (gum_boxes : ℕ)
  (licorice_boxes : ℕ)
  (sour_boxes : ℕ)
  (h_chocolate : chocolate_boxes = 3)
  (h_sugar : sugar_boxes = 5)
  (h_gum : gum_boxes = 2)
  (h_licorice : licorice_boxes = 4)
  (h_sour : sour_boxes = 7) :
  chocolate_boxes + sugar_boxes + gum_boxes + licorice_boxes + sour_boxes = 21 := by
  sorry

end NUMINAMATH_GPT_gas_station_total_boxes_l1262_126281


namespace NUMINAMATH_GPT_equilateral_triangle_area_perimeter_l1262_126200

theorem equilateral_triangle_area_perimeter (altitude : ℝ) : 
  altitude = Real.sqrt 12 →
  (exists area perimeter : ℝ, area = 4 * Real.sqrt 3 ∧ perimeter = 12) :=
by
  intro h_alt
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_perimeter_l1262_126200


namespace NUMINAMATH_GPT_share_of_B_in_profit_l1262_126214

variable {D : ℝ} (hD_pos : 0 < D)

def investment (D : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let C := 2.5 * D
  let B := 1.25 * D
  let A := 5 * B
  (A, B, C, D)

def totalInvestment (A B C D : ℝ) : ℝ :=
  A + B + C + D

theorem share_of_B_in_profit (D : ℝ) (profit : ℝ) (hD : 0 < D)
  (h_profit : profit = 8000) :
  let ⟨A, B, C, D⟩ := investment D
  B / totalInvestment A B C D * profit = 1025.64 :=
by
  sorry

end NUMINAMATH_GPT_share_of_B_in_profit_l1262_126214


namespace NUMINAMATH_GPT_paint_cost_per_liter_l1262_126298

def cost_brush : ℕ := 20
def cost_canvas : ℕ := 3 * cost_brush
def min_liters : ℕ := 5
def total_earning : ℕ := 200
def total_profit : ℕ := 80
def total_cost : ℕ := total_earning - total_profit

theorem paint_cost_per_liter :
  (total_cost = cost_brush + cost_canvas + (5 * 8)) :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_per_liter_l1262_126298


namespace NUMINAMATH_GPT_total_items_l1262_126278

theorem total_items (slices_of_bread bottles_of_milk cookies : ℕ) (h1 : slices_of_bread = 58)
  (h2 : bottles_of_milk = slices_of_bread - 18) (h3 : cookies = slices_of_bread + 27) :
  slices_of_bread + bottles_of_milk + cookies = 183 :=
by
  sorry

end NUMINAMATH_GPT_total_items_l1262_126278


namespace NUMINAMATH_GPT_steel_ingot_weight_l1262_126234

theorem steel_ingot_weight 
  (initial_weight : ℕ)
  (percent_increase : ℚ)
  (ingot_cost : ℚ)
  (discount_threshold : ℕ)
  (discount_percent : ℚ)
  (total_cost : ℚ)
  (added_weight : ℚ)
  (number_of_ingots : ℕ)
  (ingot_weight : ℚ)
  (h1 : initial_weight = 60)
  (h2 : percent_increase = 0.6)
  (h3 : ingot_cost = 5)
  (h4 : discount_threshold = 10)
  (h5 : discount_percent = 0.2)
  (h6 : total_cost = 72)
  (h7 : added_weight = initial_weight * percent_increase)
  (h8 : added_weight = ingot_weight * number_of_ingots)
  (h9 : total_cost = (ingot_cost * number_of_ingots) * (1 - discount_percent)) :
  ingot_weight = 2 := 
by
  sorry

end NUMINAMATH_GPT_steel_ingot_weight_l1262_126234


namespace NUMINAMATH_GPT_boys_from_clay_l1262_126205

theorem boys_from_clay (total_students jonas_students clay_students pine_students total_boys total_girls : ℕ)
  (jonas_girls pine_boys : ℕ) 
  (H1 : total_students = 150)
  (H2 : jonas_students = 50)
  (H3 : clay_students = 60)
  (H4 : pine_students = 40)
  (H5 : total_boys = 80)
  (H6 : total_girls = 70)
  (H7 : jonas_girls = 30)
  (H8 : pine_boys = 15):
  ∃ (clay_boys : ℕ), clay_boys = 45 :=
by
  have jonas_boys : ℕ := jonas_students - jonas_girls
  have boys_from_clay := total_boys - pine_boys - jonas_boys
  exact ⟨boys_from_clay, by sorry⟩

end NUMINAMATH_GPT_boys_from_clay_l1262_126205


namespace NUMINAMATH_GPT_function_passes_through_point_l1262_126258

-- Lean 4 Statement
theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧ (a^(x-1) + 4) = y :=
by
  use 1
  use 5
  sorry

end NUMINAMATH_GPT_function_passes_through_point_l1262_126258


namespace NUMINAMATH_GPT_distance_l1_l2_l1262_126265

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_l1_l2 :
  distance_between_parallel_lines 3 4 (-3) 2 = 1 :=
by
  -- Add the conditions needed to assert the theorem
  let l1 := (3, 4, -3) -- definition of line l1
  let l2 := (3, 4, 2)  -- definition of line l2
  -- Calculate the distance using the given formula
  let d := distance_between_parallel_lines 3 4 (-3) 2
  -- Assert the result
  show d = 1
  sorry

end NUMINAMATH_GPT_distance_l1_l2_l1262_126265


namespace NUMINAMATH_GPT_second_number_is_11_l1262_126284

-- Define the conditions
variables (x : ℕ) (h1 : 5 * x = 55)

-- The theorem we want to prove
theorem second_number_is_11 : x = 11 :=
sorry

end NUMINAMATH_GPT_second_number_is_11_l1262_126284


namespace NUMINAMATH_GPT_find_k_l1262_126222

theorem find_k (k m : ℝ) : (m^2 - 8*m) ∣ (m^3 - k*m^2 - 24*m + 16) → k = 8 := by
  sorry

end NUMINAMATH_GPT_find_k_l1262_126222


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_7_mod_17_l1262_126212

theorem least_five_digit_congruent_to_7_mod_17 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 17 = 7 ∧ (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 7 → n ≤ m) :=
sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_7_mod_17_l1262_126212


namespace NUMINAMATH_GPT_range_of_a_l1262_126251

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * x - 3 * a * Real.exp x
def A : Set ℝ := { x | f x = 0 }
def B (a : ℝ) : Set ℝ := { x | g x a = 0 }

theorem range_of_a (a : ℝ) :
  (∃ x₁ ∈ A, ∃ x₂ ∈ B a, |x₁ - x₂| < 1) →
  a ∈ Set.Ici (1 / (3 * Real.exp 1)) ∩ Set.Iic (4 / (3 * Real.exp 4)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1262_126251


namespace NUMINAMATH_GPT_largest_product_is_168_l1262_126215

open Set

noncomputable def largest_product_from_set (s : Set ℤ) (n : ℕ) (result : ℤ) : Prop :=
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∀ (x y z : ℤ), x ∈ s → y ∈ s → z ∈ s → x ≠ y → y ≠ z → x ≠ z →
  x * y * z ≤ a * b * c ∧ a * b * c = result

theorem largest_product_is_168 :
  largest_product_from_set {-4, -3, 1, 3, 7, 8} 3 168 :=
sorry

end NUMINAMATH_GPT_largest_product_is_168_l1262_126215


namespace NUMINAMATH_GPT_selection_methods_l1262_126288

theorem selection_methods (females males : Nat) (h_females : females = 3) (h_males : males = 2):
  females + males = 5 := 
  by 
    -- We add sorry here to skip the proof
    sorry

end NUMINAMATH_GPT_selection_methods_l1262_126288


namespace NUMINAMATH_GPT_geometric_seq_a5_l1262_126256

theorem geometric_seq_a5 : ∃ (a₁ q : ℝ), 0 < q ∧ a₁ + 2 * a₁ * q = 4 ∧ (a₁ * q^3)^2 = 4 * (a₁ * q^2) * (a₁ * q^6) ∧ (a₅ = a₁ * q^4) := 
  by
    sorry

end NUMINAMATH_GPT_geometric_seq_a5_l1262_126256


namespace NUMINAMATH_GPT_imaginary_part_z_l1262_126283

theorem imaginary_part_z : 
  ∀ (z : ℂ), z = (5 - I) / (1 - I) → z.im = 2 := 
by
  sorry

end NUMINAMATH_GPT_imaginary_part_z_l1262_126283


namespace NUMINAMATH_GPT_dave_deleted_17_apps_l1262_126295

-- Define the initial and final state of Dave's apps
def initial_apps : Nat := 10
def added_apps : Nat := 11
def apps_left : Nat := 4

-- The total number of apps before deletion
def total_apps : Nat := initial_apps + added_apps

-- The expected number of deleted apps
def deleted_apps : Nat := total_apps - apps_left

-- The proof statement
theorem dave_deleted_17_apps : deleted_apps = 17 := by
  -- detailed steps are not required
  sorry

end NUMINAMATH_GPT_dave_deleted_17_apps_l1262_126295


namespace NUMINAMATH_GPT_team_selection_count_l1262_126216

-- The problem's known conditions
def boys : ℕ := 10
def girls : ℕ := 12
def team_size : ℕ := 8

-- The number of ways to select a team of 8 members with at least 2 boys and no more than 4 boys
noncomputable def count_ways : ℕ :=
  (Nat.choose boys 2) * (Nat.choose girls 6) +
  (Nat.choose boys 3) * (Nat.choose girls 5) +
  (Nat.choose boys 4) * (Nat.choose girls 4)

-- The main statement to prove
theorem team_selection_count : count_ways = 238570 := by
  sorry

end NUMINAMATH_GPT_team_selection_count_l1262_126216


namespace NUMINAMATH_GPT_polygon_with_20_diagonals_is_octagon_l1262_126201

theorem polygon_with_20_diagonals_is_octagon :
  ∃ (n : ℕ), n ≥ 3 ∧ (n * (n - 3)) / 2 = 20 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_20_diagonals_is_octagon_l1262_126201


namespace NUMINAMATH_GPT_longer_leg_smallest_triangle_l1262_126226

noncomputable def length_of_longer_leg_of_smallest_triangle (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then a 
  else if n = 1 then (a / 2) * Real.sqrt 3
  else if n = 2 then ((a / 2) * Real.sqrt 3 / 2) * Real.sqrt 3
  else ((a / 2) * Real.sqrt 3 / 2 * Real.sqrt 3 / 2) * Real.sqrt 3

theorem longer_leg_smallest_triangle : 
  length_of_longer_leg_of_smallest_triangle 3 10 = 45 / 8 := 
sorry

end NUMINAMATH_GPT_longer_leg_smallest_triangle_l1262_126226


namespace NUMINAMATH_GPT_meaningful_fraction_l1262_126245

theorem meaningful_fraction (x : ℝ) : (x + 5 ≠ 0) → (x ≠ -5) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l1262_126245


namespace NUMINAMATH_GPT_functional_equation_solution_l1262_126220

noncomputable def f (t : ℝ) (x : ℝ) := (t * (x - t)) / (t + 1)

noncomputable def g (t : ℝ) (x : ℝ) := t * (x - t)

theorem functional_equation_solution (t : ℝ) (ht : t ≠ -1) :
  ∀ x y : ℝ, f t (x + g t y) = x * f t y - y * f t x + g t x :=
by
  intros x y
  let fx := f t
  let gx := g t
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1262_126220


namespace NUMINAMATH_GPT_choose_president_and_secretary_l1262_126282

theorem choose_president_and_secretary (total_members boys girls : ℕ) (h_total : total_members = 30) (h_boys : boys = 18) (h_girls : girls = 12) : 
  (boys * girls = 216) :=
by
  sorry

end NUMINAMATH_GPT_choose_president_and_secretary_l1262_126282


namespace NUMINAMATH_GPT_unique_angles_sum_l1262_126266

theorem unique_angles_sum (a1 a2 a3 a4 e4 e5 e6 e7 : ℝ) 
  (h_abcd: a1 + a2 + a3 + a4 = 360) 
  (h_efgh: e4 + e5 + e6 + e7 = 360) 
  (h_shared: a4 = e4) : 
  a1 + a2 + a3 + e4 + e5 + e6 + e7 - a4 = 360 := 
by 
  sorry

end NUMINAMATH_GPT_unique_angles_sum_l1262_126266


namespace NUMINAMATH_GPT_sum_of_first_20_terms_arithmetic_sequence_l1262_126219

theorem sum_of_first_20_terms_arithmetic_sequence 
  (a : ℕ → ℤ)
  (h_arith : ∃ d : ℤ, ∀ n, a n = a 0 + n * d)
  (h_sum_first_three : a 0 + a 1 + a 2 = -24)
  (h_sum_eighteen_nineteen_twenty : a 17 + a 18 + a 19 = 78) :
  (20 / 2 * (a 0 + (a 0 + 19 * d))) = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_20_terms_arithmetic_sequence_l1262_126219


namespace NUMINAMATH_GPT_part_a_l1262_126252

theorem part_a (x : ℝ) : (6 - x) / x = 3 / 6 → x = 4 := by
  sorry

end NUMINAMATH_GPT_part_a_l1262_126252


namespace NUMINAMATH_GPT_least_even_perimeter_l1262_126207

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

theorem least_even_perimeter
  (a b : ℕ) (h1 : a = 24) (h2 : b = 37) (c : ℕ)
  (h3 : c > b) (h4 : a + b > c)
  (h5 : ∃ k : ℕ, k * 2 = triangle_perimeter a b c) :
  triangle_perimeter a b c = 100 :=
sorry

end NUMINAMATH_GPT_least_even_perimeter_l1262_126207


namespace NUMINAMATH_GPT_power_of_two_plus_one_is_power_of_integer_l1262_126274

theorem power_of_two_plus_one_is_power_of_integer (n : ℕ) (hn : 0 < n) (a k : ℕ) (ha : 2^n + 1 = a^k) (hk : 1 < k) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_plus_one_is_power_of_integer_l1262_126274


namespace NUMINAMATH_GPT_math_problem_l1262_126227

theorem math_problem :
  -50 * 3 - (-2.5) / 0.1 = -125 := by
sorry

end NUMINAMATH_GPT_math_problem_l1262_126227


namespace NUMINAMATH_GPT_factorization_of_polynomial_l1262_126242

theorem factorization_of_polynomial (x : ℝ) : 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 :=
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l1262_126242


namespace NUMINAMATH_GPT_trigonometric_identity_l1262_126231

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = -(2 + Real.sqrt 3) / 3 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1262_126231


namespace NUMINAMATH_GPT_find_m_l1262_126244

theorem find_m (m : ℝ) 
    (h1 : ∃ (m: ℝ), ∀ x y : ℝ, x - m * y + 2 * m = 0) 
    (h2 : ∃ (m: ℝ), ∀ x y : ℝ, x + 2 * y - m = 0) 
    (perpendicular : (1/m) * (-1/2) = -1) : m = 1/2 :=
sorry

end NUMINAMATH_GPT_find_m_l1262_126244


namespace NUMINAMATH_GPT_original_decimal_l1262_126292

variable (x : ℝ)

theorem original_decimal (h : x - x / 100 = 1.485) : x = 1.5 :=
sorry

end NUMINAMATH_GPT_original_decimal_l1262_126292


namespace NUMINAMATH_GPT_base_rate_second_telephone_company_l1262_126217

theorem base_rate_second_telephone_company : 
  ∃ B : ℝ, (11 + 20 * 0.25 = B + 20 * 0.20) ∧ B = 12 := by
  sorry

end NUMINAMATH_GPT_base_rate_second_telephone_company_l1262_126217


namespace NUMINAMATH_GPT_sarah_mean_score_l1262_126257

noncomputable def john_mean_score : ℝ := 86
noncomputable def john_num_tests : ℝ := 4
noncomputable def test_scores : List ℝ := [78, 80, 85, 87, 90, 95, 100]
noncomputable def total_sum : ℝ := test_scores.sum
noncomputable def sarah_num_tests : ℝ := 3

theorem sarah_mean_score :
  let john_total_score := john_mean_score * john_num_tests
  let sarah_total_score := total_sum - john_total_score
  let sarah_mean_score := sarah_total_score / sarah_num_tests
  sarah_mean_score = 90.3 :=
by
  sorry

end NUMINAMATH_GPT_sarah_mean_score_l1262_126257


namespace NUMINAMATH_GPT_log_domain_l1262_126218

theorem log_domain (x : ℝ) : 3 - 2 * x > 0 ↔ x < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_log_domain_l1262_126218


namespace NUMINAMATH_GPT_technician_round_trip_l1262_126246

-- Definitions based on conditions
def trip_to_center_completion : ℝ := 0.5 -- Driving to the center is 50% of the trip
def trip_from_center_completion (percent_completed: ℝ) : ℝ := 0.5 * percent_completed -- Completion percentage of the return trip
def total_trip_completion : ℝ := trip_to_center_completion + trip_from_center_completion 0.3 -- Total percentage completed

-- Theorem statement
theorem technician_round_trip : total_trip_completion = 0.65 :=
by
  sorry

end NUMINAMATH_GPT_technician_round_trip_l1262_126246


namespace NUMINAMATH_GPT_total_letters_sent_l1262_126238

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end NUMINAMATH_GPT_total_letters_sent_l1262_126238


namespace NUMINAMATH_GPT_nancy_ate_3_apples_l1262_126268

theorem nancy_ate_3_apples
  (mike_apples : ℝ)
  (keith_apples : ℝ)
  (apples_left : ℝ)
  (mike_apples_eq : mike_apples = 7.0)
  (keith_apples_eq : keith_apples = 6.0)
  (apples_left_eq : apples_left = 10.0) :
  mike_apples + keith_apples - apples_left = 3.0 := 
by
  rw [mike_apples_eq, keith_apples_eq, apples_left_eq]
  norm_num

end NUMINAMATH_GPT_nancy_ate_3_apples_l1262_126268


namespace NUMINAMATH_GPT_gibbs_inequality_l1262_126255

noncomputable section

open BigOperators

variable {r : ℕ} (p q : Fin r → ℝ)

/-- (p_i) is a probability distribution -/
def isProbabilityDistribution (p : Fin r → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i, p i = 1)

/-- -\sum_{i=1}^{r} p_i \ln p_i \leqslant -\sum_{i=1}^{r} p_i \ln q_i for probability distributions p and q -/
theorem gibbs_inequality
  (hp : isProbabilityDistribution p)
  (hq : isProbabilityDistribution q) :
  -∑ i, p i * Real.log (p i) ≤ -∑ i, p i * Real.log (q i) := 
by
  sorry

end NUMINAMATH_GPT_gibbs_inequality_l1262_126255


namespace NUMINAMATH_GPT_necessarily_positive_l1262_126285

-- Definitions based on given conditions
variables {x y z : ℝ}

-- Stating the problem
theorem necessarily_positive : (0 < x ∧ x < 1) → (-2 < y ∧ y < 0) → (0 < z ∧ z < 1) → (x + y^2 > 0) :=
by
  intros hx hy hz
  sorry

end NUMINAMATH_GPT_necessarily_positive_l1262_126285


namespace NUMINAMATH_GPT_perfectCubesCount_l1262_126294

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end NUMINAMATH_GPT_perfectCubesCount_l1262_126294


namespace NUMINAMATH_GPT_minimum_value_of_v_l1262_126221

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

noncomputable def g (x u v : ℝ) : ℝ := (x - u)^3 - 3 * (x - u) - v

theorem minimum_value_of_v (u v : ℝ) (h_pos_u : u > 0) :
  ∀ u > 0, ∀ x : ℝ, f x = g x u v → v ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_v_l1262_126221


namespace NUMINAMATH_GPT_onions_on_scale_l1262_126202

theorem onions_on_scale (N : ℕ) (W_total : ℕ) (W_removed : ℕ) (avg_remaining : ℕ) (avg_removed : ℕ) :
  W_total = 7680 →
  W_removed = 5 * 206 →
  avg_remaining = 190 →
  avg_removed = 206 →
  N = 40 :=
by
  sorry

end NUMINAMATH_GPT_onions_on_scale_l1262_126202


namespace NUMINAMATH_GPT_boat_distance_along_stream_l1262_126239

-- Define the conditions
def boat_speed_still_water := 15 -- km/hr
def distance_against_stream_one_hour := 9 -- km

-- Define the speed of the stream
def stream_speed := boat_speed_still_water - distance_against_stream_one_hour -- km/hr

-- Define the effective speed along the stream
def effective_speed_along_stream := boat_speed_still_water + stream_speed -- km/hr

-- Define the proof statement
theorem boat_distance_along_stream : effective_speed_along_stream = 21 :=
by
  -- Given conditions and definitions, the steps are assumed logically correct
  sorry

end NUMINAMATH_GPT_boat_distance_along_stream_l1262_126239


namespace NUMINAMATH_GPT_q_investment_time_l1262_126271

-- Definitions from the conditions
def investment_ratio_p_q : ℚ := 7 / 5
def profit_ratio_p_q : ℚ := 7 / 13
def time_p : ℕ := 5

-- Problem statement
theorem q_investment_time
  (investment_ratio_p_q : ℚ)
  (profit_ratio_p_q : ℚ)
  (time_p : ℕ)
  (hpq_inv : investment_ratio_p_q = 7 / 5)
  (hpq_profit : profit_ratio_p_q = 7 / 13)
  (ht_p : time_p = 5) : 
  ∃ t_q : ℕ, 35 * t_q = 455 :=
sorry

end NUMINAMATH_GPT_q_investment_time_l1262_126271


namespace NUMINAMATH_GPT_hands_coincide_again_l1262_126247

-- Define the angular speeds of minute and hour hands
def speed_minute_hand : ℝ := 6
def speed_hour_hand : ℝ := 0.5

-- Define the initial condition: coincidence at midnight
def initial_time : ℝ := 0

-- Define the function that calculates the angle of the minute hand at time t
def angle_minute_hand (t : ℝ) : ℝ := speed_minute_hand * t

-- Define the function that calculates the angle of the hour hand at time t
def angle_hour_hand (t : ℝ) : ℝ := speed_hour_hand * t

-- Define the time at which the hands coincide again after midnight
noncomputable def coincidence_time : ℝ := 720 / 11

-- The proof problem statement: The hands coincide again at coincidence_time minutes
theorem hands_coincide_again : 
  angle_minute_hand coincidence_time = angle_hour_hand coincidence_time + 360 :=
sorry

end NUMINAMATH_GPT_hands_coincide_again_l1262_126247


namespace NUMINAMATH_GPT_read_both_books_l1262_126223

theorem read_both_books (B S K N : ℕ) (TOTAL : ℕ)
  (h1 : S = 1/4 * 72)
  (h2 : K = 5/8 * 72)
  (h3 : N = (S - B) - 1)
  (h4 : TOTAL = 72)
  (h5 : TOTAL = (S - B) + (K - B) + B + N)
  : B = 8 :=
by
  sorry

end NUMINAMATH_GPT_read_both_books_l1262_126223


namespace NUMINAMATH_GPT_ellipse_foci_on_y_axis_l1262_126237

theorem ellipse_foci_on_y_axis (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (m + 2)) - (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -3/2) := 
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_on_y_axis_l1262_126237


namespace NUMINAMATH_GPT_weight_of_a_is_75_l1262_126286

theorem weight_of_a_is_75 (a b c d e : ℕ) 
  (h1 : (a + b + c) / 3 = 84) 
  (h2 : (a + b + c + d) / 4 = 80) 
  (h3 : e = d + 3) 
  (h4 : (b + c + d + e) / 4 = 79) : 
  a = 75 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_weight_of_a_is_75_l1262_126286


namespace NUMINAMATH_GPT_domain_of_f_l1262_126236

-- Define the function f(x) = 1/(x+1) + ln(x)
noncomputable def f (x : ℝ) : ℝ := (1 / (x + 1)) + Real.log x

-- The domain of the function is all x such that x > 0
theorem domain_of_f :
  ∀ x : ℝ, (x > 0) ↔ (f x = (1 / (x + 1)) + Real.log x) := 
by sorry

end NUMINAMATH_GPT_domain_of_f_l1262_126236


namespace NUMINAMATH_GPT_largest_cube_edge_from_cone_l1262_126267

theorem largest_cube_edge_from_cone : 
  ∀ (s : ℝ), 
  (s = 2) → 
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_cube_edge_from_cone_l1262_126267


namespace NUMINAMATH_GPT_sin_B_value_cos_A_minus_cos_C_value_l1262_126297

variables {A B C : ℝ} {a b c : ℝ}

theorem sin_B_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) : Real.sin B = Real.sqrt 7 / 4 := 
sorry

theorem cos_A_minus_cos_C_value (h₁ : 4 * b * (Real.sin A) = Real.sqrt 7 * a) (h₂ : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := 
sorry

end NUMINAMATH_GPT_sin_B_value_cos_A_minus_cos_C_value_l1262_126297


namespace NUMINAMATH_GPT_smallest_n_for_n_cubed_ends_in_888_l1262_126248

/-- Proof Problem: Prove that 192 is the smallest positive integer \( n \) such that the last three digits of \( n^3 \) are 888. -/
theorem smallest_n_for_n_cubed_ends_in_888 : ∃ n : ℕ, n > 0 ∧ (n^3 % 1000 = 888) ∧ ∀ m : ℕ, 0 < m ∧ (m^3 % 1000 = 888) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_n_cubed_ends_in_888_l1262_126248


namespace NUMINAMATH_GPT_min_buses_needed_l1262_126291

theorem min_buses_needed (n : ℕ) : 325 / 45 ≤ n ∧ n < 325 / 45 + 1 ↔ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_buses_needed_l1262_126291


namespace NUMINAMATH_GPT_martha_black_butterflies_l1262_126264

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies : ℕ)
  (h1 : total_butterflies = 11)
  (h2 : blue_butterflies = 4)
  (h3 : blue_butterflies = 2 * yellow_butterflies) :
  ∃ black_butterflies : ℕ, black_butterflies = total_butterflies - blue_butterflies - yellow_butterflies :=
sorry

end NUMINAMATH_GPT_martha_black_butterflies_l1262_126264


namespace NUMINAMATH_GPT_original_profit_percentage_l1262_126276

theorem original_profit_percentage (C S : ℝ) 
  (h1 : S - 1.12 * C = 0.5333333333333333 * S) : 
  ((S - C) / C) * 100 = 140 :=
sorry

end NUMINAMATH_GPT_original_profit_percentage_l1262_126276


namespace NUMINAMATH_GPT_johann_mail_l1262_126299

def pieces_of_mail_total : ℕ := 180
def pieces_of_mail_friends : ℕ := 41
def friends : ℕ := 2
def pieces_of_mail_johann : ℕ := pieces_of_mail_total - (pieces_of_mail_friends * friends)

theorem johann_mail : pieces_of_mail_johann = 98 := by
  sorry

end NUMINAMATH_GPT_johann_mail_l1262_126299


namespace NUMINAMATH_GPT_fraction_ordering_l1262_126279

theorem fraction_ordering :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  (b < c) ∧ (c < a) :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  sorry

end NUMINAMATH_GPT_fraction_ordering_l1262_126279


namespace NUMINAMATH_GPT_ac_length_l1262_126235

theorem ac_length (a b c d e : ℝ)
  (h1 : b - a = 5)
  (h2 : c - b = 2 * (d - c))
  (h3 : e - d = 4)
  (h4 : e - a = 18) :
  d - a = 11 :=
by
  sorry

end NUMINAMATH_GPT_ac_length_l1262_126235


namespace NUMINAMATH_GPT_right_triangle_side_lengths_l1262_126254

theorem right_triangle_side_lengths (x : ℝ) :
  (2 * x + 2)^2 + (x + 2)^2 = (x + 4)^2 ∨ (2 * x + 2)^2 + (x + 4)^2 = (x + 2)^2 ↔ (x = 1 ∨ x = 4) :=
by sorry

end NUMINAMATH_GPT_right_triangle_side_lengths_l1262_126254


namespace NUMINAMATH_GPT_side_length_of_S2_l1262_126213

variables (s r : ℕ)

-- Conditions
def combined_width_eq : Prop := 3 * s + 100 = 4000
def combined_height_eq : Prop := 2 * r + s = 2500

-- Conclusion we want to prove
theorem side_length_of_S2 : combined_width_eq s → combined_height_eq s r → s = 1300 :=
by
  intros h_width h_height
  sorry

end NUMINAMATH_GPT_side_length_of_S2_l1262_126213


namespace NUMINAMATH_GPT_shorter_piece_length_l1262_126290

/-- A 69-inch board is cut into 2 pieces. One piece is 2 times the length of the other.
    Prove that the length of the shorter piece is 23 inches. -/
theorem shorter_piece_length (x : ℝ) :
  let shorter := x
  let longer := 2 * x
  (shorter + longer = 69) → shorter = 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_shorter_piece_length_l1262_126290


namespace NUMINAMATH_GPT_mean_profit_first_15_days_l1262_126273

-- Definitions and conditions
def mean_daily_profit_entire_month : ℝ := 350
def total_days_in_month : ℕ := 30
def mean_daily_profit_last_15_days : ℝ := 445

-- Proof statement
theorem mean_profit_first_15_days : 
  (mean_daily_profit_entire_month * (total_days_in_month : ℝ) 
   - mean_daily_profit_last_15_days * 15) / 15 = 255 :=
by
  sorry

end NUMINAMATH_GPT_mean_profit_first_15_days_l1262_126273


namespace NUMINAMATH_GPT_Michael_pizza_fraction_l1262_126225

theorem Michael_pizza_fraction (T : ℚ) (L : ℚ) (total : ℚ) (M : ℚ) 
  (hT : T = 1 / 2) (hL : L = 1 / 6) (htotal : total = 1) (hM : total - (T + L) = M) :
  M = 1 / 3 := 
sorry

end NUMINAMATH_GPT_Michael_pizza_fraction_l1262_126225


namespace NUMINAMATH_GPT_comparison_of_exponential_values_l1262_126240

theorem comparison_of_exponential_values : 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  c < a ∧ a < b := 
by 
  let a := 0.3^3
  let b := 3^0.3
  let c := 0.2^3
  sorry

end NUMINAMATH_GPT_comparison_of_exponential_values_l1262_126240


namespace NUMINAMATH_GPT_problem_statement_l1262_126270

-- Given conditions
variable (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k)

-- Hypothesis configuration for inductive proof and goal statement
theorem problem_statement : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1262_126270


namespace NUMINAMATH_GPT_p_range_l1262_126204

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem p_range :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → 29 ≤ p x ∧ p x ≤ 93 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_p_range_l1262_126204


namespace NUMINAMATH_GPT_additional_charge_l1262_126228

variable (charge_first : ℝ) -- The charge for the first 1/5 of a mile
variable (total_charge : ℝ) -- Total charge for an 8-mile ride
variable (distance : ℝ) -- Total distance of the ride

theorem additional_charge 
  (h1 : charge_first = 3.50) 
  (h2 : total_charge = 19.1) 
  (h3 : distance = 8) :
  ∃ x : ℝ, x = 0.40 :=
  sorry

end NUMINAMATH_GPT_additional_charge_l1262_126228


namespace NUMINAMATH_GPT_abc_zero_l1262_126233

-- Define the given conditions as hypotheses
theorem abc_zero (a b c : ℚ) 
  (h1 : (a^2 + 1)^3 = b + 1)
  (h2 : (b^2 + 1)^3 = c + 1)
  (h3 : (c^2 + 1)^3 = a + 1) : 
  a = 0 ∧ b = 0 ∧ c = 0 := 
sorry

end NUMINAMATH_GPT_abc_zero_l1262_126233


namespace NUMINAMATH_GPT_jelly_cost_l1262_126206

theorem jelly_cost (N B J : ℕ) (hN_gt_1 : N > 1) (h_cost_eq : N * (3 * B + 7 * J) = 252) : 7 * N * J = 168 := by
  sorry

end NUMINAMATH_GPT_jelly_cost_l1262_126206


namespace NUMINAMATH_GPT_fraction_to_decimal_l1262_126230

theorem fraction_to_decimal :
  (7 : ℝ) / 16 = 0.4375 := 
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1262_126230


namespace NUMINAMATH_GPT_determine_k_l1262_126280

noncomputable def k_value (k : ℤ) : Prop :=
  let m := (-2 - 2) / (3 - 1)
  let b := 2 - m * 1
  let y := m * 4 + b
  let point := (4, k / 3)
  point.2 = y

theorem determine_k :
  ∃ k : ℤ, k_value k ∧ k = -12 :=
by
  use -12
  sorry

end NUMINAMATH_GPT_determine_k_l1262_126280


namespace NUMINAMATH_GPT_student_count_l1262_126269

open Nat

theorem student_count :
  ∃ n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_student_count_l1262_126269


namespace NUMINAMATH_GPT_christina_has_three_snakes_l1262_126293

def snake_lengths : List ℕ := [24, 16, 10]

def total_length : ℕ := 50

theorem christina_has_three_snakes
  (lengths : List ℕ)
  (total : ℕ)
  (h_lengths : lengths = snake_lengths)
  (h_total : total = total_length)
  : lengths.length = 3 :=
by
  sorry

end NUMINAMATH_GPT_christina_has_three_snakes_l1262_126293


namespace NUMINAMATH_GPT_find_k_value_l1262_126224

variables {R : Type*} [Field R] {a b x k : R}

-- Definitions for the conditions in the problem
def f (x : R) (a b : R) : R := (b * x + 1) / (2 * x + a)

-- Statement of the problem
theorem find_k_value (h_ab : a * b ≠ 2)
  (h_k : ∀ (x : R), x ≠ 0 → f x a b * f (x⁻¹) a b = k) :
  k = (1 : R) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l1262_126224


namespace NUMINAMATH_GPT_caesars_charge_l1262_126211

theorem caesars_charge :
  ∃ (C : ℕ), (C + 30 * 60 = 500 + 35 * 60) ↔ (C = 800) :=
by
  sorry

end NUMINAMATH_GPT_caesars_charge_l1262_126211


namespace NUMINAMATH_GPT_calculate_difference_l1262_126249

def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

theorem calculate_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by
  sorry

end NUMINAMATH_GPT_calculate_difference_l1262_126249


namespace NUMINAMATH_GPT_sum_of_odds_square_l1262_126232

theorem sum_of_odds_square (n : ℕ) (h : 0 < n) : (Finset.range n).sum (λ i => 2 * i + 1) = n ^ 2 :=
sorry

end NUMINAMATH_GPT_sum_of_odds_square_l1262_126232


namespace NUMINAMATH_GPT_find_AX_l1262_126259

theorem find_AX
  (AB AC BC : ℚ)
  (H : AB = 80)
  (H1 : AC = 50)
  (H2 : BC = 30)
  (angle_bisector_theorem_1 : ∀ (AX XC y : ℚ), AX = 8 * y ∧ XC = 3 * y ∧ 11 * y = AC → y = 50 / 11)
  (angle_bisector_theorem_2 : ∀ (BD DC z : ℚ), BD = 8 * z ∧ DC = 5 * z ∧ 13 * z = BC → z = 30 / 13) :
  AX = 400 / 11 := 
sorry

end NUMINAMATH_GPT_find_AX_l1262_126259


namespace NUMINAMATH_GPT_Ana_age_eight_l1262_126262

theorem Ana_age_eight (A B n : ℕ) (h1 : A - 1 = 7 * (B - 1)) (h2 : A = 4 * B) (h3 : A - B = n) : A = 8 :=
by
  sorry

end NUMINAMATH_GPT_Ana_age_eight_l1262_126262


namespace NUMINAMATH_GPT_rope_length_in_cm_l1262_126203

-- Define the given conditions
def num_equal_pieces : ℕ := 150
def length_equal_piece_mm : ℕ := 75
def num_remaining_pieces : ℕ := 4
def length_remaining_piece_mm : ℕ := 100

-- Prove that the total length of the rope in centimeters is 1165
theorem rope_length_in_cm : (num_equal_pieces * length_equal_piece_mm + num_remaining_pieces * length_remaining_piece_mm) / 10 = 1165 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_in_cm_l1262_126203


namespace NUMINAMATH_GPT_fish_count_together_l1262_126253

namespace FishProblem

def JerkTunaFish : ℕ := 144
def TallTunaFish : ℕ := 2 * JerkTunaFish
def SwellTunaFish : ℕ := TallTunaFish + (TallTunaFish / 2)
def totalFish : ℕ := JerkTunaFish + TallTunaFish + SwellTunaFish

theorem fish_count_together : totalFish = 864 := by
  sorry

end FishProblem

end NUMINAMATH_GPT_fish_count_together_l1262_126253
